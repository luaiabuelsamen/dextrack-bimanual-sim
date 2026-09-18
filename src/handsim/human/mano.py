"""MANO hand model, loaded without chumpy.

The official ``mano_v1_2`` pickles are Python-2 chumpy objects.  chumpy does not
install against modern numpy, and we do not need any of its autodiff: every
array we want is a plain ndarray sitting inside a chumpy wrapper.  So we
unpickle with stub classes that keep the payload and throw the wrapper away,
then cache the result as an npz.

GRAB supplies ``fullpose`` (45 = 15 joints x 3 axis-angle) directly, so the
hand-PCA basis is never used, and it supplies a subject-specific hand template
(``vtemp``), so the shape blendshapes are never used either.  What is left is
the pose corrective and linear blend skinning, both implemented here.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MANO has no fingertip joints.  These are the vertex indices conventionally
# used as tips (thumb, index, middle, ring, pinky); they are the same for the
# left and right models because the topologies mirror each other.
TIP_VERTS = (745, 317, 444, 556, 673)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class _Stub:
    """Absorbs any chumpy class; keeps whatever state the pickle carried."""

    def __init__(self, *a, **k):
        self._a = a

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_s": state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy") or module.startswith("ch"):
            return _Stub
        if module == "scipy.sparse.csc" or module.startswith("scipy.sparse"):
            import scipy.sparse as sp

            return getattr(sp, name, _Stub)
        return super().find_class(module, name)


def _dechumpify(obj):
    """Recover the ndarray a chumpy object is wrapping."""
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, _Stub):
        d = obj.__dict__
        for key in ("x", "_x", "r", "_r"):
            if key in d:
                return _dechumpify(d[key])
        # Ch objects sometimes only hold their leaves.
        for v in d.values():
            if isinstance(v, np.ndarray):
                return v
        raise ValueError(f"no array inside {sorted(d)}")
    if hasattr(obj, "toarray"):  # scipy sparse
        return np.asarray(obj.toarray())
    return np.asarray(obj)


@dataclass(frozen=True)
class ManoModel:
    """A MANO hand.  ``side`` is ``"left"`` or ``"right"``."""

    side: str
    v_template: np.ndarray  # (778, 3)
    shapedirs: np.ndarray  # (778, 3, 10)
    posedirs: np.ndarray  # (778, 3, 135)
    J_regressor: np.ndarray  # (16, 778)
    weights: np.ndarray  # (778, 16)
    parents: np.ndarray  # (16,)  parents[0] == -1
    faces: np.ndarray  # (F, 3)

    @property
    def n_joints(self) -> int:
        return self.J_regressor.shape[0]


def convert(pkl: Path, out: Path) -> Path:
    """Read a MANO_{LEFT,RIGHT}.pkl and write a chumpy-free npz."""
    with open(pkl, "rb") as fh:
        raw = _Unpickler(fh, encoding="latin1").load()

    kin = np.asarray(raw["kintree_table"])
    parents = kin[0].astype(np.int64).copy()
    parents[0] = -1

    arrays = {
        "v_template": _dechumpify(raw["v_template"]).astype(np.float64),
        "shapedirs": _dechumpify(raw["shapedirs"]).astype(np.float64).reshape(778, 3, -1),
        "posedirs": _dechumpify(raw["posedirs"]).astype(np.float64),
        "J_regressor": _dechumpify(raw["J_regressor"]).astype(np.float64),
        "weights": _dechumpify(raw["weights"]).astype(np.float64),
        "parents": parents,
        "faces": np.asarray(raw["f"]).astype(np.int64),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out


def load(side: str, root: Path | None = None) -> ManoModel:
    """Load a hand, converting from the official pickle on first use."""
    from handsim.paths import DATA

    root = root or DATA / "mano" / "mano_v1_2" / "models"
    side = side.lower()
    cache = DATA / "mano" / f"mano_{side}.npz"
    if not cache.exists():
        convert(root / f"MANO_{side.upper()}.pkl", cache)
    z = np.load(cache)
    return ManoModel(side=side, **{k: z[k] for k in z.files})


# --- forward kinematics -------------------------------------------------

def _rodrigues(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = aa / safe
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape)
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    R = I + s * K + (1.0 - c) * (K @ K)
    return np.where(theta[..., None] < 1e-8, I, R)


def forward(
    model: ManoModel,
    pose: np.ndarray,
    transl: np.ndarray,
    v_template: np.ndarray | None = None,
    pose_blend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose the hand.

    ``pose`` is (T, 48) axis-angle: global orientation then 15 joints.
    ``v_template`` overrides the model template with a subject-specific one
    (GRAB's ``vtemp``).  Returns ``(verts (T,778,3), joints (T,21,3))`` where
    joints are the 16 MANO joints followed by the 5 fingertip vertices.
    """
    pose = np.atleast_2d(np.asarray(pose, dtype=np.float64))
    transl = np.atleast_2d(np.asarray(transl, dtype=np.float64))
    T = pose.shape[0]
    J = model.n_joints
    if pose.shape[1] != 3 * J:
        raise ValueError(f"pose has {pose.shape[1]} dims, expected {3 * J}")

    v0 = model.v_template if v_template is None else np.asarray(v_template, float)
    if v0.shape != model.v_template.shape:
        raise ValueError(f"template is {v0.shape}, expected {model.v_template.shape}")

    rest_j = model.J_regressor @ v0  # (J, 3)
    R = _rodrigues(pose.reshape(T, J, 3))  # (T, J, 3, 3)

    if pose_blend:
        # Pose correctives are driven by the non-root rotations minus identity.
        feat = (R[:, 1:] - np.eye(3)).reshape(T, -1)  # (T, 135)
        v_posed = v0[None] + np.einsum("vdp,tp->tvd", model.posedirs, feat)
    else:
        v_posed = np.broadcast_to(v0, (T, *v0.shape)).copy()

    # Compose world transforms down the kinematic tree.
    A = np.zeros((T, J, 4, 4))
    A[..., 3, 3] = 1.0
    A[:, 0, :3, :3] = R[:, 0]
    A[:, 0, :3, 3] = rest_j[0]
    for j in range(1, J):
        p = model.parents[j]
        local = np.zeros((T, 4, 4))
        local[:, 3, 3] = 1.0
        local[:, :3, :3] = R[:, j]
        local[:, :3, 3] = rest_j[j] - rest_j[p]
        A[:, j] = A[:, p] @ local

    joints = A[:, :, :3, 3].copy()

    # Remove the rest-pose offset so the skinning transform is relative.
    rel = A.copy()
    rel[:, :, :3, 3] -= np.einsum("tjab,jb->tja", A[:, :, :3, :3], rest_j)

    Tv = np.einsum("vj,tjab->tvab", model.weights, rel)  # (T, 778, 4, 4)
    verts = np.einsum("tvab,tvb->tva", Tv[..., :3, :3], v_posed) + Tv[..., :3, 3]

    verts += transl[:, None, :]
    joints += transl[:, None, :]
    tips = verts[:, list(TIP_VERTS), :]
    return verts, np.concatenate([joints, tips], axis=1)
