"""Read a GRAB sequence into a source-agnostic reference.

GRAB (Taheri et al., ECCV 2020) is whole-body grasping motion capture: a
subject picks up one of 51 objects and performs an intent ("drink", "lift",
"pass"), recorded at 120 Hz with the object's 6-DoF pose and a MANO fit for
both hands.  That makes it the reference source DexTrack trains on, and the
one `track.Reference` was written to accept without changes.

What this module produces, per sequence:

    object pose over time           -> `track.Reference`, unchanged downstream
    MANO joints + tips, both hands  -> the retargeting target
    a contact window                -> which frames are actually a grasp

Nothing here retargets.  Turning human fingertips into robot joint angles is a
separate problem with its own failure modes, and mixing the two is how you end
up unable to tell a bad reference from a bad retarget.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from handsim import paths
from handsim.human import mano as mano_mod

FPS = 120.0
SIDES = ("rhand", "lhand")


def _ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal binary/ascii PLY reader -- (verts, faces).

    GRAB ships .ply for both the subject hand templates and the object
    contact meshes.  trimesh would do this, but it drags in a large dependency
    tree for two element types we can parse in thirty lines.
    """
    raw = path.read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    header = raw[:end].decode("ascii", "replace").splitlines()
    body = raw[end:].lstrip(b"\r\n")

    fmt, n_v, n_f, props = None, 0, 0, []
    element = None
    for line in header:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            element = tok[1]
            if element == "vertex":
                n_v = int(tok[2])
            elif element == "face":
                n_f = int(tok[2])
        elif tok[0] == "property" and element == "vertex":
            props.append((tok[1], tok[-1]))

    np_t = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
            "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4",
            "uint": "u4", "short": "i2", "ushort": "u2"}

    if fmt == "ascii":
        lines = body.decode().split("\n")
        verts = np.array([[float(x) for x in lines[i].split()[:3]]
                          for i in range(n_v)], dtype=np.float64)
        faces = np.array([[int(x) for x in lines[n_v + i].split()[1:4]]
                          for i in range(n_f)], dtype=np.int64) if n_f else np.zeros((0, 3), np.int64)
        return verts, faces

    endian = "<" if fmt == "binary_little_endian" else ">"
    dt = np.dtype([(n, endian + np_t[t]) for t, n in props])
    vrec = np.frombuffer(body, dtype=dt, count=n_v)
    verts = np.stack([vrec["x"], vrec["y"], vrec["z"]], -1).astype(np.float64)

    faces = np.zeros((0, 3), np.int64)
    if n_f:
        off = n_v * dt.itemsize
        # face lists are (count, i, j, k); GRAB's are all triangles
        fdt = np.dtype([("n", endian + "u1"), ("v", endian + "i4", 3)])
        faces = np.frombuffer(body, dtype=fdt, count=n_f, offset=off)["v"].astype(np.int64)
    return verts, faces


@lru_cache(maxsize=64)
def _cached_ply(path_str: str):
    v, f = _ply(Path(path_str))
    return v, f


def _rodrigues(aa):
    return mano_mod._rodrigues(np.asarray(aa, float))


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    """(T,3,3) -> (T,4) wxyz, branch-selected on the largest component."""
    R = np.asarray(R, float)
    T = R.shape[0]
    q = np.zeros((T, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(T):
        m, t = R[i], tr[i]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
                    (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                    (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                    0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.sign(q[:, :1] + 1e-12)   # keep w >= 0, a continuous branch


@dataclass
class HandTrack:
    """One MANO hand over a sequence, in world coordinates."""
    side: str                 # "rhand" | "lhand"
    joints: np.ndarray        # (T, 21, 3) 16 MANO joints then 5 fingertips
    verts: np.ndarray | None  # (T, 778, 3) if requested
    wrist_pos: np.ndarray     # (T, 3)
    wrist_quat: np.ndarray    # (T, 4) wxyz
    fullpose: np.ndarray      # (T, 45) finger axis-angle, as fit

    @property
    def tips(self) -> np.ndarray:
        return self.joints[:, 16:, :]


@dataclass
class GrabSequence:
    """One GRAB clip: an object trajectory and the hands that produced it."""
    name: str
    subject: str
    obj: str
    intent: str
    gender: str
    dt: float
    obj_pos: np.ndarray       # (T, 3)
    obj_quat: np.ndarray      # (T, 4) wxyz
    hands: dict[str, HandTrack]
    obj_mesh: tuple[np.ndarray, np.ndarray]   # canonical verts, faces
    #: (T,3,3) PROPER rotation matrices: world = R @ v_object + pos.
    #: GRAB's own ObjectModel computes `matmul(v_template, rot_mats)`, i.e.
    #: v @ M, which is M.T @ v -- the TRANSPOSE of the usual convention. Storing
    #: the proper matrix here converts once, at the boundary, so nothing
    #: downstream has to remember which way round GRAB's object frame goes.
    #: Getting this wrong put the hand on the mug's body instead of through its
    #: handle while still passing every distance check.
    obj_R: np.ndarray = None

    @property
    def T(self) -> int:
        return len(self.obj_pos)

    def reference(self, side: str = "rhand"):
        """The object half, as the `track.Reference` the controller already takes."""
        from handsim.track_core import Reference
        return Reference(pos=self.obj_pos, quat=self.obj_quat, dt=self.dt,
                         name=self.name, source="grab")

    def object_world(self, k: int) -> np.ndarray:
        """Object mesh vertices at frame k, in world coordinates."""
        v, _ = self.obj_mesh
        return v @ self.obj_R[k].T + self.obj_pos[k]

    def to_object(self, pts: np.ndarray, k: int) -> np.ndarray:
        """World points -> the object's own frame at frame k."""
        return (np.asarray(pts, float) - self.obj_pos[k]) @ self.obj_R[k]

    def contact_distance(self, side: str = "rhand") -> np.ndarray:
        """(T,) min distance from any hand vertex to any object vertex.

        A point-cloud lower bound, not a true surface distance -- GRAB's
        contact meshes are dense enough (2-6k verts) that the difference is
        well under a millimetre, and it needs no mesh library.
        """
        hand = self.hands[side]
        if hand.verts is None:
            raise ValueError("load with verts=True to measure contact")
        v0, _ = self.obj_mesh
        out = np.empty(self.T)
        for k in range(self.T):
            ov = v0 @ self.obj_R[k].T + self.obj_pos[k]
            d = np.linalg.norm(hand.verts[k][:, None, :] - ov[None, :, :], axis=-1)
            out[k] = d.min()
        return out


def sequences(subject: str = "s1", root: Path | None = None) -> list[Path]:
    root = Path(root or paths.GRAB) / "grab" / subject
    return sorted(root.glob("*.npz"))


def load(path: str | Path, verts: bool = False, stride: int = 1,
         root: Path | None = None) -> GrabSequence:
    """Read one GRAB .npz and pose both hands.

    ``stride`` subsamples in time; GRAB is 120 Hz and most of that is dwell.
    ``verts`` also returns the 778 hand vertices per frame, which is what
    contact measurement needs and nothing else does.
    """
    root = Path(root or paths.GRAB)
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = root / "grab" / path
    d = np.load(path, allow_pickle=True)

    sl = slice(None, None, stride)
    op = d["object"].item()["params"]
    obj_aa = np.asarray(op["global_orient"], float)[sl]
    obj_pos = np.asarray(op["transl"], float)[sl]
    obj_M = _rodrigues(obj_aa).transpose(0, 2, 1)     # see ObjectScene.obj_R
    obj_quat = _quat_from_R(obj_M)

    gender = str(d["gender"])
    hands: dict[str, HandTrack] = {}
    for side in SIDES:
        if side not in d:
            continue
        h = d[side].item()
        p = h["params"]
        tmpl, _ = _cached_ply(str(root / h["vtemp"]))
        model = mano_mod.load("right" if side == "rhand" else "left")
        pose = np.concatenate(
            [np.asarray(p["global_orient"], float)[sl],
             np.asarray(p["fullpose"], float)[sl]], axis=1)
        transl = np.asarray(p["transl"], float)[sl]
        V, J = mano_mod.forward(model, pose, transl, v_template=tmpl)
        hands[side] = HandTrack(
            side=side, joints=J, verts=V if verts else None,
            wrist_pos=J[:, 0, :],
            wrist_quat=_quat_from_R(_rodrigues(pose[:, :3])),
            fullpose=pose[:, 3:],
        )

    obj_name = str(d["obj_name"])
    mesh = _cached_ply(str(root / "tools" / "object_meshes" / "contact_meshes"
                           / f"{obj_name}.ply"))

    seq = GrabSequence(
        name=path.stem, subject=str(d["sbj_id"]), obj=obj_name,
        intent=str(d["motion_intent"]), gender=gender,
        dt=stride / float(d["framerate"]),
        obj_pos=obj_pos, obj_quat=obj_quat, hands=hands, obj_mesh=mesh,
        obj_R=obj_M,
    )
    seq.obj_quat_aa = obj_aa       # raw axis-angle, as recorded
    return seq
