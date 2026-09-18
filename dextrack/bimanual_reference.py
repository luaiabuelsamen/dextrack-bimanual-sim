"""A two-hand reference in DexTrack's format, from their one-hand reference.

DexTrack stores, per clip, an object trajectory and one (T, 22) hand
trajectory for the right hand. A bimanual clip needs a second 22-vector. The
human's own left hand is in GRAB, but GRAB's object frame is not theirs (they
re-canonicalize the meshes, so their object rotations differ from GRAB's by a
per-object transform that has to be solved before anything can be carried
across; see `retarget.py`). That is a separate problem, and it is not the one
standing between here and two hands in their simulator.

So this builds the left hand by reflection instead. At every frame the right
hand's palm pose is reflected across a plane through the object's centre, in
the object's own frame, and the sixteen finger joints are copied unchanged.
That is exact, not approximate: `left_hand.py` builds the left hand as the
right hand mirrored in its own y = 0 plane, so the same joint values on the
left model give the mirrored finger configuration, and the only thing to
compute is where to put the palm.

Writing that out: the left hand's world transform is

    X_left = N . X_right . D

with `D` the hand-frame mirror the left model already carries and `N` the
world reflection across the chosen plane. Both are improper, so their
composition with a rotation is a proper rigid transform, which is why this
yields a pose a simulator will accept.

What it is and is not: a constructed bimanual reference, two hands placed
symmetrically about the object and moving with it, not a recording of what
the human's two hands did. It is labelled that way in the file it writes. It
exists so the two-hand environment can be built and trained against something
physically sensible while the GRAB-to-DexTrack frame map is worked out.

    python dextrack/bimanual_reference.py --ref out/refs/passive_active_info_ori_grab_s1_binoculars_see_1_nf_300.npy \
        --out out/refs/bimanual_binoculars_see_1.npy --check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

#: the hand-frame mirror the left model carries, as a matrix
D = np.diag([1.0, -1.0, 1.0])

ASSETS = Path("out/dextrack_assets")
RIGHT_URDF = ASSETS / "allegro_hand_description_right_fly_v2.urdf"
LEFT_URDF = ASSETS / "allegro_hand_description_left_fly_v2.urdf"


def pose_of(q22: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Their 22-vector's base as (position, rotation matrix)."""
    return q22[..., :3], R.from_euler("XYZ", q22[..., 3:6]).as_matrix()


def to_q22(pos: np.ndarray, rot: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Position, rotation matrix and sixteen joints back into their 22-vector."""
    out = np.empty(joints.shape[:-1] + (22,))
    out[..., :3] = pos
    out[..., 3:6] = R.from_matrix(rot).as_euler("XYZ")
    out[..., 6:] = joints
    return out


def mirror_axis(q_right: np.ndarray, obj_pos: np.ndarray, obj_rot: np.ndarray) -> int:
    """Which object axis to reflect across: the one the palm sits furthest along.

    Reflecting across the plane normal to that axis is what puts the second
    hand on the other side of the object rather than on top of the first.
    """
    p, _ = pose_of(q_right)
    local = np.einsum("tij,tj->ti", obj_rot.transpose(0, 2, 1), p - obj_pos)
    return int(np.argmax(np.abs(local).mean(0)))


def build(ref: dict, axis: int | None = None) -> dict:
    """Their reference plus a left hand, reflected across the object."""
    q_r = np.asarray(ref["robot_delta_states_weights_np"], float)
    obj_pos = np.asarray(ref["object_transl"], float)
    obj_rot = R.from_quat(np.asarray(ref["object_rot_quat"], float)).as_matrix()
    if axis is None:
        axis = mirror_axis(q_r, obj_pos, obj_rot)

    n_local = np.zeros(3)
    n_local[axis] = 1.0
    refl_local = np.eye(3) - 2 * np.outer(n_local, n_local)          # improper

    pos_r, rot_r = pose_of(q_r)
    # the world reflection at each frame: into the object frame, reflect, back out
    N = np.einsum("tij,jk,tlk->til", obj_rot, refl_local, obj_rot)
    # X_left = N . X_right . D, with N acting about the object's centre
    rot_l = np.einsum("tij,tjk,kl->til", N, rot_r, D)
    pos_l = np.einsum("tij,tj->ti", N, pos_r - obj_pos) + obj_pos

    q_l = to_q22(pos_l, rot_l, q_r[:, 6:])
    out = dict(ref)
    out["robot_delta_states_weights_np_left"] = q_l
    out["bimanual"] = True
    out["left_source"] = f"reflected from the right hand across object axis {axis}"
    return out


def check(built: dict, tol_mm: float = 0.5) -> None:
    """Place both hands from the reference and verify the left mirrors the right.

    The test is geometric and independent of how the reflection was derived:
    every left fingertip, carried back through the world reflection, must land
    on the matching right fingertip.
    """
    import mujoco
    from dextrack.left_hand import mujoco_copy, DOF_ORDER

    def load(p):
        m = mujoco.MjModel.from_xml_path(str(mujoco_copy(Path(p), Path("out/urdf_work"))))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        return m, mujoco.MjData(m), [m.jnt_qposadr[names.index(n)] for n in DOF_ORDER]

    mr, dr, ar = load(RIGHT_URDF)
    ml, dl, al = load(LEFT_URDF)
    q_r = built["robot_delta_states_weights_np"]
    q_l = built["robot_delta_states_weights_np_left"]
    obj_pos = np.asarray(built["object_transl"], float)
    obj_rot = R.from_quat(np.asarray(built["object_rot_quat"], float)).as_matrix()
    axis = int(built["left_source"].split()[-1])
    n_local = np.zeros(3); n_local[axis] = 1.0
    refl = np.eye(3) - 2 * np.outer(n_local, n_local)

    tips = ["link_3", "link_7", "link_11", "link_15", "palm_link"]
    worst = 0.0
    for k in range(0, len(q_r), max(1, len(q_r) // 12)):
        for m, d, adr, q in ((mr, dr, ar, q_r[k]), (ml, dl, al, q_l[k])):
            d.qpos[:] = 0
            d.qpos[adr] = q
            mujoco.mj_kinematics(m, d)
        N = obj_rot[k] @ refl @ obj_rot[k].T
        for t in tips:
            a = dr.xpos[mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_BODY, t)]
            b = dl.xpos[mujoco.mj_name2id(ml, mujoco.mjtObj.mjOBJ_BODY, t)]
            back = N @ (b - obj_pos[k]) + obj_pos[k]
            worst = max(worst, float(np.abs(a - back).max()))
    print(f"left hand reflects onto the right: worst {worst*1000:.3f} mm over 12 frames x {len(tips)} bodies")
    if worst * 1000 > tol_mm:
        raise SystemExit("reflection is not exact; check the mirror convention")

    # and report the geometry that decides whether this is a sane two-hand pose
    sep = np.linalg.norm(q_l[:, :3] - q_r[:, :3], axis=1)
    print(f"palm separation: {sep.min()*100:.1f} to {sep.max()*100:.1f} cm (mean {sep.mean()*100:.1f})")
    print(f"mirror axis: object axis {axis}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True, help="their one-hand reference for the clip")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--axis", type=int, default=None, help="object axis to reflect across (default: chosen)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    ref = np.load(a.ref, allow_pickle=True).item()
    built = build(ref, a.axis)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, built, allow_pickle=True)
    print(f"wrote {a.out}: {len(built['robot_delta_states_weights_np'])} frames, two hands")
    if a.check:
        check(built)


if __name__ == "__main__":
    main()
