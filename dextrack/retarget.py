"""Retarget a GRAB hand onto DexTrack's Allegro, in DexTrack's own convention.

DexTrack stores one reference per clip: an object trajectory and a (T, 22)
hand trajectory, `[x, y, z, rx, ry, rz, joint_0 .. joint_15]`, the rotation
read as intrinsic XYZ Euler, for the right hand only. A two-handed clip needs
the same thing for the left hand, and nothing in their release produces it.

The fit is done against their own URDF loaded into MuJoCo, so the numbers that
come out are already in their convention: no coordinate map, no joint-order
table, nothing to get subtly wrong between our solver and their simulator.
`left_hand.py` supplies the left URDF, which they do not ship.

What is fitted, per frame, is contact rather than posture. Each human
fingertip within `contact_tol` of the object is projected onto the object
surface and the matching robot fingertip is asked to reach that surface
point; fingertips that were not near the object keep the human's own tip
position at a tenth of the weight, which holds the posture plausible without
inventing a contact. The human wrist is a weak target on the palm, which is
what stops the redundant arm from wandering. The object is the same size for
both, so hand scale never enters: a smaller robot hand simply brings its
wrist in, which the six base degrees of freedom allow.

Solved by damped Gauss-Newton on MuJoCo's analytic body Jacobians, warm
started from the previous frame, with a step toward the previous frame's pose
carrying the null space so consecutive frames do not land in different parts
of it. MANO has five fingertips and Allegro four, so the pinky is dropped.

    python dextrack/retarget.py --clip s1/binoculars_see_1 --side lhand --out out/refs/left.npy
    python dextrack/retarget.py --clip s2/cubesmall_inspect_1 --side rhand --validate \
        out/refs/passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy

`--validate` fits the RIGHT hand and reports how close it lands to DexTrack's
own right-hand trajectory for the same clip. That is the check that the
convention, the joint order and the fit are all right; the left hand has no
ground truth of its own.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: their fingertip links, in the order this module pairs them with MANO's
ROBOT_TIPS = ["link_3", "link_7", "link_11", "link_15"]     # index, middle, ring, thumb
#: MANO tips come out thumb-first
MANO_TIPS = [1, 2, 3, 0]                                    # index, middle, ring, thumb
PALM = "palm_link"
DOF_ORDER = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + [f"joint_{i}" for i in range(16)]

#: GRAB's own contact threshold
CONTACT_TOL = 0.005
#: a tip that is not touching is held at a tenth of the weight of one that is
W_FREE = 0.1
#: the wrist is a weak anchor, not a target
W_WRIST = 0.25
#: pull toward the previous frame, which also picks a point in the null space
W_SMOOTH = 0.05
#: Levenberg-Marquardt damping
LAMBDA = 1e-3


@dataclass
class Fit:
    """A retargeted trajectory in DexTrack's convention."""
    q: np.ndarray              # (T, 22)
    tip_err_m: np.ndarray      # (T,) mean fingertip residual
    contacts: np.ndarray       # (T, 4) which tips were fitted to the surface
    frames: np.ndarray         # (T,) indices into the source sequence


def load_model(urdf: Path, meshdir: Path | None = None):
    """Their URDF as a MuJoCo model, with the 22 dof in their order."""
    import mujoco
    from dextrack.left_hand import mujoco_copy

    meshdir = meshdir or urdf.parent / "meshes"
    path = mujoco_copy(urdf, Path("out/urdf_work"), meshdir)
    m = mujoco.MjModel.from_xml_path(str(path))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    missing = [n for n in DOF_ORDER if n not in names]
    if missing:
        raise ValueError(f"{urdf} is missing {missing}; left hands need left_hand.py")
    qadr = np.array([m.jnt_qposadr[names.index(n)] for n in DOF_ORDER])
    dadr = np.array([m.jnt_dofadr[names.index(n)] for n in DOF_ORDER])
    return m, qadr, dadr


def _bodies(m):
    import mujoco
    return ([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t) for t in ROBOT_TIPS],
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PALM))


def fit_sequence(seq, side: str, urdf: Path, window: tuple[int, int] | None = None,
                 iters: int = 60, contact_tol: float = CONTACT_TOL,
                 meshdir: Path | None = None, verbose: bool = False) -> Fit:
    """Fit `side` of a GRAB sequence onto the hand in `urdf`."""
    import mujoco
    from scipy.spatial import cKDTree

    m, qadr, dadr = load_model(urdf, meshdir)
    d = mujoco.MjData(m)
    tip_ids, palm_id = _bodies(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    jidx = [names.index(n) for n in DOF_ORDER]
    qlo = np.array([m.jnt_range[i, 0] if m.jnt_limited[i] else -np.inf for i in jidx])
    qhi = np.array([m.jnt_range[i, 1] if m.jnt_limited[i] else np.inf for i in jidx])

    human = seq.hands[side]
    if human is None:
        raise ValueError(f"sequence {seq.name} has no {side}")
    start, length = window or (0, seq.T)
    frames = np.arange(start, min(start + length, seq.T))

    tree = cKDTree(seq.obj_mesh[0])                       # object frame, built once
    q = np.zeros(22)
    q[:3] = human.wrist_pos[frames[0]]                    # start the base at the wrist
    out = np.zeros((len(frames), 22))
    errs = np.zeros(len(frames))
    cons = np.zeros((len(frames), 4), bool)

    for i, k in enumerate(frames):
        tips_w = human.tips[k][MANO_TIPS]                 # (4, 3) world
        tips_o = seq.to_object(tips_w, k)                 # object frame
        dist, idx = tree.query(tips_o)
        touching = dist < contact_tol
        # a touching tip is pulled to the surface point it touched, in world
        surf_o = seq.obj_mesh[0][idx]
        surf_w = surf_o @ seq.obj_R[k].T + seq.obj_pos[k]
        target = np.where(touching[:, None], surf_w, tips_w)
        w = np.where(touching, 1.0, W_FREE)
        wrist = human.wrist_pos[k]
        q_prev = q.copy()

        for _ in range(iters):
            d.qpos[:] = 0
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, res = [], []
            for t, (bid, tgt, wt) in enumerate(zip(tip_ids, target, w)):
                jacp = np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, d, jacp, None, bid)
                rows.append(wt * jacp[:, dadr]); res.append(wt * (tgt - d.xpos[bid]))
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, None, palm_id)
            rows.append(W_WRIST * jacp[:, dadr]); res.append(W_WRIST * (wrist - d.xpos[palm_id]))
            rows.append(W_SMOOTH * np.eye(22)); res.append(W_SMOOTH * (q_prev - q))
            J = np.vstack(rows); r = np.concatenate(res)
            step = np.linalg.solve(J.T @ J + LAMBDA * np.eye(22), J.T @ r)
            q = np.clip(q + np.clip(step, -0.15, 0.15), qlo, qhi)
            if np.abs(step).max() < 1e-5:
                break

        d.qpos[:] = 0
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        err = np.linalg.norm(np.stack([d.xpos[b] for b in tip_ids]) - target, axis=1)
        out[i], errs[i], cons[i] = q, float(err[touching].mean() if touching.any() else err.mean()), touching
        if verbose and i % 25 == 0:
            print(f"  frame {k:4d}  tip err {errs[i]*1000:6.2f} mm  contacts {int(touching.sum())}/4", flush=True)

    return Fit(q=out, tip_err_m=errs, contacts=cons, frames=frames)


def main() -> None:
    import sys
    sys.path.insert(0, "src")
    from handsim.human import grab as grab_mod

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="e.g. s1/binoculars_see_1")
    ap.add_argument("--side", default="lhand", choices=["lhand", "rhand"])
    ap.add_argument("--urdf", type=Path, default=None, help="defaults to the side's fly urdf")
    ap.add_argument("--assets", type=Path, default=Path("out/dextrack_assets"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--length", type=int, default=0, help="0 = to the end of the clip")
    ap.add_argument("--stride", type=int, default=8, help="GRAB is 120 Hz; DexTrack references are 15 Hz")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--validate", type=Path, help="their reference for this clip, to compare against")
    a = ap.parse_args()

    urdf = a.urdf or a.assets / (f"allegro_hand_description_{'left' if a.side == 'lhand' else 'right'}_fly_v2.urdf")
    seq = grab_mod.load(f"{a.clip}.npz", verts=False, stride=a.stride)
    print(f"{a.clip}: {seq.T} frames at stride {a.stride}, object {seq.obj}, fitting {a.side} onto {urdf.name}")
    fit = fit_sequence(seq, a.side, urdf, window=(a.start, a.length or seq.T),
                       meshdir=a.assets / "meshes", verbose=True)
    print(f"fitted {len(fit.frames)} frames: tip error mean {fit.tip_err_m.mean()*1000:.2f} mm, "
          f"median {np.median(fit.tip_err_m)*1000:.2f} mm, contacts {fit.contacts.sum(1).mean():.2f}/4 per frame")

    if a.validate:
        ref = np.load(a.validate, allow_pickle=True).item()
        theirs = ref["robot_delta_states_weights_np"]
        n = min(len(theirs), len(fit.q))
        base = np.abs(theirs[:n, :3] - fit.q[:n, :3])
        joints = np.abs(theirs[:n, 6:] - fit.q[:n, 6:])
        print(f"against their right-hand reference over {n} frames:")
        print(f"  base translation  mean {base.mean()*1000:6.1f} mm   max {base.max()*1000:6.1f} mm")
        print(f"  finger joints     mean {np.degrees(joints.mean()):6.1f} deg  max {np.degrees(joints.max()):6.1f} deg")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(a.out, {"q": fit.q, "frames": fit.frames, "tip_err_m": fit.tip_err_m,
                        "contacts": fit.contacts, "clip": a.clip, "side": a.side,
                        "urdf": str(urdf), "stride": a.stride}, allow_pickle=True)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
