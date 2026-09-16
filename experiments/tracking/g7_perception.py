"""Stage 7: does the tracker survive a pose it had to SEE rather than be told?

The controller is frozen. Only the object pose it reads is swapped, between:

    truth     the simulator's own state, which no robot has
    depth     ICP against the known mesh, on a rendered depth image
    noise     the truth plus Gaussian error of the same magnitude as `depth`

The third condition is what separates two explanations that the second cannot
tell apart. If `depth` degrades tracking and `noise` of the same size does not,
the estimator's error is structured -- biased, or correlated across frames --
and the fix is perception. If both degrade it equally, the controller is simply
sensitive to pose error, and the fix is control.

Scored on the TRUTH in every condition: a controller that believes a wrong pose
must not also be graded against its own belief.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from oppdef.human import grab, track, perception as P


def run(seq_name, hand="shadow", alpha=0.4, cam_offset=(0.35, -0.30, 0.25),
        width=200, noise_m=0.0, seed=0):
    s = grab.load(seq_name, verts=True, stride=8)
    rt = track.ReferenceTracker(s, hand)
    gf = rt.grasp_frames()
    if len(gf) == 0:
        return None
    k0 = int(gf[0])

    cam = P.Camera(width, width, 45.0,
                   rt.ref_pos[k0] + np.array(cam_offset), rt.ref_pos[k0])
    sensor = P.DepthSensor(rt.sim.model, cam, rt.sim.obj_gids)
    model_pts = P.object_model_points(rt.sim.model, rt.sim.obj_gids, max_pts=6000)
    rng = np.random.default_rng(seed)
    out = {}

    try:
        # ---- calibrate the estimator's error, and run each condition -------
        for mode in ("truth", "depth", "noise"):
            rt.pose_source = None
            rt.reset_at(k0)
            est = P.PoseEstimator(model_pts, max_pts=4000)
            tp, tq = rt.true_obj_pose()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, tq)
            est.reset(R.reshape(3, 3), tp)
            errs, pose_err = [], []

            state = {"p": tp.copy(), "q": tq.copy()}
            if mode == "depth":
                def src(_st=state):
                    return _st["p"], _st["q"]
                rt.pose_source = src
            elif mode == "noise":
                sd = out.get("depth", {}).get("pose_err_mm", 20.0) / 1000.0

                def src(_st=state):
                    return _st["p"], _st["q"]
                rt.pose_source = src

            for k in range(k0, rt.T):
                tp, tq = rt.true_obj_pose()
                if mode == "depth":
                    cl = sensor.cloud(rt.sim.data, lookat=tp, noise_m=noise_m,
                                      rng=rng)
                    Re, te = est.update(cl)
                    state["p"], state["q"] = te, est.quat()
                    pose_err.append(float(np.linalg.norm(te - tp)))
                elif mode == "noise":
                    state["p"] = tp + rng.normal(scale=sd, size=3)
                    state["q"] = tq
                    pose_err.append(float(np.linalg.norm(state["p"] - tp)))

                rt.apply(k, alpha=alpha)
                for _ in range(rt.ctrl_every):
                    mujoco.mj_step(rt.sim.model, rt.sim.data)
                errs.append(rt.error(k)[0])

            errs = np.array(errs)
            out[mode] = {
                "mean_mm": float(errs.mean() * 1000),
                "final_mm": float(errs[-1] * 1000),
                "held_frac": float((errs < 0.05).mean()),
                "dropped": bool(errs[-1] > 0.10),
                "pose_err_mm": float(np.mean(pose_err) * 1000) if pose_err else 0.0,
                "steps": int(len(errs))}
    finally:
        rt.pose_source = None
        sensor.close()

    out["seq"] = s.name
    out["object"] = s.obj
    out["start"] = k0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="s1/mug_drink_1.npz,s1/apple_lift.npz")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--out", default="results/g7_perception.json")
    a = ap.parse_args()
    rows = []
    for nm in a.seqs.split(","):
        r = run(nm, alpha=a.alpha)
        if r is None:
            print(f"{nm}: no graspable frame")
            continue
        rows.append(r)
        print(f"\n{r['seq']} ({r['object']}), from frame {r['start']}")
        for m in ("truth", "depth", "noise"):
            v = r[m]
            print(f"  {m:6s} track {v['mean_mm']:8.1f} mm mean, "
                  f"{v['final_mm']:9.1f} final, held {v['held_frac']:.2f}"
                  + (f", pose err {v['pose_err_mm']:6.1f} mm" if v['pose_err_mm'] else ""))
    if rows:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.out}")
