"""Retarget the hand POSE, or retarget the WRENCH? Both, in physics.

The geometric version of this comparison was retracted: its poses had fingers
19 mm inside the object. This one runs both conditions through the same
physical harness, where a pose that cannot be realised simply does not score.

    pose      the finger angles come from matching the human's fingertip
              geometry (keypoint retargeting, `oppdef.retarget`). The hand's
              PLACEMENT is then searched, so the condition is given every
              chance -- only the finger angles are dictated by the human.

    closure   placement searched, fingers driven along the hand's own derived
              closure with only a scalar fraction free. This is NOT a wrench
              objective -- it is the hand's canonical grasp -- and it was
              mislabelled "wrench" in the first version of this comparison,
              which gave the wrench side ONE free number against the pose
              side's full finger specification and produced a null.

    wrench    placement AND all finger angles chosen to maximise the
              Ferrari-Canny epsilon of the contact set actually measured in
              simulation. No human data is used. This is the condition the
              thesis is about.

Both are closed in simulation, measured on real MuJoCo contacts, and pushed in
14 directions until they slip. The placement search is identical and gets the
same budget in both, so the comparison isolates what decides the fingers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from oppdef.synth import GraspScene
from oppdef.data import SyntheticSource


def keypoint_pose(hand_key, width):
    """Finger angles that best match the human's inter-fingertip geometry."""
    from oppdef.retarget import retargeter_for, transform_ref, KEYPOINT
    rt = retargeter_for(hand_key, free_base=False)
    ref = list(SyntheticSource(widths=(width,), n_per=1))[0]
    V, obj, half, sc, _demo, _wrist = transform_ref(rt, ref, width=width)
    r = rt.fit(V, half, objective=KEYPOINT, obj_pos=obj, restarts=10, scale=sc)
    out = {}
    for i, j in enumerate(rt.jids):
        n = mujoco.mj_id2name(rt.m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        out[n] = float(r.q[i])
    return out, float(r.keypoint_err_m)


def search_fingers(scene, iters=8, pop=32, seed=0):
    """Optimise placement AND every finger angle for measured epsilon.

    The geometric retargeter could not do this: free-space joint search put
    fingers 19 mm inside the object. Here the same search is safe, because a
    pose that buries its fingers is rejected by the penetration gate before it
    can score -- the simulator enforces what the surrogate could not.
    """
    base = {n[len(scene.prefixes[0]):]: t
            for n, (_qa, _a, t) in scene.finger[scene.prefixes[0]].items()}
    names = list(base)
    lo, hi = {}, {}
    for n in names:
        jid = mujoco.mj_name2id(scene.m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{scene.prefixes[0]}{n}")
        a_, b_ = scene.m.jnt_range[jid]
        lo[n], hi[n] = (a_, b_) if b_ > a_ else (-np.pi, np.pi)
    nf = len(names)
    rng = np.random.default_rng(seed)
    mu = np.concatenate([[0.0, 0.0, 0.0, 0.005, 0.9],
                         [base[n] for n in names]])
    sd = np.concatenate([[1.6, 1.6, 1.6, 0.015, 0.15],
                         [0.45] * nf])
    best, best_t = None, None
    for _it in range(iters):
        cand = rng.normal(mu, sd, size=(pop, 5 + nf))
        cand[:, 3] = np.clip(cand[:, 3], -0.03, 0.055)
        cand[:, 4] = np.clip(cand[:, 4], 0.45, 1.0)
        scored = []
        for c in cand:
            tgt = {n: float(np.clip(c[5 + i], lo[n], hi[n]))
                   for i, n in enumerate(names)}
            a = scene.attempt(c[:5], do_hold=False, finger_target=tgt)
            scored.append(((a.epsilon if a.valid else -1.0), c))
            if a.valid and (best is None or a.epsilon > best.epsilon):
                best, best_t = a, tgt
        scored.sort(key=lambda t: -t[0])
        top = np.array([c for _s, c in scored[:8]])
        if scored[0][0] > -1:
            mu = top.mean(0)
            sd = np.maximum(top.std(0),
                            np.concatenate([[0.25, 0.25, 0.25, 0.004, 0.03],
                                            [0.08] * nf]))
    return best, best_t


def search(scene, n_hands, finger_target=None, iters=6, pop=24, seed=0):
    """Search placement (and closure, when the fingers are not dictated)."""
    rng = np.random.default_rng(seed)
    dim = 5 * n_hands
    mu = np.zeros(dim)
    for h in range(n_hands):
        mu[5 * h:5 * h + 3] = [0.0, np.pi * h, 0.0]
        mu[5 * h + 3] = 0.005
        mu[5 * h + 4] = 0.9
    sd = np.tile([1.6, 1.6, 1.6, 0.015, 0.15], n_hands)
    best = None
    for _it in range(iters):
        cand = rng.normal(mu, sd, size=(pop, dim))
        cand[:, 3::5] = np.clip(cand[:, 3::5], -0.03, 0.055)
        cand[:, 4::5] = np.clip(cand[:, 4::5], 0.45, 1.0)
        scored = []
        for c in cand:
            a = scene.attempt(c, do_hold=False, finger_target=finger_target)
            scored.append(((a.epsilon if a.valid else -1.0), c))
            if a.valid and (best is None or a.epsilon > best.epsilon):
                best = a
        scored.sort(key=lambda t: -t[0])
        top = np.array([c for _s, c in scored[:6]])
        if scored[0][0] > -1:
            mu = top.mean(0)
            sd = np.maximum(top.std(0), [0.25, 0.25, 0.25, 0.004, 0.03] * n_hands)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+", default=["shadow", "leap", "allegro"])
    ap.add_argument("--widths", type=float, nargs="+",
                    default=[0.04, 0.05, 0.06, 0.07])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--masses", type=float, nargs="+", default=[0.05])
    ap.add_argument("--conditions", nargs="+",
                    default=["pose", "closure", "wrench"])
    ap.add_argument("--out", default="results/pose_vs_wrench.json")
    a = ap.parse_args()

    hdr = (f"{'hand':8s} {'w_cm':>5s} {'condition':10s} {'eps':>7s} {'cts':>4s} "
           f"{'hold_N':>7s} {'pen_mm':>6s} {'kp_err':>7s} {'s':>6s}")
    print(hdr); print("-" * len(hdr), flush=True)
    rows = []
    for hk in a.hands:
      for mass in a.masses:
        for w in a.widths:
            tgt, kperr = keypoint_pose(hk, w)
            for cond in a.conditions:
                t0 = time.time()
                sc = GraspScene(hk, (w / 2,) * 3, n_hands=1, kp_finger=a.kp,
                                mass=mass)
                use = tgt if cond == "pose" else None
                if cond == "wrench":
                    # same number of simulated attempts as the other two, spent
                    # on fingers as well as placement
                    best, use = search_fingers(sc, iters=a.iters,
                                               pop=a.pop * 5 // 4)
                else:
                    best = search(sc, 1, finger_target=use,
                                  iters=a.iters, pop=a.pop)
                hold = 0.0
                if best is not None and best.n_contacts >= 2:
                    sc.attempt(best.params, do_hold=False, finger_target=use)
                    hold, _per = sc.hold_of()
                e = best.epsilon if best else 0.0
                nc = best.n_contacts if best else 0
                pen = best.penetration_mm if best else 0.0
                print(f"{hk:8s} {w*100:5.1f} {cond:10s} {e:7.4f} {nc:4d} "
                      f"{hold:7.2f} {pen:6.2f} {kperr*100:6.2f}cm "
                      f"{time.time()-t0:6.1f}", flush=True)
                rows.append(dict(hand=hk, width=w, mass=mass, condition=cond,
                                 epsilon=float(e), n_contacts=int(nc),
                                 hold_N=float(hold), penetration_mm=float(pen),
                                 keypoint_err_m=kperr))
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
