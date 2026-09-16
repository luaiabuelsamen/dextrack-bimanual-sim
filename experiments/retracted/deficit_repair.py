"""The claim: where one hand cannot achieve force closure, two hands can.

For every (hand, object width) the approach is SEARCHED -- orientation, standoff
and closure fraction -- by cross-entropy method, once with one hand and once
with two. Each candidate is produced by closing the hand in simulation, so every
pose is physically realisable, and the best one found is then validated by
pushing the object in fourteen directions until it slips.

Two numbers per cell:

    eps        Ferrari-Canny epsilon of the REAL MuJoCo contact set
    hold_N     the largest force the grasp survives in its worst direction

A cell where the best single-hand epsilon is 0 and the best two-hand epsilon is
not is the claim of this project, stated as a measurement: the deficit is a
property of the hand and the object, and the second hand is the repair.

Nothing here uses human data. The wrench requirement is the specification and
the hands are fitted to it, which is the point -- a keypoint objective needs a
demonstration to copy, and a wrench objective does not.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from oppdef.grasping.synth import GraspScene


def cem(scene, n_hands, iters=6, pop=24, elite=6, seed=0, verbose=False):
    """Search approach parameters for the best epsilon this hand can reach."""
    rng = np.random.default_rng(seed)
    dim = 5 * n_hands
    # rx, ry, rz over the full sphere of orientations; standoff 0-5 cm;
    # closure fraction 0.5-1. A second hand starts from the opposite side,
    # which is a starting point for the search, not a constraint on it.
    mu = np.zeros(dim)
    for h in range(n_hands):
        mu[5 * h:5 * h + 3] = [0.0, np.pi * h, 0.0]
        mu[5 * h + 3] = 0.005
        mu[5 * h + 4] = 0.9
    sd = np.tile([1.6, 1.6, 1.6, 0.015, 0.15], n_hands)

    best = None
    for it in range(iters):
        cand = rng.normal(mu, sd, size=(pop, dim))
        # standoff may be NEGATIVE: for a short-fingered hand the object is
        # admitted by sitting deeper in the palm, not further out.
        cand[:, 3::5] = np.clip(cand[:, 3::5], -0.03, 0.055)
        cand[:, 4::5] = np.clip(cand[:, 4::5], 0.45, 1.0)
        scored = []
        for c in cand:
            a = scene.attempt(c, do_hold=False)
            score = a.epsilon if a.valid else -1.0
            scored.append((score, c, a))
            if a.valid and (best is None or a.epsilon > best.epsilon):
                best = a
        scored.sort(key=lambda t: -t[0])
        top = np.array([c for _s, c, _a in scored[:elite]])
        if top[:, 0].size and scored[0][0] > -1:
            mu = top.mean(0)
            sd = np.maximum(top.std(0), [0.25, 0.25, 0.25, 0.004, 0.03] * n_hands)
        if verbose:
            print(f"    iter {it}: best eps so far "
                  f"{(best.epsilon if best else 0):.4f}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+", default=["shadow", "leap", "allegro", "f5d6"])
    ap.add_argument("--widths", type=float, nargs="+",
                    default=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--out", default="results/deficit_repair.json")
    a = ap.parse_args()

    rows = []
    hdr = (f"{'hand':8s} {'w_cm':>5s} {'hands':>5s} {'eps':>7s} {'cts':>4s} "
           f"{'F_N':>7s} {'hold_N':>7s} {'pen_mm':>6s} {'s':>6s}")
    print(hdr); print("-" * len(hdr), flush=True)
    for hk in a.hands:
        for w in a.widths:
            for nh in (1, 2):
                t0 = time.time()
                try:
                    sc = GraspScene(hk, (w / 2,) * 3, mass=a.mass,
                                    n_hands=nh, kp_finger=a.kp)
                except Exception as e:
                    print(f"{hk:8s} {w*100:5.1f} {nh:5d}  BUILD FAIL "
                          f"{type(e).__name__}: {str(e)[:60]}", flush=True)
                    continue
                best = cem(sc, nh, iters=a.iters, pop=a.pop, seed=a.seed)
                if best is None:
                    print(f"{hk:8s} {w*100:5.1f} {nh:5d} {'--':>7s} {0:4d} "
                          f"{0:7.2f} {0:7.2f} {'--':>6s} {time.time()-t0:6.1f}",
                          flush=True)
                    rows.append(dict(hand=hk, width=w, n_hands=nh, epsilon=0.0,
                                     n_contacts=0, f_total=0.0, hold_N=0.0,
                                     found=False))
                    continue
                hold, per = (0.0, None)
                if best.n_contacts >= 2:
                    sc.attempt(best.params, do_hold=False)
                    hold, per = sc.hold_of()
                print(f"{hk:8s} {w*100:5.1f} {nh:5d} {best.epsilon:7.4f} "
                      f"{best.n_contacts:4d} {best.f_total:7.2f} {hold:7.2f} "
                      f"{best.penetration_mm:6.2f} {time.time()-t0:6.1f}",
                      flush=True)
                rows.append(dict(hand=hk, width=w, n_hands=nh,
                                 epsilon=float(best.epsilon),
                                 n_contacts=int(best.n_contacts),
                                 f_total=float(best.f_total), hold_N=float(hold),
                                 penetration_mm=float(best.penetration_mm),
                                 params=best.params.tolist(), found=True,
                                 per_direction_N=(per.tolist() if per is not None
                                                  else None)))
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
