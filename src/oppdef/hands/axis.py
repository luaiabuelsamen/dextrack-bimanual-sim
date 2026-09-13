"""The opposition axis: how closely can a hand's thumb meet its fingers?

RESTATED 2026-09-13. The previous implementation of this file measured the
distance from a thumb BODY ORIGIN to the MEAN of finger body origins, with no
self-collision, no joint coupling, and f5d6's eleven joints treated as
independent. Every one of those was wrong, and the numbers it produced --
including the 3.1-3.4 cm f5d6 "deficit" this project was named for -- are
retracted. See NOTES 2026-09-12.

What is measured now:

    floor    = min over poses of || thumb_tip(q) - nearest fingertip(q) ||
    aperture = max of the same quantity

subject to joint limits, with the hand's mimic couplings enforced, penalising
any pose in which two finger bodies interpenetrate, and with the fingertip
taken to be the far end of each distal link's own collision geometry rather
than a body origin. Multi-start.

The measurement and the hand registry both live elsewhere now (`hands.model`,
`hands.specs`) so this file cannot drift away from the solver again -- which is
exactly what it did.

A caveat the number cannot carry by itself: a small floor says the thumb can
approach a finger, not that the resulting contact normals, reachable object
placements and torque limits can support any particular task.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import minimize

from oppdef.hands.specs import SPECS, load, hand_joints
from oppdef.hands.model import (tip_bodies, finger_body_set, mimic_pairs,
                                apply_mimic, self_penetration, thumb_gap)

PEN_WEIGHT = 50.0


@dataclass
class Axis:
    hand: str
    floor_m: float
    aperture_m: float
    floor_self_pen_mm: float
    n_joints: int
    restarts: int
    cap_rad: float
    pen_weight: float
    tips: tuple
    note: str = ""


def _extreme(key, sign, restarts, seed, cap):
    m, cfg = load(key)
    d = mujoco.MjData(m)
    names, ids, offs = tip_bodies(m, cfg)
    bodies = finger_body_set(m, cfg, ids)
    pairs = mimic_pairs(m)
    jids = hand_joints(m, cfg)
    qadr = np.array([m.jnt_qposadr[j] for j in jids])
    r = m.jnt_range[jids]
    lo, hi = r[:, 0].copy(), r[:, 1].copy()
    bad = hi <= lo
    lo[bad], hi[bad] = -np.pi, np.pi
    lo, hi = np.clip(lo, -cap, cap), np.clip(hi, -cap, cap)

    def obj(x):
        d.qpos[:] = 0.0
        d.qpos[qadr] = x
        apply_mimic(m, d, pairs)
        mujoco.mj_kinematics(m, d)
        g = thumb_gap(m, d, ids, offs)
        return sign * g + PEN_WEIGHT * self_penetration(m, d, bodies)

    rng = np.random.default_rng(seed)
    best = None
    for i in range(restarts):
        x0 = np.zeros(len(jids)) if i == 0 else lo + rng.random(len(jids)) * (hi - lo)
        res = minimize(obj, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                       options=dict(maxiter=600))
        if best is None or res.fun < best.fun:
            best = res
    d.qpos[:] = 0.0
    d.qpos[qadr] = np.clip(best.x, lo, hi)
    apply_mimic(m, d, pairs)
    mujoco.mj_kinematics(m, d)
    # the GAP at the solution, never the penalised objective -- conflating them
    # once reported a 34.57 cm "floor" that was a penalty
    return (thumb_gap(m, d, ids, offs), self_penetration(m, d, bodies) * 1000.0,
            len(jids), names)


def opposition_axis(key, restarts=24, seed=0, cap=1.3):
    floor, pen, nj, names = _extreme(key, +1.0, restarts, seed, cap)
    aperture, _p, _n, _t = _extreme(key, -1.0, restarts, seed, cap)
    return Axis(hand=key, floor_m=floor, aperture_m=aperture,
                floor_self_pen_mm=pen, n_joints=nj, restarts=restarts,
                cap_rad=cap, pen_weight=PEN_WEIGHT, tips=tuple(names),
                note=SPECS[key].get("note", ""))


def provenance():
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True).strip()
        except Exception:
            return "unknown"
    return dict(commit=_git("rev-parse", "HEAD"),
                dirty=bool(_git("status", "--porcelain")),
                mujoco=mujoco.__version__, numpy=np.__version__,
                python=platform.python_version(), machine=platform.machine(),
                when=time.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+", default=list(SPECS))
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/opposition_axis.json")
    a = ap.parse_args()
    rows = []
    print(f"{'hand':9s} {'floor_cm':>9s} {'aperture_cm':>12s} "
          f"{'self_pen_mm':>12s} {'joints':>7s}")
    for k in a.hands:
        ax = opposition_axis(k, restarts=a.restarts, seed=a.seed)
        rows.append(asdict(ax))
        print(f"{k:9s} {ax.floor_m*100:9.2f} {ax.aperture_m*100:12.2f} "
              f"{ax.floor_self_pen_mm:12.2f} {ax.n_joints:7d}", flush=True)
    out = dict(provenance=provenance(), hands=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
