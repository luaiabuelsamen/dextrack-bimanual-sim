"""G3 Part 2: does any grasp metric predict task success?

Pre-registered in docs/G3_PREREGISTRATION.md, committed before this file
existed.

Grasps are SAMPLED, not optimised. The question is whether a metric ranks
grasps by task success across the quality range; a search that concentrates on
high-epsilon grasps would truncate exactly the variation under test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from oppdef.grasping.synth import GraspScene
from oppdef.hands.axis import provenance
from oppdef.grasping.hold import DIRECTIONS
from oppdef.grasping.task import (carry, object_path, required_wrenches, run_task,
                         grasp_wrench_capacity)

SHAPES = ("box", "cylinder", "sphere", "capsule")


def task_spec(half, mass, seg_s, axis="x"):
    traj = carry(seg_s=seg_s)
    if axis != "x":
        traj.cmd = traj.cmd.copy()
        col = {"x": 3, "y": 4, "z": 5}[axis]
        traj.cmd[:, col] = traj.cmd[:, 3]
        if col != 3:
            traj.cmd[:, 3] = 0.0
        traj.tilt_axis = axis
    P, Q = object_path(traj, np.zeros(3), np.array([1.0, 0, 0, 0]))
    inertia = mass * (2 * (2 * float(np.mean(half))) ** 2) / 12.0
    lam = float(np.linalg.norm(half))
    req = required_wrenches(P, Q, traj.dt, mass, [inertia] * 3, lam)
    return traj, req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+",
                    default=["shadow", "leap", "allegro", "f5d6"])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--half", type=float, nargs=3, default=[0.022, 0.022, 0.030])
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--per-cell", type=int, default=15)
    ap.add_argument("--max-draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/g3.json")
    a = ap.parse_args()

    prov = provenance()
    half = np.array(a.half)
    trajA, reqA = task_spec(half, a.mass, 0.5, "x")
    trajB, reqB = task_spec(half, a.mass, 0.3, "y")
    print("G3 Part 2 -- pre-registered in docs/G3_PREREGISTRATION.md")
    print(f"provenance: commit {prov['commit'][:10]} dirty={prov['dirty']}\n")
    hdr = (f"{'hand':8s} {'shape':9s} {'#':>3s} {'eps':>6s} {'hold':>6s} "
           f"{'margin':>8s} {'cts':>4s} {'F_N':>7s} {'T3a':>5s} {'T3b':>5s} {'s':>5s}")
    print(hdr); print("-" * len(hdr), flush=True)
    rows = []
    for hk in a.hands:
        for shape in a.shapes:
            # a stable digest, not Python's hash(): hash() is salted per
            # process, so rerunning the same command with the same recorded
            # seed sampled DIFFERENT grasps and the saved provenance was not
            # enough to reproduce the dataset
            cell_id = int(hashlib.sha256(f"{hk}/{shape}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(a.seed * 1_000_003 + cell_id)
            sc = GraspScene(hk, tuple(half), mass=a.mass, n_hands=1,
                            kp_finger=a.kp, shape=shape)
            kept, draws = 0, 0
            while kept < a.per_cell and draws < a.max_draws:
                draws += 1
                c = np.concatenate([rng.uniform(-np.pi, np.pi, 3),
                                    [rng.uniform(-0.03, 0.055)],
                                    [rng.uniform(0.45, 1.0)]])
                at = sc.attempt(c, do_hold=False)
                if not at.valid or at.n_contacts < 2:
                    continue
                t0 = time.time()
                mA, eps, f_tot = grasp_wrench_capacity(sc, reqA)
                mB, _e, _f = grasp_wrench_capacity(sc, reqB)
                hold, _per = sc.hold_of()
                sc.attempt(c, do_hold=False)
                rA = run_task(sc, trajA)
                sc.attempt(c, do_hold=False)
                rB = run_task(sc, trajB)
                kept += 1
                print(f"{hk:8s} {shape:9s} {kept:3d} {eps:6.3f} {hold:6.3f} "
                      f"{mA:8.2f} {at.n_contacts:4d} {f_tot:7.2f} "
                      f"{str(rA.success):>5s} {str(rB.success):>5s} "
                      f"{time.time()-t0:5.1f}", flush=True)
                rows.append(dict(
                    hand=hk, shape=shape, draw=draws,
                    epsilon=float(eps), hold_N=float(hold),
                    margin=float(mA), margin_per_N=float(mA / max(f_tot, 1e-9)),
                    margin_b=float(mB), n_contacts=int(at.n_contacts),
                    f_total=float(f_tot), penetration_mm=float(at.penetration_mm),
                    task_a=bool(rA.success), task_b=bool(rB.success),
                    slip_a=float(rA.max_slip_m), slip_b=float(rB.max_slip_m),
                    reason_a=rA.reason, reason_b=rB.reason,
                    params=c.tolist(), mass=a.mass, kp_finger=a.kp,
                    half=list(map(float, half))))
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(
                    dict(provenance=prov, args=vars(a),
                         preregistration="docs/G3_PREREGISTRATION.md",
                         rows=rows), indent=2))
            print(f"  -> {hk}/{shape}: {kept} grasps from {draws} draws",
                  flush=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
