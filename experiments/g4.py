"""G4 Part A: is the G3 task outcome reproducible?

Pre-registered in docs/G4_PREREGISTRATION.md, committed before this file
existed. A metric can only predict a stable property; if the same grasp gives
different outcomes on repeated runs, G3's null is about the simulator.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import time

import numpy as np
import mujoco

from oppdef.grasping.synth import GraspScene
from oppdef.hands.axis import provenance
from oppdef.grasping.task import run_task
from experiments.g3 import task_spec

R = 5


def load_grasps(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        blob = json.loads(pathlib.Path(f).read_text())
        rows += blob["rows"] if isinstance(blob, dict) else blob
    return rows


def one_run(sc, row, traj, rng, perturb):
    """Re-form the saved grasp, perturb the start state, run the task."""
    at = sc.attempt(np.array(row["params"]), do_hold=False)
    if not at.valid or at.n_contacts < 2:
        return None
    if perturb:
        # Amendment 1: the +/-2% mass perturbation is applied as an equivalent
        # STEADY FORCE on the object rather than by mutating body_mass.
        # Mutating mass needs mj_setConst to rebuild derived inertia, and
        # restoring the scalar afterwards did not restore everything: grasps
        # then failed to re-form intermittently, in an alternating pattern, and
        # 10 of 12 determinism-check failures were re-formation failures rather
        # than physics differences. The model is now never mutated mid-run.
        sc._mass_bias = np.array([0.0, 0.0,
                                  -9.81 * float(row["mass"])
                                  * rng.uniform(-0.02, 0.02)])
        # object start offset +/-1 mm, applied AFTER the grasp formed, so the
        # grasp itself is identical across repeats
        sc.d.qpos[sc.obj_q:sc.obj_q + 3] += rng.uniform(-0.001, 0.001, 3)
        sc.d.qacc_warmstart[:] = 0.0        # no repeat inherits another's path
        mujoco.mj_forward(sc.m, sc.d)
    else:
        sc._mass_bias = None
    return run_task(sc, traj, extra_force=getattr(sc, "_mass_bias", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+",
                    default=["shadow", "leap", "allegro", "f5d6"])
    ap.add_argument("--glob", default="results/g3_*.json")
    ap.add_argument("--repeats", type=int, default=R)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/g4.json")
    a = ap.parse_args()

    prov = provenance()
    grasps = [g for g in load_grasps(a.glob) if g["hand"] in a.hands]
    print("G4 Part A -- pre-registered in docs/G4_PREREGISTRATION.md")
    print(f"provenance: commit {prov['commit'][:10]} dirty={prov['dirty']}")
    print(f"{len(grasps)} grasps x 2 tasks x "
          f"({a.repeats} perturbed + 1 determinism check)\n")
    hdr = (f"{'hand':8s} {'shape':9s} {'#':>3s} {'task':>5s} {'orig':>5s} "
           f"{'det':>5s} {'repeats':>12s} {'unanimous':>10s} {'s':>5s}")
    print(hdr); print("-" * len(hdr), flush=True)

    rows = []
    for i, g in enumerate(grasps):
        half = np.array(g["half"])
        trajA, _rA = task_spec(half, g["mass"], 0.5, "x")
        trajB, _rB = task_spec(half, g["mass"], 0.3, "y")
        for tname, traj, orig in (("a", trajA, g["task_a"]),
                                  ("b", trajB, g["task_b"])):
            t0 = time.time()
            rng = np.random.default_rng(a.seed + 1000 * i + (0 if tname == "a" else 1))
            # one scene, reused: safe now that run_task hands gravity back,
            # and the regression test asserts a grasp re-forms identically in a
            # scene a task has already run in
            sc = GraspScene(g["hand"], tuple(half), mass=g["mass"], n_hands=1,
                            kp_finger=g["kp_finger"], shape=g["shape"])
            det = one_run(sc, g, traj, rng, perturb=False)
            reps = []
            for _ in range(a.repeats):
                r = one_run(sc, g, traj, rng, perturb=True)
                reps.append(None if r is None else bool(r.success))
            good = [x for x in reps if x is not None]
            unan = bool(good) and all(x == good[0] for x in good)
            det_ok = (det is not None and bool(det.success) == bool(orig))
            print(f"{g['hand']:8s} {g['shape']:9s} {i:3d} {tname:>5s} "
                  f"{str(orig):>5s} {str(det_ok):>5s} "
                  f"{''.join('1' if x else ('0' if x is False else '-') for x in reps):>12s} "
                  f"{str(unan):>10s} {time.time()-t0:5.1f}", flush=True)
            rows.append(dict(hand=g["hand"], shape=g["shape"], grasp=i,
                             task=tname, original=bool(orig),
                             determinism_ok=bool(det_ok),
                             det_success=(None if det is None else bool(det.success)),
                             repeats=reps, unanimous=unan,
                             n_valid_repeats=len(good),
                             majority=(None if not good
                                       else bool(sum(good) * 2 > len(good)))))
            pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(a.out).write_text(json.dumps(
                dict(provenance=prov, args=vars(a),
                     preregistration="docs/G4_PREREGISTRATION.md",
                     rows=rows), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
