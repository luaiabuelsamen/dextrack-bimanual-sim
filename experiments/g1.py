"""G1: does a wrench objective beat the pipeline practitioners actually ship?

Pre-registered in docs/G1_PREREGISTRATION.md, committed before this file
existed. Do not change the arms, outcome, or analysis here without adding an
Amendment there.

Three arms, identical budget, identical executor:

    pose_squeeze    keypoint retarget, then a searched squeeze toward the
                    hand's own closure -- what a real pipeline does
    eps_synth       maximise measured epsilon, started from the hand's closure,
                    no human data anywhere
    wrench_refine   maximise measured epsilon, started from the keypoint
                    retarget

`eps_synth` and `wrench_refine` differ ONLY in where the search starts, which
is what isolates whether the demonstration contributes anything once a wrench
objective is doing the optimising.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from oppdef.grasping.hold import DIRECTIONS
from oppdef.hands.axis import provenance
from oppdef.bench import Cell, keypoint_pose

POSE_SQUEEZE, EPS_SYNTH, WRENCH_REFINE = "pose_squeeze", "eps_synth", "wrench_refine"
ARMS = (POSE_SQUEEZE, EPS_SYNTH, WRENCH_REFINE)


def search(cell, arm, kp_vec, iters, pop, seed):
    """One arm. Every arm spends exactly `iters * pop` executor calls."""
    rng = np.random.default_rng(seed)
    nf = len(cell.names)
    closure = np.array([cell.base[n] for n in cell.names])

    if arm == POSE_SQUEEZE:
        # placement + ONE squeeze fraction: the finger pose is the retarget,
        # closed toward the hand's own closure by a searched amount
        dim = 6
        mu = np.concatenate([[0.0, 0.0, 0.0, 0.005, 0.9], [0.3]])
        sd = np.array([1.6, 1.6, 1.6, 0.015, 0.15, 0.30])

        def targets(c):
            s = float(np.clip(c[5], 0.0, 1.0))
            return cell.targets(kp_vec + s * (closure - kp_vec))
    else:
        dim = 5 + nf
        start = closure if arm == EPS_SYNTH else kp_vec
        mu = np.concatenate([[0.0, 0.0, 0.0, 0.005, 0.9], start])
        sd = np.concatenate([[1.6, 1.6, 1.6, 0.015, 0.15], [0.45] * nf])

        def targets(c):
            return cell.targets(c[5:])

    want_eps = arm in (EPS_SYNTH, WRENCH_REFINE)
    best, best_score, best_t = None, -np.inf, None
    tally = {"valid": 0, "few_contacts": 0}
    for _it in range(iters):
        cand = rng.normal(mu, sd, size=(pop, dim))
        cand[:, 3] = np.clip(cand[:, 3], -0.03, 0.055)
        cand[:, 4] = np.clip(cand[:, 4], 0.45, 1.0)
        scored = []
        for c in cand:
            tgt = targets(c)
            a = cell.scene.attempt(c[:5], do_hold=False, finger_target=tgt)
            if not a.valid or a.n_contacts < 2:
                if a.valid:
                    key = "few_contacts"
                elif "pushed" in a.reason:
                    key = "object_ejected"
                elif "buried" in a.reason:
                    key = "fingers_buried"
                elif "cannot reach" in a.reason:
                    key = "unreachable"
                elif "starts inside" in a.reason:
                    key = "pregrasp_penetrating"
                else:
                    key = "invalid"
                tally[key] = tally.get(key, 0) + 1
                scored.append((-1e9, c))
                continue
            tally["valid"] += 1
            sc = a.epsilon if want_eps else cell.fidelity()
            scored.append((sc, c))
            if sc > best_score:
                best, best_score, best_t = a, sc, tgt
        scored.sort(key=lambda t: -t[0])
        top = np.array([c for _s, c in scored[:max(pop // 4, 4)]])
        if scored[0][0] > -1e9:
            mu = top.mean(0)
            floor = ([0.25, 0.25, 0.25, 0.004, 0.03] +
                     ([0.05] if arm == POSE_SQUEEZE else [0.08] * nf))
            sd = np.maximum(top.std(0), np.array(floor))
    return best, best_t, tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+",
                    default=["shadow", "leap", "allegro", "f5d6"])
    ap.add_argument("--widths", type=float, nargs="+",
                    default=[0.03, 0.045, 0.06, 0.075, 0.09])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--out", default="results/g1.json")
    a = ap.parse_args()

    prov = provenance()
    print(f"G1 -- pre-registered in docs/G1_PREREGISTRATION.md")
    print(f"provenance: commit {prov['commit'][:10]} dirty={prov['dirty']}\n")
    hdr = (f"{'hand':8s} {'w_cm':>5s} {'sd':>3s} {'arm':14s} {'eps':>7s} "
           f"{'kp_err_cm':>9s} {'hold':>7s} {'cts':>4s} {'valid':>6s} {'s':>6s}")
    print(hdr); print("-" * len(hdr), flush=True)
    rows = []
    for hk in a.hands:
        for w in a.widths:
            kp_map, _err = keypoint_pose(hk, w)
            for sd in a.seeds:
                cell = Cell(hk, w, a.mass, a.kp)
                kp_vec = np.array([kp_map.get(n, 0.0) for n in cell.names])
                for arm in ARMS:
                    t0 = time.time()
                    best, tgt, tally = search(cell, arm, kp_vec, a.iters,
                                              a.pop, sd)
                    hold, kp_err, per = 0.0, float("nan"), None
                    if best is not None and best.n_contacts >= 2:
                        cell.scene.attempt(best.params, do_hold=False,
                                           finger_target=tgt)
                        kp_err = -cell.fidelity()
                        hold, per = cell.scene.hold_of()
                    e = best.epsilon if best else 0.0
                    nc = best.n_contacts if best else 0
                    print(f"{hk:8s} {w*100:5.1f} {sd:3d} {arm:14s} {e:7.4f} "
                          f"{kp_err*100:9.2f} {hold:7.3f} {nc:4d} "
                          f"{tally['valid']:6d} {time.time()-t0:6.1f}",
                          flush=True)
                    rows.append(dict(
                        hand=hk, width=w, seed=sd, arm=arm,
                        epsilon=float(e), n_contacts=int(nc),
                        hold_N=float(hold), keypoint_err_m=float(kp_err),
                        penetration_mm=float(best.penetration_mm if best else 0),
                        candidates=a.iters * a.pop,
                        params=(best.params.tolist() if best is not None else None),
                        finger_target=tgt, rejections=tally,
                        per_direction_N=(per.tolist() if per is not None else None),
                        n_force_dirs=len(DIRECTIONS), n_torque_dirs=len(DIRECTIONS),
                        mass=a.mass, kp_finger=a.kp,
                        seconds=round(time.time() - t0, 2)))
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps(
                        dict(provenance=prov, args=vars(a),
                             preregistration="docs/G1_PREREGISTRATION.md",
                             rows=rows), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
