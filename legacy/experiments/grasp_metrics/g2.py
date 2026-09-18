"""G2: does a demonstration specify the TASK?

Pre-registered in docs/G2_PREREGISTRATION.md, committed before this file
existed. Do not change the arms, outcomes or analysis without an Amendment
there.

    pose_squeeze   keypoint retarget + searched squeeze  -- the hand's pose
    task_generic   maximise epsilon                      -- no demonstration
    task_demo      maximise margin against the required  -- the object's
                   wrench sequence                          trajectory

Identical budget, identical executor. Success is slip in the palm frame during
the carry, which no objective is scored on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from handsim.bench import Cell, keypoint_pose
from handsim.hands.axis import provenance
from handsim.grasping.task import (carry, object_path, required_wrenches, run_task,
                         grasp_wrench_capacity)

POSE, GENERIC, DEMO = "pose_squeeze", "task_generic", "task_demo"
ARMS = (POSE, GENERIC, DEMO)


def task_spec(width, mass, seg_s):
    """The demonstration: an object trajectory and the wrenches it demands."""
    traj = carry(seg_s=seg_s)
    P, Q = object_path(traj, np.zeros(3), np.array([1.0, 0, 0, 0]))
    half = width / 2
    inertia = mass * (2 * (2 * half) ** 2) / 12.0
    lam = float(np.linalg.norm([half] * 3))
    req = required_wrenches(P, Q, traj.dt, mass, [inertia] * 3, lam)
    return traj, req


def search(cell, arm, kp_vec, req, iters, pop, seed):
    rng = np.random.default_rng(seed)
    nf = len(cell.names)
    closure = np.array([cell.base[n] for n in cell.names])
    if arm == POSE:
        dim = 6
        mu = np.concatenate([[0.0, 0.0, 0.0, 0.005, 0.9], [0.3]])
        sd = np.array([1.6, 1.6, 1.6, 0.015, 0.15, 0.30])

        def targets(c):
            s = float(np.clip(c[5], 0.0, 1.0))
            return cell.targets(kp_vec + s * (closure - kp_vec))
    else:
        dim = 5 + nf
        mu = np.concatenate([[0.0, 0.0, 0.0, 0.005, 0.9], closure])
        sd = np.concatenate([[1.6, 1.6, 1.6, 0.015, 0.15], [0.45] * nf])

        def targets(c):
            return cell.targets(c[5:])

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
            if arm == DEMO:
                # per UNIT contact force: the margin is support x f_total,
                # so scoring it raw rewards squeezing rather than contact
                # placement -- run 1's demo arm squeezed 3.6x harder than the
                # generic arm and corr(margin, f_total) was +0.961.
                _m, _e, _f = grasp_wrench_capacity(cell.scene, req)
                sc = _m / max(_f, 1e-9)
            elif arm == GENERIC:
                sc = a.epsilon
            else:
                sc = cell.fidelity()
            scored.append((sc, c))
            if sc > best_score:
                best, best_score, best_t = a, sc, tgt
        scored.sort(key=lambda t: -t[0])
        top = np.array([c for _s, c in scored[:max(pop // 4, 4)]])
        if scored[0][0] > -1e9:
            mu = top.mean(0)
            floor = ([0.25, 0.25, 0.25, 0.004, 0.03] +
                     ([0.05] if arm == POSE else [0.08] * nf))
            sd = np.maximum(top.std(0), np.array(floor))
    return best, best_t, tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+",
                    default=["shadow", "leap", "allegro", "f5d6"])
    ap.add_argument("--widths", type=float, nargs="+",
                    default=[0.045, 0.06, 0.075])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--seg-s", type=float, default=0.5)
    ap.add_argument("--out", default="legacy/results/g2.json")
    a = ap.parse_args()

    prov = provenance()
    print("G2 -- pre-registered in docs/G2_PREREGISTRATION.md")
    print(f"provenance: commit {prov['commit'][:10]} dirty={prov['dirty']}\n")
    hdr = (f"{'hand':8s} {'w_cm':>5s} {'sd':>3s} {'arm':13s} {'margin':>8s} "
           f"{'eps':>6s} {'P1 task':>8s} {'slip_cm':>8s} {'P2 hold':>8s} "
           f"{'valid':>6s} {'s':>6s}")
    print(hdr); print("-" * len(hdr), flush=True)
    rows = []
    for hk in a.hands:
        for w in a.widths:
            traj, req = task_spec(w, a.mass, a.seg_s)
            kp_map, _e = keypoint_pose(hk, w)
            for sd in a.seeds:
                cell = Cell(hk, w, a.mass, a.kp)
                kp_vec = np.array([kp_map.get(n, 0.0) for n in cell.names])
                for arm in ARMS:
                    t0 = time.time()
                    best, tgt, tally = search(cell, arm, kp_vec, req,
                                              a.iters, a.pop, sd)
                    margin = eps = f_tot = 0.0
                    p2 = 0.0
                    res = None
                    if best is not None and best.n_contacts >= 2:
                        # re-form the chosen grasp, measure both primaries
                        cell.scene.attempt(best.params, do_hold=False,
                                           finger_target=tgt)
                        margin, eps, f_tot = grasp_wrench_capacity(cell.scene, req)
                        p2, _per = cell.scene.hold_of()       # P2, static probe
                        cell.scene.attempt(best.params, do_hold=False,
                                           finger_target=tgt)
                        res = run_task(cell.scene, traj)      # P1, the task
                    ok = bool(res.success) if res else False
                    slip = res.max_slip_m if res else float("nan")
                    print(f"{hk:8s} {w*100:5.1f} {sd:3d} {arm:13s} {margin:8.2f} "
                          f"{eps:6.3f} {str(ok):>8s} {slip*100:8.2f} {p2:8.3f} "
                          f"{tally['valid']:6d} {time.time()-t0:6.1f}", flush=True)
                    rows.append(dict(
                        hand=hk, width=w, seed=sd, arm=arm,
                        task_success=ok, max_slip_m=float(slip),
                        hold_N=float(p2), margin=float(margin),
                        epsilon=float(eps), f_total=float(f_tot),
                        n_contacts=int(best.n_contacts if best else 0),
                        reason=(res.reason if res else "no grasp found"),
                        candidates=a.iters * a.pop, rejections=tally,
                        params=(best.params.tolist() if best is not None else None),
                        finger_target=tgt, mass=a.mass, kp_finger=a.kp,
                        seg_s=a.seg_s, seconds=round(time.time() - t0, 2)))
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps(
                        dict(provenance=prov, args=vars(a),
                             preregistration="docs/G2_PREREGISTRATION.md",
                             rows=rows), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
