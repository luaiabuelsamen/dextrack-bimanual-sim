"""One budget-matched experiment: does the OBJECTIVE matter, holding search equal?

Every earlier version of this comparison was confounded. The pose condition
searched 5 placement parameters while the wrench condition searched 5 plus every
finger angle, and the runner additionally gave wrench `pop*5//4` candidates --
180 against 144 -- under a comment claiming the budgets were equal. The outcome
measured was whether search finds a passing grasp, so extra attempts and extra
freedom both went straight to the endpoint.

Here the two conditions are identical in every respect except the scalar they
maximise:

    same search space     placement (5) + every finger angle
    same candidate count  iters x pop, both conditions
    same seeds            the same CEM seeds, cell by cell
    same object           one cube, and the reference grasp is built on it
    same outcome          worst direction of a 14-force + 14-torque probe

    POSE    maximise fidelity to the human's inter-fingertip geometry,
            measured on the pose the simulator actually settles at
    WRENCH  maximise the Ferrari-Canny epsilon of the measured contact set

Anything that separates them now is the objective, because nothing else differs.
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
from oppdef.hands.tips import tip_offset, tip_points
from oppdef.retarget import retargeter_for, transform_ref, tip_graph, correspond
from oppdef.hold import DIRECTIONS
from oppdef.hands.axis import provenance

POSE, WRENCH = "pose", "wrench"


def reference_graph(hand_key, width):
    """The human's inter-fingertip vectors, in this hand's frame, to scale."""
    rt = retargeter_for(hand_key, free_base=False)
    ref = list(SyntheticSource(widths=(width,), n_per=1))[0]
    V, _obj, _half, sc, _demo, _wrist = transform_ref(rt, ref, width=width)
    return rt.target_graph(V, sc)


class Cell:
    """One (hand, width) scene plus the machinery both conditions share."""

    def __init__(self, hand_key, width, mass, kp):
        self.scene = GraspScene(hand_key, (width / 2,) * 3, mass=mass,
                                n_hands=1, kp_finger=kp)
        s = self.scene
        self.pfx = s.prefixes[0]
        self.tip_b = s.tip_bids[self.pfx]
        self.tip_off = [tip_offset(s.m, b) for b in self.tip_b]
        self.palm_b = s.palm_bid[self.pfx]
        self.names = [n[len(self.pfx):] for n in s.finger[self.pfx]]
        self.base = {n[len(self.pfx):]: t
                     for n, (_q, _a, t) in s.finger[self.pfx].items()}
        self.lo, self.hi = {}, {}
        for n in self.names:
            j = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.pfx}{n}")
            a, b = s.m.jnt_range[j]
            self.lo[n], self.hi[n] = (a, b) if b > a else (-np.pi, np.pi)
        self.A_ref = reference_graph(hand_key, width)

    def achieved_graph(self):
        """Inter-fingertip vectors of the settled pose, in the PALM frame.

        Palm frame, not world: the base is free, so a world-frame graph would
        change when the hand is merely rotated and the fidelity score would
        measure placement instead of finger shape.
        """
        s = self.scene
        P = tip_points(s.m, s.d, self.tip_b, self.tip_off)
        R = s.d.xmat[self.palm_b].reshape(3, 3)
        G, _pairs = tip_graph((P - s.d.xpos[self.palm_b]) @ R)
        return G

    def fidelity(self):
        """Negative mean inter-fingertip vector error, in metres."""
        G = self.achieved_graph()
        n = min(len(G), len(self.A_ref))
        return -float(np.linalg.norm(G[:n] - self.A_ref[:n], axis=1).mean())

    def targets(self, vec):
        return {n: float(np.clip(vec[i], self.lo[n], self.hi[n]))
                for i, n in enumerate(self.names)}


def cem(cell, objective, iters, pop, seed):
    rng = np.random.default_rng(seed)
    nf = len(cell.names)
    mu = np.concatenate([[0.0, 0.0, 0.0, 0.005, 0.9],
                         [cell.base[n] for n in cell.names]])
    sd = np.concatenate([[1.6, 1.6, 1.6, 0.015, 0.15], [0.45] * nf])
    best, best_score, best_t = None, -np.inf, None
    tally = {"valid": 0, "few_contacts": 0}
    for _it in range(iters):
        cand = rng.normal(mu, sd, size=(pop, 5 + nf))
        cand[:, 3] = np.clip(cand[:, 3], -0.03, 0.055)
        cand[:, 4] = np.clip(cand[:, 4], 0.45, 1.0)
        scored = []
        for c in cand:
            tgt = cell.targets(c[5:])
            a = cell.scene.attempt(c[:5], do_hold=False, finger_target=tgt)
            # Both conditions select among GRASPS. Fidelity alone is maximised
            # by a hand matching the human's finger shape in mid-air, touching
            # nothing -- which is what the pose condition did on its first run
            # (0 contacts, best score). A retargeting pipeline that returns a
            # non-grasp is not the baseline anyone ships, so the contact
            # requirement is applied identically to both.
            if not a.valid or a.n_contacts < 2:
                key = ("few_contacts" if a.valid
                       else a.reason.split("(")[0].strip()[:40] or "invalid")
                tally[key] = tally.get(key, 0) + 1
                scored.append((-1e9, c)); continue
            tally["valid"] += 1
            sc = a.epsilon if objective == WRENCH else cell.fidelity()
            scored.append((sc, c))
            if sc > best_score:
                best, best_score, best_t = a, sc, tgt
        scored.sort(key=lambda t: -t[0])
        top = np.array([c for _s, c in scored[:max(pop // 4, 4)]])
        if scored[0][0] > -1e9:
            mu = top.mean(0)
            sd = np.maximum(top.std(0),
                            np.concatenate([[0.25, 0.25, 0.25, 0.004, 0.03],
                                            [0.08] * nf]))
    return best, best_score, best_t, tally


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
    ap.add_argument("--out", default="results/matched.json")
    a = ap.parse_args()

    hdr = (f"{'hand':8s} {'w_cm':>5s} {'sd':>3s} {'cond':7s} {'eps':>7s} "
           f"{'kp_err_cm':>9s} {'hold':>7s} {'cts':>4s} {'pen':>5s} {'s':>6s}")
    print(hdr); print("-" * len(hdr), flush=True)
    rows = []
    prov = provenance()
    print(f"provenance: commit {prov['commit'][:10]} dirty={prov['dirty']} "
          f"mujoco {prov['mujoco']}\n", flush=True)
    for hk in a.hands:
        for w in a.widths:
            for sd in a.seeds:
                cell = Cell(hk, w, a.mass, a.kp)
                for cond in (POSE, WRENCH):
                    t0 = time.time()
                    best, _score, tgt, tally = cem(cell, cond, a.iters,
                                                   a.pop, sd)
                    hold, kp_err, per = 0.0, float("nan"), None
                    if best is not None and best.n_contacts >= 2:
                        cell.scene.attempt(best.params, do_hold=False,
                                           finger_target=tgt)
                        kp_err = -cell.fidelity()
                        hold, per = cell.scene.hold_of()
                    e = best.epsilon if best else 0.0
                    nc = best.n_contacts if best else 0
                    pen = best.penetration_mm if best else 0.0
                    print(f"{hk:8s} {w*100:5.1f} {sd:3d} {cond:7s} {e:7.4f} "
                          f"{kp_err*100:9.2f} {hold:7.3f} {nc:4d} {pen:5.2f} "
                          f"{time.time()-t0:6.1f}", flush=True)
                    rows.append(dict(hand=hk, width=w, seed=sd, condition=cond,
                                     epsilon=float(e), n_contacts=int(nc),
                                     hold_N=float(hold),
                                     keypoint_err_m=float(kp_err),
                                     penetration_mm=float(pen),
                                     candidates=a.iters * a.pop,
                                     # everything needed to replay the chosen
                                     # grasp without repeating the search
                                     params=(best.params.tolist() if best
                                             is not None else None),
                                     finger_target=tgt,
                                     per_direction_N=(per.tolist()
                                                      if per is not None else None),
                                     n_force_dirs=len(DIRECTIONS),
                                     n_torque_dirs=len(DIRECTIONS),
                                     rejections=tally,
                                     mass=a.mass, kp_finger=a.kp,
                                     seconds=round(time.time() - t0, 2)))
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps(
                        dict(provenance=prov, args=vars(a), rows=rows),
                        indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
