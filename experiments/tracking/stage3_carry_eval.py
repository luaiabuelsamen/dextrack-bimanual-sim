"""Held-out starts for the carry-seed policies: does PPO generalise past the
frame it trained around?

`stage3_carry.py` scored each policy from the carry start it was trained
around (+/-2 frames). That is the one start where it is guaranteed to have
seen the state. This rolls every policy AND the feedforward from every
other frame of the reference, at a stride, and reports the fraction of starts
that end held and clean, split into the trained neighbourhood and everything
else. Same seeds file, same offsets, same checkpoints, live end state.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from oppdef.human import grab, track as T, rl
from experiments.tracking.stage2_carry import contact_state, LOST_M, CLEAN_GRIP_X, W


def summary(errs, cs):
    pen, n, nb, sided, grip = cs
    held = bool(errs[-1] <= LOST_M)
    return {"mean_mm": float(errs.mean() * 1000), "end_mm": float(errs[-1] * 1000),
            "end_held": held, "end_pen_mm": pen * 1000, "end_bodies": nb,
            "end_sided": sided, "end_grip_n": grip,
            "end_clean": bool(held and pen < 0.003 and nb >= 2 and sided < 0.8
                              and grip / W < CLEAN_GRIP_X)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="results/stage2_carry_robust_seeds.json")
    ap.add_argument("--stage3", default="results/stage3_carry.json")
    ap.add_argument("--ckpt", default="results/ppo_carry")
    ap.add_argument("--out", default="results/stage3_carry_eval.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--near", type=int, default=2, help="trained neighbourhood half-width")
    a = ap.parse_args()

    seeds = {f"{r['subject']}/{r['seq']}": r
             for r in json.loads(Path(a.seeds).read_text())["rows"]}
    trained = json.loads(Path(a.stage3).read_text())["rows"]
    out = []
    for t in trained:
        key = f"{t['subject']}/{t['seq']}"
        row = seeds[key]
        seq = grab.load(f"{key}.npz", verts=True, stride=8)
        rt = T.ReferenceTracker(seq, hand="shadow")
        rt.apply_wrist_offset(np.asarray(row["offset"], float))
        net = rl.make_policy(rt.n_obs, rt.n_action)
        net.load_state_dict(torch.load(Path(a.ckpt) / f"ppo_{t['seq']}_carry.pt",
                                       map_location="cpu"))
        net.eval()
        k0 = int(t["start"])
        starts = list(range(0, rt.T - 5, a.stride))
        if k0 not in starts:
            starts.append(k0)
        per = []
        for k in sorted(starts):
            rt.reset_at(k)
            ff = summary(rt.rollout(start=k, steps=rt.T - k).pos_err, contact_state(rt.sim))
            errs, _ = rl.evaluate(rt, net, start=k)
            pp = summary(errs, contact_state(rt.sim))
            per.append({"start": k, "near": abs(k - k0) <= a.near, "ff": ff, "ppo": pp})

        def agg(sel, who):
            if not sel:
                return {}
            return {"n": len(sel),
                    "held": sum(x[who]["end_held"] for x in sel),
                    "clean": sum(x[who]["end_clean"] for x in sel),
                    "median_mm": float(np.median([x[who]["mean_mm"] for x in sel]))}
        near = [x for x in per if x["near"]]
        far = [x for x in per if not x["near"]]
        rec = {"seq": t["seq"], "subject": t["subject"], "trained_start": k0,
               "T": int(rt.T), "stride": a.stride,
               "near": {"ff": agg(near, "ff"), "ppo": agg(near, "ppo")},
               "far": {"ff": agg(far, "ff"), "ppo": agg(far, "ppo")},
               "starts": per}
        out.append(rec)
        f, p = rec["far"]["ff"], rec["far"]["ppo"]
        print(f"{t['seq']:24s} T={rt.T:3d} trained@{k0:3d}  held-out starts n={f.get('n',0):2d}: "
              f"FF held {f.get('held',0):2d} clean {f.get('clean',0):2d} med {f.get('median_mm',0):7.1f} mm | "
              f"PPO held {p.get('held',0):2d} clean {p.get('clean',0):2d} med {p.get('median_mm',0):7.1f} mm",
              flush=True)
        Path(a.out).write_text(json.dumps({"stride": a.stride, "near": a.near,
                                           "rows": out}, indent=1))


if __name__ == "__main__":
    main()
