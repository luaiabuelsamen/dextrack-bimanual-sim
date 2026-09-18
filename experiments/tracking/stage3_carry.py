"""Stage 3 on carry-scored seeds: can PPO add anything to a grasp that carries?

Every earlier stage-3 row started from a static-hold seed and either tracked a
burial or dropped a grasp, and the open-loop carry showed the controller barely
moved either outcome. The carry-scored seeds (`stage2_carry.py`, validated and
rendered by `render_carry.py`) are the first that end a feedforward carry held
AND out of the object. This trains one PPO policy per such seed, from the same
start frame the carry was scored on, and reports policy against feedforward on
the identical initial condition -- tracking error and the live end state
(penetration, contacts, links, grip), because a lower error that ends inside
the object is not an improvement.

Start frames are the carry start, not `grasp_frames()`: that is a static hold
test, and the whole point of these seeds is that they were not selected by one.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import mujoco
import torch

from handsim.human import grab, track as T, rl
from experiments.tracking.stage2_carry import contact_state, LOST_M


def feedforward(rt, k0):
    rt.reset_at(k0)                # rollout() does not reset; evaluate() does
    errs = rt.rollout(start=k0, steps=rt.T - k0).pos_err
    return errs, contact_state(rt.sim)


def policy(rt, net, k0):
    errs, _acts = rl.evaluate(rt, net, start=k0)
    return errs, contact_state(rt.sim)


def summary(errs, cs):
    pen, n, nb, sided, grip = cs
    ok = np.flatnonzero(errs <= LOST_M)
    return {"mean_mm": float(errs.mean() * 1000),
            "end_mm": float(errs[-1] * 1000),
            "carried_frac": float((int(ok[-1] + 1) if len(ok) else 0) / len(errs)),
            "end_held": bool(errs[-1] <= LOST_M),
            "end_pen_mm": pen * 1000, "end_contacts": n, "end_bodies": nb,
            "end_sided": sided, "end_grip_n": grip,
            "end_clean": bool(errs[-1] <= LOST_M and pen < 0.003 and nb >= 2
                              and sided < 0.8)}


def one(row, a):
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = T.ReferenceTracker(seq, hand="shadow")
    rt.apply_wrist_offset(np.asarray(row["offset"], float))
    k0 = int(row["start"])
    span = rt.T - k0
    starts = np.array(sorted({int(np.clip(k0 + d, 0, rt.T - 1))
                              for d in range(-a.jitter, a.jitter + 1)}))
    ff = summary(*feedforward(rt, k0))
    cfg = rl.RLConfig(n_envs=a.envs, horizon=min(224, max(64, span + 8)),
                      seed=a.seed, w_rot=a.w_rot, w_rot_lin=a.w_rot_lin)
    cfg.iters = max(24, a.steps // (cfg.n_envs * cfg.horizon))
    t0 = time.time()
    net, _log = rl.train(rt, cfg, starts=starts, verbose=a.verbose)
    train_s = time.time() - t0
    pol = summary(*policy(rt, net, k0))
    # a second feedforward AFTER training, so any drift in the tracker's state
    # between the two rollouts shows up as a disagreement here rather than as
    # a policy effect
    ff2 = summary(*feedforward(rt, k0))
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), Path(a.save) / f"ppo_{row['seq']}_carry.pt")
    return {"seq": row["seq"], "subject": row["subject"], "object": row["object"],
            "start": k0, "span": int(span), "starts": starts.tolist(),
            "steps": int(cfg.iters * cfg.n_envs * cfg.horizon),
            "horizon": int(cfg.horizon), "train_s": train_s,
            "feedforward": ff, "feedforward_after": ff2, "policy": pol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="results/stage2_carry.json")
    ap.add_argument("--out", default="results/stage3_carry.json")
    ap.add_argument("--save", default="results/ppo_carry")
    ap.add_argument("--seqs", default="", help="subset; default: every CLEAN row")
    ap.add_argument("--steps", type=int, default=160_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--jitter", type=int, default=2,
                    help="also start episodes this many frames either side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-rot", type=float, default=0.35)
    ap.add_argument("--w-rot-lin", type=float, default=0.0)
    ap.add_argument("--min-robust", type=int, default=0,
                    help="keep rows whose robust_clean is at least this")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    rows = json.loads(Path(a.seeds).read_text())["rows"]
    if a.seqs:
        want = set(x.strip() for x in a.seqs.split(","))
        rows = [r for r in rows if r["seq"] in want]
    else:
        rows = [r for r in rows if r.get("end", {}).get("clean")
                and r.get("robust_clean", 8) >= a.min_robust]
    print(f"{len(rows)} carry-scored seeds; {a.steps} steps x {a.envs} envs each",
          flush=True)
    out = []
    for i, r in enumerate(rows):
        try:
            rec = one(r, a)
        except Exception as e:                            # noqa: BLE001
            print(f"  {r['seq']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        out.append(rec)
        f, p = rec["feedforward"], rec["policy"]
        print(f"[{i+1}/{len(rows)}] {rec['seq']:22s} span {rec['span']:3d}  "
              f"FF {f['mean_mm']:7.1f} mm (end {f['end_pen_mm']:5.2f} mm, "
              f"{f['end_bodies']} links, {f['end_grip_n']:5.0f} N, "
              f"held={int(f['end_held'])})   PPO {p['mean_mm']:7.1f} mm "
              f"(end {p['end_pen_mm']:5.2f} mm, {p['end_bodies']} links, "
              f"{p['end_grip_n']:5.0f} N, held={int(p['end_held'])})  "
              f"{rec['train_s']/60:.0f} min", flush=True)
        Path(a.out).write_text(json.dumps(
            {"seeds": a.seeds, "steps": a.steps, "envs": a.envs,
             "jitter": a.jitter, "seed": a.seed, "w_rot": a.w_rot,
             "w_rot_lin": a.w_rot_lin, "rows": out}, indent=1))


if __name__ == "__main__":
    main()
