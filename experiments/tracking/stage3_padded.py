"""Does a grasp-seeded policy still let go when trained on as much data?

Stage 3's headline -- grasp-seeded references drop, burial-seeded ones track --
is confounded. The burial-seeded references are also the best-supported:
mug_drink_2 trained on 27 start frames and tracked at 5.7 mm; hammer_use_2
trained on 4 and let go. Both differences run the same way and that run cannot
separate them.

This removes the confound the cheap way, which a peer session used on the mug:
override `grasp_frames` and train on every candidate start rather than only the
ones that hold under feedforward. If hammer still lets go with a comparable
number of starts, sample size is not the explanation and the seed class is.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from oppdef.human import grab, track, rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="hammer_use_2")
    ap.add_argument("--steps", type=int, default=160_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grips", default="results/stage2_grips_g9.json")
    ap.add_argument("--out", default="results/stage3_padded.json")
    a = ap.parse_args()

    grips = {x["seq"]: x for x in json.loads(Path(a.grips).read_text())["rows"]}
    row = grips[a.seq]
    seq = grab.load(f"{row['subject']}/{a.seq}.npz", verts=True, stride=8)
    rt = track.ReferenceTracker(seq, hand="shadow")
    rt.apply_wrist_offset(np.asarray(row["offset"], float))

    accepted = rt.grasp_frames()
    padded = np.arange(0, rt.T, a.stride)
    print(f"{a.seq}: {len(accepted)} accepted starts, {len(padded)} padded "
          f"(T={rt.T}); seed {row['n_contact']} contacts at {row['grip_n']:.1f} N",
          flush=True)

    span = int(rt.T - (accepted[0] if len(accepted) else 0))
    cfg = rl.RLConfig(n_envs=a.envs, horizon=min(224, max(64, span + 8)),
                      seed=a.seed)
    cfg.iters = max(24, a.steps // (cfg.n_envs * cfg.horizon))
    t0 = time.time()
    net, log = rl.train(rt, cfg, starts=padded, verbose=True)

    k0 = int(accepted[0]) if len(accepted) else 0
    errs, _ = rl.evaluate(rt, net, start=k0)
    pen_mm, _ = rt.sim.penetration()
    grip_n, ncon = track.total_grip(rt.sim)
    out = {
        "seq": a.seq, "object": row["object"], "steps": a.steps,
        "accepted_starts": int(len(accepted)), "padded_starts": int(len(padded)),
        "seed_contacts": int(row["n_contact"]), "seed_grip_n": float(row["grip_n"]),
        "ppo_mm": float(errs.mean() * 1000),
        "end_pen_mm": float(pen_mm * 1000), "end_contacts": int(ncon),
        "end_grip_n": float(grip_n), "seconds": time.time() - t0,
    }
    print(f"\n{a.seq}: PPO {out['ppo_mm']:.1f} mm  ends {out['end_pen_mm']:.2f} mm in, "
          f"{ncon} contacts, {grip_n:.0f} N   ({len(padded)} starts vs "
          f"{len(accepted)} accepted)", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
