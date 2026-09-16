"""Stage 5, done the way DexTrack does it: distil the PPO POLICIES.

Every earlier attempt cloned a trajectory optimiser, and none of them worked:

    plain cloning of MPPI's correction   held-out ratio 2.235 (worse than a constant)
    cloning the absolute command         ratio 0.716, and drops the object in the loop
    DAgger, four variants                round 0 best, every round after it worse

The diagnosis was the same each time and it points here. A trajectory optimiser's
output is not a function of the state, and an open-loop expert cannot say how to
recover from a state the policy wandered into. A PPO policy is state-conditioned
by construction and IS a closed-loop expert -- that is why DexTrack distils one.

So: train PPO per reference, roll each policy out on its own reference, record
what it saw and what it did, and fit one network to all of it. Held out by
OBJECT, because frames within a clip are near-duplicates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import mujoco

from oppdef.human import grab, track, rl, distill


def per_reference(row, hand="shadow", steps=200_000, seed=0, verbose=False,
                  n_envs=16):
    """Train one PPO policy and return it with its environment."""
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = track.ReferenceTracker(seq, hand=hand)
    rt.synthesize_grasp()
    gf = rt.grasp_frames()
    if len(gf) == 0:
        return None, None, None
    # The training horizon must COVER the reference. PPO learns a correction
    # over the window it trains on and does not extrapolate past it: on
    # mug_drink_1 a 64-step policy scored 10535 mm on a 111-step rollout and a
    # 160-step policy scored 29.4 mm, tracking every frame. Matching the horizon
    # to the task was worth more than a reshaped reward or ten times the steps.
    span = int(rt.T - gf[0])
    cfg = rl.RLConfig(n_envs=n_envs, horizon=min(224, max(64, span + 8)), seed=seed)
    cfg.iters = max(24, steps // (cfg.n_envs * cfg.horizon))
    net, _log = rl.train(rt, cfg, starts=gf, verbose=verbose)
    return rt, net, gf


def harvest(rt, net, gf, per_start=200, max_starts=6):
    """On-policy states and the policy's own commands, as distillation data."""
    cfg = rl.RLConfig()
    scale = np.concatenate([np.full(3, cfg.a_pos), np.full(3, cfg.a_rot),
                            np.full(rt.n_action - 6, cfg.a_fin)])
    O, A = [], []
    for k0 in np.asarray(gf)[:max_starts]:
        k0 = int(k0)
        rt.reset_at(k0)
        for k in range(k0, min(k0 + per_start, rt.T)):
            o = rt.observe(k)
            with torch.no_grad():
                a = net.dist(torch.as_tensor(o, dtype=torch.float32)[None]).mean.numpy()[0]
            rt.apply(k, np.clip(a, -1, 1) * scale)
            for _ in range(rt.ctrl_every):
                mujoco.mj_step(rt.sim.model, rt.sim.data)
            O.append(o)
            A.append(distill.expert_label(rt, k))   # the ABSOLUTE command it produced
    return np.array(O), np.array(A)


def main(a):
    inv = json.loads(Path("results/grab_inventory.json").read_text())["rows"]
    rows = [r for r in inv if "error" not in r and r["rhand_hold_len"] >= 25]
    seen, picked = {}, []
    for r in sorted(rows, key=lambda r: (-r["rhand_hold_len"], r["seq"])):
        if seen.get(r["object"], 0) >= 1:
            continue
        picked.append(r)
        seen[r["object"]] = 1
        if len(picked) >= a.n:
            break

    store, envs = [], []
    for i, r in enumerate(picked):
        t0 = time.time()
        rt, net, gf = per_reference(r, steps=a.steps, seed=a.seed,
                                    hand=a.hand, n_envs=a.envs)
        if rt is None:
            print(f"[{i+1}/{len(picked)}] {r['seq']}: no graspable frame", flush=True)
            continue
        O, A = harvest(rt, net, gf)
        errs, _ = rl.evaluate(rt, net, start=int(gf[0]))
        store.append({"obj": r["object"], "seq": r["seq"], "O": O, "A": A,
                      "ppo_mm": float(errs.mean() * 1000)})
        envs.append((rt, r["object"]))
        print(f"[{i+1}/{len(picked)}] {r['seq']:26s} {r['object']:12s} "
              f"PPO {errs.mean()*1000:7.1f} mm  {len(O)} transitions "
              f"({time.time()-t0:.0f}s)", flush=True)

    if len(store) < 3:
        print("too few references solved")
        return

    objs = sorted({s["obj"] for s in store})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(objs)
    test = set(objs[:max(1, int(round(0.3 * len(objs))))])
    tr = [s for s in store if s["obj"] not in test]
    print(f"\nholding out objects {sorted(test)}", flush=True)

    from oppdef.learning.bc import train as bc_train
    O = np.concatenate([s["O"] for s in tr]).astype(np.float32)
    A = np.concatenate([s["A"] for s in tr]).astype(np.float32)
    model = bc_train(O, A, seed=a.seed, epochs=a.epochs)

    print("\nheld-out objects, distilled policy vs that reference's own PPO:")
    held = []
    for (rt, obj), s in zip(envs, store):
        if obj not in test:
            continue
        gf = rt.grasp_frames()
        k0 = int(gf[0]) if len(gf) else 0
        rt.reset_at(k0)
        ff = rt.rollout(start=k0, steps=rt.T - k0)
        pe = distill.policy_rollout(rt, model, start=k0)
        held.append({"seq": s["seq"], "object": obj,
                     "feedforward_mm": float(min(ff.pos_err.mean()*1000, 9e5)),
                     "ppo_mm": float(s["ppo_mm"]),
                     "distilled_mm": float(min(pe.mean()*1000, 9e5))})
        print(f"  {s['seq'][:24]:24s} {obj:12s} feedforward "
              f"{held[-1]['feedforward_mm']:8.1f} mm   "
              f"its own PPO {s['ppo_mm']:8.1f} mm   "
              f"DISTILLED {held[-1]['distilled_mm']:8.1f} mm", flush=True)

    # Persisted, because a stage that only prints cannot be audited later and
    # every number in this pipeline has had to be re-derived at least once.
    Path(a.out).write_text(json.dumps({
        "hand": a.hand, "steps": a.steps, "seed": a.seed,
        "held_out_objects": sorted(test),
        "per_reference": [{"seq": s["seq"], "object": s["obj"],
                           "ppo_mm": s["ppo_mm"],
                           "transitions": int(len(s["O"]))} for s in store],
        "held_out": held,
    }, indent=1))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--steps", type=int, default=160_000)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--out", default="results/g9_ppo_distill.json")
    main(ap.parse_args())
