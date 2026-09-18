"""Does action chunking beat the one-step MLP on the bimanual task?

Front 5 built the chunked policy; this measures it. Both policies see exactly
the same demonstrations, the same observation, the same training budget and the
same evaluation seeds, and differ only in the horizon -- `horizon=1` IS the
old baseline, so the comparison is a controlled one rather than a new pipeline
being compared against an old one.

Control rate matters and is easy to get wrong. Demonstrations are decimated by
`stride` (the expert is recorded at the 500 Hz sim rate, which no policy should
be asked to run at), so a chunk of H actions covers H*stride simulator steps.
The evaluation therefore queries the policy every `stride` steps and HOLDS its
action in between -- which is what a real control loop does, and what keeps the
executed sequence on the distribution the policy was fitted to. Querying every
sim step instead would consume a chunk H times too fast.

Baselines are run every time. A learned policy that does not beat do-nothing
and random has not been shown to do anything.

EVERY HORIZON IS TRAINED FROM SEVERAL SEEDS, because one run per horizon does
not measure the horizon. Run once each, this comparison reported horizon 8 at
0/8 with the peg driven 6.9 cm the WRONG way, sitting between horizon 1 and
horizon 16 at 8/8 -- a hole that no property of chunking explains. Retrained
with a different torch seed on the same demonstrations, horizon 8 scored 8/8.
The spread across seeds is larger than any difference between horizons, so a
single-seed table would have reported initialisation luck as an architecture
result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from handsim.learning.bc import rollout_expert, observe
from handsim.envs.bimanual import BASE_HALF
from handsim.policy import train


def collect_episodes(n, stride=4, seed0=1000):
    """Per-episode (obs, act) pairs -- NOT concatenated.

    Chunking needs episode boundaries: a window that runs off one demonstration
    into the next teaches the policy to follow a trajectory with a different peg
    pose, and it looks exactly like ordinary training noise.
    """
    eps, meta = [], []
    t0 = time.time()
    for i in range(n):
        o, a, r, _ = rollout_expert(seed=seed0 + i)
        if not r["success"]:
            print(f"  demo {i:3}: DISCARDED (peg {r['peg_out_m']*100:.1f} cm)",
                  flush=True)
            continue
        eps.append((o[::stride], a[::stride]))
        meta.append(r)
        print(f"  demo {i:3}: peg {r['peg_out_m']*100:5.2f} cm  "
              f"{len(o[::stride])} pairs", flush=True)
    print(f"{len(eps)}/{n} demos kept in {time.time()-t0:.0f}s")
    return eps, meta


def run_episode(seed, policy=None, mode="policy", stride=4, n_steps=3470):
    """One evaluation episode, paired with the expert's draw for that seed."""
    from handsim.control.expert import Expert, palm_ctrl
    from handsim.envs.bimanual import LH_HOME

    rng = np.random.default_rng(seed)
    kw = dict(hinge_friction=float(rng.uniform(0.9, 1.8)),
              base_mass=float(rng.uniform(0.06, 0.12)))
    ex = Expert(two_handed=True, **kw)
    e = ex.e
    ex.pre_frac, _ = ex.start_at_pregrasp(
        palm_ctrl(LH_HOME, [0.0, -0.085, 0.044]))
    box0 = np.array([0.0, 0.0, BASE_HALF[2]])
    out0 = ex.out0
    if policy is not None:
        policy.reset()

    rs = np.random.default_rng(seed + 77)
    ctrl = np.zeros(e.m.nu)
    lo, hi = e.m.actuator_ctrlrange[:, 0], e.m.actuator_ctrlrange[:, 1]
    lim = e.m.actuator_ctrllimited.astype(bool)
    for k in range(n_steps):
        if mode == "policy" and k % stride == 0:
            # queried at the rate it was TRAINED at, held in between
            ctrl = policy(observe(e, k / n_steps))
        elif mode == "random":
            ctrl = ctrl + rs.normal(0, 0.01, e.m.nu)
        e.d.ctrl[:] = np.where(lim, np.clip(ctrl, lo, hi), ctrl)
        mujoco.mj_step(e.m, e.d)

    out = e.peg_out() - out0
    lift = float(e.box_pos()[2] - box0[2])
    return dict(peg_out_m=float(out), box_lift_m=lift,
                tilt_deg=e.box_tilt_deg(),
                success=bool(out >= 0.08 and lift < 0.02
                             and e.box_tilt_deg() < 15.0))


def evaluate(label, episodes, **kw):
    rs = [run_episode(s, **kw) for s in episodes]
    ok = np.array([r["success"] for r in rs], float)
    peg = np.array([r["peg_out_m"] for r in rs]) * 100
    print(f"  {label:22s} success {ok.mean()*100:5.1f}% "
          f"({int(ok.sum())}/{len(ok)})   peg {peg.mean():6.2f} +/- "
          f"{peg.std():5.2f} cm", flush=True)
    return dict(label=label, success=float(ok.mean()),
                peg_mean_cm=float(peg.mean()), peg_std_cm=float(peg.std()),
                episodes=[float(p) for p in peg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=24)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ensemble-m", type=float, default=0.01)
    ap.add_argument("--out", default="legacy/results/chunk_bc.json")
    a = ap.parse_args()

    print(f"=== collecting {a.demos} demos (stride {a.stride}) ===", flush=True)
    eps, meta = collect_episodes(a.demos, stride=a.stride)
    if not eps:
        raise SystemExit("no successful demonstrations")

    seeds = [9000 + i for i in range(a.eval_episodes)]
    rows = []
    print("\n=== baselines ===", flush=True)
    rows.append(evaluate("do-nothing", seeds, mode="donothing",
                         stride=a.stride))
    rows.append(evaluate("random", seeds, mode="random", stride=a.stride))

    print("\n=== expert (upper reference) ===", flush=True)
    ex_ok, ex_peg = [], []
    for s in seeds:
        _o, _a, r, _ = rollout_expert(seed=s, record=False)
        ex_ok.append(r["success"]); ex_peg.append(r["peg_out_m"] * 100)
    print(f"  {'expert':22s} success {np.mean(ex_ok)*100:5.1f}% "
          f"   peg {np.mean(ex_peg):6.2f} +/- {np.std(ex_peg):5.2f} cm")
    rows.append(dict(label="expert", success=float(np.mean(ex_ok)),
                     peg_mean_cm=float(np.mean(ex_peg)),
                     peg_std_cm=float(np.std(ex_peg))))

    for h in a.horizons:
        base = "MLP (horizon 1)" if h == 1 else f"chunked (horizon {h})"
        print(f"\n=== {base} ===", flush=True)
        per_seed = []
        for ts in a.train_seeds:
            p = train(eps, horizon=h, epochs=a.epochs, seed=ts,
                      m=a.ensemble_m, log=None)
            r = evaluate(f"{base} seed {ts}", seeds, policy=p, mode="policy",
                         stride=a.stride)
            r["horizon"], r["train_seed"] = h, ts
            rows.append(r)
            per_seed.append(r)
        sc = np.array([r["success"] for r in per_seed])
        pg = np.array([r["peg_mean_cm"] for r in per_seed])
        print(f"  {'-> across seeds':22s} success {sc.mean()*100:5.1f}% "
              f"+/- {sc.std()*100:4.1f}   peg {pg.mean():6.2f} +/- "
              f"{pg.std():5.2f} cm", flush=True)
        rows.append(dict(label=f"{base} SUMMARY", horizon=h,
                         success_mean=float(sc.mean()),
                         success_std=float(sc.std()),
                         peg_mean_cm=float(pg.mean()),
                         peg_std_cm=float(pg.std()),
                         n_seeds=len(a.train_seeds)))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
