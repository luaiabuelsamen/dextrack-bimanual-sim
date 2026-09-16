"""Does the contact bookkeeping in `rl.Pool.step` cost training time?

The penetration cost and hold proxy leave the reward bit-for-bit unchanged at
their defaults (tests/test_rl_pen.py), but the STEP is not unchanged: every
env on every control step now walks the contact list, and -- unless gated --
calls `mj_contactForce` per contact. On a buried hand that is ~30 contacts x
12 envs x 160 steps ~ 58k calls an iteration, and this repo has been bitten
by "reward identical, training 15% slower" before (a per-actuator Python loop
was 11% of wall clock).

So: a MANY-contact reference with its buried stage-2 seed (bowl_drink_1 sits
on 94 contacts, gamecontroller_play_1 on 57), and N PPO iterations timed on

    3216767            rl.py as it was before the terms existed
    branch gated       the branch at defaults (force only on logged iterations)
    branch always-on   the branch with log_every=1 (force every step)

run ALTERNATING -- A B C A B C ... -- so thermal drift and whatever else the
machine is doing land on all three equally, same seed, median s/iter each.

    PYTHONPATH=src python experiments/infra/bench_rl_step.py --iters 3 --reps 3

Do not read the numbers while another run holds the cores.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from oppdef import paths
from oppdef.human import grab, track, rl

if str(paths.REPO) not in sys.path:     # track.py imports experiments.* itself
    sys.path.insert(0, str(paths.REPO))


def load_rl_at(rev: str):
    """`rl.py` exactly as committed at `rev`, as its own module."""
    src = subprocess.run(
        ["git", "-C", str(paths.REPO), "show", f"{rev}:src/oppdef/human/rl.py"],
        capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader(f"rl_{rev}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclasses resolve string annotations here
    exec(compile(src, f"rl.py@{rev}", "exec"), mod.__dict__)
    return mod


def build(ref: str, hand: str, grips_path: Path):
    """The tracker as g9_ppo_distill.per_reference builds and seeds it."""
    grips = {x["seq"]: x for x in json.loads(grips_path.read_text())["rows"]}
    row = grips[ref]
    seq = grab.load(f"{row['subject']}/{ref}.npz", verts=True, stride=8)
    rt = track.ReferenceTracker(seq, hand=hand)
    rt.apply_wrist_offset(np.asarray(row["offset"], float))
    gf = rt.grasp_frames()
    if len(gf) == 0:
        rt.synthesize_grasp()
        gf = rt.grasp_frames()
    if len(gf) == 0:
        raise SystemExit(f"{ref}: no graspable frame")
    return rt, gf, row


def main(a):
    base = load_rl_at(a.rev)
    t0 = time.time()
    rt, gf, row = build(a.ref, a.hand, Path(a.grips))
    rt.reset_at(int(gf[0]))
    cs = rl.contact_summary(rt.sim)
    print(f"{a.ref}: stage-2 seed {row['n_contact']} contacts; at frame {int(gf[0])} "
          f"now {cs.n} contacts on {cs.bodies} bodies, {cs.pen * 1000:.2f} mm, "
          f"{cs.grip:.0f} N  ({time.time() - t0:.0f}s to build)", flush=True)
    span = int(rt.T - gf[0])
    horizon = a.horizon or min(224, max(64, span + 8))

    def cfg_for(mod, **kw):
        c = mod.RLConfig(n_envs=a.envs, horizon=horizon, iters=a.iters, seed=a.seed)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    variants = [
        (a.rev, base, lambda: cfg_for(base)),
        ("branch gated", rl, lambda: cfg_for(rl)),
        ("branch always-on", rl, lambda: cfg_for(rl, log_every=1)),
    ]
    times = {name: [] for name, _m, _c in variants}
    # Untimed warm-up: torch's first import and first network otherwise land
    # on whichever variant runs first (5.8 s against 0.85 s on a 1-rep smoke).
    for _name, mod, mk in variants:
        c = mk()
        c.iters = 1
        mod.train(rt, c, starts=gf, verbose=False)
    print(f"{a.envs} envs x {horizon} horizon x {a.iters} iters, {a.reps} reps, alternating",
          flush=True)
    for rep in range(a.reps):
        for name, mod, mk in variants:
            cfg = mk()
            t0 = time.perf_counter()
            mod.train(rt, cfg, starts=gf, verbose=False)
            dt = (time.perf_counter() - t0) / a.iters
            times[name].append(dt)
            print(f"  rep {rep}  {name:18s} {dt:7.3f} s/iter", flush=True)

    ref_med = statistics.median(times[a.rev])
    print("median s/iter:")
    for name, _m, _c in variants:
        med = statistics.median(times[name])
        print(f"  {name:18s} {med:7.3f}   x{med / ref_med:.3f} of {a.rev}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default="bowl_drink_1",
                   help="reference; needs a row in --grips")
    p.add_argument("--rev", default="3216767", help="baseline commit of rl.py")
    p.add_argument("--iters", type=int, default=3, help="PPO iterations per timing")
    p.add_argument("--reps", type=int, default=3, help="alternating repetitions")
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--horizon", type=int, default=0, help="0 = g9's rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hand", default="shadow")
    p.add_argument("--grips", default=str(paths.RESULTS / "stage2_grips_g9.json"))
    main(p.parse_args())
