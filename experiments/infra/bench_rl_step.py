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
    branch defaults    the branch, contact set read on logged iterations only
                       (--log-every, default 10; the line says how many of
                       the timed iterations that was)
    branch always-on   the branch with log_every=1 (read every step)

run ALTERNATING -- A B C A B C ... -- so thermal drift and whatever else the
machine is doing land on all three equally, same seed, each iteration timed
on its own (TrainLog.t_iter; patched into the baseline's source the same
way), median s/iter over every timed iteration EXCEPT each run's final one,
which train() always measures. Each line says how many of the timed
iterations ran the contact summary.

    PYTHONPATH=src python experiments/infra/bench_rl_step.py --iters 10 --reps 3

Do not read the numbers while another run holds the cores.

Measured on a quiet machine, the first version (force gated, contact list
still walked in Python every step) came out x1.053 (gamecontroller) and
x1.079 (bowl) of 3216767; always-on x1.060 / x1.086. So the loop itself was
the cost, not mj_contactForce, and the summary is now numpy over
`d.contact.geom` / `d.contact.dist` and only run on measured steps.
`--summary-only` times that function alone on the buried seed state -- the
per-contact Python loop it replaced against the array version, with and
without the force read -- so the change is measurable without a PPO run:

    PYTHONPATH=src python experiments/infra/bench_rl_step.py --summary-only
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
import mujoco

from handsim import paths
from handsim.human import grab, track, rl

if str(paths.REPO) not in sys.path:     # track.py imports experiments.* itself
    sys.path.insert(0, str(paths.REPO))


#: The per-iteration timer the branch's train() has (TrainLog.t_iter), applied
#: to the baseline's source textually so its final iteration can be excluded
#: like-for-like. Each marker must match exactly once in rl.py at 3216767; if
#: the revision differs, the bench falls back to total/iters and says so.
_TIMER_PATCH = [
    ("from __future__ import annotations\n",
     "from __future__ import annotations\nimport time\n"),
    ("    frac_alive: list = field(default_factory=list)\n",
     "    frac_alive: list = field(default_factory=list)\n"
     "    t_iter: list = field(default_factory=list)\n"),
    ("        for it in range(cfg.iters):\n",
     "        for it in range(cfg.iters):\n            _t_it = time.perf_counter()\n"),
    ("            log.reward.append(float(R.mean()))\n",
     "            log.t_iter.append(time.perf_counter() - _t_it)\n"
     "            log.reward.append(float(R.mean()))\n"),
]


def load_rl_at(rev: str):
    """`rl.py` exactly as committed at `rev`, as its own module, with the
    per-iteration timer patched in if its source still matches."""
    src = subprocess.run(
        ["git", "-C", str(paths.REPO), "show", f"{rev}:src/handsim/human/rl.py"],
        capture_output=True, text=True, check=True).stdout
    if "t_iter" not in src and all(src.count(old) == 1 for old, _new in _TIMER_PATCH):
        for old, new in _TIMER_PATCH:
            src = src.replace(old, new)
    elif "t_iter" not in src:
        print(f"note: {rev} could not be timed per iteration; using total/iters",
              flush=True)
    spec = importlib.util.spec_from_loader(f"rl_{rev}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclasses resolve string annotations here
    exec(compile(src, f"rl.py@{rev}", "exec"), mod.__dict__)
    return mod


def build(ref: str, hand: str, grips_path: Path):
    """The tracker as g9_ppo_distill.per_reference builds and seeds it."""
    # Keyed on subject/seq: GRAB has 80 sequence names that exist under more
    # than one subject, and a bare-name dict silently keeps one of them (it
    # voided a training run on 2026-09-16; tests/test_seed_keys.py guards it).
    # A bare --ref is accepted only when exactly one subject has it.
    rows = json.loads(grips_path.read_text())["rows"]
    grips = {f"{x['subject']}/{x['seq']}": x for x in rows}
    if ref in grips:
        key = ref
    else:
        hits = [k for k in grips if k.split("/", 1)[1] == ref]
        if len(hits) != 1:
            raise SystemExit(f"--ref {ref!r} is {'ambiguous' if hits else 'absent'} in "
                             f"{grips_path}: {hits or sorted(grips)[:8]} -- give subject/seq")
        key = hits[0]
    row = grips[key]
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = track.ReferenceTracker(seq, hand=hand)
    rt.apply_wrist_offset(np.asarray(row["offset"], float))
    gf = rt.grasp_frames()
    if len(gf) == 0:
        rt.synthesize_grasp()
        gf = rt.grasp_frames()
    if len(gf) == 0:
        raise SystemExit(f"{ref}: no graspable frame")
    return rt, gf, row


def summary_loop(sc, force=True):
    """The per-contact Python loop `rl.contact_summary` replaced (b14c384),
    kept here verbatim as the micro-benchmark's baseline and cross-check."""
    m, d = sc.model, sc.data
    objs, hands = set(sc.obj_gids), set(sc.hand_gids)
    f = np.zeros(6)
    pen, n, grip, bodies = 0.0, 0, 0.0 if force else float("nan"), set()
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ({g1, g2} & objs and {g1, g2} & hands):
            continue
        n += 1
        pen = max(pen, -float(c.dist))
        bodies.add(int(m.geom_bodyid[g1 if g1 in hands else g2]))
        if force:
            mujoco.mj_contactForce(m, d, i, f)
            grip += abs(float(f[0]))
    return rl.ContactSummary(pen=pen, n=n, bodies=len(bodies), grip=grip)


def bench_summary(rt, cs, reps: int, envs: int, horizon: int):
    """Time the summary alone on the current (buried) state of rt.sim."""
    masks = rl.contact_masks(rt.sim)
    cases = [
        ("loop, no force", lambda: summary_loop(rt.sim, force=False)),
        ("loop, force", lambda: summary_loop(rt.sim, force=True)),
        ("numpy, no force", lambda: rl.contact_summary(rt.sim, force=False, masks=masks)),
        ("numpy, force", lambda: rl.contact_summary(rt.sim, force=True, masks=masks)),
    ]
    a, b = summary_loop(rt.sim), rl.contact_summary(rt.sim, masks=masks)
    assert (a.pen, a.n, a.bodies) == (b.pen, b.n, b.bodies) and abs(a.grip - b.grip) < 1e-6, (a, b)
    print(f"summary on {cs.n} hand-object contacts of {rt.sim.data.ncon} "
          f"({cs.bodies} bodies), {reps} calls each, alternating:", flush=True)
    us = {name: [] for name, _f in cases}
    for _rep in range(5):
        for name, fn in cases:
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            us[name].append((time.perf_counter() - t0) / reps * 1e6)
    ref = statistics.median(us[cases[0][0]])
    for name, _f in cases:
        med = statistics.median(us[name])
        per_iter = med * 1e-6 * envs * horizon
        print(f"  {name:16s} {med:8.1f} us/call   x{med / ref:.3f}   "
              f"= {per_iter:6.3f} s/iter at {envs} envs x {horizon}", flush=True)


def main(a):
    t0 = time.time()
    rt, gf, row = build(a.ref, a.hand, Path(a.grips))
    rt.reset_at(int(gf[0]))
    cs = rl.contact_summary(rt.sim)
    print(f"{a.ref}: stage-2 seed {row['n_contact']} contacts; at frame {int(gf[0])} "
          f"now {cs.n} contacts on {cs.bodies} bodies, {cs.pen * 1000:.2f} mm, "
          f"{cs.grip:.0f} N  ({time.time() - t0:.0f}s to build)", flush=True)
    span = int(rt.T - gf[0])
    if a.summary_only:
        bench_summary(rt, cs, a.calls, a.envs,
                      a.horizon or min(224, max(64, span + 8)))
        return
    base = load_rl_at(a.rev)
    horizon = a.horizon or min(224, max(64, span + 8))

    def cfg_for(mod, **kw):
        c = mod.RLConfig(n_envs=a.envs, horizon=horizon, iters=a.iters, seed=a.seed)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    variants = [
        (a.rev, base, lambda: cfg_for(base)),
        ("branch defaults", rl, lambda: cfg_for(rl, log_every=a.log_every)),
        ("branch always-on", rl, lambda: cfg_for(rl, log_every=1)),
    ]
    times = {name: [] for name, _m, _c in variants}
    # The FINAL iteration of every run is excluded from the timing: train()
    # measures the contact set on `it == iters-1` whatever log_every is, so
    # with it in, no --iters value gets the "defaults" column down to a real
    # run's cadence (3 iters: 2 of 3 measured). Without it, --iters 10
    # --log-every 10 times 9 iterations of which exactly one (it 0) is
    # measured -- 11%, within a point of a 60-iteration run's 7/60.
    meas = {name: measured_iters(mk()) for name, _m, mk in variants}

    def frac(name):
        k, n = meas[name]
        return f"{n * a.reps} timed, {k * a.reps} measured"
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
            _net, log = mod.train(rt, cfg, starts=gf, verbose=False)
            total = time.perf_counter() - t0
            ti = list(getattr(log, "t_iter", []))
            if len(ti) == a.iters and a.iters > 1:
                per = ti[:-1]                       # all but the final iteration
            else:
                per = [total / a.iters] * max(a.iters - 1, 1)
            times[name].extend(per)
            print(f"  rep {rep}  {name:18s} {statistics.median(per):7.3f} s/iter "
                  f"(median of {len(per)}; final iteration excluded)", flush=True)

    ref_med = statistics.median(times[a.rev])
    print("median s/iter over all timed iterations:")
    for name, _m, _c in variants:
        med = statistics.median(times[name])
        print(f"  {name:18s} {med:7.3f}   x{med / ref_med:.3f} of {a.rev}   "
              f"{frac(name)}", flush=True)


def measured_iters(cfg) -> tuple[int, int]:
    """(measured, timed) iterations per run for `cfg`, the final iteration
    excluded, by rl.train's own rule: every log_every-th (and the last, which
    is why it is excluded), or all of them when a term prices the contact
    set. 3216767's config has no log_every and its step reads nothing."""
    timed = max(cfg.iters - 1, 1)
    le = getattr(cfg, "log_every", None)
    if le is None:
        return 0, timed
    if any(getattr(cfg, k, 0.0) for k in ("w_pen", "w_force", "w_hold")):
        return timed, timed
    return sum(1 for it in range(timed) if it % le == 0), timed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default="bowl_drink_1",
                   help="reference as subject/seq (a bare name only if unambiguous); needs a row in --grips")
    p.add_argument("--rev", default="3216767", help="baseline commit of rl.py")
    p.add_argument("--iters", type=int, default=10,
                   help="PPO iterations per run; the last is not timed")
    p.add_argument("--reps", type=int, default=3, help="alternating repetitions")
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--horizon", type=int, default=0, help="0 = g9's rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10,
                   help="RLConfig.log_every for the 'branch defaults' variant; "
                        "with --iters below it, most timed iterations are measured")
    p.add_argument("--hand", default="shadow")
    p.add_argument("--grips", default=str(paths.RESULTS / "stage2_grips_g9.json"))
    p.add_argument("--summary-only", action="store_true",
                   help="time contact_summary alone on the seed state; no PPO")
    p.add_argument("--calls", type=int, default=2000,
                   help="--summary-only: calls per timing")
    main(p.parse_args())
