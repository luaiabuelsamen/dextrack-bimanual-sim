# opposition-deficit

**Question.** When a robot hand cannot oppose the way a human hand does, what
should human hand-object interaction data be retargeted *to*?

Read `NOTES.md` top to bottom for the record, including every retraction.

## What is established

**The opposition axis, two dimensions, four hands.** The closest a hand's thumb
can come to its fingers (REACH) predicts how many block widths it can actually
hold (HOLD). Measured by one identical, derived procedure per hand -- nothing
hand-specific is hard-coded.

| hand | opposition floor | widths feasible | **widths held** |
|---|---|---|---|
| Shadow (24 DoF) | 0.00 cm | 8 | **7** |
| LEAP (16 DoF) | 0.00 cm | 8 | **5** |
| Allegro (16 DoF) | 0.00 cm | 7 | **3** |
| Dexmate f5d6 (11) | **3.08 cm** | 4 | **0** |

f5d6 is the only hand that cannot oppose, and the only one that holds nothing --
not even the four widths it can nominally close on. Its 3.08 cm replicates a
3.1 cm measured months earlier by a completely different method.

**A bimanual task that provably needs two hands.** Two LEAP hands, a peg in a
socket whose 1.20 N of friction exceeds the 0.78 N base weight, so a one-handed
pull lifts the whole base instead of extracting anything.

| | peg out | base lift | success |
|---|---|---|---|
| two-handed expert | 12.98 cm | +0.57 cm | yes |
| one-handed control | 9.07 cm | **+6.10 cm** | no |

6/6 across socket-friction and base-mass settings. The control is run, not
assumed.

**GPU physics, locally, without degrading the scene.** JAX CUDA works on this
Jetson (the note that it does not is out of date). The MJX/warp port strips only
VISUAL geoms -- proven bit-identical over 400 steps -- and leaves collision
geometry and the solver untouched. 2,655 env-steps/s at 256 worlds.

**The first learned policy.** BC from 20 randomised expert demos: 67% +/- 47%
against 0% for do-nothing and random, matching the expert on 2 of 3 seeds.

## Layout

    GOAL.md      the four claims the project exists to earn or kill
    PLAN.md      milestones with a kill criterion each
    NEXT.md      what is built, what is not, the one next step
    TODO.md      outstanding work and what "done" means
    NOTES.md     the measurement log -- every number traces here
    EXPERIMENTS.md  simulator protocol, tasks, success criteria
    scripts/     live code; scripts/attic/ is superseded, with reasons
    results/     live results; results/retracted/ may not be quoted
