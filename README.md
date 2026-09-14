# opposition-deficit

[![tests](https://img.shields.io/badge/tests-46%20fast%20%2B%20slow-2a78d6)](tests/)
[![license](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)

A simulation workbench for **how a grasp should be specified when transferring
manipulation between embodiments**, and a measurement log that records, in
public, every claim this project has had to withdraw.

> **The name is a historical artifact.** It refers to an "opposition deficit"
> that does not exist — the measurement behind it was taken at the wrong point
> on every hand. See [Status](#status). Renaming is pending G4.

---

## The question, and where it has moved

It started as: *human hand-object data is retargeted by matching the hand's
pose; should it instead specify the wrench the object requires?*

Two pre-registered experiments answered the human-data half: **no**. A wrench
objective does beat the pipeline practitioners ship, but a demonstration adds
nothing to it — once as an initialisation (G1), once as a task specification
(G2). So the live question is now narrower and more basic:

**Do the metrics this field optimises actually predict whether a grasp does the
job?** That is G3, and the answer is *partly, and not the one you would expect*.

## Status

| claim | status | evidence |
|---|---|---|
| A wrench objective beats the shipped retargeting pipeline | **supported** | pre-registered, n=60, McNemar p = 0.0075 |
| …by *selecting* better, not searching more | **supported** | finds fewer valid grasps, survives 2.3× as often |
| Human demonstration data helps | **not supported, twice** | G1 initialisation p = 1.0000; G2 specification p = 0.180, worse on magnitude |
| Contact **force** predicts task success | **supported** | ρ +0.325, AUC 0.800, clustered by grasp |
| ε predicts task success | **weakly** | AUC 0.684, below the pre-declared ρ = 0.3 |
| Every metric inverts on capsules | **robust across three bug fixes** | see the figure below |
| The peg task requires two hands | **supported** | 13.49 cm vs 5.32 cm, by force balance |
| An opposition deficit exists among these hands | **RETRACTED** | all four oppose within 2.8 mm once measured correctly |
| A second hand repairs that deficit | **WITHDRAWN** | force-only data labelled as wrench; budget uncontrolled |
| BC / action chunking show learned control | **RETRACTED** | open-loop replay scores 8/8 on the same task |
| "No metric predicts task success" (earlier G3) | **WITHDRAWN** | it was three bugs — see below |

**[NOTES.md](NOTES.md) is the authoritative log. Read it before quoting any
number.** Withdrawn work lives under `results/retracted/`,
`figures/retracted/`, `experiments/retracted/`, each with a README saying what
was wrong.

## Headline: which metrics predict task success?

238 sampled grasps × 4 hands × 4 object shapes × 2 carry tasks. Grasps are
**sampled, not optimised** — an optimiser would remove the quality variation
being tested. Inference is clustered **by grasp**, since the two tasks share one.

![metric prediction](figures/g3_metrics.png)

Exactly one metric crosses the pre-declared ρ = 0.3: **total contact force**
(AUC 0.800). The strongest indicator of whether a grasp completes a carry is
**how hard the hand squeezes**, not where its contacts sit. Ferrari–Canny ε —
the field's geometric quality measure — is genuinely informative at AUC 0.684,
and clearly weaker. That is a deflating result and it is reported as one.

The robust observation is the right panel: **every metric inverts on capsules.**
Whatever these metrics capture does not transfer across object geometry, and
that survived three separate bug fixes.

## Tasks

Full protocols and validity gates: **[docs/TASKS.md](docs/TASKS.md)**.

### T1 — grasp and hold under a wrench probe

Pre-grasp → close → squeeze → release, then push and twist along 14 force and
14 torque directions until it slips. No floor and no gravity, so a dropped
object is unambiguous and a resting one cannot be mistaken for a held one.

| wrench objective | pose objective |
|---|---|
| ![wrench](figures/task_grasp_wrench.gif) | ![pose](figures/task_grasp_pose.gif) |
| ε 0.642 · 9 contacts · holds **1.345 N** | ε 0.358 · 4 contacts · holds **0.000 N** |

Same hand, same 6 cm cube, same seed, same budget — only the objective differs.

### T3 — carry under gravity

Lift 8 cm, tilt 60° **about the object's own centre**, carry 8 cm, set down.
Scored by *slip in the palm frame*, because the base is position-controlled and
tracking error would measure controller lag rather than the grasp. Two variants
with genuinely different wrench profiles: task A loads torque about x (peak
0.0113), task B about y (0.0307).

### T2 — bimanual peg extraction

Socket friction (1.20 N) exceeds the base's weight (0.78 N), so a one-handed
pull lifts the base instead of extracting the peg. Two-handed by force balance,
not by assumption.

| two-handed expert | one-handed control |
|---|---|
| ![two handed](figures/task_peg_two_handed.gif) | ![one handed](figures/task_peg_one_handed.gif) |
| peg out **13.49 cm**, base lift +0.44 cm — success | peg out 5.32 cm, base lift **+10.26 cm** — failure |

### M1 — the opposition axis

`make axis`. Fingertips derived from each distal link's collision geometry,
mimic couplings enforced, self-collision scoped to the fingers, distance to the
*nearest* fingertip.

| hand | floor | aperture | DoF |
|---|---:|---:|---:|
| Allegro | 0.06 cm | 30.44 cm | 16 |
| Shadow | 0.21 cm | 25.18 cm | 24 |
| Dexmate f5d6 | 0.26 cm | **12.73 cm** | 11 |
| LEAP | 0.34 cm | 31.95 cm | 16 |

All four oppose within 2.8 mm — there is no deficit. The spread that *does*
survive is **aperture**, where f5d6 is an outlier by 2.4×.

## Where this is headed

Each goal is pre-registered before it is written, with its decision rule fixed
in advance, including the branch where the result kills the idea.

| | question | outcome |
|---|---|---|
| **G1** | does a wrench objective beat the pipeline people ship? | **yes** (p = 0.0075); the demonstration adds nothing (p = 1.0000) |
| **G2** | does the demonstration specify the *task*? | **no** — and conditioning on it costs grasp strength |
| **G3.1** | can all four hands even be placed? | **fixed** — Shadow went from 0/320 valid grasps to 16–38% |
| **G3.2** | do grasp metrics predict task success? | contact force does (AUC 0.800); ε weakly (0.684) |
| **G4** | *is the task outcome even reproducible?* | **running** — a go/no-go on all of the above |

**G4 is the gate.** If the same grasp gives different outcomes under ±1 mm and
±2% perturbations, none of G3 means anything and the right move is a better
contact model, not another sweep. Its threshold (80% unanimity) was fixed
before the code was written.

If G4 passes, the next question is the capsule inversion: it is the one
observation that survived every bug fix, and it points at contact *geometry*
rather than scalar wrench summaries.

## Layout

```
src/oppdef/
  hands/      tips.py (fingertip derivation) · model.py (shared kinematics)
              axis.py (the opposition metric) · specs.py (closure synthesis)
              f5d6.py (URDF repairs: tip frames, mimic couplings, torque limits)
  synth.py    grasp synthesis by closing in simulation
  task.py     object trajectories, required wrenches, the carry executor
  hold.py     the 6-D wrench probe
  bench.py    the shared benchmark cell
  metrics/    Ferrari-Canny epsilon from real MuJoCo contacts
  vec.py      batched stepping (CPU / MJX / Warp) with measured parity
experiments/  live experiments; retracted/ holds the withdrawn ones
docs/         task definitions, pre-registrations, external reviews
results/      live results with provenance; retracted/ holds the rest
NOTES.md      the measurement log and every retraction
```

## Reproduce

```bash
make install
make test                 # fast + slow invariants
make axis                 # the opposition axis, with provenance
make g1 && make g1-analysis    # pre-registered three-arm comparison
python experiments/g3.py       # metric-vs-task dataset
python experiments/g3_analysis.py
make expert               # the bimanual peg task and its one-handed control
make render-tasks         # re-render every GIF by replaying saved grasps
```

Renders need `MUJOCO_GL=egl`. GPU stepping uses `mujoco_warp`; **MJX-JAX does
not run on this machine** — `gpusolverDnCreate` fails, so no MJX number should
be quoted from it.

## How this repository is meant to be read

Experiments that can decide something are **pre-registered** — hypotheses,
analysis and decision rule committed before the code exists. Results carry
provenance and, for every selected grasp, the placement, finger targets,
per-direction probe values and rejection counts, so a figure traces back to the
pose behind it without re-running a search. Withdrawn work is kept, labelled,
and its generator made to refuse to run.

The measurement log is longer than the results. Three of this project's
headline findings were killed by its own later checks, and one of those checks
(G4) found a bug on its first smoke run that invalidated the result it was
built to test. That ratio is the honest one so far.
