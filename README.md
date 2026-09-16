# dextrack-bimanual-sim

[![simulation](https://img.shields.io/badge/simulation-MuJoCo-2a78d6)](src/oppdef/human/track.py)
[![license](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)

A simulation workbench for **two-handed dexterous tracking from human motion**,
and a public measurement log. Real [GRAB](https://grab.is.tue.mpg.de/) hand-object
references drive retargeting, grasp synthesis, per-reference PPO and two-hand
experiments, following [DexTrack](https://meowuu7.github.io/DexTrack/).

The log is the point. Every claim this project has had to withdraw is recorded
here with the measurement that killed it, including the one it used to be named
after. **[NOTES.md](NOTES.md) is authoritative — read it before quoting any
number.**

> **Renamed from `opposition-deficit`** (2026-09-15). That name referred to a
> deficit that does not exist: the measurement behind it was taken at the wrong
> point on every hand, and the claim is retracted. The Python package is still
> `oppdef`; renaming it is a mechanical change deliberately deferred so it does
> not collide with active work.

## What this repository establishes

Two research programs share this codebase. Both are measured; one is finished
and one is live.

### A · Grasp quality and retargeting objectives — settled

Does the objective everyone retargets against actually predict whether a grasp
does the job? 238 sampled grasps × 4 hands × 4 shapes × 2 carry tasks,
pre-registered, clustered by grasp.

| finding | |
|---|---|
| **A wrench objective beats the shipped keypoint pipeline** | pre-registered, n=60, McNemar p = 0.0075 — and by *selecting* better, not searching more |
| **Contact force predicts task success** | ρ +0.325, **AUC 0.800**; the strongest single predictor tested |
| **Every metric inverts on capsules** | survived three bug fixes and an external audit — whatever these metrics capture does not transfer across object geometry |
| ε is informative but weak | AUC 0.684, below the pre-declared threshold |
| Outcomes are reproducible under perturbation | 0.928 unanimity, CI [0.897, 0.956] |
| The peg task genuinely requires two hands | 13.49 cm vs 5.32 cm, by force balance |

Four claims from this program were **retracted or withdrawn**, including the one
the repository was named after. That history is in [Status](#status) and
[NOTES.md](NOTES.md), not buried.

### B · Human motion → robot tracking — live, blocked at stage 2

The DexTrack pipeline rebuilt on real [GRAB](https://grab.is.tue.mpg.de/)
references: retarget contacts, form a grasp, train a tracker per clip, distil,
extend to two hands, then test under depth perception.

| built and validated | |
|---|---|
| GRAB reconstruction | recall **0.919** against the dataset's own contact labels (0.158 under a transposed-rotation bug, since fixed) |
| Convex decomposition | a mug handle is a genuine hole, not a filled hull — 0.92× volume |
| Instrumentation | every rollout GIF carries penetration, grip force and contact count on its face, with a per-frame JSON manifest and a replay check |
| Stages 1, 4, 7 | human reference, homotopy curriculum, depth→pose evaluation of a frozen tracker |

**Stage 2 is broken and everything downstream inherited it.** The retarget
matched fingertip positions with no non-penetration constraint, so every
reported tracking number was measured on a hand *inside* the object. Those
numbers are withdrawn. The defect is now located, decomposed and half fixed:

| | state | |
|---|---|---|
| **A** arm-side placement | **fixed** | `binoculars_see_1` 175 m → **106 mm** |
| **B** finger reach | open | hand now feasible but 50–62 mm out of reach |
| **C** vessel wrist target | open | hand placed *through* cups and mugs |

**→ [docs/STAGE2.md](docs/STAGE2.md)** is the full diagnosis and where to pick
up. RL retraining is blocked behind all three.

### Where to start

| you want to | read |
|---|---|
| continue the live work | [docs/STAGE2.md](docs/STAGE2.md) |
| know what a number means before quoting it | [NOTES.md](NOTES.md) — authoritative |
| find your way around the code | [src/oppdef/README.md](src/oppdef/README.md) |
| see what an external review found | [docs/REVIEW_HANDOFF.md](docs/REVIEW_HANDOFF.md) |
| reproduce a figure | [Reproduce](#reproduce) |

---

## Status

| claim | status | evidence |
|---|---|---|
| A wrench objective beats the shipped retargeting pipeline | **supported** | pre-registered, n=60, McNemar p = 0.0075 |
| …by *selecting* better, not searching more | **supported** | finds fewer valid grasps, survives 2.3× as often |
| References improve the earlier synthetic objectives | **benefit not established** | G1 initialisation p = 1.0000; G2 specification p = 0.180; this does not test all uses of human data |
| Contact **force** predicts task success | **supported** | ρ +0.325, AUC 0.800, clustered by grasp; survives G4 |
| ε predicts task success | **weakly** | AUC 0.684 (0.679 excluding non-reproducible grasps), below the pre-declared ρ = 0.3 |
| Every metric inverts on capsules | **robust across three bug fixes and an audit** | see the figure below |
| GRAB tracking rollouts are physically valid | **NOT SUPPORTED** | 8–14 mm penetration on 541/541 frames at 1530–7549 N mean on a 1.96 N object; grasp search was climbing toward it |
| The retarget can produce a usable grasp | **NOT SUPPORTED** | its grasps are either interpenetrating or non-contacting; on `phone_call_1` the only 0 mm-penetration grasp found never touches the object |
| Scoring the grasp search on tracking improved it (`d5b49ca`, 248 m → 35 mm) | **WITHDRAWN** | compared an unphysical grasp against a failed one; the 35 mm is achieved at 19.4 mm penetration and 4551 N |
| Perturbed repeats agree | **high agreement, with replay failures** | G4: 0.928 unanimity, CI [0.897, 0.956]; five grasps did not re-form |
| The peg task requires two hands | **supported** | 13.49 cm vs 5.32 cm, by force balance |
| An opposition deficit exists among these hands | **RETRACTED** | all four oppose within 2.8 mm once measured correctly |
| A second hand repairs that deficit | **WITHDRAWN** | force-only data labelled as wrench; budget uncontrolled |
| Earlier peg-task BC / action chunking demonstrate feedback | **RETRACTED** | open-loop replay scores 8/8 on that task; separate from the new GRAB PPO result |
| "No metric predicts task success" (earlier G3) | **WITHDRAWN** | it was three bugs — see below |

**[NOTES.md](NOTES.md) is the authoritative log. Read it before quoting any
number.** Withdrawn work lives under `results/retracted/`,
`figures/retracted/`, `experiments/retracted/`, each with a README saying what
was wrong.

## Physics rollouts

The first time this pipeline was stepped as real physics rather than kinematic
playback. **All four fail**, and the GIFs are built to show *why* rather than
assert it: the object is translucent, so the hand inside it is visible, and any
link buried deeper than 2 mm turns **red**.

| Mug — per-clip PPO | Bowl — bimanual grasp search |
|---|---|
| ![Shadow hand with finger links buried inside a translucent GRAB mug](figures/physics_mug.gif) | ![Two Shadow hands with fingers inside a translucent bowl](figures/physics_bowl.gif) |

| Binoculars — bimanual grasp search | Camera — bimanual grasp search |
|---|---|
| ![Left hand buried in the binoculars while the right hand grips nothing](figures/physics_binoculars.gif) | ![A camera inverted between two Shadow hands](figures/physics_camera.gif) |

| Clip | Position error | Orientation error | Max penetration | Mean grip | vs object weight |
|---|---:|---:|---:|---:|---:|
| [Mug](figures/physics_mug.json) | 29.4 mm | 44.8° | 13.8 mm | 1562 N | 796× |
| [Bowl](figures/physics_bowl.json) | 18.0 mm | 45.9° | 9.8 mm | 2325 N | 1185× |
| [Binoculars](figures/physics_binoculars.json) | 32.6 mm | 20.0° | 10.9 mm | 7553 N | 3850× |
| [Camera](figures/physics_camera.json) | 34.4 mm | 138.4° | 11.4 mm | 5353 N | 2729× |

The object weighs **1.96 N**. The position-error column is what this project
used to report; watch the red links and the two columns after it for why that
column meant nothing. `make render-tracking` reproduces the set, and every
figure is saved per frame in the manifests beside the GIFs.

**→ [docs/STAGE2.md](docs/STAGE2.md)** — the full diagnosis and where to pick up.

## The question, and where it has moved

It started as: *human hand-object data is retargeted by matching the hand's
pose; should it instead specify the wrench the object requires?*

Two earlier synthetic experiments found no demonstrated benefit from their
specific use of a reference: as an initialisation (G1), and as a task
specification (G2). These comparisons do not rule out useful information in
real human demonstrations. They motivated a narrower diagnostic question:

**Do the metrics this field optimises actually predict whether a grasp does the
job?** That is G3, and the answer is *partly, and not the one you would expect*.

The current work uses real [GRAB](https://grab.is.tue.mpg.de/) human motion
with a pipeline inspired by [DexTrack](https://meowuu7.github.io/DexTrack/):
retarget contacts, establish a feasible grasp, train a tracker per reference,
then distil a shared policy. Two-hand experiments currently use joint
retargeting and grasp search. See [docs/PIPELINE.md](docs/PIPELINE.md).

## Earlier benchmark: which metrics predict task success?

238 sampled grasps across 4 hands and 4 object shapes, evaluated on 2 carry
tasks. Grasps are **sampled, not optimised** — an optimiser would remove the
quality variation being tested. Inference is clustered **by grasp**, since the two tasks share one.

![metric prediction](figures/g3_metrics.png)

Exactly one tested metric crosses the pre-declared ρ = 0.3: **total contact
force** (AUC 0.800). Ferrari–Canny ε is informative at AUC 0.684. These are
predictive associations in this sampled benchmark; they do not establish that
extra squeeze causes success or that contact placement is unimportant.

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

## The pipeline

Seven stages from a GRAB clip to a tracked object: human reference → retarget →
per-reference PPO → homotopy curriculum → distillation → two hands → depth
perception. **Stage 2 is broken and everything downstream inherited it.**

**→ [docs/PIPELINE.md](docs/PIPELINE.md)** — every stage, what it measures, what
it established, and what was withdrawn, including the GRAB reconstruction
validation, the convex-decomposition fix, the PPO horizon result and the
perception finding.

**→ [docs/STAGE2.md](docs/STAGE2.md)** — the live problem.

## Layout

```
src/oppdef/        package README: src/oppdef/README.md
  human/           THE LIVE PIPELINE. GRAB -> retarget -> grasp -> PPO -> distil
                   grab.py mano.py decompose.py  data in
                   retarget.py                   human contacts -> joint traj
                   scene.py track.py             one hand or two, in physics
                   rl.py distill.py homotopy.py  PPO, distillation, curriculum
                   grasp.py perception.py        grasp fitting; depth -> pose
  grasping/        grasp quality and synthesis, shared by both bodies of work
                   epsilon.py                    Ferrari-Canny from real contacts
                   synth.py hold.py task.py      synthesis, wrench probe, carry
                   retarget_pose.py              single-pose retarget (keypoint vs eps)
  hands/           tips.py (fingertip derivation, load-bearing) · model.py
                   axis.py (the opposition metric) · specs.py · f5d6.py
  envs/            simulation environments; bimanual.py provides _add_base_dof
  learning/        bc.py is imported by human/distill.py
  track_core.py    reference-agnostic tracking core (not the GRAB tracker)
  embodiment.py paths.py objects.py   hand/object abstractions, asset paths
  bench.py policy.py sensing.py data.py vec.py control/ sim/ viz/
                   earlier synthetic-grasp work; backs the supported G3 findings
experiments/       grouped by program -- see experiments/README.md
  grasp_metrics/   program A: g1-g5, matched, the peg task, figures
  tracking/        program B: grab_*, render_tracking, g7-g9
  infra/           backend parity and batched-stepping benchmarks
  retracted/       withdrawn experiments, each saying what was wrong
docs/              task definitions, pre-registrations, external reviews
results/ figures/  live results with provenance; retracted/ holds the rest
attic/             superseded scripts, two kept as retraction evidence
NOTES.md           the measurement log and every retraction
```

`human/` imports from `grasping/`, `hands/`, `envs/`, `learning/`,
`embodiment.py` and `paths.py`, so none of those are removable legacy.
[`src/oppdef/README.md`](src/oppdef/README.md) says what each module is for and
which are load-bearing. The package is still named `oppdef` after the retracted
claim; renaming it is mechanical and deferred.

## Reproduce

```bash
make install
make test                 # fast + slow invariants
make axis                 # the opposition axis, with provenance
make g1 && make g1-analysis    # pre-registered three-arm comparison
python experiments/grasp_metrics/g3.py       # metric-vs-task dataset
python experiments/grasp_metrics/g3_analysis.py
make expert               # the bimanual peg task and its one-handed control
make render-tasks         # synthetic grasp / peg GIFs
make render-tracking     # the physical GRAB rollouts at the top of this README
```

Tracking renders require the local GRAB/MANO assets, Menagerie models, and
`pip install -e ".[viz,learn]"`. Install `coacd` if the convex-decomposition
cache has not been generated yet. Set `OPPDEF_DATA` and `OPPDEF_MENAGERIE`
to those local assets when they are outside the default workspace layout.
The mug demo uses the checked-in PPO checkpoint and its original hold-scored
grasp setup. Each new GIF has an adjacent JSON
manifest with the exact source hashes, search settings, errors, and replay check.
The full physics-state cache stays in `out/tracking_gifs/`; use
`python -m experiments.render_tracking mug bowl binoculars camera --render-only`
to adjust presentation without rerunning the experiment.

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
