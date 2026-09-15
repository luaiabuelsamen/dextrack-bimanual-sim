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

Since G4 the project has moved to the thing those experiments were always
circling: **tracking control for dexterous manipulation, learned from real
human hand-object motion** — the DexTrack pipeline, rebuilt here in simulation
and validated locally, then extended to two hands. The first two stages are
built and measured; see [From human motion to a robot hand](#from-human-motion-to-a-robot-hand).

## Status

| claim | status | evidence |
|---|---|---|
| A wrench objective beats the shipped retargeting pipeline | **supported** | pre-registered, n=60, McNemar p = 0.0075 |
| …by *selecting* better, not searching more | **supported** | finds fewer valid grasps, survives 2.3× as often |
| Human demonstration data helps | **not supported, twice** | G1 initialisation p = 1.0000; G2 specification p = 0.180, worse on magnitude |
| Contact **force** predicts task success | **supported** | ρ +0.325, AUC 0.800, clustered by grasp; survives G4 |
| ε predicts task success | **weakly** | AUC 0.684 (0.679 excluding non-reproducible grasps), below the pre-declared ρ = 0.3 |
| Every metric inverts on capsules | **robust across three bug fixes and an audit** | see the figure below |
| Task outcomes are reproducible under perturbation | **supported** | G4: 0.928 unanimity, CI [0.897, 0.956] |
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
| **G4** | *is the task outcome even reproducible?* | **yes** — 0.928 unanimity, CI [0.897, 0.956]; the gate says **go** |

**G4 was the gate, and it passed.** Under ±1 mm and ±2% perturbations the same
grasp gives the same outcome 92.8% of the time (clustered CI [0.897, 0.956]),
and the majority outcome reproduces G3 at 99.6%. Its 80% threshold was fixed
before the code was written. One caveat is recorded rather than buried: 5 of
228 G3 grasps (2.2%) do not re-form standalone — they existed only given
accumulated scene state — and excluding them moves ε from AUC 0.684 to 0.679
and contact force from 0.800 to 0.795. Nothing changes.

| **G5** | does a retargeted human grasp hold the real object? | pre-registered; running |

**The pipeline being built**, stage by stage. Each stage is gated on a
measurement, not on the previous stage having compiled:

| stage | what it does | state |
|---|---|---|
| 1. human reference | GRAB clip → object pose over time, both MANO hands | **done** — hand closes to 0.1–0.6 mm of the object and holds |
| 2. retarget | human contact points → robot joint trajectory | **done** — ~13 mm to the human's contacts, 0 mm penetration after settling |
| 3. per-reference tracking | **PPO** per clip (plus MPPI + homotopy) | **solved** — 29.4 mm mean, **111/111** frames within 50 mm, object never dropped |
| 4. homotopy curriculum | solve an easier reference, deform it into the hard one | **built** — walks λ 0.35 → 1.0 holding throughout |
| 5. distillation | one neural tracking controller across references | **built, fails in the loop** — regression ratio 0.716 held out, but the policy drops the object on every held-out reference |
| 6. bimanual | both hands on one object | **tracking** — 3 of 4 references inside 200 mm (56/117/165 mm), all 4 hold |
| 7. perception | depth → pose estimator → evaluate the *frozen* tracker | **built** — and it already says something (below) |

Every number in this section is from the **corrected** pipeline. An earlier set
was computed with the object's rotation transposed and is
[retracted](NOTES.md).

## From human motion to a robot hand

![GRAB mug_drink_1](figures/grab_mug_drink_1.gif)

*GRAB `s1/mug_drink_1`: the human right hand (red) reaching, taking the mug **by
its handle**, drinking, and setting it down. The left hand stays 53 cm away,
which is what "drink" should look like. Drawn as the MANO surface — a stick
skeleton cannot show whether a hand is curled.*

> Two earlier versions of this caption were wrong, and the second one was wrong
> because the pipeline was. It first claimed the handle from a thumbnail, then
> claimed the **body** on the strength of a measurement taken while the object's
> rotation was **transposed** (GRAB poses objects as `v @ R`, not `v @ R.T`).
> GRAB's own per-vertex contact labels put 41–100% of this clip's contacts on
> the handle. See [the retraction](NOTES.md).

The reference is validated against **GRAB's own per-vertex contact labels**, not
against itself. Every earlier check here compared the reconstructed hand to the
reconstructed object and asked whether they were close; they always were, and
for a while they were close in the wrong place — the object's rotation was
transposed and a 0.1 mm minimum-distance check passed the whole time.

`experiments/grab_validate.py` scores the reconstruction against the labels the
dataset ships: **recall 0.919, minimum 0.846** over the sequences checked, where
recall is the fraction of GRAB's own contacted vertices the reconstruction also
marks as touched. Under the transposed transform that number was 0.158.

![Shadow hand carrying the mug](figures/track_shadow_mug_drink_1.gif)

*The end of the pipeline so far: the retargeted Shadow hand carrying the real
GRAB mug along the human's own trajectory, including the tilt of the "drink"
intent. **Kinematic playback** — the object is placed at the reference pose and
the hand at the pose the SE(3) feedforward produces. Whether the grasp survives
physics is exactly what G5 and the tracking stage measure, and is not claimed
here.*

![retargeted Shadow hand](figures/retarget_shadow_mug_drink_1.png)

*The same grasp retargeted onto a Shadow hand, rendered in MuJoCo against the
real GRAB mesh. The handle is a genuine hole, not a filled-in hull.*

![GRAB bimanual handover](figures/grab_bimanual_waterbottle.gif)

*GRAB `s1/waterbottle_offhand_1`: the left hand (blue) holds the bottle, both
hands share it through the transfer, then the right hand (red) carries it away.
The `offhand` intent is a genuine hand-to-hand handover, which is what makes the
bimanual stage a data problem already solved rather than one to be invented.*

Surveying all **291 sequences** (`s1` + `s2`, 51 objects) for frames where both
hands are within 5 mm of the object at once — `results/grab_inventory.json`:

| | sequences | bimanual | rate |
|---|---|---|---|
| `offhand` (hand-to-hand transfer) | 40 | 37 | **92.5%** |
| `lift` | 62 | 30 | 48.4% |
| `pass` | 74 | 25 | 33.8% |
| `inspect` | 33 | 11 | 33.3% |
| all intents | **291** | **136** | **46.7%** |

63 of those hold with both hands for ≥15 consecutive frames, which is the subset
the bimanual stage trains on. "Bimanual" here is measured contact, not the fact
that GRAB happens to track two hands.

The **robot** side of this does not work yet, and there is deliberately no
picture of it here. Two Shadow hands are built and simulated together
(`scene.build_bimanual`, real left models from Menagerie rather than a mirrored
right one), but the two hands are retargeted in separate single-hand scenes, so
neither solver sees the other: on `gamecontroller_play_1` they interpenetrate by
11.7 mm across 42 contacts and the left pushes the right clean off the object.
Separating them along the contact normal removes the interpenetration and makes
tracking *worse* (105 mm → 255 m), which says the two poses are mutually
inconsistent rather than merely overlapping. See [NOTES](NOTES.md).

Two things had to be fixed before this picture was honest:

**MuJoCo collides a mesh as its convex hull.** For GRAB that is not a small
approximation — the mug's hull is **3.52×** the mug's own volume, because it
fills both the cup's cavity and the handle's hole. Every handle grasp in the
dataset would have been physically impossible, and a correctly placed hand reads
as 20–40 mm of penetration that is not there. Convex decomposition brings the
mug to 0.92× (`src/oppdef/human/decompose.py`).

### Stage 3: per-reference tracking — solved by PPO

| controller | mean | final | frames within 50 mm |
|---|---|---|---|
| open-loop feedforward | 8197.3 mm | 52878 mm | 53/111 |
| PPO, training horizon 64 | 10535.3 mm | 62601 mm | 57/111 |
| **PPO, training horizon 160** | **29.4 mm** | **31.9 mm** | **111/111** |

844,800 control steps. The whole story is the horizon: PPO learns a correction
over the window it trains on and does not extrapolate past it, and the
evaluation is a 111-step rollout. A reshaped reward was a wash and 614k steps at
horizon 64 did not help — matching the horizon to the task did.



A tracking episode has to START from a formed grasp. The hold window begins
where the human's hand first comes within 5 mm of the object — contact
starting, not the grasp being formed — and an episode begun there drops the
object before it has tracked anything. **17 of 24 sampled frames hold the object
on their own**, and from one of those:

| start | frames kept | mean position error |
|---|---|---|
| 60 | **59 of 59** | **11.9 mm** |
| 75 | 44 of 44 | 14.4 mm |
| 35 | 24 of 84 (drops) | — |

MPPI over the feedforward closes that marginal case: from frame 35 it tracks to
the end at **53.0 mm** mean, against a feedforward that drops the object and
ends 76 m away. From frame 20 both fail, correctly — the hand has not closed yet.

Getting here required fixing a defect that invalidated every earlier physical
number: the retargeter targeted fingertip **body origins**, and Shadow's real
tip is 32–36 mm beyond one, so the tip geom was buried by its own radius —
**4469 N** of contact force on a 0.2 kg mug. Targeting the derived tip gives
64 N and zero penetration.

### The retarget is a prior, not a starting state

G5 asked whether a retargeted pose holds the object under gravity. Pre-registered
threshold 0.50; measured **0.318**, clustered CI [0.270, 0.367] over 279
sequence-clusters. **FAIL.** Its failures make **0.1 contacts at reset** against
17.4 for its successes — they are not slipping grasps, they are poses that never
touch. The retargeting solves fingertip positions against a static object;
nothing in it asks the result to close on anything.

G5's decision rule, fixed before the experiment ran, says to initialise tracking
from a grasp **synthesised near the human's contact set**. It is right:

| initialisation | holds the object |
|---|---|
| retargeted pose, as fitted | 4/24 = **0.167** |
| closing the fingers on it | 0.276 (from 0.288 — no effect) |
| **searching near it for a pose that holds** | 19/24 = **0.792** |

Closing alone cannot work, because the failures are not touching anything to
close on. The hand has to be repositioned. The correction is a constant wrist
offset in the object frame — the trajectory's shape is untouched, only the grasp
moves — and it is kept only when it improves the whole reference (unguarded, it
took one clip from 6 graspable frames to 1). Guarded: **16 → 38** graspable
frames over six references, never worse.

### Stage 7: the estimator's error is structured, and that is the finding

The controller is frozen and only the object pose it reads is swapped. Scored on
the truth in every condition, on `mug_drink_1`:

| pose the controller reads | pose error | tracking error | outcome |
|---|---|---|---|
| simulator ground truth | — | 101.5 mm | held |
| **ICP on rendered depth** | 38.4 mm | **637.3 mm** | **dropped** |
| ground truth + Gaussian noise | **61.8 mm** | 102.5 mm | held |

Random error *larger* than the estimator's costs almost nothing, while the
estimator's own error destroys tracking. So ICP's error is **structured** —
biased and drifting, not random — and the fix is perception, not control. The
noise condition exists precisely to separate those two, and neither of the other
two rows can distinguish them alone.

**Retargeting cannot be done in the dataset's world frame.** GRAB puts the
object 0.8–1.7 m above the origin while the floating hand base has 0.6 m of
travel; the fit pins itself against its limits and reports 486 mm of error.
Solved in the *object* frame — which is the frame the result is used in — the
same fit reaches 6.7 mm.

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
  human/      mano.py (MANO without chumpy) · grab.py (GRAB references)
              decompose.py (convex parts) · retarget.py (human -> robot)
              scene.py (hand + real object) · track.py (tracking env)
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
