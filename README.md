# opposition-deficit

[![simulation](https://img.shields.io/badge/simulation-MuJoCo-2a78d6)](src/oppdef/human/track.py)
[![license](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)

A simulation workbench for **turning human hand-object motion into dexterous
robot tracking**. Real GRAB references drive retargeting, grasp search,
per-reference PPO, and experiments with two robot hands.

> **The name is a historical artifact.** It refers to an "opposition deficit"
> that does not exist — the measurement behind it was taken at the wrong point
> on every hand. The original claim is retracted; the name remains for continuity.
> See [Status](#status) and the full [measurement log](NOTES.md).

---

## Physics rollouts

These recordings step a free object under gravity in MuJoCo. The camera follows
the reference path (cyan); the measured object path is amber. Each frame
shows position and orientation error. Right hands are silver; left hands
are teal. Forearms are hidden for visibility.

| Mug — per-clip PPO | Bowl — bimanual grasp search |
|---|---|
| ![PPO-controlled Shadow hand tracking a GRAB mug reference](figures/physics_mug.gif) | ![Two Shadow hands tracking a bowl reference in physics](figures/physics_bowl.gif) |

| Binoculars — bimanual grasp search | Camera — bimanual grasp search |
|---|---|
| ![Two Shadow hands tracking binoculars in physics](figures/physics_binoculars.gif) | ![Two Shadow hands tracking a camera in physics](figures/physics_camera.gif) |

Measured from the exact rollouts shown above:

| Clip / saved measurements | Mean position error | Frames under 50 mm | Mean orientation error |
|---|---:|---:|---:|
| [Mug](figures/physics_mug.json) | 29.4 mm | 111/111 | 44.8° |
| [Bowl](figures/physics_bowl.json) | 18.0 mm | 131/131 | 45.9° |
| [Binoculars](figures/physics_binoculars.json) | 32.6 mm | 138/138 | 20.0° |
| [Camera](figures/physics_camera.json) | 34.4 mm | 154/161 | 138.4° |

**Scope:** selected, already-grasped reference windows with 0.20 kg objects
and floating, externally driven wrists. The mug uses a PPO policy trained
for that clip. The two-hand examples use jointly fitted grasps and wrist
offsets searched on the displayed trajectory, followed by feedforward
control. They do not demonstrate a learned bimanual policy or held-out
generalization.

Position tracking is promising; **orientation tracking remains incomplete**.
The camera clip is a clear failure of orientation tracking despite its small
position error; its GRAB intent label does not mean the robot completed that task.
The angular errors compare the full reference orientation without an
object-symmetry adjustment. Each capture matches the existing evaluator
exactly after reset. Reproduce with `make render-tracking`; the
[renderer](experiments/render_tracking.py) saves per-frame errors, contact
counts, source/model hashes, and search settings beside every GIF.

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
retargeting and grasp search. See [From human motion to a robot hand](#from-human-motion-to-a-robot-hand).

## Status

| claim | status | evidence |
|---|---|---|
| A wrench objective beats the shipped retargeting pipeline | **supported** | pre-registered, n=60, McNemar p = 0.0075 |
| …by *selecting* better, not searching more | **supported** | finds fewer valid grasps, survives 2.3× as often |
| References improve the earlier synthetic objectives | **benefit not established** | G1 initialisation p = 1.0000; G2 specification p = 0.180; this does not test all uses of human data |
| Contact **force** predicts task success | **supported** | ρ +0.325, AUC 0.800, clustered by grasp; survives G4 |
| ε predicts task success | **weakly** | AUC 0.684 (0.679 excluding non-reproducible grasps), below the pre-declared ρ = 0.3 |
| Every metric inverts on capsules | **robust across three bug fixes and an audit** | see the figure below |
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

## Where this is headed

Each goal is pre-registered before it is written, with its decision rule fixed
in advance, including the branch where the result kills the idea.

| | question | outcome |
|---|---|---|
| **G1** | does a wrench objective beat the pipeline people ship? | **yes** (p = 0.0075); the demonstration adds nothing (p = 1.0000) |
| **G2** | does the demonstration specify the *task*? | **no** — and conditioning on it costs grasp strength |
| **G3.1** | can all four hands even be placed? | **fixed** — Shadow went from 0/320 valid grasps to 16–38% |
| **G3.2** | do grasp metrics predict task success? | contact force does (AUC 0.800); ε weakly (0.684) |
| **G4** | how sensitive are outcomes to small perturbations? | 0.928 unanimity, CI [0.897, 0.956]; exact replay failures remain |

**G4 measured perturbation sensitivity.** Under ±1 mm and ±2% perturbations,
92.8% of the sets of five repeats were unanimous (clustered CI [0.897, 0.956]),
and majority outcomes agreed with G3 at 99.6%. Its 80% threshold was fixed
before the code was written. One caveat is recorded rather than buried: 5 of
228 G3 grasps (2.2%) do not re-form standalone — they existed only given
accumulated scene state — and excluding them moves ε from AUC 0.684 to 0.679
and contact force from 0.800 to 0.795. Exact replay failures remain a limitation;
unanimity under perturbations alone does not validate physical realism.

**G5:** the raw retarget held in 0.318 of trials, below its registered 0.50
threshold. This triggered grasp synthesis before tracking (details below).

**The pipeline being built**, stage by stage. Each stage is gated on a
measurement, not on the previous stage having compiled:

| stage | what it does | state |
|---|---|---|
| 1. human reference | GRAB clip → object pose over time, both MANO hands | **done** — hand closes to 0.1–0.6 mm of the object and holds |
| 2. retarget | human contact points → robot joint trajectory | **done** — ~13 mm to the human's contacts, 0 mm penetration after settling |
| 3. per-reference tracking | **PPO** per clip (plus MPPI + homotopy) | **mug checkpoint reproduced** — 29.35 mm mean position error, 111/111 frames under 50 mm; other clips still fail |
| 4. homotopy curriculum | solve an easier reference, deform it into the hard one | **built** — walks λ 0.35 → 1.0 holding throughout |
| 5. distillation | one neural tracking controller across references | **incomplete** — reported held-out position error is 158–456 mm; that error alone does not verify continued grasp retention |
| 6. bimanual | joint retargeting + wrist-offset search | **selected physical rollouts demonstrated** — see measured GIFs above; no learned bimanual controller yet |
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

63 of those hold with both hands for ≥15 consecutive frames, which identifies
reference windows for the bimanual stage. "Bimanual" here is measured contact, not the fact
that GRAB happens to track two hands.

The robot side now fits both Shadow hands in one scene, commands both hands
during physics, and searches constant wrist offsets against tracking over the
full reference. The physical rollouts at the top of this README show this
version. Earlier independently fitted hands interpenetrated; that failure and
its corrections remain in [NOTES](NOTES.md). A successful two-hand clip does
not establish that two hands are necessary: a matched one-hand comparison is
still needed for that claim.

Two things had to be fixed before this picture was honest:

**MuJoCo collides a mesh as its convex hull.** For GRAB that is not a small
approximation — the mug's hull is **3.52×** the mug's own volume, because it
fills both the cup's cavity and the handle's hole. Every handle grasp in the
dataset would have been physically impossible, and a correctly placed hand reads
as 20–40 mm of penetration that is not there. Convex decomposition brings the
mug to 0.92× (`src/oppdef/human/decompose.py`).

### Stage 3: a reproduced per-reference PPO result

| controller | mean | final | frames within 50 mm |
|---|---|---|---|
| open-loop feedforward | 8197.3 mm | 52878 mm | 53/111 |
| PPO, training horizon 64 | 10535.3 mm | 62601 mm | 57/111 |
| **PPO, training horizon 160** | **29.4 mm** | **31.9 mm** | **111/111** |

844,800 control steps. Extending the training horizon was the useful change
in this comparison: evaluation spans 111 steps. The reproduced checkpoint
also depends on the hold-scored grasp setup used during training; its weights
alone do not specify a reproducible episode. The displayed result measures
position tracking. Its 44.8° mean orientation error remains a limitation.



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

### Establishing a grasp before tracking

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

The larger independent-noise perturbation preserves tracking in this example,
while the ICP estimate does not. That suggests temporal structure, bias, or
outliers matter beyond average pose error. It motivates tests of estimator
latency and drift, and of controller recovery; one clip does not isolate the
cause or establish that control changes cannot help.

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
