# The pipeline, stage by stage

Detail moved out of the README, which is an orientation document. For the live
problem see [STAGE2.md](STAGE2.md); for what is authoritative see
[NOTES.md](../NOTES.md).

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
| 2. retarget | human contact points → robot joint trajectory | **0.275 → 0.850 hold rate, 40 objects, reproduces exactly at frozen HEAD** (`b7e2e49`; the night's "does not reproduce" was a subject-key collision, `c407fba`) — of 34 holds, 20 grasps, 8 mixed, 6 burials; six failures are one geometric class (featureless convex primitives); see [STAGE2.md](STAGE2.md) |
| 3. per-reference tracking | **PPO** per clip (plus MPPI + homotopy) | **not physical, blocked on stage 2** — 29.35 mm is achieved on 13.8 mm of penetration (1562 N placement force on replay, 2832 N live at rollout end); `gamecontroller`'s 27.2 mm withdrawn. Every checkpoint is invalid once the initial condition changes |
| 4. homotopy curriculum | solve an easier reference, deform it into the hard one | **built** — walks λ 0.35 → 1.0 holding throughout |
| 5. distillation | one neural tracking controller across references | **incomplete** — reported held-out position error is 158–456 mm; that error alone does not verify continued grasp retention |
| 6. bimanual | joint retargeting + wrist-offset search | **fails under physics** — and 2 of 3 clips are effectively one-handed; see [docs/STAGE2.md](STAGE2.md) |
| 7. perception | depth → pose estimator → evaluate the *frozen* tracker | **built** — and it already says something (below) |

Every number in this section is from the **corrected** pipeline. An earlier set
was computed with the object's rotation transposed and is
[retracted](../NOTES.md).

## From human motion to a robot hand

![GRAB mug_drink_1](../figures/grab_mug_drink_1.gif)

*GRAB `s1/mug_drink_1`: the human right hand (red) reaching, taking the mug **by
its handle**, drinking, and setting it down. The left hand stays 53 cm away,
which is what "drink" should look like. Drawn as the MANO surface — a stick
skeleton cannot show whether a hand is curled.*

> Two earlier versions of this caption were wrong, and the second one was wrong
> because the pipeline was. It first claimed the handle from a thumbnail, then
> claimed the **body** on the strength of a measurement taken while the object's
> rotation was **transposed** (GRAB poses objects as `v @ R`, not `v @ R.T`).
> GRAB's own per-vertex contact labels put 41–100% of this clip's contacts on
> the handle. See [the retraction](../NOTES.md).

The reference is validated against **GRAB's own per-vertex contact labels**, not
against itself. Every earlier check here compared the reconstructed hand to the
reconstructed object and asked whether they were close; they always were, and
for a while they were close in the wrong place — the object's rotation was
transposed and a 0.1 mm minimum-distance check passed the whole time.

`experiments/tracking/grab_validate.py` scores the reconstruction against the labels the
dataset ships: **recall 0.919, minimum 0.846** over the sequences checked, where
recall is the fraction of GRAB's own contacted vertices the reconstruction also
marks as touched. Under the transposed transform that number was 0.158.

![Shadow hand carrying the mug](../figures/track_shadow_mug_drink_1.gif)

*The end of the pipeline so far: the retargeted Shadow hand carrying the real
GRAB mug along the human's own trajectory, including the tilt of the "drink"
intent. **Kinematic playback** — the object is placed at the reference pose and
the hand at the pose the SE(3) feedforward produces. Whether the grasp survives
physics is exactly what G5 and the tracking stage measure, and is not claimed
here.*

![retargeted Shadow hand](../figures/retarget_shadow_mug_drink_1.png)

*The same grasp retargeted onto a Shadow hand, rendered in MuJoCo against the
real GRAB mesh. The handle is a genuine hole, not a filled-in hull.*

![GRAB bimanual handover](../figures/grab_bimanual_waterbottle.gif)

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
its corrections remain in [NOTES](../NOTES.md). A successful two-hand clip does
not establish that two hands are necessary: a matched one-hand comparison is
still needed for that claim.

Two things had to be fixed before this picture was honest:

**MuJoCo collides a mesh as its convex hull.** For GRAB that is not a small
approximation — the mug's hull is **3.52×** the mug's own volume, because it
fills both the cup's cavity and the handle's hole. Every handle grasp in the
dataset would have been physically impossible, and a correctly placed hand reads
as 20–40 mm of penetration that is not there. Convex decomposition brings the
mug to 0.92× (`src/handsim/human/decompose.py`).

### Stage 3: a per-reference PPO result, on an invalid initial condition

| controller | mean | final | frames within 50 mm |
|---|---|---|---|
| open-loop feedforward | 8197.3 mm | 52878 mm | 53/111 |
| PPO, training horizon 64 | 10535.3 mm | 62601 mm | 57/111 |
| **PPO, training horizon 160** | **29.4 mm** | **31.9 mm** | **111/111** |

844,800 control steps. Extending the training horizon was the useful change
in this comparison: evaluation spans 111 steps. The reproduced checkpoint
also depends on the hold-scored grasp setup used during training; its weights
alone do not specify a reproducible episode.

**This whole table is measured on a penetrated grasp** and none of it should be
read as physical. The 29.4 mm is achieved at 13.8 mm of interpenetration under
1530 N on a 1.96 N object, and the accompanying 44.8° orientation error is the
object rotating inside a cage it was never held by — see
[docs/STAGE2.md](STAGE2.md). The comparison between
rows may still be informative about horizon, since all three share the same
initial condition, but the winning row is not a working controller. The
`gamecontroller` 27.2 mm figure reported elsewhere from this stage is
**withdrawn**: its policy was trained on the same invalid initial condition.



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

That fix is real, but the number it earned was measured on a retargeted pose in
isolation, under settling, and **it does not survive the rest of the pipeline**.
On `mug_drink_2` the retarget alone still leaves 11.08 mm and 4411 N before
grasp search runs at all. Settling cannot recover it either: with gravity off
for 0.5 s, 12.7 mm becomes 11.8 mm while the object is ejected 28 mm and every
contact is lost. A penetrated pose is not a grasp the simulator can repair.

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

### Stage 7: withdrawn — all three conditions dropped the object

**Withdrawn 2026-09-15 (`1d138b7`).** The stored stage-7 result was a single
reference, `mug_drink_1`, started from frame 0 — the approach, not the grasp —
and the object was dropped in *all three* conditions: truth 101.5 mm (held
0.22), depth 637.3 mm (held 0.02), noise 102.5 mm (held 0.10). Depth being 6×
worse than noise of the same magnitude would have been a real finding — a
structured estimator error, so the fix is perception rather than control — but
the truth condition drops the object too, so all three measure the same
failure. The second reference in the stage's default list had no graspable
frame, so a stage built to compare three pose sources ran on one clip that
failed in all of them. It is now seeded from stage 2's wrist offset like stage
3, and records the seed's contact count and grip force, so a perception number
cannot be read as a perception result when it is a burial being tracked. The
section below is kept as the record of what was claimed.

### Stage 7 as previously written

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

