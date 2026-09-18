# G2 pre-registration — does a demonstration specify the *task*?

Written and committed **before** the experiment was implemented or run. The
task machinery (`src/handsim/grasping/task.py`) was built and calibrated first, on G1's
saved grasps; that pilot is declared below. No G2 arm has been run.

## Why

G1 found that a demonstration contributes nothing beyond initialisation when
the target is a static grasp (`eps_synth` vs `wrench_refine`: 21/60 vs 21/60,
exact tie, p = 1.0000). The rejoinder is that a static grasp is not what a
demonstration is *about*: a record of a human moving an object specifies where
the object goes, not which joint angles to use. G1 gave the demonstration only
an initial guess. Here it supplies the **specification**.

## The task — T3, carry under gravity

The demonstration is an **object** trajectory: lift 8 cm, tilt 60° about its own
centre, translate 8 cm sideways, set down. Smooth (minimum-jerk) segments of
0.5 s with brief holds; ~4 s total; gravity ON.

*Object-centred, not hand-centred.* A demonstration records what the object did
and says nothing about a palm. The executor therefore commands a compensating
base translation so that a rigidly-held object rotates about **its own centre**
— otherwise the required wrench would depend on where the hand happened to be
and would not be a task specification at all.

**Required wrench sequence.** From the specified object motion: force
*m*(**a** − **g**), torque **I·α + ω×Iω** about the object's centre, scaled by
the object's characteristic length exactly as `wrench_set` scales its torque
rows, so the two live in the same space. 80 samples along the path. For the
default 50 g, 6 cm cube this demands 0.40–0.58 N of force and up to 0.014 of
scaled torque.

**Success — measured on the object, never on any objective.** *Slip in the palm
frame*: the object's position relative to the hand must stay within **15 mm** of
where the grasp left it, it must never exceed 60 mm (a drop), and the object
must actually be carried at least 50 mm.

> Slip, not tracking error against the command. The base is position-controlled
> with finite gain, so the whole hand lags; scored against the command, a
> perfectly held object fails for the actuator's reasons and the experiment
> would be measuring controller gain.

## Arms — three, identical budget, identical executor

Every arm gets the same number of simulated attempts. All are formed by the
same pre-grasp → close → squeeze → release protocol (T1) and then run T3.

| arm | searches | objective | uses the demonstration for |
|---|---|---|---|
| `pose_squeeze` | placement + squeeze fraction | fidelity to the keypoint retarget, among grasps | the hand's **pose** |
| `task_generic` | placement + all finger angles | Ferrari–Canny ε (worst case over every direction) | **nothing** |
| `task_demo` | placement + all finger angles | task margin against the required wrench sequence | the **object's trajectory** |

ε is the worst case over all directions; the task margin is the worst
resistible-to-required ratio over the directions *this task* demands. That is
the whole difference, and it is the only thing `task_demo` knows that
`task_generic` does not.

## Pre-declared outcomes — two co-primaries

* **P1 (reliability):** task success, binary, as defined above.
* **P2 (magnitude):** the worst-direction force the selected grasp survives on
  the static T1 probe, run on the same grasp.

P2 is declared because G1 found, *exploratorily*, that demonstration-initialised
search produced stronger grasps (0.525 vs 0.223 N, p = 0.0163) while producing
no more reliable ones. That finding has never had a pre-registered test. It gets
one here without a separate run.

## Pre-declared analysis

Paired by cell. Cells are hand × width × seed: hands shadow, leap, allegro,
f5d6; widths 4.5, 6.0, 7.5 cm; seeds 0, 1, 2. n = 36 per arm.

McNemar exact (two-sided binomial on discordant pairs) for P1; Wilcoxon
signed-rank on non-tied cells for P2. Reported seed-stratified as well as
pooled, because **cells are 12 configurations at 3 optimiser seeds, not 36 task
instances**. Valid candidates per budget reported for every arm, as the
search-versus-selection diagnostic.

Two comparisons:

* **H1:** `task_demo` > `pose_squeeze` — does an object-side specification beat
  a hand-side one on a task?
* **H2 (the question):** `task_demo` > `task_generic` — does the demonstration's
  task information buy anything a generic worst-case objective does not?

## Pre-declared decision rule

* **H2 significant on P1** → *human data specifies the task, not the grasp.*
  The project has a thesis again, in a form that makes human demonstration
  necessary rather than decorative.
* **H2 null on P1, significant on P2** → the demonstration buys grasp
  *strength* but not task *reliability*, confirming G1's exploratory finding
  under pre-registration. A narrower claim, reported as such.
* **H2 null on both** → *object-side wrench specification is sufficient; human
  hand-object data is dispensable for this class of problem.* Two independent
  pre-registered nulls. This is a publishable result and will be reported as
  the headline, not buried.
* **H1 null** → neither object-side arm beats the shipped pipeline on a task,
  which would contradict G1 and require explaining before anything else is
  claimed.

No arms, widths, seeds, thresholds or outcome definitions will be added after
seeing the numbers. Anything added later appears under Amendments, labelled
exploratory.

## Declared pilot

Segment duration and the slip threshold were chosen on a pilot over G1's
**already-saved** grasps (statically optimised, not task-optimised), varying
0.14–0.70 s. At 0.5 s that pilot gave `pose_squeeze` 4/12 task successes and
`wrench_refine` 7/11 — discriminating, neither floored nor saturated. The pilot
touched no G2 arm and no G2 objective.

Known during the pilot and stated here: `pose_squeeze` grasps that fail the task
do so by **slipping out of the hand** (~60 mm), not by lagging, and the task
margin of one failing grasp was 0.85 — below 1, meaning it could not supply the
required wrench. Whether margin predicts failure in general is **not** a
declared outcome of this experiment.

## Known limitations, stated up front

Synthetic reference, cubes only, one trajectory family, single hands, assisted
grasp formation (the object is pinned while the grasp forms, released before
anything is measured). The task margin and the T1 probe share a contact model
with the objectives, though task success does not. Shadow scored 0/60 across
all arms in G1 at this budget; if it does so again the cause is search power at
29 DoF, not the hand.

## Amendments

### Amendment 1 — 2026-09-13, after run 1 returned an uninterpretable null

**Run 1 is reported as null and uninterpretable.** Its pre-declared rule fired
the H1-null branch, which requires the contradiction with G1 to be explained
before anything is claimed. The mandated diagnosis found two defects, one of
them a bug in my own objective:

    arm             P1 task   P2 hold_N   margin    eps   valid/budget
    pose_squeeze     5/36       0.154      8.22   0.1008     21.8
    task_generic     7/36       0.253     11.01   0.1862     31.1
    task_demo        4/36       0.185     39.50   0.1402     30.4

    H1  discordant 3 vs 4   p = 1.00000
    H2  discordant 1 vs 4   p = 0.37500

1. **Half the cells were empty.** 17-18 of 36 produced no grasp at all, and
   those are exactly shadow (9/9) and f5d6 (9/9). Shadow is the 29-DoF search
   power limit already seen in G1 (0/60 there). f5d6 cannot grasp cubes of
   4.5-7.5 cm — a structural limit of a hand whose aperture is 12.73 cm, not a
   budget artifact.

2. **`task_demo`'s objective was dominated by squeeze force.** The margin is
   `support x f_total`, hence linear in contact force: `corr(margin, f_total) =
   +0.961`. Maximising it rewards squeezing rather than contact placement, and
   the arm duly learned to squeeze 3.6x harder than `task_generic` (29.3 N vs
   8.1 N). That is a defect in how I formulated the objective, not a property
   of task-conditioned specification.

The quantities were nonetheless predictive of task success — successes averaged
margin 35.9 against failures' 16.7, epsilon 0.367 against 0.103 — which is what
makes run 1 a broken experiment rather than a negative result.

**Changes for run 2**, fixed before it is executed:

* the `task_demo` objective uses the margin **per unit contact force**
  (`f_total = 1`), isolating contact geometry from squeeze strength. The
  reported margin still includes force, so the two remain distinguishable.
* the candidate budget rises from 144 to 320 for every arm equally, to give
  Shadow a fair chance at 29 DoF. **No hand is dropped**: excluding hands after
  seeing which ones failed would be cherry-picking, and f5d6's structural
  failure at these widths is itself worth reporting.
* everything else — task, arms, co-primaries, analysis, decision rule — is
  unchanged.

Run 1's results are kept at `results/retracted/g2_run1_*.json`.
