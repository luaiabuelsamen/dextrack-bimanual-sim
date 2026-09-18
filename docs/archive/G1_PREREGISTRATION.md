# G1 pre-registration — does a wrench objective beat the pipeline people ship?

Written and committed **before** the experiment was implemented or run. Commit
this file first; any deviation is recorded in an amendment section below rather
than edited in.

## Why

One result survives the retractions: under a matched budget, a wrench objective
selects better grasps than a fingertip-fidelity objective from the same
feasible set (`NOTES.md`, 2026-09-13). But the fidelity arm in that experiment
was *the most faithful grasp found by free search*, which nobody ships. The
pipeline practitioners actually use is **keypoint retarget, then a budgeted
squeeze / contact refinement**. If that matches a wrench-conditioned method,
the surviving result is about search objectives in grasp synthesis, not about
how to retarget human hand-object data, and this project's framing is wrong.

## Arms — three, identical budget, identical executor

All three are closed in the same `GraspScene`, measured on real MuJoCo
contacts, and scored by the same probe. Every arm gets the **same number of
simulated attempts** (`iters x pop`). How an arm spends that budget is the
method under test.

| arm | initialised from | searches | objective |
|---|---|---|---|
| `pose_squeeze` | keypoint retarget | placement + squeeze fraction | fidelity to the retargeted pose, among grasps |
| `eps_synth` | the hand's own derived closure | placement + all finger angles | measured epsilon |
| `wrench_refine` | keypoint retarget | placement + all finger angles | measured epsilon |

`pose_squeeze` holds the finger pose at the retarget and closes it by a
searched fraction toward the hand's closure — the budgeted squeeze. The only
difference between `eps_synth` and `wrench_refine` is **where the search
starts**: the hand's own closure, or the human demonstration. That isolates
whether the demonstration contributes anything once a wrench objective is doing
the optimising.

## Pre-declared outcome

**Primary (binary, per cell):** does the arm yield a grasp that survives the
sampled disturbance probe — every sampled direction holding at the smallest
rung (0.25 N / 0.25 N·m·λ), with final displacement < 20 mm, final rotation
< 15°, and contact retained?

**Secondary:** the worst-direction magnitude in newtons; valid candidates found
per budget (the search-vs-selection diagnostic).

Cells are (hand × width × seed). Hands: shadow, leap, allegro, f5d6. Widths:
3.0, 4.5, 6.0, 7.5, 9.0 cm. Seeds: 0, 1, 2. n = 60 per arm.

## Pre-declared analysis

Paired by cell. McNemar exact (two-sided binomial on discordant pairs) for the
primary; Wilcoxon signed-rank on non-tied cells for the secondary. Results
reported seed-stratified as well as pooled, because **60 cells are 20
configurations at 3 optimiser seeds, not 60 task instances**.

Two comparisons, both declared now:

* **H1 (the thesis):** `wrench_refine` > `pose_squeeze` on the primary.
* **H2 (does the human data help?):** `wrench_refine` vs `eps_synth`.

## Pre-declared decision rule

* **H1 not significant** (`p > 0.05`): the wrench objective does not beat the
  shipped pipeline. The retargeting framing is not supported; the surviving
  contribution is about objectives for grasp *synthesis* and the project should
  be reframed and renamed accordingly.
* **H1 significant, H2 not:** a wrench objective beats the shipped pipeline,
  but the human demonstration contributes nothing beyond initialisation — the
  contribution is the objective, not the retargeting.
* **H1 significant and `wrench_refine` > `eps_synth`:** the demonstration
  carries information a wrench objective alone does not, which is the only
  outcome that supports the project as originally framed.

A null is a result and will be reported as one. No additional arms, widths,
seeds, probe thresholds or outcome definitions will be introduced after seeing
the numbers; anything added later appears under Amendments and is labelled
exploratory.

## Known limitations, stated up front

Synthetic reference (`SyntheticSource`), cubes only, one analytic grasp family,
single hands, assisted grasp formation (the object is pinned while the grasp
forms and released before measurement), and an evaluation that shares its
contact model with the objective. This experiment can refute the framing; it
cannot by itself establish a claim about human data.

## Amendments

*(none)*
