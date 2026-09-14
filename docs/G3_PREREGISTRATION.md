# G3 Part 2 pre-registration — does any grasp metric predict task success?

Written and committed **before** the data-collection code existed. Part 1 (the
placement repair) is complete and its acceptance is recorded in `NOTES.md`.

## Why this question

G1 found a wrench objective beating the shipped pipeline on a **static
worst-case probe** (p = 0.0075). G2 found that advantage does not appear on a
**carry task** (p = 0.549), and the diagnosis was that the two outcomes barely
rank grasps the same way:

    corr(static probe hold, task success)   Spearman +0.141
    corr(epsilon,           task success)   Spearman +0.185
    corr(task margin,       task success)   Spearman -0.157

on 57 grasps across two hands. Ferrari–Canny ε is the field's standard grasp
metric and this project spent weeks optimising against it. If it does not
predict whether a grasp does a job, that is worth knowing, and it explains why
G1 and G2 disagreed.

## Dataset

Grasps are **sampled**, not optimised: the question is whether a metric ranks
grasps by task success across the quality range, so a search that concentrates
on high-ε grasps would truncate exactly the variation being tested.

* 4 hands: shadow, leap, allegro, f5d6
* 4 object shapes: box, cylinder, sphere, capsule (half-extents 22 × 22 × 30 mm,
  mass 50 g). A cube alone hides any dependence on where contacts sit — every
  face is identical.
* up to **15 valid grasps per (hand, shape)**, drawn from uniform random
  approach parameters, capped at 400 draws per cell. Cells that cannot reach 15
  are reported with the count they achieve; no cell is dropped.
* target n ≈ 240 grasps.

## Metrics recorded per grasp (the predictors)

`epsilon` (Ferrari–Canny from real MuJoCo contacts) · `hold_N` (worst direction
of the static 14-force + 14-torque probe) · `margin` (task-conditioned, raw) ·
`margin_per_N` (per unit contact force) · `n_contacts` · `f_total` ·
`penetration_mm`.

## Outcomes (the thing predicted)

Two tasks, each scored by **slip in the palm frame** exactly as in G2:

* **T3a** — lift 8 cm, tilt 60° about x, carry 8 cm, place (0.5 s segments).
* **T3b** — a different wrench profile: tilt about **y**, faster segments
  (0.3 s), so the required force and torque directions differ from T3a.

A grasp's outcome is recorded per task; the pooled primary uses both.

## Pre-declared analysis

**Primary:** Spearman rank correlation between each metric and task success,
pooled over both tasks, with its p-value. **Co-primary:** ROC AUC of each
metric as a classifier of task success.

**Grouping:** correlations are also reported per hand and per shape. Cells are
not independent replications of a task; the pooled figure is the headline but
the per-group figures are reported alongside it and any disagreement is stated.

**Declared comparison set:** the seven metrics above. Reporting the best of
seven without saying so would be a multiple-comparisons error, so the
Holm-corrected p-values are reported next to the raw ones.

## Pre-declared decision rule

* **Some metric achieves Spearman ρ > 0.3 with Holm-corrected p < 0.05** →
  that metric predicts task success; it should be what the project optimises,
  and G1's result is meaningful to the extent it used that metric.
* **All metrics fall below ρ = 0.3, or none survives correction** → the
  standard grasp metrics do not predict task success in this setting. That is
  the headline result, it explains the G1/G2 disagreement directly, and it is a
  benchmark critique rather than a failure of this project.
* **ε specifically shows ρ ≤ 0** → stronger still: the field's standard metric
  is uninformative or inverted for task outcomes here, which would need saying
  plainly and carefully.

Either outcome is reported. No metric, task, shape or hand will be added after
seeing the numbers; anything added appears under Amendments and is labelled
exploratory.

## Known limitations, stated up front

Simulation only, one object size, 50 g, four hands, two task variants of the
same carry family, assisted grasp formation (object pinned while the grasp
forms, released before measurement). ε, the margin and the static probe all
share a contact model with each other; **task success does not**, which is the
point. A null here is a statement about these metrics on these tasks, not a
proof that no grasp metric can predict manipulation.

## Amendments

*(none)*
