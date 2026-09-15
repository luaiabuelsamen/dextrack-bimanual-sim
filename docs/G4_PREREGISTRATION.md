# G4 Part A pre-registration — is the G3 null about metrics, or about the simulator?

Written and committed **before** the code existed. This is a go/no-go on the G3
headline and, with it, on this line of work.

## Why

G3 found that no standard grasp metric predicts task success (ε at AUC 0.544
against a coin flip's 0.500). But the strongest of its seven predictors was
**penetration depth, negatively** (ρ = −0.272, AUC 0.332): grasps whose fingers
press further into the object fail more often. That is a statement about
contact realism, not grasp quality, and its being the best available predictor
raises the obvious alternative explanation:

> if task success is dominated by contact-model artifacts, then **nothing**
> would predict it — including a perfect metric — and G3's null is a property
> of my simulator rather than of grasp metrics.

There is a clean test. A metric can only predict a *stable* property. If the
same grasp gives different outcomes on repeated runs, the outcome is noise and
no metric could have predicted it.

## Method

Every one of G3's **158 saved grasps** is re-formed from its stored approach
parameters and re-run on both carry tasks, **R = 5** times each, under
perturbations of the kind a real system would face:

* object start position offset by ±1 mm, uniform, applied after the grasp forms
  and is released, so the grasp itself is unchanged;
* object mass scaled by ±2%, uniform;
* solver warm-start reset, so no repeat inherits another's convergence path.

Plus **one unperturbed repeat** per (grasp, task) as a determinism check. MuJoCo
is deterministic, so this must agree with G3's original outcome exactly; if it
does not, the pipeline has non-determinism independent of the perturbations and
that is reported before anything else.

## Pre-declared primary

**Unanimity rate:** the fraction of (grasp, task) pairs whose R = 5 perturbed
repeats all agree with one another, with a Wilson 95% confidence interval.

**Secondary:** agreement between the majority-vote outcome and G3's original
outcome; unanimity broken down by hand, by shape, and by whether the original
outcome was success or failure.

## Pre-declared decision rule

* **Unanimity ≥ 80%** (lower CI bound above 0.75) → the task outcome is a
  stable property of the grasp. G3's null then says the metrics genuinely fail
  to predict something real, and the benchmark critique stands on firm ground.
* **Unanimity < 80%** → the outcome is noise-dominated. **G3's headline is
  withdrawn** and restated as *"task success in this setup is not reproducible"*,
  which is a defect in the simulation, not a finding about metrics. The right
  next move is then a different contact model or hardware — not another sweep.
* **Determinism check fails** (unperturbed repeat disagrees with G3) → stop and
  fix that first; nothing else in this experiment is interpretable until the
  pipeline is deterministic.

No threshold, perturbation or outcome definition will change after seeing the
numbers. Anything added appears under Amendments, labelled exploratory.

## What is deliberately NOT being done

The perturbations are not tuned. ±1 mm and ±2% are chosen as plainly smaller
than anything a real system controls to, and are fixed here before any run. If
the outcome is unstable to perturbations this small, that is the finding.

## Amendments

### Amendment 1 — 2026-09-14, after run 1 tripped its own determinism gate

Run 1 gave unanimity 438/456 = 0.961, which passes the primary. It did **not**
pass the determinism gate: the unperturbed repeat matched the saved G3 outcome
in only 444/456 = 0.974, below the 0.99 this pre-registration requires. Per the
rule, that is fixed before anything is interpreted.

Diagnosis: **10 of the 12 disagreements were grasps failing to RE-FORM**, not
physics differing, and they alternated within a cell. The cause is the mass
perturbation. Changing `body_mass` requires `mj_setConst` to rebuild derived
inertia, and restoring the scalar afterwards did not restore everything, so a
scene reused across repeats drifted.

**Change:** the ±2% mass perturbation is applied as the equivalent steady
vertical force on the object during the carry, rather than by mutating the
model. The model is never mutated mid-run. Everything else — ±1 mm offset,
warm-start reset, R = 5, the 80% threshold, the decision rule — is unchanged.

A second defect, in the analysis rather than the experiment: the saved
`unanimous` field ignores repeats whose grasp failed to re-form, so it read
456/456 where the honest count was 438/456. All reporting now uses one strict
definition — every declared repeat must re-form **and** agree — and the
grasp-clustered bootstrap uses the same one. Flagged by the codex review before
it could reach a result.
