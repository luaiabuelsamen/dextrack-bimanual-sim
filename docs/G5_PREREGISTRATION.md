# G5 pre-registration — does a retargeted human grasp hold the real object?

Written and committed **before** the experiment was run. This is the go/no-go on
the tracking stage of the DexTrack pipeline.

## Why

The pipeline is: human reference → retarget → per-reference tracking (RL +
trajectory optimisation) → homotopy curriculum → distil one controller. Stages
one and two are built and validated kinematically: the MANO fit puts the human
hand within 0.1–0.6 mm of the object through the grasp, and the retarget puts
the robot's fingertips within ~13 mm of the human's contact points with ~4 mm of
penetration on non-fingertip links.

Neither of those numbers is a grasp. The retarget is **kinematic**: it matches
positions and never asks whether the resulting contact set can carry a load. A
position servo commanded to a pose that leaves a 1 mm gap applies no force, and
the object falls. On a single sequence (`s1/mug_drink_1`, Shadow) 16 of 20
sampled frames held — but one sequence, one object and one hand is an anecdote,
and the four failures were all in the frames where the human hand was still
closing, which is a pattern that either generalises or does not.

This matters because it decides what the tracking stage *is*. If retargeted
grasps mostly hold, tracking is a control problem layered on a working grasp. If
they mostly do not, then the first job of the tracking optimiser is to **find a
grasp at all**, and a tracking reward alone will not do it.

## Hypotheses

**H1 (gate).** A retargeted pose taken from the human's hold window holds the
object against gravity for 1.0 s in **≥ 50%** of sampled frames, pooled over
sequences.

**H2.** The hold rate is lower in the first 20% of the hold window (the human
closing) than in the middle 60%. Directional, one-sided.

**H3 (exploratory, no decision attached).** Hold rate differs by hand
(Shadow / LEAP / Allegro) and by object convexity (hull-to-mesh volume ratio).

## Method

Sequences: every `s1` and `s2` GRAB sequence whose right-hand hold window is at
least 20 frames at stride 8 (15 Hz), restricted to objects whose convex
decomposition is cached. Sampling one frame per sequence would confound "this
grasp" with "this sequence", so **10 frames evenly spaced across each hold
window** are used, and inference is clustered by sequence.

Per frame: the object is placed at the origin with identity orientation (the
frame the retarget solved in), the hand is set to the retargeted configuration,
position servos are commanded to hold it there, 0.25 s of settling is simulated
and discarded, then 1.0 s is simulated and the object's displacement measured.

**Held** := displacement < 5 cm over the 1.0 s. Free fall over that interval is
~62 cm, so the threshold is not near the boundary between holding and falling.

The hand is commanded to stay exactly where the retarget put it. No squeeze, no
closing, no search. This measures the retarget, not a grasp synthesiser.

## Analysis, fixed now

* Hold rate with a Wilson interval, and a **bootstrap clustered by sequence**,
  because 10 frames from one sequence are not 10 independent observations —
  the same mistake was already made once in this repository with 316
  observations that were 158 clusters of 2.
* H2 by the same clustered bootstrap on the difference in rate between window
  positions.
* H3 reported with intervals and **no** decision rule attached; three hands and
  a continuous covariate on the same data is not a confirmatory test.

## Decision rule

* **H1 passes** → build the per-reference tracking optimiser on top of the
  retargeted grasp, initialising from it.
* **H1 fails** → the tracking stage must include grasp establishment. The
  optimiser is then initialised from a *synthesised* grasp near the human's
  contact set rather than from the retarget, and the retarget becomes a prior on
  where to search rather than a starting state.

Either outcome is informative and neither is a failure of the pipeline. What
would be a failure is building the tracking stage without knowing which regime
it is in.

## What this does NOT establish

That the grasp can carry the object along the human's trajectory. Holding
against gravity is a static test; tracking adds inertial loads and rotation.
G5 is a prerequisite for the tracking experiment, not a substitute for it.
