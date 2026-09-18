# G5 result — **H1 FAILS**. The retarget is a prior, not a starting state.

Pre-registration: `docs/G5_PREREGISTRATION.md`, committed before the code.
Full output: `results/g5_analysis.txt`. Data: `results/g5_hold.json`.

## Verdict

| hypothesis | pre-declared rule | result | verdict |
|---|---|---|---|
| **H1** | hold rate ≥ 0.50 | **0.255**, clustered CI [0.217, 0.294] | **FAIL** |
| **H2** | middle of window holds better than early | −0.018, CI [−0.060, +0.022] | **not supported** |

2730 observations over **273 sequence-clusters**, 51 objects, Shadow.
Inference is clustered by sequence; the Wilson interval that ignores clustering
is [0.239, 0.272], visibly too narrow, which is why the pre-registration fixed
the clustered bootstrap in advance.

## What fails, and how

The failures are **not slipping grasps — they are non-grasps**:

| | contacts at reset | penetration |
|---|---|---|
| held | **17.4** | 12.0 mm |
| fell | **0.1** | 8.0 mm |

A pose that falls makes essentially no contact at all when placed in the
simulator. The retargeting solves for fingertip positions against a *static*
object and is scored on how near the human's contact points it lands; nothing in
it requires the resulting configuration to close on anything.

H2 being unsupported matters too, because it kills the explanation I had reached
for from a single sequence. On `mug_drink_1` the early frames drop and the middle
ones hold, and it was tempting to conclude the hold window simply starts too
early. Across 273 sequences that pattern does not exist (−1.8 points, CI
spanning zero). `mug_drink_1` is an easy case, not a representative one.

## Where it fails

Hold rate by object, worst first — every one of them a smooth primitive:

    spherelarge  0.000   spheremedium 0.000   cubelarge 0.014
    cylindermedium 0.017  cubemedium 0.020    pyramidlarge 0.029

and by intent, `drink` 0.623 and `clean` 0.850 against `lift` 0.247,
`pass` 0.215 and `offhand` 0.166. Grasps that wrap a handle or a rim survive;
fingertip contacts on a large smooth convex surface do not.

## Decision, as pre-registered

> **H1 fails** → the tracking stage must include grasp establishment. The
> optimiser is then initialised from a *synthesised* grasp near the human's
> contact set rather than from the retarget, and the retarget becomes a prior on
> where to search rather than a starting state.

This is now the design of stage 3, not a patch to it. `ReferenceTracker.establish`
(pre-grasp → close → squeeze to a force target) already exists and was built
before this result; G5 says it is mandatory rather than optional, and that the
number to beat is **0.255**.

## What this does not say

That the references are unusable, or that 25% is the ceiling. It measures one
specific thing: whether the *kinematic retarget alone*, commanded as a position
target, holds the object. It was never optimised to.
