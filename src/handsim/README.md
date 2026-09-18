# Package layout

The package is still called `handsim`, after the retracted "opposition deficit"
claim the repository was originally named for. Renaming it is mechanical and
deliberately deferred; see the [top-level README](../../README.md).

Two bodies of work live here. Read this before assuming a module is dead —
several of the older ones are load-bearing for the newer ones.

## The live pipeline

**`human/`** — human motion to robot tracking, the DexTrack pipeline. GRAB
clip → retargeted trajectory → grasp → per-reference PPO → distillation, plus
the two-hand variants and the depth-perception stage.

| module | role |
|---|---|
| `grab.py`, `mano.py` | load GRAB sequences and MANO hand meshes |
| `decompose.py` | convex decomposition of object meshes (handles are holes, not hull) |
| `retarget.py` | **human contacts → robot joint trajectory.** Currently the weak link — it has no non-penetration constraint; see the top-level README |
| `scene.py` | build the MuJoCo scene, one hand or two |
| `track.py` | `ReferenceTracker`, `BimanualTracker`, grasp synthesis, MPPI |
| `rl.py`, `distill.py`, `homotopy.py` | PPO per reference, distillation, curriculum |
| `grasp.py`, `perception.py` | grasp fitting; depth → pose estimation |

## Shared foundation

Used by both bodies of work. **`human/` imports from all of these**, so none of
it is legacy in the sense of removable.

| module | role |
|---|---|
| `grasping/` | grasp quality and synthesis: `epsilon.py` (Ferrari–Canny and the wrench hull), `synth.py`, `hold.py`, `task.py`, `retarget_pose.py` |
| `hands/` | hand models and specs. `tips.py` distinguishes a fingertip from a body origin — the retarget depends on it, and getting it wrong produced a 4469 N contact force |
| `envs/` | simulation environments. `bimanual.py` provides `_add_base_dof`, which `human/scene.py` calls |
| `embodiment.py`, `paths.py`, `objects.py` | hand/object abstractions and asset paths |
| `learning/bc.py` | the trainer `human/distill.py` imports |

## Earlier synthetic-grasp work

Backs the **supported** G3 findings (contact force predicts task success at
AUC 0.800; every metric inverts on capsules). Not part of the tracking
pipeline, but its results still stand and its code must stay runnable.

`bench.py`, `track_core.py` (reference-agnostic tracking core), `policy.py`,
`sensing.py`, `data.py`, `vec.py`, `control/`, `sim/`, `viz/`.

## Naming

Two modules used to shadow each other. Resolved:

| was | is | why |
|---|---|---|
| `handsim/retarget.py` | `grasping/retarget_pose.py` | single-pose retargeting under a choice of objective (keypoint vs ε); `human/retarget.py` is the trajectory version and imports it |
| `handsim/track.py` | `track_core.py` | the reference-agnostic tracking core; `human/track.py` is the GRAB-specific tracker |

## Where the measurements live

`results/` and `figures/`, with withdrawn work under `retracted/` in each,
every directory carrying a README saying what was wrong. `attic/` holds
superseded scripts, two of which are kept as evidence for retractions.
**[NOTES.md](../../NOTES.md) is the authoritative log.**
