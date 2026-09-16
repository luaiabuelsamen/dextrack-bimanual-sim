# Experiments

Grouped by research program. See the [top-level README](../README.md) for what
each program established.

## `grasp_metrics/` — program A, settled

Does the objective a retargeter optimises predict whether the grasp does the
job? Sampled grasps across four hands and four object shapes, pre-registered,
inference clustered by grasp.

| script | question |
|---|---|
| `inventory.py` | which hand/object cells are even feasible |
| `g1.py`, `g1_analysis.py` | does a demonstration help as an *initialisation*? |
| `g2.py`, `g2_analysis.py` | does it help as a *task specification*? |
| `g3.py`, `g3_analysis.py` | **do grasp metrics predict task success?** the main result |
| `g4.py`, `g4_analysis.py` | are those outcomes reproducible under perturbation? |
| `g5.py`, `g5_analysis.py` | does a retargeted human grasp hold under gravity? |
| `matched.py`, `fig_matched.py` | matched-budget comparison |
| `bimanual_expert.py` | the two-handed peg task and its one-handed control |
| `render_tasks.py`, `fig_g3.py` | figures |

## `tracking/` — program B, live

GRAB clip → retargeted trajectory → grasp → per-reference PPO → distillation,
extended to two hands, then evaluated under depth perception.
**Currently blocked at the retarget — see [docs/STAGE2.md](../docs/STAGE2.md).**

| script | role |
|---|---|
| `grab_inventory.py` | which GRAB sequences have usable contact |
| `grab_decompose.py` | convex decomposition so handles are holes |
| `grab_validate.py` | reconstruction against GRAB's own contact labels |
| `grab_render.py`, `grab_retarget_render.py`, `grab_track_render.py`, `grab_bimanual_render.py` | kinematic renders |
| `render_tracking.py` | **physics** rollouts with penetration/force instrumentation |
| `g7_perception.py` | depth → pose estimate → frozen tracker |
| `g8_distill.py`, `g9_ppo_distill.py` | distillation across references |

## `infra/`

`vec_bench.py` batched-stepping throughput and agreement; `mjx_parity.py` and
`warp_parity.py` backend parity. MJX-JAX does not run on this machine —
`gpusolverDnCreate` fails, so no MJX number should be quoted from it.

## `retracted/`

Withdrawn experiments, kept with a README saying what was wrong. Nothing here
should be developed further.
