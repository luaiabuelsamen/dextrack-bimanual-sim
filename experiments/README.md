# experiments/

Runnable experiments on the MuJoCo pipeline in `src/handsim/`. Run from the
repository root with `PYTHONPATH=src:.`.

| directory | what it holds |
|---|---|
| `tracking/` | the live pipeline's experiments: GRAB validation and rendering (`grab_*.py`), the carry-scored stage-2 search and its checks (`stage2_carry.py`, `carry_robust.py`, `carry_merge.py`, `render_carry.py`), per-clip PPO from carry seeds (`stage3_carry.py`, `stage3_carry_eval.py`), the README renderer (`render_tracking.py`), scoring on DexTrack's success rule (`dextrack_metric.py`), and the distillation and perception stages (`g8_distill.py`, `g9_*.py`, `g7_perception.py`). |
| `infra/` | backend benchmarks: batched stepping, warp and MJX parity. |

The GPU-side work on DexTrack's own trainer is not here; it is in
[`../dextrack/`](../dextrack/README.md). The retired grasp-metrics benchmark
and every retracted experiment are in [`../legacy/`](../legacy/README.md).
