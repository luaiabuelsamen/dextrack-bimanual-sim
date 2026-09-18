# `dextrack/` — DexTrack in Isaac Gym, instrumented

Everything that runs DexTrack's trainer on a rented GPU and measures what
its policies do to the object. The Jetson side of the repository (`src/`,
`experiments/`) renders and keeps the books; nothing here needs it.

## What is in this directory

| file | what it is |
|---|---|
| `task_hook.diff` | the patch to DexTrack's `tasks/allegro_hand_tracking_generalist.py`: two initialization fixes (`AUDIT_FIX=3`, `AUDIT_PARK=1 AUDIT_PARK_DZ=0.003`), a per-step logger (`AUDIT_LOG=<file>.npy`), an all-environment probe at evaluation (`AUDIT_ALLENV=1`), and the penetration term in the reward (`AUDIT_PEN_W`, `AUDIT_FORCE_W`, `AUDIT_PEN_GATE`, `AUDIT_PROBE_SPACING`, `AUDIT_PROBE_SDF`). Everything is env-var gated; with nothing set, their task is unchanged. |
| `penetration_torch.py` | the GPU penetration probe: the hand's URDF collision boxes and spheres sampled on a 2 mm grid (33k points per hand), the object as the face planes of its convex parts or as a 1 mm signed-depth volume. Batched over environments; 64 ms per step at 1024 environments on the cube, 124 ms on a 64-hull apple. The sparse setting (20 points per link) exists for comparison and is not safe to train against: a policy learned it (`docs/STAGE2.md`). |
| `audit.py` | the offline audit of a logged rollout: penetration from 400-point sampling of the same primitives, links touching, PhysX net contact force on touching links as a multiple of the object's weight, position and rotation error, their success flags. Independent of the probe; this is what judges a trained policy. |
| `render.py` | replays the logged Isaac Gym poses in MuJoCo as a camera (no physics), object translucent, any link past 2 mm red, the audit's numbers on every frame. Isaac Gym cannot render inside a headless container. |
| `pod/setup_pod.sh`, `pod/DATA.md`, `pod/data.sha256` | a fresh pod from the five release archives: clone, verify, extract, venv, smoke (see below). |
| `pod/run.py`, `../runs/*.json` | the spec-driven launcher and the specs of every run worth repeating. |
| `pod/ft_any.sh`, `pod/scratch_cube.sh`, `pod/dense_eval.sh`, `pod/sweep.sh`, `pod/ft_dense.sh` | the evaluation and sweep scripts as they ran; `sweep.sh` is the sparse-probe sweep, kept as the record of the mistake. |
| `pod/gen_batch.sh`, `pod/summarize_gen.py` | the 43-clip generalist audit. |
| `pod/patches/` | the incremental patches that became `task_hook.diff`, in order. |
| `pod/wandb_utils.py` | the trainer's W&B observer, restored: the release imports it commented out and the class was missing. |
| `track.py` | the ledger, W&B evaluation runs and Hub upload (see Observability). |
| `pod/train_cmd_cube.txt`, `pod/gen_cmd.txt` | DexTrack's own test command lines for the cube checkpoint and the generalist; every run is a `sed` of one of these. |

## Bringing up a pod

`pod/setup_pod.sh` on a fresh pod with the five archives from
[`pod/DATA.md`](pod/DATA.md) in `/workspace/inputs/`: it clones the
[DexTrack fork](https://github.com/luaiabuelsamen/DexTrack/tree/audit)
(branch `audit`, the hook as a commit) and this repository, verifies the
archives against `pod/data.sha256`, extracts them, builds the Python 3.8
venv with the pinned stack, and refuses to declare the pod ready until a
4-environment smoke run reports a penetration line. About fifteen minutes
plus the archive upload.

## Launching a run

Every run is a JSON spec in [`runs/`](runs/): the clip, the base command
family (`cube` or `gen`), the starting checkpoint or none, the term
weight, gate and probe settings, epochs, environments, seed. On the pod:

```bash
python /workspace/run.py /workspace/specs/cube_w1000_dense.json                 # train, then evaluate base and fine-tuned
python /workspace/run.py /workspace/specs/cube_scratch.json --tag cube_scratch_s2 --set seed=2
python /workspace/run.py /workspace/specs/apple_w100_gated.json --dry-run       # print the resolved commands
```

The launcher writes `<tag>.spec.json` beside the logs with the spec, the
resolved commands, both repositories' commits and timings; `track.py add`
folds that into the ledger row, so any row can be relaunched from its
spec. The shell scripts in `pod/` are what ran before the launcher existed
and are kept as the record.

## The protocol

1. **Train** at 1024 environments in their trainer with the term on
   (`AUDIT_PEN_W=<per metre> AUDIT_PROBE_SPACING=0.002 AUDIT_PROBE_SDF=0.001`).
   The trainer prints, every 200 steps, the probe's depth over environments
   still tracking the object.
2. **Evaluate** base and fine-tuned at 16 environments, deterministic player,
   `AUDIT_ALLENV=1 AUDIT_PROBE_SPACING=0.002` and no volume: the judge is
   the exact plane test on the dense grid, never the measure the policy
   trained on. Every environment's mean penetration, the fraction of frames
   over 2 mm, grip on touching links, tracking error and end error are
   printed on one `AUDIT_ALLENV` line.
3. **Audit env 0 offline** (`audit.py`) with the independent 400-point
   sampler, and **render it** (`render.py`) before a number is written down.

## Tests and CI

`tests/test_probe.py` pins the probe on synthetic geometry with known
answers: the dense grid recovers a 3 mm corner and a 2 mm sphere depth,
the volume matches the plane test within a third of a voxel, chunking is
invisible, and the sparse set reads no more than the dense one on the
edge geometry a policy exploited. CI (`.github/workflows/ci.yml`) runs
them with the MuJoCo pipeline's fast invariants on every push.

## Observability

Three layers, each usable without the others:

1. **The ledger**, `results/dextrack_audit/runs.jsonl`: one row per run with
   the clip, checkpoint, term weight, probe settings, epochs, and the exact
   test's base and fine-tuned metrics (per-env values included), plus the
   commit and, when present, the env-0 audit, the render, the checkpoint,
   the W&B and Hub links. `track.py add` writes a row; `track.py backfill`
   parses a directory of evaluation logs. The README's tables are read off it.
2. **Weights & Biases**, project `dextrack-bimanual`. Training runs come from
   the trainer's restored observer (`pod/wandb_utils.py`: rl_games' reward,
   episode-length and loss scalars, plus `pen/*` from the hook every 200
   steps, with every `AUDIT_*` setting in the config). Evaluation runs
   (`job_type=eval`) come from `track.py`: base and fine-tuned metrics side
   by side, per-env tables, the per-frame penetration curve, the render.
   Runs are grouped by clip. `AUDIT_WANDB=0` turns it off for a smoke run.
3. **The Hub**, private dataset `luaia/dextrack-bimanual-runs`: `track.py
   publish` uploads a run's logs, summaries, audit, render and checkpoint
   under `runs/<tag>/` and refreshes the ledger. Private, and it stays so:
   every log carries object trajectories derived from GRAB, which is
   licensed and not redistributable.

## Running it

```bash
# on the pod, after setup_pod.sh and a copy of DexTrack + data under /workspace
cd /workspace
./scratch_cube.sh cube_scratch 3000                       # the from-scratch control
AUDIT_PEN_GATE=0.1 ./ft_any.sh apple_w100 ori_grab_s10_apple_eat_1 \
    gen_cmd.txt ./ckpts/grab_trajs_tracking_ckpt.pth 100 0.002 0.001 150
# on the Jetson
PYTHONPATH=src:. python dextrack/audit.py --log out/x.npy --hand-urdf ... --obj-dir ... --out out/x.json
PYTHONPATH=src:. python dextrack/render.py --log out/x.npy --audit out/x.json --obj-dir ... --out figures/x.gif
```

The pod keeps DexTrack, its data and checkpoints under `/workspace`
(persistent); the interpreter lives on the container disk and needs
`uv python install 3.8` after every restart. `graphics_device_id=-1` for
headless; ninja must be on `PATH` for the gymtorch build. A community pod
costs $0.22 per hour; a 150-epoch fine-tune on the cube is about 25 minutes.

Results live in `results/dextrack_audit/` (audit logs and JSONs, the
generalist's 43 audits, fine-tune logs and checkpoints under `finetune/`).
The numbers are in `docs/STAGE2.md`; the reasoning in `NOTES.md`.
