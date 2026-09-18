# Working in this repository

This is a measurement project on dexterous hand-object tracking: DexTrack's
policies in Isaac Gym on a rented GPU, instrumented for interpenetration, on
the way to two hands on one object. Read `README.md` for where it stands,
`docs/BRIEF.md` for the goal and its stop rules, `docs/STAGE2.md` for the
evidence, and `NOTES.md` before quoting any number.

## Layout, and where new work goes

| directory | what belongs there |
|---|---|
| `dextrack/` | everything that runs or measures DexTrack in Isaac Gym: the task hook, the probe, the audit, the renderer, pod scripts. New GPU-side work goes here. |
| `src/handsim/` | the MuJoCo pipeline (GRAB → retarget → carry search → PPO → distil → two hands). Importable package; `PYTHONPATH=src:.`. |
| `experiments/tracking/` | runnable experiments on the MuJoCo pipeline; `experiments/infra/` backend benchmarks. |
| `results/` | live results with provenance; `results/dextrack_audit/` for everything from the pod. |
| `figures/` | live renders, each with its per-frame JSON manifest beside it. |
| `docs/` | `BRIEF.md`, `STAGE2.md`, `PIPELINE.md`, the DexTrack issue draft. `docs/archive/` is history. |
| `legacy/` | the retired grasp-metrics benchmark, retracted experiments, their results and figures, the attic. Read-only. Do not extend it, import from it, or copy its patterns. |
| `NOTES.md` | the dated lab log: what was run, what it showed, what was withdrawn. Append; never rewrite history. |

New tooling goes in `dextrack/` or `src/handsim/`; new experiments in
`experiments/`; new numbers in `results/` with the command that produced
them; new renders in `figures/`. Nothing goes at the repository root.

## The rules that keep the numbers honest

1. **Render before believing.** No contact number enters a document until the
   rollout behind it has been rendered and looked at.
2. **A large effect is a measurement bug until it survives a second, independent
   measurement.** The measure a policy optimises is never the measure that
   judges it (a sparse probe used as a reward was learned by the policy; the
   dense grid found the true effect was half).
3. **Replay is placement, not physics.** Numbers from a replayed trajectory are
   labelled as such.
4. **Count what you can rebuild.** A count is quoted only for poses rebuilt from
   stored offsets, rolled to the end, perturbed, and rendered.
5. **Record misses.** A run that made things worse is written up with the same
   care as one that helped.
6. **Seeds are keyed on subject and sequence**; GRAB sequence names collide
   across subjects.

## Working on the pod

Nothing runs on a paid machine that has not run at 4 environments first.
Smoke, then scale. Stop the pod when idle; report spend. Every run is a
`sed` of DexTrack's own command line (`dextrack/pod/*_cmd*.txt`), driven by
env-var-gated hooks in `dextrack/task_hook.diff`; their trainer is otherwise
untouched. Training at 1024 environments; evaluation at 16 with the exact
plane test on the dense grid; env 0 audited offline and rendered. A `DONE`
marker copied from another machine is not a result: check the log timestamp.

## Style

Python 3.10+, numpy/torch, no frameworks beyond what the pipeline needs.
Scripts take arguments and write JSON with provenance (command, commit, seed).
Docstrings say what a measurement means, not what the code does. Commit
messages state the result in numbers, one line, and end with the
co-author line. Do not add a penetration penalty, a compliant wrist, or a
static hold objective to the MuJoCo pipeline: all three are ruled out by
measurement (`docs/BRIEF.md`, "Ruled out").
