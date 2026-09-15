# Review handoff for Claude

**Latest assessment, 2026-09-15 at `7212fa2`:** real GRAB references, per-clip
PPO, and joint bimanual retargeting are meaningful progress. The mug checkpoint
reproduces its reported position tracking when its original grasp setup is
restored. Orientation tracking and generalization remain open. Read the
[current tracking review](#tracking-review-and-readme-demonstrations) before
acting on earlier audit sections; they describe historical checkpoints.

## Recommendation to Claude (2026-09-14)

Claude: continue the project with one concrete next question: **can selecting
contacts for a particular task improve successful execution over a practical
pose-plus-squeeze baseline?** The next milestone should establish a small,
trustworthy experiment. Check the findings below against the current source
before acting; the detailed audit records the reviewed commit and evidence.

Recommended order of work:

1. **Repair the task and measurement path.** Ensure task B's requested Y tilt
   reaches the commanded motion and required-wrench calculation. Pair each
   task outcome with its own margin. Score object orientation and completion
   of the requested motion, alongside position retention. Finish stable seed
   handling and analysis that accounts for repeated outcomes from one grasp.
   Record protocol corrections explicitly before collecting replacement data.
2. **Validate a small benchmark.** Use one hand, two objects, and approximately
   10–20 fixed grasps with known successes and failures. A competent scripted
   grasp must complete the task; open-hand, no-contact, and missing-tilt controls
   must fail the appropriate endpoint. Check replay, reused scenes, and task
   execution order. Save motion traces and inspect representative episodes.
3. **Separate selection quality from search quality.** Compare pose, epsilon,
   task-margin, and direct task-rollout scores on the same candidate grasps.
   Evaluate rollout-based selection on separate perturbations. Then compare
   complete pose-plus-squeeze, generic synthesis, and task-conditioned pipelines
   under a common executor and matched search budgets.
4. **Test whether task information helps.** Compare correct, swapped, and
   missing task descriptions on tasks requiring different contact choices.
   Look for changed grasp selection and improved held-out completion. Test
   actual human demonstrations before claiming a human-data contribution.
5. **Use G4 to diagnose reliability.** Distinguish exact replay failures from
   sensitivity to perturbations. Require all declared repeats when reporting
   unanimity, and report formation failures. Amend the interpretation openly:
   repeatability does not establish physical realism, and variable outcomes
   can still have predictable success probabilities.
6. **Keep the current stack through this milestone.** Retain CPU MuJoCo as the
   reference. Revisit acceleration or RL tooling after profiling a validated
   workload. Make the next research decision from held-out task outcomes and
   uncertainty, with a practical improvement threshold declared in advance.

For the handoff back to the user, report **what you fixed, what remains open,
what evidence supports each claim, and the next decision**. Include runnable
checks, saved episode provenance, failure counts, and task-outcome comparisons.
Label outcomes as supported, ruled out within the tested scope, or inconclusive.
Keep withdrawn results clearly identified in the current README and goal summary.

The pattern to avoid is promoting an unexpected result into a new thesis before
checking that the experiment performs and measures its declared task. The
success criterion for this milestone is a reproducible task comparison whose
interpretation survives those checks. See the
[detailed recommendation and diagnostics](#recommendation-after-gravity-fix).

## Review history

Prepared 2026-09-12 from a repository review and a follow-up comparison of the technology stack. The user requested an assessment of the methodology, project direction, possible local minima, and whether alternatives such as PufferLib would help.

**Main assessment.** The research question is worth pursuing, but the main bottleneck is experimental validity. The largest risk is interpreting limitations of a scene, controller, or search procedure as intrinsic limitations of a robot hand. The strongest direction is to estimate which task forces and torques an embodiment cannot reliably supply, then allocate the minimum additional assistance needed. That question remains useful even if f5d6 can sometimes grasp and Ferrari–Canny epsilon does not yield a universal success curve.

**Stack recommendation.** Keep CPU MuJoCo as the reference, NumPy/SciPy for analysis, and PyTorch for the current learning experiments. Retain direct MuJoCo Warp as the first GPU scaling candidate. Make additional MJX-JAX work conditional on a specific need. Evaluate PufferLib only if profiling identifies a training/runtime bottleneck that justifies its integration cost. Physics fidelity, environment batching, and policy training are separate decisions.

**Scope and evidence.** The reviewed commit was `ac865234e05a623ee37b5f3fb3331d5fbcc0cd46`, with an active dirty working tree. Uncommitted work included `src/oppdef/synth.py`, `experiments/deficit_repair.py`, four `results/deficit_*.json` files, and a modification to `results/bimanual_expert.json`. Four synthesis processes were writing results during the review. Treat those files as unfinished experiments, and re-check source locations and observations against the current checkout before acting. This document records review findings and recommendations; it does not replace the measurement and retraction history in [NOTES.md](../NOTES.md).

The review inspected code, plans, recent commits, saved results, installed package metadata, and primary external sources. It also ran the non-GPU test suite and two focused CPU diagnostics. No GPU benchmark or PufferLib installation was performed for this handoff. The initial review changed no project files; this follow-up creates the handoff document.

The intended direction in [GOAL.md](GOAL.md) is to explain when human pose fidelity fails to preserve manipulation, then recover the demonstrated object interaction by reallocating contacts across hands. The implementation currently has four tracks:

| Track | Methods in use | Evidence currently supported |
|---|---|---|
| Hand characterization | Multi-start L-BFGS-B joint search, thumb-to-finger-mean gap, derived closure trajectory, gravity holds | Behavior of particular models under a particular closure procedure; intrinsic capability limits are not established |
| Retargeting | Synthetic fingertip references; keypoint, epsilon, and blended objectives | Optimization machinery; the main physically realizable grasp comparison was retracted |
| Active grasp synthesis | Cross-entropy search over orientation, standoff, and closure; simulated closure and force disturbances; one versus two hands | Preliminary conditional grasp stability, without demonstration transfer |
| Execution and scaling | Scripted expert, MPPI, behavior cloning, action chunking, CPU/MJX/Warp interfaces | Useful infrastructure, with limited evidence of feedback adaptation |

The strongest methodological habits are analytic checks, explicit controls, multiple training seeds, and visible retractions. Preserve them. The recurring weakness is that invalid trials can become scientific findings before the relevant checks are enforced. Passing software tests establishes the invariants those tests cover; it does not establish that an experiment measures the intended phenomenon.

The six main research local minima are:

1. **Treating a failed controller or search as a physical impossibility.**

   The existing f5d6 hold benchmark imports the whole Vega model and adds finger actuators while leaving its arm joints unactuated. In a diagnostic of the 5 cm box trial, the palm moved **1.05336 m during closure**, leaving zero object contacts. The initial palm position was approximately `[0.67575, -0.22946, 1.11012]`; after closure it was `[-0.14942, -0.20560, 0.45582]`. All seven right-arm joints had no actuators. The initial state also contained hand-object penetration and `clear_start=False`.

   The saved f5d6 rows already show `clear_start=False`, zero grip force, and approximately 19.64 m of free fall for every attempted width. This is evidence of an invalid fixture, not evidence that opposition alone explains failure. See [hand_bench.py](../src/oppdef/envs/hand_bench.py), especially `build`, `open_until_clear`, and `trial`, and [hand_axis_f5d6.json](../results/hand_axis_f5d6.json).

   More broadly, the thumb-to-finger-mean distance uses body reference points and does not encode surface extent, contact normals, self-collision, or the full range of alternative grasps. A nonzero floor does not prove that larger objects cannot be grasped. A finite multi-start search is also not an impossibility certificate. The hold bench follows one derived joint trajectory, so it measures the interaction between that controller and the hand.

   Establish a fixed-wrist, correctly actuated reference for every hand. Reject invalid setups before reporting hold failures. Compare multiple feasible grasp strategies before attributing a boundary to morphology. The new synthesis code already freezes non-hand joints; that correction needs to inform the interpretation and remeasurement of the older headline result.

2. **Replacing keypoint error with another overextended proxy.**

   Ferrari–Canny epsilon measures worst-direction resistance in a normalized six-dimensional force-and-torque space under a particular contact-force model. Holding against gravity is a narrower requirement. The saved LEAP 6 cm cell has `eps=0` and `held=True`; the 11 cm cell also holds despite a reconstructed `delta` of approximately 1.83. See [hand_axis.json](../results/hand_axis.json) and [grasp_metrics](../src/oppdef/metrics/epsilon.py).

   These examples do not invalidate epsilon as a geometric grasp descriptor. They undermine treating `delta=1` as an already established universal boundary for these physical outcomes. Multiplying epsilon by the measured sum of normal forces also does not model the complete set of forces that the actuators can redistribute across contacts.

   The new hold validation applies fourteen translational force directions and checks translation and remaining contact. It does not apply pure torques or gate on orientation retention. Therefore its minimum force is not a direct experimental measurement of the full six-dimensional epsilon ball. See [hold.py](../src/oppdef/hold.py) and `GraspScene.hold_of` in [synth.py](../src/oppdef/synth.py).

   Prefer task-specific wrench feasibility or margin under stated actuation and friction assumptions. Include gravity-only, torque-sensitive, and environment-supported cases where appropriate. Validate predictions on held-out physical outcomes rather than using the optimized score as its own validation. Task wrench coverage and limitations of contact-only force models have substantial prior art; [Grasp planning to maximize task coverage](https://rpal.cse.usf.edu/publications/ijrr2015b.pdf) is a relevant starting point.

3. **Assuming simulated closure makes every synthesized grasp physically realizable.**

   Closing in MuJoCo is an improvement over the retracted point-fingertip surrogate. It still requires validation of contact compliance, actuation, and initialization. The partial f5d6 results inspected during the review included a single-hand 3 cm grasp with `hold_N=1.0` and **4.66 mm penetration**, and a two-hand counterpart with `hold_N=2.0` and **10.61 mm penetration**. See [deficit_f5d6.json](../results/deficit_f5d6.json). These are provisional outputs, not established counterexamples to a physical hardware claim.

   `GraspScene.attempt` checks initial object penetration, but its final acceptance records penetration without rejecting on that quantity. Its displacement gate is not a penetration gate. The object is also pinned during closure and then released before measurement. That can support a conditional grasp-stability experiment, but acquisition from an unassisted scene remains a separate question.

   Validate acceptable penetration against geometry scale, solver settings, and force limits. Check actuation and coupling against the intended embodiment. Record peak as well as final penetration and forces. Keep externally assisted grasp formation distinct from unassisted acquisition. A second simulator is not necessary to begin this work, and CPU MuJoCo itself remains a model rather than hardware ground truth.

4. **Improving learned policies on a task that can be solved by replay.**

   A review diagnostic recorded the action sequence from expert seed 1000, then replayed it using only elapsed time from the existing policy-evaluation reset. It used no state feedback for action selection and no retraining. It succeeded on all three tested evaluation seeds:

   | Evaluation seed | Peg extraction | Base lift | Success |
   |---|---:|---:|---|
   | 9000 | 13.511 cm | 0.429 cm | True |
   | 9001 | 13.517 cm | 0.420 cm | True |
   | 9002 | 13.455 cm | 0.453 cm | True |

   This is a small missing-baseline diagnostic, not a complete benchmark. It shows that success on those conditions does not establish useful feedback control. The evaluation still supplies its existing calibrated pre-grasp initialization; this is not an unassisted perception-to-action result.

   `rollout_expert` randomizes friction and mass but does not implement the XY-placement randomization described in its module introduction. See [learning/bc.py](../src/oppdef/learning/bc.py). The recorded action-chunking null is consistent with a task already saturated by a one-step MLP, but is not evidence against action chunking generally.

   Add trajectory replay and a simple feedback controller as baselines. Before architecture expansion, vary pose/geometry or apply disturbances that require corrective actions. Keep training-seed uncertainty separate from variation across evaluation episodes. Preserve the existing honest reporting of the three-seed spread.

5. **Drifting from retargeting into generic grasp synthesis.**

   The active [deficit_repair.py](../experiments/deficit_repair.py) searches grasps without human data and maximizes epsilon. It is useful supporting work, but it does not test whether a demonstrated interaction transfers better under a new representation. Real demonstration readers remain unfinished in [data.py](../src/oppdef/data.py).

   Two hands outperforming one can result from a larger contact or actuator budget. That alone does not show that a deficit metric predicted the required assistance, or that a retargeting method recovered a demonstrated task. The peg expert likewise tests stabilization and extraction under a specified control protocol; its one-handed control does not exhaust every possible one-hand strategy.

   Return the central comparison to the same demonstration, object task, executor, and compute budget. Compare pose-plus-squeeze, an established contact-aware baseline, and the proposed method. Report resource budgets and test whether any advantage survives appropriate force-budget controls. Measure when assistance is needed and how much is needed. Distinguish extra hand assistance from support supplied by a fixture or the environment when that distinction matters to the task.

6. **Building infrastructure around an insufficiently distinctive contribution.**

   A primary-source literature spot-check found that [DexMachina](https://project-dexmachina.github.io/) already studies bimanual functional retargeting through object-state tracking and contact guidance across different robot hands. The recent [C2Dex preprint](https://arxiv.org/abs/2608.07045) explicitly uses object-side contacts as transfer targets across embodiments. This is not a complete novelty review, but it is enough to require a more specific contribution than object-centric retargeting alone.

   [DexGraspBench](https://github.com/JYChen18/DexGraspBench/) already provides multi-hand evaluation with simulation success, force-closure metrics, penetration, and contact quality. The project's own NOTES had already identified it. Reuse or cross-check that machinery before rebuilding another full benchmark.

   A potentially distinctive contribution is predicting missing task capability across embodiments and allocating minimal assistance, with evidence that the prediction transfers to held-out hands or objects. More policies, backends, or hand registrations do not themselves earn that claim.

The stack decision should follow the workload. **MJX and PufferLib occupy different layers.** MJX supplies accelerated MuJoCo physics through a JAX API; PufferLib supplies an RL training/runtime ecosystem and its own environment conventions. Choosing a trainer does not replace the need for a contact simulator. PufferLib's current 4.0 documentation describes native CUDA training, a PyTorch fallback, C environments, and OpenMP rollout execution with buffered transfers. It also explicitly notes the removal of the older Python/third-party integration path. Do not assume a tutorial for an older Gym wrapper applies to 4.0. See the [current PufferLib documentation](https://puffer.ai/docs.html).

My recommended choices for this repository are:

| Component | Recommendation | Reason and decision trigger |
|---|---|---|
| Reference physics and debugging | CPU MuJoCo | Existing scenes, real contact extraction, short experiments, and diagnostics already work here; preserve the exact evaluation model |
| Contact metrics and search | NumPy/SciPy plus existing CEM/MPPI tools | No policy learner is needed for most immediate claims; profile closure rollouts and convex-hull evaluation separately |
| Current policy fitting | PyTorch | The BC implementation already uses it; a JAX migration has no demonstrated research benefit |
| NVIDIA GPU simulation | Direct MuJoCo Warp, after fixing its adapter | Best first candidate given the working local path and current PyTorch code; validate full task outcomes and resource use |
| JAX-based physics | MJX-JAX only for a specific benefit | Reconsider for a JAX-native executor, a supported non-NVIDIA target, or validated simulator gradients; the present search and BC do not require simulator differentiation |
| Packaged GPU RL environment framework | Evaluate mjlab when RL is justified | A closer architectural fit for MuJoCo Warp plus PyTorch than building an entire RL environment lifecycle locally |
| PufferLib | Optional measured integration experiment | Consider if a stable task needs substantially faster collection/training and the environment binding pays for itself |

There is an important naming distinction: current MJX supports both **MJX-JAX** and **MJX-Warp**. The latter uses MuJoCo Warp behind the JAX API. The repo's `MjxVec` currently selects the default implementation without an explicit `impl`, while `WarpVec` uses Warp directly. Upstream guidance favors the Warp implementation for many NVIDIA contact-heavy workloads; MJX-JAX has different scaling characteristics and supports differentiation that MJX-Warp currently lacks. See the [MJX implementations documentation](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-implementations). Record the actual implementation in every benchmark rather than using the label “MJX” ambiguously.

For a future GPU RL framework, [mjlab](https://github.com/mujocolab/mjlab) provides MuJoCo Warp environment infrastructure and native MuJoCo access; MuJoCo's own documentation identifies it as a PyTorch training route. [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) is the corresponding ecosystem to inspect for a JAX/MJX executor. These are candidates to evaluate, not dependencies installed or validated on this Jetson during the review. Neither framework is required to complete the immediate scientific comparison.

PufferLib's headline throughput should not be projected onto dexterous mesh-contact physics. Its native runtime also changes more than sampling speed: its documented learning algorithm and default model differ from a plain PPO/MLP baseline. Hold those choices fixed where possible when attributing an improvement to infrastructure. On this specific machine, integration also needs an architecture check: the inspected 4.0 [build script](https://github.com/PufferAI/PufferLib/blob/4.0/build.sh) selects a Linux `amd64` raylib archive without distinguishing aarch64. That is a concrete setup risk, not proof that PufferLib cannot run on Jetson. A migration could simply exchange one native-dependency problem for another.

Several current implementation issues should be resolved before changing frameworks:

1. **The installed environments are not aligned.** Package metadata was read directly, without importing or initializing GPU libraries:

   | Package | Existing CPU environment | `.venv-mjx` |
   |---|---|---|
   | Python / architecture | 3.10.12 / aarch64 | 3.10.12 / aarch64 |
   | MuJoCo | 3.8.0 | 3.13.0 |
   | mujoco-mjx | Not installed | 3.9.0 |
   | mujoco-warp | Not installed | 3.13.0 |
   | warp-lang | Not installed | 1.16.0 |
   | JAX / jaxlib | Not installed | 0.6.2 / 0.6.2 |
   | PyTorch | 2.7.1a0+gite2d141d | 2.7.1a0+gite2d141d |
   | NumPy / SciPy | 2.2.6 / 1.15.3 | 2.2.6 / 1.15.3 |
   | PufferLib | Not installed | Not installed |

   The CPU interpreter used was `/home/jetson3/projects/dextrack_vega/.venv/bin/python`. These version differences do not by themselves prove incompatibility. They mean comparing an old CPU result with a GPU result may mix package-version and backend changes. Select and record a tested compatible package set, and compare CPU and GPU under the same MuJoCo version where supported. Preserve existing environments while testing replacements.

   The current Menagerie commit was `8161bba264d7fa7c99ca301e91e7fb44737676ad`. Record asset revisions, scene hashes, package versions, compiler settings, and any local upstream patches alongside results. [pyproject.toml](../pyproject.toml) currently provides broad version constraints rather than a reproducible experiment lock.

2. **The Warp adapter multiplies per-world capacities by the world count.** In [vec.py](../src/oppdef/vec.py), `WarpVec.__init__` passes `nconmax=nconmax * N` and `njmax=njmax * N`. The installed Warp `put_data` source defines both arguments per world; `naconmax` is the separate total-contact argument. Upstream documents the same [batch-size semantics](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#batch-sizes).

   With defaults and `N=256`, the wrapper requests a total contact capacity of `256 * 256 * 256 = 16,777,216`, rather than `256 * 256 = 65,536`. Constraint storage likewise grows with an extra factor of N. This can make GPU scaling look dramatically worse than necessary. This finding is established by the call and allocation semantics; its runtime impact was not benchmarked here. Fix the capacity units and verify overflow handling and memory growth before comparing backends. `reset` also hardcodes capacities instead of preserving constructor settings.

3. **The Warp reset drops heterogeneous initial states.** `WarpVec.reset` flattens a supplied state and uses only the first world's qpos/qvel to initialize every world. This differs from the CPU backend's per-world reset contract and undermines independent initial-state randomization. Add backend-specific tests with distinct initial states and independent resets. The existing [vector tests](../tests/test_vec.py) exercise the CPU path, so their success does not validate this Warp behavior.

4. **GPU state is copied to NumPy on every step.** `MjxVec.step/state` passes through NumPy, and `WarpVec.step/state` constructs a new device action array and copies qpos/qvel back with `.numpy()` each step. This introduces synchronization and conversions, even on a machine with shared physical memory. Those calls are visible in [vec.py](../src/oppdef/vec.py); their cost needs profiling.

   For a GPU training loop, keep state, observations, actions, and rewards on the device and export only diagnostics at deliberate intervals. Use the appropriate Warp/PyTorch interoperability rather than routing everything through NumPy. The official [MuJoCo Warp learning-framework documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#learning-frameworks) confirms both PyTorch and JAX integration paths. Keep a simple CPU adapter for analysis; one interface need not force identical memory representations on every backend.

5. **The batching layer is not integrated into the active synthesis path.** `deficit_repair.cem` calls `GraspScene.attempt` sequentially for each candidate. BC collection likewise runs sequential expert rollouts. The new `CpuVec`/`WarpVec` wrappers are primarily used by the vector benchmark and parity code. MPPI separately uses `mujoco.rollout` already. Therefore a faster vector microbenchmark will not automatically accelerate the experiments currently producing results.

   Profile model construction, repeated closure optimization, candidate simulation, epsilon computation, disturbance testing, and Python overhead. Cache reusable models/closures, then batch independent candidates where the pin/release lifecycle can be preserved. CPU MuJoCo already exposes threaded rollout execution in its [Python rollout interface](https://mujoco.readthedocs.io/en/stable/python.html#rollout). A trainer migration is unnecessary to exploit that.

6. **Parity is narrower than its headline.** The current [warp_parity.py](../experiments/warp_parity.py) imports `warp_fix` as a top-level module, although the packaged implementation is `oppdef.sim.warp_fix`. Its Warp success calculation also omits the tilt clause used by the CPU expert. These are source-inspection findings; the GPU command was not rerun during this handoff.

   Use the same complete outcome definition on both backends, with an independent audit of the raw trajectory. Validate contact-rich rollouts including successes, failures, and conditions near the task boundary. Agreement on the scripted expert and one control is useful but does not establish agreement for every learned policy or grasp. Preserve collision geometry and actuation during performance comparisons; parameter changes need a recorded model-accuracy experiment.

7. **MJX availability is a local package/driver finding, not a universal platform conclusion.** NOTES records working JAX GPU matrix multiplication but cuSolver failures for linear algebra, making the tested MJX-JAX path unusable. Direct Warp was the working GPU route in that record. Do not repeat the earlier claim that all JAX is unavailable on Jetson, and do not assume a successful matrix multiply validates MJX stepping. Check a minimal actual physics step before a full scene. A new MJX-Warp configuration is a separate candidate, not a demonstrated fix for this environment.

A framework comparison should measure **time to a valid scientific result**. The NOTES numbers of roughly 17,022 CPU environment-steps/s and 2,655 Warp environment-steps/s came from different experiments; they do not establish a controlled speed ratio. Benchmark one representative contact-rich scene and protocol, with the same controls, control frequency, physics substeps, batch sizes, solver settings, and output requirements. Separate cold compilation from warmed execution. Report memory, accepted valid grasps per minute, completed disturbance tests per minute, and full collection/inference/update time when training is involved. A benchmark that requests all state on the CPU each step answers a different question from a device-resident training loop.

PufferLib becomes worth a bounded trial when the task is stable, a policy needs many interactions, and the measured rollout/training overhead is material. Compare the complete pipeline against the existing implementation at matched task quality and learning budget. Include integration and maintenance cost in the decision. Retain an independently executable MuJoCo evaluation path whichever trainer is selected.

The recommended next work for Claude is ordered below. These are proposed deliverables for the user to adopt, not claims that implementation has already happened:

1. **Re-establish the hand-capability reference.** Fix the f5d6 fixture and actuation semantics, enforce valid initial states, and measure palm stability, contact, and forces throughout closure. Re-run the affected cells into a separate result artifact. Completion means the old zero-hold finding is either replaced or re-earned with a functioning setup and a documented search scope.
2. **Make the physical predictor comparison valid.** State whether the target is gravity retention, force disturbance tolerance, full wrench resistance, or a demonstrated task. Implement the corresponding trajectory-level outcome checks, valid contact tolerances, and force-budget controls. Completion means epsilon is evaluated against a physically independent outcome, including counterexamples and held-out cells.
3. **Run a small demonstration-transfer comparison.** Use real demonstration data, pose-plus-squeeze, an established contact-aware method, and the proposed method, with the same executor. Include replay and simple feedback baselines where learning is evaluated. Completion means a reproducible task-outcome effect, or a clearly reported null, rather than a higher optimized score.
4. **Repair the narrow stack defects and record a reproducible environment.** Address capacity units, per-world reset behavior, parity import/criterion consistency, and version provenance. Add targeted tests for these actual failure modes. Avoid a broad framework rewrite as a prerequisite to the first three deliverables.
5. **Profile, then choose one scaling path.** Start with the existing CPU path; compare corrected direct Warp only if compute is limiting. If substantial RL becomes necessary, evaluate mjlab or a bounded PufferLib integration against the chosen workload. Revisit MJX-JAX when its benefits are specifically needed. Completion means a measured improvement in end-to-end valid experiment throughput while retaining the same scientific outcome.

The project should not require proving that f5d6 is universally incapable of grasping, that pose fidelity is always harmful, or that all hands share one exact scalar threshold. A narrower transferable prediction of task capability and needed assistance would be a stronger result than a broader claim supported by fragile scenes.

**Verification details.** The initial review ran:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  /home/jetson3/projects/dextrack_vega/.venv/bin/python \
  -m pytest -q -m 'not gpu'
```

Result: **56 passed in 46.76 s**. The default system interpreter did not have `oppdef` installed, so the working CPU environment and explicit `PYTHONPATH` were used. Passing this suite does not cover the Warp-specific findings above. This documentation-only follow-up did not repeat the physics suite.

The replay diagnostic can be reproduced from the repository root with the same CPU interpreter:

```python
from oppdef.learning.bc import rollout_expert, run_policy

_, actions, source_result, _ = rollout_expert(1000, record=True)
assert source_result["success"]

def replay(observation):
    # Read only episode phase; no joint, object, or contact feedback.
    index = min(int(round(float(observation[-3]) * len(actions))),
                len(actions) - 1)
    return actions[index]

for seed in (9000, 9001, 9002):
    print(seed, run_policy(seed, act_fn=replay, n_steps=len(actions)))
```

For the f5d6 diagnostic, build `HandBench` at width 0.05 m and mass 0.05 kg using `derive_flex`, call `plan` and `open_until_clear`, and measure the palm before and after the same pinned closure sequence used by `trial`. Check the actuator map for the right-arm joints and count object contacts afterwards.

That diagnostic initially exposed a separate concurrency issue: the external URDF compiler uses shared filenames `_dextrack_stripped.urdf` and `_dextrack_raw.xml` beside the original asset. A concurrent compile failed while another process was using them. The successful diagnostic compiled an isolated temporary URDF with absolute mesh paths, applying the same GLB stripping, and supplied its MJCF only within the diagnostic process. Make compiler temporaries unique before parallelizing f5d6 scene construction. Do not disturb active result writers to reproduce this review.

All external stack documentation linked here was checked on 2026-09-12. Backend support and APIs are version-dependent. The literature references establish relevant prior work; they are not a comprehensive novelty assessment or an assertion that the proposed narrower contribution has already been published.

## Follow-up audit after the claimed resolution

Reviewed 2026-09-13 at commit `30df0fe` (following the axis repair in `8cc4b63`). The short verdict is: **the implementation repairs several concrete defects from the completion review, but the original project goal is still not achieved.** The new experiment earns a narrower synthetic result about fixed-budget grasp search. It does not establish opposition deficit, human-interaction retargeting, the claimed anti-correlation, or bimanual repair of a demonstrated task.

### What was actually fixed

The response handled the most serious model errors candidly:

- The old f5d6 results and the deficit cohort were retracted. The README now says that the previous opposition ordering was an artifact.
- The f5d6 compiler repair restores five mimic constraints. A fresh `GraspScene` has `neq=5`, six independently actuated finger joints, and no actuator on the five dependent joints.
- The six independent f5d6 actuators now have force limits: ±1.0 N·m on the two independent thumb joints and ±0.5 N·m on the four independent finger joints.
- Fingertip measurement was moved from distal-body origins to points derived from collision geometry. Regression tests now cover the observed f5d6 tip, coupling, and effort-limit failures.
- `SyntheticSource` now describes the same cube used by the physical experiment and places the analytic reference points on its faces.
- The replacement comparison in [matched.py](../experiments/matched.py) gives both conditions the same parameter space, 144 attempted candidates, and the same three random seeds for each hand/width.
- The replacement log uses the accurate phrase “survives the probe” and explicitly withdraws the claim that the sampled pure-force/pure-torque test represents every possible six-dimensional wrench.
- The follow-up commit adds `hands/model.py`, rewires the public `hands/axis.py` command to the corrected tip/coupling/self-collision implementation, and adds public-entry-point regression tests.
- The matched result files now include provenance, arguments, selected placement/finger targets, per-direction values, rejection counts, mass/gain, and elapsed time. `experiments/fig_matched.py` is checked in and derives its plotted discordance values from the saved rows.

I reran the complete non-GPU suite after these changes: **61 tests passed in 52.14 seconds**. I also inspected a newly built f5d6 scene and confirmed the five equalities, six actuators, and their force limits directly.

### What the replacement result supports

The saved arithmetic is correct. Across four hand models, five cube widths, and three CEM seeds, the stored results contain 60 paired search runs:

| Outcome | Pose objective | Wrench objective | Paired result |
|---|---:|---:|---|
| Mean fingertip-graph error | 7.96 cm | 12.64 cm | Pose is more faithful |
| Mean epsilon | 0.0544 | 0.2124 | Wrench wins, but this is its optimized score |
| Mean sampled-probe threshold | 0.025 N | 0.223 N | 19 wrench wins, 1 pose win, 40 ties |
| Positive sampled-probe threshold | 3/60 | 21/60 | 18 wrench-only, 0 pose-only |

The nominal exact discordant-pair calculation is `p=7.63e-6`, and the stored magnitudes reproduce the reported Wilcoxon value of approximately `0.000183`. The effect is also visible at stricter ladder thresholds: at 0.49 N the counts are 2 pose versus 14 wrench, and at 0.96 N they are 0 versus 6.

The defensible claim is:

> In this synthetic, assisted-formation cube benchmark, CEM guided by contact epsilon finds grasps that survive the sampled disturbance probe more often than the same CEM guided by fingertip-shape fidelity, at 144 candidates.

That is useful evidence that the optimization objective affects search efficiency. It is not yet a result about retargeting human interaction or opposition-deficient hands.

### What remains unresolved

1. **The pose condition is not the baseline proposed in the review or used by a normal retargeting pipeline.** Both conditions search every finger angle freely, and both discard candidates with fewer than two contacts. The pose condition then selects the contact-producing candidate with the smallest synthetic fingertip-graph error. This is a constrained grasp-synthesis objective. It is not keypoint retargeting followed by squeeze/contact refinement, and the log acknowledges that baseline is still owed. Because only 3/60 pose searches survive even the smallest 0.25 N rung and Shadow is 0/15 under both methods, the comparison mostly distinguishes which score guides a small stochastic search through a sparse feasible set.

2. **The new objective and evaluation remain closely aligned.** Epsilon rewards worst-direction capacity of the measured contact wrench set. The evaluation applies pure forces and pure torques to the same simulated contact model. The dynamic rollout is more informative than scoring epsilon twice, but it is not independent in the strong scientific sense of a held-out task, object family, physical system, or demonstrated wrench trajectory. Friction, geometry, contact placement, and actuator assumptions are shared.

3. **The 60 “cells” are not 60 diverse task instances.** They are 20 hand/width configurations, each repeated with three optimizer seeds. The synthetic reference is one analytic grasp family scaled with cube width; its random jitter is a common translation of every fingertip and cancels from the inter-fingertip objective. Seed-level variation is relevant to an algorithm-success estimand, but the nominal cell-level test should not be presented as evidence of transfer across demonstrations, shapes, or tasks. Report seed-stratified results as well: the survival discordances are 4/0, 6/0, and 8/0 for seeds 0, 1, and 2, with nominal two-sided exact p-values 0.125, 0.03125, and 0.0078125.

4. **The canonical axis and provenance blockers are now largely fixed, but the metric itself remains a weak proxy.** [hands/axis.py](../src/oppdef/hands/axis.py) now shares `hands/model.py`, uses collision-derived fingertip points, applies mimic relationships, scopes self-collision, records provenance, and is covered by public-entry-point tests. The matched rows also carry the selected parameters and per-direction traces. The remaining scientific issue is conceptual: nearest fingertip distance says that two points can approach each other; it does not say that the resulting contact normals, reachable object placement, friction, and torque limits can supply a particular task wrench. Keep the axis as a geometric descriptor, and add a task-wrench capability measure before calling it an “opposition deficit.”

   The checked-in `results/opposition_axis.json` has not been regenerated after this repair: its provenance still names commit `247a62f`, while the public-axis fix is `8cc4b63`. Run `make axis` from a clean tree and review the output before quoting that artifact.

5. **Replayability is now substantially fixed, with provenance still needing one clean rerun.** Each `results/matched_*.json` row now saves selected parameters, finger targets, per-direction values, rejection counts, timing, and run arguments, and the file-level metadata records commit, package versions, machine, and dirty state. The recorded runs have `dirty=true` because the working tree contains an unrelated pre-existing `results/bimanual_expert.json` modification. Their metadata also says commit `247a62f`, while the writer/provenance changes are in the later `30df0fe` commit. For a paper-quality artifact, rerun from a clean commit and add source asset/model hashes so an external reviewer can confirm the exact physical model bytes.

6. **The live matched figure generator is now fixed.** `experiments/fig_matched.py` computes the plotted discordances from the input rows, and the old hardcoded generator was moved to `experiments/retracted_fig_thesis.py`. The retracted figure is quarantined. This item is resolved, subject to keeping the retracted path clearly out of release instructions.

7. **The second-hand causal claim remains unfixed.** The new commits did not rerun the two-hand comparison with a controlled force/torque budget during disturbance. The old 1.983× ratio is still a descriptive normalization by pre-disturbance contact force, based on 14 force directions. It does not establish that contact allocation, rather than resources or controller response, causes the gain. The old strict count remains 17 improvements, 5 declines, and 1 tie.

8. **The original scientific endpoints remain open.** There is no real human corpus, no held-out demonstrated object trajectories, no task-specific required wrench sequence, no evidence that lower pose error harms task success, no transferable `delta=1` curve, and no experiment in which deficit prediction triggers a measured amount of second-hand assistance that recovers the same demonstrated task. The corrected axis currently removes the project's defining independent variable: these four hands do not span an opposition-deficit range under the revised metric.

### Recommended handoff to Claude

Treat `30df0fe` as a useful recovery checkpoint, not completion.

1. Validate the consolidated embodiment metric and separate it from task capability. The stale public path is repaired and tested. Next, choose an opposition measure tied to task wrench capability; nearest fingertip distance alone can say a hand “opposes” even when the resulting contact normals, reachable object placement, and force limits cannot support the task.
2. Build the missing practical baseline: keypoint/contact-state retargeting followed by a fixed, budgeted squeeze or contact refinement. Compare this with wrench-conditioned refinement and generic epsilon grasp synthesis. Keep candidate counts and executor identical, and report valid candidates per budget as a diagnostic.
3. Finish provenance for every selected grasp: input reference ID, object geometry, seed, full search settings, model/asset hashes, placement, joint targets, settled state, contacts, forces, per-direction traces, and rejection reasons. The current artifacts cover most of these, but not source asset hashes or a clean working tree.
4. Freeze a held-out evaluation before looking at outcomes. Use multiple reference grasps and object shapes, group inference by held-out demonstration/object rather than treating optimizer seeds as new task instances, and include sensitivity to the 0.25 N pass threshold.
5. Run one complete task chain: demonstration → required object wrench/trajectory → retargeted contacts → matched executor → task outcome. Then add the second hand under an enforced total resource budget. This is the experiment that can reconnect the useful objective-search signal to C3/C4.
6. Keep CPU MuJoCo as the reference while these questions are unresolved. MJX, Warp, or PufferLib may reduce runtime later, but current failures concern model semantics, baselines, provenance, and experimental design.

The local minimum has shifted. Earlier, the risk was explaining simulator defects as hand anatomy. Now it is stopping at a statistically strong comparison between two synthetic search scores and calling that the representation thesis. The new experiment is a solid ablation for a future paper; it is not yet the paper's central result.

## Follow-up audit after G3 and G4 pre-registration

Reviewed 2026-09-14 at local `61a06df` (`G3 RESULT` at `2d7947d`; origin is
still at that G3 result). This is progress toward an honest benchmark, but it
does not close the original goal. G3 directly tests task success rather than
the static probe, and its pooled effect sizes are small: epsilon has AUC 0.544
and Spearman rho +0.074, while the task-conditioned margin has rho +0.004.
The result plausibly explains why the static G1 win did not transfer to G2's
carry task. It remains simulation-only, one object size and mass, two related
carry variants, and f5d6 contributes only four grasps.

### What changed

The placement repair is now exercised across all four hands and four shapes;
Shadow is no longer structurally empty. G3 samples grasps before measuring
metrics, which is the right direction for a predictor study because an
optimizer would remove the quality variation being tested. The pre-registration
declares the metric set, Holm correction, task outcome, and null decision rule.
G4 then pre-registers a useful go/no-go question: whether the same saved grasp
has a stable task outcome under small mass and placement perturbations. That is
the right next gate before interpreting a metric null.

### G3 analysis caveats that must be fixed or explicitly re-analyzed

1. `experiments/g3.py` seeds each hand/shape cell with
   `a.seed + hash(hk + shape) % 9973`. Python's `hash()` is salted per process,
   so a rerun with the same command and recorded seed can sample different
   grasps. Replace it with a stable mapping (for example an explicit cell index
   or a cryptographic digest) and rerun the saved artifacts. The current
   `dirty=false` provenance does not make the data reproducible while this
   remains in the protocol.
2. The analysis duplicates every grasp's metric once for task A and once for
   task B, then applies ordinary Spearman p-values and AUC calculations. The
   two outcomes share the same grasp and are therefore clustered. The point
   estimates are still descriptive, but inferential p-values and Holm decisions
   should be recomputed with a grasp-level paired permutation, cluster bootstrap,
   or an equivalent repeated-measures model. The same dependence issue applies
   to confidence intervals for G4's pair-level unanimity statistic.
3. The G3 conclusion should be phrased as “these seven metrics did not predict
   these two carry outcomes in this sampled simulation,” not as a general
   failure of grasp metrics. The shape breakdown already shows why: epsilon is
   positive on spheres and slightly negative on capsules. Treat geometry/task
   interaction as a hypothesis for the next experiment.

### G4 status

`docs/G4_PREREGISTRATION.md` is committed and has a clear stopping rule, but
the experiment has not been run. `experiments/g4.py` is currently untracked,
so there is no committed implementation or result to review yet. Before running
it, verify that the determinism repeat is compared against the exact saved G3
outcome, that perturbations are applied after grasp formation as intended, and
that failed/rejected repeats are counted in the declared denominator rather
than silently removed from unanimity. Report cluster-aware intervals by grasp,
hand, and task. If the lower unanimity bound fails the pre-registered gate,
withdraw the G3 metric headline and repair the simulator/contact model before
another metric sweep.

### Current recommendation

The project is now headed toward a defensible simulator/metric benchmark, not
yet a demonstration-transfer or opposition-deficit paper. First make G3
reproducible and reanalyze it at the grasp level; then run G4 and honor its
go/no-go rule. If G4 passes, build a held-out task/object evaluation and a
real retargeting baseline. If it fails, stop tuning epsilon and repair the
contact/task simulator. The MJX-versus-PufferLib decision is still downstream
of this: neither framework resolves a target that is not predictive or an
outcome that is not reproducible. Keep the current CPU MuJoCo path as the
reference until those gates pass.

### Verification on this checkout

`python -m py_compile experiments/g4.py` and `git diff --check` pass. The full
non-GPU suite currently reports **66 passed, 1 failed** in 87.6 seconds. The
failure is `tests/test_claims.py::test_opposition_floor_separates_f5d6_from_the_others`,
which imports the removed `oppdef.hands.axis.extremes` function. That test still
asserts the retracted f5d6 opposition claim, so it should be rewritten or
removed alongside the claim rather than restoring the old API. Targeted axis
and model tests remain the relevant passing coverage; the repository should not
be called green until the stale test is resolved.

## Recommendation after gravity fix

Reviewed 2026-09-14 at `07c0f52`. My recommendation is to continue with a
smaller, falsifiable question: **does selecting contacts for a particular task
improve completion of that task over a practical pose-plus-squeeze baseline?**
Keep demonstration transfer and additional-hand assistance as hypotheses to
test after the executor and outcome measure work. The project has useful
infrastructure and increasingly effective checks; its scientific direction
still depends on establishing a trustworthy experiment.

The latest commit restores gravity after `run_task`, including early returns,
and moves the affected G3 data and figure into the retracted directories.
Finding this through G4's replay check is evidence that the check was useful.
It also supersedes my earlier interpretation of G3's AUC and correlation
numbers. Replacement `results/g3_*.json` files are being collected locally;
their partial contents are not a completed result.

### Immediate issues before another interpretation of G3

- **Task B's tilt is discarded.** `experiments/g3.py:29` puts its Y rotation in
  `traj.cmd[:, 4]`, but `object_path` and `base_command` in
  `src/oppdef/task.py:89` and `:110` use only column 3, the X rotation. A direct
  diagnostic on this checkout produces requested Y rotation **1.0472 rad**,
  emitted base rotation **0 rad**, object-path rotation **0 rad**, and required
  torque **0**. Task B currently exercises a faster translation, not the
  registered Y-tilt task. G4 imports the same helper, so repeating it cannot
  detect this specification error. Fix the full path from task specification
  through commands and evaluation, and record the correction before collection.
- **The task-specific predictor is paired with the wrong outcome.**
  `experiments/g3.py:96` saves both `margin` for A and `margin_b` for B, but
  `experiments/g3_analysis.py:73` duplicates A's `margin` and `margin_per_N` for
  both outcomes. Use B's margin, divided by the matching force for its normalized
  version, for B observations in both pooled and subgroup analyses. The earlier
  statement that the margin contains no information is not established by this
  analysis.
- **The scored outcome is position retention, with no orientation criterion.**
  `src/oppdef/task.py:173` measures object position in the palm frame. The success
  check at `:234`–`:250` tests that displacement and total carried distance; it
  never checks object orientation or completion of the tilt. Retention is a
  useful diagnostic, but the manipulation endpoint also needs the intended
  object translation, orientation, and phase completion. Measure controller
  tracking separately so a base-control failure is not attributed to a grasp.

These are code-path findings, not guesses about why a particular grasp failed.
They are reasons to validate a small bank of episodes before repeating a full
hand/shape sweep. The stable sampling-seed and clustered-inference issues in
the previous audit also remain. One clarification to that audit: Python's
salted `hash()` prevents exact resampling from the recorded seed; saved grasp
parameters can still support replay once simulation state is correctly restored.

### Make G4 a diagnostic, not a verdict on physics

I would revise my earlier endorsement of G4's decision rule. Identical-state
replay, sensitivity to physical perturbations, and physical realism are three
different questions. A simulation can reproduce a wrong task perfectly. A
real grasp near a slip boundary can be sensitive to a millimetre of placement
change. Neither low unanimity nor high unanimity alone identifies a simulator
defect or validates a benchmark critique.

Also, a metric can predict a *probability* of success even when individual
outcomes vary. For illustration, a grasp with independent success probability
0.8 has five-repeat unanimity probability `0.8^5 + 0.2^5 = 0.328`. That does
not make its 80% success probability unpredictable. Preserve the registered
unanimity statistic as a sensitivity diagnostic, but amend its interpretation
explicitly. Estimate success rates under the declared perturbations, with
uncertainty, and distinguish exact replay failures from physical sensitivity.
Five repeats give only a coarse estimate per grasp.

The current untracked G4 implementation also removes `None` repeats before
calling a pair unanimous and continues after a failed determinism check.
Require all five valid repeats for the registered statistic; report formation
failures separately and stop interpretation if exact replay fails. A 1 mm
object displacement after closure changes contact geometry, so inspect induced
penetration instead of describing the contact state as unchanged. MuJoCo uses
a soft-contact model, and penetration by itself does not establish invalid
physics. [MuJoCo contact-model documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#soft-contact-model).

The task rollout also uses the same MuJoCo scene and contact solver as the
static probe. It has a different outcome measure; the claims in the G3 protocol
and NOTES that it does not share their contact model should be corrected.

### A bounded next milestone

1. **Validate one complete task.** Start with one hand, two object shapes, and a
   fixed bank of roughly 10–20 grasps spanning known successes and failures.
   Demonstrate that a scripted competent grasp completes the requested motion,
   no-contact and open-hand controls fail, and an object carried without the
   required tilt fails the manipulation endpoint. Check exact replay, fresh
   versus reused scenes, and A/B execution order. Save command, hand, and object
   pose traces and inspect representative episodes. This is a validation bank,
   not the final held-out evaluation.
2. **Isolate selection from search.** On the same candidate bank, compare pose
   score, epsilon, task margin, and a more expensive score from direct task
   rollouts. Use the rollout score as a diagnostic upper comparator on separate
   evaluation perturbations; do not select and evaluate on the same trials. If
   successful candidates exist but a cheap score ranks them poorly, there is a
   concrete predictor problem. If every candidate fails, investigate formation
   and execution before drawing conclusions about metrics. Then compare full
   search pipelines with the same executor and budget.
3. **Test whether the task input changes a useful decision.** Use tasks that
   require meaningfully different grasps. Compare the correct task description,
   a swapped description, and a generic objective while holding the candidate
   set and executor fixed. If correct information changes selection and improves
   held-out task completion, task conditioning earns its place. Claim a benefit
   from human demonstrations only after testing actual demonstrations and an
   appropriate missing/shuffled-input control. Analytic trajectories establish
   a task-specification experiment.
4. **Freeze the evaluation and decision before scaling.** Define the smallest
   success-rate improvement worth the added cost, group repeated trials by
   grasp/object, and report effect sizes with uncertainty. Hold out objects and
   task trajectories, not frames. Failure to reach statistical significance
   does not establish equivalence. Continue a method when its held-out gain
   exceeds the declared practical threshold with adequate evidence; narrow or
   stop that method when a valid, sufficiently precise experiment rules the
   gain out. Treat wide intervals as inconclusive.

I would time-box the validation milestone to about one working week, then
review the evidence before starting a larger research phase. That is a
planning recommendation, not an estimate that all tasks can be completed in a
week. If validation is still failing, continue focused simulator repair and
defer broad method claims and architecture expansion.

### Direction and stack

The local minimum to avoid is **finding an unexpected null, turning it into a
new headline, and discovering a protocol defect in the next experiment**.
Pre-registration and transparent retraction help, but neither replaces
validating that the code implements the registered task. Use fewer experiments
with stronger acceptance checks. Update the README and current-goal summary
with the active claim and its status; retain old claims as labelled history.

Keep CPU MuJoCo as the reference for this milestone. MJX supplies accelerated
MuJoCo simulation; PufferLib supplies reinforcement-learning tooling and its
own environment ecosystem. Choosing a trainer does not repair a task command
or success criterion. My recommendation is to benchmark an acceleration or
training change only after a valid workload demonstrates a throughput problem,
using time to a completed, validated experiment as the measure.
[MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html),
[PufferLib repository](https://github.com/PufferAI/PufferLib).

The promising contribution is a measured improvement in completing a
manipulation task, with a clear account of what task information made that
improvement possible. Additional-hand allocation becomes a strong extension
once the one-hand limit is demonstrated under a controlled resource budget.
That direction remains worth investigating without assuming the original
opposition-deficit story will return.

Verification for this addendum: the two new gravity/isolation regression tests
pass (**2 passed in 48.97 s**). The direct task-helper diagnostic produced the
rotation and torque values above, and `git diff --check` passed. This follow-up
did not repeat the full suite or run the G3/G4 experiments. It updates the review
document; source and actively written result files were not changed.

## Tracking review and README demonstrations

Reviewed 2026-09-15 at `7212fa2`. I support the current direction as a
demonstration-driven simulation workbench. It now has actual human-motion
input, a measurable manipulation trajectory, and a saved neural controller
that can be independently replayed. Those address major gaps in the earlier
synthetic-grasp project. This is useful engineering progress; evidence for a
new generalizable method still needs a separate comparison.

The user requested new GIFs and a README update if the work supported them.
`experiments/render_tracking.py` captures actual MuJoCo dynamics, then renders
the captured states in a separate `MjData`. The object has a free joint, gravity
is active, and it is not welded to a hand. Floating wrists are driven through
mocap constraints and fingers through actuators. These are assisted-start,
per-reference simulation examples. The new GIFs and adjacent JSON files in
`figures/physics_*` contain measured position and orientation errors, contact
counts, source/model hashes, and exact search/checkpoint settings. Display
changes do not alter the captured physics.

### What the replay establishes

The saved `ppo_mug_drink_1_h160.pt` checkpoint reproduces **29.3509 mm** mean
position error and **111/111** frames below 50 mm. Its original hold-scored
grasp synthesis, seed 0, and start frame 5 are necessary parts of the protocol.
Loading the weights on the unrefined retarget instead gave 1620.7 mm. Restoring
the original setup also reproduced the feedforward baseline's 8197.3 mm error.
The renderer's recorded errors match `rl.evaluate` exactly from the same reset.

The new bowl example uses joint fitting and wrist search scored over all 131
frames, seed 0, 10 samples and two rounds. It measures **18.0 mm** mean position
error, with **131/131** frames below 50 mm. Its independently reset rollout
matches the capture exactly. This is a fresh result under the saved settings,
not a reproduction of the older 28.1 mm headline. The bimanual controller is
searched feedforward, not a learned two-hand policy. Each remaining GIF carries
its own complete measured values in the adjacent manifest.

### What remains open

1. **Position tracking is not full pose tracking.** The mug and bowl rollouts
   average **44.8°** and **45.9°** of orientation error respectively, despite
   their small position errors. The camera is more severe: **34.4 mm** mean
   position error alongside **138.4°** mean orientation error. The binoculars
   example measures **32.6 mm** and **20.0°**. `BimanualTracker.rollout` returns a
   zero-filled `rot_err`; the renderer computes the actual quaternion error
   independently. Include true orientation error in the evaluation API and
   define task-appropriate angular tolerances, including object symmetries.
   For a handled mug, orientation changes are functionally meaningful.
2. **Per-reference fitting is not a held-out task result.** These grasp offsets
   are optimized on the reference being shown. PPO is trained for its own
   reference and reads simulator state and future reference poses. Keep that
   scope explicit; demonstrate recovery from perturbations and compare with
   action replay before attributing robustness to feedback. A matched one-hand
   baseline is still needed to establish the benefit or necessity of two hands.
3. **Distillation remains incomplete.** Position errors of 158–456 mm alone do
   not establish that the object stayed held. Record contact histories, relative
   motion, orientation, and task completion on held-out objects. The larger G9
   run was still active during this review; its partial output is not a final
   evaluation and was not used to label these GIFs.
4. **Persist the full trained system.** The mug weights were saved without the
   grasp/configuration needed to use them. `g9_ppo_distill.py` prints results but
   does not persist each policy and its fitted environment. Save checkpoints,
   grasp offsets, initialization, horizon, splits, seeds, model hashes, and
   evaluation traces together before scaling the experiment further.

My next recommendation to Claude is to make position-and-orientation task
completion reproducible from a saved artifact, then test perturbation recovery
and held-out objects. Retain the current CPU reference until that comparison
is established. Keep the README gallery as illustrations of measured runs,
with aggregate success claims supported by a separately defined evaluation.

Verification: the full non-GPU suite at the reviewed checkout reports
**68 passed, 1 failed** (152.52 s). The existing failure is
`tests/test_retarget.py::test_every_hand_builds_with_a_floating_base`:
`f5d6_left` is registered but absent from `hands/specs.py`, producing a
`KeyError`. It is unrelated to the Shadow demonstrations. Rendering and
documentation do not repair that missing closure specification; it remains
an outstanding implementation issue.
