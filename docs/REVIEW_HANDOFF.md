# Review handoff for Claude

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
