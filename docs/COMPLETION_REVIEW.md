The original goal is **not achieved**. The latest work contains a reproducible result in a synthetic grasp benchmark, but the completion claim is blocked by errors in the f5d6 model and pose baseline, an unequal search budget, and the absence of demonstrated task transfer.

Reviewed on 2026-09-12 at commit `d4df676bf48fc17cfbae713d1d3aff5fc5038c6f`, following [the earlier handoff](REVIEW_HANDOFF.md). This review inspected the new code and results, recalculated the reported statistics, reran the non-GPU tests, reproduced a positive f5d6 comparison in physics, and checked the source URDF against the simulated model. Detailed numerical evidence, fitted controls, a kinematic witness, environment versions, and the URDF hash are in [COMPLETION_REVIEW_EVIDENCE.json](COMPLETION_REVIEW_EVIDENCE.json).

There has been useful progress. The new synthesis fixture freezes the unactuated robot joints that invalidated the previous f5d6 bench. The work acknowledges the open-loop replay baseline and retracts earlier interpretations of BC and chunking. The new pose comparison uses measured simulator contacts and a separate physical disturbance outcome. The Warp capacity/reset fixes and compiler serialization address concrete engineering problems. Those improvements should be retained.

The headline arithmetic checks out:

| Saved comparison | Cells | Pose passes | Wrench passes | Wrench-only / pose-only | Nominal exact paired p |
|---|---:|---:|---:|---:|---:|
| All confirmation files | 65 | 17 | 30 | 14 / 1 | 0.00097656 |
| Allegro | 26 | 17 | 25 | 9 / 1 | 0.02148438 |
| f5d6 | 39 | 0 | 5 | 5 / 0 | 0.0625 |
| Excluding overlapping pilot cells | 57 | 17 | 28 | 12 / 1 | 0.00341797 |

Here a pass means the stored `hold_N` is positive. The exact paired calculation uses a two-sided binomial test on discordant pairs, consistent with the [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html) and [SciPy binomtest documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). These are nominal calculations on the saved cells; correct arithmetic does not resolve the experimental defects below.

I also reproduced the f5d6, 35 mm cube, 50 g cell with the current default protocol: the pose condition returned `hold_N=0`, and the wrench condition returned `hold_N=0.96`, with the same epsilon, contact counts, and penetration values as the saved result. The following findings concern what that result establishes.

1. **The f5d6 success uses a different mechanism from the source robot model.** The URDF specifies five affine mimic relationships: each distal finger joint follows its proximal joint. The compiled synthesis model has `neq=0`, and [GraspScene](../src/oppdef/synth.py#L89) adds independent position actuators to all eleven finger joints. The reproduced successful grasp violates those URDF relationships substantially. For the little finger, the actual distal angle is `+0.401400` rad, while its proximal angle requires `-1.162083` rad under the URDF relationship: a difference of `1.563483` rad, approximately 89.6 degrees. The middle and ring fingers differ by 0.491 and 0.702 rad. None of the eleven finger actuators has a force limit enabled, despite effort limits in the source URDF. The present result describes the independently actuated simulated variant. It cannot establish the corresponding capability of the stated underactuated model. Both comparison conditions need the same correctly coupled model, and affected results need to be rerun.

2. **The f5d6 pose objective and opposition axis use joint origins as fingertips.** The [hand registry](../src/oppdef/embodiment.py#L66), [closure specification](../src/oppdef/hands/specs.py#L40), and [axis definition](../src/oppdef/hands/axis.py#L48) name `R_ff_l2`, `R_mf_l2`, `R_rf_l2`, `R_lf_l2`, and `R_th_l2`. [Retargeter.forward](../src/oppdef/retarget.py#L266) reads those bodies' origins. Their terminal joints rotate about those same origins, so moving the terminal joints does not move any tracked point. This was verified numerically for all five joints: the maximum tracked-point displacement was exactly zero. In the reproduced pose fit all five distal targets are zero; replacing them with the wrench condition's distal targets leaves the keypoint error exactly unchanged at `0.020700794850280736` m. Thus the baseline is insensitive to motions that change actual finger contact geometry.

   The source URDF already supplies distinct `R_*_tip` frames. For example, the index fingertip is offset `[-0.0300, 0, -0.0400]` m from `R_ff_l2`, a 50 mm difference. Using these tip locations and explicitly enforcing the URDF mimic relationships, a 12-start search found a configuration with a thumb-to-finger-mean gap of **18.11 mm**, with independent joints capped at ±1.3 rad and dependent joints within their limits. This falls below the latest experiment's **20 mm deficit cutoff**. It is a kinematic witness, not an assertion that the optimizer found the global minimum; physical grasp feasibility still requires contact and self-collision checks. The important implication is that f5d6's assignment to the deficit group depends on an incorrect fingertip definition. Repair the marker and actuation conventions before using the opposition axis to select hands or explain effects.

3. **The comparison does not have the advertised equal search budget, and its pose reference needs repair.** [The runner](../experiments/pose_vs_wrench.py#L158) gives the wrench condition `pop=a.pop*5//4`, versus `a.pop` for pose, despite the adjacent comment saying the attempt counts are equal. At the defaults that is **180 versus 144 candidate attempts**. Because the endpoint is whether search finds a passing grasp, the extra opportunities are a direct confound; their contribution has not been measured. The methods also use different parameter spaces and elite counts, so the claim that placement search is identical is too strong. Compare them with explicitly matched candidate budgets and report time as well.

   The [synthetic source](../src/oppdef/data.py#L90) describes a box with half-extents `[width/2, 0.025, 0.025]`, while the [physical comparison](../experiments/pose_vs_wrench.py#L155) builds `[width/2, width/2, width/2]`. Only the 50 mm width matches all three dimensions. In the default reference, two points declared to be contacts are 12.56 and 7.44 mm outside even the reference box. Fix the geometry and validate the source grasp. Add a pose-plus-squeeze/contact-refinement baseline with correctly observed fingertips and coupled joints. The current experiment cannot distinguish a limitation of pose fidelity from a weak or misrepresented pose baseline.

4. **The expanded sweep is not a clean independent confirmation.** [The log](../NOTES.md#L1748) describes choosing survival after four null analyses. That can motivate a legitimate new test. However, eight f5d6 cells in `conf_f5d6_0.05.json` overlap `pvw6_f5d6.json` at the same width, mass, and deterministic search seed, and every saved outcome is identical. Rerunning those cells adds no independent evidence. Excluding them still produces a favorable nominal result, as the table shows; overlap alone does not explain away the effect. The larger concern is that there are two hand designs, one CEM seed per cell, related box geometries, and no held-out demonstration sequences. Sixty-five grid cells do not establish sixty-five independent replications of an effect attributable to hand opposition. Keep the exploratory result, then freeze the corrected model, cohort definition, outcome, and analysis before evaluating fresh objects/sequences and multiple independent seeds. The standalone f5d6 paired result is also weaker than the pooled headline: five discordant successes yield nominal two-sided `p=0.0625`.

5. **The second-hand normalization does not isolate contact allocation from actuator budget.** The 1.983× ratio is reproducible, but [the figure](../experiments/fig_thesis.py#L65) divides held force by `f_total`, which [the simulator records before the disturbance](../src/oppdef/synth.py#L458). Position actuators remain active during the probe, so the normal forces can change as the object loads the fingers. Dividing by initial force does not enforce an equal total force or actuation budget throughout the test. This supports a descriptive efficiency ratio under the tested controllers; it does not establish that contact placement causes the advantage independently of extra resources. Enforce a common stated budget during the disturbance and log the achieved force/torque histories. The current files give **17 strict improvements, 5 declines, and 1 tie**, rather than the log's 18 strict improvements. Their saved per-direction arrays contain **14 force directions**, so this panel also uses the older force-only evaluation while its axis label says “wrench.”

6. **The measured outcome is substantially narrower than the original goal.** [The original goal](GOAL.md#L53) requires prediction across demonstrations and tasks, an adverse relationship between pose fidelity and task capability, and demonstrated object trajectories recovered through bimanual contact allocation. The new “wrench” search maximizes generic epsilon without consuming a demonstrated trajectory or required task-wrench sequence. It is grasp synthesis. Its result rows even copy the same input pose-fit error into every condition, rather than measuring the executed wrench condition's pose error. They cannot establish the claimed pose-error tradeoff.

   [The hold probe](../src/oppdef/synth.py#L277) tests 14 pure-force and 14 pure-torque directions, with torque normalized by object size. It does not test arbitrary combined force-and-torque directions. A positive result means every sampled direction passes at least the first 0.25 rung, each for 0.6 seconds, with final displacement under 20 mm, final rotation under 15 degrees, and a contact remaining. It does not mean survival under “ANY 6-D wrench.” A zero means some sampled direction fails the minimum tested rung; it does not establish that no grasp or no positive disturbance tolerance exists. The object is pinned during closure, and gravity is disabled. This measures disturbance resistance after assisted grasp formation. The log itself [acknowledges](../NOTES.md#L1761) that no experiment composes wrench retargeting and bimanual recovery of a demonstrated task.

The status against the original claims is therefore:

| Original claim | Status after this review |
|---|---|
| C1: keypoint error loses predictive value across human grasps as opposition falls | Unestablished; no human corpus evaluation, and the axis/markers require correction. |
| C2: epsilon/deficit predicts success across hands, objects, and tasks with one curve | Preliminary grasp-level correlation exists; the transferable curve and task threshold are unestablished. |
| C3: improving pose fidelity actively reduces task capability | Unestablished; actual executed pose errors and a controlled tradeoff are missing, and the baseline has structural defects. |
| C4: a predicted amount of second-hand assistance recovers demonstrated tasks | Unestablished; static two-hand force tests and a separate scripted task do not demonstrate this composition or predict required assistance. |

For Claude, the next work should be bounded by the following acceptance criteria:

- Correct the model first: fingertip locations must correspond to the source's actual tips; FK must respond to the joints that move those tips; mimic relationships and actuator limits must be consistent across retargeting, synthesis, and evaluation. Add focused regression checks for these observed defects. Recompute the opposition axis before defining any deficit cohort.
- Freeze one experiment with matched search opportunities, a declared physical outcome, and a consistent object/reference geometry. Include pose plus grasp refinement, generic grasp synthesis, and the proposed demonstration-conditioned method. Save seeds, candidate counts, failed-attempt reasons, selected joint targets, model/environment versions, and raw evaluation traces. Use at least the three independent seeds required by the project's own [rules](GOAL.md#L127), and hold out objects and demonstration sequences.
- For the allocation claim, enforce and measure the chosen total budget during perturbation. Label the existing 14-direction data as force-only. Derive figure counts and p-values from the selected dataset; the current figure hardcodes the left-panel p-value and discordant counts.
- Demonstrate one complete transfer: source interaction → required object motion/wrench → robot contacts → execution → independently checked task success, with the second hand added when the specified deficit calls for it. If real data is unavailable, label a procedural sequence as synthetic and keep the human-transfer claim open. Preserve null results and the original stopping criteria.
- Update the README and other headline documents when findings are retracted. The README still states that f5d6 holds nothing, despite the log retracting the experiment behind that statement. Keep the earlier logs as history, but make the current status unambiguous.

The main local minimum is continuing to enlarge or reframe a synthetic static-grasp sweep until its statistics look decisive while the model, representation, and task remain unresolved. A second is treating a model-building artifact as a property of hand anatomy. The fastest useful next step is a corrected, small experiment that could falsify the proposed mechanism.

The stack recommendation from the earlier handoff remains applicable. CPU MuJoCo is sufficient to expose and resolve these problems. The new Warp capacity and reset changes are useful, but [Warp parity](../experiments/warp_parity.py#L90) still omits the tilt condition included by the CPU expert. No GPU benchmark was rerun for this review. A migration to MJX, PufferLib, or another training framework does not address the present completion blockers; select a scaling path after the corrected experiment identifies a measured throughput bottleneck.

Verification: **56 non-GPU tests passed in 20.40 seconds** using:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  /home/jetson3/projects/dextrack_vega/.venv/bin/python \
  -m pytest -q -m 'not gpu'
```

The additional checks described above were review diagnostics, not changes to the implementation. The positive f5d6 physics reproduction can be replayed without repeating CEM using the saved evidence:

```python
import json
from oppdef.synth import GraspScene

with open("docs/COMPLETION_REVIEW_EVIDENCE.json") as stream:
    evidence = json.load(stream)

for row in evidence["f5d6_35mm_50g_reproduction"]:
    scene = GraspScene(
        row["hand"], (row["width"] / 2,) * 3,
        mass=row["mass"], n_hands=1, kp_finger=1.0,
    )
    attempt = scene.attempt(
        row["params"], do_hold=False,
        finger_target=row["finger_targets"],
    )
    held, per_direction = scene.hold_of()
    print(row["condition"], attempt.valid, held, scene.m.neq)
```

Expected under the reviewed model: pose `True, 0.0, 0`; wrench `True, 0.96, 0`. Those values are a reproduction target for the old model, not acceptance criteria for the corrected one. The existing changes to `src/oppdef/hold.py` and `results/bimanual_expert.json` were present before this review; no implementation or existing result files were intentionally edited.
