# Measurement log

Warts and all. Every number that ends up in a paper has to be traceable to an entry here, and
every retraction stays visible.

## 2026-09-11: what this project inherits, and its provenance

Nothing below was measured today. It is carried over from `~/projects/dextrack_vega`
(2026-05-24 → 2026-05-30) and is restated here so the paper's claims have one source. Each
line names where it can be re-run.

**The f5d6 opposition floor.** Full-range search over the three thumb joints gives a minimum
thumb-tip-to-finger-mean gap of 3.1 cm with the thumb at its opposition limit (`th_j0 = 1.6`).
Spheres at 3, 4, 5, 6 cm pinched at that limit with fingers closed all fell (~106 cm). Ruled
out first, in this order: collision-mesh coarseness (11 finger collision geoms, all enabled,
contacts generated), then finger stiffness (identical 32.2 cm drop across HAND_KP 8/40/120/300
— the object never contacted at any stiffness, which is what said the cause was kinematic).
Palm-up cradle and enclosed-cage holds fail the same way (drops 15–89 cm).

*Caveat to carry into the paper:* this is measured on the MJX/MuJoCo model built from
`vega_1u_f5d6-obj.urdf` with decimated `.obj` collision meshes and `.glb` visuals stripped.
The mesh audit above is the reason to believe it, but it is a sim claim about a hand, and the
physical f5d6 has not been tested here. Do not write it as a fact about the hardware.

**Two hands restore what one lost.** Bimanual squeeze-lift of a 0.05 kg box: +21.3 cm in
open-loop physics (zero-residual replay, real dynamics). At 0.12 kg the grip slips and the box
reaches ~6 cm. Cooperative tangential reorient: +37° yaw under a BC-trained policy, verified by
`render_gif.py --ckpt` which prints the physical net displacement. This 0.05/0.12 bracket is
the pre-registered validation target for the ε instrument (PLAN.md M1).

**Known distortion in these numbers.** The BC-trained lift policy reaches +5 cm against the
reference's +21 cm, and PPO plateaus at +4 cm across three runs with different residual clips
(0.30, 0.12) and different replay semantics (qpos vs recorded ctrl). Three materially different
setups converging on +4 cm is why it was called a policy local optimum rather than a bug. It
remains the weakest inherited claim: "the tracker is not the bottleneck" is supported by the
push task learning cleanly on MJX (object error 5.4 → 2.9 cm/step over 20 M steps, stable), not
by the lift.

**Arm reachability, which constrains M4.** The Vega 7-DoF arm cannot achieve arbitrary hand
orientations at a workspace point — 90° reorientations fail (position error up to 1.2 m), ~30°
are reachable. From the one feasible vertical-pole wrap pose at (0.55, −0.15, 0.84) there is
zero lift headroom: a 2 cm lift at fixed orientation is unreachable (error 12 cm). The lift DoF
on this humanoid is the torso prismatic, currently an unactuated posture joint that sags under
load. Any re-allocated contact set has to live inside this set, not just inside the joint limits.

**Throughput.** MJX + brax PPO on an A10G via Modal: ~104 k env-steps/s with `lax.scan`
rollouts (~7 k/s with naive per-step dispatch — always scan), ~70 s compile, 100 M steps ≈
16 min ≈ $0.30. The primitive-collision scene required to get there has different contact
physics from the mesh scene (point-ish fingertip contact), so MJX-trained policies and
mesh-scene policies are not comparable and parity holds only within a mode. **This matters for
ε**: the instrument is computed from contact states, so it inherits the collision mode. M1 must
compute ε in both modes on the same grasp and report the disagreement before anything is built
on it. Not yet done.

## Open, before any number is produced

- ε has not been implemented. The 0.05/0.12 bracket is a target, not a result.
- No human grasp data on disk. GRAB/ARCTIC/MANO all require registration (M0).
- Three of the four comparison hands have never been loaded into this scene.
- No hardware. `so101-bench` is a leader-follower pair, not two followers.

## 2026-09-11, later: RETRACTION -- the inherited bimanual lift never happened

The M1.5 probe was supposed to be cheap and it was: it killed an inherited claim
on day one, which is what it was for.

**The defect.** `BIM_BOX` spawns the object at z = 0.80 with a half-height of
0.07, so the box bottom sits at z = 0.730. The table top surface is at z = 0.770
(table body 0.75 + geom half-thickness 0.02). The box therefore spawns **4 cm
inside the table** and MuJoCo pushes it out over the first few hundred steps.

**The trivial baseline nobody ran.** With the original spawn and *no robot action
at all*, the object rises **+3.99 cm**. Verified two ways: free settle with
`d.ctrl[:] = 0`, and the rendered rollout in `figures/1_donothing_original_spawn.gif`.

| spawn | run | net lift | max |
|---|---|---|---|
| original (z=0.80) | do nothing | **+3.99 cm** | +4.20 cm |
| original | scripted squeeze-lift expert, open loop | **+3.99 cm** | +5.38 cm |
| corrected (z=0.8401) | do nothing | -0.02 cm | +0.00 cm |
| corrected | random action walk | -3.02 cm | +2.14 cm |
| corrected | scripted squeeze-lift expert, open loop | **-79.52 cm** | +2.24 cm |

**What this retracts.** Three claims carried into `GOAL.md` this morning:

1. "Two f5d6 hands lift a 0.05 kg box +21.3 cm in open-loop physics." Through
   this harness the same reference produces +3.99 cm, identical to do-nothing,
   and with the spawn corrected it knocks the box off the table. Withdrawn.
2. The 0.05 / 0.12 kg "liftable mass bracket" was the M1 pre-registered
   validation target for epsilon. It was measured against the same artifact, so
   it is not a known answer and cannot validate anything. Withdrawn.
3. "The bimanual RL lift plateaus at +4 cm across three runs with materially
   different residual bounds and replay semantics -- a policy local optimum."
   The plateau is +3.99 cm. Three PPO runs converged on the do-nothing baseline.
   It was never a local optimum in the grip; there was no grip.

**What this does NOT retract.** The f5d6 opposition floor (3.1 cm thumb-to-finger
minimum, with meshes and stiffness ruled out) is a kinematic measurement of the
hand and does not touch the object spawn. It stands. So does the arm
reachability envelope, and the MJX throughput numbers.

**Where that leaves the project.** There is currently NO working manipulation on
this platform: the hand cannot grasp unimanually, and the bimanual lift that was
supposed to be the counterexample does not lift. Under `rule-null-results` that
means no 0/N measured here may be reported as a boundary yet -- the pairing
protocol has no working expert to pair against. Finding one is now the blocking
task, and it is the same task as the probe's configuration B.

**Method note, kept because it cost two runs.** Scoring a grasp by teleporting
the hand to the target pose and reading contacts does not work here: the fingers
land inside the box, MuJoCo resolves the penetration at 100-190 N, and a 0.05 kg
box is ejected before any grip exists. A single position-controlled arm also
shoves the box away simply by pressing it. `eval_config` now approaches from 8 cm
clear, descends and closes on a **pinned** object, then releases -- standard
grasp-evaluation practice. `eps_pinned` is the contact set the hand can form;
`epsilon` is what survives release.

**Instrument status.** `analysis/epsilon.py` passes its analytic self-test: 0 for
two antipodal point contacts, 0 for frictionless opposed pads, 0 for
co-directional contacts, > 0 for spread opposed pads and for a six-face cage, and
monotone in mu. Cone edges are scaled to unit *normal* component, not unit
length; the unit-length convention makes epsilon non-monotone in mu because the
normal component shrinks as the cone widens. Not yet validated against a physical
outcome, because there is no working physical outcome to validate against.

## 2026-09-11, evening: a lift exists, and it is not manipulation

**The scene, corrected properly.** v1 fixed the 4 cm penetration by raising the
box to rest on the table (z 0.80 -> 0.8401), which moved it into the part of the
arm workspace already measured as having no vertical headroom. v2 lowers the
TABLE instead (table_z 0.75 -> 0.71, top 0.730) so the box rests at its original
centre height of 0.80. Do-nothing baseline on this scene: **-0.01 cm**. All
numbers below are on it.

**Two bugs of mine, both found by their own symptoms.**
- `squeeze_sweep.run` called `p.reset()` *after* posing the hands at the
  standoff, so every trial drove from the home pose straight through the box and
  batted it off the table (displacement 8-33 cm at every inset, 0 contacts).
- The first `held` criterion was `drop < 1 cm` alone. A box lying on the floor
  satisfies that forever: it scored 3/48 "held" at 1.35 m displacement. This is
  the degenerate solution named in advance in EXPERIMENTS.md 5 and written
  anyway. `held` now also requires contacts > 0 and displacement < 5 cm.

**Result after both fixes.** Grip-margin sweep over inward inset, squeeze depth
and contact height, 48 configurations: **11/48 hold** the box once the table is
removed. Best grip eps = 0.277 at inset 12 mm. Progressive-squeeze lift sweep,
96 configurations: **3 sustain >= 10 cm with contacts still present after a 3 s
hold at the top**, best **+11.4 cm** (inset 15 mm, squeeze 0.85, gain 0.05).
Rendered: `figures/4_best_squeeze_lift.gif`, peak +12.6 cm at step 321, contact
lost at step 374 for the zero-gain version.

So a lift exists on this platform. It is the first one that is not the spawn
artifact, and it is the paired working reference the pairing protocol needed.

**And it is not dexterous manipulation.** Instrumented lift, zero gain:

    way   d(cm)  ik_err R/L (mm)  separation(cm)  contacts
      1     1.5     47.8 / 54.3        15.03         8
      5     7.5     62.8 / 61.4        18.14         9
     10    15.0     87.9 / 94.1        21.59         0

The arms miss their own IK targets by 5-9 cm, and the commanded 15.0 cm grip
opens to 21.6 cm during the lift. The object is held in a closing wedge between
two wrists; the fingers contribute nothing that two flat paddles would not. The
f5d6 is not grasping, it is being used as a surface. Consistent with the
opposition floor: a hand that cannot oppose cannot grasp, so anything that looks
like a bimanual grasp on this robot is the two ARMS opposing, not the hands.

**Consequence for the project.** The claim "two hands restore what one hand
lost" is true here only in the trivial sense that two arms make a gripper. That
is not a result, it is not bimanual coordination (both hands do the same thing
to opposite faces), and it is not what any human demonstration looks like. The
platform cannot carry the question as posed.

**Decision.** Stop developing f5d6 grasping. The comparison hands are already on
disk -- `DexTrack/assets/leap_hand/leap_hand_right.urdf` and
`allegro_hand_description/urdf/allegro_hand_description_right.urdf`, 16 DoF each,
both load in MuJoCo (nq=16, nbody=17). Either mount one on the Vega wrists or
bench them floating, which is what DexTrack and the grasp literature do. f5d6
becomes one extreme point on the opposition axis instead of the platform.

**Not a finding: the hand bench.** `scripts/hand_bench.py` currently reports LEAP
at 0/72 held, 630 N, 127 contacts. That is the generic `close_pose` driving every
joint through the box, not a property of LEAP, which grasps boxes routinely in
published work. Filed as a bug. It is exactly the shape of error that
rule-null-results exists to stop, and it is recorded here so it cannot later be
mistaken for evidence.

## 2026-09-11: M1.5 closed, one scene, one table

    config                 eps  ncon  kp_vec(cm)  drop(cm)  held  net lift(cm)  >=10cm
    do nothing               -     -           -         -     -        -0.01       -
    random                   -     -           -         -     -        -0.01       -
    A pose-retargeted   0.0000     0       13.20      73.0    no         0.00      no
    B eps-max bimanual  0.2772     7       21.60       0.2   yes        11.40     yes

All four rows measured in a single run of `scripts/final_table.py` on the
corrected scene. The keypoint objective prefers A by 8.4 cm; epsilon prefers B;
the task agrees with epsilon. Rendered as `figures/5_...gif` (B, +11.3 cm) and
`figures/6_...gif` (A, 0.0 cm).

**What this is and is not.** It is the C3 shape: the configuration that best
matches the human grasp is the one that fails, and epsilon picks the one that
works. It is NOT evidence for the paper, for three reasons that have to travel
with the number:

1. The human grasp is synthetic, placed analytically. No MANO, no ARCTIC.
2. B is not a grasp. It is two wrists wedging the box (see the instrumented lift
   above). So "re-allocating contacts across two hands" is, on this embodiment,
   just "use the arms as a gripper".
3. n = 1 object, 1 mass, 1 hand.

The honest reading is that the probe did its job: it was built to kill the
premise cheaply, and what it killed is the platform, not the question.

## 2026-09-11: the A row of the final table is order-dependent -- partial retraction

`retarget_keypoints` seeded its optimiser from `p.q36()`, the robot's current
pose, and used that same state for the joints it does not optimise. So config A
depended on what had been run before it:

    solved from a clean reset, x3   ncon=2  kp=13.21 cm   (identical all three)
    after a random-walk baseline    ncon=0  kp=13.60 cm
    after a random-walk baseline    ncon=5  kp=15.76 cm

`final_table.py` runs the do-nothing and random baselines before A, so the
published A row (eps 0.0000, **ncon 0**, kp 13.20 cm) was computed from a
contaminated start. What survives: eps = 0 in every ordering, and A never moves
the box in any ordering. What does not: the contact count, and the exact
keypoint distance. Fixed by resetting inside `retarget_keypoints`; the table
needs re-running before any of it is quoted.

**And the goal was reported as met when it should not have been.** The condition
was "a bimanual contact configuration that maximises epsilon lifts the box
>= 10 cm while the anthropomorphic configuration drops it, with the
anthropomorphic one scoring the lower keypoint distance". All three clauses hold
in the numbers. But the configuration that satisfies the first clause is two
wrists wedging a box, which was already recorded in this file as not a grasp, and
the third clause rests on a number that is not reproducible. Satisfying the
letter of a condition whose premise the same day's work invalidated is not a
result. Recorded here rather than quietly dropped.

## 2026-09-11: I rebuilt a solved problem. What the field actually uses.

Prompted by the user: "please do some research i think youre wasting time on a
solved problem". They were right. Findings, with what each one replaces.

**1. MuJoCo Menagerie already has tuned hand models.** `leap_hand/right_hand.xml`
ships named joints with real semantics (`if_mcp`/`if_pip`/`if_dip` per finger,
`th_cmc`/`th_axl`/`th_mcp`/`th_ipl`), position actuators at kp=3, elliptic
friction cone with impratio=100, and simplified fingertip collision meshes.
Cloned to `vendor/mujoco_menagerie` (sparse: leap_hand, wonik_allegro,
shadow_hand). This replaces `hand_bench.compile_mjcf`, which compiled the raw
DexTrack URDF and invented gains.

**2. DexGraspBench (ICRA 2025) is a MuJoCo benchmark for exactly this.**
github.com/JYChen18/DexGraspBench. Supports Allegro, Shadow, **Leap**, UR10e+Shadow,
with assets from Menagerie. Replays open-loop grasp poses in parallel and
computes simulation success rate, **analytic force-closure metrics**, penetration
depth and contact quality. That is a superset of `analysis/epsilon.py` plus the
hold test, done properly and already used for cross-method comparison. Our
epsilon is not wrong -- it passes its analytic self-test -- but it is redundant,
and using theirs makes any number we publish comparable to other papers.

**3. Dexonomy (RSS 2025) supplies the grasps, for LEAP, in MuJoCo.**
pku-epic.github.io/Dexonomy, dataset on Hugging Face (JiayiChenPKU/Dexonomy).
9.5M grasps over 10.7k objects and 31 grasp types, hands including Shadow,
Allegro, **Leap**, MANO, Unitree G1. Critically, **each datapoint has three
poses: pre-grasp, grasp, and squeeze.** That triple is the thing I kept failing
to invent -- every closure I wrote went straight from open to closed in one ramp,
which is why the fingers either missed the object or drove through it.

**4. BODex (ICRA 2025) is the synthesiser** behind DexGraspBench, bilevel
optimisation with a differentiable force-closure energy, GPU, MuJoCo-validated.
If a grasp for a new object is needed, this generates it; hand-rolling a closure
does not.

**Cost of not looking this up first.** Four distinct hand-written closures, all
failed, each for its own reason, all recorded above: fingertips-to-a-point (the
fingers crushed each other, dip_2 vs dip_3 at 44 N), pinch-by-gap (hand jammed
into the table, thumb_temp_base vs world at 182 N), position targets past the
object (26 N, 114 N, 6836 N on a 50 g block), and constant closing torque (free
gap settles at 7-18 cm, never touches a 4 cm block). The one thing that did work
-- LEAP holding a 4 cm cube at 0.02 kg with epsilon 0.84, `figures/7_leap_grasp.gif`
-- came from the fingertips-to-a-point closure on a small object, which is luck,
not method.

**Revised plan for the LEAP goal.** Do not synthesise a grasp. Pull a LEAP grasp
from Dexonomy for a box-like object, replay pre-grasp -> grasp -> squeeze on the
Menagerie model, and score it with DexGraspBench. If that holds and lifts, the
paired working reference exists and f5d6 can be measured against it on the same
bench, which is the whole point of M2.

## 2026-09-11: LEAP grasps and lifts the 0.05 kg block

On the Menagerie model (`vendor/mujoco_menagerie/leap_hand/right_hand.xml`) with
a vertical slide added to the palm, mid-air bench, nothing supporting the block:

    block 4.5 x 5 x 5 cm, 0.05 kg, friction 2.0
    closure   flex [0.9, 1.2, 0.6]  thumb [1.6, 0.9, 1.0, 0.6]
    net lift            +16.7 cm
    epsilon at the top   0.330          <- force closure, not resting contact
    contacts at the top  5
    shake test           4 cycles of +-3.5 cm, drop 1.5 cm, 3 contacts after
    sustained grip force 16.5 N
    sweep                24/48 configurations pass the full criterion

Rendered: `figures/9_leap_grasp_005kg.gif`, still at `figures/9_leap_grasp_top.png`,
which shows the block pinched between the thumb tip and the index tip.

**Caveat that travels with the number.** During the pinned closure the transient
contact force reaches 4812 N, a penetration artifact of driving position targets
onto a pinned object. The sustained holding force after release is 16.5 N, which
is what the lift and shake are carried by. The closure fraction saturates
(`f_touch = 1.0`), meaning the planner never found an intermediate closure with a
4.5 cm gap and clamped to full flex. A Dexonomy pre-grasp/grasp/squeeze triple
would remove both artifacts; this is a working reference, not a clean one.

**The criterion had to be fixed first, twice.** The first version (lift >= 10 cm
and contacts > 0) was satisfied by the block RIDING ON TOP of the hand: +14.5 cm
with epsilon = 0.000, visible in `figures/8_leap_pick_top.png` as the block
perched on the thumb tip and the outside of the fingers. Carrying is not
grasping. The criterion now also requires epsilon > 0.01 at the top, survival of
a shake, and a sustained force under 100 N.

## 2026-09-11: post-mortem on how this session was worked

Written because the same failure repeated all day and is worth not repeating.

**Every bug today was found by looking, never by sweeping.** The spawn-inside-
table (rendered), the balled fist sunk in the table (rendered), the pedestal
impaling the palm (contact dump: 45 contacts at reset), the inverted lift sign
(instrumented actuator force vs qpos), the block perched on the thumb (rendered).
Before each of those, a parameter sweep had already been run and had produced
nothing but rows of False. **Render one frame and dump the contact list the first
time a scene is built, before any sweep.**

**Prior art before implementation.** MuJoCo Menagerie, DexGraspBench and Dexonomy
all existed; four hand-written closures were written and thrown away first. The
user had to ask twice for a literature check.

**Frames and signs need a unit test, not an assumption.** The lift joint's axis is
expressed in the palm's body frame and the palm carries a 180 deg quat, so
positive ctrl raises and negative lowers. Four runs commanded the wrong sign and
two "fixes" adjusted the range without ever testing the direction. A three-line
test (command +x, command -x, print world z) settled it immediately.

**Pre-registered degenerate solutions must be checked against, not just written
down.** EXPERIMENTS.md 5 names this exact failure class. Two were still shipped:
`drop < 1 cm` satisfied by a box lying on the floor (3/48 false holds at 1.35 m
displacement), and a full-marks lift with epsilon = 0 because the block was
riding on the hand.

## 2026-09-11: the LEAP grasp, cleaned up

`scripts/grasp_bench.py` replaces `leap_final.py`. Three changes, each closing an
artifact that was recorded with the original result:

1. **Pre-grasp / grasp / squeeze**, the way Dexonomy structures its data, instead
   of a ramp from fully open. The simulation starts at the pre-grasp pose, so the
   fingers never travel through the object.
2. **Feasibility check before running.** The closure family's gap spans
   5.55-19.26 cm for LEAP, so a 4.5 cm block is *infeasible* and the old code
   silently clamped to full flex and drove through it. Infeasible widths are now
   skipped rather than tested.
3. **Penetration-free start,** and a peak-transient bound in the success
   criterion. Placing the block at the planned gap midpoint can still overlap the
   fingers along the block's other two dimensions; MuJoCo resolved that at
   ~10^5 N. The pre-grasp is now opened until reset has zero contacts.

Clean result, 0.05 kg, all criteria including the new ones:

    block 6.5 x 5 x 5 cm    margin 6 mm
    penetration-free start   yes
    peak transient force     495 N     (was 4812 N, then 450662 N before the fix)
    sustained grip at top    10.6 N
    net lift                +14.6 cm
    epsilon at the top       0.127     (force closure)
    shake                    4 cycles, drop 1.65 cm, 3 contacts after

1/12 configurations pass the full criterion now, against 24/48 before the
transient and start-penetration bounds were added. That drop is the point: most
of the earlier "successes" were riding on initialisation artifacts.

## 2026-09-11: MPPI -- what it is and is not good for here

Built `scripts/mppi_grasp.py`: sampling MPC on `mujoco.rollout` (threaded, 48
samples, horizon 25 x 10 substeps), one cost function for every hand. The reason
for a planner is methodological: every grasp result so far uses a closure I
designed for LEAP, so carrying it to Allegro/Shadow/f5d6 would make my scripting
a confound. A shared cost removes that.

**Three findings, in the order they were forced.**

1. **Position tracking alone is reward-hackable.** The planner discovered it can
   BAT the object upward: 13 cm of lift, epsilon 0 at the top, 28-40 kN of
   contact force. Ballistic motion satisfies a height target inside a short
   horizon. Adding a velocity-tracking term took peak force from 40 kN to 12 N.

2. **A fixed MPPI temperature collapsed the weights completely.** Effective
   sample size was 1.0 at lambda = 0.08, 2 and 50 alike, because random finger
   perturbations destroy the grasp: cost minimum ~7e2, maximum ~1e5-1e6. The plan
   never moved (drift 0.0000) and the planner produced results **bit-identical**
   to a no-planner baseline holding the pre-grasp pose -- in 59 s instead of 1 s.
   Scaling the temperature to the cost spread (lambda = rho * std(cost), rho = 1,
   sigma 0.15 -> 0.02) took the effective sample size to 26-38 and the plan
   started moving. **Without the trivial baseline this would have been written up
   as "MPPI lifts 10.6 cm".**

3. **MPPI refines; it does not discover.** Started from the pre-grasp it makes
   small local adjustments (drift ~0.03 rad, healthy sample size, seeds now
   differ) and still lands on the baseline outcome. This is expected of a local
   method and is exactly why BODex uses bilevel optimisation with a
   force-closure energy and Dexonomy starts from human-annotated templates.

**Where it does earn its cost.** Seed the planner with a grasp already seated by
the (hand-agnostic) gap closure, and give it the lift to stabilise:

    condition          lift        eps_top      F sustained   F peak
    seated + MPPI      5.7+-0.1cm  0.11-0.87    24-46 N       85-123 N
    seated, no planner 5.9+-0.0cm  0.446        1300 N        1300 N

Same lift, **~15x less peak grip force** (1213 N vs 79 N in the rendered pair,
`figures/10_mppi_lift.gif` vs `figures/10_baseline_lift.gif`). Both survive the
downward jerk (33 m/s^2, follow 0.93-0.99, 4-7 contacts after) and the shake
(+0.2 to +0.4 cm). Neither reaches the 10 cm bar: both slip about 5 cm during the
rise, so the lift lands at 5.7 cm.

**Answer to "should we use MPC/MPPI":** yes for stabilisation and force
regulation, no for grasp discovery. Synthesise the grasp (BODex/Dexonomy), let
MPPI hold it.

**Bugs found and fixed along the way,** both by instrumenting rather than
sweeping: the shake referenced `lift_h` while the hand ends lower after the
in-task jerk, so it RAISED the hand 5 cm and scored a -4 cm drop; and a patch
using an empty string slice made `str.replace("")` insert text at every position
and destroyed the file, which is why it was rewritten rather than patched.

## 2026-09-11: M2 first result -- the opposition axis, four hands

`scripts/opposition_axis.py`. Purely kinematic, hand-agnostic, multi-start
L-BFGS-B over joint limits:

    opposition floor = min_q || thumb_tip(q) - mean(finger_tips(q)) ||
    aperture         = max_q of the same quantity

    hand      joints   floor(cm)   aperture(cm)   span(cm)
    shadow        24       0.00          20.26      20.26
    allegro       16       0.00          23.76      23.76
    leap          16       0.00          22.93      22.93
    f5d6          11       3.08          10.22       7.14

**The f5d6 number independently replicates.** `dextrack_vega` measured 3.1 cm in
May by a completely different method -- an exhaustive sweep of the three thumb
joints in the assembled Vega scene. This is a continuous optimisation over all
eleven hand joints in a separately compiled model, and it lands on 3.08 cm. Two
methods, one number; the opposition floor is a real property of the hand.

**What the axis shows.** Three of the four hands have a floor of exactly zero --
the thumb can be brought to the finger mean, so there is no object too small to
pinch. f5d6 is the only hand with a nonzero floor, and its aperture is also less
than half the others'. It is not "a worse dexterous hand"; it is a different kind
of device, one that cannot oppose at all and can therefore only press objects
against an external surface or against a second hand.

**Caveat on the metric.** Thumb-tip-to-finger-MEAN reaching zero does not prove a
useful pinch -- the thumb could be reaching into the middle of the palm between
spread fingers. The floor is a necessary condition for opposition, not a
sufficient one. A per-pair version (min over thumb-to-each-finger) and a
grasp-quality version (max epsilon over object sizes) are the natural refinements,
and the grasp bench already provides the second.

**Next for M2**: run the clean grasp bench across all four hands on a shared
object set, so the axis gains a second dimension -- what each hand can actually
hold, not just what it can reach.

## 2026-09-11: the opposition floor, rendered

`scripts/render_axis.py` puts each hand at its measured floor pose and marks the
thumb tip (blue) and the finger-tip mean (green), joined by a capsule.

- `figures/11_floor_f5d6.png` -- floor 3.08 cm. The thumb rests ON TOP of the
  four-finger stack and points the same direction as they do. It never rotates
  round to face them, so at the closest configuration the joint limits allow, the
  two markers remain about 3 cm apart. This is the mechanism behind the number.
- `figures/11_floor_leap.png` -- floor 0.00 cm. Only one marker is visible,
  because the thumb tip is exactly coincident with the finger mean.

**Stated plainly, because it is easy to misread the axis table as a claim about
grasping:** f5d6 has never grasped anything in this project and is not claimed to
have. The rendered pose is the closest approach, not a grasp.

**Two rendering notes worth keeping.** `MjvLight` has no `directional` field in
this MuJoCo build, so adding a scene light raises AttributeError whose traceback
is then buried under an EGL_NOT_INITIALIZED error from the Renderer destructor --
the EGL message is a symptom, not the cause. And a second EGL Renderer cannot be
created in a process after the first is closed, so the script renders one hand
per invocation.

## 2026-09-11: bimanual environment + scripted expert, with the one-handed control

`scripts/bimanual_env.py`, `scripts/bimanual_expert.py`. Two Menagerie LEAP
hands combined with `MjSpec.attach` (prefixes rh_/lh_, 46 joints, 44 actuators),
each on six position-controlled base DoF. Not f5d6 (3.08 cm opposition floor) and
not on the Vega arms (their reachable-orientation set would confound every result
with a limitation of the robot rather than of the hands).

**Task.** A peg stands in a socket in a free-standing base. The socket carries
1.20 N of joint friction; the base weighs 0.78 N. Extraction therefore needs more
upward force than the base weighs, so a one-handed pull lifts the whole base
instead. One hand presses the base flange, the other grasps the peg and pulls.

**Result.**

    condition            peg out   base moved   base lifted   tilt    success
    two-handed expert    12.98 cm     1.50 cm      +0.57 cm    6.9 deg   yes
    one-handed control    9.07 cm     7.15 cm      +6.10 cm   26.0 deg   no

The control fails in the designed way: it lifts the base 6.1 cm off the table.
Across six (socket friction, base mass) combinations spanning margins of
1.27x to 2.04x, **6/6 have the expert succeeding and the control failing**, so
this is not a knife-edge. `figures/13_expert_two.gif`, `13_expert_one.gif`,
`15_final_two.png`.

**What had to be fixed, all found by looking rather than sweeping.**

1. *Hands collided with each other* at reset (rh_palm vs lh_if_ds). The exposed
   flange was 2.5 cm wide, forcing the hands together; widened the base in y to
   give 6 cm strips and moved the stabilising hand to -y, away from the right
   hand's thumb.
2. *The stabilising hand never touched the box* -- 0.00 N, and two-handed and
   one-handed runs were identical. The palm's contact surface is ~5 mm below its
   origin, not the 2.8 cm assumed. Calibrated: z=0.052 gives 0.9 N and the box
   slides 8.66 cm under a 3 N tug; z=0.048 gives 7.6 N and 0.57 cm; z=0.044
   gives 14.3 N and no movement.
3. *The grasp closure did not match the object.* LEAP's closure gap spans
   5.32-19.26 cm, so the first knob's 4.4 cm faces were infeasible -- the fingers
   closed straight past it onto the lid. Replaced the hard-coded grasp offset
   with a per-run calibration that fits the closure fraction to the object width
   and measures the grasp-centre offset kinematically.
4. *The hand closed on itself.* Squeezing to frac 1.0 put rh_rf_tip against
   rh_th_tip at 12.8 N after the object escaped. The squeeze now stops at the
   hand's own gap floor plus a margin.
5. *The original task was a hinged lid, and the arc beat me.* The hand had to
   track a rotating target, and instead pressed the lid closed (85 N on the
   knob's top face, lid going 5.9 deg -> 2.2 deg). A peg is a pure vertical pull.
   Equivalent in principle, far less fragile to script.
6. *The approach motion was the problem, not the grasp.* Descending from above
   drove the fingers onto the peg's top face (163 N, gripping nothing); sweeping
   in laterally shoved the base 8 cm. The expert now STARTS from a
   penetration-free pre-grasp, which is what grasp_bench does and how Dexonomy
   structures its data -- pre-grasp / grasp / squeeze, not a reach. Reaching is a
   separate problem and is not what this environment exists to test.

**Status against NEXT.md.** Steps 1-3 are done: hands that grasp, a task that
needs both, and a scripted expert that solves it with its control failing. Step 4
(MJX port) and step 5 (BC then RL) are next. Nothing has been learned yet, and
the expert is the warm start when it is.

## 2026-09-11: MJX port, without degrading the physics

Two results, one of which overturns a claim carried since May.

**1. JAX CUDA works on this Jetson.** The note that "JAX has NO working CUDA
build on aarch64/Tegra" is out of date. `jax_cuda12_pjrt-0.6.2-manylinux2014_aarch64`
and `jax_cuda12_plugin-0.6.2-cp310-manylinux2014_aarch64` exist and are genuine
aarch64 ELF objects. They fail out of the box only because the plugin looks for
CUDA libraries in the standard locations and Jetson does not use them. Putting
the venv's own `site-packages/nvidia/*/lib` directories on `LD_LIBRARY_PATH`
yields `backend: gpu, devices: [CudaDevice(id=0)]`. Launcher: `mjx_env.sh`,
venv `.venv-mjx` (jax 0.6.2, mujoco 3.9.0, mujoco-mjx). Modal is therefore for
SCALE, not for access -- MJX can be iterated on the Orin.

**2. The port needs no physics substitutions.** The May recipe deleted collision
meshes, replaced fingertips with spheres, swapped Newton for CG and added
armature until it stopped diverging. That changes contact physics, which is why
it could only claim "parity within primitive mode" -- a policy trained there is
about a different world.

The distinction that recipe missed: on this scene the 252,089 mesh vertices are
almost entirely **visual** geoms (contype=0, conaffinity=0, density=0 in
Menagerie's `visual` class). They generate no contacts and carry no mass.
Deleting them is provably a no-op; deleting collision meshes is not. The actual
collision meshes here are the eight fingertips at 52 vertices each.

`scripts/mjx_port.py` strips only visual-only geoms (both conditions checked:
no collision AND no mass, since a contype=0 geom still contributes inertia
unless its density is zero) plus any mesh asset left unreferenced, and then
PROVES the strip is neutral by replaying an identical 400-step control sequence
through the full and stripped models on CPU:

    full     ngeom 179  nmesh 21  meshvert 252089
    stripped ngeom 145  nmesh  4  meshvert     208
    max |state difference| over 400 steps : 0.000e+00

Bit-identical. 1200x fewer mesh vertices, zero change to the dynamics, no
collision geometry touched, no solver swapped.

`mjx.put_model` then succeeds in **2.1 s** on the stripped model with Newton and
the elliptic cone left exactly as the expert was validated with.

**Open at time of writing:** the first `jax.jit(mjx.step)` compile has been
running over 10 minutes on this scene. That is an XLA compile cost over ~145
collision geoms, not a physics problem, and it is a one-time cost per shape --
but it needs measuring before the training loop is designed around it. The
honest next number is compile time and steps/s under `vmap` + `lax.scan`, which
is how MJX is actually used; stepping one environment from Python, as the parity
check does, is the worst case for compilation.

## 2026-09-11: MJX status -- what works, what does not

**Works.**
- JAX on the Orin GPU: `backend: gpu, devices: [CudaDevice(id=0)]`, reproducible
  via `mjx_env.sh`. Overturns the May claim. Modal is for scale, not access.
- The physics-neutral strip: visual-only geoms removed, 252089 -> 208 mesh
  vertices, bit-identical over 400 steps of identical controls (max difference
  0.000e+00). No collision geometry touched, no solver swapped.
- `mjx.put_model` on the stripped model: 2.1 s, Newton + elliptic cone intact.

**Does not work.**
- `jax.jit(mjx.step)` on this scene: a single-environment compile exceeded 15
  minutes and was abandoned; a 64-world `vmap` compile died silently, almost
  certainly OOM (the Orin has 15 GB of UNIFIED memory shared with the GPU, and
  ~6 GB was free). So the blocker is the XLA compile and memory, not the model.
- `mujoco_warp`: `KeyError` on the CCD kernel's shared-memory metadata
  (`..._cuda_kernel_forward_smem_bytes`) on sm_87. Reproduces on warp 1.16 and
  1.17, with mujoco aligned to 3.13, and after clearing the kernel cache. It is
  triggered by the four fingertip convex meshes. This is an upstream bug, not a
  configuration error.

**Read on why.** 145 collision geoms is a lot for MJX-JAX, which compiles a
whole-program kernel over a static collision-pair list. The Orin's 15 GB unified
memory makes the compile itself the constraint. None of this is a reason to
degrade the scene -- it is a reason to size the experiment to the device.

**Next measurements, in order, before any design decision:**
1. Compile time vs geom count -- one LEAP hand alone (~73 geoms) versus two.
   If it is superlinear, the bimanual scene is simply past this device's budget
   and the answer is Modal for training, Orin for development.
2. The smallest `nworld` that compiles and runs here, with its throughput.
3. Only if 1 and 2 both fail: substitute the eight fingertip collision meshes
   with fitted primitives AND MEASURE THE COST -- re-run the expert and the
   one-handed control in the substituted scene and report how the numbers move.
   That is the difference between a quantified approximation and May's
   unmeasured one.

## 2026-09-11: both MJX blockers fixed -- the scene runs on the Orin GPU

**Blocker 1, mujoco_warp's CCD crash, diagnosed and fixed.** It is an upstream
bug, and the diagnosis is what made it fixable. `convex_narrowphase` builds a
SEPARATE specialised CCD kernel for each convex geom-type pair (box-box,
box-mesh, mesh-mesh, ...) and asks warp for a launch block size for each. That
query does `module.load(device)` and reads `module_exec.meta[<kernel>_smem_bytes]`.
Once the module is loaded for the FIRST pair, warp returns the cached
`ModuleExec`, whose binary and metadata both predate the later kernels -- so the
SECOND distinct pair type raises

    KeyError: ..._ccd_kernel_<hash>_cuda_kernel_forward_smem_bytes

That is why a single-mesh toy scene works (tested: boxes-only OK, one convex
mesh OK) and two LEAP hands plus a box does not. Isolating it that way was the
step that turned "warp is broken here" into a one-line cause.

`scripts/warp_fix.py` unloads the kernel's module on the missing key, so the
next load rebuilds with every kernel currently registered, and falls back to
`naconmax` if that still fails. Returning a grid size ALONE is not sufficient --
the kernel is genuinely absent from the loaded module and the following
`wp.launch` then fails with `CUDA error 500: named symbol not found`. Both
branches touch only module compilation and launch width, never the simulation.

**Blocker 2, the XLA compile, is sidestepped rather than fought.** With warp the
same unmodified scene compiles in ~1 s (48 s cold, then cached) against an XLA
compile that exceeded 15 minutes for one environment and OOM-killed at 64.

**A capacity bug found on the way, which matters more than the speed.** warp's
default `njmax` is 64. The real expert episode peaks at **ncon 44, nefc 167**, so
constraints were being silently dropped -- `nefc overflow - please increase njmax
beyond 64`. That is a physics error, not a performance one. Sized to
`nconmax=256, njmax=512`.

**Where it lands, with correct capacity:**

    nworld      compile      throughput
         1      48 s cold        35 env-steps/s
       256      cached        2,655 env-steps/s
      2048      cached        2,098 env-steps/s

2048 is slower per env-step than 256, consistent with memory pressure on the
Orin's 15 GB unified memory at njmax=512. 256 worlds is the current sweet spot.

**Parity, CPU MuJoCo vs warp, 300 steps of identical controls:** max qpos
difference 3.3e-7 through step 50, then chaotic divergence (1.0e-2 by step 100,
3.1e-1 by step 200). That is expected -- warp is float32, MuJoCo CPU is float64,
and this is a contact-rich system. Step-level agreement is therefore the wrong
test past the first few dozen steps. **The test that matters is whether the TASK
OUTCOME survives: the expert succeeding and the one-handed control failing when
both are run under warp. That is not done yet and is the next thing.**

## 2026-09-12: outcome parity, CPU vs warp -- the gap, stated as a number

Recorded the expert's 3470 control steps on CPU and replayed them verbatim
through warp (the expert is open-loop, so this is a fair replay).

    condition            backend   peg out   base moved   base LIFT
    two-handed expert    CPU        12.96 cm     1.54 cm     +0.59 cm
    two-handed expert    warp       13.42 cm     2.75 cm     +0.63 cm
    one-handed control   CPU         9.02 cm     7.09 cm     +6.08 cm
    one-handed control   warp        1.32 cm     6.67 cm     +3.95 cm

**Against the pre-registered criterion (peg >= 8 cm, base displacement < 2 cm,
tilt < 15 deg) the two-handed expert FAILS under warp** -- not on the peg, which
comes out further than on CPU (13.42 vs 12.96 cm), but on lateral base
displacement: 2.75 cm against a 2 cm threshold.

**The honest reading, without moving the goalposts.** Two of the three clauses
reproduce cleanly and one does not:

- peg extraction: reproduces, and the two-handed/one-handed separation survives
  (13.42 vs 1.32 cm under warp; 12.96 vs 9.02 on CPU).
- base LIFT -- the quantity the task is actually built around, since the socket
  friction exceeds the base weight -- separates by ~6x on CPU (0.59 vs 6.08) and
  ~6x under warp (0.63 vs 3.95). Reproduces.
- base LATERAL displacement differs between backends by 1.2 cm on the same
  control sequence, and the threshold was set at 2.0 cm. **The threshold sits
  inside the backend noise, so it is not a usable discriminator.** That is a
  defect in my criterion, not a property of the task.

I am NOT rewriting the criterion to make this pass. The measured transfer gap is
**1.2 cm of lateral base displacement on an identical open-loop control
sequence**, and any future criterion has to be chosen with that in mind and
stated before the run, not after. The next criterion should discriminate on base
lift, which the task's own force design guarantees and which both backends agree
on -- but it must be fixed in advance and re-validated on CPU first.

Also worth noting: the one-handed control diverges far more (9.02 vs 1.32 cm of
peg travel) than the two-handed case. That is expected -- the one-handed run is
the unstable one, with the base tumbling -- and it is a reminder that the
divergence is largest exactly where the dynamics are least constrained.

## 2026-09-12: criterion v2 validated on both backends

Criterion v2 was written into `bimanual_env`'s docstring BEFORE the run that
used it: peg extracted >= 8 cm, base LIFT < 2 cm, base tilt < 15 deg. Lateral
displacement is still reported but no longer gates, because CPU and warp differ
by 1.2 cm on it with identical controls and the old threshold was 2.0 cm.

    condition            CPU ok   warp ok   CPU peg   warp peg
    two-handed expert      True      True     12.96     13.42
    one-handed control    False     False      9.02      1.24

Outcome parity holds: both backends agree on both verdicts. Training in warp and
evaluating on CPU is now licensed for this task, with the lateral-displacement
gap (1.2 cm) recorded as the known difference.

## 2026-09-12: M2 complete -- the opposition axis gains its second dimension

`opposition_axis.py` measured REACH (how close the thumb gets to the fingers).
`hand_axis_bench.py` adds HOLD: across a shared set of block widths, which can
each hand actually close on and hold against gravity, nothing underneath.

Everything mechanical is derived per hand by one identical procedure, never
hard-coded. `hand_specs.derive_flex` solves for the closure over ALL joints at
once, using the same thumb-to-finger objective as the axis. Two cheaper versions
were tried and both got LEAP's sign backwards -- deriving `if_mcp = -0.31` where
the validated closure uses +0.9 -- because a closure is a joint COMBINATION and
single-joint probes cannot see it.

    hand      closed gap   gap range      feasible   HELD   widths held (cm)
    shadow       0.25 cm                        8      7    2,3,4,5,6,7,9
    leap         0.80 cm                        8      5    3,6,7,9,11
    allegro      2.41 cm                        7      3    6,7,11
    f5d6         3.40 cm   3.4 - 7.8 cm         4      0    none

**The ordering is monotone in the opposition floor.** The hand that can bring
its thumb closest to its fingers is feasible on the most widths and holds the
most of them; f5d6, with the highest floor and much the narrowest gap range,
holds nothing at all -- not even the four widths it can nominally close on. So
the floor is not merely a necessary condition, it predicts the outcome.

f5d6 holding 0/4 is a 0/N, and it is allowed to stand as a boundary here because
the pairing protocol is satisfied by construction: three other hands were run
through the identical procedure on the identical widths and held 3, 5 and 7 of
them.

**A bug worth recording.** The first run reported every hand as infeasible
everywhere, with shadow's gap curve spanning 10.8-12.0 cm when its closed gap is
0.25 cm. The Menagerie hands ship their own actuators (`if_mcp_act`), and the
bench looked only for `act_<joint>`, so it found none and swept a completely
unactuated hand. Now every flex joint is mapped to whatever actuator drives it,
with an assertion that at least four are wired.

## 2026-09-12: behaviour cloning -- the first learned policy in this project

`scripts/bc.py`, collect -> train -> evaluate, entirely local. 20 randomised
expert demos (socket friction 0.9-1.8 N, base mass 0.06-0.12 kg), 17360
(obs, act) pairs, 52-dim observation, 44-dim action. Plain MLP 52-512-512-44,
ReLU, MSE on z-normalised data, Adam 1e-3, 240 epochs, torch on the Orin GPU.
Six held-out episodes, drawn with seeds the demos never used.

    method          success                peg out (cm)
    do-nothing      0/6                   -0.00 +/- 0.00
    random          0/6                   -0.00 +/- 0.00
    expert          6/6                   12.89 +/- 0.08
    BC (3 seeds)    67% +/- 47%            8.01 +/- 6.65

**BC beats both trivial baselines and, when it works, matches the expert**
(seeds 0 and 2: 6/6 at 12.71 and 12.70 cm against the expert's 12.89). **Seed 1
collapses completely** -- 0/6, driving the peg 1.4 cm the wrong way. The seed
spread is 47 percentage points against a 67-point gap to baseline, so by
RULES.md #3 this is a real but marginal result and must be reported with the
spread, never as "BC solves the task".

**The phase input is what made it work at all.** Without it BC scored 0/6 on
every seed and pushed the peg 2.9 cm backwards. The scripted expert is open-loop
once calibrated, so its action is a function of TIME as well as state: two
moments with near-identical states -- descending versus holding -- demand
different actions, and a state-only policy can only average them. Adding
[phase, sin 2*pi*phase, cos 2*pi*phase] is what let the policy represent the
demonstrator at all. That is a property of cloning a time-indexed expert, not a
hyperparameter.

The obvious next levers for the seed fragility, none of them tried yet: action
chunking (ACT-style), predicting deltas rather than absolute position targets,
more demonstrations, or an ensemble.

## 2026-09-12: superseded results moved out of reach

`results/retracted/` now holds 18 result files with a per-file table saying why
each is there. The two that matter:

- `final_table.json` -- its config-A row was order-dependent (the retargeting
  optimiser seeded from the robot's current pose). eps = 0 and "never moves the
  box" survive every ordering; the contact count and exact keypoint distance do
  not. Never re-run, because the direction it served was abandoned.
- `hand_bench.json` -- LEAP scoring 0/72 is a bug in my closure, not a property
  of LEAP. Recorded as such so it can never be misread as a finding.

## 2026-09-12: where RL has headroom, measured rather than assumed

Before running RL it is worth knowing whether there is anything for it to win.
The scripted expert turns out to be robust on every difficulty axis tried:

    socket friction  1.2 -> 4.4 N     5/5, peg 12.98 cm, unchanged
    peg friction     1.0 -> 0.15      5/5, peg 12.98 -> 13.86 cm
    start-pose jitter 0 -> 3 cm       5/5, peg 12.98 cm, unchanged

The last one is not robustness, it is irrelevance: the expert commands absolute
position targets through kp=4000 servos, so where the hand STARTS does not
matter. A residual has nothing to improve on any of these.

The axis where it does break is PERCEPTION -- being wrong about where the peg
is, which is the one thing an open-loop expert cannot absorb:

    perception error   expert success   peg out (cm)
        0.0 cm             5/5             12.98
        1.0 cm             5/5             12.65
        1.5 cm             5/5             12.12
        2.0 cm             4/5             11.67
        3.0 cm             2/5              6.57

That is the headroom, and it is the classic case for learning over scripting: a
closed-loop policy observes the actual peg state and can correct for a wrong
belief about it. The comparison at 3 cm is the experiment.

## 2026-09-12: does the learned policy beat the expert under perception error? No.

The headroom was real (expert 5/5 at 0 cm error, 2/5 at 3 cm), so the experiment
is well posed: at 3 cm of peg-position error, does a closed-loop policy that can
SEE the peg beat an open-loop expert that cannot?

    condition                          success        peg out (cm)
    expert, 3.0 cm perception error    3/6            11.82 +/- 0.82
    BC clean-trained, 3.0 cm error     61% +/- 21%    10.32 +/- 8.01
    BC noise-augmented, 3.0 cm error   50% +/- 41%     4.26 +/- 8.46

**No result.** Clean-trained BC is nominally ahead of the expert, 61% against
50%, but the seed spread is 21 points against an 11-point gap. By RULES.md #3
that is not a result and must not be reported as one. Noise augmentation, which
was meant to be the fix, is if anything WORSE -- 50% with a 41-point spread.

**Why the augmentation failed, which is the useful part.** I duplicated the
dataset with corrupted peg observations but the SAME actions. That teaches the
policy that the action is independent of the peg reading -- i.e. it trains the
policy to ignore the observation it was supposed to learn to use, making it more
open-loop rather than less. The augmentation was self-defeating by construction.

Doing it properly needs the policy's own visited states labelled by the expert
(DAgger), so that a wrong peg reading is paired with the CORRECTIVE action
rather than the nominal one. That is the next experiment, not a tuning change.

**So the honest status of learning in this project:** BC reproduces the expert
on a clean task (67% +/- 47% against 0% baselines) and does not yet beat it
anywhere. No RL has been run, and the measurements above are the reason -- there
is no headroom on the axes the expert already handles, and on the one axis where
there is headroom, the obvious cloning approach does not claim it.

## 2026-09-12: figures

- `figures/16_opposition_axis.png` -- the project's central result in one frame:
  closing distance per hand against which block widths it holds. Same ordering on
  both panels. Palette validated (blue/orange, CVD dE 24.7, normal-vision 33.6
  against a #fcfcfb surface); the first attempt used green/red and FAILED CVD
  separation at dE 4.1, so the states also carry glyphs and are never colour
  alone.
- `figures/17_bc_policy.gif` -- the BC policy, not the scripted expert,
  extracting the peg: 12.67 cm, base lift +0.78 cm, success.

## 2026-09-12: restructured into a package

The repo was 19 live scripts reaching each other through `sys.path.insert`, with
external paths hard-coded in five of them. A refactor in that state could
silently invalidate any published number, and adding anything meant guessing
which file owned what.

    src/oppdef/      library code -- no CLI, no path games
      paths.py       menagerie / vega urdf / dextrack, one place, env-overridable
      metrics/ hands/ envs/ control/ sim/ learning/ viz/
    experiments/     every runnable thing: python -m experiments.<name>
    tests/           13 invariants
    attic/           superseded code, with the table of reasons
    docs/            goal, plan, protocol, next
    NOTES.md         this file -- the only authoritative numbers

`pyproject.toml` + `Makefile`; editable install verified; all 13 modules import;
the axis figure regenerates through the new entry point; `make expert` still
reproduces 12.98 / 9.07 cm.

**The tests are the point.** Each one is a bug that actually happened this
session, written so it cannot happen again silently:

- epsilon's five analytic ground truths, including monotonicity in mu -- the
  reason cone edges are scaled to unit NORMAL component rather than unit length
- the object rests ON its support rather than 4 cm inside it, and the
  do-nothing baseline is zero (the defect that made three PPO runs look like
  they had learned a 4 cm lift)
- the two hands do not touch at reset (they did, at a 2.5 cm flange)
- extraction costs more upward force than the base weighs, so the task IS
  bimanual by construction rather than by assertion
- the peg's graspable face lies inside LEAP's closure range (a face below it is
  infeasible and the fingers close straight past)
- the visual strip is bit-identical, so the MJX port cannot start changing
  physics unnoticed
- the expert succeeds and the one-handed control fails BY LIFTING THE BASE
- f5d6's opposition floor separates it from LEAP

Two stale-path bugs surfaced during the move and were caught by those tests
rather than by a later wrong number, which is the whole argument for having them.

## 2026-09-12: published

`github.com/luaiabuelsamen/opposition-deficit`, **private** -- the direction is
unpublished and aimed at RSS 2027, and private->public is one command while the
reverse does not un-index anything. `gh repo edit --visibility public` when the
paper is out or the advisor needs it wider.

Portability work before pushing: the Makefile's interpreter defaults to
`python3` and is overridable; `paths.py` fails with the fix rather than a stack
trace, and the author's own checkouts are marked as dev defaults; `make vendor`
sparse-clones the Menagerie models. Three of the four hands need nothing beyond
that clone -- only the f5d6 rows need a URDF that is not redistributed here.
Audited clean for secrets and credentials before the first push.

**A transport gotcha worth keeping.** Pushing 56 MB from this Jetson failed
every way -- SSH dropped with "unexpected disconnect while reading sideband
packet" and HTTPS with "GnuTLS recv error (-12)" -- and it was NOT size:
an 11-commit chunk of text-only commits failed identically while SSH auth
succeeded. What fixed it was the transport options, now persisted in the repo's
own config:

    core.sshCommand = ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
                          -o IPQoS=throughput -o Compression=no
    pack.threads = 1

With those, the history pushed in chunks of three commits and then completed in
one. Anyone hitting a stalled push from a Tegra board should try this before
assuming the pack is too large.

## 2026-09-12: infra front 4 -- sensing

`src/oppdef/sensing.py`. An observation is now a declared `ObsSpec` instead of
52 numbers assembled inline, and every channel states whether it is ONBOARD or
PRIVILEGED:

    FULL      137d  [proprio_pos=44 proprio_vel=44 tactile=32 object_pose=7*
                     object_vel=6* task=1* phase=3]
    ONBOARD   123d  [proprio_pos=44 proprio_vel=44 tactile=32 phase=3]
    LEGACY     55d  [proprio_pos=44 object_pose=7* task=1* phase=3]

`LEGACY` is exactly what the first BC saw, kept so the gap is legible: no
velocities, no tactile, and three privileged channels including `task`, which IS
the success metric. `ObsSpec.onboard_only()` makes the sim-to-real question an
ablation you run rather than a caveat you write.

**Tactile now exists and is role-discriminative.** 32 touch sensors, one per
fingertip collision pad. Over an expert episode:

    right hand (grasps the peg)   5/16 pads active, peak 10.78 N
    left hand  (presses the base) 3/16 pads active, peak  2.91 N

The two hands' roles are visible in the tactile channel alone. Adding the
sensors does not perturb the dynamics -- the expert still extracts 12.98 cm --
which is asserted in `tests/test_sensing.py` rather than assumed.

## 2026-09-12: infra front 3 -- embodiment registry

`src/oppdef/embodiment.py`. One registry replaces three modules that each knew
about models their own way. Adding a hand was a three-file edit; adding an arm
was impossible.

    make(hand="leap")                  floating hand
    make(hand="leap", count=2)         two, prefixed rh_/lh_
    make(hand="leap", arm="ur5e")      MOUNTED at the arm's attachment site
    make(hand="leap", free_base=True)  6 position-controlled base DoF

    hand  leap       16 DoF     arm  ur5e        6 DoF
    hand  leap_left  16         arm  panda       8
    hand  allegro    16         arm  xarm7       8
    hand  shadow     24         arm  so_arm100   6  <- the SO-101 family, the
    hand  f5d6       11                             lab's real hardware

All nine load, asserted in `tests/test_embodiment.py` rather than assumed.
Composition verified: leap+ur5e (nq 22), shadow+so_arm100 (nq 30), leap+xarm7
(nq 29), two LEAP hands (nq 32).

**Two traps found by building it.** MuJoCo namespaces bodies and joints on
`MjSpec.attach` but NOT assets, so a hand and an arm that both define a material
called "black" fail to compile -- every hand+arm build died on that until each
attach got a prefix. And attaching to the worldbody puts the hand at the origin
*beside* the arm rather than on it; mounting needs the arm's declared site, so
`arm=` now means mounted and a test asserts the palm is not at the origin.

Menagerie sparse checkout widened to include franka_emika_panda,
universal_robots_ur5e, trs_so_arm100 and ufactory_xarm7. `menagerie_xml` now
resolves any `meshdir` spelling against the model's own directory -- the
previous version matched two known spellings and `trs_so_arm100` uses a third.

## 2026-09-12 — retargeting (front 2), and a units defect in the bimanual env

### Defect found in already-validated infra: the wrist could not rotate

`envs/bimanual._add_base_dof` wrote `range=[-3.2, 3.2]` for the three base
hinges, intending radians, into a spec that compiles angles in **degrees**. The
matching actuator `ctrlrange` stayed ±3.2 **radians**. Compiled:

    rh_rz_act   ctrlrange [-3.2, 3.2]   jnt_range [-0.0559, 0.0559]

So every commanded wrist rotation past 3.2° saturated at the joint limit, with
no error anywhere. `control/expert.best_rz` sweeps rz over `linspace(-pi, pi,
48)` and scores each — it was selecting among poses the hand could not take.

**Retraction:** any statement that the expert's wrist angle was *optimised* is
withdrawn. The expert's measured numbers were measured and stand.

Fixed to `[-180, 180]` (degrees, matching the spec's unit). Verified: a
commanded `rz = 1.0 rad` now reaches `qpos = 1.0000 rad`; before it would have
clamped at 0.0559.

Re-measured after the fix, same protocol:

| run | peg out | base moved | base z | tilt | ok |
|---|---|---|---|---|---|
| two-handed expert (was 12.98 cm) | **13.49 cm** | 3.03 cm | +0.44 cm | 3.1° | yes |
| one-handed control | 5.32 cm | 12.36 cm | +10.26 cm | 43.1° | no |

The claim the task rests on — it requires two hands — survives the fix, and the
expert is slightly better with a wrist that can actually turn.

### Retargeting: six corrections before the comparison measured anything

The first run reported `eps = 0.0000, 0 contacts` for all four hands under both
objectives. Causes, in the order they were found — each produced plausible
output while being wrong:

1. **Epsilon is exactly flat before first contact.** With no fingertip
   touching, eps is 0 in every direction, so L-BFGS-B's numerical gradient is
   zero and the search terminates where it started. Not a tuning problem — the
   shape of the function. Fixed with the DexGraspNet/BODex decomposition
   (E_fc + E_dis + E_pen): a reach term gives gradient before contact, a
   penetration term keeps tips out of the interior. Epsilon is still what is
   reported, measured by the same hard geometric test.
2. **Thumb correspondence was inverted.** The reference stacked the thumb
   first, `Hand.tip_names` stacks it last, so the frame fit paired the human's
   thumb with the robot's index finger and dropped the robot's thumb entirely.
3. **Wrist-relative vectors are not comparable across hands.** The body a model
   calls the wrist is a modelling choice; Shadow's palm body sits 25 cm from its
   own fingertips (it includes the forearm), LEAP's 8 cm. **Rendered, the object
   came out buried inside LEAP's palm.** Now fitted on inter-fingertip vectors,
   which is what DexPilot and AnyTeleop optimise and what has no origin to
   disagree about.
4. **Every way of fitting a scale was wrong.** Least squares collapsed toward
   zero when the shapes differed (0.50 for LEAP, whose fingers are ~0.8× the
   reference's) and buried the object 2 cm into the palm; a span ratio gave
   3.26 because it compared a *closed* human grasp against the robot at its
   *open* zero posture. Not fitted at all now: the demonstration held a box of
   known width, so testing width `w` scales it by exactly `w / w_ref`, and `w`
   is swept. Scale became a measurement instead of a fudge factor.
5. **Fingertips are near-planar**, so inter-tip vectors are blind to which side
   of that plane the palm is on — the one DoF deciding whether the fingers point
   at the object or away from it. LEAP's palm was placed *above* the box with
   its fingers extending further above, and every keypoint fit hovered ~15 cm
   clear. The rotation fit now includes a unit palm direction.
6. **The base hinges could not turn** (the units defect above), so the hand
   could not face the object at all. The optimiser's bounds are opened in
   `retargeter_for`; the model is left alone so the env keeps the limits its
   results were measured under.

**Guard added:** every row now reports the DEMONSTRATION's own epsilon beside
the fits. Without it there was no way to see that the question was ill-posed
rather than the hands bad. The synthetic reference scores eps ≈ 0.21 with 4/5
contacts on its own object, so it is a real grasp — `test_demonstration_is_
itself_a_grasp` locks that in.

### Convergence control — is this measuring retargeting or my optimiser?

Asked before reporting anything, because BLEND was beating pure EPSILON
(0.2614 vs 0.1400), which is only possible if the epsilon search is failing.

    leap keypoint   restarts  4: kp 3.23 cm  eps 0.0000  cts 0
                    restarts 12: kp 1.41 cm  eps 0.0000  cts 0
                    restarts 24: kp 1.41 cm  eps 0.0000  cts 0
                    restarts 24, maxiter 2000: kp 1.41 cm  eps 0.0000  cts 0
    leap epsilon    restarts  4: kp 4.64 cm  eps 0.1400  cts 3
                    restarts 12: kp 7.53 cm  eps 0.4026  cts 4

Keypoint is **converged** — error plateaus at 1.41 cm from 12 restarts on, and
contacts stay 0. That result is real. Epsilon at 4 restarts was **not**
converged (0.1400 → 0.4026 at 12), which is what produced the BLEND inversion.
**Comparison runs must use >= 12 restarts**; the 4-restart smoke numbers are not
reportable.

Nothing about the keypoint-vs-epsilon comparison is claimed yet: the reference
is `SyntheticSource`, an analytic stand-in, not MANO. Any result carries that
caveat until ARCTIC/Dexonomy lands.

## 2026-09-12 — policy (front 5) and scale (front 6)

### MJX is unavailable on this machine, and the reason is specific

The note carried since May said JAX had no working CUDA build on aarch64/Tegra.
That was overturned earlier: `mjx_env.sh` gets `backend gpu`, and GEMM runs.
The correction to the correction is that **not all of JAX works**:

    matmul          ok
    jnp.linalg.cholesky   INTERNAL: cuSolver internal error
    jnp.linalg.solve      gpusolverDnCreate(&handle) failed: cuSolver internal error

`gpusolverDnCreate` cannot create a handle at all, so this is not about scene
size or MJX: any MJX step fails, including on a two-body cart-pole. Tried and
rejected:

* system CUDA (`/usr/local/cuda/targets/aarch64-linux/lib`) ahead of the pip
  wheels — cuSPARSE then fails to load and JAX falls back to CPU entirely;
* `LD_PRELOAD` of JetPack's `libcusolver.so.11` (145 MB) over the pip wheel's
  (298 MB), keeping the pip path order — same cuSolver error.

**So MJX is not a usable backend here and no MJX number should be quoted from
this machine.** `mujoco_warp` is the GPU path that works, and it is what the
`warp` backend in `oppdef.vec` uses.

### Batched stepping

`oppdef.vec` puts CPU, MJX and warp behind one interface, each reporting its
divergence from single-world CPU MuJoCo rather than being trusted.

    scene: nq 52  nv 51  nu 44  ngeom 145  (bimanual, visual geoms stripped)

    AGREEMENT (8 worlds, identical controls, 60 steps)
      world-to-world           0.000e+00     (bitwise)
      vs single-world mj_step  1.261e-08

    THROUGHPUT (cpu, 8 cores)
      n=  1     2206 steps/s   1.00x
      n=  8     9035 steps/s   4.10x
      n= 32    13435 steps/s   6.09x
      n=128    15212 steps/s   6.90x
      n=256    17022 steps/s   7.72x

The 1.26e-08 is not error in the batch: the vec backend re-seeds state each
step, so the solver warm-start does not carry across steps the way it does in a
continuous `mj_step` loop. Worlds agree bitwise with each other.

    warp vs cpu, cart-pole, 40 steps:  max 4.060e-07   final 2.187e-07

which is float32-vs-float64, the expected size.

Bugs fixed on the way: `mujoco.rollout` takes one `MjData` per THREAD (the
worlds are a work queue through that pool), so seeding the batch from that pool
made it infer the batch size as `nthread`; passing a list of N models made it
infer N. Both raise the moment `n != nthread`, which is the normal case.

### Action chunking

`oppdef.policy` adds an ACT-style chunked policy: predict the next H actions,
execute with temporal ensembling. `horizon=1` reproduces the MLP baseline
exactly so the comparison is measurable rather than asserted.

The ensembler was wrong first: it kept only each chunk's FIRST element, which
is a moving average of fresh predictions, not an ensemble. The prediction about
NOW made i steps ago is element i of the chunk emitted then. With identical
chunks `[1,2,3,4]` it settled on 1.0 where the answer is 2.5.

Episodes are chunked separately — a window running off one demonstration into
the next teaches the policy to follow a trajectory with a different peg pose,
and it looks exactly like ordinary training noise.

### Retargeting: the palm is now fixed for every condition

Letting the palm float and placing it at the human's wrist offset rescaled by
finger length landed it ~2 cm out, and 2 cm is the difference between every
contact and none — so the comparison was measuring wrist placement, not the
objective. With the palm fixed at the hand's own pose and the object at its
opposition-axis midpoint, only the fingers differ, and a `+squeeze` condition
gives the keypoint baseline what practitioners actually ship.

Smoke result (allegro, 5 cm box, palm fixed):

    demo        eps 0.1059  3 contacts
    keypoint    eps 0.1603  4 contacts   kp err 3.03 cm
    +squeeze    eps 0.1603  4 contacts   (never worse than its input)
    epsilon     eps 0.4537  4 contacts   kp err 5.93 cm

2.8x the epsilon, bought by letting keypoint error roughly double. Still on
`SyntheticSource`, which is an analytic stand-in and not MANO.

## 2026-09-12 — retargeting result: four hands, four widths, palm fixed

Protocol: the palm is fixed at the hand's own pose with the object at the
midpoint of its opposition axis, identically for every condition, so only the
finger angles differ. The demonstration is scaled to the box under test
(`w / w_ref`). Contacts are geometric (nearest point on the box surface, that
face's normal), so these are grasp-synthesis-style predictions to be validated
in simulation, not physical results.

    mean Ferrari-Canny epsilon over 16 hand x width cells

      human demo            0.1194     eps = 0 in  0/16
      keypoint              0.0451     eps = 0 in 11/16
      keypoint + squeeze    0.0511     eps = 0 in 10/16
      epsilon-optimised     0.3570     eps = 0 in  0/16

Per hand, keypoint achieves force closure at NO width on **leap** and **f5d6**;
on **allegro** at three of four; on **shadow** at two of four. The
epsilon objective achieves it everywhere, at 0.19-0.47.

The sharpest single cell is leap at a 3 cm box: keypoint matches the
demonstration's inter-fingertip geometry to **0.01 cm** -- essentially exactly --
and gets **epsilon = 0 with zero contacts**, while the epsilon objective from
the same palm pose reaches 0.246. Reproducing the human's hand shape to a
tenth of a millimetre produces no grasp at all.

Squeezing barely helps: it rescues one cell of eleven (f5d6 at 3 cm, 0 ->
0.096). So this is not "the keypoint pose just needed closing".

Figure: `figures/retarget_epsilon.png` (`make figures-retarget`).

**Caveats that travel with every number above.** The reference is
`SyntheticSource`, an analytic stand-in, NOT MANO -- ARCTIC/Dexonomy
registration is still pending. Contacts are geometric, not simulated. And the
epsilon condition is an optimiser given the epsilon objective, so it should win
on epsilon; what is informative is the SIZE of the gap and that keypoint lands
at exactly zero, which is a qualitative failure rather than a smaller number.

## 2026-09-12 — action chunking: a null, and a nearly-reported artefact

Three training seeds per horizon, eight evaluation episodes each, identical
demonstrations and evaluation seeds throughout. Policy queried every `stride`
sim steps and its action held in between, matching the rate it was fitted at.

    with temporal ensembling (m = 0.01)
      do-nothing                    0%
      random                        0%
      expert                      100%          peg 13.42 +/- 0.15 cm
      MLP (horizon 1)             100% +/-  0.0  peg 12.84 +/- 0.41 cm   3/3 seeds
      chunked (horizon 8)        66.7% +/- 47.1  peg  6.58 +/- 9.52 cm   2/3 seeds
      chunked (horizon 16)       66.7% +/- 47.1  peg  6.85 +/- 9.17 cm   2/3 seeds

    without ensembling (m = 50, newest chunk only)
      chunked (horizon 8)        66.7% +/- 47.1  peg  6.58 +/- 9.57 cm
      chunked (horizon 16)       33.3% +/- 47.1  peg -0.11 +/- 8.92 cm

**Action chunking does not help on this task and destabilises it.** The
one-step MLP succeeds on 3/3 seeds; both chunked horizons lose a seed, and the
lost seeds do not degrade gracefully -- they drive the peg ~6-7 cm the WRONG
way, the same signature as the phase-less MLP failure.

### The artefact this nearly became

Run once per horizon, the table read: horizon 1 -> 8/8, horizon **8 -> 0/8**,
horizon 16 -> 8/8. A hole at 8 between two working horizons is not something
any property of chunking explains, and it was tempting to report it as one.
Retraining horizon 8 with a different torch seed on the same demonstrations
gave 8/8. **The spread across training seeds is larger than any difference
between horizons**, so the single-seed table would have reported initialisation
luck as an architecture result. Every horizon is now trained from several seeds;
`chunk_bc.py` documents why.

### A hypothesis that was wrong

The natural explanation was temporal ensembling: it averages predictions made
up to H queries ago, and the demonstrator is open-loop and time-indexed, so lag
should hurt. A single-seed probe agreed (m = 50 gave the best number seen).
Run properly across three seeds it does not hold -- the failing seed fails
either way, at -6.88 cm with ensembling and -6.95 cm without, so **the
instability is in the trained chunk predictor, not in the ensembler.**
Ensembling does help horizon 16 (66.7% vs 33.3%), but it does not fix the
failure mode.

### What this is not

A claim about action chunking in general. This task is SATURATED -- the
one-step MLP already matches the expert (12.84 vs 13.42 cm), so there is no
headroom for a better architecture to show value in, and the demonstrator is an
open-loop time-indexed script, which is the regime where committing to a
multi-step intent buys least. Three seeds is also few: 66.7% vs 33.3% is one
seed. What the experiment does establish is that the infrastructure works end
to end and that horizon comparisons on this task need seeds, not one run.

## 2026-09-12 — RETRACTION: the retargeting epsilons are not physical

The hold test (`oppdef.hold`) was built to check whether the epsilon this
project computes predicts what a hand can physically hold. Run on LEAP's own
epsilon-optimal pose at a 5 cm box, **every condition failed before the test
even started**: 44 contacts at t=0, the object interpenetrating the hand by
19 mm, flung 70 cm away within 0.3 s.

Measured penetration of the fitted poses (box half-extent 25 mm):

    condition           eps      body pen     tip pen
    keypoint         0.0000     33.65 mm    13.91 mm
    keypoint+squeeze 0.0000     33.65 mm    13.91 mm
    blend            0.3598     28.17 mm    34.26 mm
    epsilon          0.3581     16.38 mm    19.11 mm

The fingers are inside the box. Two compounding defects in
`geometric_epsilon`, both mine:

1. **Fingertips are treated as POINTS at body origins.** The real fingertip
   geom has extent and an offset, so a body origin sitting 6 mm outside the
   surface can have its collision geom 19 mm inside. The "tip distances" this
   module reported were distances for a point that is not where the finger is.
2. **Non-fingertip links are not modelled at all.** The optimiser is free to
   drive proximal and middle phalanges straight through the object, and it
   does. A fingertip-only penetration term cannot see this, which is why
   DexGraspNet and BODex carry a whole-hand `E_pen` rather than a fingertip one.

**What is retracted.** Every epsilon in the 2026-09-12 retargeting table, as a
statement about physically realisable grasps. The comparison remains valid as
what it literally is -- what each OBJECTIVE optimises under a point-fingertip,
no-body-collision idealisation -- and the qualitative gap (keypoint lands at
exactly zero in 11/16) was measured under that idealisation for both
conditions equally. But no claim that an epsilon-optimised pose is a better
GRASP survives, because none of these poses is a grasp.

**What replaces it.** The correct contact set already exists in this repo and
predates the retargeting module: `metrics.epsilon.object_contacts` reads real
MuJoCo contacts -- true positions, true normals, true friction -- and it is what
the validated hold bench (M2, the opposition-floor result) uses. The retargeting
objective must be rebuilt on it: set qpos, run collision, extract contacts,
compute epsilon. Slower per evaluation, and correct.

`retarget.Penetration` is the first piece: exact whole-hand penetration depth
from MuJoCo's own collision detection, ready to enter the energy.

**The method worked.** The geometric surrogate looked right, produced a clean
monotone figure and a quotable headline, and was wrong. Nothing caught it until
an object was placed in a simulator and pushed. That is the whole argument for
the physics gate, and it is now a standing requirement: no epsilon enters a
finding until the pose behind it has been held against a wrench in MuJoCo.

## 2026-09-12 — the second hand, measured in physics

The retracted retargeting result was geometric. This one is not: every grasp
below is produced by CLOSING a hand in simulation (pre-grasp -> close ->
squeeze), its epsilon comes from real MuJoCo contacts, and it is then validated
by pushing the object in 14 directions until it slips.

### First: does epsilon predict holding at all?

57 LEAP grasps on a 5 cm box, approach sampled at random, each hold-tested:

    eps bin        n    mean force sustained in the worst direction
    0.00-0.15     21    0.55 N
    0.15-0.25     11    1.14 N
    0.25-0.35     15    1.87 N
    > 0.35        10    2.64 N

Monotone, ~5x across the range. Per-grasp it is noisy (Spearman +0.39, and the
force ladder is coarse), but the instrument tracks the physics in aggregate.
This is the check the geometric epsilon never had.

### Then: what does a second hand buy?

Approach searched by CEM (6 iterations x 24 samples) per hand per width, once
with one hand and once with two, then hold-tested.

    hand        w    eps_1   eps_2   hold_1  hold_2
    allegro   3.0   0.1774  0.6308    0.50    2.00
    allegro   4.0   0.3982  0.6200    0.50    4.00
    allegro   5.0   0.6438  0.6563    2.00    4.00
    allegro   6.0   0.6390  0.6241    2.00    8.00
    allegro   7.0   0.4686  0.6518    2.00    4.00
    allegro   8.0   0.4562  0.6692    1.00    8.00
    allegro   9.0   0.3980  0.5761    0.50    8.00
    leap      4.0   0.5214  0.6313    4.00    8.00
    leap      7.0   0.5946  0.7882    4.00    8.00
    leap      8.0   0.4197  0.6798    4.00    8.00
    shadow    3.0   0.5591  0.6350    1.00    4.00
    shadow    6.0   0.5759  0.7103    2.00    2.00
    ...
    mean sustained force   one hand 2.33 N    two hands 4.54 N   (1.95x)
    two hands >= one in 22/23 cells, strictly greater in 16

Figure: `figures/deficit_repair.png` (`make deficit-fig`).

**No human data is involved anywhere in this result.** The wrench requirement is
the specification and the hands are fitted to it. That is the half of the thesis
that does not need the ARCTIC/Dexonomy registration to be answered, and it is
now answered in physics.

### The boundary, which is as informative as the claim

f5d6 grasps a 3 cm box (eps 0.65) and a 4 cm box (0.43) and then fails
completely from 5 cm up -- with ONE hand and with TWO. Its open hand cannot
admit the object at all; every sampled approach is rejected as starting inside
it. So:

**A second identical hand repairs a WRENCH deficit, not an APERTURE deficit.**

Where a hand can reach around the object but cannot oppose well enough to
resist the wrench, the second hand supplies the missing directions and the
sustained force roughly doubles. Where the hand cannot admit the object in the
first place, a second copy of the same hand adds nothing. That distinction was
not in the thesis as written and it sharpens it.

### Caveats carried with these numbers

* The force ladder tops out at 8 N, so cells where both conditions saturate
  understate the gain (leap reaches the ceiling at 6 of 7 widths).
* CEM is stochastic and was run at ONE seed per cell. Shadow at 8 cm is the
  single regression (1.00 -> 0.50 N) and is most likely search variance, not a
  property of the hand.
* Penetration is gated at 3 mm and reported per cell; at kp = 3 the search had
  been finding "grasps" buried 5-19 mm deep carrying 40-310 N, which is how
  epsilon gets manufactured out of buried fingers. At kp = 1 the same
  approaches sit near 2 mm and 16 N.
* Objects are boxes. Two hands of the same type only; no arm, no torso.
* The keypoint-vs-wrench comparison has NOT been redone on this pipeline. That
  half of the thesis still rests on the retracted geometric run and has to be
  rebuilt here.

## 2026-09-12 — review response: two retractions, one control that holds, one null

An external review (`docs/REVIEW_HANDOFF.md`) identified defects in work already
reported here. Two of them invalidate claims. Acting on them in order of damage.

### RETRACTION 1: the BC and action-chunking results are on a task that needs no feedback

The review supplied the baseline this project never ran: record the expert's
action sequence once, then replay it indexed ONLY by elapsed time -- no joint,
object or contact feedback of any kind -- against the standard evaluation seeds.

    seed 9000  peg 13.51 cm   9004  13.43 cm
    seed 9001  peg 13.52 cm   9005  13.59 cm
    seed 9002  geg 13.45 cm   9006  13.46 cm
    seed 9003  peg 13.47 cm   9007  13.49 cm      8/8 success

Open-loop replay averages **13.49 cm and succeeds 8/8** -- better than the BC MLP
(12.84 cm, 3/3 seeds) and level with the expert (13.42 cm).

**Retracted:** every framing of the BC and action-chunking results as evidence
about learned control. A policy that matches an open-loop replay has not been
shown to use its observations. The measurements stand; what they measure is not
feedback. The chunking null in particular is uninformative: a task solvable by
replay cannot distinguish architectures.

The evaluation also hands every policy the expert's calibrated pre-grasp, so it
was never perception-to-action. And `rollout_expert` documents an XY-placement
randomisation it does not implement -- only friction and mass vary.

### RETRACTION 2: f5d6's "holds 0 of 8 widths" was an invalid fixture

The review's diagnostic of the old hold bench: during closure on a 5 cm box the
palm MOVED 1.05 m (from ~[0.676, -0.229, 1.110] to [-0.149, -0.206, 0.456]),
all seven right-arm joints had no actuators, the start state already penetrated
the object, and every saved row shows `clear_start=False` with ~19.64 m of free
fall.

That is a hand falling, not a hand failing to grasp. **Retracted:** "f5d6 is the
only hand that cannot oppose, and the only one that holds nothing." The
opposition-floor MEASUREMENT (3.4 cm) is untouched; the hold column derived
from it is not.

The replacement is already in this file: with non-hand joints frozen, actuators
added (the URDF ships none), and the base able to reach, f5d6 grasps a 3 cm box
at eps 0.65 and a 4 cm box at 0.43, and fails from 5 cm up because its open hand
cannot admit the object. A narrow working range, not an incapacity.

### The two-hand claim survives a force-budget control

The review's objection: two hands bring twice the actuators and more contacts,
so of course they hold more. Normalising the sustained force by the total
contact normal force actually measured:

    hold per newton of contact force    one hand 0.0849    two hands 0.1685
                                        (1.98x, better in 18/23 cells)
    mean contacts                       one hand 9.3       two hands 12.6

Two hands are ~2x better per newton they apply, while bringing only 1.35x the
contacts. The advantage is not the budget and it is not simply more contact
points; it is where the contacts are. That is the claim, and it now has its
control.

### NULL: retargeting the wrench does NOT beat retargeting the pose

The retracted geometric comparison said keypoint retargeting reaches eps = 0 in
11 of 16 cells. Rebuilt in physics -- both conditions closed in simulation, the
placement searched with the same budget for each, only the finger angles decided
differently -- it does not replicate:

    mean force sustained    pose 2.48 N    wrench 1.85 N
    wrench > pose in 7/24 cells, pose > wrench in 6/24, tied in 11
    corr(keypoint error, wrench advantage)  Spearman +0.06

**The thesis as stated is not supported by this experiment.** Where both
conditions produce a grasp, copying the human's fingertip geometry is as good as
optimising the wrench, and slightly better on average. The predicted
anti-correlation -- wrench pulling ahead as keypoint error grows -- is absent
(+0.06).

What survives is narrower and about RELIABILITY, not magnitude: the pose
condition produced no holdable grasp in 8 of 24 cells against the wrench
condition's 5, and the cells where it collapses are the informative ones --
f5d6 at 4 cm (pose eps 0.000 vs wrench 0.434), shadow at 8 cm (0.000 vs 0.519),
allegro at 8 cm (0.175 vs 0.456). Pose fidelity fails completely on some
hand-object pairs where a wrench objective still finds a grasp; it is not
generally worse.

Caveat on the comparison itself: the wrench condition searches one extra
parameter (closure fraction) that the pose condition has fixed by the human, so
it has slightly more freedom. That asymmetry favours wrench, and wrench still
did not win on magnitude.

### Also fixed, from the review

* `WarpVec` passed `nconmax * N` and `njmax * N`; both are PER WORLD (`naconmax`
  is the total). At N=256 that requested 16.8M contacts instead of 65k.
* `WarpVec.reset` flattened its argument and used world 0's state for all
  worlds, silently defeating per-world randomisation, and rebuilt with
  hardcoded capacities rather than the constructor's.
* `warp_parity` imported `warp_fix` as a top-level module; the packaged path is
  `oppdef.sim.warp_fix`.
* The "MJX is unavailable on this Jetson" note should read **MJX-JAX**: the
  cuSolver failure is in the JAX linear-algebra path. MJX-Warp routes through
  MuJoCo Warp, which works here, and is a separate untested candidate.

### Standing limitations this does not fix

The hold test applies 14 translational force directions and gates on
translation and remaining contact. It applies no pure torques and does not gate
on orientation, so its "minimum force" is not a measurement of the full 6-D
epsilon ball. Objects are boxes. Two hands of the same type, no arm, no torso.
CEM is stochastic at one seed per cell.

## 2026-09-12 — the null was my own experiment's fault; the corrected result, and its limit

### The flaw

The "wrench" condition in the first pose-vs-wrench run was not a wrench
objective. It drove the fingers along the hand's single DERIVED CLOSURE with
only a scalar fraction free, while the pose condition received a full 16-DoF
finger specification from the human. One free number against sixteen. NOTES
even recorded the opposite ("the wrench condition searches one extra
parameter"), which was wrong.

That condition is now labelled `closure`, which is what it is, and a real
wrench condition was added: placement AND every finger angle chosen to maximise
the epsilon of the contact set MEASURED in simulation. Free-space joint search
is safe here in a way it was not for the geometric retargeter -- a pose that
buries its fingers is rejected by the penetration gate before it can score.

    Wilcoxon, wrench vs closure (hold force):  p = 0.0498

So the mislabelled condition was materially weaker, and the earlier null was an
artefact of my own parameterisation.

### The corrected comparison (n = 32 cells: 4 hands x 8 widths)

Finer force ladder (1.4x per rung; the doubling ladder had tied 11 of 24 cells
for want of resolution).

    EPSILON (real MuJoCo contacts)   pose 0.3688    wrench 0.4621
      Wilcoxon p = 0.0039                                      SIGNIFICANT

    HOLD FORCE (independent outcome) pose 2.66 N    wrench 2.96 N
      Wilcoxon p = 0.674 ; 12 wins vs 8 losses, sign test p = 0.503   NOT

    outright failures (held 0 N)     pose 12/32     wrench 6/32
      sign test p = 0.238                                            NOT

### What this does and does not establish

Retargeting to a wrench specification produces significantly better contact
sets than retargeting the hand pose, measured in wrench space (p = 0.004).

**That result is circular and must be reported as such.** The wrench condition
optimises epsilon and is then scored on epsilon. The review warned about
exactly this -- "validate predictions on held-out physical outcomes rather than
using the optimized score as its own validation" -- and it is the right warning.

The non-circular test is the hold force, and it is **not significant**
(2.96 vs 2.66 N, p = 0.67). Outright failures halve (12 -> 6) in the right
direction and also fail significance (p = 0.24). So:

**The thesis is supported on the metric it is stated in, and not yet
demonstrated on an independent physical outcome.** The effect direction is
consistent across both independent measures; the experiment is underpowered to
call it.

What would settle it, in order of cost: more cells (widths, shapes, masses,
several CEM seeds per cell) to lift n; a task-level outcome (lift, carry,
insert) instead of a static wrench probe, which is both independent of epsilon
and closer to what a paper claims; and real demonstrations in place of
`SyntheticSource`, without which "retargeting" is still a stand-in.

### Standing, from this session

Supported, with controls:
* epsilon from real contacts predicts physical holding, monotone, ~5x across
  its range (57 grasps).
* A second hand raises sustained force per newton APPLIED by 1.98x (18/23
  cells) while adding only 1.35x the contacts -- allocation, not budget. This
  one is not circular: the outcome is force, the control is force.

Retracted this session:
* the geometric retargeting table (fingers inside the object);
* BC and action chunking as evidence about learned control (open-loop replay
  scores 8/8 at 13.49 cm);
* f5d6 "holds 0 of 8 widths" (the palm fell 1.05 m during closure).

## 2026-09-12 — FINAL: the wrench-over-pose claim does not survive an independent test

The force-only probe was the weaker half of the measurement, and I said so: it
probes forces, epsilon is six-dimensional, and opposition is what buys TORQUE
resistance. A grasp with poor opposition resists forces through friction and
fails on moments, so a force-only probe should miss exactly the effect the
thesis predicts. Confirmed on two known-good grasps -- torque is the binding
direction, not force:

    allegro 5 cm : worst force 1.882   worst torque 1.345
    allegro 8 cm : worst force 0.686   worst torque 0.490

So `hold_of` became a true six-dimensional probe: pure torques about the same
14 directions, scaled by the object's characteristic length, with orientation
gated at 15 degrees -- a check a translation-only test never made.

### Result (n = 32 paired cells, 4 hands x 8 widths)

    6-D worst wrench      pose 1.274    closure 0.995    wrench 1.067
      wrench wins 10, pose wins 11, 11 tied
      Wilcoxon p = 0.808 ; sign test p = 1.000

    epsilon (circular)    pose 0.3688   wrench 0.4621    p = 0.0039
    outright failures     pose 15/32    wrench  9/32

**The thesis's first half is not supported.** Optimising for wrench-space grasp
quality produces significantly better contact sets by its own measure and does
NOT produce grasps that physically resist larger wrenches than matching the
human's finger pose. Tested twice, with the wrench condition correctly
parameterised both times:

    force-only probe   p = 0.674
    6-D wrench probe   p = 0.808

Two independent probes, both null, in the same direction as each other. I am
stopping here rather than running further probe variants: continuing to vary
the outcome measure until one reaches significance is p-hacking, and this
project's whole method is the opposite of that.

### What the per-hand breakdown suggests, as a hypothesis and not a result

    hand      n   pose    wrench   advantage
    shadow    8   1.042   1.349    +0.307
    leap      8   2.861   1.458    -1.403
    allegro   8   1.192   1.281    +0.090
    f5d6      8   0.000   0.181    +0.181

Three of four hands show a small wrench advantage and LEAP shows a large pose
advantage that drives the mean. The f5d6 column is the one worth a second look:
pose retargeting produces **zero** surviving grasps at all eight widths, while
the wrench objective produces some. That is the thesis's conditional form -- the
advantage appears on the hand that cannot oppose -- but with magnitudes near the
noise floor, mostly zeros, and n = 8, it is a hypothesis for a powered
experiment, not a finding.

### Standing conclusions from this session

Supported, non-circularly:
* epsilon from real contacts predicts physical holding, monotone, ~5x across
  its range (57 grasps).
* A second hand raises sustained force per newton APPLIED by 1.98x (18/23
  cells) while adding only 1.35x the contacts. Outcome in force, control in
  force; the objective does not mark its own homework.

Not supported:
* that retargeting to a wrench specification beats retargeting the hand pose,
  on any independent physical outcome tested.

Retracted this session: the geometric retargeting table; BC and action chunking
as evidence about learned control; f5d6 "holds 0 of 8 widths".

## 2026-09-12 — the thesis, in the form it was actually stated

The pooled test was the wrong test and I ran it first. The claim is CONDITIONAL
-- "on hands which cannot oppose like a human hand" -- and pooling four hands
averages that condition away: LEAP (floor 0.80 cm, human-like) contributed
-1.403 to the mean and dominated it.

Tested conditionally, on the deficit hands the thesis is about (opposition
floor > 2 cm: allegro 2.41, f5d6 3.40), against the outcome declared BEFORE the
run -- does the method yield a grasp that survives ANY six-dimensional wrench
disturbance? -- at 65 cells (2 hands x 13 widths x up to 3 masses):

    retarget the POSE      17/65 cells yield a viable grasp
    retarget the WRENCH    30/65

    discordant pairs: wrench-only 14, pose-only 1
    McNemar exact  p = 0.0010

Per hand:

    allegro (floor 2.41 cm)   pose 17/26   wrench 25/26   discordant 9 vs 1
    f5d6    (floor 3.40 cm)   pose  0/39   wrench  5/39   discordant 5 vs 0

**On f5d6, matching the human's fingertip geometry yields a grasp that survives
a wrench disturbance in ZERO of 39 cells.** The wrench objective yields five.
That is the thesis's own extreme case: a hand that cannot oppose like a human
hand gets nothing from being told to hold the object like one.

Where BOTH methods produce a viable grasp they are equally strong (0.544 vs
0.519 N, p = 0.736). So the effect is on WHETHER a grasp exists, not on how
much it holds -- which is the sharper claim and the one the pooled magnitude
test could never have found.

This outcome is independent of epsilon: epsilon is what the wrench condition
optimises, and survival under an applied 6-D wrench is measured afterwards by
the simulator.

### Honest accounting of how this was reached

Four analyses preceded it, all null: pooled force-only (p = 0.674), pooled 6-D
(p = 0.808), deficit-hands magnitude (p = 0.717), and the floor interaction
(Spearman +0.03). The survival outcome was identified as a signal in the third
(4 discordant vs 0), DECLARED as the pre-specified test, and only then run at
3.3x the cells. It held and strengthened (14 vs 1). That sequence is
exploratory-then-confirmatory, and it is reported as such rather than as a
single clean hypothesis test.

The magnitude claim remains null under every test. Anyone quoting this result
should quote it as survival, not as strength.

### What is still not established

* `SyntheticSource` is an analytic stand-in, not MANO. Until ARCTIC/Dexonomy
  lands, "the human pose" is my construction of one.
* Boxes only; two hands of the same type; no arm, no torso.
* One CEM seed per cell. The search is stochastic and a failure to find a grasp
  is not proof that none exists -- which matters more for the pose condition,
  whose finger angles are fixed, than for the wrench condition, which searches
  them.
* Both halves of the thesis now have support, on different experiments. They
  have not been shown to compose: nothing here demonstrates a wrench-retargeted
  bimanual grasp recovering a demonstrated task.

### Both halves, together

    1. retarget the wrench, not the pose (deficit hands, survival)
         30/65 vs 17/65   McNemar p = 0.0010
    2. the second hand repairs the deficit (force held per newton APPLIED)
         1.98x, 18/23 cells, only 1.35x the contacts

Figure: `figures/thesis.png`.

## 2026-09-12 — RETRACTION: the f5d6 model was wrong, and the opposition floor with it

A completion review (`docs/COMPLETION_REVIEW.md`) found that the f5d6 simulated
throughout this project is not the f5d6 in the URDF. I verified every claim
before acting on it. All three hold.

**1. The tracked fingertips were distal JOINT ORIGINS.** Each finger's terminal
joint rotates about the exact point being tracked, so moving it displaced the
tracked position by **0.0000 mm** -- measured, all five fingers. Every objective
reading those points was blind to the last joint of every finger. The URDF
carries real tip frames 27.6-50.0 mm further out (`R_*_tip`); MuJoCo's importer
merges them away because they hang off fixed joints, and nothing here noticed.

**2. The five mimic couplings were dropped.** In the URDF each distal joint
follows its proximal one (x1.13 to x1.35), so the right hand has **six**
independent joints. The compiled model had `neq = 0` and this project actuated
all **eleven** independently -- roughly twice the true freedom.

**3. No actuator force limits**, against 0.5 N*m per finger joint and 1.0 at
the thumb in the URDF.

### What that does to the headline number

Recomputed with the real tips and the mimic coupling enforced:

    f5d6 opposition floor    3.40 cm  ->  1.44 cm

and the ordering this project is built on inverts:

    shadow  0.25      shadow  0.25
    leap    0.80      leap    0.80
    allegro 2.41  ->  f5d6    1.44
    f5d6    3.40      allegro 2.41

**RETRACTED:**

* "f5d6 is the only hand that cannot oppose" -- it opposes better than Allegro.
  The README still says it holds nothing; that is now wrong twice over (the
  hold bench was already retracted as an invalid fixture).
* The claim that the 3.1 cm floor measured in May was "independently replicated
  by a completely different method". Both measurements used the same wrong
  marker convention, so the second replicated the first's error. Agreement
  between two methods that share a defect is not corroboration.
* **The deficit cohort, and with it the McNemar headline.** The cohort was
  "opposition floor > 2 cm" = {allegro 2.41, f5d6 3.40}. Corrected, f5d6 is
  1.44 cm and does not qualify. **39 of the 65 cells behind
  p = 0.0010 were f5d6.** The pooled result does not stand.

What remains of it is Allegro alone: 17/26 vs 25/26, discordant 9 vs 1,
nominal p = 0.0215 -- one hand, and still confounded by the search-budget
defect below. It is a lead, not a result.

### Other confirmed defects in the same result

* **Unequal search budget.** The runner gives wrench `pop * 5 // 4` and pose
  `pop`: **180 versus 144 candidate attempts**, while the adjacent comment
  claims they are equal. The outcome measured is whether search finds a passing
  grasp, so extra attempts go straight to the endpoint. I had named this
  confound in conversation and then shipped the result without checking my own
  code for it.
* **Object geometry mismatch.** `SyntheticSource` describes half-extents
  `[w/2, 0.025, 0.025]`; the comparison builds a cube `(w/2, w/2, w/2)`. Only
  w = 50 mm agrees. Two points declared contacts in the reference sit 12.56 and
  7.44 mm outside even the reference box.
* **The second-hand figure is mislabelled and miscounted.** Its saved
  per-direction arrays are length **14 -- force only** -- while the axis says
  "wrench". The counts are **17 improvements, 5 declines, 1 tie**, not the
  18/23 recorded earlier. The 1.983x ratio reproduces, but dividing by
  `f_total` measured BEFORE the disturbance does not hold the actuation budget
  fixed during it, so it is a descriptive efficiency ratio, not a demonstration
  that allocation causes the advantage.
* **The figure hardcodes its p-value and discordant counts** rather than
  deriving them from the data it plots.
* **"Survives ANY 6-D wrench" overstates the probe.** It tests 14 pure-force
  and 14 pure-torque directions, not arbitrary combined wrenches, and a pass
  means every sampled direction survives at least the smallest rung.

### Status

Standing, unaffected: epsilon from real contacts predicts physical holding
(~5x across its range); the geometric-retargeting, BC/chunking and f5d6-hold
retractions.

Not standing: the wrench-over-pose headline (invalid cohort, unequal budget);
the opposition-floor ordering; the second-hand claim as stated (mislabelled
probe, budget not held fixed during the disturbance).

Repaired in code: `hands/f5d6.py` adds the real tip frames and the mimic
couplings and records the effort limits; `paths.compile_urdf` applies it;
`specs` and `embodiment` now name the real tips. `tests/test_f5d6_model.py`
locks all three defects.

## 2026-09-12 — the opposition axis, re-derived. There is no deficit.

The completion review established that f5d6's fingertips were tracked at distal
joint origins. Auditing the other three hands found the same defect in all of
them: on shadow, leap and allegro the most distal joint of every finger moves
the tracked body by **0.000 mm**. The opposition axis -- the measurement this
project is named after -- was taken at the wrong point on every hand.

### The rule, derived rather than tabled

`hands/tips.py` takes the fingertip to be the point of the distal link's own
collision geometry lying furthest from that body's origin. No per-hand offsets
to maintain, and it works for a capsule, a box or a mesh.

It can be checked against ground truth exactly once, on f5d6, whose URDF names
its own tip frames. The rule recovers them to **0.4 mm**:

    R_ff_l2 -> tip   derived 50.4 mm    URDF 50.0 mm
    R_th_l2 -> tip   derived 27.8 mm    URDF 27.6 mm

### Three further corrections, each found by the previous one failing

* **Thumb-to-MEAN-of-fingertips is not a distance the hand can close.** With
  true tips, shadow, leap and allegro all measured 0.00 cm with zero
  self-penetration: a thumb sitting in the middle of splayed fingers scores
  zero while touching nothing. The floor is now thumb to NEAREST fingertip.
* **Self-collision has to be scoped to the fingers.** An unscoped penalty was
  dominated by a 6.57 mm overlap between `head_l1` and `head_l3` -- the robot's
  HEAD -- plus palm/thumb-base overlaps present at rest. Constant across poses,
  so they penalised nothing and masked everything.
* **Reporting the penalised objective as a distance** gave f5d6 a "floor" of
  34.57 cm, which is a penalty, not a gap.

### The corrected axis

    hand      corrected    as previously reported
    allegro     0.06 cm         2.41 cm
    shadow      0.21 cm         0.25 cm
    f5d6        0.26 cm         3.40 cm
    leap        0.34 cm         0.80 cm

**All four hands oppose within 0.06-0.34 cm of each other -- a spread of
2.8 mm.** f5d6, the hand this project is named around and the one that
motivated the whole thesis, opposes BETTER than LEAP.

### The cohort cannot be defined

The conditional thesis needs hands that "cannot oppose like a human hand". On
the corrected axis no such hand is present, and there is no ordering to
condition on: the spread across four hands is smaller than the measurement's
own sensitivity to how a fingertip is defined. **The deficit cohort is
withdrawn, and with it every result that used it.**

That includes the last standing headline (30/65 vs 17/65, McNemar p = 0.0010):
its cohort was {allegro, f5d6} on floors of 2.41 and 3.40 cm, both of which
were artifacts.

### What replaces it

One budget-matched experiment across all four hands, with the cohort removed
rather than redefined -- if opposition does not vary, the comparison should be
run unconditioned and reported as such. `experiments/matched.py`: both
conditions search the SAME space (placement + every finger angle), with the
SAME candidate count and the SAME seeds, on the same cube, and differ only in
the scalar they maximise. Both are required to produce an actual grasp, because
fidelity alone is maximised by a hand matching the human's shape in mid-air
holding nothing -- which is exactly what the pose condition did on its first
run.

Also corrected in the physical model: mimic-driven joints are no longer given
their own actuators (they fought the constraint that defines them), f5d6 now
has **6** finger actuators rather than 11, and the URDF's torque limits
(1.0 N*m thumb, 0.5 fingers) are applied. `SyntheticSource` now builds its
reference grasp on the cube actually tested; previously two of its five
declared contacts were 12.56 and 7.44 mm outside the box.

## 2026-09-12 — the budget-matched experiment

Cohort removed rather than redefined, because the corrected opposition axis has
no spread to condition on. All four hands, five widths, three CEM seeds.

Both conditions are identical except for the scalar they maximise: the same
search space (placement + every finger angle), the same 144 candidates, the
same seeds cell by cell, the same cube, and both required to produce an actual
grasp. Every confound named in `docs/COMPLETION_REVIEW.md` -- 180 vs 144
candidates, mismatched parameter spaces, a reference grasp whose contacts sat
outside the box -- is closed.

    n = 60 paired cells

    keypoint error       pose  7.96 cm    wrench 12.64 cm
      the pose condition is the more faithful one, as it must be: a manipulation
      check that each objective is in fact optimising what it claims

    epsilon              pose 0.0544     wrench 0.2124    p < 0.0001  (circular)

    hold, 6-D probe      pose 0.025 N    wrench 0.223 N
      wrench wins 19, pose wins 1, of 20 non-tied
      Wilcoxon p = 0.0002    sign test p < 0.0001

    yields a grasp that survives the probe
      pose 3/60    wrench 21/60    discordant 18 vs 0    McNemar p < 0.0001

Per hand:

    shadow    survives  0 vs  0
    leap      survives  3 vs 11
    allegro   survives  0 vs  8
    f5d6      survives  0 vs  2

**Under a matched budget the objective decides the outcome, and the outcome is
independent of the objective's own scorecard.** Optimising the wrench yields a
grasp that survives a disturbance seven times more often than optimising
fidelity to the human's fingertip geometry, and the one metric where pose wins
is fidelity itself.

### What this does and does not say

It is a statement about SEARCH EFFICIENCY toward viable grasps at a fixed
budget, not about the best grasp each objective could reach given unlimited
compute. Both conditions are weak in absolute terms -- the pose condition finds
a surviving grasp in 3 of 60 cells -- and 144 candidates is a small budget for a
29-dimensional space. Shadow, the highest-DoF hand, fails under BOTH conditions
at this budget; that is a power limit, not a property of the hand.

The pose condition is "the most faithful grasp this search found", not
"keypoint retarget followed by a contact refinement", which is what a
practitioner ships and what the review asked for. That baseline is still owed.

The probe tests 14 pure-force and 14 pure-torque directions, not arbitrary
combined wrenches; "survives the probe" means every sampled direction holds at
the smallest rung, and is not the same as "survives any 6-D wrench" -- language
this log used incorrectly before.

Still synthetic throughout: `SyntheticSource` is my construction of a human
grasp, not MANO. And with the deficit cohort withdrawn, nothing here supports
the CONDITIONAL half of the thesis -- that the advantage grows on hands that
cannot oppose -- because no such hand is present among these four.

## 2026-09-13 — follow-up audit response

`docs/REVIEW_HANDOFF.md` was extended with an audit of commit `247a62f`. Its
verdict -- a useful recovery checkpoint, not completion -- is accurate. Acting on
the items that were concrete defects rather than open science.

### The canonical axis command was still reporting the retracted number

The worst finding, and the one I should have caught: `hands/axis.py` kept its
OWN hand table, its own `load()`, its own distance function. The correction had
gone into `derive_flex` only, so `make axis` still used body-origin markers,
thumb-to-finger-MEAN, eleven independent f5d6 joints and no self-collision.
A reader running the shipped command would have got the withdrawn answer.

Fixed by removing the duplication rather than patching it twice:

* `hands/model.py` now answers "where is a fingertip", "which joints are
  coupled" and "is the hand inside itself" once, for every caller.
* `hands/axis.py` is the axis METRIC (floor and aperture, multi-start, with
  provenance) and nothing else; `specs.derive_flex` is closure SYNTHESIS. The
  audit was right that those are different concepts sharing one function.
* `tests/test_axis.py` runs the PUBLIC entry point, and asserts the f5d6 floor
  is not the retracted 3.40 cm, that the floor is free of self-penetration,
  that the registry tips are the ones used, that the mimic coupling is applied
  during the kinematic search, and that self-penetration is finger-scoped.

Output of the corrected command, now with aperture:

    hand      floor_cm   aperture_cm   joints
    leap          0.34        31.95       16
    allegro       0.06        30.44       16
    shadow        0.21        25.18       24
    f5d6          0.26        12.73       11

### The independent variable moved rather than vanished

The audit notes the corrected axis "removes the project's defining independent
variable". On the FLOOR, yes: 0.06-0.34 cm across four hands is no spread at
all. But the aperture does span a range, and f5d6 is the outlier by a factor of
2.4 -- 12.73 cm against 25-32 cm.

That also matches what the physical experiments kept showing and I kept
mis-attributing: f5d6 fails on larger cubes because it cannot ADMIT them, not
because it cannot oppose. "A second identical hand repairs a wrench deficit,
not an aperture deficit" (2026-09-12) was the same observation arriving early
with the wrong name attached.

This is a hypothesis with a plausible mechanism, not a result. It needs the
measure tied to task wrench capability the audit asks for -- a small floor says
the thumb can approach a finger, not that the contact normals, reachable
placements and torque limits support any particular task.

### Seed-stratified, as asked

    seed 0:  4 wrench-only vs 0 pose-only   exact p = 0.1250
    seed 1:  6 vs 0                         exact p = 0.0312
    seed 2:  8 vs 0                         exact p = 0.0078
    pooled: 18 vs 0                         p = 7.6e-06

Identical to the auditor's independent calculation. **The 60 cells are 20
hand x width configurations repeated at 3 optimiser seeds, not 60 task
instances**, and the figure now says so in its own output.

### Provenance and replay

`results/matched_*.json` now carries the commit, dirty flag, package versions,
full argument set, and per row: the selected placement parameters, the selected
finger targets, the per-direction probe array with its force/torque split, a
histogram of rejection reasons, mass, gain and elapsed time. A reader can replay
the chosen grasp without repeating the search.

### Withdrawn figures quarantined

`figures/thesis.png` is now `figures/retracted/thesis_RETRACTED.png` with a
README, and its generator refuses to run and explains why. The live figure has
a checked-in generator, `experiments/fig_matched.py`, which derives its
p-values and counts from the files it plots -- the withdrawn one typed them into
the title, so it could not disagree with its data.

### Still owed, and not claimed

The audit's items 1 (a task-capability opposition measure), 2 (keypoint
retarget followed by a budgeted squeeze -- the baseline a practitioner actually
ships), 7 (the second-hand claim under an enforced budget DURING the
disturbance) and 8 (every original scientific endpoint) are open. The
second-hand figure remains withdrawn rather than fixed. Nothing in this entry
moves the thesis; it makes the repository honest about where the thesis stands.

### Selection, not search: the diagnostic the audit asked for

The audit's sharpest doubt about the matched experiment was that it "mostly
distinguishes which score guides a small stochastic search through a sparse
feasible set" -- i.e. that wrench wins by finding more grasps, not better ones.
The per-run rejection histograms now settle it:

    valid candidates found, out of 144      pose 28.0     wrench 29.2

Both objectives reach the feasible set at the same rate. They then select
differently from it, and the selection is what separates them:

    survives the probe                      pose 3/60     wrench 21/60

So the effect is SELECTION, not search luck. That is a stronger statement than
the one made yesterday, and it is the one the diagnostic supports.

The replayable re-run reproduces the earlier run exactly (same seeds,
deterministic): keypoint error 7.96 vs 12.64 cm, hold 0.025 vs 0.223 N,
Wilcoxon p = 0.00018, survival 3/60 vs 21/60, discordant 18-0, p = 7.6e-06.

What this still is not: 60 independent task instances (it is 20 configurations
at 3 optimiser seeds); an evaluation independent of the contact model the
objective is scored on; or a comparison against the baseline a practitioner
ships, which is a keypoint retarget followed by a budgeted squeeze. Those
remain owed.

## 2026-09-13 — G1: the wrench objective wins, the human data does not

Pre-registered in `docs/G1_PREREGISTRATION.md` (commit `84715fd`), with the
analysis script committed at `58e7326` **before any result existed**. Three
arms, identical budget (144 executor calls each), identical executor and probe.
n = 60 cells: 4 hands x 5 widths x 3 optimiser seeds.

    arm              survives   hold_N     eps   kp_err_cm   valid/144
    pose_squeeze       9/60      0.118   0.0972      7.12       23.9
    eps_synth         21/60      0.223   0.2124     12.64       29.2
    wrench_refine     21/60      0.525   0.1931      9.60       19.8

**H1 (the thesis): supported.** `wrench_refine` beats the pipeline people
actually ship -- keypoint retarget followed by a budgeted squeeze:

    survival    discordant 15 (wrench only) vs 3 (pose only)   p = 0.0075
    magnitude   18 wins vs 6 of 24 non-tied, Wilcoxon           p = 0.0018

**H2 (does the demonstration help?): NO.** `eps_synth` uses no human data
anywhere -- it starts from the hand's own closure -- and matches
`wrench_refine` exactly:

    survival    21/60 vs 21/60, discordant 7 vs 7, McNemar p = 1.0000

The pre-declared decision rule for this branch, quoted from the
pre-registration without modification:

> **H1 significant, H2 not:** a wrench objective beats the shipped pipeline,
> but the human demonstration contributes nothing beyond initialisation -- the
> contribution is the objective, not the retargeting.

### What this settles

The project's thesis is "human hand-object data should be retargeted as a
wrench specification on the object, not as a hand pose". G1 splits that claim
in two and the halves go opposite ways:

* *as a wrench specification, not as a hand pose* -- **supported**, against the
  real baseline this time (p = 0.0075).
* *human hand-object data* -- **not supported**. Generic wrench-space synthesis
  with no demonstration at all does exactly as well. Whatever the demonstration
  contributes, it is not the ability to find a grasp that holds.

Selection, not search, again and more sharply: `wrench_refine` finds the
FEWEST valid grasps of the three arms (19.8 per 144, against pose_squeeze's
23.9) and yet survives more than twice as often (21 vs 9). It is not looking
harder. It is choosing better.

### Exploratory, labelled as such -- not pre-registered

On magnitude rather than survival, `wrench_refine` does beat `eps_synth`
(0.525 vs 0.223 N; 17 wins vs 9 of 26 non-tied, Wilcoxon p = 0.0163). So the
demonstration may buy a *stronger* grasp while not buying a *more reliable*
one. That was not the declared outcome, it is one comparison among several
looked at after the fact, and it needs its own pre-registered test before it
means anything.

Per hand the ordering is inconsistent, which is consistent with H2's null:

    allegro   pose  3   synth  8   refine 12
    leap      pose  6   synth 11   refine  6
    f5d6      pose  0   synth  2   refine  3
    shadow    pose  0   synth  0   refine  0

Shadow fails under all three arms at this budget -- 144 candidates in a 29-D
space -- as it did in the matched experiment. That is a power limit, not a
property of the hand.

### Consequence for the project's framing

The name `opposition-deficit` already encoded a retracted claim (there is no
deficit on the corrected axis). G1 removes the other half: the contribution
that survives is about the OBJECTIVE used to select a grasp, not about
transferring human demonstrations. Reframing is now the honest move, and the
remaining goals (G2-G5) should be re-derived against the claim that actually
holds rather than the one the repository is named after.

What would still rescue the original framing is a demonstration-conditioned
quantity the object-side objective cannot reconstruct -- a required task wrench
TRAJECTORY, not a static grasp. That is G4's task chain, and G1 says it has to
carry the whole weight of the "human data" half of the thesis.

## 2026-09-13 — G2: the demonstration does not specify the task either, and a placement bug I mis-diagnosed twice

Pre-registered in `docs/G2_PREREGISTRATION.md` (commit `00a6a4f`), analysis
committed at `6c88121` before any result existed. Run 1 was uninterpretable and
is recorded as Amendment 1; run 2 is the result.

### Run 2 (320 candidates per arm, objective normalised)

    arm             P1 task   P2 hold_N   margin     eps   valid/320
    pose_squeeze    11/36       0.224      5.61   0.1176      42.3
    task_generic    13/36       0.471     14.25   0.2393      80.3
    task_demo        8/36       0.180     14.77   0.1439      79.4

    H1  task_demo vs pose_squeeze   discordant 4 vs 7    p = 0.549
    H2  task_demo vs task_generic   discordant 2 vs 7    p = 0.180
        P2 magnitude                0.180 vs 0.471 N     p = 0.0026

**H2 is null on reliability and significantly NEGATIVE on magnitude.**
Conditioning the objective on the demonstrated wrench trajectory does not help
and makes the grasp measurably weaker. Both G1's and G2's answers to "does the
demonstration earn its keep" are now no — once for initialisation, once for
specification.

The likely mechanism, offered as a hypothesis: the task margin is a *minimum
over the nominal required wrenches*, which is a narrow target. Optimising it
yields a grasp barely adequate in exactly the demanded directions, while
execution departs from nominal — contact transitions, slip, controller lag —
and there is no margin left for the parts the specification did not name.
Generic epsilon, being worst-case over every direction, keeps that reserve.

### H1 is null too, and the pre-declared rule required explaining why

G1 found wrench objectives beating the shipped pipeline 21/60 vs 9/60 on the
STATIC probe. On the task they do not (13/36 vs 11/36, n.s.). The diagnosis:

    corr(static probe hold, task success)   Spearman +0.141
    corr(epsilon,           task success)   Spearman +0.185
    corr(task margin,       task success)   Spearman -0.157

Group means separate strongly (successes averaged 0.636 N on the static probe
against failures' 0.146 N), but the *ranking* barely transfers. **The static
worst-case probe and a carry task are not the same outcome**, and a result
established on one does not carry to the other. That is worth more than either
null: this project spent weeks optimising against a static probe.

### RETRACTION: "Shadow is a search-power limit" was wrong, twice

Shadow scored 0/60 in G1 and 0/36 here, and both times I wrote that 144
candidates in a 29-DoF space was underpowered. At 320 candidates Shadow
produces **0.0 valid grasps**, and the rejection histogram says why:

    shadow  (8640 candidates)   pregrasp_penetrating  8498  (98.4%)   valid 0
    f5d6    (8640 candidates)   pregrasp_penetrating  8220  (95.1%)   valid 3
    allegro (8640 candidates)   valid 3473 (40.2%)

Shadow is not finding poor grasps. It is never being **placed**: the open hand
starts inside the object and the retraction along the palm→fingertip axis
cannot clear it — unsurprising for a hand whose model includes a forearm and
whose grasp centre sits 25 cm from the palm body. f5d6 is the same at 95.1%.

**Retracted:** every statement in this log attributing Shadow's zeros to search
power. Two of four hands have contributed nothing to G1 or G2, for a reason in
the placement code, and I labelled it a property of the search twice.

This does not change G1's or G2's statistics — McNemar uses only discordant
pairs, and structurally empty cells are ties — but the effective comparison in
both was **allegro and leap**, on 18 cells, not four hands on 36 or 60.

The fix is a placement routine that approaches along a ray from outside the
object's bounding sphere rather than putting the hand's grasp centre at the
object and retracting. That is the next thing to build, and until it exists no
experiment here has evidence about Shadow at all.

### What G2 settles and what it does not

Settles: a demonstration conditioning the *objective* on a required wrench
trajectory does not improve task success, and costs grasp strength. Together
with G1, two independent pre-registered tests say the human demonstration is
not earning its place in this pipeline.

Does not settle: whether a demonstration helps when it supplies something other
than a wrench requirement — contact semantics, which part of the object affords
what, a sequence of subgoals. G2 tested one operationalisation of "specifies
the task", not the idea.

Still synthetic throughout; still two of four hands unable to be placed.

## 2026-09-14 — G3 Part 1: the placement bug, fixed

Two of four hands had contributed nothing to any experiment in this repository.
Shadow produced **0 valid grasps out of 320** and f5d6 **3 of 320**, with 98.4%
and 95.1% of candidates rejected before a finger ever moved. I called that a
search-power limit twice. It was a placement bug, and fixing it took four
attempts, each of which traded one hand against another.

### What was actually wrong -- two faults, not one

**1. `qpos = 0` is not an open hand.** It is whatever zero means in a given
model file. For Shadow it leaves the hand occupying its own grasp volume, so
*no placement existed* that was both clear when open and touching when closed:
wherever the closed fingers could reach the object, the open hand already
overlapped it by 11-35 mm. `hands/axis.aperture_pose` now derives the maximally
open posture the same way the closure is derived -- multi-start, mimic enforced,
self-collision scoped -- and caches it per hand.

**2. The grasp centre is computed at the CLOSED pose**, so for a hand with a
large palm, or a forearm as Shadow's model has, that point lies inside the
hand's own volume; placing the object there puts the object inside the hand.
Penetration is also not monotone in standoff -- f5d6 measured 0.5 mm, then
10 mm, then 26 mm, then 0 as the object passes the fingers -- so a bisection
finds nothing.

### Four attempts, and what each one cost

Recorded because the pattern is the lesson: each fix was a *heuristic standing
in for a search*, and each suited one hand at another's expense.

    deepest clear placement        shadow 32.5%  leap 47.5%  allegro 10.0%  f5d6 0%
    most predicted contacts        shadow  2.5%  leap 20.0%  allegro 22.5%  f5d6 0%
    index the clear set uniformly  shadow 16.7%  leap  0.0%  allegro  0.0%  f5d6 3.3%
    nearest feasible + adaptive open  -- below

The version that works keeps the original rule (grasp centre at the object plus
a searched standoff) and falls back to the ray scan **only when that overlaps**,
taking the nearest feasible placement; and it opens the hand by the SMALLEST
fraction toward the aperture that clears, rather than always maximally. A hand
that needs no extra opening gets none, which is why Allegro stopped regressing.

### Acceptance, measured as the experiments actually run (CEM, 144 candidates)

    hand        3.0 cm        4.5 cm        6.0 cm
    shadow    23/144 16.0%  50/144 34.7%  55/144 38.2%
    leap      84/144 58.3%  81/144 56.2%  89/144 61.8%
    allegro   31/144 21.5%  73/144 50.7%  74/144 51.4%
    f5d6      94/144 65.3%  74/144 51.4%   0/144  0.0%

Best epsilon found ranges 0.36-0.65 for every hand. **Shadow went from 0.0/320
to 16-38%.** The single remaining zero is f5d6 at 6 cm, which is the physical
limit measured independently -- it failed at 5 cm and above consistently, and
its aperture is the narrowest of the four at 12.73 cm. That is a finding, not a
bug.

### What this invalidates

Nothing is retracted by this fix, but G1 and G2 were both effectively
**allegro and leap on 18 cells**, not four hands on 60 and 36. Both are worth
re-running now that all four hands can be placed, and Part 1's acceptance was
the precondition for doing so.

## 2026-09-14 — G3 Part 2: standard grasp metrics do not predict task success

Pre-registered in `docs/G3_PREREGISTRATION.md` (commit `f8257b7`), analysis
committed before any data existed. **158 sampled grasps** across 4 hands x 4
object shapes, each run on two carry tasks with different required wrench
profiles: 316 observations, 102 successes (32.3%).

Grasps are SAMPLED rather than optimised, because the question is whether a
metric ranks grasps across the quality range; a search would truncate exactly
the variation under test.

    metric            Spearman         p    p(Holm)     AUC
    epsilon             +0.074    0.1921     0.7684   0.544
    hold_N              +0.181    0.0012     0.0075   0.545
    margin              +0.004    0.9369     1.0000   0.503
    margin_per_N        -0.027    0.6364     1.0000   0.484
    n_contacts          +0.105    0.0615     0.3077   0.564
    f_total             +0.057    0.3139     0.9416   0.535
    penetration_mm      -0.272    0.0000     0.0000   0.332

**No metric reaches the pre-declared rho = 0.3 with Holm p < 0.05.**

Ferrari-Canny epsilon -- the field's standard grasp metric, and the thing this
project spent weeks optimising -- has **AUC 0.544**. A coin flip is 0.500.

The strongest signal among the seven is **penetration depth, negatively**
(rho -0.272, AUC 0.332): grasps whose fingers press further into the object
fail more often. That is a statement about contact realism in the simulator,
not about grasp quality, and it being the best available predictor is itself
the finding.

The task-conditioned margin -- the quantity G2 built and optimised -- is
`rho = +0.004`. It carries no information about whether the task succeeds.

### This explains the G1/G2 disagreement exactly

G1 established that a wrench objective beats the shipped pipeline **on a static
worst-case probe**. G2 found no such advantage **on a carry task**. G3 says why:
the static probe and the task rank grasps almost independently (`hold_N` at
AUC 0.545), so a result established on one has no reason to appear on the other.

Every optimisation in this repository has been against a target that does not
predict the outcome anyone cares about.

### Not uniform across shapes, which is worth stating

    metric          box    capsule   cylinder   sphere
    epsilon      +0.067    -0.139     +0.016   +0.451
    hold_N       +0.149    +0.295     +0.060      nan

Epsilon is a decent predictor on spheres (+0.451) and slightly inverted on
capsules (-0.139). A metric that works on one object family and not another is
consistent with the pooled null and suggests the failure is about contact
GEOMETRY rather than about wrench-space reasoning in general.

### Limitations, stated rather than discovered later

158 grasps at one object size and one mass, four hands, two task variants of a
single carry family, simulation only. **f5d6 contributed 4 grasps** -- one per
shape, each cell hitting the 400-draw cap -- so it is effectively absent and
its column is undefined. Sampled grasps are low quality on average (32.3% task
success), so the range tested is wide but bottom-heavy. epsilon, the margin and
the static probe share a contact model with one another; task success does not,
which is the entire point of the comparison.

A null here is a statement about **these metrics on these tasks**. It is not a
proof that no grasp metric can predict manipulation success.

Figure: `figures/g3_metrics.png` (`make g3-fig`); statistics derived from the
saved rows, not typed in.

## 2026-09-14 — G3 re-collected: the earlier null was three bugs, not a finding

The first G3 dataset is **withdrawn**. A codex review plus G4's own smoke run
found three defects, each of which suppressed real signal:

1. **`run_task` leaked gravity.** `GraspScene` forms grasps weightless on
   purpose; `run_task` switched gravity on for the carry and never handed it
   back. Within a cell, grasp #0 was formed weightless and every grasp after it
   under gravity. Saved grasps did not re-form.
2. **Task B had no rotation at all.** `task_spec` moved the tilt into
   `cmd[:,4]`, but `object_path` and `base_command` both read the angle out of
   `cmd[:,3]` regardless. The "second wrench profile" demanded **exactly zero
   torque** — it was one task at two speeds. `Trajectory` now carries its
   `tilt_axis`: task A peaks at 0.01134 about x, task B at 0.03065 about y.
3. **`hash()` is salted per process**, so the per-cell seed did not reproduce
   the dataset across runs. Replaced with a SHA-256 digest.

Inference is now clustered **by grasp**: task A and task B share a grasp, so
316 observations are 158 clusters of 2, and ordinary p-values on the duplicated
rows understate the variance. Spearman and AUC carry grasp-level bootstrap CIs.

### The clean result (238 grasps, 4 hands x 4 shapes, 2 carry tasks)

    metric              rho          rho 95% CI   p(Holm)     AUC      AUC 95% CI
    f_total          +0.325     [+0.228,+0.407]    0.0000   0.800  [0.730,0.857] *
    hold_N           +0.297     [+0.097,+0.472]    0.0040   0.606  [0.531,0.689]
    margin           +0.274     [+0.162,+0.381]    0.0000   0.753  [0.656,0.843]
    epsilon          +0.203     [+0.087,+0.323]    0.0000   0.684  [0.582,0.787]
    margin_per_N     +0.177     [+0.080,+0.271]    0.0000   0.664  [0.575,0.747]
    n_contacts       +0.138     [+0.027,+0.243]    0.0240   0.625  [0.525,0.720]
    penetration_mm   -0.136     [-0.261,+0.008]    0.0627   0.375  [0.255,0.508]

**The earlier headline is withdrawn.** It read "no metric predicts task
success, epsilon at AUC 0.544". On the repaired bench epsilon reaches
**AUC 0.684** with a CI excluding chance, and the task margin — which the
contaminated run put at rho +0.004 — reaches **0.753**. The null was three
bugs.

### What the clean data does say

Exactly one metric crosses the pre-declared rho = 0.3: **total contact force**,
at rho +0.325 and **AUC 0.800**, the best predictor of the seven. The strongest
single indicator of whether a grasp completes a carry is **how hard the hand is
squeezing** — not where its contacts sit. Epsilon, the field's geometric
quality measure, is genuinely informative but clearly weaker.

That is a deflating result rather than a triumphant one, and it should be
reported that way. It also sits awkwardly beside G2, where the arm that learned
to squeeze 3.6x harder did *worse* — but G2 ran on the broken task B, so that
comparison is not currently worth anything either.

### Geometry interaction, unchanged by the repair

    metric              box   capsule  cylinder    sphere
    epsilon          +0.261    -0.176    +0.185    +0.226
    margin           +0.355    -0.097    +0.317    +0.239

Every metric inverts on capsules. Whatever the metrics capture, it does not
transfer across object geometry, and that survived the bug fixes — which makes
it the most robust observation here and the obvious next hypothesis.

### f5d6 forms grasps and completes nothing

60 grasps, **0 task successes on either task**. Its per-hand correlation is
undefined because there is no variation to correlate. The placement repair let
it grasp; it still cannot carry. Given its aperture is the narrowest of the
four (12.73 cm) that is plausible, but it is unexplained and should not be
quietly pooled away.

### Standing caveat

This is one object size, one mass, four hands, two variants of a single carry
family, in simulation. Whether the task outcome is even reproducible is exactly
what G4 Part A is testing; until that returns, none of the above is safe.

## 2026-09-14 — G4 Part A: the task outcome is reproducible (gate PASSES, with one caveat)

Pre-registered in `docs/G4_PREREGISTRATION.md` (`61a06df`), Amendment 1 recorded
after run 1 tripped its own determinism gate. 228 grasps x 2 tasks, 5 perturbed
repeats each (±1 mm object offset, ±2% mass as an equivalent steady force,
warm-start reset) plus an unperturbed determinism check.

    PRIMARY   unanimity of the 5 repeats   423/456 = 0.928
              Wilson 95%                   [0.900, 0.948]
              grasp-clustered 95%          [0.897, 0.956]

    SECONDARY majority vote vs G3's saved outcome   454/456 = 0.996

**PASS.** The pre-declared gate was 80% with a lower bound above 0.75; the
clustered lower bound is 0.897. The task outcome is a stable property of the
grasp, so G3's correlations describe something real rather than noise.

Unanimity is lowest where grasps are marginal: leap 0.842 and allegro 0.865
against shadow 0.992 and f5d6 1.000 — and f5d6's perfect score is because it
fails every task, which is consistency without information.

### The determinism gate is NOT met, and the reason is worth recording

    unperturbed repeat vs G3's saved outcome   444/456 = 0.974   (gate: 0.99)

Of the 12 disagreements, 10 were grasps failing to re-form and 2 genuine flips.
Run 1 blamed the mass perturbation, which was real: mutating `body_mass` needs
`mj_setConst` and restoring the scalar did not restore everything. Amendment 1
removed that. The residual has a different cause, and the check that found it
is worth stating because it reversed my assumption:

Re-running the failing pairs with a **fresh scene per repeat** made them fail
**consistently** (no grasp at all), where a reused scene made them alternate.
So those grasps do not re-form standalone; they formed in G3 only because that
cell's scene carried accumulated state from earlier draws. **5 of 228 G3 grasps
(2.2%) are not independently reproducible.**

### Robustness: excluding them changes nothing

    metric          rho all   rho kept   AUC all   AUC kept
    f_total          +0.325     +0.320     0.800      0.795
    margin           +0.274     +0.268     0.753      0.747
    epsilon          +0.203     +0.198     0.684      0.679
    hold_N           +0.297     +0.300     0.606      0.608

Kept-only clustered CIs: f_total AUC [0.726, 0.854], epsilon AUC [0.568, 0.781].
Every conclusion in G3 survives. The gate failure is real, bounded at 2.2%, and
demonstrably immaterial to the result it was guarding.

### Where that leaves the project

G4 was set up as a go/no-go on the whole line of work, and it says **go**. The
outcome being measured is stable, so the G3 finding stands: contact force is
the best predictor of task success (AUC 0.800), epsilon is real but weaker
(0.684), and every metric inverts on capsules.

That capsule inversion is now the most robust unexplained observation in the
repository — it survived a gravity leak, a dead task axis, an unstable seed, and
a reproducibility audit. It is the obvious next question, and it points at
contact geometry rather than scalar wrench summaries.

---

## 2026-09-14 — real human data: GRAB + MANO, read, retargeted, and put in physics

The human-data gate is closed. Everything below was measured on this machine.

**MANO without chumpy.** The official `mano_v1_2` pickles are Python-2 chumpy
objects and chumpy does not install against modern numpy. They are unpickled
with stub classes that keep the payload and discard the wrapper, then cached as
npz (`src/oppdef/human/mano.py`). LBS and the pose correctives are implemented
directly. Checks: the rest pose reproduces the template to **0.0000 mm**, the
skinning weights partition unity to 1.0000, and the kinematic tree comes out as
five chains of three off the wrist.

**The reference is correct, and this is not assumable.** On `s1/mug_drink_1`
the minimum distance from any right-hand vertex to any mug vertex falls
1281 → 53 → **0.5 mm**, stays in **0.1–0.6 mm for 48 frames**, then recedes to
1265 mm. The left hand never comes closer than 534 mm, which is what the
"drink" intent should look like. Sub-millimetre contact for the whole grasp
requires the MANO fit, the subject template, `flat_hand_mean`, the pose
convention and the object transform to be simultaneously right.

**MuJoCo collides meshes as convex hulls, and for GRAB that is fatal.**
Measured hull-to-mesh volume ratios: mug **3.52×**, airplane **3.51×**, body
6.56×, banana 2.00×, binoculars 1.60×, apple 1.05×. The mug's hull fills the cup
*and* the handle's hole — every handle grasp GRAB records would be impossible
against it. CoACD brings the mug to 0.92× and the airplane to 0.82×; cached on
the mesh hash so a changed threshold cannot silently reuse a stale
decomposition. Without this, a correctly placed hand read as 22.6 mm mean
penetration; with it, 14.3 mm, of which the rest was real.

**Three bugs, each of which returned a plausible wrong answer rather than an
error.** Recorded because each was found by a measurement, not by reading code:

1. `mj_jacBody` reads `d.cdof`, which `mj_kinematics` does **not** fill. The
   Jacobian was identically zero, every Gauss-Newton step was zero, and the fit
   returned its seed pose while reporting 1324 mm of error as though it had
   converged. `mj_comPos` fixes it.
2. Retargeting in GRAB's world frame is impossible: the object sits 0.8–1.7 m up
   and the floating base has 0.6 m of travel. The fit pinned itself against its
   limits at **486 mm**. Re-expressed in the object frame — which is the frame
   the result is consumed in — the same solver reaches **6.7 mm**.
3. Penalising *fingertip* contact as penetration fights the contact targets
   directly: it bought 4 mm of penetration for 27 mm of contact accuracy
   (6.7 → 33.6 mm). The penalty now excludes fingertip geoms; at w_pen = 2.0 the
   trade is 12.9 mm contact error for 3.9 mm penetration. w_pen = 15 is worse on
   **both** — the steps destabilise the Gauss-Newton solve.

**Retargeting, validated per finger** (Shadow, `mug_drink_1`, hold window):
index 12.5 mm, middle 10.4 mm, ring 14.2 mm, thumb 14.5 mm, pinky 22.3 mm.
The pinky is the worst and is in contact in only 13.8% of frames, so it is
mostly unconstrained — the ordering is the one the grasp implies, not a defect.
Correspondence confirmed thumb→`rh_thdistal`, index→`rh_ffdistal`.

**First look at holding.** A three-frame sample suggested the grasps were
failing — two of three showed ~14 m of displacement. That is not a drop, it is
**free fall** (½·9.81·1.75² = 15 m) with zero contacts. Across the full 119-frame
window the retargeted poses make contact in **117/119** frames (mean 25.7
contacts) and **16 of 20** sampled frames hold the object for 1 s. The failures
are concentrated in the frames where the human hand is still closing. This is
why G5 samples across the window and clusters by sequence rather than taking one
frame per clip.

**Dataset.** 198 `s1` + 93 `s2` sequences, 51 objects. The bimanual subset is
real: sequences whose intent is `offhand` (hand-to-hand transfer) have both hands
in contact simultaneously, and the full inventory is in
`results/grab_inventory.json`.

**Feedforward is exact.** Composing the retargeted object-frame grasp with the
reference trajectory and solving the six base DoF by IK reproduces the fingertip
targets to **0.000 mm**, with the fingers frozen at their retargeted values so
the grasp shape cannot drift.

**CORRECTION — the jumpy commands were my solver, not the base's Euler chart.**
An earlier version of this entry attributed a 0.70 rad frame-to-frame jump in
the base command to gimbal in the three base hinges, on the evidence that
constraining the solve to stay near the previous frame traded the error straight
back (max_step 0.35 → 171 mm residual). That inference was wrong. The jump is
present in the **retargeted palm pose itself**, before any base
parameterisation: measured in the object frame, the fitted palm moved 157 mm and
0.715 rad between frames while the human's own wrist moved at most 52.3 mm, and
an SE(3) feedforward with no IK at all reproduced it (0.825 rad).

The actual cause is **non-convergence in the retargeting solve**. Five fingertip
targets are fifteen constraints on twenty-nine DoF, so there is a
fourteen-dimensional null space; stopped at twelve iterations each frame halts
wherever its trust-region path reached, and consecutive frames land in different
parts of it. Raising `iters` to 80:

    iters   contact err   palm turn max   palm step max
       12       16.6 mm       0.715 rad        157.3 mm
       80       16.7 mm       0.275 rad         52.6 mm

— a strict improvement, at no accuracy cost, and the palm now moves like the
human wrist it was fitted to (52.6 mm against 52.3 mm). With the converged fit
the *hinge* base has **zero** steps above 0.3 rad, which is the check that
settles it. Intermediate iteration counts are not monotone (60 was worse than
either), because with a trust region the path matters and not only the optimum.

The nearest-surface-point targets were checked and exonerated: they jump no more
than the human's own fingertips (28.8 mm against 30.4 mm).

**The SE(3) wrist was kept anyway**, on its own merits rather than as this fix.
`scene.build(base="mocap")` puts the hand on a free joint driven by a welded
mocap body, commanded as a pose. Three serial hinges genuinely do lose a degree
of freedom at gimbal lock, a tracking controller should be producing SE(3)
rather than Euler angles, and the transfer between the two representations is
exact: the mocap scene reproduces the hinge-base fit to **0.0000 mm** on both
the palm and all five fingertips.

## 2026-09-14 (later) — the retarget is not a grasp, and the measurement that shows it

Building the tracking stage turned up a defect in the retargeting that
invalidated every physical number taken from it, and then a finding that is not
a defect at all.

**The fingertip was the body origin again.** `oppdef.hands.tips` exists because
all four hands in this project once tracked a distal body's ORIGIN as its
fingertip; the new retargeter did the same thing. The derived tip lies **32–36 mm**
beyond the origin on Shadow, so placing the origin on the object surface buries
the tip geom by its own radius. Consequences, measured in the tracking scene:

    tip targeting     initial penetration     total contact force (0.2 kg mug)
    body origin              12.7 mm                     4469 N
    derived tip               0.0 mm                     64–85 N

4469 N is 450 kgf on a coffee mug. Every rollout taken before this fix was
measuring the constraint solver relaxing that, not a grasp. Fingertips are also
no longer *exempt* from the penetration penalty but given a 2 mm allowance: a
fingertip on the surface is the grasp, a fingertip 12 mm inside is not.

**A kinematic retarget cannot be a grasp, and both horns were measured.** With
the tips correctly on the surface a position servo is already at its target and
applies no force at all: the object free-falls from frame 0 (302 m over 7.9 s,
which is exactly ½·9.81·7.9²). Burying them to get force instead gives the 4469 N
above, and merely letting that relax ejects the object — it moved 28 mm and lost
every contact during a 0.2 s settle with gravity off.

So the grip has to be built, not fitted, which is what grasp synthesis in this
repository already does (pre-grasp, close, squeeze). Three things that had to be
right before closing did anything, each found by measurement:

1. Closing toward the derived closure POSTURE moved the fingertips *away* from
   the object, 134 mm → 209 mm. A hand wrapped round a mug is already more
   flexed than its generic closure, so the blend opens it. The closing
   *direction* — the sign of each actuator's own derived closure — is the right
   quantity.
2. That direction must exclude the wrist. Shadow's `WRJ1` sits on the palm body
   itself, so driving it "toward closure" swings the whole hand off the object:
   closing to 1.21 rad produced 0.10 N because the fingers were being carried
   away faster than they closed. The finger set is selected by body descent.
3. `apply()` rewrites `d.ctrl` from the feedforward every frame, which silently
   discarded the squeeze that had just been established.

With all three fixed, closing reaches **81.6 N across 14 contacts** — and the
object is still lost on the first frame, because the contact set is one-sided:
the squeeze that generates enough force to hold against Shadow's weak finger
servos (gains 0.4–1.5 N·m/rad) also extrudes a rigid object. **This is the gap
the tracking stage exists to close**, and it is not closable by a kinematic fit
plus a heuristic squeeze. MPPI over feedforward corrections does not close it
either: 24 samples at horizon 4 moved the hold from 22 to 25 frames of 119,
with corrections of 9 mm and 0.16 rad — local search cannot recover an object
that has already been released.

**Open-loop feedforward, for the record.** Before the tip fix, the feedforward
carried the mug for 27 of 119 frames: contacts fell 25 → 8 → 0 as the reference
accelerated from 3.7 to 17 mm/frame, with the object rotating 45° in the hand
first. The hand was also commanded to the REFERENCE while the object lagged
behind it, so the palm–object gap grew 133 → 167 mm — a separate defect, fixed
by blending toward the object's actual pose, which on its own did not save the
grasp either.

**Control rate.** One control frame must last one reference frame. Hardcoding
`ctrl_every=10` against a 15 Hz clip ran the hand at 50 Hz — 3.3× ahead of the
object — and threw it 13 m. Now derived from `seq.dt / timestep`.

**`MocapHand.command` snapshotted `qpos` after zeroing the free joint**, so
restoring it teleported the hand to the origin on every control step. 253 m of
"tracking error" was the hand leaving, not the object.

## 2026-09-14 (later still) — CORRECTION: tracking works; I was starting it in the wrong place

The entry above concludes that "a kinematic retarget cannot be a grasp" and that
the tracking gap "is not closable by a kinematic fit plus a heuristic". **The
first claim is wrong and the second is too strong.** Both came from starting
every tracking episode at window frame 0.

The hold window begins where the human's hand first comes within 5 mm of the
object — the moment contact *starts*, not the moment the grasp is *formed*.
Started there, the hand is still closing, so it drops the object within a frame,
and that reads as "tracking failed" when nothing has been tracked yet.

Measured on `mug_drink_1` (Shadow), holding each frame's retargeted pose for 1 s:

    window frame   10    30    50    70    90
    hinge base    drop  drop  HELD  HELD  HELD
    mocap base    drop  drop  HELD  HELD  HELD

Identical in both scenes, so the mocap/weld env was never the problem either —
another thing the earlier entry suspected. **17 of 24 sampled frames hold the
object on their own.**

Starting from a frame that holds, the open-loop feedforward tracks the rest of
the reference:

    start frame   frames kept        mean position error
        35          24 of 84            (drops)
        60          59 of 59              11.9 mm
        75          44 of 44              14.4 mm

And MPPI closes the marginal case rather than failing to:

    start 35   feedforward  18608 mm mean, drops at frame 59
               MPPI            53.0 mm mean, tracks to the end
    start 20   both fail — correctly; at frame 20 the hand has not closed yet

So the honest picture is: a retargeted pose IS a grasp over most of the hold
window, open-loop feedforward tracks from one, and MPPI rescues the marginal
starts. What remains true from the earlier entry is the fingertip-origin defect
(4469 N → 64 N), the control-rate bug, and the `MocapHand.command` snapshot bug
— those were real and are fixed. What was wrong was the conclusion drawn while
every episode was being started before the grasp existed.

---

## 2026-09-14 — **RETRACTION: the object's rotation was transposed.** Everything downstream is void.

Found because the GIF looked wrong to a reader and I was asked to look again.

**The bug.** GRAB's own `tools/objectmodel.py` poses the object as

    vertices = torch.matmul(v_template, rot_mats) + transl     # v @ R

which is `R.T @ v` — the transpose of the usual convention. I wrote `v @ R.T + p`.
Every object pose in this pipeline was therefore rotated the wrong way.

**Why nothing caught it.** Every check was self-referential: reconstructed hand
against reconstructed object. Both were wrong together, so they stayed in
contact, and the headline validation — minimum hand-to-object distance falling
to 0.1–0.6 mm — passed the whole time. A minimum over 778 vertices is satisfied
by one grazing vertex; I later replaced it with contact AREA (100–145 vertices
within 5 mm), and that passed too, because the hand was genuinely touching the
mug — just in the wrong place, its body instead of its handle.

**The external ground truth I had not used.** GRAB ships
`contact['object']`: per frame, per object vertex, which body part touches it.
Scored against it, on `mug_drink_1`:

    object transform      recall   IoU     contacts on the handle
    v @ R.T + p  (mine)    0.158   0.045            4.8%
    v @ R   + p  (GRAB)    0.847   0.456           76.6%

GRAB's labels put 41–100% of that clip's contacts on the handle, which is what
"drinking from a mug" looks like and what the render did not show. Fixed at the
boundary: `GrabSequence.obj_R` now stores the PROPER rotation matrix, converted
once at load, and `to_object` / `object_world` are the only two places the
convention appears. Across 8 sequences the fixed pipeline scores **recall 0.919,
minimum 0.846**.

`experiments/grab_validate.py` now scores the reconstruction against GRAB's
labels, so this class of error cannot pass silently again.

**What this retracts.** Everything computed through the object frame:

* the **GRAB inventory** (`results/grab_inventory_TRANSPOSED.json`) — contact
  masks, hold windows, and the 136/291 bimanual count;
* **G5 in full** (`results/g5_hold_TRANSPOSED.json`) — the 0.255 hold rate, both
  hypotheses, and the decision rule I drew from it. G5's *pre-registration*
  stands; its result does not;
* every **retargeting** number (13 mm contact error, 32–36 mm tip offsets are
  fine — those are robot-side — but the targets they were fitted to were wrong);
* every **tracking** number, including the 11.9 mm feedforward result and the
  MPPI 53.0 mm rescue;
* the claim that the mug is grasped by its **body**. It is grasped by the
  handle. My "5.6% on the handle" measurement was computed under the transposed
  transform.

**What survives**, because it never touched the object frame: the MANO loader
and its rest-pose check; the convex decomposition (hull ratios are properties of
the meshes); the fingertip-origin fix (4469 N → 64 N is robot-side); the
control-rate and `MocapHand.command` bugs; and the left-hand registration.

**The lesson, and it is the second time in this project.** A measurement that
compares a reconstruction against itself validates nothing. The opposition
deficit died the same way — every hand measured at its own body origin — and
`hands/tips.py` was written to stop it. Here the external check existed, shipped
in the dataset, and I did not look for it until a picture forced me to.

## 2026-09-15 — stages 3-7 on the corrected pipeline

Re-measured after the transposed-rotation retraction. All numbers below are from
the corrected transform.

**Reference pipeline, validated externally.** `experiments/grab_validate.py`
scores the reconstruction against GRAB's own per-vertex contact labels, over all
291 sequences: **recall mean 0.840, median 0.890, 282/291 above 0.5**. Restricted
to hand labels, because GRAB labels face and torso contacts too (16 and 23 sit
42–50 mm from either hand; 27 and above sit 2–4 mm) and a hand model cannot
cover them. The unrestricted figure is 0.830.

The corrected inventory reproduces **136/291 bimanual (46.7%)** — the same
headline as the transposed run, by coincidence. 192 of 291 sequences have a
different right-hand hold length and 14 bimanual flags flipped.

**Stage 3 is better than the retracted numbers**, because the retargeting
targets are now right: 21 of 24 sampled frames hold the object unaided (was 17),
and the open-loop feedforward tracks `mug_drink_1` for **116 of 116 frames at
35.3 mm** from frame 0 — the whole reference, not a suffix. From frame 65 it is
14.9 mm with every frame inside 50 mm.

**Stage 4 works.** The homotopy walks λ = 0.35 → 0.70 → 1.0 holding at every
level (50.9 → 51.7 → 52.9 mm) on a start the feedforward drops. Its value
against plain MPPI is untested: on this reference MPPI alone already succeeds,
so the curriculum matches rather than beats it.

**Stage 6 is built and does not work.** Two hands, one physics scene, dynamics
driven through the welds. Every sequence tried loses the object. Two distinct
causes, separated by measurement:

* on `gamecontroller_play_1` the hands **interpenetrate** — 42 hand-hand
  contacts at 11.7 mm — and the left pushes the right off the object entirely
  (0 object contacts for a hand the human had on it). Each hand is fitted in its
  own single-hand scene, so neither solver sees the other;
* on `camera_takepicture_2`, `binoculars_see_1` and `bowl_drink_1` there is **no**
  inter-hand contact at all, and the object is still lost — the same grasp-quality
  problem as one hand.

Separating the hands along the contact normal fixes the first and makes tracking
worse (105 mm → 255 m): clearing the other hand also breaks the grasp. So the
poses are mutually inconsistent, not merely overlapping, and the real fix is to
fit both hands against each other. `BimanualScene.side_view` exposes that (it
folds the other hand's geoms into the obstacle set); the solver cannot yet reach
the wrist DoF in that scene. **Recorded as not working rather than reported as
built.**

**Stage 7 already discriminates.** Frozen controller, only its pose input
swapped, scored on the truth throughout:

    pose the controller reads      pose error   tracking   outcome
    simulator ground truth               --      101.5 mm   held
    ICP on rendered depth            38.4 mm     637.3 mm   DROPPED
    ground truth + Gaussian noise    61.8 mm     102.5 mm   held

Noise **larger** than the estimator's error costs nothing; the estimator's own
error loses the object. Its error is therefore structured — biased and drifting
— and the fix is perception, not control. Neither of the other two rows can
distinguish those, which is why the noise condition was included.

Three perception bugs, each found by measurement: ICP must correspond
cloud→model (matching a full mesh onto a one-sided cloud drags it until it
straddles the visible face: 26.9 mm seed → 43.7 mm); model points must be
sampled over mesh FACES, since a convex part carries 64 vertices and ~12 mm
spacing caps the achievable pose at ~28 mm; and MuJoCo puts the eye at
`lookat - distance * forward`, so azimuth and elevation say where the camera
LOOKS, not where it sits — asking for (0.3, −0.3, 0.3) gave (−0.28, 0.32, −0.30).

## 2026-09-15 (later) — what the seven stages actually do

**G5, re-run twice on the corrected pipeline.** Pre-registered threshold 0.50:

    retarget objective          hold rate   clustered CI        verdict
    fingertips only (W_JOINT=0)   0.318     [0.270, 0.367]       FAIL
    + intermediate joints (0.45)  0.200     [0.161, 0.238]       FAIL

H2 unsupported in both (+0.001 and +0.024, intervals spanning zero), so the
"the window starts before the grasp is formed" explanation I drew from one
sequence does not generalise.

**Matching the human better makes the grasp worse.** This is the session's
sharpest result and it arrived by accident, from fixing something a reader
spotted in a render. Adding the intermediate finger joints as targets moves the
retarget onto the mug's handle the way the human holds it -- handle contacts
31.8% → 38.8%, distance to the human's own contact points 19.6 → 13.1 mm -- and
costs 12 points of hold rate, with non-overlapping intervals. The gap survives
grasp synthesis (0.500 → 0.750 from tips-only against 0.250 → 0.650 from the
wrap-aware fit), so the repair stage does not absorb it. `W_JOINT` defaults to 0
and the trade-off is recorded as the finding: it is this project's founding
claim arriving from the other direction.

**Grasp synthesis is what unblocks the pipeline**, exactly as G5's decision rule
said it would. Searching a small neighbourhood of the retargeted wrist for a
pose that holds -- scored in physics -- takes 4/24 to 19/24. Closing the fingers
without repositioning does nothing (0.288 → 0.276): the failures make 0.1
contacts and have nothing to close on. Applied as a constant wrist offset in the
object frame and kept only when it improves the whole reference; unguarded it
took one clip from 6 graspable frames to 1. Guarded: 16 → 38 graspable frames
over six references, never worse.

**Stage 5 runs end to end and does not generalise.** 16 references attempted, 5
solved, 632 transitions over 5 objects, held out by object:

    train              MAE 0.00297   ratio to predict-the-mean 0.122
    held-out objects   MAE 0.04648   ratio to predict-the-mean 1.840

A ratio above 1 means the policy is worse than a constant. With three training
objects that is what should happen, and it is reported rather than buried: the
statement is about data scale, not about the method. The collection also ran
with W_JOINT = 0.45, which is now known to be the worse setting, so the 5-of-16
solve rate is a floor.

**Stage 6 still does not track.** Joint bimanual fitting works -- fitting both
hands in one scene takes inter-hand contact from 42 at 11.7 mm to 0.1 at 2.4 mm,
and needed a hinge-base variant of the bimanual scene because a free joint gives
the solver no wrist DoF to move. Tracking still fails on every sequence tried,
for the one-handed grasp-quality reason. Grasp synthesis is what it needs next
and has not been applied to it.

## 2026-09-15 — stage 6: the two-handed grasp is fixed; tracking is partly

Applying the same physics-scored wrist search to both hands, alternating one at
a time (a joint 12-D search needs far more samples for the same coverage, and
every sample costs a rollout). Scored by a two-handed `hold_test`:

    sequence              object          hold drop        tracking error
                                       before -> after    before -> after
    gamecontroller_play_1 gamecontroller  981.6 -> 3.40 cm   97 m -> 151 m
    camera_takepicture_2  camera            1.3 -> 0.13 cm  188 m -> 131 m
    binoculars_see_1      binoculars      232.5 -> 0.05 cm   21.8 m -> 0.19 m
    bowl_drink_1          bowl              7.3 -> 0.24 cm  160.9 m -> 0.12 m

**All four now hold.** That was the blocking failure, and it is gone: every
sequence tried went from losing the object outright to holding it within
0.05–3.4 cm. Two of the four then track to 120–190 mm, against 21 and 161
METRES before — three orders of magnitude. The other two hold and still lose the
object during the motion, which is the one-handed pattern: a grasp that survives
gravity need not survive inertia, and that is what the MPPI correction is for.
It has not been applied to the two-handed case.

The prerequisites were two structural fixes, both measured rather than assumed:
fitting both hands in ONE scene so each sees the other (inter-hand contact 42 at
11.7 mm -> 0.1 at 2.4 mm), which in turn needed a hinge-base variant of the
bimanual scene, because a free joint leaves the solver no wrist DoF to move.

## 2026-09-15 — stage 5 fixed by changing the LABEL, and PPO implemented

**Distillation works once the target is a function of the state.** Distilling
MPPI's per-step correction cannot work and the reason is measurable: regressing
it on the observation gives a linear R^2 of **0.075 in sample**. A sampling
optimiser's correction is dominated by its own noise draw. Labelling the
ABSOLUTE command instead -- palm pose in the object frame plus finger targets --
gives a target that is a function of the state by construction:

    label                      train ratio   HELD-OUT OBJECT ratio
    MPPI correction               0.265             2.235
    absolute command              0.044             0.734

Ratios are against predict-the-mean, so below 1 is better than a constant.
13 episodes, 1525 transitions, 11 objects, held out by object. The correction
label was worse than a constant on unseen objects; the absolute label
generalises. Same data, same network, same split -- only the target changed.

This is also the clearest statement of why DexTrack distils an RL policy rather
than a trajectory optimiser's output, and it was worth getting wrong to find.

**PPO is implemented** (`human/rl.py`): pooled environments over one shared
model, Gaussian policy initialised near zero so it starts as the feedforward
and only has to learn the correction, GAE, clipped objective, dense bounded
reward with an alive bonus so that dropping the object is not the cheapest way
to end an episode.

**It does not yet beat the feedforward, and the reason is budget.** 40
iterations x 40 steps x 10 envs is 16,000 control steps in 445 s -- about 36
control steps a second, each of which is 33 MuJoCo steps. The policy came out at
10412 mm against the feedforward's 8197 mm. DexTrack trains on millions of
steps; 16k is not a fair test of PPO, it is a test of 16k steps. A longer run is
the next measurement, and the honest cost on this machine is roughly 8 hours per
million control steps.

## 2026-09-15 — stage 6 now tracks: the rollout was never commanding the fingers

`BimanualTracker.rollout` set mocap targets and never touched `d.ctrl`. All 40
actuators sat at zero, so the position servos drove every finger to its OPEN
configuration the instant stepping began. The hands held the object at reset
only because `place` writes qpos directly, and then let go. Nothing in the
two-handed rollout was commanding the fingers at all.

With that fixed, plus joint bimanual fitting and two-handed grasp synthesis, the
four longest bimanual references go from losing the object outright to tracking
it:

    sequence                object          tracking error
                                          start of day -> now
    bowl_drink_1            bowl            160.9 m  ->   56 mm
    camera_takepicture_2    camera          188.3 m  ->  117 mm
    binoculars_see_1        binoculars       21.8 m  ->  165 mm
    gamecontroller_play_1   gamecontroller   97.1 m  -> 1615 mm

Three of four inside 200 mm, none in free fall. The grasp itself holds in all
four (0.11-1.47 cm of drift over a 0.8 s hold test, against 1.3-998 cm before).

Two-handed grip establishment is implemented and left OFF by default: it is not
a uniform win. It raises camera's in-tolerance frames from 22/161 to 55/161 and
costs gamecontroller (1615 mm -> 91903 mm) and binoculars (165 mm -> 21373 mm).
Recorded with both numbers rather than tuned to the average.

One merge bug worth keeping: combining the two sides' servo targets by taking
whichever was non-zero silently drops a target of exactly zero, which is a
legitimate target. Assigned by actuator ownership instead.

## 2026-09-15 — stage 5: the regression metric improved and the policy still fails

Rolling the distilled policy out in the simulator, on objects it never saw:

    held-out object   feedforward   distilled policy   frames in tolerance
    camera              31.1 mm        23549 mm             0/146
    hammer              18.2 mm         3180 mm             0/14
    hand                23.6 mm        14620 mm             0/99

**It drops the object every time.** The held-out MAE ratio is 0.716 -- better
than predicting a constant, robust across five object splits -- and that is not
sufficient. A command error that would be unremarkable in a regression metric
breaks contact, and once contact is broken nothing the policy does afterwards
matters. The feedforward it is cloning tracks these same references to 18-31 mm.

So the earlier entry overstated it. Changing the label from MPPI's correction to
the absolute command fixed the *learnability* problem, which was real and
measurable (R^2 0.075 -> a target that is a function of the state, ratio 2.235
-> 0.716). It did not make the stage work. Behaviour cloning of a contact-rich
tracking controller does not survive its own errors: the training distribution
contains only states the demonstrator visited, and the first command error takes
the policy somewhere it has never seen.

That is the standard argument for DAgger and for training IN the loop, and it is
why DexTrack distils an RL policy rather than cloning a trajectory optimiser.
The measurement to record is that MAE was the wrong scoreboard, and the rollout
is the right one -- exactly the mistake this repository made once before with a
behaviour-cloning result that open-loop replay matched.

## 2026-09-15 — PPO works, and my evaluation was out of distribution

Stage 3 done the way DexTrack does it, rather than with MPPI standing in.

**Throughput first, because it was the blocker.** MuJoCo releases the GIL inside
`mj_step`, so the environment pool steps its physics in threads: 8 envs go from
9,676 to 48,979 MuJoCo steps/s here, 5.06x. `ctrl_for` was calling `np.clip`
once per actuator in a Python loop -- 9,400 times per 25 control steps, 11% of
wall clock -- and is now one matmul. Together with 24 environments instead of 8:
**36 -> 195 control steps/s**, which is two hours per million instead of eight.
The feedforward is bit-identical afterwards (8197.3 mm), which is the check that
this was a rewrite and not a behaviour change.

**Two runs of 614,400 control steps each.** Training is healthy in both: reward
rises monotonically and the alive fraction sits at 0.98.

    evaluation                          feedforward   PPO (v1)   PPO (v2)
    one 111-step rollout from gf[0]        8197 mm     10535 mm   12413 mm
    12 starts x 64 steps, mean             19.7 mm         --      22.7 mm
    12 starts x 64 steps, MEDIAN           18.7 mm         --       7.9 mm
    starts where PPO is better                 --          --      11/12

**The first row is an out-of-distribution evaluation and I wrote it down twice
before noticing.** The policy trains on 64-step windows from randomly drawn
graspable starts, and I was scoring it on a single 111-step rollout -- 1.7x its
training horizon, from one fixed start. On the distribution it was actually
trained on it beats the feedforward on **11 of 12 starts**, with a median error
of 7.9 mm against 18.7 mm.

So stage 3 works, with the honest qualifier attached: PPO learns a tracking
correction that is better than the feedforward over the horizon it was trained
for, and does not extrapolate to horizons well beyond it. Longer training
windows are the obvious next measurement, and they cost linearly.

Reward shaping is recorded as a wash rather than a win. The first reward,
`exp(-e/0.02)`, is 0.007 at 10 cm, so a drifting policy has no gradient back;
the second added a wide exponential and a linear term. On the training
distribution v2 is clearly good (median 7.9 mm); on the long rollout it is worse
than v1 (12413 vs 10535 mm). Both sit inside the same out-of-distribution
artifact, so the comparison between them is not worth much.

## 2026-09-15 — DAgger does not fix stage 5, and the reason is the expert

Plain cloning fails in the loop because the training set contains only states
the expert visited. DAgger is the textbook answer: roll the policy out, relabel
the states it actually reached, retrain. Ten references, three rounds, held out
by object:

    round   transitions   gamecontroller   camera   knife
      0         972           523 mm       1096 mm   164 mm
      1        1944           664 mm       1330 mm   469 mm
      2        2916           942 mm       1848 mm  1124 mm

**Every round is worse than the last.** That is not a tuning failure, it is the
wrong expert. DAgger assumes the expert can say what to do from a state the
policy wandered into. Mine is the feedforward, which is a function of the frame
index alone -- it does not observe the object, and its command from a drifted
state is the same command it would give from a good one. Relabelling
off-distribution states with an action that does not recover teaches the policy
that no recovery is needed, and the dataset fills up with exactly that lesson.

So stage 5 needs a CLOSED-LOOP expert, and the cheapness that made this
attractive -- relabelling costs one FK call because the expert is a function of
k -- is precisely the property that makes it useless. MPPI and PPO both qualify
and both cost a real query per label.

Worth recording: round 0 here (plain cloning, 164-1096 mm) is far better than
the earlier cloning result (3180-23549 mm), because this pipeline establishes a
grip and synthesises a grasp first. The earlier number was measuring a bad
starting state as much as a bad policy.

## 2026-09-15 — stage 5: four DAgger variants, and round 0 wins every time

Held-out-object tracking error (mm), 10 references, objects held out:

    variant                          round 0   round 1   round 2
    feedforward expert                 594      821      1305
    MPPI expert (closed loop)          594      672      1396
    MPPI + drop-lost-states filter   18431    21034        --
    MPPI + beta decay, no filter       594      751      3012

(means over camera / gamecontroller / knife.)

**Plain cloning is the best of the four, and every iteration makes it worse.**
That is a reproducible negative across expert choice, beta schedule and state
filtering, not a tuning accident.

The MPPI expert does help where the theory says it should -- round 1 improves on
the feedforward expert (672 vs 821, and gamecontroller 443 vs 664 mm) because it
can say how to get back from a drifted state, which the feedforward cannot. It
just is not enough to stop the compounding.

The filter row is my own bug and is kept as one. Dropping states whose tracking
error exceeds 12 cm looked like removing hopeless data; applied in round 0 as
well, it removed the states the EXPERT visits late in an episode, where its own
error naturally grows, and left a policy that had never seen the end of a
trajectory. Held-out error went from 523 mm to 17994 mm on a dataset only 20%
smaller. Fixed to apply only to policy-driven rounds.

So stage 5 does not work. Round 0 reaches 164-1096 mm on held-out objects
against an expert that tracks the same references to 18-31 mm, and no DAgger
variant closes it. What has NOT been tried is distilling the PPO policy itself
rather than a trajectory optimiser -- which is what DexTrack actually does, and
which now has a working stage 3 to draw from. That costs one PPO run per
reference (~50 min here) and is the honest next step rather than another
relabelling scheme.

## 2026-09-15 — **stage 3 solved.** PPO tracks the whole reference when its horizon covers it.

    controller                          mean      final    frames within 50 mm
    open-loop feedforward             8197.3 mm  52878 mm       53 / 111
    PPO, training horizon  64        10535.3 mm  62601 mm       57 / 111
    PPO, training horizon 160           29.4 mm     31.9 mm    111 / 111

844,800 control steps, 6237 s. **Every frame inside 50 mm and the object never
leaves the hand**, against a feedforward that loses it halfway. 279x better than
the baseline it corrects.

The whole story is the horizon. PPO learns a correction over the window it is
trained on and does not extrapolate past it; the evaluation is a 111-step
rollout, so a 64-step policy is being asked for something it never saw and a
160-step policy is not. I reported "PPO loses" twice before working that out,
and the fix was not a better reward or more steps -- the reshaped reward was a
wash and 614k steps at horizon 64 did not help. It was matching the horizon to
the task.

Reward shaping is still recorded as a wash: v1 and v2 differ by less than the
horizon does, and both were measured inside the same out-of-distribution
artifact.

This also makes stage 5 worth another attempt for the first time. Every previous
distillation cloned a trajectory optimiser or an open-loop feedforward, and the
failures traced to exactly that. There are now policies worth distilling.

## 2026-09-15 — stage 5: distilling PPO policies finally generalises without dropping

Six references, PPO trained per reference with a horizon covering the clip, the
policies' own on-policy states and commands distilled into one network, held out
by object:

    per-reference PPO       hammer 2.5   camera 17.7   mouse 62.3   binoculars 60.7 mm
                            phone 216250   gamecontroller 211786 mm   (2 of 6 fail)

    held-out object    feedforward    its own PPO    DISTILLED
    gamecontroller      247682 mm      211786 mm       456.5 mm
    hammer                  18.2 mm         2.5 mm     158.6 mm

Against every earlier distillation attempt, which dropped the object on every
held-out reference (3180-23549 mm, i.e. free fall), **the distilled policy now
keeps it**: 158-456 mm is drift, not a drop. On gamecontroller it is 540x better
than the expert it was distilled from, because that expert fails and the network
has learned from the four references that do not.

It is still far short of the per-reference policies (2.5-17.7 mm), and with four
training references that is a data-scale statement rather than a verdict.

The sequence of failures that led here is worth keeping, because each one was
diagnosed rather than guessed:

    cloning MPPI's correction        ratio 2.235, worse than a constant -- the
                                     label had R^2 0.075 on the observation
    cloning the absolute command     ratio 0.716 and dropped the object in the
                                     loop -- a regression metric is not a rollout
    DAgger, four variants            round 0 best every time -- an open-loop
                                     expert cannot say how to recover
    distilling PPO policies          keeps the object on held-out objects

Which is the argument DexTrack's design makes, arrived at from the wrong end
four times.

## 2026-09-15 — the stage 6 "discrepancy" was my own unit label

I recorded binoculars as showing a search score of 46.9 mm against a 92752 mm
rollout and called the bimanual numbers untrustworthy. There is no such bug.
`track_score` returns a **composite** -- a clipped mean error plus half the
fraction of dropped frames -- which is unitless, and my test script printed it
with a "mm" suffix beside a genuine millimetre error. A score of 0.0469 printed
as "46.9 mm" looked like a 2000x disagreement.

Checked directly, the code repeats to the digit:

    before search   score 0.6642   rollout mean 96709 mm   dropped 0.877 of frames
    repeat          score 0.6642   rollout mean 96709 mm   dropped 0.877
    after search    score 0.6589   rollout mean 93575 mm   dropped 0.870
    repeat          score 0.6589   rollout mean 93575 mm   dropped 0.870

So stage 6's numbers stand as measured, and binoculars genuinely fails --
87% of its frames have the object dropped. The two that work, work:
**bowl 28.1 mm (103/131 frames in tolerance)** and **camera 44.1 mm (151/161)**.
Two of four, honestly measured, with no outstanding reproducibility question.

The lesson is narrow and worth the entry: never print an objective and a metric
with the same unit suffix. I spent an hour on a phantom.

## 2026-09-15 — stage 6: all four bimanual references track, once the search is scored on tracking

    sequence                object          start of day   now      frames in tolerance
    bowl_drink_1            bowl              160.9 m      28.1 mm      103/131
    binoculars_see_1        binoculars         21.8 m      29.2 mm      125/138
    camera_takepicture_2    camera            188.3 m      44.1 mm      151/161
    gamecontroller_play_1   gamecontroller     97.1 m      65.2 mm       50/192

**Four of four, all inside 70 mm**, from four references that were losing the
object by tens or hundreds of metres. Three ingredients, each found by a
measurement that contradicted an assumption:

1. the rollout never commanded `d.ctrl`, so the servos drove every finger open
   the moment stepping began -- the hands held at reset only because `place`
   writes qpos directly;
2. the two hands were fitted in separate scenes and interpenetrated by 11.7 mm
   across 42 contacts, the left pushing the right off the object. Fitting them
   in one scene needed a hinge-base variant, because a free joint leaves the
   solver no wrist DoF;
3. the grasp search was scored on HOLDING. Holding is necessary and not
   sufficient: `gamecontroller_play_1` has 15 frames that hold under gravity
   and a feedforward that ended 248 m away. Scored on tracking over the whole
   reference, the same search finds 65.2 mm.

Seed variance is real and recorded rather than hidden: binoculars gives 29.2,
48.0 and 71.3 mm over three seeds, the last holding only 4 of 138 frames. The
search is a random restart over wrist offsets and it does not always find the
basin.

The same third fix carried to the ONE-handed stage, where the search was still
scored on `grasp_frames`: `phone_call_1` 248398 -> 11676 mm and
`gamecontroller_play_1` 247682 -> 35.4 mm. Those were precisely the two
references per-reference PPO could not rescue, because PPO learns a bounded
correction and cannot repair a baseline that is 248 m wrong.

## 2026-09-15 — stage 3 coverage, after the grasp search was scored on tracking

Per-reference PPO, same budget, before and after the search objective changed
from holding to tracking:

    reference                hold-scored search   track-scored search
    mouse_use_1                    62.3 mm              29.8 mm
    gamecontroller_play_1      211786 mm                27.2 mm
    phone_call_1               216250 mm              5069 mm
    camera_takepicture_2           17.7 mm              57.7 mm

The two catastrophic failures are gone. `gamecontroller_play_1` improves by a
factor of 7800, and the reason is not that PPO got better -- it is that PPO
learns a BOUNDED correction (6 mm of palm translation per control step) and
cannot repair a baseline that is 248 m wrong. Give it a feedforward that is
merely imperfect and it works; give it one that has thrown the object across the
room and no amount of training helps.

`camera` gets worse (17.7 -> 57.7 mm), so this is not free. `phone_call_1` is
still poor at 5069 mm and is the remaining single-hand failure.

## 2026-09-15 — stage 2 is the whole problem, and it is measured

Two scans, one mine and one from a peer session working the same tree, settle
what has been going wrong all day.

**Every frame of every reference penetrates.** Across 20 s1 sequences, 2866
frames, **23 frames are under 1 mm**. Per-sequence minimum penetration: median
2.51 mm, max 17.15 mm, and 7 of 20 never get below 5 mm at their BEST frame.
`flashlight_on_2` spends all 144 frames with its palm inside the object;
`binoculars_see_1` spends all 155 with its FOREARM inside, 17.15 mm deep.

**The split is structural.** Sequences that reach a clean frame are ones where
the deepest body is a distal link. Sequences that never do are ones where it is
the forearm, the palm, or a proximal link:

    forearm   binoculars 17.15, camera_takepicture_1
    palm      flashlight_on_2 12.82, scissors_use_2 8.36
    proximal  cup_drink_2 11.73, cup_lift 9.40, mug_drink_1 6.81

That is exactly what a retarget with no non-penetration constraint on the ARM
would produce. Fingertips can sometimes land outside the object by luck of the
pose; a forearm placed by matching a human wrist cannot.

**And the feasible poses do not hold the object.** Raising the fit's penetration
weight works -- on `mug_drink_1`, w_pen 1 -> 25 takes median penetration
10.80 mm -> 0.96 mm -- and the resulting grasp tracks WORSE, not better:

    reference      w_pen   pen      force   residual   tracking
    mug_drink_1      1.0  10.13 mm  2294 N     4.0x      33.9 mm
    mug_drink_1     25.0   1.35 mm  1202 N    12.3x     148.7 mm
    binoculars       1.0   0.02 mm    44 N    32.6x  176578 mm

The binoculars row is the clearest: 0.02 mm of penetration, 44 N, and the object
on the floor. Contacts fall from 30 to 2.8 as penetration clears, because the
contacts WERE the penetration. A peer sweep of 283 candidate wrist offsets found
**0** that were simultaneously non-penetrating and touching.

**So every physics remedy attempted today was compensating for stage 2.**
Closing under physics, adaptive finger opening, wrist retraction along the
contact normal, penetration terms in the search objective, multi-restarts --
each addressed a symptom of a fit that has no feasibility constraint. Two of
them are still worth keeping on their own merits (closing under physics is the
standard way to make a grasp, and it took `gamecontroller_play_1` from 4411 N to
10.5 N), but none of them can repair a wrist that is inside the object, and
wrist retraction along contact normals actively diverges when the hand is
wrapped -- normals oppose, so escaping one drives others deeper (binoculars:
17.15 -> 62.44 mm over 11 iterations).

**What stage 2 needs.** A hard non-penetration constraint on the arm-side bodies
-- forearm, wrist, palm -- treated as a rigid-body placement problem, with the
fingers free to find contact afterwards. Matching the human's wrist pose is what
puts the forearm inside the object, and a Shadow forearm is not shaped like a
human's. The contact REGION on the object is what should be matched; the robot's
joint angles will not resemble the human's and should not be asked to.

Everything downstream is invalid until that lands: stage 3's 29.4 mm, stage 6's
28-65 mm, every PPO checkpoint, and the stage 5 distillation numbers below.

## 2026-09-15 — the arm-side constraint is necessary and not sufficient

Constraining forearm/wrist/palm/knuckles at 60x the finger weight, measured
end-to-end through the rollout:

    reference          w_arm   arm pen    force   residual   tracking
    binoculars_see_1     1.0   5.20 mm    117 N     29.5x   175528 mm
    binoculars_see_1      60   2.47 mm   2473 N     15.4x      106.2 mm
    mug_drink_1          1.0   7.04 mm   2294 N      4.0x       33.9 mm
    mug_drink_1           60   0.48 mm   1195 N      8.3x      146.8 mm
    flashlight_on_2      1.0  12.96 mm    782 N      1.7x        8.5 mm
    flashlight_on_2       60   1.39 mm    386 N      4.3x      107.5 mm

**binoculars_see_1 goes from 175 metres to 106 mm.** That reference never
reached within 17 mm of feasible at any of its 155 frames and was unreachable by
every remedy tried today; constraining the arm makes it tractable. That is the
constraint earning its place.

**And mug_drink_1 and flashlight_on_2 get worse**, 33.9 -> 146.8 mm and 8.5 ->
107.5 mm. This is not a regression to fix, it is the same finding again: those
two were "working" BECAUSE of penetration. flashlight_on_2 tracked at 8.5 mm
with its palm 12.96 mm inside the object. Remove the burial and the fake grip
goes with it.

So the arm constraint is necessary -- it is the only thing that has made a
forearm-blocked reference tractable -- and it is not sufficient. Residual is
4.3-15.4x everywhere and forces are 361-2488 N on a 1.96 N object, so none of
these is an equilibrium. What is still missing is a finger-side contact set that
is a grasp rather than an overlap, and that cannot come from placing fingertips
either. It has to come from closing, which is what `reset_grasp` does and which
works when the hand starts outside the object (gamecontroller_play_1, 4411 N ->
10.5 N) and not otherwise.

Stage 2 therefore needs both halves: the arm-side feasibility constraint, which
now exists, and finger contact established by closing from a feasible pre-grasp,
which exists but only fires correctly when the arm constraint has already put
the hand outside. Those two were built in the wrong order today.

## 2026-09-15 — what the finger half has to solve, measured

After the arm-side constraint lands, the fingertips are **50-62 mm from the
object surface**:

    flashlight_on_2   49.5  51.7  50.1  15.8  70.0 mm   median 50.1
    mug_drink_1       58.1  59.5  57.8  43.1  90.7 mm   median 58.1
    binoculars_see_1  72.6  62.3  60.1  34.2  76.0 mm   median 62.3

`establish_grip` closes by at most 1.2 rad, which moves a fingertip 30-40 mm. So
it cannot reach, and closing finds nothing -- every closed configuration came
out at 0.0 N with the object on the floor.

That is the precise shape of the remaining problem, and it is a real tension
rather than a tuning gap: the arm-side constraint and the fingertip targets
share one wrist, so pushing the forearm out of the object pushes the fingers out
with it. Constraining the arm makes the pose feasible and simultaneously makes
it unable to grasp.

What that means for stage 2: the wrist pose has to be CHOSEN so that the arm
clears and the fingers are within closing range -- a pre-grasp placement
problem, over approach directions, not another weight in the same objective.
Both terms currently pull on the same six DoF and one of them always wins.

I attempted a one-dimensional approach line search along the palm direction and
reverted it. It computed the fingertip clearance and then never tested it, so it
advanced the full 90 mm regardless and drove the hand past the object; every
reference came back at 0.0 N. Recorded because the failure is informative about
the shape of the fix: an approach search needs a termination condition on the
FINGER side and a feasibility condition on the ARM side simultaneously, and
getting one of the two wrong is worse than not doing it -- the unsearched
arm-constrained pose at least tracks binoculars at 106 mm.

## 2026-09-15 — approach line search, second attempt, also reverted

Two attempts, both wrong in the same area — keeping the hand's state consistent
across `qpos`, the mocap target and the weld that ties them.

    attempt 1   computed the fingertip clearance and never tested it, so it
                advanced the full 80 mm regardless and drove the hand past the
                object.
    attempt 2   terminated correctly on arm-side contact, but synced the mocap
                by calling `palm_pose()`, which zeroes qpos internally and so
                computed the palm as if the base were at the origin. Reading the
                palm from the live `d.xpos` instead still advanced the full
                80 mm with zero contacts on all three references.

Reverted both. The measurement that motivated them stands and is the useful
part: after the arm-side constraint the fingertips sit 50-62 mm from the object
surface while the closing routine moves them 30-40 mm, so the wrist has to be
PLACED rather than traded off. But three passes at it produced nothing that
works, and an unsearched arm-constrained pose still tracks binoculars at 106 mm,
which is better than a broken search.

Recorded rather than quietly dropped because the failure is specific and the
next person (me, later) should not rediscover it: this codebase drives the hand
through a mocap body welded to a free joint, so ANY routine that repositions the
wrist has to update `qpos`, the mocap target, and call `mj_forward`, in that
order, and must not call `palm_pose()` to read the current palm because that
helper zeroes qpos by design. Two of my three attempts died on exactly that.

## 2026-09-15 — translating the wrist cannot close the finger gap: the trade is 1:1

The approach line search now works. Doing it in the FITTING scene -- six
hinge/slide joints, no mocap body, no weld -- avoids the state-synchronisation
bug that killed three attempts in the simulation scene. `set_q`, `mj_forward`,
read the contacts.

Its answer is that the search is not the missing piece:

    reference          arm allowance   advance   arm pen   tip gap   contacts
    binoculars_see_1        0.5 mm      0.0 mm    2.47 mm   62.3 mm      6
    binoculars_see_1       10.0 mm      8.0 mm    9.77 mm   57.4 mm      7
    mug_drink_1             0.5 mm      0.0 mm    0.48 mm   58.1 mm      5
    mug_drink_1            10.0 mm      8.0 mm    8.25 mm   51.6 mm      7
    flashlight_on_2         0.5 mm      0.0 mm    1.39 mm   50.1 mm      3
    flashlight_on_2        10.0 mm      8.0 mm    8.54 mm   43.5 mm      3

At a 0.5 mm clearance the hand cannot advance AT ALL -- the arm is already at
its contact limit while the fingertips are 50-62 mm away. Paying 10 mm of arm
penetration buys 4.9-6.6 mm of finger gap. **The trade is about 1:1**, so
closing a 60 mm gap by translation would cost 60 mm of arm inside the object,
which is worse than where the day started.

So the fingertips are not far from the object because the hand is in the wrong
PLACE. They are far because of the hand's POSE -- the wrist's orientation and
the finger configuration that the fit produced once the arm term started
pushing. No amount of sliding fixes that, and this rules the translation fix out
rather than leaving it as an untried option.

What that leaves for stage 2, stated as precisely as tonight's measurements
allow. Three distinct problems, not one:

1. arm-side feasibility -- SOLVED. Arm-side penetration is 0.00 mm at the median
   in 29 of 40 sequences; binoculars_see_1 went from unreachable at all 155
   frames to tracking at 106 mm.
2. finger reach -- OPEN, and not a placement problem. The wrist ORIENTATION and
   the finger configuration have to be re-solved together under the arm
   constraint, rather than the fit being run and the hand then moved.
3. the vessel cluster -- OPEN and separate. Four mug variants plus a cup sit at
   33-39 mm arm penetration with exactly ZERO finger contact, and a peer sweep
   of w_arm from 60 to 1000 moves it 0.35 mm. The whole hand is inside the
   vessel; the wrist TARGET is wrong for concave objects, not the penalty.

`advance_to_contact` is kept. It is correct, it is cheap, and it is the tool
that measured the 1:1 trade -- which is the result that matters here.

## 2026-09-15 — CORRECTION: the fingers DO reach. The poses are valid and still not grasps.

Two of my conclusions tonight were artifacts of measuring at frame 0 of the
hold window, which is a 20x outlier.

**Frame 0 is not representative.** Median fingertip gap to the object surface,
over the whole window, in the fitting scene:

    reference          frame 0   median   min    frames under 10 mm
    binoculars_see_1    62.3 mm   2.2 mm  1.2 mm      153 / 155
    flashlight_on_2     50.1 mm   4.3 mm  1.1 mm      141 / 144
    mug_drink_1         58.1 mm   3.7 mm  2.0 mm      115 / 116

So "the fingertips are 50-62 mm away after the arm constraint" is wrong, and the
"finger reach" problem I named as stage 2's second piece does not exist. The fit
puts the fingers within a few millimetres on 98% of frames. Every recent
measurement I took used `reset_at(0)` or `reset_grasp(0)`, which starts at the
one unrepresentative frame. Same error class as reading G5's H2 from one
sequence, and as printing an objective with a millimetre suffix: measuring at an
unrepresentative point and generalising from it. Third time today.

**And the corrected measurement is worse news, not better.** Starting from a
representative frame:

    reference         start   arm pen    force   residual   tracking
    binoculars_see_1      0   2.47 mm   2472 N    15.4x       106 mm
    binoculars_see_1     67   0.00 mm    5.5 N     1.4x     54818 mm
    flashlight_on_2      10   0.00 mm   18.6 N     6.6x    128922 mm
    flashlight_on_2      10   0.00 mm    0.0 N     1.0x     21146 mm  (closed)
    mug_drink_1          57   0.00 mm  108.6 N     8.6x     24300 mm

At a good start frame the configuration is **physically valid** -- zero
penetration, forces of 5-109 N rather than thousands -- and it **drops the
object**. The frame-0 configurations that appeared to track at 106-147 mm were
doing it on 386-2472 N of manufactured contact.

That is the finding, stated more sharply than anything earlier today: the
retarget produces poses that are geometrically correct, physically valid, and
not grasps. A fingertip 2 mm from a surface exerts no force. Penetration was
never the disease -- it was the only thing making the contact sets look like
grasps, and removing it reveals that nothing underneath was holding the object.

So stage 2 does not have three problems. It has one: the pipeline has no step
that produces GRIP. `reset_grasp` is meant to be that step and reaches only
0-13 N against a target of 8 N, which is the number to chase next.

## 2026-09-15 — the contacts are a touch, not a grip. Which is this project's own thesis.

Closing at a valid start frame, zero penetration, sweeping the force target:

    reference          target   grip reached   object drop in 1 s
    binoculars_see_1     8 N        7.8 N          513.8 cm
    binoculars_see_1    40 N        2.6 N          489.9 cm
    binoculars_see_1   150 N        2.6 N          489.9 cm
    mug_drink_1          8 N        9.8 N          480.3 cm
    mug_drink_1         40 N       42.1 N          359.5 cm
    mug_drink_1        150 N       44.8 N          275.8 cm

Free fall over one second is 490 cm. So at 44.8 N of measured contact force on a
1.96 N object, the object is still falling at essentially the unimpeded rate.
Force is being generated and it is not opposing gravity.

That is the end of the chain the whole day has been following:

    the poses penetrate           -> no, that was one frame in a hundred
    the fingers cannot reach      -> no, median gap is 2-4 mm
    the pose is not feasible      -> no, 0.00 mm penetration at a good frame
    closing cannot build force    -> no, it reaches 44.8 N
    the contacts do not OPPOSE    -> yes

A contact set that touches an object at several points on the same side
produces force without producing a grasp. Raising the force target just presses
harder in a direction that does not hold. This is exactly the quantity
Ferrari-Canny epsilon measures, and `src/oppdef/grasping/epsilon.py` has
implemented it since the synthetic project -- and nothing in `human/` has ever
imported it. The GRAB pipeline scores held-ness, tracking error, penetration and
now equilibrium residual, and not one of those distinguishes a grasp that
resists a wrench from one that presses.

The earlier epsilon measurement on these contact sets is not evidence against
this: it was computed on PENETRATED configurations, where deep overlap
manufactures 20-42 well-distributed contacts and any static wrench metric reads
strongly force-closed. On the valid configurations measured here -- zero
penetration, a handful of contacts -- it has not been computed at all.

So the single outstanding question for stage 2 is whether the retarget can be
made to produce OPPOSING contacts, and the natural next measurement is epsilon
on the valid contact sets rather than the penetrated ones. This project is
called opposition-deficit; it turns out the DexTrack pipeline built inside it
never checked for opposition.

## 2026-09-15 — epsilon = 0. The retarget produces non-force-closed contact sets.

Ferrari-Canny computed from MuJoCo's own narrowphase, all object geoms, contact
normals oriented into the object, friction from the model:

    reference          state                   contacts   epsilon   grip
    binoculars_see_1   frame 0 (penetrated)        6       0.00000  2472.5 N
    binoculars_see_1   good frame, direct          6       0.06889    21.5 N
    binoculars_see_1   good frame, closed          1       0.00000     2.6 N
    mug_drink_1        frame 0 (penetrated)        5       0.00000  1194.9 N
    mug_drink_1        good frame, direct          9       0.00000    71.3 N
    mug_drink_1        good frame, closed          4       0.00000    42.1 N

**Zero in five of six**, including the valid, non-penetrating, 42 N
configurations. Epsilon zero means the origin is not interior to the convex hull
of the contact wrench set: the grasp cannot resist an arbitrary wrench, however
hard it presses. That is "the contacts do not oppose", measured with the metric
this repository has had since the synthetic project and which nothing in
`human/` has ever imported.

It also explains the whole day in one line. A non-force-closed contact set
cannot hold an object no matter how much normal force it generates, so:
penetration was the pipeline's only way of faking a grasp, removing it exposed
the absence, raising the force target pressed harder in a direction that does
not hold, and every downstream stage was learning to track an object that was
never actually held.

**One discrepancy I cannot resolve and will not paper over.** A peer session
computing epsilon on the same penetrated configurations, with its own multi-geom
implementation, reported 0.155-0.344. I get 0.00000 on those. We are using
different code and quite possibly a different normal-sign or friction
convention, and I have not reconciled them. What is internally consistent in MY
numbers is the comparison that matters here -- the valid closed configurations
are zero -- but the penetrated row should not be quoted from this table until
the two implementations are reconciled.

**Next measurement**, and it is now a single well-posed question: can the fit be
made to produce contact sets with epsilon > 0? The machinery exists
(`grasping/epsilon.py`, `wrench_set`, `epsilon_from_wrenches`), the cost is one
narrowphase pass plus a convex hull, and it is the term the objective has never
had. Held-ness, tracking error, penetration and equilibrium residual between
them cannot distinguish a grasp that resists a wrench from one that presses.

## 2026-09-15 — geometric opposition does NOT separate the human from the robot

I added `approach_opposition` -- for each fingertip, the outward surface
direction at its nearest object point, scored as |mean unit direction|, so 0 is
opposed and 1 is all on one side. It reads 1.000 on every robot configuration
tried, which looked like the missing quantity.

It is not. Measured at the middle of the hold window, same object, same frame:

    reference          HUMAN tips   HUMAN whole-hand   ROBOT tips
    binoculars_see_1      0.836          0.879            0.830
    flashlight_on_2       0.528          0.447            0.607
    mug_drink_1           0.937          0.949            0.945

The human's own contact set -- the one that demonstrably holds the object, with
100-145 vertices within 5 mm -- is as one-sided by this measure as the robot's,
and on binoculars and mug it is MORE so. So whatever makes the human grasp work
is not captured by the spread of outward surface directions, and a search
scored on this would not find it.

That is my fourth wrong hypothesis today, after the finger-reach problem, the
1:1 translation trade, and the epsilon-as-search-gradient idea. Recorded with
the measurement rather than deleted, because the negative is informative: it
rules out the cheapest geometric proxy for "opposition" and it does so using the
human demonstration as the positive control, which is the right way to test any
proposed grasp-quality term in this pipeline. Any future candidate should be
required to separate the human's contact set from the robot's before it is put
in an objective.

A peer session measured the quantity that DOES separate carrying from failing,
on contact sets rather than surface geometry: configurations that carry engage
10 and 3 distinct hand BODIES at contact one-sidedness 0.055 and 0.191, while
those that fail engage ONE body at ~1.0. That is a wrap versus a touch, and it
is a property of how many links engage and from which directions -- not of where
the fingertips happen to sit. Both their measurement and mine point at the same
conclusion from opposite sides: fingertip geometry is the wrong level of
description for this problem.

## 2026-09-15 — the wrap search is correct and finds nothing. The neighbourhood is empty.

`wrap_score` wired into the grasp search, wide rotation (0.35 rad), penetration
gated at 3 mm:

    reference          stage     pen       force   bodies  one-sided
    binoculars_see_1   before   2.47 mm   2472 N      1      0.708
    binoculars_see_1   wrap     2.47 mm   2472 N      1      0.708
    flashlight_on_2    before   1.39 mm    386 N      1      0.999
    flashlight_on_2    wrap     1.39 mm    386 N      1      0.999
    cup_lift           before  11.65 mm   4368 N      8      1.000
    cup_lift           wrap    11.65 mm   4370 N      8      1.000

Nothing moves, and that is the gate working rather than failing. Every sampled
candidate exceeds the 3 mm penetration allowance, so every one scores worst-case
and none is strictly better than the baseline, so the search keeps what it had.

This is the third independent route to the same conclusion: a peer's 283-candidate
sweep found zero poses that both contact the object and stay out of it; my w_pen
sweep bought 1.35 mm of penetration at 148.7 mm of tracking; and now a
human-validated wrap metric, correctly gated, finds no better pose anywhere in
the retarget's neighbourhood.

**So the fix cannot be a search around the retarget's output.** Every such search
-- translation, rotation, wrist offsets, opposition, wrap -- is looking in a
neighbourhood that contains no valid grasp. The retarget's OBJECTIVE has to
change so that its output lands somewhere else, and `wrap_score` is now the term
to change it toward: many links, opposing, at bounded penetration, validated
against the human demonstration rather than assumed.

Two wiring errors of mine on the way here, both recorded because they each cost a
full measurement cycle: the search branch was gated on `objective == "track"`, so
"wrap" silently fell through to the old hold-based path and the validated score
was never called; and before that, the "oppose" score omitted arm penetration
entirely, so widening the rotation search from 0.08 to 0.45 rad took
binoculars_see_1 from 2.47 mm of arm penetration to 15.52 mm and scored it an
improvement. Five wiring errors today, each found by a measurement that did not
match what the code was supposed to do.

## 2026-09-15 — stopping. Sixth wiring error, and that is now the limiting factor.

I tried the objective change that `wrap_score` motivates: surface-CONTACT targets
for the middle phalanges, so the fit asks links other than the five fingertips to
reach the object. It produced byte-identical results at W_MID_CONTACT 0.0 and
0.7 on all three references -- penetration, force, engaged bodies, one-sidedness
and drop all unchanged -- which means it was inert, not that it failed. Reverted
rather than committed, because unverified code that appears to do nothing is
worse than no code.

That is the sixth wiring error today. The others:

    the search branch gated on `objective == "track"`, so "wrap" silently fell
      through to the hold-based path and the validated score was never called
    the "oppose" score omitted arm penetration, so widening the rotation search
      scored a 2.47 -> 15.52 mm regression as an improvement
    `palm_pose` zeroes qpos, so using it to sync the mocap commanded the hand to
      the origin
    the approach search computed its termination condition and never tested it
    `track_score` printed with a "mm" suffix beside a real millimetre error,
      which cost an hour chasing a 2000x discrepancy that did not exist

Every one was caught by a measurement disagreeing with what the code was supposed
to do, none by reading the code. The rate is now roughly one per attempt, which
means further attempts tonight add noise rather than results. The next step --
rewriting the retarget's objective around wrap_score -- is well-posed and
deserves better than a seventh.

What stands, and what the next session starts from:

    arm-side feasibility constraint   arm penetration 0.00 mm median in 29/40
                                      sequences; binoculars_see_1 from
                                      unreachable at all 155 frames to 106 mm
    wrap_score                        the only grasp-quality term that separates
                                      the human's contact set from the robot's;
                                      penetration gated, not weighted
    advance_to_contact                correct, cheap, and the tool that measured
                                      the 1:1 translation trade
    the diagnosis                     in this pipeline tracking came FROM
                                      penetration; every downstream number was
                                      measuring the contact solver
    the method                        any proposed grasp term must separate the
                                      human's contact set from the robot's
                                      before it enters an objective. That test
                                      kills epsilon and fingertip spread in
                                      minutes.

## 2026-09-15 — the fit targets the wrong links, and the human data says so

Retracting my "inert" conclusion above: the middle-phalanx contact term SHOULD
have fired, so its no-op was a wiring bug and not a dead end. The human's middle
phalanx joints are within the 15 mm contact tolerance on 2-5 fingers of every
reference tested:

    reference          TIP gaps (mm)                MID gaps (mm)         mid<15mm
    binoculars_see_1   3.1  7.8  6.5 10.8  0.9      5.8 10.3 18.4 24.5 16.1   2/5
    flashlight_on_2   10.3 15.4 12.8 23.3 27.9      7.8  2.9  5.4 24.0 13.1   4/5
    cup_lift           6.8 21.8 11.6 18.8  5.5      4.8  7.7  6.3  5.3  0.5   5/5
    mug_drink_1       12.8 13.1 13.6 16.4  1.1      7.6  8.2  9.8 26.0 27.7   3/5

On `cup_lift` every one of the five middle phalanges is CLOSER to the object
than the fingertips are -- 0.5-7.7 mm against 5.5-21.8 mm. The human is holding
that cup with the middles of its fingers, and the fit has only ever been asked
to place the tips.

That is the objective defect stated at the level it actually lives at. Not
penetration, not reach, not opposition, not force: the retarget optimises the
wrong five points. A hand whose fingertips are on the surface and whose middle
phalanges are 20 mm off is touching; the human's, with the middles at 0.5 mm, is
wrapping. And this is measurable directly from the demonstration, for every
reference, before any simulation runs -- which makes it the cheapest possible
term to add and to validate.

So the next session has a concrete, validated change rather than a direction:
give the middle phalanges surface-contact targets on the fingers where the
human's own middle joint is within tolerance -- 2 to 5 of them per reference --
and check the result with `wrap_score`, which counts exactly the engaged-link
property this would produce. My attempt at it was inert through a wiring fault
and is reverted; the idea is not what failed.

### The middle-phalanx term, wired correctly, and what it cost

The seventh wiring bug, found: `solve()` multiplied every shape target by
`W_JOINT`, which is 0. Last session I changed where the middle target POINTS
while it was still being multiplied by zero, so byte-identical output was the
only possible result. The whole `joint_targets` path was dead code.

Wired properly (per-level weights, target offset off the surface by the link
radius so the body origin is not driven into it) the term fires -- 27k-35k
Jacobian rows against 0, on 2.2-3.4 fingers per frame -- and it does what it
was designed to do. It also does not help, and the way it fails is the useful
part. Three sweeps, all on the ungated components because `wrap_score` itself
saturates here (86-100% of frames are past its 3 mm gate, both arms, so it
returns 2.0 and one-sidedness 1.0 and can see nothing):

1. **Tighten the middle target.** Links 4->5 (binoculars) and 5->6 (mug),
   mug one-sidedness 0.669 -> 0.390. Penetration 3.2->5.9, 9.0->13.0, 5.0->8.2.
   Arm penetration stays 0.00, so W_PEN_ARM holds and the burial is finger-side.
   By link, the middle phalanx itself stays inside the allowance (0.0-2.2 mm);
   the depth lands on the TIP and the PROXIMAL link. Over-constraint: the fit
   pins the fingertip to a surface vertex at full weight while this term pins
   the middle too, and a Shadow finger cannot reach both points on the object's
   curvature the way the human's did.
2. **Release the tip where the middle contacts.** Burial relocates rather than
   leaving: tip 8.2->4.0 on the mug, below even baseline, while the middle goes
   2.2->6.0. Worst-link depth beats baseline only on flashlight. Total is
   roughly conserved -- the middle phalanx cannot reach this surface without
   burying something.
3. **Release the wrist** (a Shadow hand is bigger than the human hand that
   produced the demonstration, so a second contact might be unreachable from
   the human's wrist -- and peer's `orient_to_clear` result pointed here).
   Refuted: at w_wrist=0 binoculars and mug collapse from 5-6 links to ONE, at
   one-sidedness 0.71 and 0.49. The hand drifts off the object rather than
   wrapping it. Low penetration bought by grazing.

Every knob in this objective is a position, and a position objective has no
term that says HOLD THE OBJECT. It can only say put these points there -- and
one buried finger satisfies that, and so does one grazing fingertip. Tighten it
and links are bought with burial; loosen it and contact degenerates. There is
no setting in between because nothing in the objective distinguishes the two
failures from a grasp.

That is this project's founding claim arriving at the retarget from the inside,
and it is the fourth independent route to the same empty neighbourhood after
peer's 283-candidate sweep, the w_pen sweep, and the wrap search. The next term
needs FORCE content -- a wrench the contact set can resist -- not another point.

Shipped defaulted off (`W_MID = 0.0`), the same discipline `W_JOINT` is held to:
defaults reproduce the previous fit bit for bit, verified, and the machinery is
there for the force-based objective to use. 44 tests pass.

## Stage 2 has a number: 0.275 -> 0.850, and 20 of the 34 are real grasps

Forty references, one per object, right hand, Shadow. Retarget -> CEM wrist
search scored by a physical hold test -> hold. Hold rate:

    raw retarget       11/40   0.275
    after the search   34/40   0.850   95% CI [0.725, 0.950]

clustered by object, 4000 resamples. The retarget is a prior on where to search
and not a grasp, which is what G5 said; this is that decision rule measured at
scale rather than on a subsample.

What the 34 are, by contact count at the accepted grasp:

    <= 12 contacts (a grasp)   20/34      median 8 contacts, median drop 6.6 mm
    13-30                       8/34
    > 30 (burial)               6/34      bowl 94 contacts at 35.5 kN

So 59% of the successes are grasps, 18% are burials that the hold test cannot
distinguish from grasps, and the rest are in between. That ratio is the honest
headline, not the 0.850 -- and it is only visible because the contact count and
grip force are recorded next to the drop distance. A stage reporting hold rate
alone would have called this 85% and been wrong about a sixth of it.

**The six failures are one geometric class.** cubelarge, cubesmall,
pyramidlarge, spherelarge, spheremedium, flashlight -- every one a featureless
convex primitive, every one `inspect` intent, and all at 0-1 contacts with an
equilibrium residual of 0.99-1.00, i.e. free fall. Nothing to hook. A mug has a
handle and a rim, a bowl has a lip, a bunny has ears; a sphere has a tangent
plane everywhere and needs a real precision grasp with opposed normals, which
is exactly what a position-space objective cannot ask for. The failure class and
the missing objective term are the same finding from two directions.

### Frame 0 is the approach, not the grasp -- on the bimanual path too

Credit to the peer session for pinning this on the one-handed path. It was
costing stage 6 everything: from frame 0, `gamecontroller_play_1` dropped the
object 3.2 m, drove MuJoCo to NaN in QACC, and reported a mean tracking error of
312 KILOMETRES. From the first frame that holds on its own: drop 6.6 mm, held,
131 mm. Added `BimanualTracker.grasp_frames`.

That 312 km deserves its own line. It was an integrator artifact being reported
as a measurement -- a mean over frames that came after the object was already
lost. `BimanualTracker.rollout` now stops once the object passes 30 cm from the
reference and reports how far it got, so a failure says "131 mm over 8% of the
clip" instead of a number that looks like a tracking error and is not one.

### First honest bimanual number

Same reference, same grasp search, the left hand PARKED rather than deleted so
the model and therefore the contact solver's problem are identical:

    gamecontroller_play_1   two-handed  131.0 mm over 8% of the clip
                            one-handed  647.7 mm over 1%

That is the project's claim as a measured gap rather than an assertion, on one
reference so far.

### The search degrades its own best input

Caught by the peer session in my own results file: on `apple_eat_1` the raw
retarget already held at a drop of 0.23 mm, with 9 contacts and an equilibrium
residual of 0.034x -- the one genuine equilibrium grasp the sweep was handed.
The CEM search moved it OFF that, to 5.17 mm.

Small in absolute terms and it still counts as a hold, but it is the same defect
as the burial, seen from the other side. The score cannot tell a grasp from a
burial, so it also cannot tell that it already had a grasp; drop distance of
0.23 mm and 5.17 mm are both "held", and nothing in the objective prefers the
first. A search that can degrade its best input is not selecting for the thing
it is supposed to be selecting for -- it is wandering inside a level set, and
the burials are simply where the wandering ends up when the level set is wide.

Note for anyone re-running: the first candidate tried is always the unperturbed
retarget, so the search CAN only improve its own score. That it got worse on the
quantity we care about while improving its own score is the point.

## Stage 6 has a number, and it discriminates

Eight bimanual references, one per object, both hands Shadow, started from the
first frame that holds on its own. Four have such a frame; all four hold after
the two-handed grasp search. Each was then run again with the left hand PARKED
rather than deleted, so the model and the contact solver's problem are identical
and only the hand's reach changes:

    reference               two-handed            one-handed
    gamecontroller_play_1    97.4 mm / 14%        2496.7 mm /  1%
    binoculars_see_1        112.0 mm / 10%         138.9 mm /  9%
    bowl_drink_1             64.5 mm / 54%        2392.9 mm /  1%
    teapot_pass_1           115.4 mm / 17%        2498.5 mm /  2%
    median                  104.7 mm / 16%        2444.8 mm /  1%

Two hands track further on 4 of 4. What makes this worth trusting is that it is
NOT uniform: `binoculars_see_1` shows essentially no two-handed advantage --
112 mm over 10% against 139 mm over 9% -- and that is correct. Binoculars are
held one-handed all the time; the second hand in that clip steadies rather than
carries. The three that collapse without it are a game controller held for
two-thumb use, a bowl, and a teapot being passed. A measure that said "two hands
always win" would be measuring the harness; this one says two hands win where
the task needs them, which is the claim.

The percentages are the honest part and they are low: 16% of the clip at the
median. These are feedforward rollouts truncated at the frame the object is
lost, with no tracking controller -- stage 6 evaluated at stage 2's level of
machinery. The gap is the result, not the absolute number.

Four references have no frame that holds at all: camera, mug, flute, doorknob.

### Three refuted hypotheses about the stage 2 -> stage 3 gap

Stage 2 holds 34/40. Stage 3 discarded 4 of 6 references as having "no graspable
frame". Something is lost at the boundary, and it is none of the following:

1. **That stage 3 never closes the grip.** True as a description -- `reset_at`
   places the retargeted angles, which are an open hand -- and irrelevant.
   Adding `adopt_grip` to carry stage 2's achieved angles across changes the
   graspable-frame count by nothing: 0 -> 0 on phone_call_1 and banana_eat_1,
   23 -> 23 on apple_eat_1, 27 -> 27 on bowl_drink_1, and 6 -> 3 on
   binoculars_lift, which is worse.
2. **That the grip simply needs establishing in the tracking scene.** Worse than
   nothing: `reset_at(k, grip=8.0)` takes `gamecontroller_play_1` from 30
   graspable frames to 1. The mocap body is welded to the hand and closing
   against that weld is a different problem from closing on a hinge base.
3. **That the two scenes disagree about the grasp itself.** They mostly do not.
   The same achieved state, transferred by palm pose and joint values, survives:
   apple_eat_1 at 30.9 mm on 7 contacts and 10.1 N against 875 N in the scene
   that produced it, banana at 43.9 mm, bowl at 17.0 mm. Only phone_call_1 fails
   outright, at 0 contacts.

So the grip transfers, closing it again hurts, and carrying it across does not
buy frames. The gap is real and unexplained, and it is the thing standing
between a stage 2 that works and a stage 3 that runs on more than two clips.

Three of my probes in this investigation were wrong before any of the above was
true -- a joint-name map that returned 0/29 because the mocap scene prefixes
every joint with `hand_`, a transfer of the seed q instead of the achieved state
which sent an open hand and read 0 contacts, and the grip hypothesis itself.
The first two produced dramatic, publishable-looking numbers (3486 mm! the
scenes disagree completely!) that were entirely my own error. Recording that
because the pattern is the point: in this pipeline a large effect is evidence of
a bug in the measurement until it survives being measured a second way.

## The stage 2 -> stage 3 gap: five hypotheses, and the one that was right

Stage 2 holds 34/40. Stage 3 discarded 4 of 6 references as having "no graspable
frame", which sets the sample size for every stage after it -- at a one-in-three
survival rate the cost of stages 4 and 5 triples. Five hypotheses, four dead:

1. **Stage 3 never closes the grip.** True as a description and irrelevant.
   `adopt_grip` carries stage 2's achieved angles across and changes the count
   by nothing: 0 -> 0 phone_call_1, 0 -> 0 banana_eat_1, 23 -> 23 apple_eat_1,
   27 -> 27 bowl_drink_1, 6 -> 3 binoculars_lift.
2. **It needs re-establishing in the tracking scene.** Worse than nothing:
   `reset_at(k, grip=8.0)` takes gamecontroller_play_1 from 30 frames to 1.
3. **The scenes disagree about the grasp.** Mostly not. The same achieved state
   transfers: apple at 30.9 mm on 7 contacts, banana 43.9 mm, bowl 17.0 mm.
4. **The PRESS is left behind** -- a position servo makes force from the gap
   between target and angle, and transferring the achieved angles as both
   commands a hand that touches without pressing (apple carried across at
   10.1 N from a grasp holding at 875 N). Real defect, correctly fixed by
   `adopt_press` with all 20 finger actuators mapped by name, and it buys
   nothing: 0 -> 0, 0 -> 0, 23 -> 23, 27 -> 27, 3 -> 4.
5. **The weld is compliant and the hand gets pushed.** No: palm drift during a
   hold is 1.8-3.9 mm on both a reference that holds and one that drops.

Meanwhile the hand is demonstrably in contact the whole time -- phone_call_1
makes 3 to 11 hand-object contacts at 2-12 mm penetration with the palm 69-110
mm from the object. It touches, it presses, and it drops.

**What was actually missing: the wrist offset.** Stage 2's CEM search optimises
the WRIST POSE; that is its entire output. Every transfer above carried fingers
-- angles, then servo press -- and left the one thing the search produces behind.
Applying it with `apply_wrist_offset`:

    reference           retarget   +wrist   +wrist+grip   offset
    phone_call_1               0        1             5   17.2 mm
    banana_eat_1               0        2             2  -17.7 mm
    alarmclock_lift            0        8             4   12.9 deg
    binoculars_lift            6        8             2  -15.5 mm
    apple_eat_1               23       23            23   exactly 0
    bowl_drink_1              27       27            27  -15.7 mm

Three references go from ZERO graspable frames to some. `apple_eat_1`'s offset
is exactly zero -- the unperturbed retarget won its search -- and its count does
not move, which is the control this needed. And the wrist alone beats the wrist
plus the grip, consistent with hypotheses 1 and 4: binoculars_lift is 8 with the
offset and 2 once the fingers come too.

So stage 3 now seeds from `results/stage2_grips.json` rather than re-searching,
and falls back to its own search when a reference is not in the sweep. Stage 2
searches in the FITTING scene where closing works; stage 3's own search runs in
the mocap scene where it does not, which is the same asymmetry as hypothesis 2
and remains unexplained at the mechanism level.

Four wrong guesses before the right one, each of which sounded obvious while I
was writing it. The one that worked was the only one I had not thought to check
because it was the part I assumed was already being carried.

### Lazy imports make a shared working tree a version-skew hazard

A peer session's 80-minute training run crashed in its end phase with

    AttributeError: 'HoldResult' object has no attribute 'eq_place'
      grasp.py:157  eq = float(res.eq_place)

and nothing was wrong at HEAD. `eq_place` was added to `HoldResult` in track.py
at 22:24; that process had loaded track.py at 22:04. It survived anyway, because
`synthesize_grasp` does `from oppdef.human import grasp as G` LAZILY inside the
method -- so the first call to it, at 23:26, imported the NEW grasp.py against
the OLD HoldResult already resident in memory. New reader, old class.

Two agents editing one working tree while long jobs run makes this routine
rather than exotic. A module imported at the top is pinned at process start and
the process is at least self-consistent; a module imported inside a method is
whatever is on disk the first time that line runs, which may be an hour of
commits later. The failure surfaces at the end of a long run, which is the worst
possible time, and it looks like a bug in the code rather than in the clock.

Mitigations, in order of value: save before you evaluate, so a crash in an end
phase cannot cost the training; write results incrementally; and do not edit
`src/` while someone's job is running. The peer's run survived on the first of
those. Mine survived on timing alone -- every src/ edit happened before launch,
which was luck rather than discipline.

## Stage 7 was not working, and I repeated that it was

I told the user stage 7 was "already working and untouched", carrying that claim
forward from a summary rather than opening the file. The stored result is:

    seq mug_drink_1, start 0, ONE reference
      truth   101.5 mm mean, 152.6 final, held 0.22, dropped TRUE
      depth   637.3 mm mean, 8927.7 final, held 0.02, dropped TRUE
      noise   102.5 mm mean, 128.0 final, held 0.10, dropped TRUE

Every condition drops the object. `start: 0` is the approach frame -- not
hardcoded, the script asks `grasp_frames()` and that is what came back -- and
the second reference in its default list returned no graspable frame at all, so
a stage designed to compare three pose sources ran on one clip that failed in
all three.

The comparison it exists to make is still the right one, and it is worth saying
why: if `depth` degrades tracking and `noise` of the same magnitude does not,
the estimator's error is structured and the fix is perception; if both degrade
equally the controller is merely pose-sensitive and the fix is control. On the
stored numbers depth is 6x worse than noise of the same size, which would be a
real finding -- structured error -- except that the truth condition drops the
object too, so all three are measuring the same failure and none of them is
measuring perception.

Fixed the same way as stage 3: seed the wrist from stage 2, which is what gives
these references a graspable frame in the first place. The stage now also
records how many graspable frames it had and the contact count and grip force of
the seed it started from, so a perception number cannot be read as a perception
result when it is a burial being tracked.

The lesson is the cheap one. I had the file open in the same session where I
found that frame 0 is the approach and that the gate was dropping references,
and I still asserted the stage worked because a summary said so. One `cat` of
the results file would have caught it.

## The 0.850 hold rate does not reproduce across code versions

Two stage-2 sweeps share five references. Three of the five disagree:

    reference              sweep A                     sweep B
    apple_eat_1            raw   0.2 mm -> 5.2, n= 9   raw 727.3 mm -> 4.5, n=18
    cubelarge_inspect_1    raw 4569   mm -> FAILED      raw 4569.8 mm -> 7.1, n= 3
    phone_call_1           raw 4940   mm -> 8.7, n= 3   raw 4940.6 mm -> 15.3, n= 6

`cubelarge_inspect_1` is one of the six failures the 0.850 headline is computed
against, and in the second sweep it holds at 7.1 mm on 3 contacts. The raw
column is the UNSEARCHED retarget pose, with no CEM involved, so this is not
search noise.

The hold test itself is deterministic. Same reference, same pose, three fresh
environments and three reuses of one, in a single process:

    apple_eat_1           drop 5.17 mm, 9 contacts, 874.9 N, eq 1.108   x6
    cubelarge_inspect_1   drop 3080.60 mm, 0 contacts, 0.0 N            x6

Identical to the last digit every time, and both differ from BOTH sweeps. So the
simulator is reproducible and the two sweeps were running different code. Sweep
A ran for thirty minutes while I was editing `src/`, which is the version-skew
hazard recorded above -- except that there it cost a peer a crash, which is
loud, and here it silently changed a headline number.

What this means for the 0.850, stated plainly: the rate is real in the sense
that the search does move most references from dropped to held, but the exact
figure and the exact membership of the six-failure class belong to a code state
that no longer exists, and the featureless-convex-primitive story is weakened by
cubelarge holding on a rerun. It needs one clean sweep at a frozen HEAD before
it is quoted anywhere. The 20-of-34-are-grasps breakdown carries the same
caveat.

I verified the middle-phalanx change was behaviour-neutral by comparing the new
code's default against the new code at `w_mid=0`, which is not a test of
anything -- both were the same build. The comparison that mattered was against
the previous commit and I did not make it.

## Stage 7 is blocked on stage 4, and the seeding proved it rather than fixing it

Seeding the wrist from stage 2 did what it was supposed to: every reference now
has graspable frames where the stored run had one clip and one failure. Five
references, frames found: hammer 4, knife 1, mug_drink_2 1, apple 6, phone 7.

The perception comparison is still meaningless, and now it is clear why.

    reference        truth      depth      noise     held (truth)
    hammer_use_2      1801 mm    1682 mm    3159 mm     0.33
    knife_lift        2746       3700       1611        0.04
    mug_drink_2      21618      18240      40774        0.04
    apple_eat_1      18962      35139      20130        0.03
    phone_call_1     60103      19060      17289        0.09

Errors in METRES in every cell, and the ordering is incoherent -- `depth` beats
`truth` on phone_call_1 by a factor of three and on hammer_use_2 slightly, which
cannot happen if the numbers are measuring pose error, because truth has none.
They are measuring where the object came to rest after being dropped. A stage
whose whole design is a three-way comparison cannot be run when all three arms
fail for a reason none of them is about.

The dependency is structural rather than a bug. Stage 7 swaps the pose source
under a FROZEN controller, and the frozen controller here is the feedforward,
which this repository established long ago does not hold a grasp through motion
-- that is the entire reason stages 3 and 4 exist. Measuring the perception gap
on the feedforward asks what a broken controller does with worse information.

So stage 7 should consume stage 4's DISTILLED policy, not the feedforward, and
it cannot produce a meaningful number until that policy exists. That is a real
ordering constraint in the pipeline that was not visible while the stage was
running on one clip that happened to fail quietly.

The stored numbers from before -- 101.5 mm truth against 637.3 depth, which
looked like structured estimator error and was carried onto a front page -- were
the same failure at a smaller magnitude, on a single reference started at the
approach frame.

## Stage 3, first two rows: what tracks is penetration

Ordered so the cleanest seeds train first, which put the answer 4.5 hours earlier
than inventory order would have.

    reference          seed class          PPO        end pen   end con   end grip
    flashlight_on_2    THIN   (0.6 N)     112.9 mm    4.86 mm      3       207 N
    hammer_use_2       GRASP  (20.7 N)  80250.1 mm    0.00 mm      0         0 N

`hammer_use_2` is the best seed stage 2 produced -- 4 contacts, 20.7 N on a 2 N
object, equilibrium 0.04, the only reference in the set that is a grasp by force
as well as by geometry. The policy trained from it LETS GO. Eighty metres.

Read with the peer session's mug 2x2, which found a policy trained on the raw
6 mm / 119 N grasp dropping from every start while both policies tracked the
buried initial condition at 22-23 mm, this is one coherent statement:

    burial-seeded references track        (mug 22.2 mm at 1330x weight)
    grasp-seeded references drop          (hammer, 80 m, 0 contacts)
    thin-seeded ones re-bury to hold on   (flashlight, 0.6 N -> 207 N, 4.86 mm)

What tracks in this pipeline is penetration. The policy either has burial handed
to it, manufactures it, or fails. Nothing in the reward asks for a grasp, and a
policy cannot invent one from a 20 N contact set it was given.

**Hammer's end state is the trap, and it arrived unprompted.** 0.00 mm
penetration, 0 contacts, 0 N. On a penetration criterion alone that is the
cleanest row in the run; it is the object lying on the floor. This is why the
readout requires tracking under 50 mm AND ending un-buried before it will call a
row a tracking result, and it is the same both-ends problem that killed
penetration depth as a grasp metric and that `equilibrium_residual` was built
for. Any penetration penalty added to the PPO reward makes hammer's behaviour
OPTIMAL unless something prices losing the object at the same time.

So the missing reward term is not a penalty on penetration. It is a price on
letting go -- and the penalty, alone, would make the failure worse while making
the metric look better.

### Deferred src/ fixes (found by a peer's test audit, deliberately not applied yet)

`src/` is frozen while g9 runs -- editing it mid-run is the version-skew hazard
that silently moved the 0.850 earlier tonight -- so these are recorded rather
than fixed:

1. `src/oppdef/viz/floor_poses.py:14` imports `tip_ids, joint_set` from
   `oppdef.hands.axis`, removed in 8cc4b63 on 13 Sep. Nothing imports it and no
   test covers it, so it has been dead for three days: the only import failure
   across all 88 non-retracted modules.
2. **`src/oppdef/human/track.py` imports `experiments.tracking.grab_inventory`
   at lines 569 and 1667.** Package code depending on `experiments/` being on
   sys.path. It works under the Makefile's `PYTHONPATH := $(CURDIR)` and would
   break under a plain `pip install -e .`. The dependency points the wrong way:
   `grab_inventory` supplies `contact_mask` and `longest_run`, which are data
   routines the library needs, so they belong in `oppdef/human/` with the
   experiment importing them rather than the reverse. This is the one worth
   doing properly.
3. `src/oppdef/bench.py:3` cites `experiments/matched.py` and `experiments/g1.py`,
   both now under `grasp_metrics/`. Cosmetic.

Also: the Makefile sets `PYTHONPATH := $(CURDIR)`, the repo root, not `src`. In
a git worktree that silently tests the MAIN checkout's `oppdef` through the
editable install, so a worktree needs `PYTHONPATH=src:.` explicitly or its test
results describe the wrong code.

### The burial effect is confounded with training-set size

Three stage-3 rows in, the binding constraint is not seed class but how much
data each policy got:

    reference          seed class   ppo          grasp frames   transitions
    flashlight_on_2    THIN         112.9 mm          2             148
    hammer_use_2       GRASP      80250.1 mm          4             566
    knife_lift         thin-end      87.7 mm          1              25

Every one of these policies was trained from a handful of start states, because
`grasp_frames` returns single digits for them. The burial-seeded references
coming later in the run have far more: gamecontroller 30, bowl 27, apple 23.

So when those rows track well -- and the mug 2x2 says they will -- there will be
TWO differences between them and hammer, not one. They start buried, and they
are trained on five to thirty times as many start states. The burial effect and
the sample-size effect run in the same direction, and this run cannot separate
them.

Writing it down before the rows land, because afterwards it will be tempting not
to. The clean experiment is a burial-seeded reference trained on only the number
of start frames its grasp-seeded counterpart had, or a grasp-seeded reference
with its start set padded the way the peer session padded the mug's (they
overrode `grasp_frames` and trained on all 23 candidate starts rather than the
2 it accepted). The second is cheaper and is the one to run.

Note also that knife's 87.7 mm mean is unexplained. Its end state matches
hammer's exactly (0.00 mm, 0 contacts, 0 N) but its error is three orders of
magnitude smaller, over roughly thirteen seconds of simulated time in which a
released object would fall hundreds of metres. Either it was held until near the
end, or it is resting on something, or the reference barely moves so the error
against it stays small after release. Not measured; the row should carry no
weight either way.

## Both scenes pin the hand, and burial is self-cancelling

A peer session proposed that the mocap weld is the structural difference from
DexTrack: with the hand's base effectively rigid, an object can never push the
hand out, so penetration costs nothing and only moves the object. That would
explain "penetration is the grip", the closing failure against the weld, and the
burial-tracks result in one stroke. Measured, the picture is broader and the
predicted fix is weaker than it looks.

**Both scenes pin the hand.** The fitting/hold scene is not a free base:

    x_act / y_act / z_act     kp = 4000 N/m,  kv = -200,  force UNLIMITED
    rx_act / ry_act / rz_act  kp =  200,      kv =  -20,  force UNLIMITED
    mocap weld                solref [0.01, 1.0]
                              solimp [0.9, 0.95, 0.001, 0.5, 2.0]

(I first checked for actuators named `x`, `y`, `z` -- those are the JOINT names
-- found none, and nearly recorded "the hinge base is a damped free body". The
actuators are `x_act` and friends. Fifth wrong reading of the night caught by
looking twice.)

So stage 2's grasp search runs against a stiff servo-held base and stage 3
against a stiff weld. The mechanism applies to the SEARCH as much as to
tracking, and would explain stage 2's 35 kN bowl as well as stage 3's burial
tracking.

**But burial does not push.** `total_grip` sums the MAGNITUDES of the normal
forces; it is not a net push. A buried hand has dozens of contacts pointing in
opposing directions and their vector sum is near zero -- which is precisely what
the equilibrium residual found when a settled 94-contact, 35.5 kN bowl read 0.00
net force against a 1.90 placement reading. At kp = 4000 N/m a genuine 35 kN net
push would displace the base by metres; the object moves 1.6 mm.

So the forces are already cancelling and a softer base gives them less to push
against, not more. Prediction, recorded before the experiment: softening the
weld changes the burial numbers modestly and does not stop burial from tracking.

Worth running anyway -- it is cheap, it is the clearest structural difference
from DexTrack anyone has named, and my predictions have been wrong four times
tonight on the stage 2/3 gap alone. What would make it decisive is logging the
NET force on the hand rather than the summed magnitude: the same vector sum the
equilibrium residual takes over the object, taken over the hand. Nothing in the
pipeline measures that today, and it is the quantity base compliance acts on.

### Refuted: the forces do NOT cancel while tracking

The prediction above is wrong and the peer session measured it. Net force on the
hand during tracking, from the per-frame residuals persisted in
`figures/physics_*.json` (net contact force on the hand is the negative of that
on the object, so away from gravity's 1x the residual IS the net hand force in
object weights):

    rollout       median   p10    p90    min    net N    summed |F|   ratio
    binoculars      158x    84x   347x    57x    310      6766        0.05
    bowl            184x   133x   272x    38x    361      2207        0.16
    camera          308x   124x   650x    36x    604      4502        0.13
    mug (PPO)       202x    85x   282x    17x    397      1486        0.27

Zero frames of 541 under 2x weight. Cancellation takes 5-27% of the magnitude,
not all of it, and 300-600 N survives as a net push -- which at the base servos'
kp = 4000 N/m is a ten-centimetre displacement. The weld does continuous
mechanical work holding the hand inside the object, every control frame.

Where my prediction failed: I took a measurement of a SETTLED hold -- the
94-contact bowl reading 0.00 net at rest -- and generalised it to tracking,
which never settles. The weld re-drives the hand to a fresh reference pose every
control frame, so tracking is a sequence of placements. I had written the
placement-versus-settled distinction into a docstring two hours earlier and then
reasoned from the settled number anyway. Same shape as reading frame 0 as the
grasp.

This is the strongest structural account anyone has given of "penetration is the
grip", and it makes the compliant-base experiment decisive rather than
suggestive by the criterion I set for it.

Two caveats to carry with it. The residual sums over every contact involving the
OBJECT, so if an object ever touches something other than the hand those forces
are not hand forces and the third-law step fails for them; scene.py has no floor
or table, so this looks clean, but a bimanual clip would break it. And the
readings are taken after each control frame's substeps rather than at reset --
which is the right place for this argument, since it is the state the weld holds
the hand in, but a reader will assume reset unless told.

And the caveat that is really the finding: all four rollouts are burial-class.
There is no clean-grasp tracking rollout in this repository to contrast against.

### Un-refuted: the forces DO cancel, and the rigid base is ruled out

The refutation above is itself retracted, by measurement, and the original
prediction stands. Live net force on the buried hand during tracking is **0.4x
object weight at the median**, p90 2.8x -- not the 300-600 N the previous entry
recorded.

That 300-600 N was a replay artifact. `render_tracking`'s `--render-only` path
does `d.qpos[:] = state; mj_forward` for every frame -- a fresh placement with
no velocity and no warm-started constraints -- and then `meta.update(diag)`
overwrites the live diagnostics that capture had recorded. A fresh placement
reads hundreds of times weight by construction, which is exactly the
placement-versus-settled distinction already in this file; the manifests'
per-frame residual column is a replay value wearing a live label. Credit to the
peer session for finding it in their own number.

**The weld sweep then rules the rigid base out cleanly.** OLD mug policy from
frame 5, weld time constant raised on the built model, with the no-contact
following-error control at each setting:

    tc     control follow    buried tracking   end state              base off target
    0.01   0.41 / 1.11 mm    23.2 mm 111/111   10.83 mm, 12 bodies    0.44 mm (= control)
    0.03   8.9 / 29 mm       29.5 mm 111/111   10.80 mm, 12 bodies    9.56 mm (= control)
    0.10   54 / 159 mm       252 mm            10.82 mm, 12 bodies    58 mm   (= control)
    0.30   238 / 443 mm      642 mm            10.81 mm, 12 bodies    246 mm  (= control)

At every stiffness the base's displacement equals its contact-free following
error, so contact displaces the base by nothing. At twenty times the compliance
burial tracks the same and ends identically -- 10.80 mm against 10.83 mm on
twelve bodies. The tc 0.10 and 0.30 cells are uninterpretable for burial because
the following error swamps the signal, which is what the control was added to
detect.

So the burial is a GEOMETRIC EQUILIBRIUM: the contact set is self-cancelling and
holds itself in place without the base resisting anything. It is not a stiffness
question at either stage, and the DexTrack-style free-floating base would not
fix it. What remains on the table is unchanged: the retarget objective, the
reward's silence about grasping, and sample scale.

Three retractions deep on one question in two hours, between two sessions, and
the thing that settled it each time was a measurement rather than an argument.

## Row 6 breaks "burial tracks": it is start count that differs

Six stage-3 rows, ordered by tracking error:

    reference          PPO         seed class   end pen  end con  end grip  starts
    mug_drink_2           5.7 mm   BURIAL        13.18     15      4834 N     27
    knife_lift           87.7      grasp          0.00      0         0        1
    flashlight_on_2     112.9      thin           4.86      3       207        2
    hammer_use_2      80250.1      grasp          0.00      0         0        4
    phone_call_1     195633.5      grasp          0.00      0         0        9
    mouse_use_1      291168.2      BURIAL         0.00      0         0        2

`mouse_use_1` is the row that matters. Its seed is 19 contacts at 4335 N --
2210x the object's weight, so BURIAL class by force, though contact count alone
would have called it mixed. It drops harder than anything else in the run.

So two burial-class rows point in opposite directions, and what differs between
them is the number of start frames: 27 against 2. The burial-seeded median is
now 145,587 mm, which is not a story about burial at all.

**"Burial-seeded tracks, grasp-seeded drops" does not survive row 6.** The
honest six-row statement is narrower: exactly one row of six both tracks well
and ends still holding the object, and that row is simultaneously the most
buried seed and the best supported by a factor of three. Everything else either
lets go or re-buries a thin contact set. Which of those two properties does the
work is exactly what this run cannot say, and mouse is the proof -- it has the
burial and not the data, and it fails.

Note also that error alone misleads here: knife at 87.7 mm and flashlight at
112.9 mm look far better than hammer's 80 m, and knife ends with zero contacts
just as hammer does. Reading the error column without the end state would rank a
dropped object above a held one.

That promotes the 2x2 from tidy-up to the load-bearing experiment:

                            few starts             many starts
    hammer (grasp seed)     80,250 mm (4) done     queued, all candidates
    mug_drink_2 (burial)    queued, capped to 4    5.7 mm (27) done

Capping the burial row to four starts is the cleaner half -- rescuing hammer can
fail for reasons unrelated to sample size, whereas removing data from the one
row that works tests one thing only.

### Pre-registered: what rows 7-9 should show

Start-frame counts computed in advance by a peer session, the way g9 computes
them (stage-2 offset applied, `grasp_frames` at stride 5):

    camera_takepicture_2    T=79   10 of 16   seed 32 con,  5,800 N ( 2956x)
    gamecontroller_play_1   T=62    8 of 13   seed 57 con, 14,042 N ( 7157x)
    bowl_drink_1            T=131  27 of 27   seed 94 con, 35,522 N (18105x)
    binoculars_see_1        T=155   1 of 31   unseeded, stage 2 FAILED -- read nothing

The 30 I quoted earlier for gamecontroller was a different configuration: it came
from a probe that called `rt.synthesize_grasp()`, this stage's own search, not
the stage-2 wrist offset that g9 applies. Both numbers are right for what they
measured; g9's own `grasp_frames` field settles which applies to the row.

bowl at 27 is a `mug_drink_2` replicate and does not discriminate. camera at 10
and gamecontroller at 8 sit between hammer's 4 and mug's 27, so they are the
informative rows.

**My prediction, recorded before they run.** Both camera and gamecontroller
track well and end buried, on 8-10 starts. That would weaken start count as the
explanation, because mouse failed on 2 and these would succeed on 8, putting the
threshold somewhere between 2 and 8 rather than near 27 -- and it would leave
seed depth as the better predictor, since these two seeds are 2956x and 7157x
object weight against mouse's 2210x and mug's 269x.

If instead either drops, start count gains and the 2x2 becomes the only thing
that can settle it.

Stating it in advance because on this question I have now been wrong about the
mechanism four times and right once, and a prediction written after the fact is
worth nothing.

## WITHDRAWN: the stage 3 seed-class analysis. 7 of 10 rows got another subject's grasp

GRAB has **80 sequence names that exist under more than one subject**.
`camera_takepicture_2` is one: s1's version holds for 161 frames, s2's for 76.

My `stage2_grips.py --seqs` handler resolves bare names with
`by = {r["seq"]: r for r in rows}`, which silently keeps one subject, and
`g9_ppo_distill` looks seeds up the same way -- `grips = {x["seq"]: x for x in
rows}`, then `if row["seq"] in grips`. So g9 selected its reference from the
inventory, loaded the clip for THAT row's subject, and applied a wrist offset
computed on a different subject's version of the same-named sequence.

    reference                g9 subj   seed subj   match
    mouse_use_1              s1        s2          NO
    phone_call_1             s1        s2          NO
    gamecontroller_play_1    s1        s2          NO
    camera_takepicture_2     s1        s2          NO
    binoculars_see_1         s1        s1          yes
    hammer_use_2             s1        s2          NO
    knife_lift               s1        s1          yes
    flashlight_on_2          s1        s2          NO
    bowl_drink_1             s1        s1          yes
    mug_drink_2              s1        s2          NO

Caught because a peer's pre-registered start count (10, computed on s2) did not
match the row's own `grasp_frames` field (28, computed on s1). Neither number
was wrong; they described different clips.

**What is withdrawn.** Every seed-class label in the stage 3 table -- grasp,
burial, thin -- describes a grasp belonging to a different recording than the
policy trained on. "hammer_use_2, grasp seed, 20.7 N" is s2's hammer grasp
applied to s1's hammer clip. The runs are valid training runs of "retarget plus
an arbitrary wrist perturbation", and nothing more. The grasp-versus-burial
story, the burial-tracks/grasp-drops headline, the mouse counter-example and my
pre-registered prediction all rest on those labels and go with them. Only knife,
bowl and binoculars are correctly labelled, and knife has one start frame while
binoculars failed stage 2 outright.

**What survives**, because it is live measurement of whatever state was actually
reached rather than a claim about its provenance:

    mug_drink_2      5.7 mm, ends 13.18 mm inside, 15 contacts, 4834 N (2466x)
    camera_tp_2     34.6 mm, ends  9.32 mm inside,  8 contacts,  509 N ( 260x)
    flashlight     112.9 mm, ends  4.86 mm inside,  3 contacts,  207 N
    knife           87.7 mm, ends 0.00 / 0 / 0 -- object on the floor
    hammer        80250.1 mm, ends 0.00 / 0 / 0
    phone        195633.5 mm, ends 0.00 / 0 / 0
    mouse        291168.2 mm, ends 0.00 / 0 / 0

and with it: no row tracks under 50 mm while ending un-buried; four of seven
finish under 3 mm of penetration because the object is on the floor; and reading
the error column alone would rank knife's dropped object above camera's held one.

Fix is to key on `subject/seq` everywhere a seed is looked up, and rerun. The
shape of the bug is the night's recurring one -- a silent key collision that
produced plausible numbers for two hours -- and the thing that exposed it was
two sessions computing the same quantity independently and getting different
answers.

### The one correctly-labelled tracking row is a burial at 14,850x weight

`bowl_drink_1` is subject-matched (s1 seed, s1 clip), so unlike seven of the ten
its seed label describes the reference it trained on:

    seed        94 contacts, 35,522 N (18,105x object weight), equilibrium 1.90
    tracking    8.2 mm
    end state   17.50 mm inside, 81 contacts, 29,118 N -- 14,850x weight
    starts      27 of 27 candidates accepted

It tracks better than all but one row in the run and ends deeper inside the
object than any of them. This is the clearest single instance of "the tracking
number is measuring the contact solver": a hand 17.5 mm inside a mug-sized bowl,
pressing with the weight of a small car, following the reference to 8 mm.

Of the three subject-matched rows, this is the only one that carries
information. `knife_lift` is matched but trained on ONE start frame and its
87.7 mm is unexplained. `binoculars_see_1` is matched and failed stage 2
outright. So the correctly-labelled evidence is: one burial seed, which tracked
and stayed buried; one grasp seed with almost no training data, which let go.

That is not enough to support the seed-class claim and it is not nothing. It is
consistent with it, on n=1 per arm, with the grasp arm confounded by sample
size -- which is exactly what the corrected rerun and the 2x2 exist to settle.

## Stages 4 and 5 ran end to end. The distilled policy does not transfer.

Nine policies trained, distilled into one network, held out by OBJECT:

    held out          feedforward    its own PPO    DISTILLED
    hammer_use_2          906.0 mm      80250.1 mm     847.6 mm
    knife_lift           2734.3          87.7           81.1
    camera_takepicture_2    36.0          34.6         1030.3

    distilled beats feedforward on 2/3
    distilled within 2x of that reference's own PPO on 2/3

Read the headline counts and then ignore them, because the row that matters is
camera. It is the one held-out reference where the feedforward already worked --
36.0 mm, the best feedforward number in the set -- and the distilled policy
turns it into 1030.3 mm, twenty-nine times worse. On the two references where
feedforward had already failed at 906 and 2734 mm, the distilled policy also
fails, at 848 and 81 mm, and "beats feedforward" is a comparison between two
failures. Nothing here is under the 50 mm bar except knife's 81.1 mm, which is
close to it and belongs to the row with a single start frame and an unexplained
error.

So the stage 4-5 mechanism works -- policies harvest, a network fits them, it
evaluates on unseen objects -- and the thing it produces does not transfer. That
is a real result about this pipeline and not a bug: the policies being distilled
are the stage 3 policies, seven of ten of which were seeded from another
subject's grasp, and of the ones that "succeed" none ends holding the object
un-buried. A network fitted to that mixture has been shown burial and dropping,
and it reproduces both.

Stage 3's population statement, on all nine rows: **no row tracks under 50 mm
while ending un-buried. Five of nine finish under 3 mm of penetration with a
median of 0.00 mm, because the object is on the floor.**

Also worth noting as a small vindication of the subject-keying fix: the readout
now prints `?` for the seed class of every row in this run, because the stored
rows predate the subject field. It refuses to label rather than mislabel, which
is what it should do.

## The corrected seeds invert the population

Subject-qualified seeds for g9's ten picks, against the contaminated ones the
first run actually trained on:

    reference                s2 (used, wrong)              s1 (correct)
    mouse_use_1              19 con   4335.4 N BURIAL       4 con     3.6 N grasp
    phone_call_1              6 con      2.8 N grasp        3 con     7.5 N grasp
    gamecontroller_play_1    57 con  14042.2 N BURIAL       5 con     1.9 N thin
    camera_takepicture_2     32 con   5800.5 N BURIAL       2 con     6.0 N grasp
    binoculars_see_1          5 con    379.3 N FAILED       5 con   379.3 N FAILED
    hammer_use_2              4 con     20.7 N grasp        7 con    29.9 N grasp
    knife_lift                4 con      2.1 N grasp        4 con     2.1 N grasp
    flashlight_on_2           3 con      0.6 N thin         8 con    35.4 N grasp
    bowl_drink_1             94 con  35521.5 N BURIAL      94 con 35521.5 N BURIAL
    mug_drink_2               9 con    528.2 N BURIAL       8 con  3250.8 N BURIAL

    contaminated:  5 BURIAL, 3 grasp, 1 thin, 1 failed
    corrected:     6 grasp,  2 BURIAL, 1 thin, 1 failed

The population is inverted. The run whose results were reported all night was
trained predominantly from burials; the corrected run trains predominantly from
grasps. Three references flip class outright -- mouse from 4335 N to 3.6 N,
gamecontroller from 14,042 N to 1.9 N, camera from 5800 N to 6.0 N -- and
flashlight moves the other way, from a 0.6 N touch that was below the object's
own weight to a 35.4 N grasp.

Only three rows are unchanged, and they are the three that were subject-matched
all along: binoculars (failed), knife, bowl. `mug_drink_2` stays BURIAL under
both but is deeper when correct, 528 N against 3251 N.

So v2 is not the same experiment with better labels. It is the first time this
pipeline trains a full set of policies from grasps rather than burials, which is
precisely the cell every version of the seed-class question has lacked. If all
six grasp-seeded policies drop the object, that is the cleanest statement of the
problem this project has produced. If any tracks and ends un-buried, it is the
first such row in the repository.

### Deferred: g9's internal seed label is contact-count only

`g9_ppo_distill._seed_kind` classifies on `n_contact > 30` alone, while
`g9_analysis.kind` is force-aware (burial above 30 contacts OR 200x object
weight, thin below one object weight). They disagree: v2's order line calls
`mug_drink_2` a grasp on 8 contacts where the analysis calls it a burial at
3251 N, 1657x weight.

The disagreement affects only the training ORDER and the printed mixture line,
not any measurement -- the analysis is the authority for every reported label.
Left unfixed because v2 is running and src/ and the experiment modules it loaded
must not move under it, which is the version-skew rule the 0.850 taught.

Fix after v2: give `_seed_kind` the same force-aware thresholds, or better, have
both call one shared classifier so they cannot drift again. Two implementations
of one definition is how the contact-count label survived long enough to
mislabel mug_drink_2 in the first place.

### The reward-branch benchmark: the gate does not save it

Measured by a peer session on a quiet machine, medians over three alternating
reps, 12 envs:

    gamecontroller (45 contacts, 15.4 kN)   6.743 s/iter baseline
                                            7.098 gated     (x1.053)
                                            7.148 always-on (x1.060)
    bowl (40 contacts, 11.9 kN)            13.914 baseline
                                           15.013 gated     (x1.079)
                                           15.107 always-on (x1.086)

Spread within a variant is under 0.06 s, so the 5-8% is real. Gating the
`mj_contactForce` call buys 0.7%; the rest is the Python loop over `d.contact`
reading dist, geoms and body ids for every contact, every environment, every
control step.

So the branch does not land as written, and the review call to measure before
landing was the right one: at defaults the reward would have been provably
unchanged while training ran 5-8% slower, which is the regression nobody looks
for. The fix is to vectorise the summary over the contact arrays with numpy and
gate the whole thing, not just the force call.

## The contamination produced a controlled experiment we could not have designed

`camera_takepicture_2` has now been trained twice on the SAME s1 clip with two
different wrist offsets, and nothing else differing:

    wrist offset              starts   tracking    end state
    s2, burial (32 con, 5800 N)   28    34.6 mm    9.32 mm inside, 8 con, 509 N
    s1, grasp  ( 2 con,  6.0 N)   33  2587.0 mm    0.00 mm, 0 contacts, 0 N

Same reference, same recording, same horizon, same seed, same everything except
the initial hand pose -- and the grasp arm has MORE training data, 33 start
frames against 28. The burial tracks at 34.6 mm and ends inside the object. The
grasp drops it.

This is the within-reference control that every version of the seed-class
question has lacked, and it exists only because the seq-name collision
accidentally trained the same clip from another recording's grasp. The bug cost
a night of mislabelled results and produced the one comparison that settles the
confound.

**Sample size is not the explanation.** The grasp-seeded arm has more data and
fails. The 33 start frames also exceed every row in the old run except none --
it is the largest start set in either run, on the lightest seed (2 contacts,
6.0 N, equilibrium 0.01), on the one reference where the FEEDFORWARD already
tracked at 36.0 mm.

That last detail is worth stating on its own: on this clip the feedforward
carries the object to 36 mm and a policy trained from a physically valid grasp
carries it to 2587 mm. The policy is worse than no policy. Whatever PPO learns
from a light contact set here, it is not how to keep hold of the object.

Five more grasp-seeded rows follow. If they agree, the statement is that in this
pipeline a policy can track only what it is handed buried, and that a valid
grasp is not something it can learn to keep.

### Two within-clip pairs now, and they disagree about burial

`mouse_use_1` has also been trained twice on the same s1 clip:

    wrist offset                 starts   tracking      end state
    s2, burial (19 con, 4335 N)      2   291,168 mm    0.00 / 0 / 0
    s1, grasp  ( 4 con,   3.6 N)    32   116,130 mm    0.00 / 0 / 0

Both arms drop. Set beside camera's pair --

    camera, s2 burial (32 con, 5800 N)   28      34.6 mm   9.32 mm inside
    camera, s1 grasp  ( 2 con,  6.0 N)   33    2,587.0 mm  0.00 / 0 / 0

-- the two pairs agree that the grasp arm drops and disagree about the burial
arm: camera's burial tracks at 34.6 mm, mouse's drops at 291 m. So burial is
NOT sufficient for tracking. Mouse's burial arm had 2 start frames against
camera's 28, so sample size is the obvious candidate for the difference, and
that is the one direction in which the confound survives -- burial with data
tracks, burial without data does not.

What the grasp side now has is the strongest evidence in either run:
`mouse_use_1` at 32 start frames and 1176 transitions is the **best-supported
policy trained tonight**, from a 4-contact 3.6 N seed at equilibrium 0.10, and
it drops the object 116 metres. Camera at 33 frames does the same. Two
independent references, the two largest training sets of the night, both clean
grasps, both let go.

So the asymmetry is now stated precisely: a policy given more data and a valid
grasp still drops the object, while a policy given burial sometimes tracks and
sometimes does not. Sample size does not rescue the grasp arm. It may well be
what separates the two burial arms, which is a smaller and different question.

### knife reproduces bit-for-bit, which retroactively supports the 0.850 diagnosis

`knife_lift` is subject-matched in both runs, so v2 trains it from the identical
seed the first run used. It returns 87.65951234648011 mm, 1 start frame, 25
transitions, ending 0.00/0/0 -- the old row to every digit.

It carries nothing about the seed-class question. What it does give is a free
determinism check on the whole stage-3 path -- retarget, wrist offset,
grasp_frames, PPO with its threaded pool, harvest, evaluate -- across a seed-file
rewrite and a run boundary.

That matters for an earlier entry. The stage-2 sweeps disagreed on three of five
shared references, and I attributed it to code having changed between them rather
than to nondeterminism, on the strength of an in-process repeat check. This is
the same conclusion from the other side and across processes: when the code does
not change, this pipeline reproduces exactly. So the 0.850's irreproducibility
was a version difference, as diagnosed, and the clean sweep at frozen HEAD is the
right remedy rather than a hope that the number settles down.

It also means every difference between the old run and v2 is attributable to the
seed change, since nothing else moved. That is what makes the camera and mouse
within-clip pairs interpretable at all.

## Six grasp-class seeds, six drops

v2, seeds keyed on their own subject, grasp-class rows in training order:

    reference          seed                 starts  tracking     end state
    camera_tp_2        2 con,  6.0 N        33      2,587.1 mm   0.00 / 0 / 0
    phone_call_1       3 con,  7.5 N         1     42,603.0      0.00 / 0 / 0
    mouse_use_1        4 con,  3.6 N        32    116,129.9      0.00 / 0 / 0
    knife_lift         4 con,  2.1 N         1         87.7      0.00 / 0 / 0
    gamecontroller     5 con,  1.9 N (thin)  3     51,777.5      0.00 / 0 / 0
    hammer_use_2       7 con, 29.9 N        11     32,462.6      0.00 / 0 / 0
    flashlight_on_2    8 con, 35.4 N         4        218.9      0.00 / 0 / 0

Every one ends with the object on the floor: zero penetration, zero contacts,
zero newtons. Three of them are well supported -- camera at 33 start frames,
mouse at 32 and hammer at 11 are the three largest training sets produced
tonight on either run -- so this is not an artifact of thin sampling, and the
thin rows (phone, knife at 1) agree with the well-supported ones.

The claim this supports, and its exact limits: **on GRAB with a Shadow right
hand, at 160,000 control steps per reference, a PPO policy trained from a
physically valid grasp lets go of the object, and more training data does not
change that.** A separate 845,000-step run on a valid mug grasp, from a peer
session, also dropped from every start, so the budget is not the explanation
either.

The burial arm is outside the claim. Policies that track at all (5.7-34.6 mm in
the earlier run) came only from burial-class seeds and ended with the hand 9-17
mm inside the object, but camera's burial arm tracked while mouse's dropped, so
burial is necessary in everything measured and not sufficient. v2's own burial
rows have not run yet.

The sharpest single number remains camera: the feedforward alone carries that
object to 36.0 mm, and the policy trained from its valid grasp carries it to
2,587 mm. **The policy is worse than no policy.**

### flashlight is a third pair, and it inverts

    flashlight, s2 thin  (3 con,  0.6 N)   112.9 mm   ends 4.86 mm in, 3 con, 207 N
    flashlight, s1 grasp (8 con, 35.4 N)   218.9 mm   ends 0.00 / 0 / 0

The 0.6 N seed -- less than the object's own weight, barely touching -- produced
a policy that MANUFACTURED a grip, driving contact force to 207 N and burying
4.86 mm, and held on. The 35.4 N seed, a real grasp, let go. It is the only case
in either run of a policy acquiring contact rather than losing it. Recorded as
an anomaly, not as evidence: one row, and the thin arm had 2 start frames.

## Sample size is now ruled out in both directions

v2's first burial row settles what the grasp rows could only half-answer:

    seed class   reference          starts   tracking      end state
    BURIAL       mug_drink_2 (s1)      4       26.7 mm     11.10 mm in, 9 con, 1574 N
    grasp        hammer_use_2 (s1)    11    32,462.6 mm     0.00 / 0 / 0
    grasp        mouse_use_1 (s1)     32   116,129.9 mm     0.00 / 0 / 0
    grasp        camera_tp_2 (s1)     33     2,587.1 mm     0.00 / 0 / 0

A burial-class seed with **four** start frames tracks to 26.7 mm and finishes
with the hand 11 mm inside the object at 802x its weight. Grasp-class seeds with
eleven, thirty-two and thirty-three start frames all let the object go.

So the confound is dead in both directions. More data does not rescue a grasp,
and very little data does not prevent a burial from tracking. Whatever separates
these rows, it is not how much the policy saw.

It also weakens the one place the sample-size story survived. Mouse's burial arm
dropped on 2 start frames and I recorded "burial with data tracks, burial
without data does not" as the live remaining question; mug_drink_2 tracks on 4,
which is barely more. So mouse's burial arm is more likely to be something about
that clip -- a flat mouse on a surface -- than a data threshold.

`mug_drink_2` has also now been trained from two different burial seeds on the
same clip, and both track and both end buried:

    s2 seed,  528 N, 27 starts    5.7 mm    ends 13.18 mm in, 15 con, 4834 N
    s1 seed, 3251 N,  4 starts   26.7 mm    ends 11.10 mm in,  9 con, 1574 N

Four within-clip pairs now exist (camera, mouse, hammer, flashlight) plus this
same-class pair, and across all of them the pattern that holds is the seed's
CLASS, not its start count, not its subject, and not the clip.

## v2 complete: perfect separation by seed class

Ten references, seeds keyed on their own subject, labelled from the seed file the
run actually used:

    reference              PPO          seed     con | end pen  end con  end grip
    bowl_drink_1               8.2 mm   BURIAL    94 |  17.50      81     29118 N
    mug_drink_2               26.7      BURIAL     8 |  11.10       9      1574
    knife_lift                87.7      grasp      4 |   0.00       0         0
    flashlight_on_2          218.9      grasp      8 |   0.00       0         0
    camera_takepicture_2   2,587.1      grasp      2 |   0.00       0         0
    hammer_use_2          32,462.6      grasp      7 |   0.00       0         0
    phone_call_1          42,603.0      grasp      3 |   0.00       0         0
    gamecontroller_play_1 51,777.5      thin       5 |   0.00       0         0
    mouse_use_1          116,129.9      grasp      4 |   0.00       0         0
    binoculars_see_1     131,158.2      failed       |   0.00       0         0

    from a BURIAL seed (2): median      17.5 mm
    from a GRASP seed  (6): median  17,524.9 mm

**A factor of one thousand between the classes, and not a single row crosses.**
Both burial-class seeds track and end with the hand inside the object. All six
grasp-class seeds, the thin one and the stage-2 failure end at 0.00 mm
penetration, zero contacts and zero newtons -- the object on the floor.

Eight of ten rows finish under 3 mm of penetration, which on a depth criterion
would be the cleanest run in the repository. Zero rows track under 50 mm while
ending un-buried.

Start counts do not separate the classes: the burial rows have 27 and 4 starts,
the grasp rows 1 to 33. The two largest training sets in the run are grasp-class
and both drop; the smallest burial set is four frames and it tracks.

So, for GRAB on a Shadow right hand at 160k control steps per reference:
**whether a tracking policy works is determined by whether its initial hand pose
is inside the object.** Nothing else measured here predicts it -- not the number
of start frames, not the subject, not the clip, not the object, and not the
training budget (a separate 845k-step run on a valid grasp also dropped from
every start).

And the corollary that matters for everything upstream: every tracking number
this repository has ever reported was produced from a buried initial condition.

## RETRACTED: "the policy is worse than no policy"

I compared two different initial conditions and the sentence reached the README
before v2's own feedforward number existed. v2's held-out row:

    camera_takepicture_2   feedforward 2256.4 mm   own PPO 2587.1 mm   distilled 992.7 mm

The 36.0 mm feedforward I quoted was from the FIRST run, where camera carried
the s2 BURIAL offset. From the s1 valid grasp the feedforward is 2256.4 mm. So
"feedforward 36 mm against policy 2587 mm" is feedforward-from-a-burial against
policy-from-a-grasp, and the factor of seventy measures the initial condition
changing, not the policy.

At a fixed initial condition:

    from the s1 valid grasp    feedforward 2256.4 mm    PPO 2587.1 mm
    from the s2 burial         feedforward   36.0 mm    PPO   34.6 mm

The policy is marginally worse than feedforward from the grasp and marginally
better from the burial. Both differences are noise beside the two-order-of-
magnitude gap between the conditions.

**The corrected finding is stronger than the retracted one.** The feedforward
drops a valid grasp too. So this is not "RL cannot learn to keep a grasp it was
handed" -- it is that nothing in this pipeline keeps hold of a valid grasp,
open-loop or learned, and everything holds a burial. Stage 3 was never the
variable. The initial condition determines the outcome and the controller barely
moves it.

That also reframes the six-of-six result one level up: those six policies drop
the object, and so would no policy at all. The failure is not in the tracking
stage.

I made this error ninety minutes after establishing that the initial condition
is the only thing that matters in this pipeline, by quoting a number from one
run against a number from another without checking they shared a condition. The
guard is the one I have been asking the peer session for all night: state the
condition beside the number.

## Stages 4 and 5 on correct seeds: the direction is right, the level is failure

    held out               feedforward    own PPO    DISTILLED
    camera_takepicture_2     2,256.4 mm   2,587.1 mm    992.7 mm
    knife_lift               2,734.3         87.7         87.5
    gamecontroller_play_1  175,291.4     51,777.5        450.5

    median                   2,734.3       2,587.1        450.5
    distilled beats feedforward           3/3
    distilled within 2x of own PPO        3/3

This is the reverse of the contaminated run, where distillation was 29x worse
than feedforward on camera. On correct seeds the distilled network is the best
of the three controllers on every held-out object, and it is better than the
per-reference policy that trained on that object's own data -- which is the
direction DexTrack's distillation claim predicts, reproduced here for the first
time.

And every one of those numbers is a dropped object. 450 mm, 88 mm and 993 mm are
all far past the 50 mm bar, so the honest statement is that distillation is
consistently the least bad of three failing controllers. The mechanism behaves
as designed; what it is fed cannot hold an object, so what comes out cannot
either.

That is the whole pipeline in one line. Stages 3, 4 and 5 all work in the sense
that each does what it was built to do, and none of it matters, because every
one of them is downstream of an initial condition that either buries the hand or
drops the object. Fixing stage 4 would improve nothing. The variable is upstream
of all of them.

### Five harness failures, one failure: watching the wrapper, not the worker

Tonight produced five, across two sessions, and they are the same bug:

1. `pkill -f <pattern>` matched my own shell's command line and killed it. Three
   times.
2. A peer's watcher used `pgrep -f g9_ppo_distill.py` as its exit test, and the
   watcher's own command line contained that string, so it waited on itself
   forever and never fired.
3. A peer's result watcher read `d.get("rows")` where the file's key is
   `per_reference`, so it counted zero before and after the row it was waiting
   for.
4. My chained job fired EARLY: it waited on the pid of a previous chain, and
   when I killed that chain to reorder the queue, the wait returned immediately
   and launched a 12-thread PPO run alongside a job that was already using the
   machine.
5. My chained job would have fired LATE: the peer's re-bench waited on my
   wrapper script's pid, and the wrapper does not exit when the interesting work
   finishes -- it goes on to the next job in the chain. Caught before it
   mattered only because I noticed the sweep starting.

Four of the five are one mistake: **the process being watched was not the process
doing the work.** A pattern can match the watcher; a wrapper outlives its
children; a chain's pid is not its current job's pid. The fix is to record the
WORKER's pid at launch and watch that, or better, to have the worker touch a
sentinel file when it is genuinely done and watch the file. A file cannot match
itself, cannot outlive the work, and says what it means.

The remaining one, the wrong dict key, is the same family as the `obj`/`object`
crash and the seq-name collision: an instrument that reads a name and gets
silence rather than an error.
