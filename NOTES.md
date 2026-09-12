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
