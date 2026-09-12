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
