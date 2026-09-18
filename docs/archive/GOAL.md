# opposition-deficit: goal

Written 2026-09-11. Successor to the `dextrack_vega` workstream, which spent four months
discovering one thing the hard way and should now be pointed at it deliberately.

Everything marked **[measured]** below was measured on this machine and is traceable to
`~/projects/dextrack_vega` (README, commit history, and the project log). Everything marked
**[hypothesis]** is what this project exists to earn or kill.

---

## The goal in one sentence

Show that human hand-object interaction should be retargeted as a **wrench specification on
the object**, not as a hand pose — because on any hand that cannot oppose the way a human
hand opposes, pose-fidelity retargeting is not merely lossy but *anti-correlated* with task
success, and the second arm is the repair.

## The claim we are trying to earn

Every pipeline that turns human hand data into robot motion — DexPilot, AnyTeleop,
`dex-retargeting`, DexTrack's MANO→Shadow step, and the retargeting stage at the end of
FlowHOI/StableHand — optimizes the same objective: make the robot's keypoints match the
human's. That objective is reported with a keypoint error, and the keypoint error is the
number that stands in for "did the retargeting work".

It cannot stand in for that, because it is computed in the hand's own frame and manipulation
is decided in the object's. The gap is invisible on an anthropomorphic 24-DoF hand, which is
why nobody has had to look at it. It is enormous on the hands people actually deploy.

We have the extreme case already characterized:

1. **The Vega f5d6 hand cannot oppose at all.** A full-range search over the three thumb
   joints (`th_j0 ∈ [-0.02, 1.60]`, `th_j1 ∈ [-0.35, 0.18]`, `th_j2 ∈ [-0.43, 0.27]`) gives a
   minimum thumb-tip-to-finger-mean gap of **3.1 cm**, with the thumb pinned at its
   opposition limit. Spheres of 3, 4, 5 and 6 cm were all pinched at that limit with fingers
   closed; all four fell (~106 cm drop). Palm-up cradle and enclosed-cage holds fail the same
   way. Ruled out as causes: collision-mesh coarseness (11 finger collision geoms, all
   enabled, contacts generated), finger stiffness (identical drop at HAND_KP 8/40/120/300 —
   the object never contacted). It is a kinematic property of the hand. **[measured]**
2. **Two hands restore exactly what one hand lost.** Two f5d6 hands on opposite ±y faces of a
   0.05 kg box lift it **+21.3 cm** in open-loop physics; at 0.12 kg the grip slips (~6 cm).
   A cooperative tangential sweep yaws the same box **+37°** under a trained policy. Neither
   manipulation is available to one f5d6 hand at any pose. **[measured]**
3. **The tracker is not the bottleneck.** PPO on MJX learns the tracking task (push:
   object error 5.4 → 2.9 cm/step, stable, 20 M steps). The bimanual lift plateaus at +4 cm
   across three runs with materially different residual bounds and replay semantics — a
   policy local optimum on a grip with no margin, not a learning failure. **[measured]**

So the embodiment is characterized, the executor works, and the thing standing between a
human demonstration and this robot is a **representation** problem. That is the paper.

### The four claims

**C1 — Keypoint retargeting error does not predict manipulation success on
opposition-deficient hands.** Across a human grasp corpus retargeted to several robot hands,
the correlation between per-hand keypoint error and physical task success collapses as the
hand's opposition capability falls. On f5d6 we expect it to be indistinguishable from zero.
**[hypothesis]**

**C2 — One number does predict it.** The Ferrari–Canny ε-metric — the radius of the largest
origin-centred ball inside the convex hull of the grasp wrench space — computed from the
retargeted contact set, predicts success across hands, objects and tasks with a single curve.
Define the **opposition deficit** δ = w_required / ε, and the claim is that success is a step
in δ at δ = 1, wherever the hand and object came from. **[hypothesis]**

**C3 — The two objectives are anti-correlated, not merely different.** On a hand with δ > 1,
*reducing* keypoint error *reduces* ε. The human's contact set assumes an opposition the
robot does not have, so the pose that best matches the human is close to the pose where the
robot's own wrench cone is smallest. If true, this is the result: the field's retargeting
objective is actively wrong for the hands the field deploys, and the metric it reports hides
that. **[hypothesis]**

**C4 — Re-allocating contacts across the two hands covers the deficit.** Retargeting that
reproduces the demonstrated *object* trajectory while choosing contacts to maximize bimanual
ε recovers tasks that pose retargeting gets 0/N on — and it does so while **losing** on the
keypoint metric. The amount of second hand it has to spend is predicted by δ. **[hypothesis]**

C3 and C4 are the ones worth the year. C1 is the setup, C2 is the instrument.

## Why this lab and not another

The specific asset is that the hardest case is already built and already measured. Nobody
sets out to study an opposition-deficient hand; you only get there by buying one and spending
four months proving it cannot grasp. `dextrack_vega` has: the Vega + f5d6 MuJoCo scene, the
MJX-compatible primitive-collision variant, a tracking env with a bounded-residual action
space, a working MJX/brax PPO recipe at ~104 k env-steps/s on an A10G (100 M steps ≈ 16 min ≈
$0.30), and a **scripted bimanual expert that works in physics** — which is what the pairing
protocol in `rule-null-results` demands before any 0/N of ours is allowed to mean anything.

The comparison hands are on disk too: Allegro, LEAP and Shadow URDFs under
`~/projects/DexTrack/assets/`. That turns "our hand is bad" into a measured axis with four
points on it, which is the difference between an excuse and a result.

## The advisor fit, stated plainly

Zuo's group produces the *input* to this problem and stops at its boundary. StableHand
recovers world-space bimanual hand motion explicitly "for supervising robot policy learning —
wrist trajectories track the end-effector and finger articulations specify the grasp pose".
FlowHOI generates hand poses, object poses and contact states, then "illustrat[es] the
feasibility of retargeting generated HOI representations to real-robot execution pipelines"
on four tasks. In both, the retargeting step is the unexamined last mile, and in both the
representation already carries the thing we need — **contact state**. This project is the
missing half, it is bimanual, it is hand-object interaction, and it needs nothing from that
group except the generator we would otherwise have to build.

The natural v2, which we should not scope into v1: replace mocap-grade hand input with
StableHand-style estimates from egocentric video, and let its four per-channel quality
signals decide how much of the human contact specification to trust. Deficit-aware
retargeting and quality-aware retargeting are the same optimization with two different
uncertainty sources.

## What would kill this

- **C1 fails**: keypoint error does predict success on f5d6. Then the premise is wrong and
  the pivot is to C2 alone as a cheap screening metric — a workshop paper, not a paper.
- **δ does not transfer across hands.** If each hand needs its own curve, there is no
  instrument, only a per-hand tuning story. Check this at M2 before building anything.
- **Someone has published it.** Nearest neighbours to position against: DexMachina (bimanual
  dexterous RL from ARCTIC demos, but on hands that *can* oppose), `dex-retargeting`/DexPilot
  /AnyTeleop (pose objective, unimanual), functional/contact-based grasp transfer,
  DexMimicGen and BiDexHD (bimanual data generation, not retargeting objectives), and the
  underactuated-hand grasp-analysis literature (Dollar/Odhner) which owns the ε-metric
  machinery. **Check after the question is fixed, not before** — that is the standing rule.
- **The hardware gate never closes.** See below; this is the real schedule risk.

## Rules this project inherits

From `~/projects/research/RULES.md` and `rule-null-results`, the ones that bite here:

- **A 0/N is a defect until proven a boundary.** Every "pose retargeting fails" number must
  be paired with the scripted bimanual expert succeeding under the identical protocol, and
  with that same expert failing across the claimed boundary. We already have the expert.
- **Trivial baselines every time**: do-nothing, random, and the scripted expert.
- **Three seeds and the spread**; held out by object and by grasp sequence, never by frame.
- **Success defined before the run, computed twice.** Degenerate solution to watch: "ε > 0"
  is satisfied by an object resting on the table. Success is the object trajectory, not the
  contact.
- **Sim is not evidence about a robot.** Either the hardware gate closes or every number in
  the paper is labelled simulation and the transfer gap is stated as an open quantity.

## The hardware gate

RULES.md #1 is non-negotiable and this project does not currently satisfy it. There is no
physical Vega here, and `so101-bench` is a leader-follower teleop pair, not two followers.

The cheap close: **a second SO-101 follower**. A parallel jaw is the extreme point of the
opposition axis — one opposition direction, fixed width, δ > 1 for almost everything — so a
bimanual SO-101 is not a downgrade from Vega, it is the *other* end of the same axis, and the
eval harness, trial runner and paired-trial protocol already exist and are already validated
at n=40. That is the recommended path and it should be decided in week 1, because a 14-week
schedule with hardware at the end has no slack for procurement.

## Target

RSS 2027 (deadline ≈ January 2027) with RA-L as the rolling fallback. ICRA 2027 is four days
away and is not a target.
