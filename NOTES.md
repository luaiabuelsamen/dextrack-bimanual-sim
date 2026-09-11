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
