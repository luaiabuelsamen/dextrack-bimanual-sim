# Experimental design

Written 2026-09-11, before any number exists. Success criteria are here so they are fixed
before collection (RULES.md #5), and so a later edit to them is visible in git.

---

## 1. The load-bearing decision: most of this paper does not train anything

C1, C2 and C3 are claims about a **retargeting objective** and a **quasi-static contact
property**. None of them requires a policy. The measurement is:

> put the hand at a pose → `mj_forward` → read contact points and normals → compute ε → and
> separately, run a short physics rollout to see whether the hold survives.

That is kinematics plus a 3-second rollout. No learning, no GPU, no reward function, and no
reward hacking to guard against. The previous project spent most of four months on the RL
layer and the finding that survived was kinematic (the 3.1 cm thumb floor). This one puts the
learning last on purpose.

Learning enters once, at M4, and only as the *executor* — and there it is held fixed across
conditions, which is the whole point (§5).

**So: play with the task first.** Weeks 1–5 produce the headline figure with zero training.

## 2. Simulator and the two-mode protocol

MuJoCo, both modes that `dextrack_vega/assets.py` already builds. This is not a free choice —
the contact mode changes the instrument, so it has to be a stated protocol rather than a
default:

| | mesh mode (`collision="mesh"`) | MJX primitive mode |
|---|---|---|
| geoms | 48, 26 meshes, per-finger collision | 15, 0 meshes, fingertips = r=8 mm spheres |
| solver | Newton, elliptic cone | CG 10 iters, pyramidal cone |
| runs on | CPU, ~130 sps single env, ~366–400 sps at 16 process workers (8-core Orin) | A10G on Modal, ~104 k steps/s |
| role | **referee** | **screen** |

The primitive scene deletes the finger collision meshes. For a paper whose entire subject is
contact geometry, that scene cannot be the instrument. So:

- **Every ε, every grasp search, every hold test is scored in mesh mode.** It is cheap enough
  (§3) that there is no reason not to.
- MJX is used only to screen large configuration sweeps and to train the M4 executor.
- **M1 gate:** compute ε in both modes on the same 200 grasps and report the disagreement as a
  distribution, not a sentence. If the two modes disagree in *sign* about δ>1 on more than 5%
  of grasps, MJX is dropped from the instrument entirely and used for training only.

If M4 needs contact-faithful scale, the one infra bet worth making is **`mujoco_warp`** — GPU
sim that keeps the mesh scene. That was already scoped as the eventual GPU path before MJX was
chosen for expedience. Do not make that bet before M3 says the paper is real.

## 3. Cost, so the plan is checkable

- **Static ε at one pose:** `mj_forward` on the 48-geom scene, ~1–3 ms. A 100 k-pose grasp
  search is ~5 minutes on one core.
- **Dynamic hold test:** 3 s at 50 Hz control = 150 steps. At 366 sps across 16 workers that
  is ~2.4 tests/s, so 10 000 hold tests ≈ 70 minutes on the Jetson.
- **M2's full axis** (4 hands × ~60 objects × grasp search + hold tests): hours, not days, and
  no cloud spend.

The expensive part of M2 is not compute, it is porting three more hands into the scene.

## 4. Tasks

Selection rule: the set must **span δ**, must include a case where the effect is **absent**
(or C3 has no negative control), and every task must live inside the Vega arm's measured
reachable set — in-plane, ≤30° reorientation, table-supported, no reliance on the unactuated
torso lift.

| | task | why it is in the set | δ regime | status |
|---|---|---|---|---|
| **T1** | bimanual squeeze-lift, mass swept 0.02–0.30 kg | the calibration task; mass is a clean continuous knob on w_required, and the 0.05/0.12 kg bracket is already measured | sweeps through δ=1 | reference + BC policy exist |
| **T2** | planar reorient by tangential push | **negative control.** Nonprehensile, object stays supported, no force closure needed → δ<1 everywhere. Pose retargeting should work here, and C3 predicts the two objectives stay parallel | δ<1 | reference + BC policy exist (+37° verified) |
| **T3** | articulated open (ARCTIC laptop / notebook / box lid) | one hand stabilizes the base while the other actuates — the roles are **asymmetric and reassignable**, which is the case where re-allocation has something to choose. ARCTIC-native, so the human demo is real | mixed | not built |
| **T4** | handover / regrasp | extreme re-allocation: contact migrates between hands by construction | crosses δ=1 twice | scripted 2-phase demo exists |

T1 and T2 carry C1–C3 and need no human data — which is what lets weeks 1–5 run while the
ARCTIC licence is still pending. T3 and T4 carry C4.

**Object set for M2's axis** (the δ grid, not the tasks): boxes, cylinders and spheres
parameterized by graspable width 2–12 cm × mass 0.02–0.30 kg × friction {0.5, 1.0, 2.0}.
Held out **by object**, never by frame or by pose.

## 5. Success criteria, fixed now

Defined before collection, and computed by two independent routes with disagreement flagged.

**T1 lift — success** = object z rises ≥ 10 cm above its rest height **and** is still within
5 cm of the hand midline at t = 3 s after the lift completes. Two routes: (a) object qpos
from the physics rollout, (b) net displacement printed by `render_gif.py`, which is the check
that already caught a policy whose metrics looked good while the box never moved.

**T2 reorient — success** = object yaw changes ≥ 30° with the object never leaving the table
(z within 1 cm of rest throughout). Two routes: quaternion-derived yaw, and a top-down
frame-difference angle from the rendered rollout.

**T3 articulated — success** = the object's articulation DoF reaches ≥ 80% of the
demonstrated range, base displacement ≤ 2 cm.

**T4 handover — success** = the object ends supported by the receiving hand alone (donor
contacts zero) and never drops below 2 cm of rest.

**Degenerate solutions being watched, named in advance:**
- "ε > 0" is satisfied by an object resting on a table. ε is never a success criterion; it is
  a predictor. Success is always the object trajectory.
- transient height from flinging — already observed from PPO on this exact task. Hence the
  "still held at t = 3 s" clause in T1.
- zero rotation error on a reference with no rotation — already observed, misled a whole run.
  Rotation criteria are absolute, never error-relative.

**Baselines, every task, every time:** do-nothing (hold start pose), random action walk, and
the scripted bimanual expert. Three seeds minimum on anything learned, mean **and** spread; if
the seed spread exceeds the gap to the baseline, there is no result and the text says so.

## 6. Where training does happen (M4), and how it is held fixed

The executor is the existing DexTrack-style tracker: bounded cumulative-residual position
targets around a kinematic reference, BC warm-start from the zero-residual rollout when PPO
stalls (it stalled on exactly this task before), brax PPO on MJX with the settings that
worked — `entropy_cost=1e-3` (1e-2 collapsed), `lr=3e-4`, `discounting=0.98`,
`num_envs=2048`, `unroll_length=16`, and `done` set to true termination only, never the
episode-length limit (that bug collapsed the policy below random init).

**The comparison is not "our policy vs their policy".** It is:

> **one tracker, two references** — pose-retargeted vs ε-re-allocated — trained with identical
> hyperparameters, seeds and step budget, evaluated on identical held-out objects.

Holding the executor fixed is what makes the result a claim about the *representation*
instead of a claim about RL tuning. Any tracker improvement must be applied to both arms of
the comparison or not at all.

## 7. The week-1 probe: C3 in miniature, on assets that already exist

This is the first thing to run, and it can run tomorrow with no human data, no licence, no
GPU and no training. It exists to kill the project cheaply if it deserves killing.

On the T1 box with two f5d6 hands, compare two contact configurations:

- **A — the anthropomorphic configuration:** what pose retargeting of a human grasp on this
  box produces. For f5d6 that is the thumb-versus-fingers antipodal pinch, which lands on the
  measured 3.1 cm opposition floor.
- **B — the ε-maximizing configuration:** search the bimanual contact space (hand placement on
  the ±y faces, approach angle, finger closure) for maximum ε, with no requirement that it
  resemble a human grasp.

Then measure, for both: ε, keypoint distance to the human grasp, and physical lift outcome.

**What each result means:**
- B lifts, A drops, and A has the *lower* keypoint distance → **C3 in miniature.** The
  objective the field optimizes picks the configuration that fails. Proceed to M2 with
  confidence and a figure already in hand.
- Both lift → the box is below δ=1 and the task is too easy; raise mass until the bracket is
  found. Not a kill.
- Neither lifts → the ε search is broken or the instrument is wrong. Fix before M2. This is
  also the pairing-protocol check, since the scripted expert is known to lift this box.

Budget: one day. If it takes three, something in the instrument is wrong and that is worth
knowing on day three rather than week six.
