# The brief (2026-09-16, evening)

Written for whoever picks this up, in the form the project's owner asked for:
the setup, what is established, what is ruled out, one bounded goal, the
guardrails that stop it turning into a table of numbers, and what is and is
not solvable on the hardware we have. Supersedes `GOAL.md` (the retracted
opposition-deficit programme) and `NEXT.md` (2026-09-11).

## Setup

GRAB human demonstrations (MANO hand and object pose, subsampled to 15 Hz),
a Shadow right hand in MuJoCo, convex-decomposed objects (23 to 42 collision
geoms each), mocap-welded wrist, position-servoed fingers. A DexTrack-shaped
pipeline: reference, retarget and grasp search, per-clip PPO, homotopy and
distillation, two hands, perception. Stages 3 to 7 all run. One Jetson Orin,
8 cores, 15 GB, GPU slower than the CPU for this scene. No x86 GPU.

## What is established, measured, on this board

1. **A static hold test selects burials.** Of 34 stage-2 holds, 6 were the
   hand inside the object and most of the rest partly inside. Every policy
   that tracked was tracking a burial; every clean grasp seed dropped, six
   of six. Open-loop, burial carries the object without a controller (bowl
   9.9 mm with nothing in the loop, 8.2 mm with PPO).
2. **The property is dynamic.** Every pose that carries starts buried and
   relaxes into a contact set under motion. A static gate rejects all of
   them. `docs/STAGE2.md`, `figures/burial_relaxation.png`.
3. **Scoring the end of an open-loop carry finds grasps.** Held, under 3 mm,
   at least two links, opposed, under 40x the object's weight, over the whole
   remaining reference, hill-climbed on the wrist offset from the stage-2
   seed. Over four random restarts, **20 of 40 references** have a pose that
   passes and survives at least half of eight 0.5 mm / 0.6 degree wrist
   perturbations; 8 to 13 per single restart; the union was still growing at
   four. Every one rebuilt from its stored offset and rendered.
   `results/stage2_carry_fcap_*`, `figures/physics_carry_*.gif`.
4. **PPO on those seeds keeps the grasp and halves the tracking error** on
   every start the feedforward carries from, and rescues nothing where it
   does not. `results/stage3_carry*.json`.
5. **On DexTrack's own success rule we are at 5 percent strict, 20 percent
   loose of 40**, against their per-clip PPO baseline's 38.6 and 54.8 on 197
   sequences. Translation is fine (18 of 20 seeds under 6 cm). **Rotation is
   the whole gap**: 20 to 95 degrees, the object swinging in a light pinch.
   `results/dextrack_metric.json`.

## Ruled out by measurement; do not re-run

Penetration-weighted retargeting; opening, retracting or advancing fingers
or wrist; closing to a force target from a penetrating start; Ferrari-Canny
epsilon and one-sidedness as search objectives; a compliant wrist; more
training data; the policy class; a penetration penalty in the reward; every
reset-time scalar as a discriminator (four tried, none separates); a 25-frame
scoring window (overfits); an uncapped force penalty (makes burial seeds
drop); softer MuJoCo contact solref, weaker finger servos and a compliant
weld, alone or together (all worse, controls break); GPU physics on this
board (Warp 2.6k steps/s at 256 worlds, below the CPU; MJX compiles for
15 minutes per environment on this scene).

## What DexTrack did, read from their code and paper

No stage 2. Keypoint retarget with no penetration term, straight into
per-trajectory PPO at 8192 PhysX environments with soft PD (kp 20) on an
actuated free base and a capped depenetration velocity. Reward: object pose,
fingertip-to-object distance as a gate, hand-pose guidance. Success: 10 cm,
20 or 40 degrees, a hand-pose bound. **Penetration is never measured.** Their
flywheel replaces kinematic references with RL rollouts, which is our
relaxation applied to whole trajectories. Their result is a claim about
tracking at scale with a metric that cannot see burial; ours is a claim about
grasps at small scale with a metric that can. They are not the same claim,
and nobody has measured their policies on ours.

## The goal, bounded

**Establish, with published policies instrumented per frame, whether
simulated dexterous tracking success is being earned by interpenetration,
and show a stage-2 procedure that produces grasps which pass both their rule
and ours on a stated fraction of GRAB.**

Concretely, three deliverables and nothing else:

- **D1, the audit.** DexTrack's released checkpoint rolled on GRAB clips with
  penetration, contact count, links and grip force read live every frame,
  exactly as `render_tracking.py` does for ours. Report their success rate
  against their burial rate on the same clips. One x86 machine with any
  NVIDIA GPU, one environment, a few hours. This is the only item that
  cannot run here.
- **D2, the milestone on this board.** Rotation in the carry score and
  weighted in the PPO reward. Target: **at least 22 of 40 references pass
  DexTrack's loose rule and at least 16 of 40 pass ours**, from the same
  rollouts. Pass or fail, reported with the per-seed spread.
- **D3, the write-up.** The burial mechanism with the GIFs, the carry-scored
  fix, the two rules side by side on our pipeline and, if D1 lands, on
  theirs. The four buried references and the twelve non-contacting ones are
  one paragraph each saying why they are out of scope.

## Status after D2 (2026-09-17)

D2 ran: strict 2 → 6, loose 8 → 13, ours 16 of 40 (target met), loose
target 22 not met and not reachable from 18 robust seeds. See
`STAGE2.md`. The next lever is seed coverage for the 22 references without
a robust seed, not the tracker. D1 (the audit) is unchanged and still needs
an x86 machine.

## Status after D1 (2026-09-17)

D1 ran on a rented 3090. The released code does not run its own
checkpoints (initialization launches the object); with two switchable
initialization fixes, two of three released GRAB checkpoints track to
millimetres while holding the object at 1–2 mm of interpenetration with
5–10 mm excursions on 5–8 % of frames, one is lost mid-clip. Their
successes are not burials of our kind. See `STAGE2.md`.

## Status after the Isaac Gym session (2026-09-17, night)

D1 done at scale: the generalist holds 17 of 43 clips at a median 1.7 mm of
interpenetration, 36 % of frames over 2 mm. New: a GPU penetration probe as
a reward term in their trainer cuts a released policy's interpenetration
by 22 % (2.87 to 2.23 mm on a 2 mm surface grid; the sparse probe it was
trained on claimed 46 %, and the policy had partly learned the probe) and
grip by 36 % with every rollout still held (16 of 16), for 1.5 mm of
error, in 150 epochs. The lever now is the same term on the dense grid,
then on the generalist across clips, scored on both rules. See `STAGE2.md`.

## Guardrails

1. **Render before believing.** No count enters a document until every pose
   behind it has been rebuilt from its stored offset, rolled to the end of
   its reference, perturbed eight times, and looked at. The 25-frame window,
   the in-process knife-edge and the 100x camera were each caught by the
   next instrument down, never by the number.
2. **Both rules, same rollouts, always.** Every table carries DexTrack's
   strict and loose columns beside ours. A row that passes one and not the
   other is the finding, not an inconvenience.
3. **Restarts are the method.** A single search seed is a lottery ticket
   (8 to 13 of 40). Quote the union over a stated number of seeds and the
   per-seed spread. Never quote the union alone.
4. **State n and set.** 40 references, one per object, subject s1, is not
   197 sequences. Say so in every comparison.
5. **Nothing runs on a paid machine that has not run here.** Every script
   dry-runs on this board at small scale, writes its results back as it
   goes, and terminates itself. First paid run is a ten-minute smoke. A
   dollar cap is set before the first run and reported against.
6. **Two errors a day is the process working.** Each retraction is recorded
   inline in NOTES.md with the measurement that killed it. An error tally
   is not a stopping rule.
7. **Decide what not to run.** The physics grid, the reset discriminators
   and the simulator knobs are done. Any new experiment must name the
   deliverable it feeds, or it does not run.

## What can be solved, and how

| problem | solvable here? | how |
|---|---|---|
| rotation gap on carry seeds | yes, two days | rotation error in the carry score; rotation weight in the PPO reward matched to DexTrack's; resweep four seeds; rescore on both rules |
| count depends on the search seed | yes, five hours | eight restarts per reference as the standard procedure; report union and spread |
| distillation that keeps a grasp | yes, one day | stages 4 and 5 on the robust seeds from the carry start, end state instrumented |
| whether published policies are buried | **no, needs x86** | D1; the 940MX if Isaac Gym runs on compute capability 5.0, otherwise a rented 4090 for an evening |
| RL at DexTrack scale without a stage 2 | no, and optional | a batched Warp tracker built and parity-checked here first; one rented evening; only if D1 and D2 leave the question open |
| the four references that stay buried | not with this hand and simulator | out of scope; one paragraph |
| the twelve references whose seed never touches | not with a local search | out of scope; where brute-force RL would pay and we cannot |

## A good week

Monday and Tuesday: D2, rotation into score and reward, resweep, retrain,
rescore. Wednesday: restarts and distillation, the tables. Thursday: D1 on
whatever x86 machine exists, or the smoke run on a rented one with the cap
set. Friday: the write-up draft with every GIF and both rules in every
table. If D2 fails its target, the write-up says the light pinch is this
hand's ceiling on these objects and shows the rotation numbers; that is a
result, not a failure to report.

## What only a person can do

Ask a labmate for a Shadow-hand MuJoCo contact setup. Ask the DexTrack
authors whether they ever measured penetration. Find out whether a lab
workstation or a Savio allocation exists before renting anything. Decide,
after D2, whether the audit paper or the pipeline paper is the one to write.
