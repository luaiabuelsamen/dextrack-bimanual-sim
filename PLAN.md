# Plan

Experimental design — simulator protocol, task set, success criteria, training plan — is in
`EXPERIMENTS.md`. Read it before M1.

Ordered, with a kill criterion on every milestone. Nothing here is allowed to run for more
than its stated budget without producing the number it promised.

---

## M0 — Two gates, opened today (day 1)

Both are latency, not work, so they start before anything else.

1. **Data.** GRAB and ARCTIC both need registration; turnaround is 1–2 weeks and has blocked
   this workstream once already (`RETARGETING.md`: "that's a setup gate, not code"). Register
   for both plus MANO today. ARCTIC is the better fit — bimanual, articulated objects,
   ground-truth MANO and object pose, and it is the benchmark the advisor's own StableHand is
   evaluated on. GRAB is the fallback and is what DexTrack used.
2. **Hardware.** Decide the second SO-101 follower. See GOAL.md's hardware gate. If the
   answer is no, the paper is simulation-only and the abstract has to say so in the first
   sentence.

**Kill:** none. If both gates stay shut past week 3, the project is a sim-only analysis paper
and should be re-scoped to that explicitly rather than drifting into it.

## M0.5 — Novelty check (day 2, *after* the question above is fixed)

Position against DexMachina, `dex-retargeting`/DexPilot/AnyTeleop, DexMimicGen, BiDexHD,
functional-grasp-transfer, and the underactuated-hand grasp-analysis literature. One page in
`notes/RELATED.md` saying, for each, what it optimizes and why it does not cover δ > 1.

**Kill:** if ε-aware bimanual contact re-allocation is published, fall back to the
quality-aware variant (GOAL.md, "the natural v2") and re-run this check on that.

## M1 — The instrument, validated against a known answer (week 1–2)

Implement ε (Ferrari–Canny) over MuJoCo contact states: contact points, normals, friction
cones linearized at 8 edges, convex hull of the wrench space, largest inscribed ball.

The validation is pre-registered because we already know the answer: on the bimanual
squeeze-lift, ε must exceed the gravity wrench for the **0.05 kg** box (which lifts +21.3 cm)
and fall below it for the **0.12 kg** box (which slips at ~6 cm). Compute the crossing mass
and compare to the measured 0.05/0.12 bracket.

**Kill:** if the predicted crossing mass falls outside [0.05, 0.12] kg, the instrument is
wrong and C2 cannot be built on it. Fix or abandon before M2.

**Second check, same week:** ε must be below threshold for every one of the four failed
marginal grasps (3/4/5/6 cm spheres, unimanual f5d6) and above it for the scripted bimanual
expert on the same objects. Same protocol, both directions — this is the pairing protocol,
run once at the start so every later 0/N inherits it.

## M1.5 — The week-1 probe: C3 in miniature (1 day, runs before the data gate closes)

Two bimanual contact configurations on the T1 box — the anthropomorphic one that pose
retargeting produces (the 3.1 cm opposition floor) versus the ε-maximizing one — scored on ε,
keypoint distance, and physical lift. No human data, no licence, no GPU, no training.

Full protocol and the meaning of each outcome in `EXPERIMENTS.md` §7.

**Kill:** neither configuration lifts a box the scripted expert is known to lift → the ε
search or the instrument is broken. Fix before M2.

## M2 — The opposition axis (week 2–3)

Four hands — f5d6, LEAP, Allegro, Shadow (URDFs all on disk under `DexTrack/assets/`) — against
a shared object set spanning size, aspect ratio and mass. For each (hand, object): search grasp
poses, record max achievable ε, and measure physical hold in sim. Produces δ per cell.

Budget: scored in mesh mode on CPU per `EXPERIMENTS.md` §2 — a 100 k-pose grasp search is
~5 minutes on one core and 10 k hold tests ~70 minutes at 16 process workers. No cloud spend.
The real cost is porting: three more hands into the mesh scene, and `assets.py`'s URDF-repair
path is the template. MJX is used here only to screen wide sweeps before mesh scores them.

**Kill:** if δ does not collapse the four hands onto one success curve, there is no
instrument. Say so, and the project becomes a per-hand characterization — much smaller.

## M3 — The blindness result (week 3–5)

Retarget a held-out set of human grasps (ARCTIC/GRAB) to each hand with standard keypoint
retargeting. For every retargeted grasp record: keypoint error, ε, δ, and physical success.

- **C1 figure**: success vs keypoint error, one panel per hand. The correlation should decay
  along the opposition axis.
- **C3 figure**: within a single grasp, sweep the retargeting objective from pure keypoint to
  pure ε and plot both. If the two curves cross with opposite slopes on f5d6 and stay parallel
  on Shadow, C3 is earned and that is the paper's headline figure.

**Kill:** C1 fails (keypoint error predicts success on f5d6) → the premise is dead, and the
honest move is to publish M2 as the characterization it is.

## M4 — Re-allocation (week 5–9)

The method. Retarget to the object's demonstrated trajectory, not the hand pose: choose the
contact allocation across both hands that maximizes bimanual ε subject to arm reachability
(Vega's reachable-orientation set is already characterized and is tight — 90° reorientations
are unreachable, ~30° are fine). The DexTrack tracker executes the result; the reference
format (`ReferenceTrajectory`) and the bounded-residual action space already exist, and the
existing BC-from-feasible-reference recipe is the warm start when PPO stalls.

Baselines, all three, every task: do-nothing, random, and the scripted bimanual expert.
Three seeds, mean and spread, held out by object.

**Kill:** re-allocation does not beat pose retargeting on task success at matched compute.

## M5 — Hardware (week 9–14)

Whichever gate closed. Paired trials on the existing protocol, n stated as arms not rollouts,
failure categories counted, and a quantified sim→real transfer gap as one number with its
protocol. If the gate never closed, this milestone is replaced by a labelled simulation
section and an explicit statement that the transfer gap is unmeasured.

## M6 — Write (week 12–16)

Paper skeleton written at M3, before M4 produces data — the house pattern (`embodied-2026`:
"analysis for the sweep, written ahead of the data"). Every number traceable to a NOTES.md
entry; every retraction left visible.
