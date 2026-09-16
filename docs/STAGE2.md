# Stage 2: the retarget

**Start here if you are picking this up.** Stage 2 turns a GRAB clip into a
robot joint trajectory. It is the blocker for everything downstream: stages 3–6
were all reporting numbers measured on poses that intersect the object, and
those numbers are withdrawn.

Status as of 2026-09-15.

| subproblem | state | evidence |
|---|---|---|
| **A. Arm-side placement** — forearm/wrist/palm inside the object | **fixed** | `W_PEN_ARM=60` → `binoculars_see_1` 175 m → **106 mm**; arm-side penetration 0.00 mm median over 40 sequences |
| **B. Finger reach** — fingertips now too far to close | **open** | after A, fingertips sit 50–62 mm from the surface; `establish_grip` closes ≤1.2 rad ≈ 30–40 mm of tip travel |
| **C. Vessel wrist target** — hand placed *inside* cups and mugs | **open, untouched** | 33–39 mm arm-side with **0.00 mm** finger contact; not weight-limited |

RL/DexTrack is blocked behind all three: every PPO checkpoint is a correction to
a specific initial condition, so changing the grasp invalidates it. Retraining
is ~25 min per reference.

---

## The defect

The retarget matched the human's fingertip **positions** with no non-penetration
constraint. For the Shadow hand on these objects, that optimum is inside the
object continuously — not at the grasp moment, at *every frame*.

Scanned across 20 s1 sequences, object at its reference pose, no closure:

```
23 of 2866 frames are under 1 mm of penetration
13 of 20 sequences never reach a collision-free frame at all
```

| reference | min | median | clean frames | deepest body |
|---|---:|---:|---:|---|
| `binoculars_see_1` | 17.15 mm | 25.12 mm | 0/155 | **forearm** |
| `flashlight_on_2` | 12.82 mm | 13.66 mm | 0/144 | **palm** |
| `cup_drink_2` | 11.73 mm | 12.28 mm | 0/117 | ffproximal |
| `mug_drink_1` | 6.81 mm | 10.82 mm | 0/116 | ffproximal |
| `camera_takepicture_1` | 0.00 mm | 14.09 mm | 13/143 | forearm |
| `gamecontroller_play_1` | 0.00 mm | 5.46 mm | 1/192 | ffdistal |

The split is structural: sequences that never come clean are dominated by a
large **non-finger** body; those that do are dominated by distal links. A Shadow
forearm is not shaped like a human's, so placing the wrist where the human's
wrist was puts the arm inside the object.

## What this invalidated

The penetration manufactured contacts, which produced grip, which flattered
every downstream metric. Measured on the four published rollouts: 8–14 mm of
interpenetration on **541 of 541 frames**, at 796–3850× the object's weight, up
to 13 kN on a 1.96 N object.

Three consequences, each measured:

- **Position error was meaningless.** The cage constrains translation, so
  position error stays in the tens of millimetres however badly the grasp fails.
  Rotation inside the cage is nearly unconstrained — hence 44.8° / 45.9° /
  138.4°. Orientation was the honest signal all along.
- **The grasp search was *rewarded* by penetration.** Depth buys contacts,
  contacts buy force, force buys grip, grip lowers tracking error. On
  `mug_drink_2` the search took the retarget from 11.08 mm / 4411 N to
  17.47 mm / 8815 N.
- **Every static grasp metric is blind to it, ε included.** Ferrari–Canny on the
  four reset contact sets gives ε 0.155–0.344 and δ 0.0005–0.0045 — strongly
  force-closed on paper — and ranks them *backwards*, the camera highest-ε and
  worst-outcome. ε takes the contact set as given, and the contact set is an
  artifact.

## What was ruled out, by measurement

Everything downstream of the retarget. None of these is a matter of tuning.

| remedy | result |
|---|---|
| Reweight `w_pen` | plateaus: 11.08 → 8.29 mm at w=4, then flat at 12 and 30 |
| Search harder for a better pose | **0 of 283** candidates across 3 references both touch the object and stay out of it; every candidate under 2 mm has <3 contacts |
| Open the fingers before closing | clears nothing at any amount: mug 10.13 → 10.64 / 9.91 / 11.35 / 11.36 mm at 0.3 / 0.8 / 1.5 / 3.0 rad; palm invariant at ~6 mm |
| Retract the wrist along the contact normal | **diverges** — binoculars runs to a 186 mm offset with penetration *rising* 17.15 → 62.44 mm, because a wrapped hand cannot be translated out (normal alignment 0.325) |
| Let it settle | gravity off 0.5 s takes 12.7 → 11.8 mm while ejecting the object 28 mm and losing every contact |
| Advance the wrist to contact before closing | **0.0 mm of advance** on 4 references — the arm is already touching while the tips are 43 mm out, so translation drives the arm deeper first |
| Close the fingers to a force target | close stops after **one 0.015 rad step**: penetrating contacts already read 362 N and 2487 N against an 8 N target, so the grip is declared formed before the hand closes |

And the result that explains why: **making the pose feasible makes the tracking
worse**, because the contacts and the penetration are the same thing.

| reference | w_pen | penetration | force | tracking |
|---|---:|---:|---:|---:|
| `mug_drink_1` | 1.0 | 10.13 mm | 2294 N | 33.9 mm |
| `mug_drink_1` | 25.0 | **1.35 mm** | 1202 N | **148.7 mm** |
| `binoculars_see_1` | 1.0 | 0.02 mm | 44 N | 176578 mm |

---

## A. Arm-side placement — fixed

Weight non-penetration on bodies not strictly below a knuckle (forearm, wrist,
palm, knuckles) at 60× the finger weight.

| reference | w_arm | arm penetration | force | residual | tracking |
|---|---:|---:|---:|---:|---:|
| `binoculars_see_1` | 1 | 5.20 mm | 117 N | 29.5× | 175528 mm |
| `binoculars_see_1` | **60** | 2.47 mm | 2473 N | 15.4× | **106.2 mm** |
| `flashlight_on_2` | 1 | 12.96 mm | 782 N | 1.7× | 8.5 mm |
| `flashlight_on_2` | 60 | 1.39 mm | 386 N | 4.3× | 107.5 mm |

**`flashlight_on_2` getting worse is not a regression.** It tracked at 8.5 mm
with its palm 12.96 mm inside the object — the burial *was* the grip.

Across 40 sequences with the constraint active:

```
median arm-side penetration   0.00 mm     <- fixed
median finger penetration     6.26 mm     <- subproblem B
median arm-side > 3 mm        11/40       <- subproblem C
max    arm-side > 3 mm        30/40, but only 6% of frames at the median
```

## B. Finger reach — open

The constraint puts the hand outside the object and thereby out of reach.
Fingertip-to-surface distance after A: **50.1 mm** (mug), **58.1 mm**
(flashlight), **62.3 mm** (binoculars). `establish_grip` closes at most 1.2 rad,
moving a tip 30–40 mm. So the hand is now feasible and cannot reach.

Equilibrium residual remains 4.3–15.4× at 361–2488 N, so no configuration is an
equilibrium yet: the arm is out, the finger contact is still an overlap.

Three attempts at an approach search were made and all three reverted, all on
the same bug class — keeping the hand consistent across `qpos`, the mocap
target, and the weld. **Anything repositioning the wrist must update `qpos`,
then the mocap target, then `mj_forward`, and must not call `palm_pose()` to
read the current palm** (it zeroes `qpos` internally). Two of the three died on
exactly that.

## B and C are probably one defect: the wrist *pose*

Tested 2026-09-15 evening. `advance_to_contact` exists in `human/retarget.py`,
is **never called**, and advances **0.0 mm** on all four references tested.

Its gate is `pen > 0.5 mm`, but the starting pose already carries 0.48–8.17 mm
of residual arm penetration — `W_PEN_ARM` is a soft penalty and never reaches
zero — so the exit condition trips before the first step. That is a one-line
bug, and fixing it does not rescue the idea. With a *relative* gate (terminate
when penetration exceeds its starting value plus 1 mm), three of four still
advance 0.0 mm; only `mug_drink_2` moves, 8 mm, closing its tip gap 17.2 → 12.1 mm.

The reason is structural. On `mug_drink_1` the arm is penetrating at 0.48 mm
**while the fingertips are 42.9 mm away**. The arm is already the binding
contact. Translating the hand toward the object drives the arm deeper before the
fingers get anywhere, so the allowance is zero. No translation brings the
fingers in.

That makes B and C the same defect seen from two sides. The wrist *pose* —
position **and orientation** — presents the arm to the object where the fingers
ought to be. For vessels it shows as 33 mm of arm inside with zero finger
contact; elsewhere as the arm grazing at 0.5–2.5 mm with the tips 43 mm out.
`W_PEN_ARM` constrains where the arm may *be*; nothing constrains where the
fingers *point*.

So the open question is probably one, not two: **how should the wrist be posed
relative to the object, given it cannot be the human's wrist pose?** Orientation
is the part nothing has touched.

## C. Vessel wrist target — open, untouched

Eleven of 40 sequences retain a median arm-side problem, and they cluster on
objects with a concavity:

| | arm-side | finger |
|---|---:|---:|
| `mug_drink_3` | 39.28 mm | **0.00 mm** |
| `mug_toast_1` | 34.27 mm | **0.00 mm** |
| `mug_drink_2` | 33.65 mm | **0.00 mm** |
| `mug_drink_4` | 33.30 mm | **0.00 mm** |
| `cup_pour_1` | 25.80 mm | 10.26 mm |
| `bowl_drink_1` | 16.38 mm | 17.42 mm |
| `cup_lift` | 12.26 mm | 13.58 mm |

33–39 mm of arm inside the object with **zero** finger contact is a hand through
the vessel, not around it. It is **not weight-limited** — sweeping `W_PEN_ARM`
on `mug_drink_2`:

| w_arm | 60 | 150 | 400 | 1000 |
|---|---:|---:|---:|---:|
| arm median | 33.65 mm | 32.92 mm | 32.85 mm | 33.30 mm |

A 16.7× increase moves it 0.35 mm. The wrist *target* is wrong for this
geometry; no penalty relocates a wrist that has been told where to be.

These five zero-finger sequences should be held out as a separate category.
"The robot cannot be placed at all" and "the robot is placed and tracks badly"
are different failures, and averaging them makes the distribution unreadable.

---

## Instrumentation

`make render-tracking` captures and renders the four reference rollouts. Each
GIF carries penetration, grip force as a multiple of object weight, and contact
count on its face, and flags `CONTACT SET IS AN ARTIFACT / NOT A GRASP` above
2 mm or 40× weight. Each has a JSON manifest with per-frame errors, penetration,
force, equilibrium residual, source and model hashes, and a replay check against
the evaluator.

**Equilibrium residual** is the scalar worth watching: net unbalanced wrench on
the object at reset, gravity included, in multiples of object weight. **0× is
equilibrium, 1× is freefall**, and these measure 142–337×. Unlike penetration
depth it rejects both failure ends — a hand that misses the object scores
exactly 1×, where depth would flatter it with 0 mm.

Scan scripts used for the tables above are not committed; they are short and
read-only, calling `ReferenceTracker` and `mj_forward` per frame at roughly
0.07 s per configuration.

---

## Working this with more than one agent

Two sessions worked this on 2026-09-15 and the coordination cost was real. What
went wrong, so it does not go wrong again:

- **They shared one working tree**, not separate checkouts. Same directory, same
  branch. A `git add -A` from either picks up the other's in-flight edits — and
  did: commit `50bff98` describes a restructure but also carries the
  `W_PEN_ARM` block, which belonged to the other session's uncommitted work.
- **Stage explicit paths. Never `git add -A` or `git add .`.** This was agreed
  and then violated by the agent who proposed it.
- **Split by file, not by task.** What worked: one session owned `src/` and
  `NOTES.md`, the other owned `README.md`, `docs/`, and
  `experiments/tracking/render_tracking.py`. Announce before crossing.
- **Reading the other's uncommitted diff is high value.** The per-body
  breakdown that found the buried palm — and later the buried forearm — came
  from testing code that had not been committed yet, not from waiting for a
  report.

The division that actually paid off was **one agent implementing, one agent
measuring**. Most of the findings above came from the measuring side testing the
implementing side's work and reporting what it did rather than what it was
supposed to do. Three hypotheses from the measuring side were wrong and were
retracted the same way: one-sidedness of the contact normals, force closure as a
discriminator, and wrist retraction along contact normals. Each was cheap to
test and each wrong answer narrowed the problem.
