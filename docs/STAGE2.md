# Stage 2: the retarget

**Start here if you are picking this up.** Stage 2 turns a GRAB clip into a
robot joint trajectory. It is the blocker for everything downstream: stages 3–6
were all reporting numbers measured on poses that intersect the object, and
those numbers are withdrawn.

Status as of 2026-09-15.

| subproblem | state | evidence |
|---|---|---|
| **A. Arm-side placement** — forearm/wrist/palm inside the object | **fixed** | `W_PEN_ARM=60` → `binoculars_see_1` 175 m → **106 mm**; arm-side penetration 0.00 mm median over 40 sequences |
| **B + C. The wrist pose** — the arm reaches the object before the fingers do | **open, and now one question** | on `mug_drink_1` the arm penetrates at 0.48 mm while the tips are 42.9 mm out; for vessels the palm is 33 mm inside with **zero** finger contact |

**What stage 2 has to produce, stated precisely:** a configuration with **many
contacts, on opposing sides, at bounded penetration** — a wrap. That is what
separates the configurations that carry from the ones that do not (10 and 3
engaged bodies at one-sidedness 0.055/0.191, against 1 body at ≈1.0), and
nothing in the pipeline currently searches for it.

**The hard finding, and the reason this is not a tuning problem:** across 3019
sampled wrist poses under a 2 mm penetration gate, **not one made contact at
all**. Every configuration that touches these objects penetrates them. That is
the same result as the earlier 0-of-283 over translation, now over orientation.

**Before proposing anything, look at it.** See
[Look at the pose before trusting the number](#look-at-the-pose-before-trusting-the-number).
Three conclusions recorded in this document were wrong until someone rendered
the scene, and the corrections are kept inline rather than deleted.

RL/DexTrack is blocked behind this: every PPO checkpoint is a correction to a
specific initial condition, so changing the grasp invalidates it. Retraining is
~25 min per reference, and nothing should be retrained until a grasp survives a
rollout.

---

## The reference is clean — a sub-millimetre wrap exists

Worth establishing before anything else, because it bounds what stage 2 has to
achieve. Minimum distance from any MANO hand vertex to any object vertex, in the
GRAB reference itself:

| reference | min gap | median during hold | frames < 1 mm |
|---|---:|---:|---:|
| `cup_lift` | **0.19 mm** | 0.99 mm | 194/467 |
| `mug_drink_1` | **0.47 mm** | 1.51 mm | 18/167 |
| `binoculars_see_1` | **0.14 mm** | 0.91 mm | 113/236 |
| `flashlight_on_2` | **0.11 mm** | 0.81 mm | 112/194 |

The human hand closes to a fraction of a millimetre and stops there, for
hundreds of frames. Two consequences:

1. **The 8–17 mm of robot penetration is the retarget's doing**, not a flaw
   inherited from the data. The reference is not asking the robot to intersect
   anything.
2. **A sub-millimetre wrap of these exact objects exists**, because a human
   performs one on every frame. So the earlier result — 3019 sampled wrist poses
   under a 2 mm gate, none making contact — is a statement about the Shadow hand
   and the space being searched, not about the objects being ungraspable.

**Caveat, and it matters.** This is an unsigned point-cloud distance, the same
quantity as `GrabSequence.contact_distance`. It cannot report a negative value,
so it shows the human achieving sub-millimetre contact but does **not by itself
prove non-penetration** — a hand a few millimetres inside would still report a
small positive nearest-vertex distance. Establishing that properly needs a
signed test against the convex decomposition, which has not been run.

### Which links the human uses — and a retraction

**Retracted.** An earlier revision of this section claimed, from MANO joint
positions, that "the human contacts with fingertips and distal phalanges;
middle phalanges sit 14–20 mm out", and used that to argue a middle-phalanx
contact term would be the wrong fix. That argument does not hold and the
measurement behind it could not support it.

`human/retarget.py` already carries `W_MID = 0.6`, a middle-phalanx **contact**
target — explicitly distinguished there from a joint-position target — with
per-finger surface gaps measured against a 15 mm tolerance:

| reference | TIP gaps (mm) | MID gaps (mm) | fires |
|---|---|---|---|
| `binoculars_see_1` | 3.1 7.8 6.5 10.8 0.9 | 5.8 10.3 18.4 24.5 16.1 | 2/5 |
| `flashlight_on_2` | 10.3 15.4 12.8 23.3 27.9 | 7.8 2.9 5.4 24.0 13.1 | 4/5 |
| `cup_lift` | 6.8 21.8 11.6 18.8 5.5 | 4.8 7.7 6.3 5.3 0.5 | 5/5 |
| `mug_drink_1` | 12.8 13.1 13.6 16.4 1.1 | 7.6 8.2 9.8 26.0 27.7 | 3/5 |

On `cup_lift` every middle phalanx is closer to the object than the fingertips
are. The claim that the human holds that cup with the middles of its fingers is
measured, per finger, at the surface.

**Why my contradiction was invalid.** I measured MANO *joint centres*, which sit
roughly 8–10 mm inside the finger, and aggregated as a minimum over the five
fingers then a median over engaged frames. That is a different quantity from a
per-finger surface gap at the grasp, and it cannot refute one. The two are not
comparable and I treated them as though they were.

The general lesson, which cost more than this one entry: **read the module
before contradicting its measurements.** The rationale was in the file, with the
numbers, above the constant it justifies.

What survives from my measurement is only the weakest reading — that the human
engages more than the five fingertips — which `W_MID` already encodes.

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

### And such poses exist — orientation is the missing degree of freedom

The one constructive result. Sweeping wrist rotation (rx, ry, rz over ±40°,
7 values each) crossed with an advance along the palm axis (−20, 0, +20 mm) —
1029 poses per reference, fitting scene, frame 0:

| reference | start (arm / gap) | best found | arm<1 mm **and** gap<15 mm |
|---|---|---|---:|
| `mug_drink_1` | 0.48 mm / 42.9 mm | 0.00 mm / 23.1 mm | 0/1029 |
| `binoculars_see_1` | 2.47 mm / 44.2 mm | 0.00 mm / **−11.6 mm** | **47**/1029 |
| `flashlight_on_2` | 1.39 mm / 23.5 mm | 0.00 mm / **3.1 mm** | **21**/1029 |

Two of three reach **zero** arm penetration with the fingertips touching, at
rotations of 12–40°. A negative gap means the tip is inside the surface
envelope, i.e. in contact. These are the poses the pipeline has never found, and
they sit within 40° of where the fit was already putting the wrist.

The advance matters but is not sufficient alone: +20 mm was best in all three,
yet translation *by itself* advances 0.0 mm. It only becomes available once
rotation stops the arm being the leading contact.

`mug_drink_1` is the holdout at 0/1029, though its best still improves the gap
42.9 → 23.1 mm with the arm fully clear. A mug is the vessel geometry already
known to be the hardest class.

**Caveats.** Static check at frame 0 in the fitting scene, not a trajectory and
not physics. The 15 mm threshold was chosen because `establish_grip` moves a tip
30–40 mm, so anything under it is reachable — it is not a claim about grasp
quality. A coarse 7³ grid, so the counts are existence evidence rather than a
distribution. And it says nothing about whether the resulting grasp is stable,
only that it is geometrically available.

**What it suggests.** The fit solves fingertip positions and wrist *position*,
with `W_PEN_ARM` constraining where the arm may be. Wrist *orientation* is
inherited from the human and never searched — and it is what decides whether the
arm or the fingers reach the object first.

### Under physics: one real grip, and the score's limit

`orient_to_clear` is implemented in `human/retarget.py` (not wired into the
default fit). Applying its frame-0 delta to the whole trajectory, recomputing
the feedforward, and measuring **in the simulation scene at reset**:

| reference | | penetration | force | ×weight | contacts | residual |
|---|---|---:|---:|---:|---:|---:|
| `binoculars_see_1` | baseline | 56.77 mm | 90888 N | 46324× | 84 | 15.40× |
| | **oriented** | 9.63 mm | **14.3 N** | **7×** | 8 | **4.84×** |
| `flashlight_on_2` | baseline | 17.33 mm | 769 N | 392× | 6 | 4.33× |
| | oriented | 0.00 mm | 0.0 N | 0× | **0** | **1.00×** |
| `cup_lift` | baseline | 11.65 mm | 4642 N | 2366× | 43 | 4.85× |
| | oriented | 0.00 mm | 0.0 N | 0× | **0** | **1.00×** |

`binoculars_see_1` drops from 90 kN to **14.3 N — 7× object weight**, the first
plausible grip this pipeline has produced on a bimanual reference, at 8 contacts
and a residual of 4.84×.

The other two land at residual **1.00×, which is freefall**: zero contacts,
nothing held. Clean, and not a grasp. That is the familiar dichotomy — except
the tips are now 1.5–3.1 mm out rather than 42–62 mm, which is well inside
`establish_grip`'s 30–40 mm of travel.

**Closing afterwards does not help.** `binoculars` 14.3 → 14.8 N, and the other
two stay at zero contacts. So the fingers are near the surface but not *facing*
it: curling them does not engage. A minimum-distance score cannot distinguish a
hand poised to grasp from one merely adjacent, which is exactly what an
opposition or force-closure term is for. `grasp_epsilon` and `one_sidedness`
exist in `human/track.py` for this; combining them with the orientation search
is the obvious next step and has not been tried.

### Targeting light contact, not exact touching: 1–2× object weight

Changing the score's target from *touching* (`gap = 0`) to **1 mm of contact**
(`|gap + 1 mm|`), and allowing a 30 mm advance:

| reference | penetration | force | ×weight | contacts | residual |
|---|---:|---:|---:|---:|---:|
| `binoculars_see_1` | 4.19 mm | **4.2 N** | **2×** | 6 | 1.59× |
| `flashlight_on_2` | 6.46 mm | **1.8 N** | **1×** | 1 | 1.77× |
| `cup_lift` | 0.54 mm | **1.0 N** | **1×** | 1 | 0.94× |

All three make contact at forces of **1–2× object weight**. `binoculars_see_1`
was 46324× that morning, and 7× under the touching target. The two that were in
freefall now hold contact.

A gap target of exactly zero lands just outside contact in physics; a millimetre
of intended overlap lands just inside. That is the whole difference between
freefall and a grip.

**An opposition term does not help, and hurts.** Adding `w_opp · one_sidedness`
to the score changes nothing on two references and wrecks the third:

| `cup_lift` | penetration | force | ×weight | residual |
|---|---:|---:|---:|---:|
| `w_opp` = 0 | 0.54 mm | 1.0 N | 1× | 0.94× |
| `w_opp` = 0.02 | 12.71 mm | 490.5 N | **250×** | **34.37×** |

`one_sidedness` returns 1.0 where no contacts exist, so it drives the search
toward contact — and the search obtains contact by burying. This is the same
corruption as ε: an opposition measure computed on a manufactured contact set
rewards manufacturing one. Any such term needs a penetration gate before it can
be used as an objective.

### Rolled out, it does not track — and that is the real finding

The measurement the section above was missing. Same configurations, rolled out:

| reference | baseline mean | **oriented mean** | frames < 50 mm |
|---|---:|---:|---|
| `binoculars_see_1` | 106.2 mm | **170032 mm** | 2/155 → 1/155 |
| `flashlight_on_2` | 107.5 mm | **149409 mm** | 1/144 → 1/144 |
| `cup_lift` | **23.1 mm** | **116743 mm** | **128/128** → 1/128 |

The oriented poses have plausible contact forces at reset and cannot carry the
object at all. `cup_lift` was tracking at 23.1 mm with every frame under 50 mm —
on 2366× object weight of penetration — and orientation destroys it.

**So do not read the 1–2× weight result as progress toward tracking.** It is
progress toward physical validity, and the two are in direct opposition here.
This is the third independent confirmation of the same fact:

| measured | finding |
|---|---|
| `w_pen` sweep, end to end | 1.35 mm penetration bought at 148.7 mm of tracking |
| 283 sampled candidates | zero both touch the object and stay out of it |
| orientation search, rolled out | 1–2× weight grips fly the object 100+ metres |

**In this pipeline, tracking comes from penetration.** 1–2× object weight is
what a physically valid contact set produces, and it is nowhere near enough to
carry anything. The kilonewtons were doing the carrying.

That reframes what stage 2 has to deliver. A feasible pose is necessary and
nowhere near sufficient: the grasp must also generate real grip force from
non-penetrating contact, which means the fingers have to *wrap*, not merely
touch. Nothing measured so far produces that, and no scoring function tried —
distance, held-ness, tracking error, penetration depth, equilibrium residual,
ε, one-sidedness — distinguishes a wrap from a touch.

### What separates carrying from touching: opposition, measured on real contact

Comparing configurations that carry against ones that do not, same object, same
reference:

| reference | variant | tracking | contacts | **bodies** | **one-sided** | force |
|---|---|---:|---:|---:|---:|---:|
| `cup_lift` | baseline | **23.1 mm** | 43 | **10** | **0.055** | 4642 N |
| | oriented | 116743 mm | 1 | **1** | **1.000** | 1.0 N |
| `binoculars_see_1` | baseline | 106.2 mm | 84 | **3** | **0.191** | 90888 N |
| | oriented | 170032 mm | 6 | **1** | **0.932** | 4.2 N |

The configurations that carry engage **10 and 3 distinct hand bodies** at
one-sidedness **0.055 and 0.191** — contacts distributed on opposing sides. The
ones that fail engage **one body** at one-sidedness ≈1.0: a single fingertip
pressing from a single direction. That is wrap versus touch, and
`one_sidedness` in `human/track.py` measures it correctly.

**This corrects an earlier conclusion recorded here.** One-sidedness was
reported above as harmful because adding it to the orientation score drove
`cup_lift` from 1.0 N to 490.5 N. Both observations are true and the distinction
matters: it is a **valid discriminator** between configurations that already
have contact, and an **unusable gradient** from free space, where it returns its
1.0 default and the search reaches contact by burying. Gate it on penetration
and it is the right quantity; use it unguarded as an objective and it rewards
the defect.

So the target is not "plausible contact force" and not "opposition" alone. It is
**many contacts, on opposing sides, at bounded penetration** — which is a wrap.
Nothing in the pipeline currently searches for that: `W_PEN_ARM` bounds
penetration, the distance score reaches the surface, and neither asks how many
links engage or from which directions.

**Caveats.** Frame-0 delta applied as a constant offset across the trajectory.
Two references in this comparison, three in the rollout. The rollout is
feedforward with no PPO correction, so these are not comparable to per-clip PPO
numbers. The carrying configurations here are themselves unphysical — 4642 N and
90888 N — so this identifies what distinguishes them, not a target to copy.

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

## Frame 0 is the approach, not the grasp — and what the RL half actually rides on

`reset_at`'s own docstring: *the window begins where the hand first comes within
5 mm, which is the moment contact starts, not the moment the grasp is formed.*
The mug capture uses `start = 5` for that reason. Every `reset_at(0)`
measurement in this document was taken on the approach transient. Rendered, at
frame 0 the deepest bodies are `forearm, palm, wrist` — the arm sweeping
through the object before the hand arrives.

The state at the documented start and at mid-window, feedforward rollout:

| reference | k | penetration | bodies | one-sided | force | ×W | residual | tracking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mug_drink_1` | 0 | 69.93 mm | 3 | 0.094 | 35987 N | 18342× | 8.25× | 153 mm |
| | **5** | **6.10 mm** | 7 | 0.283 | **119 N** | **61×** | — | 89408 mm |
| | 58 | 6.61 mm | 6 | 0.527 | 204 N | 104× | 2.95× | **32.7 mm** |
| `binoculars_see_1` | 0 | 56.77 mm | 3 | 0.191 | 90889 N | 46324× | 15.40× | 187 mm |
| | **5** | **6.42 mm** | 8 | 0.727 | **37 N** | **19×** | 11.47× | 161401 mm |
| | 77 | 6.03 mm | 5 | 0.486 | 24 N | 12× | 9.60× | 43457 mm |
| `flashlight_on_2` | 0 | 17.33 mm | 3 | 0.218 | 769 N | 392× | 4.33× | 134 mm |
| | **5** | **4.70 mm** | 4 | 0.963 | **27 N** | **14×** | 8.66× | 136177 mm |
| | 72 | 15.41 mm | 10 | 0.134 | 2373 N | 1209× | 10.21× | 59.7 mm |

At frame 5 the arm is clear (deepest bodies are finger links), penetration is
4.7–6.4 mm and force is 14–53× weight. **`W_PEN_ARM` is working.** Those are
the most physically plausible grasps the pipeline has produced — and every one
flies to 89–161 m under feedforward. Rendered, the mug at frame 5 is a correct
**handle pinch**, thumb and index through the loop. Its one-sidedness of 0.571
is not a defect: a handle grip is one-sided about the mug's centre of mass and
carries a lever arm feedforward cannot hold. That is precisely what a per-clip
policy is for.

### What the PPO checkpoint tracks on

| harness | initial condition at frame 5 | PPO tracking |
|---|---|---:|
| exact capture path: `synthesize_grasp("hold")` → `rl.evaluate` | **20.58 mm / 16 bodies / 13579 N** | **23.2 mm**, 111/111 |
| pristine: raw `W_PEN_ARM` fit, nothing before `evaluate` | 6.10 mm / 7 bodies / 119 N | **83984 mm**, 1/111 |

Two conclusions, one of which corrects an earlier expectation in this document:

- **The checkpoint survived `W_PEN_ARM`** — 23.2 mm at HEAD against 29.35 in the
  morning manifest, on a *different* hold-scored offset (36.8 mm vs 21.7 mm).
  The prediction that any change to the initial condition invalidates the
  policy does not hold. It survived because hold-scored synthesis takes the
  6 mm / 119 N raw grasp and re-buries it to 20.58 mm / 13.6 kN wherever the fit
  starts, and *that* is the state the policy was trained on.
- **It cannot hold the plausible grasp.** From 6 mm / 119 N the same policy
  flies. So *tracking comes from penetration* is true of PPO as well as
  feedforward. RL is blocked on retargeting — not because the checkpoint breaks
  when the fit moves, but because it breaks when the hand is un-buried.

**One unreproduced reading.** The mug's frame-5 raw fit read 5.36 mm / 5
bodies / 105 N in one script and **6.10 mm / 7 bodies / 119 N** in four
independent processes since. The fit is deterministic across processes
(identical `tr.q` hash in two separate runs), a deliberate contamination test
(fresh → `reset_at(0)` → `rollout` → re-read) returns 6.10 at every step with the
model untouched, so neither nondeterminism nor state leakage explains the 5.36.
Its cause was not found. The table uses 6.10 / 7 / 119 N; the PPO result below
was measured on that pristine harness and does not depend on which is right.

### Why synthesis re-buries, and the run that tests the alternative

**Corroboration from the other session (reported, uncommitted as of this
entry).** A 40-reference sweep — retarget → CEM wrist search scored by drop
distance → hold — with an equilibrium-residual column added:

| reference | drop | contacts | residual |
|---|---:|---:|---:|
| `apple_eat_1` | 5.2 mm | 9 | **0.03×** |
| `banana_eat_1` | 7.2 mm | 5 | 0.09× |
| `binoculars_lift` | 9.1 mm | 10 | 0.10× |
| `airplane_fly_1` | 7.8 mm | 8 | 1.28× |
| `alarmclock_lift` | 6.3 mm | 50 | 1.48× |
| `bowl_drink_1` | **1.6 mm** | **94** | **1.90×** |

Every row "holds" by drop distance. `bowl_drink_1` holds *best* — on 94
contacts at 1.90× weight, which is the re-burial. `apple_eat_1` at 0.03× on 9
contacts is a real equilibrium. **Drop-scored synthesis cannot tell a grasp
from burial**, because burial is a perfectly good way to stop an object moving
and the score asks for nothing else. That is the mechanism behind the 6 mm →
20.58 mm re-burial above, seen from the other side. The fix being implemented
is to score synthesis by equilibrium residual. First A/B (w_eq 0 → 0.05, same
seed): `airplane_fly_1` grip **385 N → 6.9 N** on a 1.96 N object — the result;
`bowl_drink_1` 35.5 kN → 11.4 kN — a 3× reduction that is still burial. An
improvement, not a fix. **Caveat on the mechanism:** the 1.90× above was read
before settling; every settled winner, the 11.4 kN bowl included, reads
**0.00×**, because a settled burial's forces cancel. So the residual is not
separating settled candidates — whatever is doing the selecting during the
search is not yet identified, and the rationale should not be quoted until it
is.

Two further items from that session, also uncommitted at this writing: frame 0
is the approach on the **bimanual** path too — a reported 312 *km* tracking
error on `gamecontroller_play_1` was an integrator artifact from a diverged
sim, fixed by starting from the first self-holding frame and truncating on
loss (6.6 mm drop, 131 mm tracking) — and the first controlled two-hand
comparison, left hand parked rather than deleted so the model is identical:
**131.0 mm two-handed against 647.7 mm solo.**

**On the middle-phalanx term, one more correction.** The retraction above stands
on its data: a joint-centre measurement cannot refute a per-finger surface
measurement. But the other session found that `solve()` scaled every shape
target by `W_JOINT = 0`, so the entire `joint_targets` path — `W_MID`
included — was dead code. Nothing tonight ever measured that term in effect.
Wired properly it buys links *with* burial (4 → 5 links, 3.2 → 5.9 mm), and it
now ships off at `W_MID = 0.0`. The line the session drew from that is the one
this document converges on: **every knob in the retarget objective is a
position, and a position objective has no term that says hold the object.**
Equilibrium residual is the first candidate for such a term — with the caveat
that it only speaks at placement (see the instrumentation note below): whether
it separates candidates *during* a search that settles them is being checked.

**The run.** PPO trained from the raw `W_PEN_ARM` fit on `mug_drink_1`, no
hold-scored synthesis, `Pool.reset` a plain `reset_at` so every environment
starts from the 6.10 mm / 119 N grasp; horizon 160, 12 envs, 440 iterations
= 844,800 control steps, starts `5..115` to match the buried checkpoint's set;
saved to `results/ppo_mug_drink_1_h160_rawfit.pt`, existing checkpoint
untouched; ends with a 2×2 of old/new policy × raw/buried initial condition.
Launched 2026-09-15 late evening. Iteration 0: reward 0.225, **alive 0.770**,
flat through iteration 30. For calibration, the original buried checkpoint's
curve (from the session that trained it):

| iter | 0 | 10 | 20 | 50 | 100 | 200 | 399 |
|---|---:|---:|---:|---:|---:|---:|---:|
| reward | 0.682 | 0.656 | 0.713 | 0.745 | 0.771 | 0.897 | 0.876 |
| alive | 0.984 | 0.985 | 0.979 | 0.984 | 0.986 | 0.979 | 0.984 |

It began at 98.4% alive and never left 0.97–0.99: it was handed a state that
already held and spent 400 iterations polishing it. The raw-fit run at 0.77 is
a different regime from its first sample, not an early transient. Compare
`alive` first — the reward scales may differ — and do not call it before
iteration 100, since the original also dipped before it rose.
Measured rate **18.8 s/iteration, ~2.3 h total** (an earlier figure of
103 s/iteration in this entry divided elapsed time by *logged lines*, which
print every 10 iterations; it was wrong). The original checkpoint took
**6237 s = 14.2 s/iteration** (NOTES.md:3209) using **24 threaded
environments**; this run uses 12. Same 844,800 control steps and horizon, but
half the batch per PPO update and twice the updates — a confound to name: a
marginal result would not be attributable to the initial condition alone, and
would be rerun at 24 environments before anything was drawn from it. Result to
follow.

The 5.36 mm reading noted above was rerun under its exact original sequence and
does not reproduce; five processes agree on 6.10 mm.

## Look at the pose before trusting the number

`experiments/tracking/inspect_pose.py` renders a configuration from two angles
with its contact summary underneath. **Use it before believing any contact
metric.**

This was learned expensively. A pose reported as *0.54 mm penetration, 1 body in
contact, one-sidedness 1.000* reads like a near-miss grasp, and several hours of
sweeps were spent treating it as one. Rendered, it is a hand sitting **beside**
the cup with its fingers closing on empty air. The same picture showed that the
baseline — the configuration everyone had been calling broken — is a *correct*
cylindrical wrap: fingers curled round the outside, thumb opposing, 10 bodies
engaged at one-sidedness 0.055. Its only defect is that the cup passes through
the palm.

That reframed two conclusions at once:

- **Opening the fingers cannot fix it.** Backing them off 0.00 → 0.25 rad leaves
  penetration flat at 11.65 → 11.38 mm, because the buried part is the palm and
  the wrist, which finger joints do not move. It also destroys the wrap — at
  0.10 rad the fingers straighten out of their curl and tracking collapses from
  23.1 mm to 103343 mm.
- **For this cup, the hand is not mis-posed — it is too big.** A Shadow palm
  cannot wrap `cup_lift` without intersecting it. Every variant that tracks —
  nine combinations of retraction and closure — sits at 11.65–16.52 mm
  penetration and 4.6–8.0 kN.

**But that does not generalise, and the check is worth recording.** Across 16
references, object bounding radius explains very little of the penetration:

```
corr(object radius, penetration)  = -0.252
corr(object radius, n bodies)     = +0.415
small objects (<105 mm) mean penetration   20.11 mm
large objects (>=105 mm) mean penetration  17.14 mm
```

Objects of near-identical radius span 2.01–56.77 mm of penetration
(`phone_call_1` 88.0 mm radius / 2.01 mm, `mouse_use_1` 87.8 mm / 33.47 mm), so
"the hand is too large" is a correct description of the cup and not an
explanation of the stage. The one real signal is that **bigger objects engage
more hand bodies** (+0.415) — more of the hand can reach a larger surface —
which is about how much wrap is available, not about penetration depth.

```python
from experiments.tracking.inspect_pose import look, contacts, sheet
rt.reset_at(0)
sheet([look(rt.sim, "baseline")], "out/x.jpg")   # contact summary auto-captioned
```

`contacts()` returns max penetration, number of distinct hand bodies, and
one-sidedness (|mean contact normal|: near 0 is opposed contacts on different
sides — a wrap; 1.0 is all one direction, or nothing touching).

## Instrumentation

`make render-tracking` captures and renders the four reference rollouts. Each
GIF carries penetration, grip force as a multiple of object weight, and contact
count on its face, and flags `CONTACT SET IS AN ARTIFACT / NOT A GRASP` above
2 mm or 40× weight. Each has a JSON manifest with per-frame errors, penetration,
force, equilibrium residual, source and model hashes, and a replay check against
the evaluator.

**Equilibrium residual** is the scalar worth watching *at reset*: net
unbalanced wrench on the object, gravity included, in multiples of object
weight. **0× is equilibrium, 1× is freefall**, and freshly placed burials
measure 142–337×. Unlike penetration depth it rejects both failure ends at
placement — a hand that misses the object scores exactly 1×, where depth would
flatter it with 0 mm. **Caveat:** a burial that has *settled* reads ~0×, the
same as a real grasp, because its constraint forces cancel at rest. The metric
discriminates how a pose was placed, not its steady state; read it at reset,
never after stepping.

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
