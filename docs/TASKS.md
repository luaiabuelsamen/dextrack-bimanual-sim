# Task definitions

Every number in this repository comes from one of the tasks below. Each is
specified here in full — scene, formation protocol, measurement, success
criterion and validity gates — so a result can be checked against the task it
claims to be about.

Validity gates matter as much as success criteria. This project has twice
recorded a "success" that was an artifact: a box resting on the floor passing a
*displacement* test, and a hand *falling* while a bench recorded it as failing
to grasp. Every gate below exists because something got through without it.

---

## T1 — Grasp and hold under a wrench probe

The main benchmark. One hand, one cube, no arm, no table.

**Scene.** A single hand on a six-DoF position-controlled floating base; one
free cube of half-extent *w*/2 and mass *m*; **no floor**; **gravity off**.

*Why no floor:* a failed grasp must be unambiguous. With nothing underneath, a
dropped object accelerates away and its displacement grows without bound —
there is no resting place that could be mistaken for a hold.

*Why gravity off:* a grasp that merely rests an object on the fingers is not a
grasp, and with gravity on it is easy to mistake one for a hold. Weight is
reintroduced as one of the probe directions.

**Formation** — pre-grasp → close → squeeze → release, the structure Dexonomy
and DexGraspBench use, and the one every hand-written closure in this project
failed without:

1. **Pre-grasp.** The hand is placed by an approach (three rotations plus a
   standoff along its own palm→fingertip axis) with the fingers open. If the
   open hand starts inside the object it is retracted along that axis — in
   *whichever direction reduces overlap*, since backing off drags a
   short-fingered hand's tips *through* the object.
2. **Close.** Finger position targets are ramped to their goal over 500 steps.
   The object is **pinned** while the grasp forms: closing on a free object
   ejects it, because the first finger to touch accelerates it away before the
   others land (measured at 5–8 cm every time).
3. **Squeeze.** 400 further steps at the held targets, so contact forces build.
4. **Release.** The pin is removed and the object is allowed to settle for 300
   steps before anything is measured. *A grasp that only holds while the object
   is pinned is not a grasp.*

**Measurement.** ε (Ferrari–Canny) computed from **real MuJoCo contacts** —
true positions, normals and friction — never from projected fingertip points.

**The probe.** A wrench is applied directly to the object and ramped along a
geometric ladder (×1.4 per rung from 0.25 N). 14 pure-force directions (6 axes
+ 8 octant diagonals) and 14 pure-torque directions about the same axes, scaled
by the object's characteristic length. Each rung runs 0.6 s.

*Why torques:* opposition buys torque resistance. A grasp with poor opposition
resists forces through friction and fails on moments, so a force-only probe
measures the half of the wrench ball where the difference is smallest — and
duly found nothing (p = 0.67) where the 6-D probe did not.

**Success (binary).** *Survives the probe* = every sampled direction holds at
least the smallest rung, with final displacement < 20 mm, final rotation < 15°,
and at least one contact remaining.

> This is **not** "survives any 6-D wrench". The probe samples pure forces and
> pure torques, not arbitrary combined wrenches, and a pass means the smallest
> rung held in every sampled direction. Earlier logs overstated this.

**Validity gates.** An attempt is discarded, not scored, if: the pre-grasp
penetrates the object by > 1 mm after retraction; the object is pushed > 5 cm
out of the hand; or the fingers end **buried > 3 mm** in the object. The last
one matters — at a higher squeeze gain the search found "grasps" pressed 5–19 mm
into a 25 mm half-extent cube carrying 40–310 N. ε rises with penetration,
because buried fingers manufacture contacts.

**Secondary measures.** Worst-direction magnitude in newtons; *valid candidates
found per budget*, which separates finding grasps from choosing among them.

| | |
|---|---|
| ![wrench objective](../figures/task_grasp_wrench.gif) | ![pose objective](../figures/task_grasp_pose.gif) |
| **wrench objective** — ε 0.642, 9 contacts, holds 1.345 N | **pose objective** — ε 0.358, 4 contacts, holds 0.000 N |

Both are Allegro on the same 6 cm cube, same seed, same budget. Each closes,
is released, then is pushed in four directions.

---

## T2 — Bimanual peg extraction

The one bimanual result that survives every retraction in this repository, and
the only task here that is *provably* two-handed.

**Scene.** Two LEAP hands on floating bases; a base block with a socket; a peg
in that socket, on a table.

**The force balance is the point.** Socket friction is **1.20 N** against a base
weight of **0.78 N**. A one-handed pull therefore lifts the *base* instead of
extracting the peg — the task cannot be done by one hand regardless of how well
that hand grasps, which is what makes it a bimanual task rather than a hard one.

**Success.** Peg extracted ≥ 8 cm, base lift < 2 cm, base tilt < 15°.

**Control.** The same expert restricted to one hand, run every time — not
assumed.

| two-handed expert | one-handed control |
|---|---|
| ![two handed](../figures/task_peg_two_handed.gif) | ![one handed](../figures/task_peg_one_handed.gif) |
| peg out **13.49 cm**, base lift +0.44 cm — **success** | peg out 5.32 cm, base lift **+10.26 cm** — **failure** |

The one-handed arm does not fail by dropping the peg; it fails by *lifting the
whole base*, which is the force balance showing up exactly where predicted.

---

## M1 — The opposition axis *(a measurement, not a task)*

How closely can a hand's thumb meet its fingers?

    floor    = min over poses of ‖ thumb_tip(q) − nearest fingertip(q) ‖
    aperture = max of the same quantity

Multi-start, subject to joint limits, with the hand's mimic couplings enforced
and any pose penalised in which two **finger** bodies interpenetrate.

**The fingertip is derived, not tabled:** the point of each distal link's own
collision geometry lying furthest from that body's origin. Checked against the
one hand whose URDF names its tip frames, the rule recovers them to 0.4 mm.

Three definitional traps, each found when the previous fix failed:

* fingertips are **not** body origins — every hand's most distal joint rotates
  about its tip body's origin, so tracking origins is blind to the last joint of
  every finger (measured displacement: exactly 0.000 mm);
* the distance is to the **nearest fingertip**, not to their mean — a thumb
  among splayed fingers is zero from the mean while touching nothing;
* self-collision must be **scoped to the fingers** — unscoped, f5d6's deepest
  overlap is between two links of the robot's *head*.

Run it with `make axis`. Provenance (commit, package versions, machine) is
written alongside the numbers.

**What the number does not say.** A small floor means two points can approach
each other. It does not mean the resulting contact normals, reachable object
placements, friction and torque limits can supply any particular task wrench.
Treat it as a geometric descriptor, not a capability.
