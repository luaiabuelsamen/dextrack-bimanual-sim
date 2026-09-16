"""Grasps produced by CLOSING a hand in simulation, not by fitting joint angles.

The retargeting module optimised 16 joint angles against a geometric surrogate
and produced poses whose fingers were 19 mm inside the object (NOTES
2026-09-12, retraction). The surrogate could not see it: fingertips were points
at body origins and no other link was modelled at all. Tightening the surrogate
is possible, but the deeper problem is that free-space joint search has no
reason to produce a REACHABLE pose, and checking one afterwards is exactly the
work the simulator already does.

So the pose is never fitted. It is produced:

    pre-grasp   hand placed around the object, fingers open, no penetration
    close       finger targets ramped toward the hand's DERIVED closure
    squeeze     held while contact forces build
    measure     epsilon from REAL MuJoCo contacts (metrics.grasp_metrics)
    hold        a test wrench applied in many directions until it slips

Every pose is physically realisable by construction, which is the property the
geometric optimiser could not give at any tolerance. What is searched is the
APPROACH -- how the hand is oriented around the object and how far it closes --
which is five numbers, not sixteen, and every one of them is a quantity a real
system also has to choose.

This is the pre-grasp / grasp / squeeze structure of Dexonomy and DexGraspBench,
and it is the same structure that every hand-written closure in this project
failed without.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from oppdef.grasping.epsilon import grasp_metrics, object_contacts
from oppdef.grasping.hold import DIRECTIONS


@dataclass
class Attempt:
    """One synthesis attempt, measured."""
    params: np.ndarray
    epsilon: float = 0.0
    n_contacts: int = 0
    f_total: float = 0.0
    delta: float = float("inf")
    penetration_mm: float = 0.0
    valid: bool = False
    reason: str = ""
    min_force_N: float = -1.0
    per_direction_N: np.ndarray | None = None
    qpos: np.ndarray | None = None


class GraspScene:
    """One hand (or two) around one free object, with nothing underneath.

    Gravity is OFF throughout. A grasp that merely rests an object on the
    fingers is not a grasp, and with gravity on it is easy to mistake one for
    a hold; with gravity off the only thing keeping the object in place is the
    contact set, which is the quantity under test. Weight is reintroduced as
    one of the test wrench directions.
    """

    #: object shapes the bench can build. A cube is not a manipulation
    #: benchmark on its own -- every face is identical, so it hides any
    #: dependence of a metric on where the contacts sit.
    SHAPES = {"box": mujoco.mjtGeom.mjGEOM_BOX,
              "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
              "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}

    def __init__(self, hand_key, obj_half, mass=0.05, friction=(1.0, 0.02, 0.001),
                 n_hands=1, margin=0.0, kp_finger=1.0, shape="box"):
        from oppdef.embodiment import make, HANDS
        from oppdef.hands.specs import derive_flex

        self.hand_key, self.n_hands = hand_key, int(n_hands)
        self.h_palm = HANDS[hand_key].palm
        self.obj_half = np.asarray(obj_half, float)
        prefixes = ("",) if n_hands == 1 else ("rh_", "lh_")
        emb = make(hand=hand_key, count=n_hands, prefixes=prefixes,
                   free_base=True)
        spec = emb.spec
        b = spec.worldbody.add_body(name="obj", pos=[0.0, 0.0, 0.0])
        b.add_freejoint(name="obj_free")
        self.shape = shape
        gtype = self.SHAPES[shape]
        gsize = [float(x) for x in self.obj_half]
        if shape in ("cylinder", "capsule"):     # (radius, half-length, _)
            gsize = [float(self.obj_half[0]), float(self.obj_half[2]), 0.0]
        elif shape == "sphere":
            gsize = [float(self.obj_half[0]), 0.0, 0.0]
        b.add_geom(name="obj_geom", type=gtype,
                   size=gsize, mass=float(mass),
                   rgba=[0.85, 0.3, 0.2, 1.0],
                   friction=[float(x) for x in friction],
                   margin=float(margin))
        spec.option.timestep = 0.002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.gravity = [0.0, 0.0, 0.0]

        # Hands that come from a URDF ship no actuators at all -- f5d6 compiled
        # with nu = 6, the floating base only, so its fingers never moved and it
        # scored zero contacts on every object. Menagerie hands bring their own.
        _flex_probe = derive_flex(hand_key)[2]
        have = {a.target for a in spec.actuators}
        kpf = float(kp_finger or 3.0)
        from oppdef.hands.f5d6 import is_dependent, effort_of
        for pfx in prefixes:
            for j in _flex_probe:
                name = f"{pfx}{j}"
                if name in have:
                    continue
                # a joint driven by a mimic coupling must NOT get its own
                # actuator: it would fight the equality constraint that defines
                # it. f5d6's right hand has six independent joints, not eleven.
                if is_dependent(name):
                    continue
                gp = [0.0] * 10; gp[0] = kpf
                bp = [0.0] * 10; bp[1], bp[2] = -kpf, -0.05
                act = spec.add_actuator(
                    name=f"fing_{name}", target=name,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                    gainprm=gp, biasprm=bp,
                    gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                    biastype=mujoco.mjtBias.mjBIAS_AFFINE)
                # torque limits from the URDF: 0.5 N*m per finger joint, 1.0 at
                # the thumb. Without them the simulated hand squeezes as hard
                # as the controller asks.
                eff = effort_of(name)
                if eff is not None:
                    act.forcelimited = 1
                    act.forcerange = [-eff, eff]
        # Everything that is not the hand or its floating base is FROZEN. f5d6's
        # file is the whole Vega robot, and its torso, lift, head and both arms
        # are unactuated: hung off a floating base they flop, and the hand was
        # flung 1.2 m from the object within half a second of closing. Deleting
        # those joints makes the robot rigid from torso to wrist, which is what
        # a bench-mounted hand actually is. For a Menagerie hand this deletes
        # nothing -- they contain only hand joints.
        # Only the HAND collides. f5d6's file is a whole 1.5 m robot, so rotating
        # the hand about the object swept its torso straight through it: 40 of
        # 40 sampled approaches were rejected as starting inside the object,
        # which reads as a hand that cannot be placed and is really a torso in
        # the way. The bodies stay (the kinematic chain is needed, and it is
        # rigid now that the joints are gone); their geoms do not.
        def _subtree_names(body):
            out = {body.name}
            for ch in body.bodies:
                out |= _subtree_names(ch)
            return out

        # The keep-set is the UNION over hands, computed before anything is
        # deleted. Deleting per hand in a loop removes the OTHER hand's geoms on
        # the second pass: every two-hand cell scored 0 contacts, and Allegro
        # would not even compile ("mass and inertia of moving bodies must be
        # larger than mjMINVAL") because a stripped body had no mass left.
        keep_bodies = set()
        for pfx in prefixes:
            for bd in spec.bodies:
                if bd.name == f"{pfx}{self.h_palm}":
                    keep_bodies |= _subtree_names(bd)
                    break
        if keep_bodies:
            for g in list(spec.geoms):
                par = g.parent
                if par is not None and par.name not in keep_bodies \
                        and par.name != "obj":
                    spec.delete(g)

        keep = {"obj_free"}    # the object must stay free to be dropped
        for pfx in prefixes:
            keep |= {f"{pfx}{d}" for d in ("x", "y", "z", "rx", "ry", "rz")}
            keep |= {f"{pfx}{j}" for j in _flex_probe}
        removed = set()
        for jt in list(spec.joints):
            if jt.name not in keep:
                removed.add(jt.name)
                spec.delete(jt)
        # An equality constraint naming a deleted joint does not disappear with
        # it -- the model then fails to compile with "unknown element in
        # equality constraint". The mimic couplings cover BOTH hands, so
        # freezing the left one strands five of them.
        for eq in list(spec.equalities):
            if getattr(eq, "name1", "") in removed or \
                    getattr(eq, "name2", "") in removed:
                spec.delete(eq)

        # The floating base's slides are limited to +/-0.6 m by the bimanual
        # env they come from. f5d6's grasp centre sits 1.4 m from the origin --
        # its hand is part of a whole robot -- so the hand physically could not
        # be brought to the object, closed on empty air, and recorded 0 contacts
        # at EVERY width, which reads exactly like an opposition deficit and is
        # not one.
        for jt in spec.joints:
            if jt.name.endswith(("x", "y", "z")) and not jt.name.endswith(
                    ("rx", "ry", "rz")) and jt.type == mujoco.mjtJoint.mjJNT_SLIDE:
                jt.range = [-2.5, 2.5]
        self.spec = spec
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)

        self.h = HANDS[hand_key]
        self.prefixes = tuple(emb.palm_bid.keys())
        self.obj_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
        self.obj_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_q = self.m.jnt_qposadr[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")]

        # the closing posture, DERIVED once (specs solves it over all joints)
        _m, _cfg, flex, _tips, _gap = derive_flex(hand_key)
        self.flex = flex
        self.finger = {}     # prefix -> {joint name: (qadr, actuator id, target)}
        self.base = {}       # prefix -> {dof: actuator id}
        self.tip_bids = {p: list(emb.tip_bid[p]) for p in self.prefixes}
        self.palm_bid = dict(emb.palm_bid)
        act_of = {}
        for a in range(self.m.nu):
            if self.m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                jn = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                       int(self.m.actuator_trnid[a, 0]))
                act_of[jn] = a
        for p in self.prefixes:
            self.finger[p] = {}
            for j, v in flex.items():
                name = f"{p}{j}"
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0 or name not in act_of:
                    continue
                lo, hi = self.m.jnt_range[jid]
                tgt = float(np.clip(v, lo, hi)) if hi > lo else float(v)
                self.finger[p][name] = (int(self.m.jnt_qposadr[jid]),
                                        act_of[name], tgt)
            self.base[p] = {dof: act_of[f"{p}{dof}"]
                            for dof in ("x", "y", "z", "rx", "ry", "rz")
                            if f"{p}{dof}" in act_of}
        self.base_q = {p: {dof: int(self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}{dof}")])
            for dof in ("x", "y", "z", "rx", "ry", "rz")}
            for p in self.prefixes}
        # The base actuators' ctrlrange must be widened with the joint range, not
        # instead of it: left at +/-0.6 m the command that brings the hand to the
        # object is clamped, and the actuator then drags the hand back toward
        # 0.6 m -- f5d6 drifted from 5 cm to 59 cm away from the object while its
        # fingers closed perfectly on nothing.
        for p_ in self.prefixes:
            for dof, a in self.base[p_].items():
                if dof in ("x", "y", "z"):
                    self.m.actuator_ctrlrange[a] = [-2.5, 2.5]

        # The pre-grasp posture is DERIVED, not assumed to be zero. `qpos = 0`
        # is whatever zero means in a model file; for Shadow it leaves the hand
        # occupying its own grasp volume, so no placement existed that was both
        # clear when open and touching when closed. See hands/axis.aperture_pose.
        from oppdef.hands.axis import aperture_pose
        _open = aperture_pose(hand_key)
        # The aperture pose is how WIDE the hand can open; how wide it SHOULD
        # open is decided per attempt. Opening maximally rescued Shadow, whose
        # zero pose occupies its own grasp volume, and starved Allegro, which
        # then closed from a fully splayed pose and missed the object. The
        # pre-grasp is the LEAST opening that clears -- `open_q(alpha)`
        # interpolates zero -> aperture and the attempt takes the smallest
        # alpha that works, so a hand that needs nothing gets nothing.
        self.aperture_q = {}
        for p_ in self.prefixes:
            self.aperture_q[p_] = {n: float(_open.get(n[len(p_):], 0.0))
                                   for n in self.finger[p_]}
        self.open_q = {p_: {n: 0.0 for n in self.finger[p_]}
                       for p_ in self.prefixes}

        if kp_finger:
            for p in self.prefixes:
                for _n, (_qa, a, _t) in self.finger[p].items():
                    self.m.actuator_gainprm[a, 0] = kp_finger
                    self.m.actuator_biasprm[a, 1] = -kp_finger

    def set_open_fraction(self, alpha):
        """Pre-grasp posture: `alpha` of the way from zero toward the aperture.

        How wide the hand SHOULD open is a per-attempt decision. Opening
        maximally rescued Shadow, whose zero pose occupies its own grasp
        volume, and starved Allegro, which then closed from a fully splayed
        pose and missed the object. The attempt takes the smallest alpha that
        clears, so a hand that needs no extra opening gets none.
        """
        a = float(np.clip(alpha, 0.0, 1.0))
        for p_ in self.prefixes:
            self.open_q[p_] = {n: a * v for n, v in self.aperture_q[p_].items()}

    # ------------------------------------------------------------------
    def _grasp_centre(self, prefix):
        tips = self.d.xpos[self.tip_bids[prefix]]
        if len(tips) < 2:
            return tips.mean(0)
        return 0.5 * (tips[-1] + tips[:-1].mean(0))

    def place(self, prefix, rot, offset, closure_frac_for_centre=1.0):
        """Orient the hand, then translate it so its grasp centre meets the object.

        Rotation first, translation second, and the translation is computed
        AFTER the rotation is applied -- rotating the hand moves its grasp
        centre, so a translation chosen beforehand puts the object somewhere
        else entirely.
        """
        bq = self.base_q[prefix]
        for dof, v in zip(("rx", "ry", "rz"), rot):
            self.d.qpos[bq[dof]] = float(v)
        for dof in ("x", "y", "z"):
            self.d.qpos[bq[dof]] = 0.0
        # measure the grasp centre at the CLOSED posture: that is where the
        # object has to be for the fingers to arrive around it
        saved = {}
        for name, (qa, _a, tgt) in self.finger[prefix].items():
            saved[qa] = self.d.qpos[qa]
            self.d.qpos[qa] = tgt * closure_frac_for_centre
        mujoco.mj_kinematics(self.m, self.d)
        centre = self._grasp_centre(prefix)
        for qa, v in saved.items():
            self.d.qpos[qa] = v
        want = np.asarray(offset, float)          # object is at the origin
        delta = want - centre
        lim = []
        for i, dof in enumerate(("x", "y", "z")):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{prefix}{dof}")
            lo, hi = self.m.jnt_range[jid]
            v = float(np.clip(delta[i], lo, hi))
            lim.append(abs(v - delta[i]) > 1e-9)
            self.d.qpos[bq[dof]] = v
        mujoco.mj_kinematics(self.m, self.d)
        return any(lim)

    def _penetration_mm(self):
        worst = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 != self.obj_gid and c.geom2 != self.obj_gid:
                continue
            worst = max(worst, -float(c.dist))
        return worst * 1000.0

    #: geometric force ladder at ~1.4x per rung. A doubling ladder left 11 of
    #: 24 comparison cells TIED -- not because the grasps were equal but
    #: because the measurement could not tell them apart.
    LADDER = tuple(round(0.25 * (1.4 ** k), 3) for k in range(15))

    def hold_of(self, ladder=None, push_steps=300, max_disp=0.02,
                max_tilt_deg=15.0, torque=True):
        """Push AND twist the object until it slips; report the worst direction.

        Forces alone are not the test epsilon deserves. Epsilon is the radius of
        the largest ball in a six-dimensional wrench space, and the half of that
        space this project cares about is torque: a grasp with poor opposition
        can still resist forces through friction, and fails on moments. Probing
        only the 14 force directions measures the half of the ball where the
        difference is smallest, and the earlier force-only comparison duly found
        nothing (p = 0.67).

        So pure torques are applied about the same directions, scaled by the
        object's characteristic length so a newton and a newton-metre are
        comparable rungs of one ladder, and the pose must survive them without
        rotating away: orientation is gated at `max_tilt_deg`, which a
        translation-only gate never checked.
        """
        ladder = ladder or self.LADDER
        lam = float(np.linalg.norm(self.obj_half)) or 0.05
        snap = (self.d.qpos.copy(), self.d.qvel.copy(), self.d.ctrl.copy())
        modes = [("f", u) for u in DIRECTIONS]
        if torque:
            modes += [("t", u) for u in DIRECTIONS]
        per = np.zeros(len(modes))

        def _tilt(q0, q1):
            dq = np.zeros(4)
            mujoco.mju_mulQuat(dq, q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
            return float(np.degrees(2.0 * np.arccos(np.clip(abs(dq[0]), -1, 1))))

        for i, (kind, u) in enumerate(modes):
            best = 0.0
            for f in ladder:
                self.d.qpos[:], self.d.qvel[:] = snap[0].copy(), snap[1].copy()
                self.d.ctrl[:] = snap[2]
                mujoco.mj_forward(self.m, self.d)
                p0 = self.d.qpos[self.obj_q:self.obj_q + 3].copy()
                q0 = self.d.qpos[self.obj_q + 3:self.obj_q + 7].copy()
                for _ in range(push_steps):
                    if kind == "f":
                        self.d.xfrc_applied[self.obj_bid, :3] = u * f
                    else:
                        self.d.xfrc_applied[self.obj_bid, 3:] = u * f * lam
                    mujoco.mj_step(self.m, self.d)
                self.d.xfrc_applied[self.obj_bid, :] = 0.0
                moved = float(np.linalg.norm(
                    self.d.qpos[self.obj_q:self.obj_q + 3] - p0))
                tilt = _tilt(q0, self.d.qpos[self.obj_q + 3:self.obj_q + 7])
                touching = len(object_contacts(self.m, self.d, self.obj_gid)[0])
                if moved < max_disp and tilt < max_tilt_deg and touching > 0:
                    best = f
                else:
                    break
            per[i] = best
        self.d.qpos[:], self.d.qvel[:] = snap[0], snap[1]
        self.d.ctrl[:] = snap[2]
        mujoco.mj_forward(self.m, self.d)
        nf = len(DIRECTIONS)
        return float(per.min()), per

    def attempt(self, params, close_steps=500, squeeze_steps=400,
                start_pen_mm=1.0, do_hold=True, pin=True, release_steps=300,
                max_pen_mm=3.0, finger_target=None):
        """params per hand: (rx, ry, rz, radial_offset, closure_fraction).

        `finger_target` overrides the closure: a dict {joint name: angle} that
        the fingers are driven to instead of a fraction of the hand's own
        derived closure. That is what makes a RETARGETED finger pose testable
        in the same harness as a searched one -- the placement search is
        identical, and only the thing that decides the finger angles differs.
        """
        params = np.asarray(params, float).reshape(self.n_hands, 5)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.obj_q:self.obj_q + 3] = 0.0
        self.d.qpos[self.obj_q + 3:self.obj_q + 7] = [1, 0, 0, 0]

        fracs = {}
        for p, pr in zip(self.prefixes, params):
            rot, off, frac = pr[:3], pr[3], float(np.clip(pr[4], 0.05, 1.0))
            fracs[p] = frac
            # offset pushes the hand back along its own palm->tips axis, so a
            # bigger object can still be approached without starting inside it
            self.place(p, rot, np.zeros(3), closure_frac_for_centre=frac)
            palm = self.d.xpos[self.palm_bid[p]]
            centre = self._grasp_centre(p)
            axis = centre - palm
            n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-9 else np.array([0.0, 0, 1.0])
            if self.place(p, rot, -axis * off, closure_frac_for_centre=frac):
                return Attempt(params=params.ravel(), valid=False,
                               reason="hand cannot reach the object: base "
                                      "translation hit its joint limit")

        # start at the DERIVED open posture
        for p in self.prefixes:
            for _n, (qa, _a, _t) in self.finger[p].items():
                self.d.qpos[qa] = self.open_q[p][_n]
        mujoco.mj_forward(self.m, self.d)

        # APPROACH ALONG A RAY, rather than placing the grasp centre at the
        # object and retracting from there.
        #
        # The grasp centre is computed at the CLOSED pose, so for a hand with a
        # large palm -- or a forearm, as Shadow's model has -- that point lies
        # inside the hand's own volume, and putting the object there puts the
        # object inside the hand. Retracting did not rescue it: 98.4% of
        # Shadow's candidates and 95.1% of f5d6's were rejected as pre-grasp
        # penetrating, so two of four hands contributed nothing to any
        # experiment in this repository (NOTES 2026-09-13).
        #
        # Penetration is also NOT monotone in standoff -- f5d6 measured 0.5 mm,
        # then 10 mm, then 26 mm, then 0 as the object passes the fingers -- so
        # a bisection finds nothing. The whole ray is scanned and the DEEPEST
        # clear placement is taken, which is "as enclosed as the open hand can
        # be without overlapping", for any hand.
        def _pen_at(t_by_hand):
            for p_, pr_, t_ in zip(self.prefixes, params, t_by_hand):
                rot_, frac_ = pr_[:3], fracs[p_]
                self.place(p_, rot_, np.zeros(3), closure_frac_for_centre=frac_)
                palm_ = self.d.xpos[self.palm_bid[p_]]
                ax_ = self._grasp_centre(p_) - palm_
                nn = np.linalg.norm(ax_)
                ax_ = ax_ / nn if nn > 1e-9 else np.array([0.0, 0, 1.0])
                self.place(p_, rot_, -ax_ * t_, closure_frac_for_centre=frac_)
            for p_ in self.prefixes:
                for _n, (qa, _a, _t) in self.finger[p_].items():
                    self.d.qpos[qa] = self.open_q[p_][_n]
            mujoco.mj_forward(self.m, self.d)
            return self._penetration_mm()

        def _closed_contacts(t_by_hand):
            """Contacts the object would have if the fingers closed from here."""
            _pen_at(t_by_hand)
            for p_ in self.prefixes:
                for _n, (qa, _a, tgt) in self.finger[p_].items():
                    goal = (tgt * fracs[p_] if finger_target is None
                            else finger_target.get(_n[len(p_):], 0.0))
                    self.d.qpos[qa] = goal
            mujoco.mj_forward(self.m, self.d)
            n = 0
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                if c.geom1 == self.obj_gid or c.geom2 == self.obj_gid:
                    n += 1
            return n

        # Place where the search asks, and fall back to the ray only if that
        # overlaps. The original rule -- grasp centre at the object plus a
        # searched standoff -- works for hands whose grasp centre lies in front
        # of the palm, and replacing it wholesale traded one hand against
        # another every time I tried (deepest-clear suited Shadow and starved
        # Allegro; most-contacts reversed it; indexing the clear set uniformly
        # starved both). So it is kept, and the scan supplies the NEAREST
        # feasible placement only when the requested one is inside the object.
        #
        # That is what rescues the hands whose grasp centre sits inside their
        # own volume: Shadow's model includes a forearm, and 98.4% of its
        # candidates used to be rejected before a finger ever moved.
        t_want = float(np.mean([pr[3] for pr in params]))
        pen_want, alpha_used = None, 0.0
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.set_open_fraction(alpha)
            pen_want = _pen_at([t_want] * self.n_hands)
            alpha_used = alpha
            if pen_want <= start_pen_mm:
                break
        if pen_want <= start_pen_mm:
            t_pick = t_want
        else:
            reach = float(np.linalg.norm(self.obj_half)) + 0.20
            clear = [t for t in np.linspace(0.0, reach, 40)
                     if _pen_at([t] * self.n_hands) <= start_pen_mm]
            if not clear:
                return Attempt(params=params.ravel(), valid=False,
                               penetration_mm=pen_want,
                               reason="no penetration-free placement along the "
                                      "approach ray")
            t_pick = min(clear, key=lambda t: abs(t - t_want))
        pen0 = _pen_at([t_pick] * self.n_hands)

        # hold the base where it was placed
        for p in self.prefixes:
            for dof, a in self.base[p].items():
                self.d.ctrl[a] = self.d.qpos[self.base_q[p][dof]]

        # The object is PINNED while the grasp forms. Closing on a free object
        # ejects it: the fingers arrive at slightly different times, the first
        # one to touch accelerates it, and it leaves before the others land --
        # measured at 5-8 cm every time. Synthesis datasets form the grasp
        # against a held object for exactly this reason, and the disturbance
        # test afterwards is what decides whether the result is real. Pinning
        # is released before anything is measured.
        def _pin():
            self.d.qpos[self.obj_q:self.obj_q + 3] = 0.0
            self.d.qpos[self.obj_q + 3:self.obj_q + 7] = [1, 0, 0, 0]
            self.d.qvel[self.m.jnt_dofadr[mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")]:][:6] = 0.0

        for k in range(close_steps):
            a = (k + 1) / close_steps
            for p in self.prefixes:
                for _n, (_qa, act, tgt) in self.finger[p].items():
                    goal = (tgt * fracs[p] if finger_target is None
                            else finger_target.get(_n[len(p):], 0.0))
                    # ramp FROM the open posture, not from zero: starting the
                    # ramp at zero makes the fingers jump to a pose the
                    # pre-grasp was never checked at
                    self.d.ctrl[act] = (1 - a) * self.open_q[p][_n] + a * goal
            mujoco.mj_step(self.m, self.d)
            if pin:
                _pin()
        for _ in range(squeeze_steps):
            mujoco.mj_step(self.m, self.d)
            if pin:
                _pin()
        if pin:
            # released, then allowed to settle on its own before measurement:
            # a grasp that only holds while the object is pinned is not a grasp
            for _ in range(release_steps):
                mujoco.mj_step(self.m, self.d)

        gm = grasp_metrics(self.m, self.d, self.obj_gid, self.obj_bid)
        moved = float(np.linalg.norm(self.d.qpos[self.obj_q:self.obj_q + 3]))
        att = Attempt(params=params.ravel(), epsilon=gm["epsilon"],
                      n_contacts=gm["n_contacts"], f_total=gm["f_total"],
                      delta=gm["delta"], penetration_mm=self._penetration_mm(),
                      valid=True, qpos=self.d.qpos.copy())
        if moved > 0.05:
            att.valid = False
            att.reason = f"object pushed {moved*100:.1f} cm out of the hand"
            return att
        # A pose that only scores well because the fingers are buried in the
        # object is not a grasp. At kp = 3 the search found "grasps" pressed
        # 5-19 mm into a 25 mm half-extent box, carrying 40-310 N; the same
        # approaches at kp = 1 sit at ~2 mm and ~16 N. Epsilon rises with
        # penetration because buried fingers manufacture contacts, so the gate
        # has to be on the pose, not on the score.
        if att.penetration_mm > max_pen_mm:
            att.valid = False
            att.reason = (f"fingers buried {att.penetration_mm:.1f} mm in the "
                          f"object")
            return att
        if do_hold and gm["n_contacts"] >= 2:
            att.min_force_N, att.per_direction_N = self.hold_of()
        return att
