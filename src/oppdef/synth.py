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

from oppdef.metrics.epsilon import grasp_metrics, object_contacts
from oppdef.hold import DIRECTIONS


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

    def __init__(self, hand_key, obj_half, mass=0.05, friction=(1.0, 0.02, 0.001),
                 n_hands=1, margin=0.0, kp_finger=1.0):
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
        b.add_geom(name="obj_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[float(x) for x in self.obj_half], mass=float(mass),
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

        if kp_finger:
            for p in self.prefixes:
                for _n, (_qa, a, _t) in self.finger[p].items():
                    self.m.actuator_gainprm[a, 0] = kp_finger
                    self.m.actuator_biasprm[a, 1] = -kp_finger

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

        # start open
        for p in self.prefixes:
            for _n, (qa, _a, _t) in self.finger[p].items():
                self.d.qpos[qa] = 0.0
        mujoco.mj_forward(self.m, self.d)

        # If the open hand starts inside the object, RETRACT it along its own
        # approach axis rather than throwing the sample away. f5d6's palm
        # overlapped by 1.6 mm at every orientation sampled, so a fixed
        # tolerance rejected 30 of 30 approaches for a hand that grasps
        # perfectly well 4 mm further back. Standoff is a free parameter of the
        # approach; the search should not be asked to guess it to the
        # millimetre.
        def _set_offsets(deltas):
            for p_, pr_, dd in zip(self.prefixes, params, deltas):
                rot_, off_, frac_ = pr_[:3], pr_[3], fracs[p_]
                self.place(p_, rot_, np.zeros(3), closure_frac_for_centre=frac_)
                palm_ = self.d.xpos[self.palm_bid[p_]]
                ax_ = self._grasp_centre(p_) - palm_
                nn = np.linalg.norm(ax_)
                ax_ = ax_ / nn if nn > 1e-9 else np.array([0.0, 0, 1.0])
                self.place(p_, rot_, -ax_ * (off_ + dd),
                           closure_frac_for_centre=frac_)
            for p_ in self.prefixes:
                for _n, (qa, _a, _t) in self.finger[p_].items():
                    self.d.qpos[qa] = 0.0
            mujoco.mj_forward(self.m, self.d)
            return self._penetration_mm()

        pen0 = self._penetration_mm()
        # Which way to retract is not obvious and is not the same for every
        # hand. Backing off along the palm->fingertip axis drags a short-fingered
        # hand's TIPS through the object -- f5d6 went from 0.5 mm of overlap to
        # 26 mm by "retracting" 6 cm. So both directions are tried and the one
        # that actually reduces overlap is taken.
        step = 0.0
        while pen0 > start_pen_mm and step < 0.06:
            step += 0.006
            cand = []
            for sgn in (+1.0, -1.0):
                cand.append((_set_offsets([sgn * step] * self.n_hands), sgn))
            best_sgn = min(cand)[1]
            pen0 = _set_offsets([best_sgn * step] * self.n_hands)
        if pen0 > start_pen_mm:
            return Attempt(params=params.ravel(), valid=False,
                           penetration_mm=pen0,
                           reason=f"pre-grasp starts inside the object "
                                  f"({pen0:.1f} mm) even retracted 5 cm")

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
                    self.d.ctrl[act] = a * goal
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
