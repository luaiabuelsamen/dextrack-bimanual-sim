"""Tracking a GRAB reference: hold the real object, follow the human's path.

The reference is expressed in the FIRST FRAME's object frame, so the object
starts at the origin with identity orientation -- exactly the frame the
retargeting solved in, which means a retargeted pose can be used as the initial
state without composing any transforms.  What the object then does in that
frame is the task.

Before any of that is worth building, one question has to be answered honestly:
does a retargeted human grasp hold the object at all under gravity?  A tracking
controller that starts from a grasp which drops the object in 200 ms is not
being evaluated on tracking.  `hold` measures precisely that, with the hand
commanded to stay exactly where the retargeting put it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from oppdef.human import grab as grab_mod

GRAVITY = np.array([0.0, 0.0, -9.81])
BASE_DOF = ("x", "y", "z", "rx", "ry", "rz")


def reference_in_first_frame(seq, k0: int):
    """The object's GRAB trajectory, re-expressed in frame k0's object frame.

    In that frame the object starts at the origin with identity orientation,
    which is where the retargeting put the hand.
    """
    R0 = seq.obj_R[k0]
    p0 = seq.obj_pos[k0]
    pos = (seq.obj_pos - p0) @ R0
    rel = np.einsum("ab,tbc->tac", R0.T, seq.obj_R)
    return pos, grab_mod._quat_from_R(rel)


def widen_base(sc, reach: float = 1.5):
    """Give the floating base enough travel to follow a GRAB clip.

    The base slides ship with 0.6 m, which is right for holding a grasp in
    place and far too little for carrying one: the mug in `mug_drink_1` moves
    1.21 m and reaches 0.88 m from its start. Pinned at the limit, the
    feedforward IK reports 166 mm of residual and looks like a solver failure
    rather than a modelling one. Widened here, in the code that needs it, so
    every earlier result keeps the limits it was measured under.
    """
    m = sc.model
    for j in sc.jids:
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n in ("x", "y", "z"):
            m.jnt_range[j] = [-reach, reach]
    for a in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if n[:-4] in ("x", "y", "z"):
            m.actuator_ctrlrange[a] = [-reach, reach]
    return sc


@dataclass
class HoldResult:
    seq: str
    object: str
    hand: str
    held: bool
    drop_m: float          # how far the object fell from where it started
    settle_m: float        # displacement over the last 25% of the hold
    n_contact: int         # hand-object contacts at the end
    pen_mm: float          # worst penetration at the start, after placement
    seconds: float


class GrabTrackEnv:
    """A robot hand and a GRAB object, in the first frame's object frame."""

    def __init__(self, seq, hand: str = "shadow", side: str = "rhand",
                 sc=None, obj_mass: float = 0.2, reach: float = 1.5):
        from oppdef.human.scene import build as build_scene

        self.seq, self.side, self.hand = seq, side, hand
        self.sc = sc or build_scene(hand=hand, obj=seq.obj, obj_static=False)
        m = self.sc.model

        widen_base(self.sc, reach)

        self.obj_jid = [j for j in range(m.njnt)
                        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.obj_qadr = int(m.jnt_qposadr[self.obj_jid])
        self.obj_vadr = int(m.jnt_dofadr[self.obj_jid])
        obj_bid = int(m.jnt_bodyid[self.obj_jid])
        if obj_mass:
            scale = obj_mass / max(float(m.body_mass[obj_bid]), 1e-9)
            m.body_mass[obj_bid] *= scale
            m.body_inertia[obj_bid] *= scale

        self._act_map = self._build_act_map()
        self._dir = None
        self.obj_vadr = self.obj_vadr if hasattr(self, "obj_vadr") else int(
            m.jnt_dofadr[self.obj_jid])

    def _build_act_map(self):
        """actuator -> the joint indices (into sc.jids) it drives.

        Shadow couples the distal two joints of each finger onto one actuator
        through a tendon, so the mapping is not one-to-one and a naive
        actuator-per-joint assignment drives the wrong targets.
        """
        m = self.sc.model
        jid_pos = {j: i for i, j in enumerate(self.sc.jids)}
        out = []
        for a in range(m.nu):
            tt = m.actuator_trntype[a]
            if tt == mujoco.mjtTrn.mjTRN_JOINT:
                j = int(m.actuator_trnid[a, 0])
                out.append((a, [jid_pos[j]] if j in jid_pos else []))
            elif tt == mujoco.mjtTrn.mjTRN_TENDON:
                t = int(m.actuator_trnid[a, 0])
                lo = int(m.tendon_adr[t])
                idx = [int(m.wrap_objid[lo + w]) for w in range(int(m.tendon_num[t]))]
                out.append((a, [jid_pos[j] for j in idx if j in jid_pos]))
            else:
                out.append((a, []))
        return out

    def ctrl_for(self, q: np.ndarray) -> np.ndarray:
        """Position-servo targets that hold joint configuration `q`."""
        m = self.sc.model
        c = np.zeros(m.nu)
        for a, idx in self._act_map:
            if idx:
                c[a] = float(np.sum(q[idx]))     # a tendon sees the sum
                c[a] = np.clip(c[a], *m.actuator_ctrlrange[a])
        return c

    def reset(self, q: np.ndarray, obj_pos=None, obj_quat=None):
        """Place the hand at `q` and the object at the reference origin."""
        m, d = self.sc.model, self.sc.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.sc.qadr] = q
        d.qpos[self.obj_qadr:self.obj_qadr + 3] = (
            np.zeros(3) if obj_pos is None else obj_pos)
        d.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = (
            np.array([1.0, 0, 0, 0]) if obj_quat is None else obj_quat)
        d.ctrl[:] = self.ctrl_for(q)
        mujoco.mj_forward(m, d)
        return d

    def obj_state(self):
        d = self.sc.data
        return (d.qpos[self.obj_qadr:self.obj_qadr + 3].copy(),
                d.qpos[self.obj_qadr + 3:self.obj_qadr + 7].copy())

    def n_hand_contacts(self) -> int:
        d = self.sc.data
        obj, hand = set(self.sc.obj_gids), set(self.sc.hand_gids)
        n = 0
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if ({g1, g2} & obj) and ({g1, g2} & hand):
                n += 1
        return n

    def hold(self, q: np.ndarray, seconds: float = 1.5,
             settle: float = 0.25, grip: float | None = None,
             ref_pose=None) -> HoldResult:
        """Command the hand to stay at `q` and see whether the object stays.

        `settle` seconds are simulated before the object's position is taken as
        the origin, so the few millimetres of contact resolution that the
        retargeting leaves are not counted as a drop.
        """
        m, d = self.sc.model, self.sc.data
        self.reset(q)
        pen0, _ = self.sc.penetration()
        if grip:
            if self._dir is None:
                self._dir = closing_direction(self.sc, self._act_map)
            pose = ref_pose or (d.qpos[self.obj_qadr:self.obj_qadr + 3].copy(),
                                d.qpos[self.obj_qadr + 3:self.obj_qadr + 7].copy())
            establish_grip(self.sc, self._dir, self.obj_qadr, self.obj_vadr,
                           pose, target_n=grip)
        n_settle = int(settle / m.opt.timestep)
        for _ in range(n_settle):
            mujoco.mj_step(m, d)
        p_ref, _ = self.obj_state()

        n = int(seconds / m.opt.timestep)
        traj = np.empty((n, 3))
        for i in range(n):
            mujoco.mj_step(m, d)
            traj[i] = d.qpos[self.obj_qadr:self.obj_qadr + 3]

        drop = float(np.linalg.norm(traj[-1] - p_ref))
        tail = traj[int(0.75 * n):]
        return HoldResult(
            seq=self.seq.name, object=self.seq.obj, hand=self.hand,
            held=bool(drop < 0.05), drop_m=drop,
            settle_m=float(np.linalg.norm(tail[-1] - tail[0])),
            n_contact=self.n_hand_contacts(), pen_mm=pen0 * 1000,
            seconds=seconds)


# --------------------------------------------------------------------------
# feedforward: carry the retargeted grasp along the reference
# --------------------------------------------------------------------------
def _base_cols(sc) -> np.ndarray:
    """Indices (into sc.jids) of the six floating-base DoF."""
    m = sc.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in sc.jids]
    return np.array([i for i, n in enumerate(names) if n in BASE_DOF])


def feedforward(sc, q_obj: np.ndarray, ref_pos: np.ndarray, ref_quat: np.ndarray,
                iters: int = 30, reach: float = 1.5,
                w_cont: float = 0.5,
                max_step: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Joint trajectory that carries a retargeted grasp along the reference.

    `q_obj[k]` is the hand configuration relative to the object, as the
    retargeting solved it.  The hand's world pose at frame k is therefore the
    object's pose composed with that, and the only thing that has to change is
    the wrist: the FINGERS are held exactly at their retargeted values, so the
    grasp shape cannot drift while the wrist is being solved for.

    `w_cont` penalises movement away from the previous frame's solution. The
    base is three slides and three hinges, so a rigid motion that passes near
    the hinges' gimbal has several parameterisations of the SAME world pose,
    and the IK is free to jump between them: measured without this term, the
    world pose stayed exact (0.000 mm residual) while the hinge commands jumped
    0.70 rad between consecutive frames, against an object rotating at most
    0.231 rad per frame. A position servo driven through that jump applies an
    impulse that has nothing to do with the task.

    That jump turned out NOT to be a solver artifact, and the distinction is
    recorded here because it changes what has to be fixed. Constraining the
    solution to stay near the previous frame trades the error straight back:

        max_step   residual mean / max      largest base step
        none          0.000 /   0.000 mm         0.700
        0.60          1.901 /  92.945 mm         0.514
        0.35          7.383 / 171.381 mm         0.304
        0.20         23.605 / 293.240 mm         0.191

    There is no nearby parameterisation of the same world pose, so the jump is
    forced by the chart: three hinges are Euler angles, and this trajectory
    passes near their gimbal. The fix is to re-parameterise the floating base
    (a free joint, or commanding SE(3) and converting) BEFORE the tracking
    stage commands it -- not to bias the feedforward. `max_step` defaults to
    off so the feedforward stays exact and the problem stays visible.

    Solved as a six-DoF inverse-kinematics problem on the fingertip positions
    rather than by inverting the base joint chain, because the chain's Euler
    convention is a property of how the base was assembled and inverting it by
    assumption is exactly the kind of thing that returns a plausible wrong
    answer.  The residual is returned so that assumption is never needed.
    """
    widen_base(sc, reach)
    m, d = sc.model, sc.data
    cols = _base_cols(sc)
    dofs = np.array([m.jnt_dofadr[j] for j in sc.jids])[cols]
    tips = np.array(sc.tip_bids)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def fk(q):
        d.qpos[:] = 0.0
        d.qpos[sc.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    # fingertip positions in the OBJECT frame, at each retargeted pose
    tips_obj = np.empty((len(q_obj), len(tips), 3))
    for k in range(len(q_obj)):
        fk(q_obj[k])
        tips_obj[k] = d.xpos[tips]

    Q = q_obj.copy()
    err = np.zeros(len(q_obj))
    q = q_obj[0].copy()
    for k in range(len(q_obj)):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(ref_quat[k], float))
        R = R.reshape(3, 3)
        target = tips_obj[k] @ R.T + ref_pos[k]

        q = q.copy()
        q[np.setdiff1d(np.arange(len(q)), cols)] = \
            q_obj[k][np.setdiff1d(np.arange(len(q)), cols)]
        for it in range(iters):
            fk(q)
            rows, res = [], []
            for i, b in enumerate(tips):
                jp[:] = 0.0
                mujoco.mj_jacBody(m, d, jp, jr, b)
                rows.append(jp[:, dofs].copy())
                res.append(d.xpos[b] - target[i])
            # The continuity term picks the BRANCH, then gets out of the way.
            # Held on to convergence it biases the answer: measured, a constant
            # w_cont = 0.5 removed every jump but cost 30 mm of placement, and
            # exact placement is the entire purpose of the feedforward. Decayed
            # to zero over the first 60% of iterations it costs nothing.
            wc = w_cont * max(0.0, 1.0 - it / (0.6 * iters))
            if wc > 0 and k > 0:
                rows.append(wc * np.eye(len(cols)))
                res.append(wc * (q[cols] - Q[k - 1][cols]))
            J, r = np.vstack(rows), np.concatenate(res)
            H = J.T @ J + 1e-6 * np.eye(len(cols))
            try:
                dq = np.linalg.solve(H, -J.T @ r)
            except np.linalg.LinAlgError:
                break
            s = np.linalg.norm(dq)
            if s > 0.3:
                dq *= 0.3 / s
            cand = q[cols] + dq
            if max_step and k > 0:
                # Stay in the basin around the previous frame. The base's three
                # hinges are an Euler chart, so one world pose has several
                # parameterisations; without this the solver lands on a distant
                # one and the command jumps 0.70 rad while the object turns by
                # at most 0.231.
                off = cand - Q[k - 1][cols]
                n_off = np.linalg.norm(off)
                if n_off > max_step:
                    cand = Q[k - 1][cols] + off * (max_step / n_off)
            q[cols] = np.clip(cand,
                              m.jnt_range[np.array(sc.jids)[cols], 0],
                              m.jnt_range[np.array(sc.jids)[cols], 1])
            if s < 1e-6:
                break
        fk(q)
        Q[k] = q
        err[k] = float(np.linalg.norm(d.xpos[tips] - target, axis=1).mean())
    return Q, err


# --------------------------------------------------------------------------
# SE(3) wrist: converting a hinge-base fit into a mocap-driven state
# --------------------------------------------------------------------------
def joint_values(sc, q) -> dict:
    """Configuration as {joint name without the scene's prefix: value}."""
    m = sc.model
    out = {}
    for i, j in enumerate(sc.jids):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if n in BASE_DOF:
            continue
        out[n[5:] if n.startswith("hand_") else n] = float(q[i])
    return out


def palm_pose(sc, q) -> tuple[np.ndarray, np.ndarray]:
    """Palm position and orientation (wxyz) at configuration `q`."""
    m, d = sc.model, sc.data
    d.qpos[:] = 0.0
    d.qpos[sc.qadr] = q
    mujoco.mj_kinematics(m, d)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, d.xmat[sc.wrist_bid].copy())
    return d.xpos[sc.wrist_bid].copy(), quat


def _mul(pa, qa, pb, qb):
    """Compose two poses: (pa,qa) applied to (pb,qb)."""
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qa, qb)
    r = np.zeros(3)
    mujoco.mju_rotVecQuat(r, pb, qa)
    return pa + r, q


def _inv(p, q):
    qi = np.array([q[0], -q[1], -q[2], -q[3]])
    r = np.zeros(3)
    mujoco.mju_rotVecQuat(r, -p, qi)
    return r, qi


class MocapHand:
    """Place and drive a free-jointed hand by its palm pose.

    The free joint sits on `hand_base`, not on the palm -- MuJoCo allows six
    DoF per body and every one of these hands already spends some of the palm's
    on a wrist. So the base pose that puts the palm where it is wanted depends
    on the finger configuration, and is measured here rather than assumed.
    """

    def __init__(self, sc):
        if sc.base_kind != "mocap":
            raise ValueError("MocapHand needs a scene built with base='mocap'")
        self.sc = sc
        m = sc.model
        self.free_q = int(m.jnt_qposadr[
            [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
             and int(m.jnt_bodyid[j]) == sc.base_bid][0]])
        self.name_to_i = {}
        for i, j in enumerate(sc.jids):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            self.name_to_i[n[5:] if n.startswith("hand_") else n] = i

    def config_from(self, values: dict) -> np.ndarray:
        q = np.zeros(len(self.sc.jids))
        for n, v in values.items():
            if n in self.name_to_i:
                q[self.name_to_i[n]] = v
        return q

    def place(self, palm_p, palm_q, values: dict):
        """Set the hand so its palm is at (palm_p, palm_q) with these joints."""
        m, d = self.sc.model, self.sc.data
        q = self.config_from(values)
        d.qpos[self.sc.qadr] = q
        d.qpos[self.free_q:self.free_q + 3] = 0.0
        d.qpos[self.free_q + 3:self.free_q + 7] = [1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        p0 = d.xpos[self.sc.wrist_bid].copy()
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, d.xmat[self.sc.wrist_bid].copy())
        # base = palm_target * (palm_when_base_identity)^-1
        pi, qi = _inv(p0, q0)
        bp, bq = _mul(np.asarray(palm_p, float), np.asarray(palm_q, float), pi, qi)
        d.qpos[self.free_q:self.free_q + 3] = bp
        d.qpos[self.free_q + 3:self.free_q + 7] = bq
        d.mocap_pos[m.body_mocapid[self.sc.mocap_bid]] = bp
        d.mocap_quat[m.body_mocapid[self.sc.mocap_bid]] = bq
        mujoco.mj_forward(m, d)
        return bp, bq

    def command(self, palm_p, palm_q):
        """Move the mocap target only; the weld drags the hand to it."""
        m, d = self.sc.model, self.sc.data
        # Snapshot BEFORE touching qpos. Taking it afterwards restored the
        # hand's free joint to the origin on every control step, teleporting
        # the hand off the object; the rollout reported 253 m of error.
        saved = d.qpos.copy()
        d.qpos[self.free_q:self.free_q + 3] = 0.0
        d.qpos[self.free_q + 3:self.free_q + 7] = [1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        p0 = d.xpos[self.sc.wrist_bid].copy()
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, d.xmat[self.sc.wrist_bid].copy())
        d.qpos[:] = saved
        mujoco.mj_kinematics(m, d)
        pi, qi = _inv(p0, q0)
        bp, bq = _mul(np.asarray(palm_p, float), np.asarray(palm_q, float), pi, qi)
        d.mocap_pos[m.body_mocapid[self.sc.mocap_bid]] = bp
        d.mocap_quat[m.body_mocapid[self.sc.mocap_bid]] = bq
        return bp, bq


def feedforward_se3(sc_fit, q_obj, ref_pos, ref_quat):
    """Palm poses that carry a retargeted grasp along the reference.

    The object-frame grasp is composed with the reference pose directly:
    `T_palm_world[k] = T_obj[k] . T_palm_obj[k]`. No inverse kinematics, and
    therefore no chart to get stuck in -- the discontinuity the hinge base
    suffers (0.70 rad between frames while the object turns 0.231) cannot arise
    here, because nothing is being solved for.
    """
    P = np.empty((len(q_obj), 3))
    Q = np.empty((len(q_obj), 4))
    vals = []
    for k in range(len(q_obj)):
        pp, pq = palm_pose(sc_fit, q_obj[k])
        wp, wq = _mul(np.asarray(ref_pos[k], float),
                      np.asarray(ref_quat[k], float), pp, pq)
        P[k], Q[k] = wp, wq / np.linalg.norm(wq)
        vals.append(joint_values(sc_fit, q_obj[k]))
    # keep the quaternion on one branch so a controller differencing it sees
    # rotation, not a sign flip
    for k in range(1, len(Q)):
        if Q[k] @ Q[k - 1] < 0:
            Q[k] = -Q[k]
    return P, Q, vals


# --------------------------------------------------------------------------
# per-reference tracking: feedforward + a correction searched by MPPI
# --------------------------------------------------------------------------
def act_map(sc):
    """actuator -> indices into sc.jids that it drives.

    Shadow couples each finger's distal two joints onto one actuator through a
    tendon, so this is not one-to-one and an actuator-per-joint assignment
    drives the wrong targets.
    """
    m = sc.model
    pos = {j: i for i, j in enumerate(sc.jids)}
    out = []
    for a in range(m.nu):
        tt = m.actuator_trntype[a]
        if tt == mujoco.mjtTrn.mjTRN_JOINT:
            j = int(m.actuator_trnid[a, 0])
            out.append((a, [pos[j]] if j in pos else []))
        elif tt == mujoco.mjtTrn.mjTRN_TENDON:
            t = int(m.actuator_trnid[a, 0])
            lo = int(m.tendon_adr[t])
            idx = [int(m.wrap_objid[lo + w]) for w in range(int(m.tendon_num[t]))]
            out.append((a, [pos[j] for j in idx if j in pos]))
        else:
            out.append((a, []))
    return out


def palm_pose_cached(rt, k):
    """The palm's pose in the OBJECT frame at reference frame k."""
    if getattr(rt, "_palm_obj", None) is None:
        rt._palm_obj = [palm_pose(rt.fit, q) for q in rt.tr.q]
    return rt._palm_obj[int(np.clip(k, 0, len(rt._palm_obj) - 1))]


@dataclass
class Rollout:
    pos_err: np.ndarray       # (T,) object position error, metres
    rot_err: np.ndarray       # (T,) object orientation error, radians
    dropped: bool
    steps: int

    @property
    def score(self) -> float:
        """Mean position error, with a dropped object scored as a full miss."""
        return float(np.mean(self.pos_err)) if not self.dropped else 1.0


class ReferenceTracker:
    """One GRAB reference, a hand that starts on it, and a controller.

    The feedforward carries the retargeted grasp along the reference exactly
    (0.000 mm, by construction). What it cannot do is react: the grasp slips,
    the object rotates in the hand, and nothing corrects it. The correction is
    what the tracking stage searches for -- here with MPPI, which needs no
    gradients through contact and is the same sampling optimiser the synthetic
    tracker used.
    """

    def __init__(self, seq, hand="shadow", side="rhand", window=None,
                 obj_mass=0.2, ctrl_every=None, w_pen=None):
        from oppdef.human.scene import build as build_scene
        from oppdef.human.retarget import retarget_sequence

        if window is None:
            from experiments.grab_inventory import contact_mask, longest_run, _tree
            mk = contact_mask(seq, _tree(seq, {}), side)
            window = longest_run(mk)
        self.seq, self.hand, self.side, self.window = seq, hand, side, window

        self.fit = build_scene(hand, seq.obj, obj_static=True)
        kw = {} if w_pen is None else {"w_pen": w_pen}
        self.tr = retarget_sequence(seq, side, hand, window=window,
                                    sc=self.fit, **kw)
        pos, quat = reference_in_first_frame(seq, window[0])
        self.ref_pos, self.ref_quat = pos[self.tr.frames], quat[self.tr.frames]
        self.P, self.Q, self.vals = feedforward_se3(
            self.fit, self.tr.q, self.ref_pos, self.ref_quat)

        self.sim = build_scene(hand, seq.obj, base="mocap", obj_static=False)
        self.mh = MocapHand(self.sim)
        m = self.sim.model
        self.obj_q = int(m.jnt_qposadr[
            [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
             and int(m.jnt_bodyid[j]) == self.sim.obj_bid][0]])
        bid = self.sim.obj_bid
        if obj_mass:
            k = obj_mass / max(float(m.body_mass[bid]), 1e-9)
            m.body_mass[bid] *= k
            m.body_inertia[bid] *= k
        self.obj_v = int(self.sim.model.jnt_dofadr[
            [j for j in range(self.sim.model.njnt)
             if self.sim.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
             and int(self.sim.model.jnt_bodyid[j]) == self.sim.obj_bid][0]])
        self._acts = act_map(self.sim)
        mm = self.sim.model
        self._act_mat = np.zeros((mm.nu, len(self.sim.jids)))
        for a, idx in self._acts:
            for j in idx:
                self._act_mat[a, j] = 1.0
        self._ctrl_lo = mm.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = mm.actuator_ctrlrange[:, 1].copy()
        self._hold_env = None
        self.grasp_fit = None
        self._grip_offset = np.zeros(self.sim.model.nu)
        self._ctrl_closure = self._raw_ctrl(self.sim.q_closure)
        # Which actuators drive FINGERS. A squeeze must not touch the wrist:
        # Shadow's WRJ1 sits on the palm body itself, and driving it "toward
        # closure" swings the whole hand off the object -- closing to 1.21 rad
        # then produced 0.10 N because the fingers were being carried away
        # faster than they were closing. Selected by body descent, not by name.
        mm = self.sim.model
        below = np.zeros(len(self.sim.jids), bool)
        for i, j in enumerate(self.sim.jids):
            b = int(mm.body_parentid[int(mm.jnt_bodyid[j])])
            while b > 0:
                if b == self.sim.wrist_bid:
                    below[i] = True
                    break
                b = int(mm.body_parentid[b])
        self._finger_act = np.array(
            [bool(idx) and bool(np.all(below[idx])) for _a, idx in self._acts])
        # One control frame must last one REFERENCE frame. Hardcoding it
        # instead played a 15 Hz clip at 50 Hz -- the hand ran 3.3x ahead of
        # the object and threw it 13 m.
        self.ctrl_every = (int(round(seq.dt / m.opt.timestep))
                           if ctrl_every is None else int(ctrl_every))
        if self.ctrl_every < 1:
            raise ValueError(f"reference dt {seq.dt} is below one timestep")
        self.T = len(self.P)
        #: 3 palm translation + 3 palm rotation (rotvec) + one per actuator
        self.n_action = 6 + len(self._acts)

    # -- state ------------------------------------------------------------
    def _raw_ctrl(self, q) -> np.ndarray:
        m = self.sim.model
        c = np.zeros(m.nu)
        for a, idx in self._acts:
            if idx:
                c[a] = np.clip(np.sum(np.asarray(q)[idx]), *m.actuator_ctrlrange[a])
        return c

    def ctrl_for(self, values: dict) -> np.ndarray:
        """Servo targets for a joint configuration.

        Vectorised: the actuator-to-joint map is a fixed 0/1 matrix, so the
        whole thing is one matmul and one clip. The per-actuator Python loop it
        replaces called np.clip 9,400 times in 25 control steps and was 11% of
        PPO's wall clock.
        """
        m = self.sim.model
        q = self.mh.config_from(values)
        return np.clip(self._act_mat @ q, self._ctrl_lo, self._ctrl_hi)

    def reset_at(self, k: int, settle: float = 0.0, grip: float | None = None):
        """Start from reference frame `k` rather than the window's first frame.

        The window begins where the human's hand FIRST comes within 5 mm of the
        object, which is the moment contact starts, not the moment the grasp is
        formed. Started there the hand is still closing and drops the object
        within a frame, and that reads as "tracking failed" when nothing has
        been tracked yet: on `mug_drink_1`, window frames 10 and 30 drop while
        50, 70 and 90 hold, in BOTH the hinge and the mocap scenes.
        """
        m, d = self.sim.model, self.sim.data
        mujoco.mj_resetData(m, d)
        self._grip_offset = np.zeros(m.nu)
        k = int(np.clip(k, 0, self.T - 1))
        self.mh.place(self.P[k], self.Q[k], self.vals[k])
        d.qpos[self.obj_q:self.obj_q + 3] = self.ref_pos[k]
        d.qpos[self.obj_q + 3:self.obj_q + 7] = self.ref_quat[k]
        d.ctrl[:] = self.ctrl_for(self.vals[k])
        mujoco.mj_forward(m, d)
        if settle > 0:
            g = m.opt.gravity.copy()
            m.opt.gravity[:] = 0.0
            for _ in range(int(settle / m.opt.timestep)):
                mujoco.mj_step(m, d)
            m.opt.gravity[:] = g
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
        if grip:
            self.establish(grip)
        return d

    def track_score(self, start=0, steps=None):
        """Search objective for the one-handed grasp. **Unitless.**

        Clipped mean tracking error plus a drop penalty, over the whole
        reference by default. `grasp_frames` -- which counts frames that hold
        under gravity -- is not this: `gamecontroller_play_1` has 15 graspable
        frames and a feedforward that still ends 248 m away. Holding is
        necessary and not sufficient, and a grasp search scored on holding
        selects for the wrong thing.
        """
        n = (self.T - start) if steps is None else steps
        self.reset_at(start)
        r = self.rollout(start=start, steps=min(n, self.T - start))
        return float(np.minimum(r.pos_err, 0.25).mean()
                     + 0.5 * np.mean(r.pos_err > 0.10))

    def synthesize_grasp(self, k=None, samples=12, rounds=2, seconds=0.4,
                         grip=8.0, seed=0, objective="track"):
        """Search near the retargeted wrist for a pose that actually holds, and
        apply that correction to the WHOLE trajectory.

        G5 measured the raw retarget holding the object in 0.318 of frames, and
        its failures making 0.1 contacts at reset against 17.4 for successes --
        they are non-grasps. Its pre-registered decision rule says to initialise
        from a synthesised grasp near the human's contact set rather than from
        the retarget. Measured on a 24-frame subsample, that takes the hold rate
        from **0.167 to 0.792**.

        The correction is a constant wrist offset in the OBJECT frame, so the
        trajectory's shape is untouched and only the grasp moves. Applying it
        per-frame instead would be a different trajectory, not a repaired one.
        """
        from oppdef.human import grasp as G
        from oppdef.human.scene import build as build_scene

        if self._hold_env is None:
            self._hold_env = GrabTrackEnv(
                self.seq, self.hand,
                sc=build_scene(self.hand, self.seq.obj, obj_static=False))
        if k is None:
            k = int(np.argmax(self.tr.n_contact))
        # Keep the correction only if it improves the WHOLE trajectory. The
        # offset is optimised at one frame, and on `cup_pass_1` the frame's
        # optimum took the reference from 6 graspable frames to 1. Guarded, this
        # can only help.
        if objective == "track":
            # Search the wrist offset directly against tracking, the way the
            # two-handed stage does. Scored on holding, the search cannot see
            # that a grasp survives gravity and not the motion.
            rng = np.random.default_rng(seed)
            sig = np.array([0.012] * 3 + [0.08] * 3)
            best = self.track_score()
            base = np.zeros(6)
            for _rnd in range(rounds):
                for dlt in rng.normal(size=(samples, 6)) * sig:
                    self.apply_wrist_offset(dlt)
                    v = self.track_score()
                    if v < best:
                        best, base = v, base + dlt
                    else:
                        self.apply_wrist_offset(-dlt)
            self.grasp_fit = None
            self.grasp_score = best
            self.grasp_frames_before = self.grasp_frames_after = len(
                self.grasp_frames())
            return base

        before = len(self.grasp_frames())
        fit = G.synthesize(self._hold_env, self.tr.q[k], samples=samples,
                           rounds=rounds, seconds=seconds, grip=grip, seed=seed)
        self.apply_wrist_offset(fit.offset)
        after = len(self.grasp_frames())
        if after < before:
            self.apply_wrist_offset(-np.asarray(fit.offset))
            fit.held = False
        self.grasp_fit = fit
        self.grasp_frames_before, self.grasp_frames_after = before, max(after, before)
        return fit

    def apply_wrist_offset(self, delta):
        """Shift every frame's wrist by `delta` (3 translation + 3 rotation)."""
        names = [mujoco.mj_id2name(self.fit.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in self.fit.jids]
        for i, n in enumerate(names):
            if n in ("x", "y", "z"):
                self.tr.q[:, i] += delta[{"x": 0, "y": 1, "z": 2}[n]]
            elif n in ("rx", "ry", "rz"):
                self.tr.q[:, i] += delta[3 + {"rx": 0, "ry": 1, "rz": 2}[n]]
        self._palm_obj = None
        self.P, self.Q, self.vals = feedforward_se3(
            self.fit, self.tr.q, self.ref_pos, self.ref_quat)

    def grasp_frames(self, seconds: float = 0.4, stride: int = 5) -> np.ndarray:
        """Which reference frames hold the object on their own.

        Used to choose where a tracking episode may start, and as the honest
        denominator for what fraction of a reference is even graspable.
        """
        ok = []
        for k in range(0, self.T, stride):
            self.reset_at(k)
            p0 = self.obj_pose()[0].copy()
            for _ in range(int(seconds / self.sim.model.opt.timestep)):
                mujoco.mj_step(self.sim.model, self.sim.data)
            if np.linalg.norm(self.obj_pose()[0] - p0) < 0.03:
                ok.append(k)
        return np.array(ok)

    def reset(self, settle: float = 0.2, grip: float | None = None):
        """Place the hand and object, then let the contact set resolve.

        The retarget leaves the fingertips a centimetre inside the object -- it
        is a kinematic fit and tips are deliberately exempt from its penetration
        penalty, since a fingertip on the surface IS the grasp. Started from
        there, the constraint solver resolves all of it at once and, because the
        welded wrist cannot yield, the object absorbs the whole impulse: it left
        at 13 m/s and the rollout reported 253 m of tracking error.

        So the scene is settled first with gravity off and the reference held,
        which lets the penetration relax without the object also falling. The
        object's velocity is then zeroed and gravity restored.
        """
        m, d = self.sim.model, self.sim.data
        mujoco.mj_resetData(m, d)
        self._grip_offset = np.zeros(m.nu)
        self.mh.place(self.P[0], self.Q[0], self.vals[0])
        d.qpos[self.obj_q:self.obj_q + 3] = self.ref_pos[0]
        d.qpos[self.obj_q + 3:self.obj_q + 7] = self.ref_quat[0]
        d.ctrl[:] = self.ctrl_for(self.vals[0])
        mujoco.mj_forward(m, d)

        if settle > 0:
            g = m.opt.gravity.copy()
            m.opt.gravity[:] = 0.0
            for _ in range(int(settle / m.opt.timestep)):
                mujoco.mj_step(m, d)
            m.opt.gravity[:] = g
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
        if grip:
            self.establish(grip)
        return d

    def establish(self, target_n: float = 8.0, open_by: float = 0.30,
                  step: float = 0.015, settle_steps: int = 10,
                  max_close: float = 1.2):
        """Pre-grasp, then close onto the object until the grip carries it.

        A kinematic retarget cannot BE a grasp, and both failure modes were
        measured. Fingertips exactly on the surface give a position servo
        nothing to push against, so it applies no force and the object
        free-falls from the first frame. Fingertips buried to get force instead
        put 4469 N on a 0.2 kg mug -- and merely letting that relax ejects the
        object, which moved 28 mm and lost every contact during a 0.2 s settle.

        So the grip is built the way grasp synthesis in this repository already
        builds one: open to a pre-grasp, then close until the contact force
        reaches a target. The closing DIRECTION is derived, never tabled -- it
        is the sign of each actuator's own derived closure. Blending toward the
        closure POSTURE instead moved the fingertips from 134 mm to 209 mm away
        from the object, because a hand wrapped round a mug is already more
        flexed than its generic closure.
        """
        m, d = self.sim.model, self.sim.data
        g = m.opt.gravity.copy()
        m.opt.gravity[:] = 0.0

        direction = np.sign(self._ctrl_closure) * self._finger_act
        base = d.ctrl.copy()

        # pre-grasp: back the fingers off so nothing starts interpenetrating
        d.ctrl[:] = np.clip(base - open_by * direction,
                            m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(60):
            mujoco.mj_step(m, d)
        d.qpos[self.obj_q:self.obj_q + 3] = self.ref_pos[0]
        d.qpos[self.obj_q + 3:self.obj_q + 7] = self.ref_quat[0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        pre = d.ctrl.copy()

        a, reached = 0.0, 0.0
        while a < max_close:
            a += step
            d.ctrl[:] = np.clip(pre + a * direction,
                                m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            for _ in range(settle_steps):
                mujoco.mj_step(m, d)
                # Pin the object while the grip forms. Left free it is simply
                # squeezed out of the hand -- closing to the limit then read
                # 0 N because the object had already been extruded.
                d.qpos[self.obj_q:self.obj_q + 3] = self.ref_pos[0]
                d.qpos[self.obj_q + 3:self.obj_q + 7] = self.ref_quat[0]
                d.qvel[self.obj_v:self.obj_v + 6] = 0.0
                mujoco.mj_forward(m, d)
            reached, _n = total_grip(self.sim)
            if reached >= target_n:
                break

        m.opt.gravity[:] = g
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        # The offset the grip is MADE of. apply() rewrites d.ctrl from the
        # feedforward every frame, which silently discarded the squeeze that
        # had just been established and dropped the object on frame 0.
        self._grip_offset = d.ctrl - self.ctrl_for(self.vals[0])
        self.grip_close = a
        self.grip_force = reached
        return a, reached

    #: Set to a callable returning (pos, quat) to feed the controller an
    #: ESTIMATE of the object's pose instead of the simulator's ground truth.
    #: The controller is otherwise untouched, so any change in the result is
    #: attributable to perception rather than to a different controller.
    pose_source = None

    def true_obj_pose(self):
        d = self.sim.data
        return (d.qpos[self.obj_q:self.obj_q + 3].copy(),
                d.qpos[self.obj_q + 3:self.obj_q + 7].copy())

    def obj_pose(self):
        if self.pose_source is not None:
            return self.pose_source()
        return self.true_obj_pose()

    def observe(self, k) -> np.ndarray:
        """What a distilled tracking policy sees at reference frame k.

        Everything is expressed relative to the OBJECT, not to the world: a
        policy trained on world coordinates learns where GRAB's subjects happen
        to stand. The lookahead is what makes this a tracking observation
        rather than a regulation one -- a controller that sees only the current
        error is always late.
        """
        m, d = self.sim.model, self.sim.data
        k = int(np.clip(k, 0, self.T - 1))
        op, oq = self.obj_pose()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, oq)
        R = R.reshape(3, 3)

        qh = d.qpos[self.sim.qadr].copy()
        vh = np.array([d.qvel[m.jnt_dofadr[j]] for j in self.sim.jids])
        palm_p = R.T @ (d.xpos[self.sim.wrist_bid] - op)
        pq = np.zeros(4)
        mujoco.mju_mat2Quat(pq, d.xmat[self.sim.wrist_bid].copy())
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, np.array([oq[0], -oq[1], -oq[2], -oq[3]]), pq)

        err_p = R.T @ (self.ref_pos[k] - op)
        err_q = np.zeros(4)
        mujoco.mju_mulQuat(err_q, np.array([oq[0], -oq[1], -oq[2], -oq[3]]),
                           self.ref_quat[k])
        ov = d.qvel[self.obj_v:self.obj_v + 6].copy()

        ahead = []
        for h in (1, 5, 10):
            j = int(np.clip(k + h, 0, self.T - 1))
            aq = np.zeros(4)
            mujoco.mju_mulQuat(aq, np.array([oq[0], -oq[1], -oq[2], -oq[3]]),
                               self.ref_quat[j])
            ahead += list(R.T @ (self.ref_pos[j] - op)) + list(aq)

        return np.concatenate([qh, vh, palm_p, rel, err_p, err_q, ov,
                               ahead]).astype(np.float64)

    @property
    def n_obs(self) -> int:
        return len(self.sim.jids) * 2 + 3 + 4 + 3 + 4 + 6 + 3 * 7

    def error(self, k):
        # scored on the TRUTH even when the controller is fed an estimate:
        # a tracker that believes a wrong pose must not also be graded by it
        p, q = self.true_obj_pose()
        k = int(np.clip(k, 0, self.T - 1))
        dq = np.zeros(4)
        c = self.ref_quat[k] * np.array([1.0, -1, -1, -1])
        mujoco.mju_mulQuat(dq, q, c)
        ang = 2.0 * np.arccos(np.clip(abs(dq[0]), -1.0, 1.0))
        return float(np.linalg.norm(p - self.ref_pos[k])), float(ang)

    # -- control ----------------------------------------------------------
    def apply(self, k, action=None, squeeze=None, alpha=None):
        """Command frame k's feedforward, optionally corrected by `action`.

        `squeeze` drives the finger targets a fraction of the way from the
        retargeted posture toward the hand's DERIVED closure. A position servo
        commanded exactly to the surface applies no force -- it yields as soon
        as the object pushes back -- so a kinematic fit has no grip at all.
        This is the ingredient the retarget cannot supply.
        """
        m, d = self.sim.model, self.sim.data
        k = int(np.clip(k, 0, self.T - 1))
        if alpha is None or alpha >= 1.0:
            p, q = self.P[k].copy(), self.Q[k].copy()
        else:
            # Place the grasp on the object where it ACTUALLY is, blended
            # toward where the reference says it should be. alpha = 1 is the
            # pure feedforward, which commands the hand to the reference and
            # therefore walks away from an object that is lagging: measured,
            # the palm-object gap grew 133 -> 167 mm while the object fell 79 mm
            # behind, and contacts went 25 -> 0. alpha = 0 rides the object
            # perfectly but never corrects it.
            op, oq = self.obj_pose()
            bp = (1.0 - alpha) * op + alpha * self.ref_pos[k]
            bq = (1.0 - alpha) * oq + alpha * self.ref_quat[k] * np.sign(
                oq @ self.ref_quat[k] or 1.0)
            bq = bq / max(np.linalg.norm(bq), 1e-12)
            p, q = _mul(bp, bq, *palm_pose_cached(self, k))
        c = self.ctrl_for(self.vals[k]) + self._grip_offset
        if squeeze:
            c = (1.0 - squeeze) * c + squeeze * self._ctrl_closure
        if action is not None:
            a = np.asarray(action, float)
            p = p + a[:3]
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, a[3:6] / (np.linalg.norm(a[3:6]) + 1e-12),
                                      float(np.linalg.norm(a[3:6])))
            nq = np.zeros(4)
            mujoco.mju_mulQuat(nq, dq, q)
            q = nq
            c = c + a[6:]
        for i in range(m.nu):
            c[i] = np.clip(c[i], *m.actuator_ctrlrange[i])
        self.mh.command(p, q)
        d.ctrl[:] = c

    def rollout(self, actions=None, start=0, steps=None, squeeze=None,
                alpha=None) -> Rollout:
        """Run from frame `start` for `steps` control frames."""
        m, d = self.sim.model, self.sim.data
        steps = self.T - start if steps is None else steps
        pe, re = np.empty(steps), np.empty(steps)
        for i in range(steps):
            k = start + i
            self.apply(k, None if actions is None else actions[i],
                       squeeze=squeeze, alpha=alpha)
            for _ in range(self.ctrl_every):
                mujoco.mj_step(m, d)
            pe[i], re[i] = self.error(k)
        dropped = bool(pe[-1] > 0.10)
        return Rollout(pos_err=pe, rot_err=re, dropped=dropped, steps=steps)

    def save(self):
        d = self.sim.data
        return (d.qpos.copy(), d.qvel.copy(), d.mocap_pos.copy(),
                d.mocap_quat.copy(), d.ctrl.copy())

    def restore(self, st):
        d = self.sim.data
        d.qpos[:], d.qvel[:], d.mocap_pos[:], d.mocap_quat[:], d.ctrl[:] = st
        mujoco.mj_forward(self.sim.model, d)



def closing_direction(sc, acts):
    """Per-actuator sign that CLOSES the fingers, derived, excluding the wrist.

    Two things this is not. It is not a blend toward the derived closure
    POSTURE: a hand wrapped round a mug is already more flexed than its generic
    closure, so blending opens it (fingertips went 134 mm -> 209 mm away). And
    it must not touch the wrist -- Shadow's WRJ1 sits on the palm body itself,
    and driving it "toward closure" swings the whole hand off the object.
    """
    m = sc.model
    raw = np.zeros(m.nu)
    for a, idx in acts:
        if idx:
            raw[a] = float(np.sum(np.asarray(sc.q_closure)[idx]))
    below = np.zeros(len(sc.jids), bool)
    for i, j in enumerate(sc.jids):
        b = int(m.body_parentid[int(m.jnt_bodyid[j])])
        while b > 0:
            if b == sc.wrist_bid:
                below[i] = True
                break
            b = int(m.body_parentid[b])
    finger = np.array([bool(idx) and bool(np.all(below[idx])) for _a, idx in acts])
    return np.sign(raw) * finger


def establish_grip(sc, direction, obj_qadr, obj_vadr, obj_pose,
                   target_n=8.0, open_by=0.30, step=0.015, settle_steps=10,
                   max_close=1.2, pre_steps=60):
    """Pre-grasp, then close until the contact force reaches `target_n`.

    G5 measured that a retargeted pose held the object in only 25.5% of frames,
    and that the failures make 0.1 contacts at reset against 17.4 for the
    successes -- they are non-grasps, not slipping grasps. The retarget solves
    fingertip POSITIONS against a static object; nothing in it asks the result
    to close on anything. So the grip is built rather than fitted, and G5's
    pre-registered decision rule makes this part of the stage rather than an
    option.

    The object is pinned to `obj_pose` while the grip forms: left free it is
    simply extruded, and closing to the limit then reads 0 N.
    """
    m, d = sc.model, sc.data
    g = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    base = d.ctrl.copy()

    d.ctrl[:] = np.clip(base - open_by * direction, lo, hi)
    for _ in range(pre_steps):
        mujoco.mj_step(m, d)
    d.qpos[obj_qadr:obj_qadr + 3] = obj_pose[0]
    d.qpos[obj_qadr + 3:obj_qadr + 7] = obj_pose[1]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    pre = d.ctrl.copy()

    a, reached = 0.0, 0.0
    while a < max_close:
        a += step
        d.ctrl[:] = np.clip(pre + a * direction, lo, hi)
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)
            d.qpos[obj_qadr:obj_qadr + 3] = obj_pose[0]
            d.qpos[obj_qadr + 3:obj_qadr + 7] = obj_pose[1]
            d.qvel[obj_vadr:obj_vadr + 6] = 0.0
            mujoco.mj_forward(m, d)
        reached, _n = total_grip(sc)
        if reached >= target_n:
            break
    m.opt.gravity[:] = g
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return a, reached, d.ctrl.copy()


def mppi_track(rt, horizon=5, samples=48, sigma_pos=0.004, sigma_rot=0.03,
               sigma_fin=0.05, rho=1.0, w_rot=0.10, w_drop=2.0,
               seed=0, settle=0.0, start=None, warm=None, progress=None):
    """Search a correction to the feedforward, one control frame at a time.

    The feedforward carries the retargeted grasp exactly in KINEMATICS and
    never squeezes, so the first time the reference accelerates the object is
    thrown: measured on `mug_drink_1`, contacts collapse from 8 to 0 at frame
    27 and the clip ends 175 m away. MPPI searches the correction that the
    kinematic fit cannot provide, without needing gradients through contact.

    The temperature is adaptive (lambda = rho * std(cost)); a fixed one is
    meaningless here because the cost scale changes by three orders of
    magnitude between a held object and a dropped one.
    """
    rng = np.random.default_rng(seed)
    n_fin = rt.n_action - 6
    scale = np.concatenate([np.full(3, sigma_pos), np.full(3, sigma_rot),
                            np.full(n_fin, sigma_fin)])

    if start is None:
        rt.reset(settle=settle)
        start = 0
    else:
        rt.reset_at(start, settle=settle)
    # A warm start is what makes a homotopy curriculum worth running: the plan
    # solved for an easier deformation of this same reference is the opening
    # guess for the harder one.
    plan = np.zeros((horizon, rt.n_action))
    chosen = np.zeros((rt.T, rt.n_action))
    if warm is not None:
        chosen[:len(warm)] = np.asarray(warm)[:len(chosen)]
    pos_err = np.full(rt.T, np.nan)
    rot_err = np.full(rt.T, np.nan)

    for k in range(start, rt.T):
        state = rt.save()
        if warm is not None and k == start:
            plan = chosen[k:k + horizon].copy()
            if len(plan) < horizon:
                plan = np.vstack([plan, np.zeros((horizon - len(plan),
                                                  rt.n_action))])
        noise = rng.normal(size=(samples, horizon, rt.n_action)) * scale
        cand = plan[None] + noise
        cand[0] = plan                      # keep the incumbent
        cand[1] = 0.0                       # and pure feedforward
        cost = np.empty(samples)
        for i in range(samples):
            rt.restore(state)
            c = 0.0
            for h in range(horizon):
                rt.apply(k + h, cand[i, h])
                for _ in range(rt.ctrl_every):
                    mujoco.mj_step(rt.sim.model, rt.sim.data)
                pe, re = rt.error(k + h)
                c += pe + w_rot * re + (w_drop if pe > 0.10 else 0.0)
            cost[i] = c

        lam = max(rho * float(np.std(cost)), 1e-6)
        w = np.exp(-(cost - cost.min()) / lam)
        w /= w.sum()
        plan = np.einsum("i,iha->ha", w, cand)

        rt.restore(state)
        rt.apply(k, plan[0])
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(rt.sim.model, rt.sim.data)
        chosen[k] = plan[0]
        pos_err[k], rot_err[k] = rt.error(k)
        plan = np.vstack([plan[1:], np.zeros(rt.n_action)])
        if progress and k % progress == 0:
            print(f"    k={k:4d}/{rt.T}  pos {pos_err[k]*1000:7.1f} mm  "
                  f"rot {np.degrees(rot_err[k]):6.1f} deg", flush=True)

    sel = ~np.isnan(pos_err)
    return Rollout(pos_err=pos_err[sel], rot_err=rot_err[sel],
                   dropped=bool(pos_err[rt.T - 1] > 0.10),
                   steps=int(sel.sum())), chosen


def total_grip(sc) -> tuple[float, int]:
    """(sum of hand-object normal force in N, number of such contacts)."""
    m, d = sc.model, sc.data
    obj, hand = set(sc.obj_gids), set(sc.hand_gids)
    f = np.zeros(6)
    tot, n = 0.0, 0
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if ({g1, g2} & obj) and ({g1, g2} & hand):
            mujoco.mj_contactForce(m, d, i, f)
            tot += abs(float(f[0]))       # normal component, contact frame
            n += 1
    return tot, n


# --------------------------------------------------------------------------
# stage 6: two hands on one object
# --------------------------------------------------------------------------
class BimanualTracker:
    """Both hands tracking one GRAB reference.

    Built on the same pieces as the one-handed tracker rather than beside them:
    each side gets its own fitting scene and retarget, and the two are driven in
    one shared physics scene. That is the only place a bimanual problem actually
    differs -- the hands interact only through the object, and only if they are
    in the same simulation.

    136 of GRAB's 291 sequences have both hands on the object at once and 63
    hold that for 15 frames or more, so the window used here is the recorded
    TWO-handed one, not the union of each hand's own.
    """

    def __init__(self, seq, hand_r="shadow", hand_l="shadow_left",
                 window=None, obj_mass=0.2, joint_fit=True, rounds=2):
        from oppdef.human.scene import build as build_scene, build_bimanual
        from oppdef.human.retarget import retarget_sequence, retarget_bimanual

        self.seq = seq
        if window is None:
            from experiments.grab_inventory import contact_mask, longest_run, _tree
            tr_ = _tree(seq, {})
            both = (contact_mask(seq, tr_, "rhand") & contact_mask(seq, tr_, "lhand"))
            window = longest_run(both)
        self.window = window
        if window[1] < 2:
            raise ValueError(f"{seq.name}: no two-handed window")

        # Fit both hands in ONE scene by default, so each sees the other.
        # Fitted separately they interpenetrate by 11.7 mm and the left hand
        # pushes the right off the object; see retarget_bimanual.
        self.joint_fit = joint_fit
        self.fit, self.tr = {}, {}
        if joint_fit:
            self.tr, self.fit_scene, views = retarget_bimanual(
                seq, window, hand_r, hand_l, rounds=rounds)
            self.fit = views
        else:
            for sd, side, hk in (("r", "rhand", hand_r), ("l", "lhand", hand_l)):
                self.fit[sd] = build_scene(hk, seq.obj, obj_static=True)
                self.tr[sd] = retarget_sequence(seq, side, hk, window=window,
                                                sc=self.fit[sd])

        pos, quat = reference_in_first_frame(seq, window[0])
        fr = self.tr["r"].frames
        self.frames = fr
        self.ref_pos, self.ref_quat = pos[fr], quat[fr]
        self.T = len(fr)

        self.P, self.Q, self.vals = {}, {}, {}
        for sd in ("r", "l"):
            self.P[sd], self.Q[sd], self.vals[sd] = feedforward_se3(
                self.fit[sd], self.tr[sd].q, self.ref_pos, self.ref_quat)

        self.sim = build_bimanual(hand_r, hand_l, seq.obj, obj_static=False)
        m = self.sim.model
        self.obj_q = int(m.jnt_qposadr[
            [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
             and int(m.jnt_bodyid[j]) == self.sim.obj_bid][0]])
        self.obj_v = int(m.jnt_dofadr[
            [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
             and int(m.jnt_bodyid[j]) == self.sim.obj_bid][0]])
        if obj_mass:
            k = obj_mass / max(float(m.body_mass[self.sim.obj_bid]), 1e-9)
            m.body_mass[self.sim.obj_bid] *= k
            m.body_inertia[self.sim.obj_bid] *= k

        # map each fitting scene's joint order onto this scene's, by NAME.
        # The bimanual scene prefixes per side ("r_rh_FFJ3"); a lone-hand scene
        # does not. Getting this wrong posed both hands at zero and looked like
        # a physics failure.
        self.jmap = {}
        for sd in ("r", "l"):
            nf = [mujoco.mj_id2name(self.fit[sd].model, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in self.fit[sd].jids]
            ns = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in self.sim.jids[sd]]
            idx = {n: i for i, n in enumerate(nf)}
            pre = f"{sd}_"
            if joint_fit:
                # both scenes carry the same per-side prefix; only the fitting
                # scene additionally has the six base DoF, which the mocap
                # scene expresses as a free joint instead
                self.jmap[sd] = [idx.get(n, -1) for n in ns]
            else:
                self.jmap[sd] = [idx.get(n[len(pre):] if n.startswith(pre) else n, -1)
                                 for n in ns]
            if any(i < 0 for i in self.jmap[sd]):
                raise ValueError(f"side {sd}: unmapped joints {ns[:3]} vs {nf[:3]}")
        self.ctrl_every = int(round(seq.dt / m.opt.timestep))

        # Actuator maps per side. Without these d.ctrl stays at zero and the
        # position servos drive every finger to its OPEN configuration the
        # instant stepping starts -- the hands held the object at reset only
        # because `place` writes qpos directly, and then let go. Nothing in the
        # rollout was commanding the fingers at all.
        self._acts = {}
        for sd in ("r", "l"):
            pos = {j: i for i, j in enumerate(self.sim.jids[sd])}
            am = []
            for a in range(m.nu):
                tgt = int(m.actuator_trnid[a, 0])
                tt = m.actuator_trntype[a]
                if tt == mujoco.mjtTrn.mjTRN_JOINT:
                    am.append((a, [pos[tgt]] if tgt in pos else []))
                elif tt == mujoco.mjtTrn.mjTRN_TENDON:
                    lo = int(m.tendon_adr[tgt])
                    idx = [int(m.wrap_objid[lo + w])
                           for w in range(int(m.tendon_num[tgt]))]
                    am.append((a, [pos[j] for j in idx if j in pos]))
                else:
                    am.append((a, []))
            self._acts[sd] = am
        self._dirs = None
        self._grip_offset = np.zeros(m.nu)

    def ctrl_for(self, sd, k):
        """Servo targets that hold side `sd`'s retargeted configuration at k."""
        m = self.sim.model
        q = self._q_for(sd, k)
        c = np.zeros(m.nu)
        for a, idx in self._acts[sd]:
            if idx:
                c[a] = np.clip(np.sum(q[idx]), *m.actuator_ctrlrange[a])
        return c

    def set_ctrl(self, k):
        """Command both hands' fingers for reference frame k.

        Assigned by actuator OWNERSHIP, not by picking whichever side produced
        a non-zero number: a servo target of exactly zero is a legitimate
        target, and merging the two sides with a non-zero test silently drops
        it.
        """
        m, d = self.sim.model, self.sim.data
        c = np.zeros(m.nu)
        for sd in ("r", "l"):
            q = self._q_for(sd, k)
            for a, idx in self._acts[sd]:
                if idx:
                    c[a] = np.clip(np.sum(q[idx]), *m.actuator_ctrlrange[a])
        d.ctrl[:] = c + self._grip_offset
        return c

    def _q_for(self, sd, k):
        qf = self.tr[sd].q[int(np.clip(k, 0, self.T - 1))]
        return np.array([qf[i] for i in self.jmap[sd]])

    def place(self, sd, k):
        m, d = self.sim.model, self.sim.data
        fq = self.sim.free_q[sd]
        d.qpos[self.sim.qadr[sd]] = self._q_for(sd, k)
        d.qpos[fq:fq + 3] = 0.0
        d.qpos[fq + 3:fq + 7] = [1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        p0 = d.xpos[self.sim.palm_bid[sd]].copy()
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, d.xmat[self.sim.palm_bid[sd]].copy())
        pi, qi = _inv(p0, q0)
        bp, bq = _mul(self.P[sd][k], self.Q[sd][k], pi, qi)
        d.qpos[fq:fq + 3] = bp
        d.qpos[fq + 3:fq + 7] = bq
        d.mocap_pos[m.body_mocapid[self.sim.mocap_bid[sd]]] = bp
        d.mocap_quat[m.body_mocapid[self.sim.mocap_bid[sd]]] = bq

    def command(self, sd, k, delta=None):
        """Move side `sd`'s mocap target only; the weld drags the hand to it.

        Distinct from `place`, which writes qpos directly. Writing qpos every
        frame would teleport the hands and silently turn a dynamics rollout into
        kinematic playback -- the object would then be "tracked" by a hand that
        never applied a force to it.
        """
        m, d = self.sim.model, self.sim.data
        fq = self.sim.free_q[sd]
        saved = d.qpos.copy()
        d.qpos[fq:fq + 3] = 0.0
        d.qpos[fq + 3:fq + 7] = [1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        p0 = d.xpos[self.sim.palm_bid[sd]].copy()
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, d.xmat[self.sim.palm_bid[sd]].copy())
        d.qpos[:] = saved
        mujoco.mj_kinematics(m, d)
        pi, qi = _inv(p0, q0)
        k = int(np.clip(k, 0, self.T - 1))
        bp, bq = _mul(self.P[sd][k], self.Q[sd][k], pi, qi)
        d.mocap_pos[m.body_mocapid[self.sim.mocap_bid[sd]]] = bp
        d.mocap_quat[m.body_mocapid[self.sim.mocap_bid[sd]]] = bq

    def inter_hand(self):
        """(worst penetration between the two hands, number of such contacts)."""
        d = self.sim.data
        gr, gl = set(self.sim.hand_gids["r"]), set(self.sim.hand_gids["l"])
        worst, n, nrm = 0.0, 0, np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in gr and g2 in gl) or (g2 in gr and g1 in gl):
                n += 1
                dep = -float(c.dist)
                if dep > worst:
                    worst = dep
                sgn = 1.0 if g1 in gr else -1.0
                nrm += np.array(c.frame[:3]) * sgn
        return worst, n, nrm

    def separate(self, offset, iters=8, margin=0.0015):
        """Push the two hands apart until they stop interpenetrating.

        The two retargets are solved in separate single-hand scenes, so neither
        sees the other hand. On `gamecontroller_play_1` that left 42 hand-hand
        contacts and 11.7 mm of interpenetration, and the left hand pushed the
        right clean off the object -- 0 object contacts for a hand the human had
        on it. Separating along the contact normal is a correction, not a fix:
        the proper answer is to fit both hands against each other, which needs
        the wrist DoF the bimanual scene does not expose to the solver.
        `offset` accumulates so the shift persists through the rollout.

        Off by default, because it is not a net win: on that sequence it does
        remove the interpenetration (42 contacts at 11.7 mm -> none) and does
        give the right hand its object contacts back (0 -> 34), and tracking
        gets WORSE, 105 mm to 255 m. Shifting a hand rigidly to clear the other
        one also breaks its grasp. The two facts together say the independently
        fitted poses are mutually inconsistent rather than merely overlapping,
        and no rigid correction repairs that -- both hands have to be fitted
        against each other, which needs the wrist DoF `side_view` exposes but
        the solver cannot currently reach.
        """
        m, d = self.sim.model, self.sim.data
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            worst, n, nrm = self.inter_hand()
            if n == 0 or worst <= margin:
                break
            u = nrm / max(np.linalg.norm(nrm), 1e-9)
            step = 0.5 * (worst + margin)
            offset["r"] = offset["r"] + u * step
            offset["l"] = offset["l"] - u * step
            for sd in ("r", "l"):
                fq = self.sim.free_q[sd]
                d.qpos[fq:fq + 3] += offset["r"] if sd == "r" else offset["l"]
                mid = m.body_mocapid[self.sim.mocap_bid[sd]]
                d.mocap_pos[mid] = d.qpos[fq:fq + 3]
        mujoco.mj_forward(m, d)
        return offset

    def reset_at(self, k=0, separate=False):
        m, d = self.sim.model, self.sim.data
        mujoco.mj_resetData(m, d)
        self.offset = {"r": np.zeros(3), "l": np.zeros(3)}
        for sd in ("r", "l"):
            self.place(sd, k)
        d.qpos[self.obj_q:self.obj_q + 3] = self.ref_pos[k]
        d.qpos[self.obj_q + 3:self.obj_q + 7] = self.ref_quat[k]
        self.set_ctrl(k)
        mujoco.mj_forward(m, d)
        if separate:
            self.separate(self.offset)
        return d

    def obj_pose(self):
        d = self.sim.data
        return (d.qpos[self.obj_q:self.obj_q + 3].copy(),
                d.qpos[self.obj_q + 3:self.obj_q + 7].copy())

    def contacts(self, sd=None):
        d = self.sim.data
        obj = set(self.sim.obj_gids)
        hands = (set(self.sim.hand_gids[sd]) if sd else
                 set(self.sim.hand_gids["r"]) | set(self.sim.hand_gids["l"]))
        return sum(1 for i in range(d.ncon)
                   if ({int(d.contact[i].geom1), int(d.contact[i].geom2)} & obj)
                   and ({int(d.contact[i].geom1), int(d.contact[i].geom2)} & hands))

    def establish(self, k=0, target_n=8.0, open_by=0.25, step=0.02,
                  settle_steps=8, max_close=1.0):
        """Close both hands onto the object until the grip carries it.

        The one-handed path has had this since G5's decision rule made it
        mandatory; the two-handed path did not, which left it commanding the
        retargeted finger ANGLES and hoping they happened to press. They need
        not: a servo already at its target applies no force. Directions are
        derived per side from that hand's own closure and exclude the wrist.

        The object is pinned while the grip forms -- left free it is squeezed
        out from between two hands even more readily than from one.
        """
        m, d = self.sim.model, self.sim.data
        if self._dirs is None:
            self._dirs = {sd: closing_direction(self.sim.side_view(sd), self._acts[sd])
                          for sd in ("r", "l")}
        direction = self._dirs["r"] + self._dirs["l"]
        g = m.opt.gravity.copy()
        m.opt.gravity[:] = 0.0
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        base = self.set_ctrl(k)

        d.ctrl[:] = np.clip(base - open_by * direction, lo, hi)
        for _ in range(50):
            mujoco.mj_step(m, d)
        d.qpos[self.obj_q:self.obj_q + 3] = self.ref_pos[k]
        d.qpos[self.obj_q + 3:self.obj_q + 7] = self.ref_quat[k]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        pre = d.ctrl.copy()

        a, reached = 0.0, 0.0
        while a < max_close:
            a += step
            d.ctrl[:] = np.clip(pre + a * direction, lo, hi)
            for _ in range(settle_steps):
                mujoco.mj_step(m, d)
                d.qpos[self.obj_q:self.obj_q + 3] = self.ref_pos[k]
                d.qpos[self.obj_q + 3:self.obj_q + 7] = self.ref_quat[k]
                d.qvel[self.obj_v:self.obj_v + 6] = 0.0
                mujoco.mj_forward(m, d)
            reached = sum(1 for i in range(d.ncon)
                          if {int(d.contact[i].geom1), int(d.contact[i].geom2)}
                          & set(self.sim.obj_gids))
            tot = 0.0
            f = np.zeros(6)
            for i in range(d.ncon):
                gg = {int(d.contact[i].geom1), int(d.contact[i].geom2)}
                if gg & set(self.sim.obj_gids) and (
                        gg & (set(self.sim.hand_gids["r"]) | set(self.sim.hand_gids["l"]))):
                    mujoco.mj_contactForce(m, d, i, f)
                    tot += abs(float(f[0]))
            if tot >= target_n:
                break
        m.opt.gravity[:] = g
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self._grip_offset = d.ctrl - self.set_ctrl(k)
        d.ctrl[:] = self.set_ctrl(k) + self._grip_offset
        return a, tot

    def hold_test(self, k=0, seconds=0.8, settle=0.15, grip=None):
        """Do the two hands hold the object where the reference starts?

        The bimanual analogue of the one-handed hold test, and the score a
        two-handed grasp search needs. Without it, stage 6 could only be
        evaluated by running a whole trajectory, which conflates the grasp with
        the tracking.
        """
        m, d = self.sim.model, self.sim.data
        self.reset_at(k)
        if grip:
            self.establish(k, target_n=grip)
        for _ in range(int(settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
        p0 = self.obj_pose()[0].copy()
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return float(np.linalg.norm(self.obj_pose()[0] - p0))

    def track_score(self, steps=60, start=0):
        """Search objective. **Unitless, not millimetres.**

        It is a clipped mean error plus a drop penalty, so it is not comparable
        to a tracking error and must not be printed beside one. Doing exactly
        that cost an hour chasing a reproducibility bug that did not exist: a
        score of 0.0469 printed as "46.9 mm" next to a 92752 mm rollout looked
        like the search and the evaluation disagreeing by 2000x, when the code
        is deterministic and repeats to the digit.

        `hold_test` scores a grasp against gravity, and a grasp can pass it and
        still fail in motion: on `gamecontroller_play_1` two synthesis seeds
        both reached an excellent static hold (0.65 cm and 0.14 cm) and then
        tracked at 100.9 mm and 248 m. Holding is necessary and not sufficient,
        which is the two-handed restatement of what the one-handed stage found.
        """
        self.reset_at(start)
        r = self.rollout(start=start, steps=min(steps, self.T - start))
        # Clipping alone hides the only failure that matters. At a 1 m clip, a
        # rollout with 130 good frames and 8 catastrophic ones scored 46.9 mm
        # while its true mean was 92752 mm, and the search happily selected it.
        # The clip keeps the objective from being dominated by how FAR a
        # dropped object flew; the drop fraction puts the drop itself back in.
        e = np.minimum(r.pos_err, 0.25)
        dropped = float(np.mean(r.pos_err > 0.10))
        return float(e.mean() + 0.5 * dropped)

    def synthesize_grasp(self, k=0, samples=10, rounds=2, sigma_pos=0.012,
                         sigma_rot=0.08, seed=0, objective="track",
                         track_steps=60):
        """Search both wrists for a two-handed pose that holds the object.

        Alternating, one hand at a time, because a joint 12-dimensional search
        needs far more samples for the same coverage and each sample costs a
        physics rollout. Scored by `hold_test`, i.e. in physics -- the same
        standard the one-handed synthesis uses, which took that stage from 4/24
        to 19/24.

        The offsets are constant in the OBJECT frame and applied to every frame,
        so the trajectory's shape is untouched and only the grasp moves. Kept
        only if the hold improves.
        """
        rng = np.random.default_rng(seed)
        sig = np.array([sigma_pos] * 3 + [sigma_rot] * 3)
        # Score over the WHOLE reference by default. A 70-step window gave a
        # grasp scoring 93.7 mm on that window and 491 m over the full 192
        # frames -- the same horizon lesson PPO taught, now for the grasp
        # search: optimise what you will be judged on.
        nsteps = (self.T - k) if track_steps is None else track_steps
        score = ((lambda: self.track_score(nsteps, k))
                 if objective == "track" else (lambda: self.hold_test(k)))
        best = score()
        applied = {"r": np.zeros(6), "l": np.zeros(6)}
        for _ in range(rounds):
            for sd in ("r", "l"):
                cand = rng.normal(size=(samples, 6)) * sig
                for dlt in cand:
                    self._shift(sd, dlt)
                    v = score()
                    if v < best:
                        best = v
                        applied[sd] = applied[sd] + dlt
                    else:
                        self._shift(sd, -dlt)
        self.grasp_drop = best
        self.grasp_offsets = applied
        return best, applied

    def _shift(self, sd, delta):
        """Shift one hand's wrist by `delta` across the whole trajectory."""
        names = [mujoco.mj_id2name(self.fit[sd].model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in self.fit[sd].jids]
        pre = f"{sd}_"
        for i, n in enumerate(names):
            b = n[len(pre):] if n and n.startswith(pre) else n
            if b in ("x", "y", "z"):
                self.tr[sd].q[:, i] += delta[{"x": 0, "y": 1, "z": 2}[b]]
            elif b in ("rx", "ry", "rz"):
                self.tr[sd].q[:, i] += delta[3 + {"rx": 0, "ry": 1, "rz": 2}[b]]
        self.P[sd], self.Q[sd], self.vals[sd] = feedforward_se3(
            self.fit[sd], self.tr[sd].q, self.ref_pos, self.ref_quat)

    def rollout(self, start=0, steps=None):
        m, d = self.sim.model, self.sim.data
        steps = self.T - start if steps is None else steps
        pe = np.empty(steps)
        for i in range(steps):
            k = start + i
            for sd in ("r", "l"):
                self.command(sd, k)        # mocap targets only; the weld pulls
            self.set_ctrl(k)
            for _ in range(self.ctrl_every):
                mujoco.mj_step(m, d)
            pe[i] = float(np.linalg.norm(self.obj_pose()[0] - self.ref_pos[k]))
        return Rollout(pos_err=pe, rot_err=np.zeros(steps),
                       dropped=bool(pe[-1] > 0.10), steps=steps)


def mppi_bimanual(bt, horizon=4, samples=24, sigma_pos=0.004, sigma_rot=0.03,
                  sigma_fin=0.05, rho=1.0, w_rot=0.10, w_drop=2.0, seed=0,
                  start=0, progress=None):
    """MPPI over BOTH wrists at once.

    Two-handed grasp synthesis fixed the holding problem -- every sequence tried
    now holds the object within 0.05-3.4 cm, where before they dropped it
    outright -- and two of four then tracked. The other two hold and still lose
    the object once it accelerates, which is exactly the one-handed pattern: a
    grasp that survives gravity need not survive inertia, and a correction
    searched in physics is what covers the difference.

    The action is 12-dimensional (a translation and a rotation per wrist). The
    fingers are deliberately NOT searched: they are already at a configuration
    that holds, the search cost grows with the dimension, and letting a sampler
    perturb 40 finger targets is a good way to lose a grasp that works.
    """
    rng = np.random.default_rng(seed)
    scale = np.concatenate([np.full(3, sigma_pos), np.full(3, sigma_rot)] * 2)
    n_a = 12

    bt.reset_at(start)
    plan = np.zeros((horizon, n_a))
    pe = np.full(bt.T, np.nan)

    def apply(k, a):
        for i, sd in enumerate(("r", "l")):
            bt.command(sd, k, delta=a[6 * i:6 * i + 6] if a is not None else None)
        bt.set_ctrl(k)

    for k in range(start, bt.T):
        st = (bt.sim.data.qpos.copy(), bt.sim.data.qvel.copy(),
              bt.sim.data.mocap_pos.copy(), bt.sim.data.mocap_quat.copy())
        noise = rng.normal(size=(samples, horizon, n_a)) * scale
        cand = plan[None] + noise
        cand[0] = plan
        cand[1] = 0.0
        cost = np.empty(samples)
        for i in range(samples):
            (bt.sim.data.qpos[:], bt.sim.data.qvel[:],
             bt.sim.data.mocap_pos[:], bt.sim.data.mocap_quat[:]) = st
            mujoco.mj_forward(bt.sim.model, bt.sim.data)
            c = 0.0
            for h in range(horizon):
                apply(min(k + h, bt.T - 1), cand[i, h])
                for _ in range(bt.ctrl_every):
                    mujoco.mj_step(bt.sim.model, bt.sim.data)
                kk = int(np.clip(k + h, 0, bt.T - 1))
                e = float(np.linalg.norm(bt.obj_pose()[0] - bt.ref_pos[kk]))
                c += e + (w_drop if e > 0.10 else 0.0)
            cost[i] = c

        lam = max(rho * float(np.std(cost)), 1e-6)
        w = np.exp(-(cost - cost.min()) / lam)
        w /= w.sum()
        plan = np.einsum("i,iha->ha", w, cand)

        (bt.sim.data.qpos[:], bt.sim.data.qvel[:],
         bt.sim.data.mocap_pos[:], bt.sim.data.mocap_quat[:]) = st
        mujoco.mj_forward(bt.sim.model, bt.sim.data)
        apply(k, plan[0])
        for _ in range(bt.ctrl_every):
            mujoco.mj_step(bt.sim.model, bt.sim.data)
        pe[k] = float(np.linalg.norm(bt.obj_pose()[0] - bt.ref_pos[k]))
        plan = np.vstack([plan[1:], np.zeros(n_a)])
        if progress and k % progress == 0:
            print(f"    k={k}/{bt.T} {pe[k]*1000:.1f} mm", flush=True)

    sel = ~np.isnan(pe)
    return Rollout(pos_err=pe[sel], rot_err=np.zeros(int(sel.sum())),
                   dropped=bool(pe[bt.T - 1] > 0.10), steps=int(sel.sum()))
