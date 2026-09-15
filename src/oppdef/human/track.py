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
    R0 = grab_mod._rodrigues(seq.obj_quat_aa[k0][None])[0]
    p0 = seq.obj_pos[k0]
    R = grab_mod._rodrigues(seq.obj_quat_aa)
    pos = (seq.obj_pos - p0) @ R0
    rel = np.einsum("ab,tbc->tac", R0.T, R)
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
             settle: float = 0.25) -> HoldResult:
        """Command the hand to stay at `q` and see whether the object stays.

        `settle` seconds are simulated before the object's position is taken as
        the origin, so the few millimetres of contact resolution that the
        retargeting leaves are not counted as a drop.
        """
        m, d = self.sc.model, self.sc.data
        self.reset(q)
        pen0, _ = self.sc.penetration()
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
