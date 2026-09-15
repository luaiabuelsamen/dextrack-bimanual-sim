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

        # The base slides ship with 0.6 m of travel, which is right for holding
        # a grasp and far too little for following a GRAB clip -- the mug in
        # `mug_drink_1` moves 1.21 m. Widened here, in the env that needs it,
        # so every result measured under the old limits keeps them.
        for j in self.sc.jids:
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n in ("x", "y", "z"):
                m.jnt_range[j] = [-reach, reach]
        for a in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            if n[:-4] in ("x", "y", "z"):
                m.actuator_ctrlrange[a] = [-reach, reach]

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
