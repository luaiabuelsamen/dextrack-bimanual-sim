"""Does the epsilon we compute predict what a hand can physically hold?

Everything measured so far is GEOMETRY. `geometric_epsilon` reads fingertip
positions off a kinematic pose, picks the nearest point on the object surface,
and builds a wrench set from it. No contact is ever simulated, no force is ever
applied, and nothing is ever dropped. That makes every retargeting number in
this project a PREDICTION, and an instrument whose predictions have not been
checked is not yet an instrument.

This module applies the check. The hand is fixed in space, the object is held in
the air with NOTHING underneath, gravity is off, and a test wrench is applied
directly to the object in many directions at increasing magnitude. The largest
magnitude the grasp survives in its WORST direction is the physical analogue of
epsilon -- epsilon is defined as the radius of the largest wrench ball the
contact set can resist, so the measurement is the same quantity in newtons.

Two details that decide whether the test means anything:

*Squeeze.* A geometric contact set assumes forces CAN be applied at the
contacts. A fingertip merely touching the surface transmits nothing, so the
joint targets are driven slightly past the fitted pose toward the hand's derived
closure. Without this every pose fails and the test measures nothing. It is
applied identically to every condition.

*No floor.* A failed grasp must be unambiguous. With nothing underneath, a
dropped object accelerates away and its displacement grows without bound --
there is no resting place that could be mistaken for a hold. This project has
twice recorded a success criterion satisfied by an object lying on a surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

#: force directions probed: the 6 axes and the 8 octant diagonals
DIRECTIONS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    + [[x, y, z] for x in (1, -1) for y in (1, -1) for z in (1, -1)], float)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)


def build_hold_scene(hand_key, obj_half, obj_pos, mass=0.05,
                     friction="1.0 0.02 0.001", kp=8.0, kv=0.1):
    """Hand fixed in space, a free box at `obj_pos`, position actuators on the
    hand's joints. Returns (model, joint_ids in the retargeter's order)."""
    from oppdef.embodiment import make
    from oppdef.grasping.retarget_pose import retargeter_for

    rt = retargeter_for(hand_key, free_base=False)
    emb = make(hand=hand_key, free_base=False)
    spec = emb.spec

    names = [mujoco.mj_id2name(rt.m, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in rt.jids]
    have = {a.target for a in spec.actuators}
    for n in names:
        if n in have:
            continue
        gp = [0.0] * 10; gp[0] = kp
        bp = [0.0] * 10; bp[1], bp[2] = -kp, -kv
        spec.add_actuator(name=f"hold_{n}", target=n,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gainprm=gp, biasprm=bp,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE)
    b = spec.worldbody.add_body(name="held", pos=[float(x) for x in obj_pos])
    b.add_freejoint(name="held_free")
    b.add_geom(name="held_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[float(x) for x in obj_half], mass=float(mass),
               rgba=[0.85, 0.3, 0.2, 1.0],
               friction=[float(x) for x in friction.split()])
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # gravity OFF: the only load is the test wrench, so the number reported is
    # the wrench the grasp resists rather than that minus the object's weight
    spec.option.gravity = [0.0, 0.0, 0.0]
    return spec.compile(), names, rt


@dataclass
class HoldResult:
    min_force_N: float          # worst direction -- the physical analogue of eps
    per_direction_N: np.ndarray
    contacts_after_settle: int
    settled: bool


def _contacts(m, d, geom_name="held_geom"):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == gid or c.geom2 == gid:
            n += 1
    return n


def hold_test(hand_key, q, joint_names_expected, obj_half, obj_pos, mass=0.05,
              squeeze=0.12, settle_steps=600, push_steps=400,
              ladder=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0), max_disp=0.02,
              model_cache=None):
    """Squeeze onto the object, then push it in every direction until it slips.

    Returns the largest force (N) sustained in the WORST direction, which is
    zero whenever the grasp cannot hold at the smallest rung of the ladder.
    """
    key = (hand_key, tuple(np.round(obj_half, 6)), tuple(np.round(obj_pos, 6)))
    if model_cache is not None and key in model_cache:
        m, names, rt = model_cache[key]
    else:
        m, names, rt = build_hold_scene(hand_key, obj_half, obj_pos, mass)
        if model_cache is not None:
            model_cache[key] = (m, names, rt)
    assert names == list(joint_names_expected), "joint order changed"

    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    act = {}
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                                   int(m.actuator_trnid[a, 0]))
            act[jn] = a
    held_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "held")
    held_q = m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "held_free")]

    # target = the fitted pose, driven `squeeze` of the way further toward the
    # hand's own closure so the fingertips actually press
    q = np.asarray(q, float)
    tgt = q.copy()
    if rt.q_closure is not None:
        cl = np.clip(rt.q_closure, rt.lo, rt.hi)
        tgt = (1 - squeeze) * q + squeeze * cl
    tgt = np.clip(tgt, rt.lo, rt.hi)

    d = mujoco.MjData(m)

    def reset_and_settle():
        mujoco.mj_resetData(m, d)
        d.qpos[qadr] = q
        d.qpos[held_q:held_q + 3] = obj_pos
        d.qpos[held_q + 3:held_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        for n, a in act.items():
            if n in names:
                d.ctrl[a] = tgt[names.index(n)]
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)

    reset_and_settle()
    n_contact = _contacts(m, d)
    start = d.qpos[held_q:held_q + 3].copy()
    settled = bool(np.linalg.norm(start - obj_pos) < max_disp)
    snapshot = (d.qpos.copy(), d.qvel.copy())

    per_dir = np.zeros(len(DIRECTIONS))
    for i, u in enumerate(DIRECTIONS):
        best = 0.0
        for f in ladder:
            d.qpos[:], d.qvel[:] = snapshot[0].copy(), snapshot[1].copy()
            mujoco.mj_forward(m, d)
            p0 = d.qpos[held_q:held_q + 3].copy()
            for _ in range(push_steps):
                d.xfrc_applied[held_bid, :3] = u * f
                mujoco.mj_step(m, d)
            d.xfrc_applied[held_bid, :] = 0.0
            disp = float(np.linalg.norm(d.qpos[held_q:held_q + 3] - p0))
            if disp < max_disp and _contacts(m, d) > 0:
                best = f
            else:
                break
        per_dir[i] = best
    return HoldResult(min_force_N=float(per_dir.min()), per_direction_N=per_dir,
                      contacts_after_settle=n_contact, settled=settled)
