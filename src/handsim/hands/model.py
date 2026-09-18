"""Shared kinematic facts about a hand, so one correction fixes every caller.

The opposition axis and the closure solver each grew their own copy of "where
is a fingertip", "which joints are coupled" and "is the hand inside itself".
When the fingertip definition turned out to be wrong, only one copy was fixed,
and the repository's canonical axis command went on reporting the retracted
number. These helpers exist so that cannot happen again: there is one place
each of those questions is answered.
"""
from __future__ import annotations

import numpy as np
import mujoco

from handsim.hands.tips import tip_offset, tip_points


def tip_bodies(model, cfg):
    """(body ids, tip offsets) for fingers-then-thumb, in that order."""
    names = list(cfg["tips"]) + [cfg["thumb"]]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    missing = [n for n, i in zip(names, ids) if i < 0]
    if missing:
        raise KeyError(f"tip bodies missing from the model: {missing}")
    return names, ids, [tip_offset(model, b) for b in ids]


def finger_body_set(model, cfg, ids):
    """Bodies belonging to the fingers, walking up to (not past) the palm.

    Scoping matters: an unscoped self-collision test on f5d6 is dominated by a
    6.6 mm overlap between the robot's HEAD links, which is constant across
    poses and therefore discriminates nothing.
    """
    out = set()
    for t in ids:
        b = int(t)
        while b > 0:
            out.add(b)
            nb = int(model.body_parentid[b])
            if nb == 0 or mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, nb) == cfg["palm"]:
                break
            b = nb
    return out


def mimic_pairs(model):
    """[(dependent qposadr, independent qposadr, multiplier, lo, hi)].

    MuJoCo equality constraints bind the SOLVER, not `mj_kinematics`, so any
    purely kinematic search must apply them by hand or it silently uses freedom
    the hand does not have.
    """
    from handsim.hands.f5d6 import MIMIC
    out = []
    for dep, (indep, mult) in MIMIC.items():
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dep)
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, indep)
        if a < 0 or b < 0:
            continue
        lo, hi = model.jnt_range[a]
        out.append((int(model.jnt_qposadr[a]), int(model.jnt_qposadr[b]),
                    float(mult), float(lo), float(hi)))
    return out


def apply_mimic(model, data, pairs):
    for qa, qb, mult, lo, hi in pairs:
        data.qpos[qa] = float(np.clip(data.qpos[qb] * mult, lo, hi))


def self_penetration(model, data, bodies):
    """Deepest overlap between two FINGER bodies, metres, 0 if none."""
    mujoco.mj_collision(model, data)
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= 0:
            continue
        if int(model.geom_bodyid[c.geom1]) in bodies and \
                int(model.geom_bodyid[c.geom2]) in bodies:
            worst = max(worst, -float(c.dist))
    return worst


def thumb_gap(model, data, ids, offs):
    """Distance from the thumb tip to the NEAREST fingertip.

    Not to their mean: a thumb sitting among splayed fingers is zero from the
    mean while touching nothing, which is how three hands measured 0.00 cm with
    no self-contact at all.
    """
    P = tip_points(model, data, ids, offs)
    return float(np.linalg.norm(P[:-1] - P[-1][None, :], axis=1).min())
