"""Hand definitions for the multi-hand grasp bench, with curl directions DERIVED.

The bench needs, per hand: where the palm is, which bodies are fingertips, and
which way each joint has to move to close the hand. The first two are naming;
the third is the one that must not be guessed. Hard-coding a sign per hand is
how per-hand bias gets into a cross-hand comparison, and guessing it is how the
earlier bench ended up driving LEAP's joints through the object and scoring it
0/72.

So `derive_flex` SOLVES for it, over all joints at once, using the same
objective the opposition axis uses. Identical procedure for every hand, no
per-hand constants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

from oppdef.paths import (MENAGERIE, VEGA_URDF,  # noqa: E402
                          compile_urdf, menagerie_xml)



# name-level facts only; everything mechanical is measured
SPECS = {
    "leap": dict(kind="mjcf", path=MENAGERIE / "leap_hand/right_hand.xml",
                 palm="palm", tips=["if_ds", "mf_ds", "rf_ds"],
                 thumb="th_ds", note="LEAP, 16 DoF"),
    "allegro": dict(kind="mjcf", path=MENAGERIE / "wonik_allegro/right_hand.xml",
                    palm="palm", tips=["ff_tip", "mf_tip", "rf_tip"],
                    thumb="th_tip", note="Wonik Allegro, 16 DoF"),
    "shadow": dict(kind="mjcf", path=MENAGERIE / "shadow_hand/right_hand.xml",
                   palm="rh_palm",
                   tips=["rh_ffdistal", "rh_mfdistal", "rh_rfdistal",
                         "rh_lfdistal"],
                   thumb="rh_thdistal", note="Shadow Hand, 24 DoF"),
    "f5d6": dict(kind="urdf", path=VEGA_URDF, palm="R_arm_l7",
                 # the REAL tips, 27.6-50.0 mm beyond the distal joint
                 # origins that were tracked before. Those origins are the
                 # axes the terminal joints rotate about, so tracking them
                 # made the last joint of every finger invisible: measured
                 # displacement exactly 0.0000 mm. See hands/f5d6.py.
                 # the distal LINKS, whose tips hands/tips.py derives from
                 # their collision geometry -- the same rule every other hand
                 # uses. The URDF's own tip frames are massless and geomless,
                 # so a floor measured between them can close to 0.6 mm while
                 # the actual links interpenetrate by 6.6 mm.
                 tips=["R_ff_l2", "R_mf_l2", "R_rf_l2", "R_lf_l2"],
                 thumb="R_th_l2",
                 joints=["R_th_j0", "R_th_j1", "R_th_j2",
                         "R_ff_j1", "R_ff_j2", "R_mf_j1", "R_mf_j2",
                         "R_rf_j1", "R_rf_j2", "R_lf_j1", "R_lf_j2"],
                 note="Dexmate f5d6, 11 joints (underactuated)"),
}


def load(key):
    cfg = SPECS[key]
    if cfg["kind"] == "mjcf":
        xml = cfg["path"].read_text()
        assets = (cfg["path"].parent / "assets").as_posix()
        xml = xml.replace('meshdir="./assets/"', f'meshdir="{assets}/"')
        xml = xml.replace('meshdir="assets"', f'meshdir="{assets}"')
        return mujoco.MjModel.from_xml_string(xml), cfg
    return mujoco.MjModel.from_xml_string(compile_urdf(cfg['path'])), cfg


def hand_joints(m, cfg):
    """Actuated hinge/slide joints belonging to the hand."""
    if cfg.get("joints"):
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
               for j in cfg["joints"]]
        return [i for i in ids if i >= 0]
    return [j for j in range(m.njnt)
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                 mujoco.mjtJoint.mjJNT_SLIDE)]


def derive_flex(key, restarts=12, seed=0, cap=1.3):
    """The hand's closing pose, solved for as a WHOLE, not joint by joint.

    Perturbing one joint at a time does not work: a closure is a joint
    COMBINATION, and single-joint probes derived if_mcp = -0.314 for LEAP where
    the validated closure in grasp_bench.py uses +0.9. Both the palm-distance and
    the tip-spread probes got it wrong for the same reason.

    Instead, solve the same problem the opposition axis already solves -- bring
    the thumb tip to the fingers' mean, over all hand joints at once, multi-start
    within joint limits. The minimiser's pose IS the closure, and scaling it by a
    fraction gives the closure family the bench sweeps. Identical procedure for
    every hand; nothing per-hand is assumed.
    """
    from scipy.optimize import minimize
    m, cfg = load(key)
    d = mujoco.MjData(m)
    tip_names = list(cfg["tips"]) + [cfg["thumb"]]
    tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t) for t in tip_names]
    fg, th = tips[:-1], tips[-1]
    # The tips are the far end of each distal link, DERIVED from its collision
    # geometry -- not the body origin. Every hand's most distal joint rotates
    # about its tip body's origin, so tracking origins measured this gap at a
    # point that moves 0.000 mm with the last joint of every finger and is not
    # the part that touches anything. See hands/tips.py.
    from oppdef.hands.tips import tip_offset, tip_points
    offs = [tip_offset(m, b) for b in tips]
    ids = hand_joints(m, cfg)
    qadr = np.array([m.jnt_qposadr[j] for j in ids])
    rng_ = m.jnt_range[ids]
    lo, hi = rng_[:, 0].copy(), rng_[:, 1].copy()
    bad = hi <= lo
    lo[bad], hi[bad] = -np.pi, np.pi
    lo, hi = np.clip(lo, -cap, cap), np.clip(hi, -cap, cap)

    # Which bodies belong to the FINGERS. The self-collision penalty has to be
    # scoped to them: f5d6's file is a whole robot, and an unscoped penalty was
    # dominated by a 6.57 mm overlap between head_l1 and head_l3 -- the robot's
    # HEAD -- plus palm/thumb-base overlaps that are present at rest. Those are
    # constant across poses, so they biased nothing and masked everything.
    finger_bodies = set()
    for t in tips:
        b = int(t)
        while b > 0:
            finger_bodies.add(b)
            nb = int(m.body_parentid[b])
            if nb == 0 or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, nb) == \
                    cfg["palm"]:
                break
            b = nb

    def _self_pen():
        mujoco.mj_collision(m, d)
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            b1 = int(m.geom_bodyid[c.geom1]); b2 = int(m.geom_bodyid[c.geom2])
            if b1 in finger_bodies and b2 in finger_bodies:
                worst = max(worst, -float(c.dist))
        return worst

    from oppdef.hands.f5d6 import MIMIC
    dep = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, dj),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ij), mult)
           for dj, (ij, mult) in MIMIC.items()
           if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, dj) >= 0]

    def gap(x):
        d.qpos[:] = 0.0
        d.qpos[qadr] = x
        # equality constraints bind the SOLVER, not mj_kinematics, so a purely
        # kinematic search would silently use coupled joints independently --
        # freedom the real hand does not have.
        for dj, ij, mult in dep:
            jr = m.jnt_range[dj]
            d.qpos[m.jnt_qposadr[dj]] = float(np.clip(
                d.qpos[m.jnt_qposadr[ij]] * mult, jr[0], jr[1]))
        mujoco.mj_kinematics(m, d)
        P = tip_points(m, d, tips, offs)
        # Distance from the thumb tip to the NEAREST fingertip, not to their
        # mean. The mean is not a physical location: a thumb sitting in the
        # middle of splayed fingers scores zero while touching nothing, which
        # is how shadow, leap and allegro all read 0.00 cm with no
        # self-penetration whatsoever. What the floor is meant to capture is
        # the narrowest thing the hand can pinch, and that is thumb-to-finger.
        g = float(np.linalg.norm(P[:-1] - P[-1][None, :], axis=1).min())
        # SELF-COLLISION is what makes this a floor rather than a tautology.
        # Measured at the true fingertips with the fingers free to pass through
        # one another, shadow, leap and allegro all reported 0.00 cm: the
        # optimiser simply drives the thumb tip into coincidence with the
        # others. The quantity only means something if the hand is solid.
        return g + 50.0 * _self_pen()

    rs = np.random.default_rng(seed)
    best = None
    for i in range(restarts):
        x0 = np.zeros(len(ids)) if i == 0 else lo + rs.random(len(ids)) * (hi - lo)
        r = minimize(gap, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(maxiter=600))
        if best is None or r.fun < best.fun:
            best = r
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ids]
    flex = {n: float(v) for n, v in zip(names, np.clip(best.x, lo, hi))}
    # report the GAP at the solution, not the penalised objective -- mixing the
    # two reported f5d6 at 34.57 cm, which is a penalty, not a distance
    d.qpos[:] = 0.0
    d.qpos[qadr] = np.clip(best.x, lo, hi)
    for dj, ij, mult in dep:
        jr = m.jnt_range[dj]
        d.qpos[m.jnt_qposadr[dj]] = float(np.clip(
            d.qpos[m.jnt_qposadr[ij]] * mult, jr[0], jr[1]))
    mujoco.mj_kinematics(m, d)
    P = tip_points(m, d, tips, offs)
    true_gap = float(np.linalg.norm(P[:-1] - P[-1][None, :], axis=1).min())
    return m, cfg, flex, tip_names, true_gap


if __name__ == "__main__":
    for key in ("leap", "allegro", "shadow", "f5d6"):
        try:
            m, cfg, flex, tips, gap = derive_flex(key)
            print(f"{key:9} {cfg['note']:38} joints {len(flex):2}  tips {len(tips)}"
                  f"  closed gap {gap*100:5.2f} cm")
            print("          " + ", ".join(f"{k}={v:+.2f}"
                                           for k, v in list(flex.items())[:4]) + " ...")
        except Exception as e:
            print(f"{key:9} FAILED: {type(e).__name__}: {str(e)[:90]}")
