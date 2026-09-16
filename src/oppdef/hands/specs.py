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
    "leap_left": dict(kind="mjcf", path=MENAGERIE / "leap_hand/left_hand.xml",
                      palm="palm", tips=["if_ds", "mf_ds", "rf_ds"],
                      thumb="th_ds", note="LEAP left, 16 DoF"),
    "allegro_left": dict(kind="mjcf",
                         path=MENAGERIE / "wonik_allegro/left_hand.xml",
                         palm="palm", tips=["ff_tip", "mf_tip", "rf_tip"],
                         thumb="th_tip", note="Allegro left, 16 DoF"),
    "shadow_left": dict(kind="mjcf", path=MENAGERIE / "shadow_hand/left_hand.xml",
                        palm="lh_palm",
                        tips=["lh_ffdistal", "lh_mfdistal", "lh_rfdistal",
                              "lh_lfdistal"],
                        thumb="lh_thdistal", note="Shadow left, 24 DoF"),
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
    #: The left f5d6 comes from the SAME URDF as the right, distinguished by
    #: link prefix rather than by a separate file -- unlike the Menagerie hands
    #: above, which ship a left_hand.xml. It was registered in embodiment.py
    #: and never here, so every bimanual path that resolved a hand through
    #: `specs.load` raised KeyError on it; all eleven joints and six links below
    #: were checked against the URDF before being written down.
    "f5d6_left": dict(kind="urdf", path=VEGA_URDF, palm="L_arm_l7",
                      tips=["L_ff_l2", "L_mf_l2", "L_rf_l2", "L_lf_l2"],
                      thumb="L_th_l2",
                      joints=["L_th_j0", "L_th_j1", "L_th_j2",
                              "L_ff_j1", "L_ff_j2", "L_mf_j1", "L_mf_j2",
                              "L_rf_j1", "L_rf_j2", "L_lf_j1", "L_lf_j2"],
                      note="Dexmate f5d6 left, 11 joints (underactuated)"),
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
    from oppdef.hands.model import (tip_bodies, finger_body_set, mimic_pairs,
                                    apply_mimic, self_penetration, thumb_gap)
    m, cfg = load(key)
    d = mujoco.MjData(m)
    tip_names = list(cfg["tips"]) + [cfg["thumb"]]
    # One shared implementation of "where is a fingertip", "which joints are
    # coupled" and "is the hand inside itself" (hands/model.py). A second copy
    # here is how the canonical axis command went on reporting a retracted
    # number after the solver had already been fixed.
    _names, ids, offs = tip_bodies(m, cfg)
    bodies = finger_body_set(m, cfg, ids)
    pairs = mimic_pairs(m)

    jids = hand_joints(m, cfg)
    qadr = np.array([m.jnt_qposadr[j] for j in jids])
    rng_ = m.jnt_range[jids]
    lo, hi = rng_[:, 0].copy(), rng_[:, 1].copy()
    bad = hi <= lo
    lo[bad], hi[bad] = -np.pi, np.pi
    lo, hi = np.clip(lo, -cap, cap), np.clip(hi, -cap, cap)

    def gap(x):
        d.qpos[:] = 0.0
        d.qpos[qadr] = x
        apply_mimic(m, d, pairs)
        mujoco.mj_kinematics(m, d)
        return thumb_gap(m, d, ids, offs) + 50.0 * self_penetration(m, d, bodies)

    rs = np.random.default_rng(seed)
    best = None
    for i in range(restarts):
        x0 = (np.zeros(len(jids)) if i == 0
              else lo + rs.random(len(jids)) * (hi - lo))
        r = minimize(gap, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(maxiter=600))
        if best is None or r.fun < best.fun:
            best = r
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids]
    flex = {n: float(v) for n, v in zip(jnames, np.clip(best.x, lo, hi))}
    # the GAP at the solution, never the penalised objective
    d.qpos[:] = 0.0
    d.qpos[qadr] = np.clip(best.x, lo, hi)
    apply_mimic(m, d, pairs)
    mujoco.mj_kinematics(m, d)
    return m, cfg, flex, tip_names, thumb_gap(m, d, ids, offs)


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
