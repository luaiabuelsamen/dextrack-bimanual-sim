"""Repairs to the Dexmate f5d6 model, which was mis-simulated in three ways.

A completion review (docs/COMPLETION_REVIEW.md) established that the f5d6 used
throughout this project is not the f5d6 in the URDF. Three defects, all
verified here before repair:

1. **The tracked fingertips were distal JOINT ORIGINS.** The terminal joint of
   each finger rotates about the very point being tracked, so moving it changed
   the tracked position by exactly 0.0000 mm -- measured, all five fingers. Any
   objective reading those points is blind to the last joint of every finger,
   and the opposition floor derived from them is measured at the wrong place.
   The URDF carries real tip frames 27.6-50.0 mm further out; MuJoCo's URDF
   importer merges them away because they hang off fixed joints.

2. **The five mimic couplings were dropped.** In the URDF each distal joint
   follows its proximal one by an affine relation (x1.13 to x1.35), so the
   right hand has SIX independent joints. The compiled model had `neq = 0` and
   this project actuated all ELEVEN independently -- roughly twice the true
   freedom. A reviewed successful grasp violated the coupling by 89.6 degrees.

3. **No actuator force limits.** The URDF gives 0.5 N*m per finger joint and
   1.0 N*m at the thumb. The simulated hand could squeeze arbitrarily hard.

Each repair makes the hand WEAKER or more constrained, never stronger, so
nothing here can manufacture a capability the real hand lacks.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

#: parent body -> (tip body name, offset), straight from the URDF joint origins
TIP_OFFSETS = {
    "R_th_l2": ("R_th_tip", (0.023, -0.0151, -0.0018)),
    "R_ff_l2": ("R_ff_tip", (-0.03, 0.0, -0.04)),
    "R_mf_l2": ("R_mf_tip", (-0.0304, 0.0, -0.0397)),
    "R_rf_l2": ("R_rf_tip", (-0.0272, 0.0, -0.042)),
    "R_lf_l2": ("R_lf_tip", (-0.0182, 0.0, -0.0306)),
    "L_th_l2": ("L_th_tip", (0.023, -0.0151, -0.0018)),
    "L_ff_l2": ("L_ff_tip", (-0.03, 0.0, -0.04)),
    "L_mf_l2": ("L_mf_tip", (-0.0304, 0.0, -0.0397)),
    "L_rf_l2": ("L_rf_tip", (-0.0272, 0.0, -0.042)),
    "L_lf_l2": ("L_lf_tip", (-0.0182, 0.0, -0.0306)),
}

#: dependent joint -> (independent joint, multiplier)
MIMIC = {
    "R_th_j2": ("R_th_j1", 1.35316), "R_ff_j2": ("R_ff_j1", 1.13028),
    "R_mf_j2": ("R_mf_j1", 1.13311), "R_rf_j2": ("R_rf_j1", 1.12935),
    "R_lf_j2": ("R_lf_j1", 1.15037),
    "L_th_j2": ("L_th_j1", 1.35316), "L_ff_j2": ("L_ff_j1", 1.13028),
    "L_mf_j2": ("L_mf_j1", 1.13311), "L_rf_j2": ("L_rf_j1", 1.12935),
    "L_lf_j2": ("L_lf_j1", 1.15037),
}

#: URDF effort limits, N*m
EFFORT = {"th_j0": 1.0, "th_j1": 1.0, "th_j2": 0.5}
for _f in ("ff", "mf", "rf", "lf"):
    EFFORT[f"{_f}_j1"] = 0.5
    EFFORT[f"{_f}_j2"] = 0.5


def effort_of(joint_name: str) -> float | None:
    """Torque limit for a finger joint, by its suffix, or None."""
    for suffix, val in EFFORT.items():
        if joint_name.endswith(suffix):
            return val
    return None


def is_dependent(joint_name: str) -> bool:
    """True if the joint is driven by a mimic coupling and must NOT be actuated."""
    return any(joint_name.endswith(d) for d in MIMIC)


def repair_mjcf(xml: str) -> str:
    """Add the real tip frames and the mimic couplings to a compiled f5d6 MJCF."""
    root = ET.fromstring(xml)
    by_name = {b.get("name"): b for b in root.iter("body")}

    added = 0
    for parent, (tip, off) in TIP_OFFSETS.items():
        b = by_name.get(parent)
        if b is None or tip in by_name:
            continue
        # a massless, geomless leaf: a coordinate frame, not a physical part,
        # so it adds no dynamics and cannot change what the hand can do
        ET.SubElement(b, "body", {"name": tip,
                                  "pos": " ".join(str(v) for v in off)})
        added += 1

    joints = {j.get("name") for j in root.iter("joint")}
    eq = root.find("equality")
    made = 0
    for dep, (indep, mult) in MIMIC.items():
        if dep not in joints or indep not in joints:
            continue
        if eq is None:
            eq = ET.SubElement(root, "equality")
        ET.SubElement(eq, "joint", {
            "name": f"mimic_{dep}", "joint1": dep, "joint2": indep,
            "polycoef": f"0 {mult} 0 0 0"})
        made += 1
    return ET.tostring(root, encoding="unicode")
