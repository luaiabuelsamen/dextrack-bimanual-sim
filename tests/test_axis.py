"""The opposition axis, as produced by the command the repository ships.

The previous version of `hands/axis.py` kept its own hand table and its own
distance function, so when the fingertip definition was corrected in the
closure solver the canonical command went on reporting the retracted number.
These tests run the PUBLIC entry point, not a private copy of the maths.
"""
import numpy as np
import pytest
import mujoco

from oppdef.hands.axis import opposition_axis, provenance
from oppdef.hands.specs import SPECS, load
from oppdef.hands.model import (tip_bodies, mimic_pairs, finger_body_set,
                                apply_mimic, self_penetration)

#: the retracted values, kept so the old numbers cannot come back unnoticed
RETRACTED_FLOOR_CM = {"shadow": 0.25, "leap": 0.80, "allegro": 2.41, "f5d6": 3.40}


@pytest.mark.slow
def test_f5d6_floor_is_not_the_retracted_value():
    """3.40 cm came from tracking distal joint origins with the finger coupling
    dropped. The corrected floor is under 1 cm."""
    ax = opposition_axis("f5d6", restarts=12)
    assert ax.floor_m * 100 < 1.0, f"floor {ax.floor_m*100:.2f} cm"
    assert abs(ax.floor_m * 100 - RETRACTED_FLOOR_CM["f5d6"]) > 1.0


@pytest.mark.slow
def test_axis_uses_the_registry_tips_not_body_origins():
    for key in ("leap", "f5d6"):
        ax = opposition_axis(key, restarts=2)
        expect = tuple(list(SPECS[key]["tips"]) + [SPECS[key]["thumb"]])
        assert ax.tips == expect


@pytest.mark.slow
def test_floor_is_below_aperture_and_free_of_self_penetration():
    for key in ("leap", "allegro"):
        ax = opposition_axis(key, restarts=8)
        assert ax.floor_m < ax.aperture_m
        # a "floor" reached by driving the fingers through each other is not a
        # floor; an earlier version reported 34.57 cm because it returned the
        # PENALTY instead of the distance
        assert ax.floor_self_pen_mm < 1.0


@pytest.mark.slow
def test_mimic_is_applied_during_the_kinematic_search():
    """Equality constraints bind the solver, not mj_kinematics, so a kinematic
    search must apply them by hand or it uses freedom the hand lacks."""
    m, cfg = load("f5d6")
    pairs = mimic_pairs(m)
    assert len(pairs) >= 5, f"expected the five right-hand couplings, got {len(pairs)}"
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    qa, qb, mult, lo, hi = pairs[0]
    d.qpos[qb] = 0.3
    apply_mimic(m, d, pairs)
    assert d.qpos[qa] == pytest.approx(np.clip(0.3 * mult, lo, hi))


@pytest.mark.slow
def test_self_penetration_is_scoped_to_the_fingers():
    """Unscoped, f5d6's deepest overlap is between head_l1 and head_l3 -- the
    robot's HEAD -- which is constant across poses and discriminates nothing."""
    m, cfg = load("f5d6")
    _n, ids, _o = tip_bodies(m, cfg)
    bodies = finger_body_set(m, cfg, ids)
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    mujoco.mj_kinematics(m, d)
    scoped = self_penetration(m, d, bodies)
    mujoco.mj_collision(m, d)
    unscoped = max([-float(d.contact[i].dist) for i in range(d.ncon)] + [0.0])
    assert unscoped > 0.004, "expected the known rest-pose overlaps"
    assert scoped < unscoped


def test_provenance_is_recorded():
    p = provenance()
    for k in ("commit", "mujoco", "numpy", "python", "when"):
        assert k in p and p[k]
