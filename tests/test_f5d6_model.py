"""The f5d6 model defects a completion review found, as regression tests.

Each of these passed silently for the whole project and changed what the
opposition floor -- the quantity this project is named after -- measured.
"""
import numpy as np
import pytest
import mujoco

from handsim.hands.f5d6 import MIMIC, TIP_OFFSETS, effort_of, is_dependent


@pytest.mark.slow
def test_tracked_fingertips_move_when_the_distal_joint_moves():
    """The tips used to be distal JOINT ORIGINS, and the terminal joint rotates
    about that very point -- moving it displaced the tracked point by exactly
    0.0000 mm on all five fingers, so every objective reading them was blind to
    the last joint of every finger."""
    from handsim.hands.specs import load
    m, _cfg = load("f5d6")
    d = mujoco.MjData(m)
    for finger in ("ff", "mf", "rf", "lf"):
        tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"R_{finger}_tip")
        assert tip >= 0, f"R_{finger}_tip missing from the compiled model"
        for suffix in ("j1", "j2"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"R_{finger}_{suffix}")
            d.qpos[:] = 0.0
            mujoco.mj_kinematics(m, d)
            p0 = d.xpos[tip].copy()
            d.qpos[m.jnt_qposadr[j]] = -0.4
            mujoco.mj_kinematics(m, d)
            moved = np.linalg.norm(d.xpos[tip] - p0)
            assert moved > 1e-3, (
                f"R_{finger}_{suffix} moves the tracked tip {moved*1000:.4f} mm")


@pytest.mark.slow
def test_mimic_couplings_are_present():
    """The URDF couples each distal joint to its proximal one, so the right
    hand has SIX independent joints. Dropped, the model had eleven and this
    project actuated all of them -- roughly twice the true freedom."""
    from handsim.hands.specs import load
    m, _cfg = load("f5d6")
    assert m.neq >= 5, f"expected mimic equality constraints, got neq={m.neq}"
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e)
             for e in range(m.neq)}
    for dep in ("R_th_j2", "R_ff_j2", "R_mf_j2", "R_rf_j2", "R_lf_j2"):
        assert f"mimic_{dep}" in names, f"{dep} is not coupled"


def test_dependent_joints_are_identified():
    for d in ("R_ff_j2", "R_th_j2", "L_lf_j2"):
        assert is_dependent(d)
    for i in ("R_ff_j1", "R_th_j0", "R_arm_j3"):
        assert not is_dependent(i)


def test_effort_limits_are_known():
    """The URDF gives 0.5 N*m per finger joint and 1.0 at the thumb; the
    simulated hand had none and could squeeze arbitrarily hard."""
    assert effort_of("R_ff_j1") == pytest.approx(0.5)
    assert effort_of("R_th_j0") == pytest.approx(1.0)
    assert effort_of("R_arm_j3") is None


def test_tip_offsets_are_not_negligible():
    """27.6-50.0 mm -- the same scale as the opposition floor itself, which is
    why tracking the wrong point changed the headline number by 2x."""
    for _parent, (_name, off) in TIP_OFFSETS.items():
        assert np.linalg.norm(off) > 0.02
