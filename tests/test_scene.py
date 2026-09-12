"""Invariants of the bimanual scene that were broken at some point and must stay
fixed. Each test corresponds to a bug that actually happened and is named for it.
"""
import numpy as np
import mujoco
import pytest

from oppdef.envs.bimanual import BimanualBox, BASE_HALF, PEG_HALF, SOCKET_FRICTION


@pytest.fixture(scope="module")
def env():
    return BimanualBox()


def test_object_rests_on_its_support_not_inside_it(env):
    """dextrack_vega's BIM_BOX spawned the object 4 cm INSIDE the table, making
    the do-nothing baseline +3.99 cm and three PPO runs look like they had
    learned a 4 cm lift."""
    mujoco.mj_forward(env.m, env.d)
    box_bottom = float(env.box_pos()[2]) - BASE_HALF[2]
    assert box_bottom >= -1e-3, f"base spawns {box_bottom*100:.2f} cm below the table"


def test_do_nothing_baseline_is_zero(env):
    """The trivial baseline must be ~0, or every lift number is contaminated."""
    mujoco.mj_resetData(env.m, env.d)
    mujoco.mj_forward(env.m, env.d)
    z0 = env.box_pos()[2]
    env.d.ctrl[:] = 0
    for _ in range(400):
        mujoco.mj_step(env.m, env.d)
    assert abs(float(env.box_pos()[2] - z0)) < 0.005


def test_hands_do_not_collide_at_reset(env):
    """At a 2.5 cm flange the two hands were forced together and touched at
    reset (rh_palm vs lh_if_ds)."""
    mujoco.mj_resetData(env.m, env.d)
    mujoco.mj_forward(env.m, env.d)
    for _ in range(300):
        env.d.ctrl[:] = env.ctrl_vec()
        mujoco.mj_step(env.m, env.d)
    gn = lambda g: mujoco.mj_id2name(env.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    cross = [(gn(c.geom1), gn(c.geom2)) for c in env.d.contact[:env.d.ncon]
             if gn(c.geom1)[:3] in ("rh_", "lh_")
             and gn(c.geom2)[:3] in ("rh_", "lh_")
             and gn(c.geom1)[:3] != gn(c.geom2)[:3]]
    assert not cross, f"hands touch at reset: {cross[:3]}"


def test_task_needs_two_hands_by_construction(env):
    """Extraction must cost more upward force than the base weighs, or the
    one-handed control could succeed and the task would not be bimanual."""
    lever = float(np.hypot(env.knob_pos()[0] + BASE_HALF[0],
                           env.knob_pos()[2] - 2 * BASE_HALF[2]))
    base_weight = float(env.m.body_mass[env.box_bid]) * 9.81
    assert SOCKET_FRICTION > base_weight, (
        f"socket {SOCKET_FRICTION:.2f} N <= base weight {base_weight:.2f} N")


def test_peg_graspable_axis_is_wide_enough_to_be_feasible(env):
    """LEAP's closure gap spans 5.32-19.26 cm; a peg face under that is
    infeasible and the fingers close straight past it."""
    assert 2 * PEG_HALF[1] >= 0.053, "peg y-face is below LEAP's closure floor"
