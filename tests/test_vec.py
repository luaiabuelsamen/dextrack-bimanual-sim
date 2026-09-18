"""Batched stepping: does N worlds mean the same physics as one world, N times?

A fast backend that disagrees with single-world MuJoCo is not a faster
simulator, it is a different one. These tests use a tiny model so they run in
the fast suite; the real scene is covered by the parity target.
"""
import numpy as np
import pytest
import mujoco

from handsim.vec import CpuVec, Batch, make_vec
from handsim.sim.port import rollout_cpu

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="cart" pos="0 0 0.2">
      <joint name="slide" type="slide" axis="1 0 0" damping="0.1"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.01" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="hinge" ctrlrange="-0.5 0.5" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(XML)


def test_identical_controls_give_bitwise_identical_worlds(model):
    v = CpuVec(model, 6)
    c = np.broadcast_to(np.array([0.3, -0.2]), (6, model.nu))
    for _ in range(40):
        s = v.step(c)
    assert np.abs(s.qpos - s.qpos[0]).max() == 0.0
    assert np.abs(s.qvel - s.qvel[0]).max() == 0.0


def test_worlds_are_independent(model):
    """Different controls must give different worlds -- the failure mode where
    every world silently runs the same trajectory would pass the test above."""
    v = CpuVec(model, 3)
    c = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.5]])
    for _ in range(40):
        s = v.step(c)
    assert np.abs(s.qpos[0] - s.qpos[1]).max() > 1e-3
    assert np.abs(s.qpos[0] - s.qpos[2]).max() > 1e-3


def test_batch_size_need_not_equal_thread_count(model):
    """`rollout` takes one MjData per THREAD and infers the batch from the
    state array. Seeding the batch from the thread pool made it read nthread
    where n was meant, and it raised as soon as the two differed."""
    for n, nthread in ((16, 4), (3, 8), (1, 1), (32, 2)):
        v = CpuVec(model, n, nthread=nthread)
        s = v.step(np.zeros((n, model.nu)))
        assert s.qpos.shape == (n, model.nq)


def test_matches_single_world_mujoco(model):
    """The reference every fast backend is measured against."""
    c1 = np.array([0.4, -0.1])
    v = CpuVec(model, 4)
    for _ in range(60):
        s = v.step(np.broadcast_to(c1, (4, model.nu)))
    ref = rollout_cpu(model, np.broadcast_to(c1, (60, model.nu)))
    got = np.concatenate([s.qpos[0], s.qvel[0]])
    assert np.abs(got - ref[-1]).max() < 1e-6


def test_reset_accepts_per_world_state(model):
    v = CpuVec(model, 3)
    qpos = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    s = v.reset(qpos=qpos)
    assert np.allclose(s.qpos, qpos)


def test_reset_clears_previous_motion(model):
    v = CpuVec(model, 2)
    for _ in range(30):
        v.step(np.full((2, model.nu), 0.5))
    s = v.reset()
    assert np.abs(s.qvel).max() == 0.0


def test_control_limits_are_applied(model):
    """A real actuator cannot be commanded past its range, so neither may a
    policy in simulation -- otherwise learned behaviour exploits headroom the
    hardware does not have."""
    v = CpuVec(model, 2)
    out = v.clamp(np.array([[5.0, 5.0], [-5.0, -5.0]]))
    assert out[0].tolist() == [1.0, 0.5]
    assert out[1].tolist() == [-1.0, -0.5]


def test_batch_flat_concatenates_pos_and_vel():
    b = Batch(qpos=np.zeros((4, 3)), qvel=np.ones((4, 2)))
    assert b.flat().shape == (4, 5)
    assert b.n == 4


def test_make_vec_dispatches(model):
    v = make_vec(model, 2, backend="cpu")
    assert v.backend == "cpu" and v.n == 2
