"""Sensing invariants: the channels exist, they fire, and privileged is separable."""
import numpy as np
import mujoco
import pytest

from handsim.sensing import ObsSpec, FULL, ONBOARD_ONLY, LEGACY, PRIVILEGED


def test_onboard_only_drops_every_privileged_channel():
    assert FULL.uses_privileged()
    assert not ONBOARD_ONLY.uses_privileged()
    for c in PRIVILEGED:
        assert not getattr(ONBOARD_ONLY, c)


def test_legacy_spec_records_what_the_first_bc_actually_saw():
    """The BC baseline had no velocities, no tactile, and three privileged
    channels including the success metric itself."""
    assert not LEGACY.proprio_vel and not LEGACY.tactile
    assert LEGACY.task and LEGACY.object_pose


@pytest.mark.slow
def test_touch_sensors_fire_and_do_not_change_physics():
    from handsim.envs.bimanual import BimanualBox
    from handsim.control.expert import Expert

    plain = Expert(two_handed=True).run(verbose=False)
    ex = Expert(two_handed=True, tactile=True)
    obs = ex.e.observer(FULL)
    assert len(obs.touch_adr) > 0, "no touch sensors were added"

    peaks = np.zeros(len(obs.touch_adr))
    orig = mujoco.mj_step

    def patched(m, d, n=1):
        orig(m, d, n)
        peaks[:] = np.maximum(peaks, [d.sensordata[a] for a in obs.touch_adr])

    mujoco.mj_step = patched
    try:
        withs = ex.run(verbose=False)
    finally:
        mujoco.mj_step = orig

    assert (peaks > 1e-6).any(), "tactile never registered any force"
    assert peaks.max() > 1.0, f"tactile peak implausibly low: {peaks.max()}"
    # adding sensors must not perturb the dynamics
    assert abs(withs["peg_out_m"] - plain["peg_out_m"]) < 1e-3


@pytest.mark.slow
def test_observation_dimension_matches_the_declared_spec():
    from handsim.envs.bimanual import BimanualBox
    e = BimanualBox(tactile=True)
    mujoco.mj_forward(e.m, e.d)
    for spec in (FULL, ONBOARD_ONLY, LEGACY, ObsSpec(proprio_pos=True)):
        o = e.observer(spec)
        assert o(0.5).shape == (o.dim,), f"{spec} declared {o.dim}, got {o(0.5).shape}"
