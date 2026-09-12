"""The claims the project rests on, as executable checks.

Marked slow: these run physics. They exist so that a refactor cannot silently
invalidate a published number.
"""
import numpy as np
import pytest


@pytest.mark.slow
def test_visual_strip_is_physics_neutral():
    """The MJX port removes visual-only geoms and NOTHING else. If this ever
    stops being bit-identical, the port has started changing the physics and
    every GPU number needs re-deriving."""
    import mujoco
    from oppdef.envs.bimanual import build
    from oppdef.sim.port import strip_visual, rollout_cpu

    full, _ = build()
    _, spec = build()
    strip_visual(spec)
    stripped = spec.compile()
    assert (full.nq, full.nv) == (stripped.nq, stripped.nv)
    assert stripped.nmeshvert < full.nmeshvert / 100

    rng = np.random.default_rng(0)
    ctrl = np.clip(rng.normal(0, 0.25, (120, full.nu)), -0.5, 0.5)
    for i in range(full.nu):
        if full.actuator_ctrllimited[i]:
            lo, hi = full.actuator_ctrlrange[i]
            ctrl[:, i] = np.clip(ctrl[:, i], lo, hi)
    err = np.abs(rollout_cpu(full, ctrl) - rollout_cpu(stripped, ctrl)).max()
    assert err == 0.0, f"strip changed the dynamics by {err:.3e}"


@pytest.mark.slow
def test_task_requires_two_hands():
    """The headline claim: the expert extracts the peg, the one-handed control
    lifts the base instead. Criterion v2 (peg >= 8 cm, base LIFT < 2 cm)."""
    from oppdef.control.expert import Expert

    two = Expert(two_handed=True).run(verbose=False)
    one = Expert(two_handed=False).run(verbose=False)
    assert two["success"], f"expert failed: {two}"
    assert not one["success"], f"one-handed control succeeded: {one}"
    assert two["peg_out_m"] >= 0.08
    assert one["box_z_rise_m"] > two["box_z_rise_m"] + 0.02, (
        "the control should fail by LIFTING the base")


@pytest.mark.slow
def test_opposition_floor_separates_f5d6_from_the_others():
    """f5d6 is the only hand that cannot bring its thumb to its fingers."""
    from oppdef.hands.axis import extremes

    f = extremes("f5d6", restarts=6)
    leap = extremes("leap", restarts=6)
    assert f["opposition_floor_m"] > 0.02, f
    assert leap["opposition_floor_m"] < 0.01, leap
    assert f["aperture_m"] < leap["aperture_m"]
