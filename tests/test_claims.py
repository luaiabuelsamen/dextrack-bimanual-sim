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
    from handsim.envs.bimanual import build
    from handsim.sim.port import strip_visual, rollout_cpu

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
    from handsim.control.expert import Expert

    two = Expert(two_handed=True).run(verbose=False)
    one = Expert(two_handed=False).run(verbose=False)
    assert two["success"], f"expert failed: {two}"
    assert not one["success"], f"one-handed control succeeded: {one}"
    assert two["peg_out_m"] >= 0.08
    assert one["box_z_rise_m"] > two["box_z_rise_m"] + 0.02, (
        "the control should fail by LIFTING the base")


@pytest.mark.slow
def test_opposition_floor_is_the_corrected_one():
    """RESTATED 2026-09-13. This test used to assert that f5d6 was the only hand
    that could not bring its thumb to its fingers, with a floor above 2 cm.

    That was measured at distal joint origins, which the terminal joints rotate
    about, with the hand's mimic couplings dropped. Corrected -- fingertips
    derived from each distal link's collision geometry, couplings enforced,
    self-collision scoped to the fingers, distance to the NEAREST fingertip --
    all four hands oppose within 2.8 mm of one another and f5d6 opposes better
    than LEAP. See NOTES.md 2026-09-13.
    """
    from handsim.hands.axis import opposition_axis

    f = opposition_axis("f5d6", restarts=8)
    leap = opposition_axis("leap", restarts=8)
    # the retracted claim was f.floor > 2 cm; it is under 1 cm
    assert f.floor_m < 0.01, f
    assert leap.floor_m < 0.01, leap
    # what DOES separate f5d6 is its aperture, not its floor
    assert f.aperture_m < leap.aperture_m
