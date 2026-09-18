"""Every embodiment this machine claims to have must actually build."""
import numpy as np
import mujoco
import pytest

from handsim.embodiment import HANDS, ARMS, make, inventory, hand_xml


@pytest.mark.slow
def test_every_registered_hand_and_arm_loads():
    bad = [(k, st) for _, k, _, _, _, st in inventory() if st != "ok"]
    assert not bad, f"registry lists models that do not load: {bad}"


@pytest.mark.slow
@pytest.mark.parametrize("hand", ["leap", "allegro", "shadow"])
def test_floating_hand_builds(hand):
    e = make(hand=hand)
    assert e.model.nq >= HANDS[hand].dof - 4
    assert all(b >= 0 for b in e.tip_bid[""]), "fingertip bodies not found"


@pytest.mark.slow
def test_two_hands_are_namespaced_and_do_not_collide_by_name():
    e = make(hand="leap", count=2, positions=[[0, -0.2, 0.3], [0, 0.2, 0.3]])
    assert set(e.palm_bid) == {"rh_", "lh_"}
    assert e.palm_bid["rh_"] != e.palm_bid["lh_"]
    assert e.model.nq == 2 * HANDS["leap"].dof


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["ur5e", "xarm7"])
def test_hand_mounts_on_arm_at_its_end_effector(arm):
    """`arm=` must mean MOUNTED, not 'both in the same scene'. MuJoCo does not
    namespace ASSETS on attach, so this also covers the material-name collision
    that made every hand+arm build fail."""
    e = make(hand="leap", arm=arm)
    d = mujoco.MjData(e.model)
    mujoco.mj_forward(e.model, d)
    palm = d.xpos[e.palm_bid["hand_"]]
    assert np.linalg.norm(palm) > 0.05, f"hand sits at the origin, not mounted: {palm}"
    assert e.model.nu > ARMS[arm].dof, "arm actuators only -- the hand is not driven"
