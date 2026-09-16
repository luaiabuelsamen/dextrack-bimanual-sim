"""Scene state must not leak between grasps formed in the same scene.

`run_task` used to switch gravity on and leave it on. Within one G3 cell that
meant grasp #0 was formed weightless and every grasp after it was formed under
gravity, so the dataset was internally inconsistent and saved grasps did not
re-form. Found by G4's reproducibility check.
"""
import numpy as np
import pytest

from oppdef.grasping.synth import GraspScene
from oppdef.grasping.task import carry, run_task


@pytest.mark.slow
def test_run_task_restores_gravity():
    sc = GraspScene("leap", (0.022, 0.022, 0.030), kp_finger=1.0)
    before = np.array(sc.m.opt.gravity, float)
    assert np.allclose(before, 0.0), "the bench forms grasps weightless"
    sc.attempt(np.array([0.0, 0.0, 0.0, 0.01, 0.9]), do_hold=False)
    run_task(sc, carry(seg_s=0.5))
    assert np.allclose(np.array(sc.m.opt.gravity, float), before), \
        "run_task left gravity on; the next grasp in this scene would differ"


@pytest.mark.slow
def test_a_grasp_reforms_the_same_in_a_reused_scene():
    """The same parameters must give the same grasp whether or not a task has
    been run in that scene since."""
    params = np.array([0.4, -0.2, 0.3, 0.01, 0.9])
    fresh = GraspScene("leap", (0.022, 0.022, 0.030), kp_finger=1.0)
    a_fresh = fresh.attempt(params, do_hold=False)

    used = GraspScene("leap", (0.022, 0.022, 0.030), kp_finger=1.0)
    used.attempt(np.array([0.0, 0.0, 0.0, 0.02, 0.8]), do_hold=False)
    run_task(used, carry(seg_s=0.5))
    a_used = used.attempt(params, do_hold=False)

    assert a_used.valid == a_fresh.valid
    assert a_used.n_contacts == a_fresh.n_contacts
    assert a_used.epsilon == pytest.approx(a_fresh.epsilon, abs=1e-9)
