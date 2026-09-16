"""Regression tests for retargeting.

Every case here is a bug this module actually shipped. Each one returned
plausible numbers while being wrong, which is why they are tests and not
comments: a silent wrong answer is the failure mode this file exists to catch.
"""
import numpy as np
import pytest
import mujoco

from oppdef.grasping.retarget_pose import (correspond, tip_graph, align_reference, box_surface,
                             geometric_epsilon, transform_ref, retargeter_for,
                             grasp_centre)
from oppdef.data import SyntheticSource


def _ref():
    return list(SyntheticSource(widths=(0.05,), n_per=1))[0]


# -- correspondence ---------------------------------------------------------
def test_correspond_pairs_thumb_to_thumb():
    """A 5-tip reference onto a 4-tip hand must drop a FINGER, not the thumb.

    Taking the first 4 of each pairs the reference's little finger with the
    hand's thumb; the frame fit then reports a believable number for a
    correspondence that is wrong.
    """
    ri, hi = correspond(5, 4)
    assert ri.tolist() == [0, 1, 2, 4]      # thumb (index 4) kept
    assert hi.tolist() == [0, 1, 2, 3]      # thumb (index 3) kept
    assert ri[-1] == 4 and hi[-1] == 3


def test_correspond_is_symmetric_in_count():
    for n_ref, n_hand in ((5, 5), (5, 4), (4, 5)):
        ri, hi = correspond(n_ref, n_hand)
        assert len(ri) == len(hi) == min(n_ref, n_hand)
        assert ri[-1] == n_ref - 1 and hi[-1] == n_hand - 1


def test_synthetic_source_puts_thumb_last():
    """The ordering convention `correspond` relies on."""
    ref = _ref()
    tips = ref.fingertips
    # thumb is the lone tip on the +x face; the fingers share -x
    assert tips[-1][0] > 0
    assert np.all(tips[:-1, 0] < 0)


# -- the frame fit ----------------------------------------------------------
def test_tip_graph_is_translation_invariant():
    tips = np.random.default_rng(0).normal(size=(5, 3))
    a, _ = tip_graph(tips)
    b, _ = tip_graph(tips + np.array([1.0, -2.0, 3.0]))
    assert np.allclose(a, b)


def test_align_recovers_a_known_rotation():
    rng = np.random.default_rng(1)
    hand = rng.normal(size=(5, 3))
    th = 0.7
    R0 = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    ref = hand @ R0.T                      # so ref @ R0 == hand
    R = align_reference(ref, hand)
    assert np.allclose(ref @ R, hand, atol=1e-8)


def test_align_uses_the_palm_to_pick_a_side():
    """Fingertips are near-planar, so inter-tip vectors alone cannot tell which
    side of that plane the palm is on -- the DoF that decides whether the
    fingers point at the object or away from it."""
    tips = np.array([[0.0, 0, 0.03], [0.0, 0, 0.01], [0.0, 0, -0.01],
                     [0.0, 0, -0.03], [0.03, 0, 0.0]])
    front, back = np.array([0.0, -0.09, 0]), np.array([0.0, 0.09, 0])
    R_f = align_reference(tips, tips, ref_wrist=front, hand_wrist=front)
    R_b = align_reference(tips, tips, ref_wrist=front, hand_wrist=back)
    assert np.allclose(R_f, np.eye(3), atol=1e-6)
    # palm on the opposite side must NOT produce the identity
    assert not np.allclose(R_b, np.eye(3), atol=1e-2)


# -- geometry ---------------------------------------------------------------
def test_box_surface_inside_and_outside():
    half = np.array([0.02, 0.02, 0.02])
    q, n, d = box_surface(np.array([0.05, 0, 0]), half)
    assert d == pytest.approx(0.03) and np.allclose(n, [1, 0, 0])
    q, n, d = box_surface(np.array([0.019, 0, 0]), half)
    assert d < 0 and np.allclose(n, [1, 0, 0])   # inside: negative distance


def test_epsilon_is_zero_when_nothing_touches():
    tips = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    eps, n = geometric_epsilon(tips, (0.02, 0.02, 0.02))
    assert n == 0 and eps == 0.0


def test_demonstration_is_itself_a_grasp():
    """If the reference scores zero, a hand scoring zero means nothing.

    The first comparison reported eps=0 for every hand and every objective, and
    without this check there was no way to see that the question was ill-posed
    rather than the hands bad.
    """
    for w in (0.04, 0.05, 0.07):
        ref = list(SyntheticSource(widths=(w,), n_per=1))[0]
        eps, n = geometric_epsilon(ref.fingertips, ref.obj_half,
                                   obj_pos=ref.obj_pos)
        assert n >= 3, f"reference grasp at w={w} makes only {n} contacts"
        assert eps > 0.05, f"reference grasp at w={w} has eps={eps}"


# -- placement --------------------------------------------------------------
@pytest.mark.slow
def test_object_is_placed_where_the_hand_can_reach():
    """A retargeter moves joints, not the world. Placed at the origin, every
    hand scored zero contacts and the comparison measured nothing."""
    rt = retargeter_for("leap")
    ref = _ref()
    _V, obj, half, _s, _demo, wrist = transform_ref(rt, ref, width=0.05)
    tips, _w = rt.forward(np.clip(rt.q_closure, rt.lo, rt.hi))
    assert np.linalg.norm(obj - grasp_centre(rt)) < 1e-9
    # the object must sit within the closed hand's own tip envelope
    assert np.linalg.norm(obj - tips.mean(0)) < 0.05
    # and the palm target must be reachable, not further than the fingers are long
    reach = np.linalg.norm(tips - _w[None, :], axis=1).mean()
    assert np.linalg.norm(wrist - obj) <= reach * 1.5


@pytest.mark.slow
def test_base_hinges_are_specified_in_the_unit_the_spec_compiles():
    """These specs compile angles in DEGREES.

    `_add_base_dof` wrote `range=[-3.2, 3.2]` intending radians, against an
    actuator `ctrlrange` of +/-3.2 radians. It compiled to 3.2 DEGREES, so every
    commanded wrist rotation past 3.2 degrees saturated at the joint limit in
    silence -- and `expert.best_rz`, which sweeps rz across the full circle, was
    choosing among poses the hand could not take. The number to guard is the
    compiled one, in radians, because that is where the bug was visible.
    """
    rt = retargeter_for("leap", free_base=True)
    names = [mujoco.mj_id2name(rt.m, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in rt.jids]
    assert {"x", "y", "z", "rx", "ry", "rz"} <= set(names)
    for i, n in enumerate(names):
        if n in ("rx", "ry", "rz"):
            lo, hi = rt.m.jnt_range[rt.jids[i]]
            assert hi == pytest.approx(np.pi, rel=1e-3), (
                f"{n} compiled to +/-{hi:.4f} rad; 0.0559 means degrees were "
                f"written where radians were meant")
            assert rt.hi[i] == pytest.approx(np.pi)


@pytest.mark.slow
def test_bimanual_wrist_actually_reaches_a_commanded_rotation():
    """The end-to-end version of the same defect: command, then measure."""
    from oppdef.envs.bimanual import BimanualBox
    e = BimanualBox()
    j = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_JOINT, "rh_rz")
    mujoco.mj_resetData(e.m, e.d)
    c = np.zeros(e.m.nu)
    c[e.act["rh_rz_act"]] = 1.0
    for _ in range(800):
        e.d.ctrl[:] = c
        mujoco.mj_step(e.m, e.d)
    assert e.d.qpos[e.m.jnt_qposadr[j]] == pytest.approx(1.0, abs=0.02)


@pytest.mark.slow
def test_every_hand_builds_with_a_floating_base():
    """Shadow's palm carries a 2-DoF wrist and f5d6's sits inside a whole robot,
    so bolting six more DoF onto the palm exceeded MuJoCo's 6-per-body limit.
    The base needs its own carrier body."""
    from oppdef.embodiment import HANDS
    for hk in HANDS:
        if hk == "leap_left":
            continue
        rt = retargeter_for(hk, free_base=True)
        names = [mujoco.mj_id2name(rt.m, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in rt.jids]
        assert {"x", "y", "z", "rx", "ry", "rz"} <= set(names), hk
        # and the optimiser must not be handed the rest of the robot: f5d6's
        # file is the whole Vega, head and BOTH hands included. Which prefix is
        # foreign depends on the side -- the left hand's own joints are L_, and
        # asserting against L_ unconditionally is what made this test fail on
        # f5d6_left rather than any defect in the hand.
        own = getattr(HANDS[hk], "joint_prefix", None) or ""
        foreign = {"R_": "L_", "L_": "R_"}.get(own)
        bad = [p for p in (foreign, "head") if p]
        assert not any(n.startswith(p) for p in bad for n in names), hk
        if own:
            assert any(n.startswith(own) for n in names), f"{hk} exposes no {own} joint"
        assert len(rt.jids) <= 32, f"{hk} exposes {len(rt.jids)} joints"
