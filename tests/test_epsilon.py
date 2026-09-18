"""The epsilon instrument's analytic ground truth.

These cases have known answers independent of any simulation, which is why they
are the first thing that must keep passing: everything downstream reads epsilon.
"""
import numpy as np
import pytest

from handsim.grasping.epsilon import epsilon_from_wrenches, wrench_set

LAM = 0.05
COM = np.zeros(3)


def eps(P, N, mu):
    return epsilon_from_wrenches(wrench_set(P, N, [mu] * len(P), COM, LAM))


def test_two_antipodal_points_are_not_force_closure():
    """Neither contact can resist torque about the line joining them."""
    assert eps([[0.04, 0, 0], [-0.04, 0, 0]], [[-1, 0, 0], [1, 0, 0]], 1.0) == 0.0


def test_spread_opposed_pads_are_force_closure():
    P, N = [], []
    for sx in (+1, -1):
        for dy, dz in ((0, 0.03), (0.03, -0.02), (-0.03, -0.02)):
            P.append([sx * 0.04, dy, dz]); N.append([-sx, 0, 0])
    assert eps(P, N, 1.0) > 0.0


def test_frictionless_opposed_pads_are_not_force_closure():
    P, N = [], []
    for sx in (+1, -1):
        for dy, dz in ((0, 0.03), (0.03, -0.02), (-0.03, -0.02)):
            P.append([sx * 0.04, dy, dz]); N.append([-sx, 0, 0])
    assert eps(P, N, 0.0) == 0.0


def test_codirectional_contacts_are_never_force_closure():
    """A hand that cannot oppose gets exactly this contact set."""
    P = [[0, 0, -0.04], [0.03, 0, -0.04], [-0.03, 0, -0.04], [0, 0.03, -0.04]]
    assert eps(P, [[0, 0, 1]] * 4, 1.0) == 0.0


def test_epsilon_is_monotone_in_friction():
    """Cone edges are scaled to unit NORMAL component for this reason; scaling
    them to unit length instead makes epsilon fall as mu rises."""
    P, N = [], []
    for sx in (+1, -1):
        for dy, dz in ((0, 0.03), (0.03, -0.02), (-0.03, -0.02)):
            P.append([sx * 0.04, dy, dz]); N.append([-sx, 0, 0])
    vals = [eps(P, N, mu) for mu in (0.2, 0.5, 1.0, 2.0)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), vals
