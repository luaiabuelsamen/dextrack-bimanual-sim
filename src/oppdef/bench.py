"""The shared benchmark cell: one hand, one object, one comparison.

Both live comparisons (`experiments/matched.py`, `experiments/g1.py`) need the
same three things: a scene, the retargeted finger pose that the human
demonstration prescribes, and a way to score how faithfully a settled pose
reproduces that demonstration. Those lived in an experiment script, so a live
experiment imported a withdrawn one to get at them. They belong in the library.
"""
from __future__ import annotations

import numpy as np
import mujoco

from oppdef.grasping.synth import GraspScene
from oppdef.data import SyntheticSource
from oppdef.hands.tips import tip_offset, tip_points
from oppdef.grasping.retarget_pose import (retargeter_for, transform_ref, tip_graph, KEYPOINT)


def keypoint_pose(hand_key, width):
    """Finger angles that best reproduce the demonstration's fingertip geometry.

    Returns (joint name -> angle, achieved inter-fingertip error in metres).
    """
    rt = retargeter_for(hand_key, free_base=False)
    ref = list(SyntheticSource(widths=(width,), n_per=1))[0]
    V, obj, half, sc, _demo, _wrist = transform_ref(rt, ref, width=width)
    r = rt.fit(V, half, objective=KEYPOINT, obj_pos=obj, restarts=10, scale=sc)
    out = {}
    for i, j in enumerate(rt.jids):
        n = mujoco.mj_id2name(rt.m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        out[n] = float(r.q[i])
    return out, float(r.keypoint_err_m)


def reference_graph(hand_key, width):
    """The demonstration's inter-fingertip vectors, in this hand's frame."""
    rt = retargeter_for(hand_key, free_base=False)
    ref = list(SyntheticSource(widths=(width,), n_per=1))[0]
    V, _obj, _half, sc, _demo, _wrist = transform_ref(rt, ref, width=width)
    return rt.target_graph(V, sc)


class Cell:
    """One (hand, object) scene plus what every comparison arm shares."""

    def __init__(self, hand_key, width, mass=0.05, kp=1.0, n_hands=1):
        self.scene = GraspScene(hand_key, (width / 2,) * 3, mass=mass,
                                n_hands=n_hands, kp_finger=kp)
        s = self.scene
        self.hand_key, self.width, self.mass = hand_key, width, mass
        self.pfx = s.prefixes[0]
        self.tip_b = s.tip_bids[self.pfx]
        self.tip_off = [tip_offset(s.m, b) for b in self.tip_b]
        self.palm_b = s.palm_bid[self.pfx]
        self.names = [n[len(self.pfx):] for n in s.finger[self.pfx]]
        self.base = {n[len(self.pfx):]: t
                     for n, (_q, _a, t) in s.finger[self.pfx].items()}
        self.lo, self.hi = {}, {}
        for n in self.names:
            j = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{self.pfx}{n}")
            a, b = s.m.jnt_range[j]
            self.lo[n], self.hi[n] = (a, b) if b > a else (-np.pi, np.pi)
        self.A_ref = reference_graph(hand_key, width)

    def achieved_graph(self):
        """Inter-fingertip vectors of the settled pose, in the PALM frame.

        Palm frame, not world: the base is free, so a world-frame graph changes
        when the hand is merely rotated and the fidelity score would then be
        measuring placement instead of finger shape.
        """
        s = self.scene
        P = tip_points(s.m, s.d, self.tip_b, self.tip_off)
        R = s.d.xmat[self.palm_b].reshape(3, 3)
        G, _pairs = tip_graph((P - s.d.xpos[self.palm_b]) @ R)
        return G

    def fidelity(self):
        """Negative mean inter-fingertip vector error, metres (higher better)."""
        G = self.achieved_graph()
        n = min(len(G), len(self.A_ref))
        return -float(np.linalg.norm(G[:n] - self.A_ref[:n], axis=1).mean())

    def targets(self, vec):
        return {n: float(np.clip(vec[i], self.lo[n], self.hi[n]))
                for i, n in enumerate(self.names)}
