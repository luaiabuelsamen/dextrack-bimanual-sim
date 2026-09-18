"""Homotopy curriculum: solve an easier reference, deform it into the hard one.

Some references cannot be tracked from a cold start -- the object moves fast
enough that a sampling optimiser searching from the feedforward never finds a
correction before the grasp is lost. DexTrack's answer is a homotopy: solve a
deformed, easier version of the SAME reference, then walk the deformation back
to the real one, carrying the solution along.

The deformation here is toward *stationarity*. At lambda = 0 the object simply
stays where it started, which is the hold problem and is nearly always solvable
from a retargeted grasp. At lambda = 1 it is the recorded human trajectory. In
between the object follows the same path at a fraction of its amplitude, so the
contact geometry is preserved -- what changes is how much the grasp is asked to
do, not what the grasp is.

That matters: deforming by slowing time down instead would change the inertial
loads without changing the path, and a solution found there transfers to a
different problem. Amplitude deformation keeps the path and scales the demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two unit quaternions."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float) * (1.0 if q0 @ q1 >= 0 else -1.0)
    d = float(np.clip(q0 @ q1, -1.0, 1.0))
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
    return q / max(np.linalg.norm(q), 1e-12)


def deform(ref_pos: np.ndarray, ref_quat: np.ndarray, lam: float,
           anchor: int = 0):
    """Scale a reference's motion toward stationarity at `anchor`.

    lam = 0 holds the object still; lam = 1 is the recorded trajectory.
    """
    p0 = ref_pos[anchor]
    q0 = ref_quat[anchor]
    pos = p0 + lam * (ref_pos - p0)
    quat = np.stack([_slerp(q0, q, lam) for q in ref_quat])
    return pos, quat


@dataclass
class CurriculumResult:
    solved_lambda: float           # the hardest level reached
    reached_full: bool
    per_level: list = field(default_factory=list)
    actions: np.ndarray | None = None
    start: int = 0

    @property
    def summary(self) -> str:
        return (f"lambda {self.solved_lambda:.2f}"
                + ("  (full reference)" if self.reached_full else "  (partial)"))


def solve_with_curriculum(rt, levels=(0.25, 0.5, 0.75, 1.0), start=None,
                          horizon=4, samples=24, drop_m=0.10, seed=0,
                          verbose=False):
    """Track a reference by walking a homotopy up to it.

    Each level is solved by MPPI warm-started from the level below, so the
    correction found for an easy deformation is the starting plan for a harder
    one. A level that ends with the object dropped stops the walk: the last
    level that held is what this reference can currently be tracked at, and
    reporting that is more useful than reporting a failure.
    """
    from handsim.human import track as T

    true_pos, true_quat = rt.ref_pos.copy(), rt.ref_quat.copy()
    if start is None:
        gf = rt.grasp_frames()
        start = int(gf[0]) if len(gf) else 0

    out = CurriculumResult(solved_lambda=0.0, reached_full=False, start=start)
    warm = None
    try:
        for lam in levels:
            rt.ref_pos, rt.ref_quat = deform(true_pos, true_quat, lam, anchor=start)
            rt._palm_obj = None          # palm poses are cached per reference
            rt.P, rt.Q, rt.vals = T.feedforward_se3(
                rt.fit, rt.tr.q, rt.ref_pos, rt.ref_quat)
            roll, acts = T.mppi_track(rt, horizon=horizon, samples=samples,
                                      start=start, seed=seed, warm=warm)
            held = not roll.dropped
            out.per_level.append({
                "lambda": float(lam), "held": bool(held),
                "mean_mm": float(roll.pos_err.mean() * 1000),
                "final_mm": float(roll.pos_err[-1] * 1000),
                "frames": int(roll.steps)})
            if verbose:
                print(f"    lambda={lam:.2f}  {'held' if held else 'DROPPED'}  "
                      f"mean {roll.pos_err.mean()*1000:7.1f} mm", flush=True)
            if not held:
                break
            out.solved_lambda = float(lam)
            out.actions = acts
            warm = acts                   # carry the solution to the next level
            if lam >= 1.0:
                out.reached_full = True
    finally:
        rt.ref_pos, rt.ref_quat = true_pos, true_quat
        rt._palm_obj = None
        rt.P, rt.Q, rt.vals = T.feedforward_se3(
            rt.fit, rt.tr.q, rt.ref_pos, rt.ref_quat)
    return out
