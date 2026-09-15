"""Stage 3 initialisation: synthesise a grasp NEAR the human's contact set.

G5's pre-registered decision rule fires here. The raw retarget holds the object
in only 25.5% of frames, and its failures make 0.1 contacts at reset against
17.4 for its successes -- they are not slipping grasps, they are poses that
never touch. So the retarget becomes a PRIOR on where to search, not the state
the tracker starts from.

Closing the fingers from a pose that is not around the object does not fix
that, and this was measured rather than assumed: adding grip establishment to
G5's own protocol moved the hold rate from 0.288 to 0.276 on a 25-sequence
subsample, which is nothing. The hand has to be repositioned as well as closed.

So this searches a small neighbourhood of the retargeted wrist pose, scoring
each candidate by whether it actually holds the object in simulation. That is
the same standard the rest of this repository uses: a grasp is what survives a
physical test, not what scores well on a geometric one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from oppdef.human import track as T


@dataclass
class GraspFit:
    q: np.ndarray               # joint configuration, the scene's order
    ctrl: np.ndarray            # servo targets that hold it
    held: bool
    drop_m: float
    n_contact: int
    grip_n: float
    offset: np.ndarray          # wrist perturbation that produced it (6,)
    tried: int


def _perturbed(env, q0, delta):
    """q0 with its floating-base DoF shifted by `delta` (3 pos + 3 rot)."""
    q = np.array(q0, float)
    names = [mujoco.mj_id2name(env.sc.model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in env.sc.jids]
    for i, n in enumerate(names):
        if n in ("x", "y", "z"):
            q[i] += delta[{"x": 0, "y": 1, "z": 2}[n]]
        elif n in ("rx", "ry", "rz"):
            q[i] += delta[3 + {"rx": 0, "ry": 1, "rz": 2}[n]]
    return q


def synthesize(env, q0, samples: int = 24, sigma_pos: float = 0.012,
               sigma_rot: float = 0.10, seconds: float = 0.5,
               grip: float | None = 8.0, seed: int = 0,
               shrink: float = 0.6, rounds: int = 2) -> GraspFit:
    """Search near `q0` for a configuration that holds the object.

    A short cross-entropy search: sample wrist offsets, keep the ones that hold
    longest, shrink around them. Two rounds is enough to matter and cheap enough
    to run per reference; the first candidate tried is always the unperturbed
    retarget, so this can only improve on it.
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(6)
    sig = np.array([sigma_pos] * 3 + [sigma_rot] * 3)
    best = None
    tried = 0

    for r in range(rounds):
        deltas = rng.normal(size=(samples, 6)) * sig + mean
        if r == 0:
            deltas[0] = 0.0                     # the retarget itself
        scored = []
        for dlt in deltas:
            q = _perturbed(env, q0, dlt)
            res = env.hold(q, seconds=seconds, settle=0.1,
                           grip=grip if grip else None)
            tried += 1
            scored.append((res.drop_m, dlt, q, res))
            if best is None or res.drop_m < best[0]:
                best = (res.drop_m, dlt, q, res)
        scored.sort(key=lambda t: t[0])
        elite = np.array([t[1] for t in scored[:max(2, samples // 4)]])
        mean, sig = elite.mean(0), np.maximum(elite.std(0), sig * shrink)

    drop, dlt, q, res = best
    env.hold(q, seconds=0.1, settle=0.1, grip=grip if grip else None)
    grip_n, ncon = T.total_grip(env.sc)
    return GraspFit(q=q, ctrl=env.sc.data.ctrl.copy(), held=bool(drop < 0.05),
                    drop_m=float(drop), n_contact=int(ncon),
                    grip_n=float(grip_n), offset=np.asarray(dlt), tried=tried)
