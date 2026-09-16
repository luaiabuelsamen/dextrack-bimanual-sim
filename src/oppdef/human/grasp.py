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
    equilibrium: float = float("nan")   # net force as a multiple of own weight
    cost: float = float("nan")          # the search objective this minimised


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
               shrink: float = 0.6, rounds: int = 2,
               w_eq: float = 0.0) -> GraspFit:
    """Search near `q0` for a configuration that holds the object.

    A short cross-entropy search: sample wrist offsets, keep the ones that hold
    longest, shrink around them. Two rounds is enough to matter and cheap enough
    to run per reference; the first candidate tried is always the unperturbed
    retarget, so this can only improve on it.

`w_eq` adds the equilibrium residual to that score. **Default 0, and that
    is a result rather than a default** -- the third term in this repository to
    be shipped off after measurement, alongside `W_JOINT` and `W_MID`.

    The motivation was sound and the diagnosis behind it stands: drop distance
    alone cannot tell a grasp from a burial, and a search asked for nothing but
    stillness will take burial, because burying the hand is an excellent way to
    stop an object moving. Measured across a 40-reference sweep, drop-only
    scoring accepts all of these as holds --

        apple_eat_1       drop  5.2 mm    9 contacts     875 N
        bowl_drink_1      drop  1.6 mm   94 contacts  35,521 N

    -- and calls the 35 kN one the better hold, because it moves less. A peer
    session independently measured this search re-burying the hand to ~20 mm
    and 13.6 kN wherever the fit started, which is how a policy trained on a
    buried initial condition kept scoring well.

    What does not work is fixing it with THIS quantity. Two measurements:

    1. `equilibrium_residual` discriminates PLACEMENT, not steady state. Its
       142-337x burial readings were all taken on a placed reset state before
       any stepping. A settled buried object has cancelling constraint forces
       and therefore near-zero NET force: the 35 kN bowl above reads 0.00 once
       it has settled. Across 16 candidates the residual's median is 1.000 --
       free fall -- and it correlates -0.56 to -0.68 with grip force and +0.47
       to +0.58 with drop. It is largely a non-contact detector here, and drop
       distance already detects non-contact.
    2. The effect on the search is real but weak and inconsistent. Six paired
       runs, w_eq 0 against 0.05, same seeds, grip force in newtons:

           airplane_fly_1   385 -> 6.9     1,968 -> 499     663 -> 684
           bowl_drink_1  35,521 -> 11,359  7,324 -> 9,219  16,301 -> 10,467

       Four of six improve, geometric mean ratio 0.33, one is worse, and the
       bowl never leaves ~10 kN -- still burial. Drop distance got worse in all
       six. The 385 -> 6.9 N row read like a 56x win on its own and is not one;
       it is the tail of a noisy distribution.

    So the term is available and off. The diagnosis it came from is the durable
    part: the search needs a quantity that separates a grasp from a burial in a
    SETTLED state, and net force is not it, because burial cancels. `equilibrium`
    is recorded on every `GraspFit` regardless, since it costs nothing and
    reading it is how the burial was caught.
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(6)
    sig = np.array([sigma_pos] * 3 + [sigma_rot] * 3)
    best = None
    tried = 0
    obj_bid = int(env.sc.model.jnt_bodyid[env.obj_jid])

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
            eq = T.equilibrium_residual(env.sc, obj_bid) if w_eq else 0.0
            if not np.isfinite(eq):
                eq = 1e3
            cost = float(res.drop_m + w_eq * eq)
            scored.append((cost, dlt, q, res, res.drop_m, eq))
            if best is None or cost < best[0]:
                best = (cost, dlt, q, res, res.drop_m, eq)
        scored.sort(key=lambda t: t[0])
        elite = np.array([t[1] for t in scored[:max(2, samples // 4)]])
        mean, sig = elite.mean(0), np.maximum(elite.std(0), sig * shrink)

    cost, dlt, q, res, drop, eq = best
    env.hold(q, seconds=0.1, settle=0.1, grip=grip if grip else None)
    grip_n, ncon = T.total_grip(env.sc)
    return GraspFit(q=q, ctrl=env.sc.data.ctrl.copy(), held=bool(drop < 0.05),
                    drop_m=float(drop), n_contact=int(ncon),
                    grip_n=float(grip_n), offset=np.asarray(dlt), tried=tried,
                    equilibrium=float(eq), cost=float(cost))
