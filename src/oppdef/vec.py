"""Batched environments: the same scene in N worlds, behind one interface.

Nothing in this project ran more than one world at a time. The expert, BC and
every sweep stepped a single `MjData` in Python, which is why a 24-demo
collection takes minutes and why no RL was attempted -- not because RL was ruled
out, but because there was no way to feed it.

Three backends, one API:

    CpuVec    N MjData stepped by `mujoco.rollout`, which releases the GIL and
              threads across worlds. Always available; the reference.
    MjxVec    `jax.vmap` over `mjx.step` on the GPU.
    WarpVec   `mujoco_warp`, N worlds in one kernel launch.

The reference matters more than the speed. A GPU backend that disagrees with
CPU MuJoCo is not a faster simulator, it is a different one, and this project
has already retracted a set of MJX numbers obtained from a scene whose contact
physics had been substituted to make it run. So `parity()` is part of the
module, not a script someone might run: every backend reports its divergence
from CPU on identical controls, and that number is what any transfer claim has
to be stated against.

Backends are constructed from a MODEL, so anything that compiles -- the
bimanual scene, a hand bench, an embodiment from `oppdef.embodiment` -- can be
batched without a bespoke wrapper.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class Batch:
    """Batched state. `qpos`/`qvel` are (N, nq) / (N, nv)."""
    qpos: np.ndarray
    qvel: np.ndarray
    sensordata: np.ndarray | None = None

    @property
    def n(self):
        return len(self.qpos)

    def flat(self):
        return np.concatenate([self.qpos, self.qvel], axis=1)


class VecEnv:
    """N copies of one model, stepped together."""
    backend = "abstract"

    def __init__(self, model, n):
        self.m, self.n = model, int(n)

    def reset(self, qpos=None, qvel=None):
        raise NotImplementedError

    def step(self, ctrl):
        raise NotImplementedError

    def state(self) -> Batch:
        raise NotImplementedError

    def close(self):
        pass

    # -- shared helpers ----------------------------------------------------
    def clamp(self, ctrl):
        """Apply the model's control limits, as a real actuator would."""
        m = self.m
        lim = m.actuator_ctrllimited.astype(bool)
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        return np.where(lim[None, :], np.clip(ctrl, lo[None, :], hi[None, :]),
                        ctrl)


class CpuVec(VecEnv):
    """N worlds on the CPU, threaded by `mujoco.rollout`.

    `rollout` is built for open-loop batches, but stepping it one step at a time
    keeps the threading while leaving the control in Python -- which is what a
    closed-loop policy needs. It releases the GIL, so this actually scales
    across cores rather than pretending to.
    """
    backend = "cpu"

    def __init__(self, model, n, nthread=None, deterministic=True):
        super().__init__(model, n)
        from mujoco import rollout
        # `rollout` wants one MjData per THREAD, not per world: the worlds are
        # a work queue fed through that pool. Handing it one per world raises
        # "Length of data not equal to nthread".
        self.nthread = int(nthread or min(self.n, 8))
        self._roll = rollout.Rollout(nthread=self.nthread)
        self.datas = [mujoco.MjData(model) for _ in range(self.nthread)]
        # Worlds are pooled across threads, so in principle a step could
        # inherit the warm-start left by the previous occupant of its thread.
        # Measured on the bimanual scene it does not: with the pool sized
        # correctly, four worlds under identical controls agree BITWISE
        # (0.0e+00) whether or not the warm-start is reset, at 1 thread and at
        # 4. The flag stays because the exposure is real and cheap to close,
        # but it is not what makes the backend reproducible -- the pool size
        # was.
        self.deterministic = bool(deterministic)
        self.nstate = mujoco.mj_stateSize(model,
                                          mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.reset()

    def reset(self, qpos=None, qvel=None):
        # The world states and the thread pool are different things and must not
        # be built from the same list: `self.datas` is nthread long, so seeding
        # the batch from it produced an initial_state with nthread rows and
        # rollout inferred the batch size from that instead of from n.
        d = mujoco.MjData(self.m)
        states = []
        for i in range(self.n):
            mujoco.mj_resetData(self.m, d)
            if qpos is not None:
                d.qpos[:] = np.asarray(qpos)[i] if np.ndim(qpos) == 2 else qpos
            if qvel is not None:
                d.qvel[:] = np.asarray(qvel)[i] if np.ndim(qvel) == 2 else qvel
            mujoco.mj_forward(self.m, d)
            states.append(self._get(d))
        self._state = np.stack(states)
        return self.state()

    def _get(self, d):
        s = np.empty(self.nstate)
        mujoco.mj_getState(self.m, d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return s

    def step(self, ctrl):
        ctrl = self.clamp(np.asarray(ctrl, float).reshape(self.n, self.m.nu))
        ws = (np.zeros((self.n, self.m.nv)) if self.deterministic else None)
        # One model, not one per world: rollout infers the batch size from the
        # longest argument, so a list of N models makes it read N where the
        # data pool says nthread, and it raises as soon as N != nthread.
        state, _sens = self._roll.rollout(
            self.m, self.datas, self._state,
            ctrl[:, None, :], nstep=1, initial_warmstart=ws)
        self._state = np.asarray(state)[:, -1, :]
        return self.state()

    def state(self):
        nq, nv = self.m.nq, self.m.nv
        # FULLPHYSICS layout: time, qpos, qvel, act, plugin state
        qpos = self._state[:, 1:1 + nq]
        qvel = self._state[:, 1 + nq:1 + nq + nv]
        return Batch(qpos=qpos.copy(), qvel=qvel.copy())


class MjxVec(VecEnv):
    """N worlds on the GPU via `jax.vmap(mjx.step)`."""
    backend = "mjx"

    def __init__(self, model, n):
        super().__init__(model, n)
        import jax
        import mujoco.mjx as mjx
        self.jax, self.mjx = jax, mjx
        self.mx = mjx.put_model(model)
        self._step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        self.reset()

    def reset(self, qpos=None, qvel=None):
        jax, mjx = self.jax, self.mjx
        d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, d)
        dx = mjx.put_data(self.m, d)
        self.dx = jax.tree.map(
            lambda x: np.broadcast_to(x, (self.n,) + np.shape(x)).copy()
            if np.ndim(x) >= 0 else x, dx)
        self.dx = jax.tree.map(lambda x: self.jax.numpy.asarray(x), self.dx)
        if qpos is not None:
            self.dx = self.dx.replace(
                qpos=self.jax.numpy.asarray(
                    np.broadcast_to(qpos, (self.n, self.m.nq))))
        if qvel is not None:
            self.dx = self.dx.replace(
                qvel=self.jax.numpy.asarray(
                    np.broadcast_to(qvel, (self.n, self.m.nv))))
        return self.state()

    def step(self, ctrl):
        ctrl = self.clamp(np.asarray(ctrl, float).reshape(self.n, self.m.nu))
        self.dx = self.dx.replace(ctrl=self.jax.numpy.asarray(ctrl))
        self.dx = self._step(self.mx, self.dx)
        return self.state()

    def state(self):
        return Batch(qpos=np.asarray(self.dx.qpos),
                     qvel=np.asarray(self.dx.qvel))


class WarpVec(VecEnv):
    """N worlds in one `mujoco_warp` launch."""
    backend = "warp"

    def __init__(self, model, n, nconmax=256, njmax=512):
        super().__init__(model, n)
        from oppdef.sim import warp_fix
        warp_fix.apply(verbose=False)
        import warp as wp
        import mujoco_warp as mjwarp
        self.wp, self.mjwarp = wp, mjwarp
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        # njmax defaults to 64 and silently DROPS constraints past it; a real
        # episode of the bimanual scene peaks at nefc 167. See NOTES.
        #
        # `nconmax` and `njmax` are PER WORLD -- `naconmax` is the separate
        # total. Multiplying them by the world count (as this did) asks for
        # n^2 capacity: at n=256 that is 256*256*256 = 16.8M contacts instead
        # of 65k, which inflates memory enormously and would make GPU scaling
        # look far worse than it is. Reported by review, 2026-09-12.
        self.mw = mjwarp.put_model(model)
        self.nconmax, self.njmax = int(nconmax), int(njmax)
        self.dw = mjwarp.put_data(model, d, nworld=self.n,
                                  nconmax=self.nconmax, njmax=self.njmax)

    def reset(self, qpos=None, qvel=None):
        """Per-world reset, matching the CPU backend's contract.

        This previously flattened whatever it was given and used the FIRST
        world's state for all of them, so independent initial-state
        randomisation silently collapsed to a single state -- and it rebuilt
        with hardcoded capacities instead of the ones the constructor chose.
        Reported by review, 2026-09-12.
        """
        d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, d)
        self.dw = self.mjwarp.put_data(self.m, d, nworld=self.n,
                                       nconmax=self.nconmax, njmax=self.njmax)
        if qpos is not None or qvel is not None:
            qp = self.dw.qpos.numpy().reshape(self.n, self.m.nq)
            qv = self.dw.qvel.numpy().reshape(self.n, self.m.nv)
            if qpos is not None:
                qp[:] = np.broadcast_to(np.asarray(qpos, float),
                                        (self.n, self.m.nq))
            if qvel is not None:
                qv[:] = np.broadcast_to(np.asarray(qvel, float),
                                        (self.n, self.m.nv))
            self.dw.qpos = self.wp.array(qp.astype(np.float32), dtype=float)
            self.dw.qvel = self.wp.array(qv.astype(np.float32), dtype=float)
        return self.state()

    def step(self, ctrl):
        ctrl = self.clamp(np.asarray(ctrl, float).reshape(self.n, self.m.nu))
        self.dw.ctrl = self.wp.array(ctrl.astype(np.float32), dtype=float)
        self.mjwarp.step(self.mw, self.dw)
        return self.state()

    def state(self):
        return Batch(qpos=self.dw.qpos.numpy().reshape(self.n, self.m.nq),
                     qvel=self.dw.qvel.numpy().reshape(self.n, self.m.nv))


BACKENDS = {"cpu": CpuVec, "mjx": MjxVec, "warp": WarpVec}


def make_vec(model, n, backend="cpu", **kw):
    return BACKENDS[backend](model, n, **kw)


# --------------------------------------------------------------------------
# the part that matters: does the fast backend agree with the reference?
# --------------------------------------------------------------------------
def parity(model, ctrl_seq, backend="mjx", n=2, **kw):
    """Step CPU and `backend` through identical controls; return the divergence.

    Reported, never tuned away. A GPU backend that disagrees with CPU MuJoCo is
    a different simulator, and the size of the disagreement is the transfer gap
    for anything trained on it.
    """
    ctrl_seq = np.asarray(ctrl_seq, float)
    ref = CpuVec(model, n)
    alt = make_vec(model, n, backend=backend, **kw)
    ref.reset(); alt.reset()
    errs = []
    for c in ctrl_seq:
        cb = np.broadcast_to(c, (n, model.nu))
        a = ref.step(cb).flat()
        b = alt.step(cb).flat()
        errs.append(np.abs(a - b).max())
    alt.close()
    return dict(backend=backend, steps=len(ctrl_seq), n=n,
                max_abs=float(np.max(errs)), final_abs=float(errs[-1]),
                per_step=[float(e) for e in errs])


def throughput(model, n, backend="cpu", steps=200, warmup=10, **kw):
    """Steps per second, counting every world. Warmup excluded (JIT)."""
    env = make_vec(model, n, backend=backend, **kw)
    rng = np.random.default_rng(0)
    c = rng.normal(0, 0.05, (n, model.nu))
    for _ in range(warmup):
        env.step(c)
    t0 = time.time()
    for _ in range(steps):
        env.step(c)
    dt = time.time() - t0
    env.close()
    return dict(backend=backend, n=n, steps=steps, seconds=dt,
                steps_per_s=steps * n / dt)
