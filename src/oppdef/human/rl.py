"""Stage 3, done properly: PPO per reference, not a sampling optimiser.

MPPI was a substitute and the substitution is what broke distillation. A
sampling optimiser's per-step correction is dominated by its own noise draw
rather than by the state -- regressing it on the observation gives a linear R^2
of **0.075 in sample** -- so there is no function for a network to learn. That
is why DexTrack trains a policy per reference and distils THAT: a policy is
state-conditioned by construction, which is the property the distillation stage
actually depends on.

Runs on CPU. torch reports this machine's CUDA driver as too old, MJX-JAX does
not initialise here at all, and MuJoCo's own stepping is the bottleneck anyway,
so the environments are pooled over one shared model with a separate MjData
each and the policy is evaluated on the whole batch at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco


@dataclass
class RLConfig:
    n_envs: int = 12
    horizon: int = 48            # control steps per environment per iteration
    iters: int = 60
    epochs: int = 6
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.97
    lam: float = 0.95
    clip: float = 0.2
    ent: float = 3e-3
    vf: float = 0.5
    max_grad: float = 1.0
    #: action scale: palm translation (m), palm rotation (rad), finger (rad)
    a_pos: float = 0.006
    a_rot: float = 0.04
    a_fin: float = 0.06
    #: Reward shaping. The first version used a single exp(-e/0.02), which is
    #: numerically flat past 5 cm: at 10 cm it is 0.007, so a policy that has
    #: let the object drift gets no gradient telling it which way back. It
    #: learned to stay ALIVE (0.98 of steps) without learning to track, and
    #: came out at 10535 mm against the feedforward's 8197 mm over 614k steps.
    #: Two scales now, one tight and one wide, so there is signal at both ends,
    #: plus a small linear term that never saturates at all.
    s_pos: float = 0.02          # metres, the precision term
    s_wide: float = 0.10         # metres, the recovery term
    w_wide: float = 0.6
    w_lin: float = 2.0           # per metre, never saturates
    s_rot: float = 0.35          # radians
    w_rot: float = 0.35
    alive: float = 0.10
    drop_m: float = 0.15
    seed: int = 0


class Pool:
    """N independent simulations of one reference, over one shared model.

    Only `MjData` is duplicated. Rebuilding the scene per environment would
    recompile the model and reload the convex decomposition N times, which
    dominates everything else on this machine.

    The physics is stepped in THREADS. MuJoCo releases the GIL inside
    `mj_step`, so this is real parallelism and not a scheduling illusion:
    measured on this Jetson, 8 environments step at 9,676 steps/s sequentially
    and 48,979 threaded, a 5.06x speedup. That is the difference between eight
    hours per million control steps and ninety minutes, which is the difference
    between PPO being testable here and not.

    The policy still runs once per batch on the main thread; only the 33
    MuJoCo steps that make up a control step are parallel.
    """

    def __init__(self, rt, n_envs: int, starts):
        self.rt = rt
        self.n = n_envs
        self.datas = [mujoco.MjData(rt.sim.model) for _ in range(n_envs)]
        self.starts = np.asarray(starts)
        self.k = np.zeros(n_envs, int)
        self._orig = rt.sim.data
        self._grip = [np.zeros(rt.sim.model.nu) for _ in range(n_envs)]
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=min(n_envs, 8))

    def close(self):
        self._pool.shutdown(wait=False)

    def _use(self, i):
        self.rt.sim.data = self.datas[i]
        self.rt._grip_offset = self._grip[i]

    def restore(self):
        self.rt.sim.data = self._orig

    def reset(self, i, rng=None):
        self._use(i)
        k0 = int(self.starts[rng.integers(len(self.starts))] if rng is not None
                 else self.starts[i % len(self.starts)])
        self.rt.reset_at(k0)
        self._grip[i] = self.rt._grip_offset.copy()
        self.k[i] = k0
        return self.rt.observe(k0)

    def reset_all(self, rng):
        return np.stack([self.reset(i, rng) for i in range(self.n)])

    def step(self, actions, cfg: RLConfig):
        """Apply one control step in every environment."""
        rt = self.rt
        obs = np.empty((self.n, rt.n_obs))
        rew = np.empty(self.n)
        done = np.zeros(self.n, bool)
        scale = np.concatenate([np.full(3, cfg.a_pos), np.full(3, cfg.a_rot),
                                np.full(rt.n_action - 6, cfg.a_fin)])
        # Phase 1: set every environment's command. This touches rt's shared
        # index metadata, so it is done serially, one env swapped in at a time.
        ks = []
        for i in range(self.n):
            self._use(i)
            k = int(self.k[i])
            ks.append(k)
            rt.apply(k, np.clip(actions[i], -1, 1) * scale)

        # Phase 2: step the physics in parallel. mj_step releases the GIL.
        nsub = rt.ctrl_every

        def _run(d):
            for _ in range(nsub):
                mujoco.mj_step(rt.sim.model, d)

        list(self._pool.map(_run, self.datas))

        # Phase 3: read the outcome, again serially.
        for i in range(self.n):
            self._use(i)
            k = ks[i]
            pe, re = rt.error(k)
            # A dense, bounded reward. A bare negative distance makes the best
            # available action "end the episode", which a drop conveniently
            # provides; the alive bonus is what removes that incentive.
            r = (np.exp(-pe / cfg.s_pos)
                 + cfg.w_wide * np.exp(-pe / cfg.s_wide)
                 - cfg.w_lin * pe
                 + cfg.w_rot * np.exp(-re / cfg.s_rot)
                 + cfg.alive)
            self.k[i] = k + 1
            if pe > cfg.drop_m or self.k[i] >= rt.T:
                done[i] = True
                r -= 1.0 if pe > cfg.drop_m else 0.0
            rew[i] = r
            obs[i] = rt.observe(int(min(self.k[i], rt.T - 1)))
        return obs, rew, done


def make_policy(n_obs, n_act, hidden=192, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)

    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.Sequential(
                nn.Linear(n_obs, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, n_act))
            self.v = nn.Sequential(
                nn.Linear(n_obs, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1))
            self.log_std = nn.Parameter(torch.full((n_act,), -1.0))
            # a near-zero initial mean means the policy starts as the
            # feedforward, which already tracks: PPO then only has to learn the
            # correction rather than rediscover the whole trajectory
            self.pi[-1].weight.data.mul_(0.01)
            self.pi[-1].bias.data.zero_()

        def dist(self, x):
            mu = self.pi(x)
            return torch.distributions.Normal(mu, self.log_std.exp())

    return AC()


@dataclass
class TrainLog:
    reward: list = field(default_factory=list)
    err_mm: list = field(default_factory=list)
    frac_alive: list = field(default_factory=list)


def train(rt, cfg: RLConfig | None = None, starts=None, verbose=True):
    """Train one tracking policy for one reference."""
    import torch

    cfg = cfg or RLConfig()
    rng = np.random.default_rng(cfg.seed)
    if starts is None:
        gf = rt.grasp_frames()
        starts = gf if len(gf) else np.array([0])

    pool = Pool(rt, cfg.n_envs, starts)
    net = make_policy(rt.n_obs, rt.n_action, seed=cfg.seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    log = TrainLog()

    obs = pool.reset_all(rng)
    try:
        for it in range(cfg.iters):
            O = np.empty((cfg.horizon, cfg.n_envs, rt.n_obs), np.float32)
            A = np.empty((cfg.horizon, cfg.n_envs, rt.n_action), np.float32)
            LP = np.empty((cfg.horizon, cfg.n_envs), np.float32)
            R = np.empty((cfg.horizon, cfg.n_envs), np.float32)
            D = np.zeros((cfg.horizon, cfg.n_envs), np.float32)
            V = np.empty((cfg.horizon + 1, cfg.n_envs), np.float32)

            for t in range(cfg.horizon):
                x = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    d = net.dist(x)
                    a = d.sample()
                    LP[t] = d.log_prob(a).sum(-1).numpy()
                    V[t] = net.v(x).squeeze(-1).numpy()
                O[t], A[t] = obs, a.numpy()
                obs, rew, done = pool.step(A[t], cfg)
                R[t], D[t] = rew, done
                for i in np.nonzero(done)[0]:
                    obs[i] = pool.reset(int(i), rng)
            with torch.no_grad():
                V[cfg.horizon] = net.v(
                    torch.as_tensor(obs, dtype=torch.float32)).squeeze(-1).numpy()

            adv = np.zeros_like(R)
            last = 0.0
            for t in reversed(range(cfg.horizon)):
                nz = 1.0 - D[t]
                delta = R[t] + cfg.gamma * V[t + 1] * nz - V[t]
                last = delta + cfg.gamma * cfg.lam * nz * last
                adv[t] = last
            ret = adv + V[:cfg.horizon]

            b_o = torch.as_tensor(O.reshape(-1, rt.n_obs))
            b_a = torch.as_tensor(A.reshape(-1, rt.n_action))
            b_lp = torch.as_tensor(LP.reshape(-1))
            b_ad = torch.as_tensor(adv.reshape(-1))
            b_rt = torch.as_tensor(ret.reshape(-1))
            b_ad = (b_ad - b_ad.mean()) / (b_ad.std() + 1e-8)

            n = len(b_o)
            for _ in range(cfg.epochs):
                for idx in torch.randperm(n).split(cfg.minibatch):
                    d = net.dist(b_o[idx])
                    lp = d.log_prob(b_a[idx]).sum(-1)
                    ratio = (lp - b_lp[idx]).exp()
                    a1 = ratio * b_ad[idx]
                    a2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * b_ad[idx]
                    pl = -torch.min(a1, a2).mean()
                    vl = ((net.v(b_o[idx]).squeeze(-1) - b_rt[idx]) ** 2).mean()
                    loss = pl + cfg.vf * vl - cfg.ent * d.entropy().sum(-1).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad)
                    opt.step()

            log.reward.append(float(R.mean()))
            log.frac_alive.append(float(1.0 - D.mean()))
            if verbose and (it % 10 == 0 or it == cfg.iters - 1):
                print(f"    iter {it:3d}  reward {R.mean():6.3f}  "
                      f"alive {1 - D.mean():.3f}", flush=True)
    finally:
        pool.restore()
        pool.close()
    return net, log


def evaluate(rt, net, start=None, deterministic=True):
    """Roll the policy out on the reference and score it on the truth."""
    import torch

    if start is None:
        gf = rt.grasp_frames()
        start = int(gf[0]) if len(gf) else 0
    rt.reset_at(start)
    cfg = RLConfig()
    scale = np.concatenate([np.full(3, cfg.a_pos), np.full(3, cfg.a_rot),
                            np.full(rt.n_action - 6, cfg.a_fin)])
    errs, acts = [], []
    for k in range(start, rt.T):
        x = torch.as_tensor(rt.observe(k), dtype=torch.float32)[None]
        with torch.no_grad():
            d = net.dist(x)
            a = (d.mean if deterministic else d.sample()).numpy()[0]
        acts.append(a)
        rt.apply(k, np.clip(a, -1, 1) * scale)
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(rt.sim.model, rt.sim.data)
        errs.append(rt.error(k)[0])
    errs = np.array(errs)
    return errs, np.array(acts)
