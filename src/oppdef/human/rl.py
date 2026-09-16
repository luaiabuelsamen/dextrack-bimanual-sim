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
    #: Penetration. Everything above prices where the object IS; nothing
    #: prices how the hand holds it there, and in this pipeline the hand
    #: reaches its targets by going INTO the object -- the contact solver's
    #: restoring force is the "grip". A policy seeded with a genuine 0.6 N touch
    #: re-buried the hand to 207 N / 4.9 mm within one rollout. This is the RL
    #: side of the gate `track.wrap_score` already applies to the grasp search,
    #: with the same 3 mm allowance. The cost is a hinge on the DEEPEST
    #: hand-object contact, in units of the allowance -- 6 mm costs w_pen, 9 mm
    #: costs 2 w_pen -- so the gradient does not vanish at the depths the
    #: retarget actually sits at (5-25 mm, docs/STAGE2.md), which a saturating
    #: shape would repeat the flat-exp mistake on. Bounded by `pen_max` (reward
    #: units, shared with the force term): unbounded, a few millimetres cost
    #: more than the ~1.5/step that staying alive on target earns, and the
    #: cheapest way to satisfy the penalty is to LET GO -- zero penetration is
    #: what a dropped object looks like, and the drop costs only 1.0 once.
    #: Default 0 leaves the reward bit-for-bit as it was: machinery for the
    #: owner to switch on, not a tuned weight.
    w_pen: float = 0.0
    pen_allow: float = 0.003     # metres, as track.wrap_score
    pen_max: float = 1.0         # cap on the pen + force cost per step, reward units
    #: The same hinge on total hand-object normal force, in multiples of the
    #: object's weight. Two measured references on a 0.2 kg object: the genuine
    #: hammer_use_2 seed holds at 20.7 N (~10x), the buried one at 207 N
    #: (~105x); the allowance sits at the former.
    w_force: float = 0.0
    force_allow: float = 10.0
    #: Holding -- a PROXY for it, not a hold condition. The penetration cost on
    #: its own makes the WORST behaviour optimal: the genuine hammer_use_2 seed
    #: under PPO ended at 0.00 mm, 0 contacts, 0 N, object on the floor, and on
    #: a penetration criterion that is the cleanest row in the run. This term
    #: pays, per step, for a hand-object contact set that PERSISTS: contacts on
    #: at least two distinct hand bodies (a single finger pressed hard against
    #: a falling object earns nothing), carrying at least `hold_ref` object
    #: weights of total normal force. Force rather than contact count because
    #: count is the burial-favouring quantity -- a buried hand has dozens -- and
    #: it saturates at one object weight because that is what supporting the
    #: object needs; more earns nothing, so it cannot reward burial either.
    #: What it does NOT do is verify equilibrium: this pipeline has measured
    #: 44.8 N of normal force on an object that still free-fell. Bounded to
    #: [0, w_hold]. Default 0, as above.
    w_hold: float = 0.0
    hold_ref: float = 1.0        # object weights of normal force for the full bonus
    #: Iterations on which grip force is measured for the log and the verbose
    #: line printed (every log_every-th, and the last). Penetration depth,
    #: contact count, body count and position error come free from the
    #: contact list and are always on; the force needs `mj_contactForce` per
    #: contact per env per step -- ~58k calls an iteration on a buried hand --
    #: so it runs only when a term prices it or the iteration is logged.
    log_every: int = 10
    seed: int = 0


@dataclass
class ContactSummary:
    """The hand-object contact set of one `MjData`, read in a single pass."""
    pen: float        #: deepest hand-object penetration, metres (0 if none)
    n: int            #: number of hand-object contacts
    bodies: int       #: distinct HAND bodies in contact, as track.wrap_score counts
    grip: float       #: total hand-object normal force, N, as track.total_grip (nan if not measured)


def contact_summary(sc, force: bool = True) -> ContactSummary:
    """One pass over `sc.data.contact` for what the reward and the log need.

    `sc.penetration()` and `track.total_grip(sc)` compute two of these; this
    walks the contact list once because it runs for every environment on every
    control step. Same identification as both: a contact counts when one geom
    is the object's and the other is the hand's.

    `force=False` skips `mj_contactForce` and reports grip as nan. Depth,
    count and bodies are read off the contact list; the force is a solver
    call per contact, and this repo has paid "reward identical, training 15%
    slower" for a per-step Python loop like that before.
    """
    m, d = sc.model, sc.data
    objs, hands = set(sc.obj_gids), set(sc.hand_gids)
    f = np.zeros(6)
    pen, n, grip, bodies = 0.0, 0, 0.0 if force else float("nan"), set()
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ({g1, g2} & objs and {g1, g2} & hands):
            continue
        n += 1
        pen = max(pen, -float(c.dist))
        bodies.add(int(m.geom_bodyid[g1 if g1 in hands else g2]))
        if force:
            mujoco.mj_contactForce(m, d, i, f)
            grip += abs(float(f[0]))       # normal component, contact frame
    return ContactSummary(pen=pen, n=n, bodies=len(bodies), grip=grip)


def object_weight(rt) -> float:
    """m g of the tracked object, N, from the model the tracker runs on."""
    m = rt.sim.model
    g = float(np.linalg.norm(m.opt.gravity)) or 9.81
    return float(m.body_mass[rt.sim.obj_bid]) * g


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
        self.weight = object_weight(rt)
        #: per-environment contact state after the latest `step`, kept as
        #: attributes rather than widening step's return so callers are
        #: unchanged: deepest penetration (m), hand-object contacts, distinct
        #: hand bodies, total normal force (N; nan unless a term prices it or
        #: `step(..., measure_grip=True)`), object position error (m), and
        #: the two shaping terms as they entered the reward (0 while their
        #: weights are 0).
        self.pe = np.zeros(n_envs)
        self.pen = np.zeros(n_envs)
        self.ncon = np.zeros(n_envs, int)
        self.nbody = np.zeros(n_envs, int)
        self.grip = np.zeros(n_envs)
        self.r_pen = np.zeros(n_envs)
        self.r_hold = np.zeros(n_envs)
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

    def step(self, actions, cfg: RLConfig, measure_grip: bool = False):
        """Apply one control step in every environment.

        `measure_grip` forces the per-contact force read (see RLConfig.log_every);
        it is on regardless whenever `w_force` or `w_hold` prices the force.
        """
        rt = self.rt
        need_force = bool(cfg.w_force or cfg.w_hold or measure_grip)
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
            # Contact state of THIS environment: `_use(i)` above made
            # `rt.sim.data` the i-th MjData, which is what the summary reads.
            # Always measured, so the log reports burying whatever the weights.
            cs = contact_summary(rt.sim, force=need_force)
            self.pe[i] = pe
            self.pen[i], self.ncon[i] = cs.pen, cs.n
            self.nbody[i], self.grip[i] = cs.bodies, cs.grip
            self.r_pen[i] = self.r_hold[i] = 0.0
            if cfg.w_pen or cfg.w_force:
                # Hinges in units of their allowance; the sum is capped at
                # pen_max so that, per step, penetration can never cost more
                # than the alive + tracking reward is worth -- otherwise the
                # optimum is to drop the object (see RLConfig). At the
                # defaults the cap equals the one-off drop penalty.
                hp = max(0.0, cs.pen - cfg.pen_allow) / cfg.pen_allow
                hf = max(0.0, cs.grip / self.weight - cfg.force_allow) / cfg.force_allow
                self.r_pen[i] = min(cfg.w_pen * hp + cfg.w_force * hf, cfg.pen_max)
                r -= self.r_pen[i]
            if cfg.w_hold and cs.bodies >= 2:
                # A persisting multi-body contact set with one object weight
                # of normal force earns the full bonus; more earns nothing.
                # A proxy for holding, not a test of it (see RLConfig).
                self.r_hold[i] = cfg.w_hold * min(
                    1.0, cs.grip / (cfg.hold_ref * self.weight))
                r += self.r_hold[i]
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
    #: Per-iteration means over every env-step. Read `pen_mm`, `frac_held`
    #: and `err_mm` TOGETHER: penetration falling while held fraction falls
    #: with it means the policy bought clean numbers by dropping the object,
    #: and held fraction high while the error grows is a hand riding a
    #: falling object down while touching it. `grip_n` is nan on iterations
    #: where the force was not measured (RLConfig.log_every).
    pen_mm: list = field(default_factory=list)      # mean deepest penetration
    frac_held: list = field(default_factory=list)   # >= 1 hand-object contact
    grip_n: list = field(default_factory=list)      # mean total normal force
    r_pen: list = field(default_factory=list)       # mean penetration cost paid
    r_hold: list = field(default_factory=list)      # mean hold bonus earned


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
            PEN = np.empty((cfg.horizon, cfg.n_envs))
            HELD = np.empty((cfg.horizon, cfg.n_envs))
            GRIP = np.empty((cfg.horizon, cfg.n_envs))
            RP = np.empty((cfg.horizon, cfg.n_envs))
            RH = np.empty((cfg.horizon, cfg.n_envs))
            PE = np.empty((cfg.horizon, cfg.n_envs))
            logged = it % cfg.log_every == 0 or it == cfg.iters - 1

            for t in range(cfg.horizon):
                x = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    d = net.dist(x)
                    a = d.sample()
                    LP[t] = d.log_prob(a).sum(-1).numpy()
                    V[t] = net.v(x).squeeze(-1).numpy()
                O[t], A[t] = obs, a.numpy()
                obs, rew, done = pool.step(A[t], cfg, measure_grip=logged)
                R[t], D[t] = rew, done
                PEN[t], HELD[t], GRIP[t] = pool.pen, pool.ncon > 0, pool.grip
                RP[t], RH[t], PE[t] = pool.r_pen, pool.r_hold, pool.pe
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
            log.err_mm.append(float(PE.mean() * 1000))
            log.pen_mm.append(float(PEN.mean() * 1000))
            log.frac_held.append(float(HELD.mean()))
            log.grip_n.append(float(GRIP.mean()))     # nan unless measured
            log.r_pen.append(float(RP.mean()))
            log.r_hold.append(float(RH.mean()))
            if verbose and logged:
                print(f"    iter {it:3d}  reward {R.mean():6.3f}  "
                      f"alive {1 - D.mean():.3f}  err {PE.mean() * 1000:6.1f} mm  "
                      f"pen {PEN.mean() * 1000:5.2f} mm  held {HELD.mean():.3f}  "
                      f"grip {GRIP.mean():7.1f} N  "
                      f"r_pen {RP.mean():.3f}  r_hold {RH.mean():.3f}", flush=True)
    finally:
        pool.restore()
        pool.close()
    return net, log


@dataclass
class EndState:
    """Where a rollout ENDED, alongside its tracking error.

    A tracking number without this cannot be read: a policy has ended a
    "successful" 111-frame track 10.83 mm inside the mug on 12 bodies, and
    another ended at 0.00 mm / 0 contacts / 0 N with the object on the floor.
    The same triple `experiments/tracking/g9_ppo_distill.py` stores per row.
    """
    pen_mm: float
    contacts: int
    bodies: int
    grip_n: float

    def __str__(self):
        return (f"ends {self.pen_mm:5.2f} mm in, {self.contacts:3d} contacts "
                f"on {self.bodies} bodies, {self.grip_n:8.1f} N")


def end_state(rt) -> EndState:
    cs = contact_summary(rt.sim)
    return EndState(pen_mm=cs.pen * 1000, contacts=cs.n, bodies=cs.bodies,
                    grip_n=cs.grip)


def evaluate(rt, net, start=None, deterministic=True, end=False, verbose=False):
    """Roll the policy out on the reference and score it on the truth.

    Returns `(errs, acts)`; with `end=True`, `(errs, acts, EndState)` -- the
    existing callers unpack two, so the end state is opt-in rather than a
    third element they would trip on. `verbose` prints it either way, so a
    rollout ending at 0.00 mm / 0 contacts / 0 N is visible.
    """
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
    es = end_state(rt)
    if verbose:
        print(f"    eval {errs.mean() * 1000:7.1f} mm  {es}", flush=True)
    if end:
        return errs, np.array(acts), es
    return errs, np.array(acts)
