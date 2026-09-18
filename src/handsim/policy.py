"""Policies: the interface, and an action-chunking model that is not an MLP.

The BC baseline was a two-layer MLP mapping one observation to one action. That
is a defensible starting point and it worked (67% +/- 47%), but it has two
specific defects that are worth naming, because both are properties of the
ARCHITECTURE rather than of the training:

1. It is Markov in a task that is not. The scripted expert is open-loop once
   calibrated, so its action depends on TIME, not only on state. Without a phase
   input the MLP scored 0/6 and drove the peg backwards (NOTES 2026-09-12).
   Feeding it the phase fixes the symptom by handing the policy a clock; it does
   not make the policy able to represent a multi-step intent.

2. It is re-decided every control step, so its errors are white. Real
   manipulation wants the opposite: commit to a motion for a while, because
   contact rewards consistency and punishes dither.

Action chunking (ACT, Zhao et al. 2023) addresses both by predicting the next H
actions from one observation and executing them with temporal ensembling, so
each executed action averages the predictions several past observations made
about this moment. The phase input then becomes a convenience rather than a
crutch: a chunk spans time by construction.

Everything here is architecture-agnostic about the backbone -- what matters is
the chunk, the ensembler, and one interface every experiment can call:

    p = policy(kind="chunk", obs_dim=..., act_dim=..., horizon=16)
    p.reset()
    a = p(obs)

`torch` is imported lazily so this module can be inspected, and its chunking
logic tested, on a machine with no torch at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# chunk bookkeeping -- pure numpy, so it is testable without torch
# --------------------------------------------------------------------------
def make_chunks(obs, act, horizon):
    """(obs_t, actions t..t+H-1) pairs from ONE episode.

    The tail is padded by repeating the last action, which is what ACT does and
    is the right boundary behaviour here: the expert holds its final pose, so a
    repeated last action is the true continuation, not a filler value.

    Episodes must be chunked SEPARATELY. Chunking a concatenated array lets a
    window run off the end of one episode into the start of the next, teaching
    the policy to follow a demonstration with a different peg pose -- a quiet
    corruption that looks like ordinary training noise.
    """
    obs = np.asarray(obs, np.float32)
    act = np.asarray(act, np.float32)
    if len(obs) != len(act):
        raise ValueError(f"obs {len(obs)} and act {len(act)} differ in length")
    n, a_dim = len(act), act.shape[1]
    out = np.empty((n, horizon, a_dim), np.float32)
    for h in range(horizon):
        idx = np.minimum(np.arange(n) + h, n - 1)     # clamp = hold the last
        out[:, h] = act[idx]
    return obs, out


def chunk_dataset(episodes, horizon):
    """Stack per-episode chunks. `episodes` is a sequence of (obs, act)."""
    X, Y = [], []
    for o, a in episodes:
        xo, ya = make_chunks(o, a, horizon)
        X.append(xo); Y.append(ya)
    if not X:
        raise ValueError("no episodes")
    return np.concatenate(X), np.concatenate(Y)


class TemporalEnsembler:
    """Averages the predictions several past chunks made about NOW.

    At step t the policy has predictions for t from the chunks emitted at
    t, t-1, ... t-H+1. ACT combines them with weights exp(-m*i) in the age i of
    the prediction, so the freshest chunk dominates while older ones damp the
    step change that occurs whenever a new chunk is emitted.

    `m=0` averages all of them equally; large `m` approaches "use the newest
    chunk only", which is chunking without ensembling.
    """

    def __init__(self, horizon, act_dim, m=0.01):
        self.h, self.a, self.m = int(horizon), int(act_dim), float(m)
        self.reset()

    def reset(self):
        # store[i] is the WHOLE chunk emitted i steps ago. Keeping only each
        # chunk's first element turns this into a moving average of fresh
        # predictions -- element i of the chunk emitted i steps ago IS the
        # prediction for now, and it is the only thing being ensembled.
        self.store = np.zeros((self.h, self.h, self.a), np.float32)
        self.n = 0

    def push(self, chunk):
        """Record a freshly predicted chunk of shape (H, act_dim)."""
        chunk = np.asarray(chunk, np.float32).reshape(self.h, self.a)
        self.store[1:] = self.store[:-1]
        self.store[0] = chunk
        self.n = min(self.n + 1, self.h)

    def predictions(self):
        """Every live prediction about the current step, freshest first."""
        return np.stack([self.store[i, i] for i in range(self.n)])

    def step(self, chunk):
        """Push a chunk and return the action to execute now."""
        self.push(chunk)
        preds = self.predictions()
        w = np.exp(-self.m * np.arange(self.n, dtype=np.float32))
        w = w / w.sum()
        return (preds * w[:, None]).sum(0)


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------
@dataclass
class Norm:
    """Observation/action standardisation, carried with the policy.

    Kept beside the weights rather than recomputed at evaluation: a policy
    evaluated under statistics other than the ones it was trained with is a
    silent failure, and this project has had enough of those.
    """
    om: np.ndarray
    os: np.ndarray
    am: np.ndarray
    as_: np.ndarray

    @staticmethod
    def fit(O, A):
        return Norm(O.mean(0), O.std(0) + 1e-6, A.mean(0), A.std(0) + 1e-6)

    def x(self, o):
        return (o - self.om) / self.os

    def y_inv(self, y):
        return y * self.as_ + self.am


class Policy:
    """What every experiment can call. `reset()` per episode, then `p(obs)`."""
    kind = "abstract"

    def reset(self):
        pass

    def __call__(self, obs):
        raise NotImplementedError

    def act_dim(self):
        raise NotImplementedError


class Scripted(Policy):
    """Wraps any callable, so the expert and the baselines share the interface."""
    kind = "scripted"

    def __init__(self, fn, act_dim, on_reset=None):
        self.fn, self._a, self.on_reset = fn, act_dim, on_reset

    def reset(self):
        if self.on_reset:
            self.on_reset()

    def __call__(self, obs):
        return np.asarray(self.fn(obs), np.float32)

    def act_dim(self):
        return self._a


class TorchPolicy(Policy):
    """An MLP backbone that emits either one action or a chunk of H.

    horizon=1 reproduces the BC baseline exactly, so the chunking claim can be
    measured against it rather than asserted.
    """
    kind = "torch"

    def __init__(self, net, norm: Norm, horizon=1, act_dim=None, m=0.01,
                 device="cpu"):
        self.net, self.norm, self.h = net, norm, int(horizon)
        self._a = int(act_dim if act_dim is not None else len(norm.am))
        self.dev = device
        self.ens = TemporalEnsembler(self.h, self._a, m) if self.h > 1 else None

    def reset(self):
        if self.ens is not None:
            self.ens.reset()

    def __call__(self, obs):
        import torch
        with torch.no_grad():
            x = torch.as_tensor(self.norm.x(np.asarray(obs, np.float32))[None],
                                dtype=torch.float32, device=self.dev)
            y = self.net(x).cpu().numpy()[0]
        if self.h == 1:
            return self.norm.y_inv(y)
        chunk = self.norm.y_inv(y.reshape(self.h, self._a))
        return self.ens.step(chunk)

    def act_dim(self):
        return self._a


def mlp(in_dim, out_dim, hidden=512, layers=2):
    import torch.nn as nn
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods += [nn.Linear(d, out_dim)]
    return nn.Sequential(*mods)


def train(episodes, horizon=1, epochs=300, hidden=512, layers=2, lr=1e-3,
          batch=256, seed=0, device=None, m=0.01, log=print):
    """Fit a policy to a list of (obs, act) episodes.

    `horizon=1` is plain behaviour cloning; `horizon>1` is action chunking.
    Episodes are kept separate so a chunk never spans two demonstrations.
    """
    import torch
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    O, Y = chunk_dataset(episodes, horizon)
    A = np.concatenate([a for _o, a in episodes])
    norm = Norm.fit(O, A)
    a_dim = A.shape[1]

    Xn = norm.x(O).astype(np.float32)
    Yn = ((Y - norm.am) / norm.as_).reshape(len(Y), -1).astype(np.float32)
    X = torch.tensor(Xn, device=dev)
    T = torch.tensor(Yn, device=dev)

    net = mlp(O.shape[1], horizon * a_dim, hidden, layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = ((net(X[idx]) - T[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if log and (ep % max(epochs // 5, 1) == 0 or ep == epochs - 1):
            log(f"    epoch {ep:3}  mse {tot/n:.5f}  (horizon {horizon})")
    net.eval()
    return TorchPolicy(net, norm, horizon=horizon, act_dim=a_dim, m=m,
                       device=dev)
