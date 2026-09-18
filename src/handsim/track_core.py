"""Tracking control: make a hand carry an object along a REFERENCE trajectory.

This is the structure DexTrack uses, built reference-agnostic on purpose. A
reference here is a sequence of object poses -- nothing about it is synthetic by
nature. `SyntheticSource` produces them today; an ARCTIC or GRAB clip produces
the same object-pose sequence tomorrow, and nothing downstream changes. That is
the whole point of drawing the boundary here: the human-data gate stops being a
blocker for the controller, the optimiser and the policy.

What this differs from, in the rest of this repository:

    grasp synthesis   pick ONE pose, close, and see if it survives a probe
    tracking          hold the object and FOLLOW a trajectory, correcting as
                      it drifts

The second needs feedback, which nothing in this project has yet demonstrated:
the one policy line that worked was matched by open-loop replay (NOTES
2026-09-12), because its task never required correcting anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from handsim.grasping.synth import GraspScene
from handsim.grasping.task import GRAVITY


@dataclass
class Reference:
    """An object trajectory to follow. Source-agnostic by construction."""
    pos: np.ndarray                  # (T, 3) object position over time
    quat: np.ndarray                 # (T, 4) object orientation, wxyz
    dt: float
    name: str = "ref"
    source: str = "synthetic"

    @property
    def T(self):
        return len(self.pos)

    def at(self, k):
        k = int(np.clip(k, 0, self.T - 1))
        return self.pos[k], self.quat[k]


def reference_from_traj(traj, p0=None, q0=None):
    """Lift a `task.Trajectory` into a Reference (object poses over time)."""
    from handsim.grasping.task import object_path
    p0 = np.zeros(3) if p0 is None else np.asarray(p0, float)
    q0 = np.array([1.0, 0, 0, 0]) if q0 is None else np.asarray(q0, float)
    P, Q = object_path(traj, p0, q0)
    return Reference(pos=P, quat=Q, dt=traj.dt, name=traj.name)


def quat_err(q, qref):
    """Geodesic angle between two quaternions, radians."""
    d = np.zeros(4)
    conj = np.array([qref[0], -qref[1], -qref[2], -qref[3]])
    mujoco.mju_mulQuat(d, np.asarray(q, float), conj)
    return float(2.0 * np.arccos(np.clip(abs(d[0]), -1.0, 1.0)))


class TrackingEnv:
    """A hand already grasping an object, asked to follow a reference.

    The grasp is formed by the existing synthesis path, then the controller
    takes over: at each control step it commands the six base DoF and the
    finger targets, and is scored on how far the OBJECT is from where the
    reference says it should be.
    """

    def __init__(self, hand_key, obj_half, shape="box", mass=0.05,
                 kp_finger=1.0, n_hands=1, ctrl_every=10):
        self.scene = GraspScene(hand_key, obj_half, mass=mass, n_hands=n_hands,
                                kp_finger=kp_finger, shape=shape)
        s = self.scene
        self.pfx = s.prefixes[0]
        self.base_a = s.base[self.pfx]
        self.base_q = s.base_q[self.pfx]
        self.finger = s.finger[self.pfx]
        self.ctrl_every = int(ctrl_every)
        self.dofs = ("x", "y", "z", "rx", "ry", "rz")
        self.n_act = len(self.dofs) + len(self.finger)
        self._grasp = None

    # -- setup -------------------------------------------------------------
    def set_grasp(self, params, finger_target=None):
        """Form the grasp the controller will start from."""
        att = self.scene.attempt(np.asarray(params, float), do_hold=False,
                                 finger_target=finger_target)
        self._grasp = (np.asarray(params, float), finger_target)
        return att

    def reset(self):
        """Re-form the grasp, switch gravity on, and settle under weight."""
        if self._grasp is None:
            raise RuntimeError("set_grasp() first")
        att = self.scene.attempt(self._grasp[0], do_hold=False,
                                 finger_target=self._grasp[1])
        if not att.valid or att.n_contacts < 2:
            return None
        self.scene.m.opt.gravity[:] = GRAVITY
        d = self.scene.d
        self.base0 = np.array([d.qpos[self.base_q[k]] for k in self.dofs])
        self.finger0 = {n: float(d.qpos[qa])
                        for n, (qa, _a, _t) in self.finger.items()}
        for _ in range(150):
            mujoco.mj_step(self.scene.m, d)
        return att

    # -- observation -------------------------------------------------------
    def observe(self, ref, k, lookahead=(1, 5, 15)):
        """Proprioception, object state, and where the reference wants it next.

        The lookahead is what makes this a tracking problem rather than a
        regulation one: a controller that only sees the current error is always
        late.
        """
        s, d = self.scene, self.scene.d
        q = [d.qpos[self.base_q[n]] for n in self.dofs]
        qd = [d.qvel[s.m.jnt_dofadr[mujoco.mj_name2id(
            s.m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.pfx}{n}")]] for n in self.dofs]
        fq = [d.qpos[qa] for _n, (qa, _a, _t) in self.finger.items()]
        op = d.qpos[s.obj_q:s.obj_q + 3]
        oq = d.qpos[s.obj_q + 3:s.obj_q + 7]
        ov = d.qvel[s.m.jnt_dofadr[mujoco.mj_name2id(
            s.m, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")]:][:6]
        cur_p, cur_q = ref.at(k)
        fut = []
        for h in lookahead:
            p_, q_ = ref.at(k + h)
            fut += list(p_ - op) + list(q_)
        return np.concatenate([q, qd, fq, op, oq, ov,
                               cur_p - op, cur_q, fut]).astype(np.float64)

    # -- stepping ----------------------------------------------------------
    def apply(self, action):
        """Action = 6 base targets (absolute) + finger targets (absolute)."""
        d = self.scene.d
        a = np.asarray(action, float)
        for i, n in enumerate(self.dofs):
            if n in self.base_a:
                d.ctrl[self.base_a[n]] = a[i]
        for j, (_n, (_qa, act, _t)) in enumerate(self.finger.items()):
            d.ctrl[act] = a[len(self.dofs) + j]

    def step(self, action):
        self.apply(action)
        for _ in range(self.ctrl_every):
            mujoco.mj_step(self.scene.m, self.scene.d)

    def error(self, ref, k):
        s, d = self.scene, self.scene.d
        p_, q_ = ref.at(k)
        op = d.qpos[s.obj_q:s.obj_q + 3]
        oq = d.qpos[s.obj_q + 3:s.obj_q + 7]
        return float(np.linalg.norm(op - p_)), quat_err(oq, q_)

    def nominal_action(self):
        """Hold still: the identity controller, and the baseline to beat."""
        return np.concatenate([self.base0,
                               [self.finger0[n] for n in self.finger]])


@dataclass
class Rollout:
    pos_err: np.ndarray
    rot_err: np.ndarray
    dropped: bool
    steps: int

    @property
    def mean_pos_cm(self):
        return float(np.mean(self.pos_err) * 100)

    @property
    def final_pos_cm(self):
        return float(self.pos_err[-1] * 100)


def rollout(env, ref, policy, drop=0.10, control_steps=None):
    """Run a controller against a reference; report tracking error."""
    n = control_steps or (ref.T // env.ctrl_every)
    pe, re_ = [], []
    for i in range(n):
        k = i * env.ctrl_every
        a = policy(env.observe(ref, k), k) if policy is not None \
            else env.nominal_action()
        env.step(a)
        p, r = env.error(ref, min(k + env.ctrl_every, ref.T - 1))
        pe.append(p); re_.append(r)
        if p > drop:
            return Rollout(np.array(pe), np.array(re_), True, i)
    return Rollout(np.array(pe), np.array(re_), False, n)


def _euler_xyz(R):
    """Angles for three sequential hinges rx, ry, rz on one body (R = Rx Ry Rz)."""
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    ry = np.arcsin(sy)
    if abs(sy) < 0.9999:
        rx = np.arctan2(-R[1, 2], R[2, 2])
        rz = np.arctan2(-R[0, 1], R[0, 0])
    else:                                   # gimbal: fold rz into rx
        rx = np.arctan2(R[2, 1], R[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz])


def feedforward(env, ref):
    """Base targets that would carry a RIGIDLY held object along the reference.

    The kinematic part of the problem, solved in closed form. Sampling MPC
    cannot discover an 8 cm coordinated reach by perturbing +/-1 cm around
    hold-still -- asked to, it dropped the object sooner than doing nothing.
    Feedforward supplies the motion; the optimiser is then left with what it is
    actually good for, which is correcting slip and dynamics.
    """
    d, s_ = env.scene.d, env.scene
    p0 = d.qpos[s_.obj_q:s_.obj_q + 3].copy()
    q0 = d.qpos[s_.obj_q + 3:s_.obj_q + 7].copy()
    pivot = d.xpos[s_.palm_bid[env.pfx]].copy()
    out = np.zeros((ref.T, 6))
    conj0 = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
    for k in range(ref.T):
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, ref.quat[k], conj0)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, dq)
        R = R.reshape(3, 3)
        out[k, 0:3] = (ref.pos[k]) - (R @ (p0 - pivot) + pivot)
        out[k, 3:6] = _euler_xyz(R)
    return out


# --------------------------------------------------------------------------
# stage 1: per-reference trajectory optimisation (the tracking demonstrations)
# --------------------------------------------------------------------------
def _save(env):
    m, d = env.scene.m, env.scene.d
    n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    s = np.empty(n)
    mujoco.mj_getState(m, d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return s


def _load(env, s):
    m, d = env.scene.m, env.scene.d
    mujoco.mj_setState(m, d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(m, d)


def mppi_track(env, ref, horizon=5, samples=32, sigma=0.012, temperature=None,
               rho=1.0, drop=0.10, w_rot=0.25, seed=0, control_steps=None,
               log=None):
    """Receding-horizon MPPI that follows a reference, in closed loop.

    Sampling MPC is used rather than a learned policy because this is the step
    that MAKES the demonstrations a policy is later distilled from -- DexTrack's
    structure: optimise each reference first, then learn one controller across
    all of them.

    The temperature is ADAPTIVE (lambda = rho * std(cost)). A fixed one is what
    collapsed the earlier MPPI in this project to an effective sample size of
    1.0 at every value tried, producing results bit-identical to no planner at
    all (NOTES 2026-09-12).
    """
    rng = np.random.default_rng(seed)
    n = control_steps or (ref.T // env.ctrl_every)
    ff = feedforward(env, ref)
    hold = env.nominal_action()
    nf = len(env.finger)

    def ff_action(step):
        """Feedforward base command at a control step, fingers held."""
        a = hold.copy()
        a[:6] = env.base0 + ff[int(np.clip(step, 0, ref.T - 1))]
        return a

    nominal = np.stack([ff_action(h * env.ctrl_every) for h in range(horizon)])
    lo = np.array([env.scene.m.actuator_ctrlrange[a][0]
                   for a in _act_ids(env)], float)
    hi = np.array([env.scene.m.actuator_ctrlrange[a][1]
                   for a in _act_ids(env)], float)
    pe, re_, ess_log = [], [], []
    for i in range(n):
        k = i * env.ctrl_every
        base = _save(env)
        noise = rng.normal(0.0, sigma, (samples, horizon, env.n_act))
        cand = np.clip(nominal[None] + noise, lo, hi)
        cost = np.empty(samples)
        for c in range(samples):
            _load(env, base)
            tot = 0.0
            for h in range(horizon):
                env.step(cand[c, h])
                p, r = env.error(ref, min(k + (h + 1) * env.ctrl_every,
                                          ref.T - 1))
                tot += p + w_rot * r
                if p > drop:
                    tot += 10.0 * (horizon - h)
                    break
            cost[c] = tot
        lam = temperature if temperature is not None else max(
            rho * float(np.std(cost)), 1e-6)
        w = np.exp(-(cost - cost.min()) / lam)
        w /= w.sum()
        ess_log.append(float(1.0 / np.sum(w ** 2)))
        nominal = np.einsum("s,shd->hd", w, cand)
        _load(env, base)
        env.step(nominal[0])
        p, r = env.error(ref, min(k + env.ctrl_every, ref.T - 1))
        pe.append(p); re_.append(r)
        if log and i % max(n // 6, 1) == 0:
            log(f"    step {i:3d}/{n}  pos {p*100:5.2f} cm  rot {np.degrees(r):5.1f} deg"
                f"  ESS {ess_log[-1]:4.1f}/{samples}")
        if p > drop:
            return Rollout(np.array(pe), np.array(re_), True, i), nominal, ess_log
        # shift, and re-seed the newly exposed end from the feedforward so the
        # horizon always knows where the reference is going next
        nominal = np.vstack([nominal[1:],
                             ff_action((i + 1 + horizon) * env.ctrl_every)[None]])
    return Rollout(np.array(pe), np.array(re_), False, n), nominal, ess_log


def _act_ids(env):
    ids = [env.base_a[n] for n in env.dofs if n in env.base_a]
    ids += [a for _n, (_q, a, _t) in env.finger.items()]
    return ids
