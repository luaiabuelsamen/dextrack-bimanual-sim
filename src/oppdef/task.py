"""A manipulation task: carry the object somewhere, under gravity.

G1 found that a demonstration contributes nothing beyond initialisation when
the target is a static grasp. The obvious rejoinder is that a static grasp is
not what a demonstration is *about* -- a human reaching for a mug is specifying
where the mug goes, not which joint angles to use. This module builds the
machinery to test that: an object trajectory, the wrench sequence that
trajectory demands of the contacts, and an executor that runs it.

The distinction that makes the test meaningful:

    epsilon         the worst wrench the grasp resists over EVERY direction
    task margin     the worst ratio of resistible-to-required over the
                    directions this task actually demands

A task whose wrenches lie in a narrow cone can be served by a grasp with poor
epsilon, and a grasp with good epsilon can still fail a task that loads it in
its one weak direction. If a demonstration carries usable information, this is
where it should show up.

Success is measured on the OBJECT, never on the objective: the object must
track where a rigidly-held object would have gone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from oppdef.metrics.epsilon import (wrench_set, object_contacts, task_margin,
                                    epsilon_from_wrenches)

GRAVITY = np.array([0.0, 0.0, -9.81])


def _smooth(t):
    """Minimum-jerk easing, so accelerations are bounded and well defined."""
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


@dataclass
class Trajectory:
    """A commanded base path and the object motion a rigid hold would produce."""
    dof: tuple                      # ('x','y','z','rx','ry','rz')
    cmd: np.ndarray                 # (T, 6) base targets over time
    dt: float
    name: str = "carry"

    @property
    def T(self):
        return len(self.cmd)


def carry(dt=0.002, lift=0.08, across=0.08, tilt=np.deg2rad(60.0),
          hold_s=0.25, seg_s=0.7, name="lift-tilt-carry-place"):
    """Lift the object, tilt it, carry it sideways, set it back down.

    The tilt is what makes the task about more than weight: a pure vertical
    lift demands one force direction and no moment at all, which any grasp that
    can hold the object at rest already supplies. Rotating it under gravity
    demands a torque, and torque is where contact placement actually matters.
    """
    segs = [                                    # (dx, dy, dz, drx), seconds
        ((0, 0, 0, 0), hold_s),
        ((0, 0, lift, 0), seg_s),
        ((0, 0, lift, tilt), seg_s),
        ((0, across, lift, tilt), seg_s),
        ((0, across, lift, 0), seg_s),
        ((0, across, 0, 0), seg_s),
        ((0, across, 0, 0), hold_s),
    ]
    cmd, cur = [], np.zeros(4)
    for target, secs in segs:
        n = max(int(secs / dt), 1)
        tgt = np.asarray(target, float)
        for k in range(n):
            a = _smooth((k + 1) / n)
            cmd.append(cur + a * (tgt - cur))
        cur = tgt.copy()
    cmd = np.asarray(cmd)
    full = np.zeros((len(cmd), 6))
    full[:, 0:3] = cmd[:, 0:3]
    full[:, 3] = cmd[:, 3]
    return Trajectory(dof=("x", "y", "z", "rx", "ry", "rz"), cmd=full, dt=dt,
                      name=name)


def object_path(traj, p0, q0):
    """The OBJECT trajectory a demonstration specifies: translation, and
    rotation about the object's own centre.

    A demonstration is a record of what the object did. It says nothing about a
    palm, so the specification cannot be phrased as a rotation about one -- an
    earlier version rotated the object about the hand, which makes the required
    wrench depend on where the hand happened to be and stops it being a task
    specification at all.
    """
    P, Q = [], []
    for c in traj.cmd:
        ang = float(c[3])
        qr = np.array([np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qr, q0)
        P.append(p0 + c[0:3])
        Q.append(q)
    return np.asarray(P), np.asarray(Q)


def base_command(traj, p0, pivot):
    """Base targets that make a rigidly-held object follow `object_path`.

    The base rotates the hand about the base body's own origin, so realising a
    rotation about the OBJECT's centre needs a compensating translation. The
    task is specified on the object; this is the only place the hand enters.
    """
    out = np.zeros((traj.T, 6))
    for k, c in enumerate(traj.cmd):
        ang = float(c[3])
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        # want: R (p0 - pivot) + pivot + t = p0 + delta
        t = (p0 + c[0:3]) - (R @ (p0 - pivot) + pivot)
        out[k, 0:3] = t
        out[k, 3] = ang
    return out


def required_wrenches(P, Q, dt, mass, inertia_diag, lam, stride=25):
    """The wrench the contacts must supply, sampled along the path.

    Force is m(a - g): the contacts carry the object's weight plus whatever it
    takes to accelerate it. Torque is I*alpha + omega x I*omega about the COM,
    scaled by `lam` exactly as `wrench_set` scales its torque rows, so the two
    live in the same space.
    """
    P = np.asarray(P, float)
    v = np.gradient(P, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    # angular velocity from the quaternion sequence
    Q = np.asarray(Q, float)
    w = np.zeros_like(P)
    for i in range(1, len(Q) - 1):
        dq = np.zeros(4)
        conj = np.array([Q[i - 1][0], -Q[i - 1][1], -Q[i - 1][2], -Q[i - 1][3]])
        mujoco.mju_mulQuat(dq, Q[i + 1], conj)
        ang = 2.0 * np.arctan2(np.linalg.norm(dq[1:]), dq[0])
        ax = dq[1:] / (np.linalg.norm(dq[1:]) + 1e-12)
        w[i] = ax * ang / (2 * dt)
    al = np.gradient(w, dt, axis=0)
    I = np.diag(np.asarray(inertia_diag, float))
    out = []
    for i in range(0, len(P), stride):
        f = mass * (a[i] - GRAVITY)
        tau = I @ al[i] + np.cross(w[i], I @ w[i])
        out.append(np.concatenate([f, tau / lam]))
    return np.asarray(out)


@dataclass
class TaskResult:
    success: bool
    max_slip_m: float          # worst object slip in the palm frame
    final_slip_m: float
    dropped: bool
    steps: int
    margin: float = 0.0
    epsilon: float = 0.0
    f_total: float = 0.0
    reason: str = ""


def _in_palm_frame(scene, pfx):
    """Object position expressed in the palm's frame."""
    d = scene.d
    b = scene.palm_bid[pfx]
    R = d.xmat[b].reshape(3, 3)
    return R.T @ (d.qpos[scene.obj_q:scene.obj_q + 3] - d.xpos[b])


def run_task(scene, traj, slip_tol=0.015, drop=0.06, settle=200,
             min_carry=0.05):
    """Execute the trajectory on an already-formed grasp. Object-side success.

    Success is **slip in the palm frame**, not tracking error against the
    commanded path. The base is position-controlled with finite gain, so the
    whole hand lags its command; scored against the command, a perfectly held
    object fails for the actuator's reasons and the experiment would be
    measuring controller gain. Slip asks the only question the grasp is
    responsible for: did the object stay where the hand put it?

    A run must also actually carry the object (`min_carry`), so a grasp that
    holds perfectly while going nowhere is not a success.

    Gravity is switched ON here. The static benchmark deliberately runs without
    it so a resting object cannot be mistaken for a held one; a carry task is
    meaningless without weight.
    """
    m, d = scene.m, scene.d
    pfx = scene.prefixes[0]
    base_q = scene.base_q[pfx]
    base_a = scene.base[pfx]
    m.opt.gravity[:] = GRAVITY

    p0 = d.qpos[scene.obj_q:scene.obj_q + 3].copy()
    start = np.array([d.qpos[base_q[dof]] for dof in traj.dof])
    pivot = d.xpos[scene.palm_bid[pfx]].copy()
    cmd = base_command(traj, p0, pivot)

    for _ in range(settle):                 # let weight load the contacts
        mujoco.mj_step(m, d)
    rest = _in_palm_frame(scene, pfx)
    if np.linalg.norm(d.qpos[scene.obj_q:scene.obj_q + 3] - p0) > drop:
        return TaskResult(False, float("nan"), float("nan"), True, 0,
                          reason="dropped before the motion started")

    worst = 0.0
    for k in range(traj.T):
        for i, dof in enumerate(traj.dof):
            if dof in base_a:
                d.ctrl[base_a[dof]] = start[i] + cmd[k][i]
        mujoco.mj_step(m, d)
        slip = float(np.linalg.norm(_in_palm_frame(scene, pfx) - rest))
        worst = max(worst, slip)
        if slip > drop:
            return TaskResult(False, worst, slip, True, k,
                              reason=f"slipped out of the hand at step {k}")
    carried = float(np.linalg.norm(
        d.qpos[scene.obj_q:scene.obj_q + 3] - p0))
    final = float(np.linalg.norm(_in_palm_frame(scene, pfx) - rest))
    if carried < min_carry:
        return TaskResult(False, worst, final, False, traj.T,
                          reason=f"object only moved {carried*100:.1f} cm")
    return TaskResult(bool(worst < slip_tol), worst, final, False, traj.T)


def grasp_wrench_capacity(scene, required, mu_default=1.0):
    """(task margin, epsilon, total contact force) for the current contact set."""
    P, N, MU, F = object_contacts(scene.m, scene.d, scene.obj_gid)
    if len(P) < 2:
        return 0.0, 0.0, 0.0
    com = np.array(scene.d.xipos[scene.obj_bid], float)
    lam = float(np.linalg.norm(scene.obj_half)) or 1.0
    W = wrench_set(P, N, MU, com, lam)
    f_tot = float(F.sum())
    return (task_margin(W, required, f_total=max(f_tot, 1e-9)),
            epsilon_from_wrenches(W), f_tot)
