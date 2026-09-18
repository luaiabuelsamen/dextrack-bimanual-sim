"""MPPI grasp-and-lift: one shared cost function, any hand.

Why a planner instead of a scripted closure. Every grasp result before this
rests on a closure hand-designed for LEAP -- which joints flex, by how much, in
what order. Carrying that to Allegro, Shadow and f5d6 would make my scripting a
confound: if a hand scores worse there is no way to separate the hand from the
closure I wrote for it. A sampling-based planner removes that. The cost is
identical for every hand; the planner finds the finger motion.

This is the MuJoCo MPC setup (its Hand task is in-hand cube reorientation), run
on `mujoco.rollout` rather than by building MJPC, whose C++/GUI build on
aarch64 Tegra is a risk with no upside here.

Task: the palm rises on a fixed schedule. The planner controls only the finger
joint targets and is scored on how well the object tracks the palm.

  cost = 200*||obj_pos - target(t)||^2        position tracking
       +   2*(1 - |quat . quat_0|)            orientation hold
       +   8*||obj_vel - lift_vel(t)||^2      velocity tracking
       + 0.5*||obj_angvel||^2
       + 0.02*||du||^2

The velocity term is not optional. With position tracking alone the planner
discovers it can BAT the object upward: 13 cm of lift, epsilon 0 at the top and
28-40 kN of contact force, because ballistic motion satisfies a height target
inside a short horizon. Adding velocity tracking took peak force from 40 kN to
12 N. No term rewards contact, force, or any particular finger -- those are
means, and naming them is how hand-specific bias gets smuggled in.

Success is a PHYSICAL test, not an analytic one: lift >= 10 cm, then a downward
jerk faster than gravity (a carried object separates and is left behind, a
gripped one follows), then a shake. Epsilon is reported alongside but does not
gate, because it demands full 6-D force closure -- sufficient for holding, not
necessary, and it fails a grip that demonstrably survives 60 m/s^2.
DexGraspBench likewise reports simulation success and analytic metrics
separately rather than gating one on the other.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco
from mujoco import rollout as mj_rollout

from handsim.envs.grasp_bench import Bench, HANDS                          # noqa: E402

FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class MPPI:
    def __init__(self, bench, horizon=25, substep=10, n_samples=48,
                 sigma=0.02, rho=1.0, nthread=8, seed=0):
        self.b = bench
        self.H, self.S, self.K = horizon, substep, n_samples
        self.sigma, self.rho = sigma, rho
        self.ess = float('nan')
        self.rng = np.random.default_rng(seed)
        self.nu = bench.m.nu
        self.ja, self.lift_a = bench.ja, bench.lift_a
        self.nstate = mujoco.mj_stateSize(bench.m, FULL)
        self.datas = [mujoco.MjData(bench.m) for _ in range(nthread)]
        self.pool = mj_rollout.Rollout(nthread=nthread)
        lo = bench.m.actuator_ctrlrange[bench.ja, 0].copy()
        hi = bench.m.actuator_ctrlrange[bench.ja, 1].copy()
        bad = hi <= lo
        lo[bad], hi[bad] = -np.pi, np.pi
        self.lo, self.hi = lo, hi
        self.U = np.tile(0.5 * (lo + hi), (horizon, 1))

    def cost(self, states, targets, quat0, U, lift_vel):
        oq = 1 + self.b.oq
        ov = 1 + self.b.m.nq + self.b.ov
        pos = states[:, :, oq:oq + 3]
        quat = states[:, :, oq + 3:oq + 7]
        vel = states[:, :, ov:ov + 3]
        avel = states[:, :, ov + 3:ov + 6]
        tr = ((pos - targets[None, :, :]) ** 2).sum(-1)
        dot = np.abs((quat * quat0[None, None, :]).sum(-1)).clip(0, 1)
        spin = 1.0 - dot
        vtrack = ((vel[:, :, 2] - lift_vel[None, :]) ** 2
                  + vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        spinv = (avel ** 2).sum(-1)
        du = np.diff(U, axis=1, prepend=U[:, :1])
        ctrl = (du ** 2).sum(-1).repeat(self.S, axis=1)[:, :tr.shape[1]]
        return (200.0 * tr + 2.0 * spin + 8.0 * vtrack + 0.5 * spinv
                + 0.02 * ctrl).sum(1)

    def plan(self, state0, lift_seq, obj0, quat0, dt, iters=2):
        nstep = self.H * self.S
        targets = obj0[None, :] + np.stack(
            [np.zeros(nstep), np.zeros(nstep), lift_seq], axis=1)
        lift_vel = np.gradient(lift_seq) / dt
        for _ in range(iters):
            noise = self.rng.normal(0, self.sigma, (self.K, self.H, len(self.ja)))
            cand = np.clip(self.U[None] + noise, self.lo, self.hi)
            cand[0] = self.U
            ctrl = np.zeros((self.K, nstep, self.nu))
            ctrl[:, :, self.ja] = cand.repeat(self.S, axis=1)
            ctrl[:, :, self.lift_a] = lift_seq[None, :]
            st = np.empty((self.K, nstep, self.nstate))
            self.pool.rollout(self.b.m, self.datas,
                              np.tile(state0, (self.K, 1)), ctrl, state=st)
            c = np.nan_to_num(self.cost(st, targets, quat0, cand, lift_vel),
                              nan=1e9, posinf=1e9)
            # ADAPTIVE temperature. A fixed lambda collapsed the weights onto a
            # single sample (effective sample size 1.0 at lambda 0.08, 2 and 50
            # alike), because random finger perturbations destroy the grasp and
            # the cost spread is ~1e5 while the minimum is ~7e2. Scaling lambda
            # to the spread takes the effective sample size to 31-48 and lets the
            # plan actually move.
            lam = max(self.rho * float(c.std()), 1e-9)
            w = np.exp(-(c - c.min()) / lam)
            w /= w.sum() + 1e-12
            self.ess = float(1.0 / np.sum(w ** 2))
            self.U = np.clip((w[:, None, None] * cand).sum(0), self.lo, self.hi)
        return self.U[0].copy()

    def shift(self):
        self.U = np.vstack([self.U[1:], self.U[-1:]])

    def plan_drift(self, U0):
        """How far the plan has moved from its seed nominal. If this stays ~0
        the planner is not planning and the result is just the pre-grasp pose
        held constant -- which is exactly what identical results across three
        seeds would indicate."""
        return float(np.abs(self.U - U0).mean())


def run(hand="leap", width=0.065, mass=0.05, lift_h=0.15, n_ctrl=60,
        substep=10, seed=0, verbose=True, plan=True, task_jerk=True,
        seat_first=True):
    b = Bench(hand, (width / 2, 0.025, 0.025), mass)
    pl, info = b.plan(width, margin=0.006)
    if pl is None:
        return dict(hand=hand, width=width, mass=mass, success=False, **info)
    f_pre, clear = b.open_until_clear(pl)
    b.place(f_pre, pl["centre"])
    pq = b.d.qpos[b.oq:b.oq + 7].copy()
    for _ in range(150):
        b.d.ctrl[:] = b.ctrl(f_pre, 0.0)
        mujoco.mj_step(b.m, b.d)
        b.d.qpos[b.oq:b.oq + 7] = pq
        b.d.qvel[b.ov:b.ov + 6] = 0.0
    # Seat the grasp with the scripted closure BEFORE handing over to MPPI.
    # MPPI refines locally; it does not discover a grasp. Started at the
    # pre-grasp it makes small adjustments (plan drift ~0.03 rad, healthy
    # effective sample size) and lands on exactly the no-planner baseline. Given
    # a seated grasp it has something to stabilise, which is what sampling MPC
    # is actually good at. The closure itself comes from the hand-agnostic gap
    # planner, so no hand-specific scripting enters here.
    start_frac = f_pre
    if seat_first:
        for i in range(1, 31):
            fr = f_pre + (pl["f_grasp"] - f_pre) * i / 30
            b.d.ctrl[:] = b.ctrl(fr, 0.0)
            for _ in range(8):
                mujoco.mj_step(b.m, b.d)
                b.d.qpos[b.oq:b.oq + 7] = pq
                b.d.qvel[b.ov:b.ov + 6] = 0.0
        for i in range(1, 21):
            fr = pl["f_grasp"] + (pl["f_squeeze"] - pl["f_grasp"]) * i / 20
            b.d.ctrl[:] = b.ctrl(fr, 0.0)
            for _ in range(8):
                mujoco.mj_step(b.m, b.d)
                b.d.qpos[b.oq:b.oq + 7] = pq
                b.d.qvel[b.ov:b.ov + 6] = 0.0
        start_frac = pl["f_squeeze"]
        for _ in range(200):                 # release
            b.d.ctrl[:] = b.ctrl(start_frac, 0.0)
            mujoco.mj_step(b.m, b.d)

    obj0 = b.d.qpos[b.oq:b.oq + 3].copy()
    quat0 = b.d.qpos[b.oq + 3:b.oq + 7].copy()

    mppi = MPPI(b, substep=substep, seed=seed)
    mppi.U = np.tile(start_frac * b.amp, (mppi.H, 1))
    U0 = mppi.U.copy()
    hold_n = n_ctrl // 5
    sched = np.concatenate([np.zeros(hold_n),
                            np.linspace(0, lift_h, n_ctrl - hold_n)])
    if task_jerk:
        # Put the jerk INSIDE the task. With a plain vertical lift the cheapest
        # solution is CARRYING -- the object rides on the fingers, which tracks
        # the hand perfectly and needs no grip. Requiring the hand to snap
        # downward mid-episode makes carrying fail during planning, so the
        # planner has to find an actual grip. The cost is unchanged.
        j0 = int(n_ctrl * 0.75)
        sched[j0:j0 + 3] = sched[j0] - 0.05
        sched[j0 + 3:] = sched[j0]
    state0 = np.empty(mppi.nstate)
    fmax, z_tr = 0.0, []
    c = np.zeros(b.m.nu)
    for k in range(n_ctrl):
        mujoco.mj_getState(b.m, b.d, state0, FULL)
        seg = sched[k:k + mppi.H]
        if len(seg) < mppi.H:
            seg = np.concatenate([seg, np.full(mppi.H - len(seg), sched[-1])])
        if plan:
            u = mppi.plan(state0, seg.repeat(substep), obj0, quat0, b.m.opt.timestep)
        else:
            u = mppi.U[0].copy()          # baseline: hold the pre-grasp pose
        c = np.zeros(b.m.nu)
        c[b.ja] = u
        c[b.lift_a] = sched[k]
        for _ in range(substep):
            b.d.ctrl[:] = c
            mujoco.mj_step(b.m, b.d)
            fmax = max(fmax, b.force_only())
        mppi.shift()
        z_tr.append(float(b.d.qpos[b.oq + 2] - obj0[2]))
        if verbose and k % 15 == 0:
            g = b.metrics()
            print(f"   step {k:>3}  cmd {sched[k]*100:>5.1f}  obj "
                  f"{z_tr[-1]*100:>+6.2f} cm  ncon {g['n_contacts']:>2}  "
                  f"F {g['f_total']:>7.2f} N  eps {g['epsilon']:.3f}", flush=True)

    top = b.metrics()
    net = z_tr[-1]

    # JERK: drive the hand down faster than gravity. A carried object separates
    # and is left behind; a gripped one follows.
    drop, dur = 0.06, 0.06
    zh0, zo0 = float(b.d.qpos[b.lift_q]), float(b.d.qpos[b.oq + 2])
    n = max(1, int(dur / b.m.opt.timestep))
    lv0 = float(sched[-1])
    for i in range(n):
        c[b.lift_a] = lv0 - drop * (i + 1) / n
        b.d.ctrl[:] = c
        mujoco.mj_step(b.m, b.d)
    acc = 2.0 * drop / (dur ** 2)
    hand_dz = float(b.d.qpos[b.lift_q]) - zh0
    obj_dz = float(b.d.qpos[b.oq + 2]) - zo0
    follow = obj_dz / hand_dz if abs(hand_dz) > 1e-4 else 0.0
    after_jerk = b.metrics()

    # Shake about the CURRENT commanded height. Referencing lift_h was wrong
    # once the task jerk was added: the hand ends the episode lower than lift_h,
    # so the "shake" raised it 5 cm and the object dutifully followed, scoring a
    # -4 cm drop.
    base_lv = float(sched[-1])
    z_pre = float(b.d.qpos[b.oq + 2])
    for _ in range(4):
        for lv in (base_lv - 0.035, base_lv):
            c[b.lift_a] = lv
            for _ in range(12):
                b.d.ctrl[:] = c
                mujoco.mj_step(b.m, b.d)
    shook = b.metrics()
    shake_drop = z_pre - float(b.d.qpos[b.oq + 2])

    return dict(hand=hand, width=width, mass=mass, planner=bool(plan),
                plan_drift=mppi.plan_drift(U0), ess=mppi.ess,
                clear_start=bool(clear),
                f_max_transient=float(fmax), net_lift_m=float(net),
                eps_top=top["epsilon"], n_top=top["n_contacts"],
                f_top=top["f_total"], jerk_accel=float(acc),
                jerk_follow=float(follow), n_after_jerk=after_jerk["n_contacts"],
                shake_drop_m=float(shake_drop), n_shake=shook["n_contacts"],
                eps_shake=shook["epsilon"], z_trace=z_tr,
                success=bool(net >= 0.10 and top["n_contacts"] > 0
                             and follow > 0.8
                             and after_jerk["n_contacts"] > 0
                             and abs(shake_drop) < 0.02
                             and shook["n_contacts"] > 0
                             and top["f_total"] < 100.0 and clear
                             and fmax < 500.0), **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="leap")
    ap.add_argument("--width", type=float, default=0.065)
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="hold the pre-grasp pose, no planning (trivial baseline)")
    ap.add_argument("--no-task-jerk", action="store_true")
    ap.add_argument("--no-seat", action="store_true")
    ap.add_argument("--out", default="legacy/results/mppi_grasp.json")
    a = ap.parse_args()
    t0 = time.time(); rows = []
    for s in range(a.seeds):
        print(f"\n=== {a.hand}  seed {s} ===", flush=True)
        r = run(a.hand, a.width, a.mass, seed=s, verbose=not a.quiet,
                plan=not a.baseline, task_jerk=not a.no_task_jerk,
                seat_first=not a.no_seat)
        rows.append(r)
        if r.get("feasible"):
            print(f"  drift {r['plan_drift']:.4f} ess {r['ess']:.1f} | lift {r['net_lift_m']*100:+.2f} cm | eps_top {r['eps_top']:.3f}"
                  f" | F_top {r['f_top']:.2f} N | Fmax {r['f_max_transient']:.1f} N"
                  f" | jerk {r['jerk_accel']:.0f} m/s2 follow {r['jerk_follow']:.2f}"
                  f" ncon {r['n_after_jerk']} | shake {r['shake_drop_m']*100:+.2f} cm"
                  f" | ok={r['success']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2, default=float))
    ok = [r for r in rows if r.get("success")]
    lifts = [r["net_lift_m"] for r in rows if r.get("feasible")]
    if lifts:
        print(f"\n{a.hand}: {len(ok)}/{len(rows)} seeds succeed;  lift "
              f"{np.mean(lifts)*100:.1f} +/- {np.std(lifts)*100:.1f} cm"
              f"   ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
