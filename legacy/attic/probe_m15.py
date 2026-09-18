"""M1.5 probe: does the keypoint objective pick the configuration that fails?

Three configurations of the Vega's f5d6 hands on the BIM_BOX (8 x 9 x 14 cm,
0.05 kg), all scored on the same three numbers -- epsilon, keypoint distance to
a human grasp, and physical outcome:

  A  pose-retargeted   : minimise keypoint distance to a human grasp of the box.
                         The human grasps it with ONE hand (thumb opposing four
                         fingers across the 8 cm face), so pose retargeting
                         preserves that allocation and produces a one-handed
                         f5d6 grasp.
  B  epsilon-max bimanual : free to re-allocate contacts across both hands.
                            Searched over face inset, contact height and squeeze.
  C  epsilon-max unimanual : the control that stops B from being "two hands beat
                             one hand". Best epsilon reachable with ONE hand, over
                             the thumb-opposition family.

The human grasp is SYNTHETIC -- an analytically placed thumb-versus-fingers
grasp on the box, not MANO, because no human corpus is on disk yet (M0 gate).
It is defined once, before any measurement, in `human_grasp()`.

Physical outcome is measured two ways, per the pre-registered criteria:
  hold : settle the grip, then remove the table's support and step 2 s. Held if
         the object drops < 1 cm. This isolates the GRIP from arm reachability,
         which on this robot is a separate known limitation.
  lift : from the settled grip, drive the arms up 10 cm. Success if the object
         rises >= 10 cm and is still within 5 cm of the hand midline at the end.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dextrack_vega"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dextrack_vega import config as C                      # noqa: E402
from dextrack_vega.envs.tracking_env import VegaTrackingEnv  # noqa: E402
from analysis.epsilon import grasp_metrics                 # noqa: E402

# Spawn height CORRECTED. dextrack_vega's BIM_BOX used z=0.80, which puts the
# box bottom at 0.730 against a table top at 0.770 -- 4 cm INSIDE the table. The
# box pops out on its own, so the do-nothing baseline there is +3.99 cm. See
# NOTES.md 2026-09-11. Here the box rests on the table.
BIM_BOX = dict(obj_type="box", obj_dims=(0.04, 0.045, 0.07),
               obj_pos=(0.55, 0.0, 0.8401), obj_mass=0.05)

OPEN = {"th_j0": 0.0, "th_j1": 0.0, "th_j2": 0.0, "ff_j1": 0.0, "ff_j2": 0.0,
        "mf_j1": 0.0, "mf_j2": 0.0, "rf_j1": 0.0, "rf_j2": 0.0,
        "lf_j1": 0.0, "lf_j2": 0.0}
SQUEEZE = {"th_j0": 1.2, "th_j1": 0.4, "th_j2": -0.6,
           "ff_j1": -0.7, "ff_j2": -0.9, "mf_j1": -0.7, "mf_j2": -0.9,
           "rf_j1": -0.7, "rf_j2": -0.9, "lf_j1": -0.7, "lf_j2": -0.9}
# Thumb-opposition pinch: thumb driven to its opposition limit, fingers curled to
# meet it. This is the family the earlier f5d6 search covered.
PINCH = {"th_j0": 1.6, "th_j1": 0.18, "th_j2": -0.43,
         "ff_j1": -1.0, "ff_j2": -1.2, "mf_j1": -1.0, "mf_j2": -1.2,
         "rf_j1": -1.0, "rf_j2": -1.2, "lf_j1": -1.0, "lf_j2": -1.2}


class Probe:
    def __init__(self, seed=0, **kw):
        self.sides = ["R", "L"]
        self.env = VegaTrackingEnv(sides=self.sides, seed=seed, **kw)
        self.m, self.d = self.env.model, self.env.data
        self.ctrl_joints = self.env.ctrl_joints
        self.obj_q = self.env._obj_qadr
        self.obj_bid = self.env._bid("object")
        self.obj_gid = self._gid("object_geom")
        self.table_gid = self._gid("table_top")
        self.floor_gid = self._gid("floor")
        self.block = {"R": 0, "L": 18}
        self.arm_qadr, self.arm_dof, self.arm_lo, self.arm_hi = {}, {}, {}, {}
        self.ft_bids, self.wrist_bid = {}, {}
        for k, s in enumerate(self.sides):
            jids = [self._jid(j) for j in C.ARM_JOINTS[s]]
            self.arm_qadr[s] = np.array([self.m.jnt_qposadr[i] for i in jids])
            self.arm_dof[s] = np.array([self.m.jnt_dofadr[i] for i in jids])
            r = self.m.jnt_range[jids]
            self.arm_lo[s], self.arm_hi[s] = r[:, 0], r[:, 1]
            self.ft_bids[s] = self.env._ft_bids[5 * k:5 * k + 5]
            self.wrist_bid[s] = self.env._bid(C.WRIST_BODY[s])
        jr = self.m.jnt_range[[self._jid(j) for j in self.ctrl_joints]]
        self.lo36, self.hi36 = jr[:, 0].copy(), jr[:, 1].copy()

    def _jid(self, n): return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
    def _gid(self, n): return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)

    # ---------------- state helpers ----------------
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        for j, v in C.HOME_POSTURE.items():
            self.d.qpos[self.m.jnt_qposadr[self._jid(j)]] = v
        self.m.geom_contype[self.table_gid] = 1
        self.m.geom_conaffinity[self.table_gid] = 1
        mujoco.mj_forward(self.m, self.d)

    def q36(self):
        return self.d.qpos[self.env._jnt_qposadr].copy()

    def set_q36(self, q):
        self.d.qpos[self.env._jnt_qposadr] = q
        mujoco.mj_forward(self.m, self.d)

    def obj_pos(self):
        return self.d.qpos[self.obj_q:self.obj_q + 3].copy()

    def ft(self, side):
        return self.d.xpos[self.ft_bids[side]].copy()

    def wrist(self, side):
        return self.d.xpos[self.wrist_bid[side]].copy()

    def fingers36(self, q, side, table):
        """Write a {suffix: value} finger table into a 36-vector."""
        q = q.copy(); b = self.block[side]
        for i, j in enumerate(self.ctrl_joints[b + 7:b + 18]):
            for suf, val in table.items():
                if j.endswith(suf):
                    q[b + 7 + i] = val
        return q

    def drive(self, target, n, sub=None, pin=False):
        """Ramp ctrl to `target` over n control steps, stepping physics."""
        sub = sub or C.CONTROL_DECIMATION
        start = self.q36()
        for k in range(n):
            a = (k + 1) / n
            s = 3 * a ** 2 - 2 * a ** 3
            self.d.ctrl[self.env._act_ctrl_idx] = start + s * (target - start)
            for _ in range(sub):
                mujoco.mj_step(self.m, self.d)
                if pin:
                    self._pin()

    def hold_ctrl(self, target, n, pin=False):
        for _ in range(n):
            self.d.ctrl[self.env._act_ctrl_idx] = target
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(self.m, self.d)
                if pin:
                    self._pin()

    def _pin(self):
        """Hold the object at its pinned pose. Standard grasp-evaluation
        practice: close the hand on a stationary object, then release. Without
        it a position-controlled arm driving into a 0.05 kg free box simply
        shoves it away, which confounds "could not form a grip" with "knocked it
        out of reach" -- observed on every unimanual candidate."""
        self.d.qpos[self.obj_q:self.obj_q + 7] = self._pin_q
        a = self.m.jnt_dofadr[self._jid("object_free")]
        self.d.qvel[a:a + 6] = 0.0

    def pin_here(self):
        self._pin_q = self.d.qpos[self.obj_q:self.obj_q + 7].copy()

    def ik_side_to(self, side, goal, iters=250, damp=0.12, step_clip=0.05):
        q_save, v_save = self.d.qpos.copy(), self.d.qvel.copy()
        qadr, dof = self.arm_qadr[side], self.arm_dof[side]
        lo, hi = self.arm_lo[side], self.arm_hi[side]
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            err = goal - self.ft(side).mean(0)
            if np.linalg.norm(err) < 3e-3:
                break
            J = np.zeros((3, self.m.nv))
            for b in self.ft_bids[side]:
                mujoco.mj_jacBody(self.m, self.d, jp, jr, b); J += jp
            J /= len(self.ft_bids[side])
            Ja = J[:, dof]
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + damp ** 2 * np.eye(3), err)
            self.d.qpos[qadr] = np.clip(self.d.qpos[qadr] + np.clip(dq, -step_clip, step_clip), lo, hi)
        sol = self.d.qpos[qadr].copy()
        self.d.qpos[:], self.d.qvel[:] = q_save, v_save
        mujoco.mj_forward(self.m, self.d)
        return sol

    # ---------------- metrics ----------------
    def eps_now(self, exclude_table=True):
        ex = (self.table_gid, self.floor_gid) if exclude_table else ()
        return grasp_metrics(self.m, self.d, self.obj_gid, self.obj_bid, ex)

    # ---------------- tests ----------------
    def hold_test(self, target, settle=40, hold_s=2.0):
        """Remove table support, hold ctrl, measure drop. Grip test only."""
        self.hold_ctrl(target, settle)
        pre = self.eps_now()
        z0 = self.obj_pos()[2]
        self.m.geom_contype[self.table_gid] = 0
        self.m.geom_conaffinity[self.table_gid] = 0
        mujoco.mj_forward(self.m, self.d)
        self.hold_ctrl(target, 10)
        free = self.eps_now()
        n = int(hold_s / self.env.dt)
        self.hold_ctrl(target, n)
        drop = float(z0 - self.obj_pos()[2])
        self.m.geom_contype[self.table_gid] = 1
        self.m.geom_conaffinity[self.table_gid] = 1
        return dict(drop_m=drop, held=bool(drop < 0.01), eps_on_table=pre,
                    eps_free=free)

    def lift_test(self, sides, contacts, fingers, lift_h=0.15, n_way=8):
        """From the settled grip, drive the arms up. Object-trajectory success."""
        up = np.array([0.0, 0.0, 1.0])
        z0 = self.obj_pos()[2]
        zmax = z0
        q = self.q36()
        for i in range(1, n_way + 1):
            dz = lift_h * i / n_way
            for s in sides:
                q[self.block[s]:self.block[s] + 7] = self.ik_side_to(s, contacts[s] + dz * up)
            for s in sides:
                q = self.fingers36(q, s, fingers[s])
            self.drive(q, 26)
            zmax = max(zmax, self.obj_pos()[2])
        self.hold_ctrl(q, 60)      # 3 s hold at the top
        zf = self.obj_pos()[2]
        mid = np.mean([self.ft(s).mean(0) for s in sides], axis=0)
        off = float(np.linalg.norm(self.obj_pos()[:2] - mid[:2]))
        return dict(net_lift_m=float(zf - z0), max_lift_m=float(zmax - z0),
                    midline_offset_m=off,
                    success=bool((zf - z0) >= 0.10 and off < 0.05))


# --------------------------------------------------------------------------
# The human grasp (synthetic; fixed before any measurement)
# --------------------------------------------------------------------------

def human_grasp(center, half):
    """A one-handed human grasp of the box: thumb pad on the +x face, four
    fingertips spread down the -x face, wrist back along -y. Opposition is
    across the 8 cm dimension, which is what a person actually does with a box
    this size. Returns (wrist, fingertips[5]) ordered [th, ff, mf, rf, lf]."""
    c = np.asarray(center, dtype=float)
    hx, _, _ = half
    th = c + np.array([+hx, 0.0, +0.010])
    ff = c + np.array([-hx, 0.0, +0.035])
    mf = c + np.array([-hx, 0.0, +0.012])
    rf = c + np.array([-hx, 0.0, -0.012])
    lf = c + np.array([-hx, 0.0, -0.035])
    wrist = c + np.array([0.0, -0.10, 0.02])
    return wrist, np.stack([th, ff, mf, rf, lf])


def keypoint_distance(p, side, h_wrist, h_ft):
    """dex-retargeting's objective: mean error of wrist->fingertip VECTORS.
    Also returns the absolute fingertip error for reference."""
    r_w, r_ft = p.wrist(side), p.ft(side)
    v_rob = r_ft - r_w
    v_hum = h_ft - h_wrist
    return (float(np.linalg.norm(v_rob - v_hum, axis=1).mean()),
            float(np.linalg.norm(r_ft - h_ft, axis=1).mean()))


# --------------------------------------------------------------------------
# Configuration A: keypoint retargeting
# --------------------------------------------------------------------------

def _bounds(lo, hi):
    lo, hi = lo.copy(), hi.copy()
    bad = hi <= lo                      # unlimited joints come back as (0, 0)
    lo[bad], hi[bad] = -np.pi, np.pi
    return lo, hi


def retarget_keypoints(p, side, h_wrist, h_ft, restarts=6, seed=0):
    """Minimise the standard dex-retargeting objective: wrist->fingertip vector
    error, plus a wrist position term (which is how the arm gets placed).

    Resets first. An earlier version seeded the optimiser from whatever pose the
    robot happened to be in and used that state for the joints it does not
    optimise, so the solution depended on execution order: three clean runs gave
    ncon=2 / kp=13.21 cm identically, but running a random-walk baseline first
    gave ncon=0 / kp=13.60 and ncon=5 / kp=15.76. The A row of an earlier
    results table was computed that way and its contact count is not meaningful.
    """
    p.reset()
    b = p.block[side]
    lo, hi = _bounds(p.lo36[b:b + 18], p.hi36[b:b + 18])
    q_full = p.q36()
    v_hum = h_ft - h_wrist

    def obj(x):
        q = q_full.copy(); q[b:b + 18] = x
        p.d.qpos[p.env._jnt_qposadr] = q
        mujoco.mj_kinematics(p.m, p.d)
        v_rob = p.ft(side) - p.wrist(side)
        return float(np.sum((v_rob - v_hum) ** 2)
                     + np.sum((p.wrist(side) - h_wrist) ** 2))

    rng = np.random.default_rng(seed)
    x0s = [np.clip(q_full[b:b + 18], lo, hi)]
    x0s += [lo + rng.random(18) * (hi - lo) for _ in range(restarts - 1)]
    best = None
    for x0 in x0s:
        r = minimize(obj, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(maxiter=500))
        if best is None or r.fun < best.fun:
            best = r
    q = q_full.copy(); q[b:b + 18] = best.x
    return q, float(best.fun)


def eval_config(p, q_target, sides, standoff=0.08, settle=25):
    """Approach clear of the object, descend and close on a PINNED object, then
    release and let the grip settle.

    Two artifacts this avoids, both observed on the first run:
      - teleporting the hand to the target pose puts fingers inside the box;
        MuJoCo resolves that penetration explosively (100+ N) and ejects it, so
        there are no contacts left to score.
      - a single position-controlled arm pressing a 0.05 kg box on a table
        shoves it away before any grip forms, so every unimanual candidate came
        back with zero contacts -- which confounds "cannot form a grip" with
        "knocked it out of reach".
    Pinning during the close is standard grasp-evaluation practice; the release
    is what makes the measurement physical again. `eps_pinned` is the contact
    set the hand can form, `epsilon` is what survives release.
    """
    p.reset()
    centre = p.obj_pos()
    p.set_q36(q_target)
    goal = {s: p.ft(s).mean(0) for s in sides}
    away = {}
    for s in sides:
        v = goal[s] - centre
        v[2] = 0.0
        nv = np.linalg.norm(v)
        away[s] = (v / nv) if nv > 1e-6 else np.array(
            [0.0, -1.0 if s == "R" else 1.0, 0.0])

    p.reset()
    q_app = q_target.copy()
    for s in sides:
        q_app[p.block[s]:p.block[s] + 7] = p.ik_side_to(s, goal[s] + standoff * away[s])
        q_app = p.fingers36(q_app, s, OPEN)
    p.set_q36(q_app)                  # clear of the object: safe to teleport to
    p.pin_here()
    p.hold_ctrl(q_app, 10, pin=True)

    q_desc = q_target.copy()          # arms in, fingers still open
    for s in sides:
        q_desc = p.fingers36(q_desc, s, OPEN)
    p.drive(q_desc, 40, pin=True)     # descend
    p.drive(q_target, 40, pin=True)   # close
    p.hold_ctrl(q_target, 15, pin=True)

    m_pinned = p.eps_now()            # the contact set the hand can form
    p.hold_ctrl(q_target, settle)     # RELEASE
    m = p.eps_now()
    m["eps_pinned"] = m_pinned["epsilon"]
    m["n_pinned"] = m_pinned["n_contacts"]
    m["f_pinned"] = m_pinned["f_total"]
    m["displaced_m"] = float(np.linalg.norm(p.obj_pos() - centre))
    m["ejected"] = bool(m["displaced_m"] > 0.05)
    return m


# --------------------------------------------------------------------------
# Configuration B / C searches
# --------------------------------------------------------------------------

def lerp_fingers(a, b, s):
    return {k: (1 - s) * a[k] + s * b[k] for k in a}


def search_bimanual(p, center, hy, insets, dzs, sqs, log):
    best = None
    for inset in insets:
        p.reset()
        for dz in dzs:
            gR = center + np.array([0.0, -(hy - inset), dz])
            gL = center + np.array([0.0, +(hy - inset), dz])
            p.reset()
            aR = p.ik_side_to("R", gR)
            aL = p.ik_side_to("L", gL)
            for sq in sqs:
                q = p.q36()
                q[0:7], q[18:25] = aR, aL
                f = lerp_fingers(OPEN, SQUEEZE, sq)
                q = p.fingers36(q, "R", f); q = p.fingers36(q, "L", f)
                mres = eval_config(p, q, ["R", "L"])
                rec = dict(inset=float(inset), dz=float(dz), sq=float(sq),
                           eps=mres["epsilon"], n=mres["n_contacts"],
                           eps_pinned=mres["eps_pinned"], n_pinned=mres["n_pinned"],
                           f_total=mres["f_total"], delta=mres["delta"],
                           displaced=mres["displaced_m"], ejected=mres["ejected"])
                log.append(rec)
                if best is None or rec["eps_pinned"] > best[0]["eps_pinned"]:
                    best = (rec, q.copy(), {"R": gR, "L": gL}, f)
    return best


def search_unimanual(p, center, dxs, dys, dzs, closures, log):
    best = None
    for dx in dxs:
        for dy in dys:
            for dz in dzs:
                g = center + np.array([dx, dy, dz])
                p.reset()
                a = p.ik_side_to("R", g)
                for cname, cl in closures:
                    q = p.q36(); q[0:7] = a
                    q = p.fingers36(q, "R", cl)
                    mres = eval_config(p, q, ["R"])
                    rec = dict(dx=float(dx), dy=float(dy), dz=float(dz),
                               closure=cname, eps=mres["epsilon"],
                               n=mres["n_contacts"], eps_pinned=mres["eps_pinned"],
                               n_pinned=mres["n_pinned"], f_total=mres["f_total"],
                               delta=mres["delta"], displaced=mres["displaced_m"],
                               ejected=mres["ejected"])
                    log.append(rec)
                    if best is None or rec["eps_pinned"] > best[0]["eps_pinned"]:
                        best = (rec, q.copy(), {"R": g}, cl)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="legacy/results/probe_m15.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    p = Probe(seed=a.seed, **BIM_BOX)
    p.reset()
    center = p.obj_pos()
    half = np.array(p.m.geom_size[p.obj_gid][:3], dtype=float)
    hy = float(half[1])
    print(f"object centre {center}  half-extents {half}  mass "
          f"{p.m.body_mass[p.obj_bid]:.3f} kg   w_req={p.m.body_mass[p.obj_bid]*9.81:.3f} N")

    h_wrist, h_ft = human_grasp(center, half)
    out = dict(object=dict(center=center.tolist(), half=half.tolist(),
                           mass=float(p.m.body_mass[p.obj_bid])),
               human_grasp=dict(wrist=h_wrist.tolist(), fingertips=h_ft.tolist()),
               configs={}, search={})

    # ---- A: pose retargeting ------------------------------------------------
    print("\n[A] retargeting keypoints ...", flush=True)
    qA, fA = retarget_keypoints(p, "R", h_wrist, h_ft, seed=a.seed)
    mA = eval_config(p, qA, ["R"])
    kA = keypoint_distance(p, "R", h_wrist, h_ft)
    holdA = p.hold_test(qA)
    print(f"    obj={fA:.5f}  eps={mA['epsilon']:.5f}  ncon={mA['n_contacts']}"
          f"  kp_vec={kA[0]*100:.2f} cm  drop={holdA['drop_m']*100:.1f} cm")
    out["configs"]["A_pose_retarget"] = dict(
        kind="unimanual", metrics=mA, keypoint_vec_m=kA[0], keypoint_abs_m=kA[1],
        retarget_objective=fA, hold=holdA, q=qA.tolist())

    # ---- C: best unimanual epsilon -----------------------------------------
    print("\n[C] searching unimanual epsilon ...", flush=True)
    logC = []
    closures = [("pinch", PINCH), ("squeeze", SQUEEZE),
                ("squeeze85", lerp_fingers(OPEN, SQUEEZE, 0.85))]
    bC = search_unimanual(p, center, [-0.02, 0.0, 0.02], [-0.02, 0.0, 0.02],
                          [-0.02, 0.02], closures, logC)
    recC, qC, gC, clC = bC
    mC = eval_config(p, qC, ["R"])
    kC = keypoint_distance(p, "R", h_wrist, h_ft)
    holdC = p.hold_test(qC)
    print(f"    best {recC}  kp_vec={kC[0]*100:.2f} cm  drop={holdC['drop_m']*100:.1f} cm")
    out["search"]["unimanual"] = logC
    out["configs"]["C_eps_unimanual"] = dict(
        kind="unimanual", best=recC, metrics=mC, keypoint_vec_m=kC[0],
        keypoint_abs_m=kC[1], hold=holdC, q=qC.tolist())

    # ---- B: best bimanual epsilon ------------------------------------------
    print("\n[B] searching bimanual epsilon ...", flush=True)
    logB = []
    bB = search_bimanual(p, center, hy, np.linspace(0.0, 0.02, 5),
                         [-0.02, 0.0, 0.02], [0.7, 0.85, 1.0], logB)
    recB, qB, gB, fB_ = bB
    mB = eval_config(p, qB, ["R", "L"])
    kB = min(keypoint_distance(p, s, h_wrist, h_ft) for s in ("R", "L"))
    holdB = p.hold_test(qB)
    print(f"    best {recB}  kp_vec={kB[0]*100:.2f} cm  drop={holdB['drop_m']*100:.1f} cm")
    out["search"]["bimanual"] = logB
    out["configs"]["B_eps_bimanual"] = dict(
        kind="bimanual", best=recB, metrics=mB, keypoint_vec_m=kB[0],
        keypoint_abs_m=kB[1], hold=holdB, q=qB.tolist())

    # ---- lift tests ---------------------------------------------------------
    print("\n[lift] B bimanual ...", flush=True)
    eval_config(p, qB, ["R", "L"])
    p.hold_ctrl(qB, 30)
    liftB = p.lift_test(["R", "L"], gB, {"R": fB_, "L": fB_})
    print(f"    {liftB}")
    out["configs"]["B_eps_bimanual"]["lift"] = liftB

    print("[lift] C eps-unimanual ...", flush=True)
    eval_config(p, qC, ["R"])
    p.hold_ctrl(qC, 30)
    liftC = p.lift_test(["R"], {"R": p.ft("R").mean(0)}, {"R": clC})
    print(f"    {liftC}")
    out["configs"]["C_eps_unimanual"]["lift"] = liftC

    print("[lift] A pose-retargeted ...", flush=True)
    eval_config(p, qA, ["R"])
    p.hold_ctrl(qA, 30)
    gA = {"R": p.ft("R").mean(0)}
    fA_tab = {suf: qA[7 + i] for i, suf in enumerate(
        [j.split("_", 1)[1] for j in p.ctrl_joints[7:18]])}
    liftA = p.lift_test(["R"], gA, {"R": fA_tab})
    print(f"    {liftA}")
    out["configs"]["A_pose_retarget"]["lift"] = liftA

    out["wall_s"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}  ({out['wall_s']:.0f} s)")

    # ---- the table the probe exists to produce ------------------------------
    print("\n" + "=" * 78)
    print(f"{'config':<22}{'eps':>9}{'F_tot(N)':>10}{'delta':>9}"
          f"{'kp_vec(cm)':>12}{'drop(cm)':>10}{'lift(cm)':>10}")
    for name in ("A_pose_retarget", "C_eps_unimanual", "B_eps_bimanual"):
        c = out["configs"][name]
        m = c["metrics"]
        dl = m["delta"]
        print(f"{name:<22}{m['epsilon']:>9.4f}{m['f_total']:>10.3f}"
              f"{(dl if np.isfinite(dl) else float('inf')):>9.2f}"
              f"{c['keypoint_vec_m']*100:>12.2f}"
              f"{c['hold']['drop_m']*100:>10.1f}"
              f"{c['lift']['net_lift_m']*100:>10.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
