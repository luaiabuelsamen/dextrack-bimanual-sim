"""LEAP picks a block off a table with a real pinch grasp.

Written after looking at a rendered frame of the previous attempt, which showed
the hand balled into a fist, sunk into the table, with the block outside it and
only the thumb touching. The contact dump agreed: thumb_temp_base against the
world at 182 N, dip_2 against dip_3 at 44 N, and about 1.5 N total on the object.

Two causes, both mine:
  1. The base was lowered to bring the grasp centre down, which drove the thumb
     through the table. Here the TABLE is placed to suit the hand instead, and
     the base starts at zero.
  2. The closure aimed every fingertip at one point, so on a small block the
     fingers collide with each other rather than with the object.

The closure here is a pinch, planned the way a grasp planner would:
  - find the pose where the thumb-tip to finger-mean gap equals the block width
  - that gap's midpoint is the grasp centre, and its direction is the opposition
    axis; orient the block so its narrow dimension lies along that axis
  - close past it by a small margin, force-seated
"""
from __future__ import annotations

import argparse, json, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.epsilon import grasp_metrics     # noqa: E402
from hand_bench import compile_mjcf, HANDS     # noqa: E402


def quat_x_to(u):
    """Quaternion rotating +x onto the unit vector u."""
    x = np.array([1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    c = float(np.dot(x, u))
    if c > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = np.cross(x, u); ax /= np.linalg.norm(ax)
    ang = np.arccos(c)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])


class Pinch:
    THUMB = "thumb_fingertip"
    FINGERS = ["fingertip", "fingertip_2", "fingertip_3"]

    def __init__(self, width, height, mass, kp=5.0, kv=0.2, table_z=0.004,
                 friction="2.0 0.05 0.002", key="leap"):
        cfg = HANDS[key]
        root = ET.fromstring(compile_mjcf(cfg["urdf"]))
        wb = root.find("worldbody")
        chains = [b for b in list(wb) if b.tag == "body"]
        base = ET.Element("body", dict(name="hand_base", pos="0 0 0"))
        ET.SubElement(base, "joint", dict(name="base_z", type="slide",
                                          axis="0 0 1", range="-0.05 0.8",
                                          damping="2"))
        ET.SubElement(base, "inertial", dict(pos="0 0 0.1", mass="0.5",
                                             diaginertia="0.01 0.01 0.01"))
        for c in chains:
            wb.remove(c); base.append(c)
        wb.append(base)

        names = [j.get("name") for j in root.iter("joint")
                 if j.get("name") and j.get("name") != "base_z"
                 and j.get("type") not in ("free", "ball")]
        act = ET.SubElement(root, "actuator")
        for n in names:
            ET.SubElement(act, "position", dict(name=f"act_{n}", joint=n,
                                                kp=str(kp), kv=str(kv)))
        ET.SubElement(act, "position", dict(name="act_base_z", joint="base_z",
                                            kp="1500", kv="120"))
        ET.SubElement(wb, "geom", dict(name="table_top", type="box",
                                       size="0.4 0.4 0.02",
                                       pos=f"0 0 {table_z - 0.02}",
                                       rgba="0.75 0.65 0.5 1"))
        obj = ET.SubElement(wb, "body", dict(name="object", pos="0 0 0"))
        ET.SubElement(obj, "freejoint", dict(name="object_free"))
        ET.SubElement(obj, "geom", dict(
            name="object_geom", type="box",
            size=f"{width/2} {height/2} {height/2}",   # narrow along local x
            rgba="0.85 0.3 0.2 1", mass=str(mass), friction=friction))
        opt = root.find("option")
        if opt is None:
            opt = ET.SubElement(root, "option")
        opt.set("timestep", "0.002"); opt.set("integrator", "implicitfast")
        vis = root.find("visual") or ET.SubElement(root, "visual")
        ET.SubElement(vis, "global", dict(offwidth="640", offheight="480"))

        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.width, self.height, self.mass, self.table_z = width, height, mass, table_z
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qadr = np.array([self.m.jnt_qposadr[i] for i in jid])
        r = self.m.jnt_range[jid]
        self.lo, self.hi = r[:, 0].copy(), r[:, 1].copy()
        bad = self.hi <= self.lo
        self.lo[bad], self.hi[bad] = -np.pi, np.pi
        self.aidx = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                f"act_{n}") for n in names])
        self.n = len(names)
        self.bz_a = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_base_z")
        self.bz_q = self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")]
        self.tb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, self.THUMB)
        self.fb = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in self.FINGERS]
        self.ogid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.tgid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        ofj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.oq = self.m.jnt_qposadr[ofj]
        self.ov = self.m.jnt_dofadr[ofj]

    # ---- pinch planning ----
    def _tips(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.tb].copy(), self.d.xpos[self.fb].copy()

    def pose_for_gap(self, gap, restarts=8, seed=0):
        """Pose whose thumb-tip / finger-mean distance equals `gap`, with the
        three fingers kept spread (so they grip the object, not each other)."""
        rng = np.random.default_rng(seed)
        best = None

        def obj(q):
            t, f = self._tips(q)
            fm = f.mean(0)
            g = np.linalg.norm(t - fm)
            spread = float(((f - fm) ** 2).sum())
            return (g - gap) ** 2 * 100.0 - 0.5 * spread

        for i in range(restarts):
            x0 = np.zeros(self.n) if i == 0 else \
                self.lo + rng.random(self.n) * (self.hi - self.lo)
            r = minimize(obj, x0, method="L-BFGS-B",
                         bounds=list(zip(self.lo, self.hi)),
                         options=dict(maxiter=800))
            if best is None or r.fun < best.fun:
                best = r
        q = np.clip(best.x, self.lo, self.hi)
        t, f = self._tips(q)
        fm = f.mean(0)
        return q, t, fm, float(np.linalg.norm(t - fm))

    def plan(self, margin=0.006):
        """Open pose at the block width + margin, closed pose at width - margin."""
        q_open, t0, f0, g0 = self.pose_for_gap(self.width + 2 * margin)
        q_close, t1, f1, g1 = self.pose_for_gap(max(self.width - 2 * margin, 0.004))
        centre = 0.5 * (t0 + f0)
        axis = (t0 - f0) / (np.linalg.norm(t0 - f0) + 1e-12)
        return dict(q_open=q_open, q_close=q_close, centre=centre, axis=axis,
                    gap_open=g0, gap_close=g1)

    # ---- dynamics ----
    def step(self, q, bz, n, pin=False):
        for _ in range(n):
            self.d.ctrl[self.aidx] = q
            self.d.ctrl[self.bz_a] = bz
            mujoco.mj_step(self.m, self.d)
            if pin:
                self.d.qpos[self.oq:self.oq + 7] = self._pq
                self.d.qvel[self.ov:self.ov + 6] = 0.0

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.ogid, self.obid, (self.tgid,))

    def hand_table_force(self):
        tot = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.tgid in (c.geom1, c.geom2) and self.ogid not in (c.geom1, c.geom2):
                f = np.zeros(6); mujoco.mj_contactForce(self.m, self.d, i, f)
                tot += abs(f[0])
        return tot

    def setup(self, plan, xy_off=(0.0, 0.0)):
        """Place the block in the planned gap, standing on the table."""
        c = plan["centre"].copy()
        c[0] += xy_off[0]; c[1] += xy_off[1]
        c[2] = self.table_z + self.height / 2        # standing on the table
        quat = quat_x_to(plan["axis"])
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = plan["q_open"]
        self.d.qpos[self.bz_q] = 0.0
        self.d.qpos[self.oq:self.oq + 3] = c
        self.d.qpos[self.oq + 3:self.oq + 7] = quat
        self.d.ctrl[self.aidx] = plan["q_open"]
        self.d.ctrl[self.bz_a] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return c

    def seat(self, plan, f_target, n_coarse=40, settle=8):
        qo, qc = plan["q_open"], plan["q_close"]
        lo, hi, found = 0.0, 1.0, False
        for k in range(1, n_coarse + 1):
            f = k / n_coarse
            self.step(qo + f * (qc - qo), 0.0, settle, pin=True)
            if self.metrics()["f_total"] >= f_target:
                lo, hi, found = (k - 1) / n_coarse, f, True
                break
        if not found:
            return qc
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            self.step(qo + mid * (qc - qo), 0.0, settle, pin=True)
            if self.metrics()["f_total"] >= f_target:
                hi = mid
            else:
                lo = mid
        q = qo + hi * (qc - qo)
        self.step(q, 0.0, settle, pin=True)
        return q

    def pick(self, plan, f_target=6.0, lift_h=0.15, n_way=15, hold_s=2.0,
             xy_off=(0.0, 0.0)):
        self.setup(plan, xy_off)
        self._pq = self.d.qpos[self.oq:self.oq + 7].copy()
        self.step(plan["q_open"], 0.0, 25, pin=True)
        ht0 = self.hand_table_force()
        q = self.seat(plan, f_target)
        g = self.metrics()
        self.step(q, 0.0, 40)                        # RELEASE, block on table
        z0 = float(self.d.qpos[self.oq + 2])
        rel0 = z0 - float(self.d.qpos[self.bz_q])
        for i in range(1, n_way + 1):
            self.step(q, lift_h * i / n_way, 40)
        self.step(q, lift_h, int(hold_s / self.m.opt.timestep))
        zf = float(self.d.qpos[self.oq + 2])
        after = self.metrics()
        slip = abs((zf - float(self.d.qpos[self.bz_q])) - rel0)
        return dict(eps=g["epsilon"], n=g["n_contacts"], f_total=g["f_total"],
                    delta=g["delta"], hand_table_N=float(ht0),
                    base_cmd=float(lift_h), base_actual=float(self.d.qpos[self.bz_q]),
                    net_lift_m=float(zf - z0), n_after=after["n_contacts"],
                    eps_after=after["epsilon"], slip_m=float(slip),
                    success=bool(zf - z0 >= 0.10 and after["n_contacts"] > 0
                                 and slip < 0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="results/leap_pinch.json")
    a = ap.parse_args()
    t0 = time.time()
    rows, best = [], None
    print(f"{'w(cm)':>7}{'h(cm)':>7}{'ftgt':>6}{'gapO':>7}{'gapC':>7}{'HT(N)':>8}"
          f"{'ncon':>6}{'F(N)':>8}{'eps':>8}{'base':>8}{'lift(cm)':>10}"
          f"{'n_end':>7}{'slip':>8}{'ok':>6}")
    for w in (0.030, 0.040, 0.050):
        for hgt in (0.06, 0.08):
            for ftgt in (4.0, 8.0):
                b = Pinch(width=w, height=hgt, mass=a.mass)
                pl = b.plan()
                r = b.pick(pl, f_target=ftgt)
                r.update(w=w, h=hgt, ftgt=ftgt, gap_open=pl["gap_open"],
                         gap_close=pl["gap_close"])
                rows.append(r)
                print(f"{w*100:>7.1f}{hgt*100:>7.1f}{ftgt:>6.1f}"
                      f"{pl['gap_open']*100:>7.2f}{pl['gap_close']*100:>7.2f}"
                      f"{r['hand_table_N']:>8.1f}{r['n']:>6}{r['f_total']:>8.2f}"
                      f"{r['eps']:>8.4f}{r['base_actual']*100:>8.2f}"
                      f"{r['net_lift_m']*100:>10.2f}{r['n_after']:>7}"
                      f"{r['slip_m']*100:>8.2f}{str(r['success']):>6}", flush=True)
                if best is None or (r["success"], r["net_lift_m"]) > \
                        (best["success"], best["net_lift_m"]):
                    best = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, best=best), indent=2))
    ok = [r for r in rows if r["success"]]
    print(f"\nsuccess {len(ok)}/{len(rows)}  ({time.time()-t0:.0f} s)")
    print("BEST:", json.dumps(best, indent=2, default=float))


if __name__ == "__main__":
    main()
