"""LEAP grasps a block off a table and lifts it.

The bench so far only held the block against gravity in mid-air. That is the
harder half of a pick, but it is not a lift, so this adds the rest: a table, the
block resting on it, and a vertical slide joint on the hand base so the hand can
actually rise.

The LEAP URDF has no palm body -- the four finger chains attach directly to the
world -- so a `hand_base` body carrying a slide joint is inserted and the chains
reparented under it.

Sequence: open above the block -> close on it (pinned, force-seated) -> release
the pin -> raise the base by `lift_h` -> hold. Success is the pre-registered
criterion: the block rises >= 10 cm, is still in contact at the end, and has not
slipped more than 2 cm relative to the hand.
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
from leap_bench import TIPS, tip_bodies        # noqa: E402


class LiftBench:
    def __init__(self, key="leap", half=(0.02, 0.02, 0.02), mass=0.05, kp=5.0,
                 kv=0.2, friction="2.0 0.05 0.002", table_gap=0.0):
        cfg = HANDS[key]
        root = ET.fromstring(compile_mjcf(cfg["urdf"]))
        wb = root.find("worldbody")

        # --- wrap the finger chains in a body with a vertical slide ---
        chains = [b for b in list(wb) if b.tag == "body"]
        base = ET.Element("body", dict(name="hand_base", pos="0 0 0"))
        ET.SubElement(base, "joint", dict(name="base_z", type="slide",
                                          axis="0 0 1", range="-0.05 0.6",
                                          damping="5"))
        ET.SubElement(base, "inertial", dict(pos="0 0 0", mass="0.5",
                                             diaginertia="0.01 0.01 0.01"))
        for c in chains:
            wb.remove(c)
            base.append(c)
        wb.append(base)

        names = []
        for j in root.iter("joint"):
            n = j.get("name")
            if not n or n == "base_z" or j.get("type") in ("free", "ball"):
                continue
            if cfg.get("keep_prefix") and not n.startswith(cfg["keep_prefix"]):
                continue
            names.append(n)

        act = ET.SubElement(root, "actuator")
        for n in names:
            ET.SubElement(act, "position", dict(name=f"act_{n}", joint=n,
                                                kp=str(kp), kv=str(kv)))
        ET.SubElement(act, "position", dict(name="act_base_z", joint="base_z",
                                            kp="2000", kv="150"))
        self.half = np.array(half, float)
        self.mass = mass
        ET.SubElement(wb, "geom", dict(name="table_top", type="box",
                                       size="0.4 0.4 0.02", pos="0 0 -0.02",
                                       rgba="0.75 0.65 0.5 1"))
        obj = ET.SubElement(wb, "body", dict(name="object", pos="0 0 0"))
        ET.SubElement(obj, "freejoint", dict(name="object_free"))
        ET.SubElement(obj, "geom", dict(name="object_geom", type="box",
                                        size=" ".join(str(v) for v in half),
                                        rgba="0.85 0.3 0.2 1", mass=str(mass),
                                        friction=friction))
        opt = root.find("option")
        if opt is None:
            opt = ET.SubElement(root, "option")
        opt.set("timestep", "0.002")
        opt.set("integrator", "implicitfast")

        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.names = names
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qadr = np.array([self.m.jnt_qposadr[i] for i in jid])
        r = self.m.jnt_range[jid]
        self.lo, self.hi = r[:, 0].copy(), r[:, 1].copy()
        bad = self.hi <= self.lo
        self.lo[bad], self.hi[bad] = -np.pi, np.pi
        self.aidx = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                f"act_{n}") for n in names])
        self.bz_a = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_base_z")
        self.bz_q = self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")]
        self.n = len(names)
        self.tips = tip_bodies(self.m, key)
        self.ogid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.tgid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        ofj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.oq = self.m.jnt_qposadr[ofj]
        self.ov = self.m.jnt_dofadr[ofj]

    # ---- kinematics ----
    def tip_pos(self, q, bz=0.0):
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.bz_q] = bz
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.tips].copy()

    def close_pose(self, G, restarts=6, seed=0):
        rng = np.random.default_rng(seed)
        best = None
        for i in range(restarts):
            x0 = np.zeros(self.n) if i == 0 else \
                self.lo + rng.random(self.n) * (self.hi - self.lo)
            r = minimize(lambda q: float(((self.tip_pos(q) - G) ** 2).sum()), x0,
                         method="L-BFGS-B", bounds=list(zip(self.lo, self.hi)),
                         options=dict(maxiter=600))
            if best is None or r.fun < best.fun:
                best = r
        return np.clip(best.x, self.lo, self.hi), float(best.fun)

    def converge_point(self, restarts=8, seed=0):
        rng = np.random.default_rng(seed)
        best = None
        for i in range(restarts):
            x0 = 0.5 * (self.lo + self.hi) if i == 0 else \
                self.lo + rng.random(self.n) * (self.hi - self.lo)
            r = minimize(lambda q: float(((self.tip_pos(q) -
                                           self.tip_pos(q).mean(0)) ** 2).sum()),
                         x0, method="L-BFGS-B", bounds=list(zip(self.lo, self.hi)),
                         options=dict(maxiter=600))
            if best is None or r.fun < best.fun:
                best = r
        return self.tip_pos(best.x).mean(0)

    # ---- dynamics ----
    def reset(self, obj_xy, obj_z, bz=0.0, q=None):
        mujoco.mj_resetData(self.m, self.d)
        q = np.zeros(self.n) if q is None else q
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.bz_q] = bz
        self.d.qpos[self.oq:self.oq + 3] = (obj_xy[0], obj_xy[1], obj_z)
        self.d.qpos[self.oq + 3:self.oq + 7] = (1, 0, 0, 0)
        self.d.ctrl[self.aidx] = q
        self.d.ctrl[self.bz_a] = bz
        mujoco.mj_forward(self.m, self.d)

    def step(self, q, bz, n, pin=False):
        for _ in range(n):
            self.d.ctrl[self.aidx] = q
            self.d.ctrl[self.bz_a] = bz
            mujoco.mj_step(self.m, self.d)
            if pin:
                self.d.qpos[self.oq:self.oq + 7] = self._pq
                self.d.qvel[self.ov:self.ov + 6] = 0.0

    def pin_here(self):
        self._pq = self.d.qpos[self.oq:self.oq + 7].copy()

    def metrics(self, exclude_table=True):
        ex = (self.tgid,) if exclude_table else ()
        return grasp_metrics(self.m, self.d, self.ogid, self.obid, ex)

    def seat(self, q_close, bz, f_target, n_coarse=50, settle=8):
        q_open = np.zeros(self.n)
        lo, hi, found = 0.0, 1.0, False
        for k in range(1, n_coarse + 1):
            f = k / n_coarse
            self.step(q_open + f * (q_close - q_open), bz, settle, pin=True)
            if self.metrics()["f_total"] >= f_target:
                lo, hi, found = (k - 1) / n_coarse, f, True
                break
        if not found:
            return q_close
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            self.step(q_open + mid * (q_close - q_open), bz, settle, pin=True)
            if self.metrics()["f_total"] >= f_target:
                hi = mid
            else:
                lo = mid
        q = q_open + hi * (q_close - q_open)
        self.step(q, bz, settle, pin=True)
        return q

    def pick(self, G, f_target=8.0, lift_h=0.12, n_way=12, hold_s=2.0):
        """Close on the block where it rests, release, then raise the base."""
        obj_z = float(self.half[2])            # resting on the table
        bz0 = float(G[2] - obj_z)              # slide so the grasp centre meets it
        q_cl, fit = self.close_pose(G - np.array([0, 0, bz0]))
        self.reset(G[:2], obj_z, bz=bz0)
        self.pin_here()
        self.step(np.zeros(self.n), bz0, 25, pin=True)
        q = self.seat(q_cl, bz0, f_target)
        g = self.metrics()
        self.step(q, bz0, 40)                  # RELEASE, still on the table
        z0 = float(self.d.qpos[self.oq + 2])
        rel0 = z0 - float(self.d.qpos[self.bz_q])
        for i in range(1, n_way + 1):
            self.step(q, bz0 + lift_h * i / n_way, 40)
        self.step(q, bz0 + lift_h, int(hold_s / self.m.opt.timestep))
        zf = float(self.d.qpos[self.oq + 2])
        after = self.metrics()
        slip = abs((zf - float(self.d.qpos[self.bz_q])) - rel0)
        return dict(G=[float(v) for v in G], fit=fit, eps=g["epsilon"],
                    n=g["n_contacts"], f_total=g["f_total"], delta=g["delta"],
                    net_lift_m=float(zf - z0), n_after=after["n_contacts"],
                    eps_after=after["epsilon"], slip_m=float(slip),
                    success=bool(zf - z0 >= 0.10 and after["n_contacts"] > 0
                                 and slip < 0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--size", type=float, default=0.02, help="box HALF-extent (m)")
    ap.add_argument("--out", default="legacy/results/leap_lift.json")
    a = ap.parse_args()
    t0 = time.time()
    rows, best = [], None
    print(f"{'half(cm)':>9}{'mass':>6}{'ftgt':>6}{'dx':>6}{'dz':>6}{'ncon':>6}"
          f"{'F(N)':>8}{'eps':>8}{'lift(cm)':>10}{'n_end':>7}{'slip(cm)':>10}{'ok':>6}")
    for s in (a.size, 0.025, 0.03):
        b = LiftBench(half=(s, s, s), mass=a.mass)
        G0 = b.converge_point()
        for ftgt in (5.0, 8.0, 12.0):
            for dx in (-0.01, 0.0, 0.01, 0.02):
                for dz in (-0.01, 0.0, 0.01):
                    G = G0 + np.array([dx, 0.0, dz])
                    r = b.pick(G, f_target=ftgt)
                    r.update(half=s, ftgt=ftgt, dx=dx, dz=dz)
                    rows.append(r)
                    if r["success"] or r["net_lift_m"] > 0.02:
                        print(f"{100*s:>9.1f}{a.mass:>6.2f}{ftgt:>6.1f}{dx:>6.2f}"
                              f"{dz:>6.2f}{r['n']:>6}{r['f_total']:>8.2f}"
                              f"{r['eps']:>8.4f}{r['net_lift_m']*100:>10.2f}"
                              f"{r['n_after']:>7}{r['slip_m']*100:>10.2f}"
                              f"{str(r['success']):>6}", flush=True)
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
