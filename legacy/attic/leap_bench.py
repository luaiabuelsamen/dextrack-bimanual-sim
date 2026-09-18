"""Floating-hand grasp bench, done against the hand's real closure geometry.

The previous bench guessed a closure ("drive every joint `frac` of the way to a
limit") and drove the fingers straight through the box: LEAP scored 0/72 held at
630 N and 127 contacts, which is a property of that closure, not of LEAP.

Here the closure is SOLVED, per hand, and hand-agnostically:

    q_close(G) = argmin_q  sum_i || fingertip_i(q) - G ||^2

i.e. "bring every fingertip to the point G". Ramping from the open pose toward
that target makes the fingers converge on G; a box sitting at G stops them, which
is a grasp. No per-hand tuning, no assumed curl direction, and the same function
works for Allegro and f5d6 -- which is required, because the whole point is to
compare hands.

Measured for LEAP before building this (scripts, not assumed):
    minimum fingertip spread     rms 0.44 cm, max pairwise gap 1.25 cm
    thumb tip to finger mean     0.00 cm        (f5d6: 3.10 cm)
    natural convergence point    (0.075, 0.004, 0.046) in the hand frame

Test: close on the box, then hold against gravity with nothing underneath. The
hand base is welded to the world, so the only thing holding the box up is the
grasp.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.epsilon import grasp_metrics            # noqa: E402
from hand_bench import compile_mjcf, HANDS            # noqa: E402

TIPS = {
    "leap": ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"],
    "allegro": None,          # resolved by heuristic below
    "f5d6": ["R_th_l2", "R_ff_l2", "R_mf_l2", "R_rf_l2", "R_lf_l2"],
}


def tip_bodies(m, key):
    names = TIPS.get(key)
    if names:
        return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    # heuristic: leaf bodies (no body has them as parent)
    parents = set(m.body_parentid[1:])
    return [b for b in range(1, m.nbody) if b not in parents]


class Hand:
    def __init__(self, key, box_half=(0.04, 0.045, 0.07), box_mass=0.05,
                 kp=5.0, kv=0.2, friction="2.0 0.05 0.002"):
        import xml.etree.ElementTree as ET
        cfg = HANDS[key]
        root = ET.fromstring(compile_mjcf(cfg["urdf"]))
        names = []
        for j in root.iter("joint"):
            n = j.get("name")
            if not n or j.get("type") in ("free", "ball"):
                continue
            if cfg.get("keep_prefix") and not n.startswith(cfg["keep_prefix"]):
                continue
            names.append(n)
        act = ET.SubElement(root, "actuator")
        for n in names:
            ET.SubElement(act, "position", dict(name=f"act_{n}", joint=n,
                                                kp=str(kp), kv=str(kv)))
        wb = root.find("worldbody")
        obj = ET.SubElement(wb, "body", dict(name="object", pos="0 0 0"))
        ET.SubElement(obj, "freejoint", dict(name="object_free"))
        ET.SubElement(obj, "geom", dict(
            name="object_geom", type="box",
            size=" ".join(str(v) for v in box_half), rgba="0.85 0.3 0.2 1",
            mass=str(box_mass), friction=friction))
        opt = root.find("option") or ET.SubElement(root, "option")
        opt.set("timestep", "0.002")
        opt.set("integrator", "implicitfast")

        self.key = key
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
        self.n = len(names)
        self.tips = tip_bodies(self.m, key)
        self.ogid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        ofj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.oq = self.m.jnt_qposadr[ofj]
        self.ov = self.m.jnt_dofadr[ofj]
        self.half = np.array(box_half, float)
        self.mass = box_mass

    # ---- kinematics ----
    def tip_pos(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.tips].copy()

    def converge_point(self, restarts=10, seed=0):
        """Where the fingertips naturally meet -- the grasp centre."""
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
        return self.tip_pos(best.x).mean(0), float(best.fun)

    def close_pose(self, G, restarts=6, seed=0):
        """argmin_q sum_i ||tip_i(q) - G||^2 -- close around the point G."""
        rng = np.random.default_rng(seed)
        best = None
        for i in range(restarts):
            x0 = np.zeros(self.n) if i == 0 else \
                self.lo + rng.random(self.n) * (self.hi - self.lo)
            r = minimize(lambda q: float(((self.tip_pos(q) - G) ** 2).sum()),
                         x0, method="L-BFGS-B", bounds=list(zip(self.lo, self.hi)),
                         options=dict(maxiter=600))
            if best is None or r.fun < best.fun:
                best = r
        return np.clip(best.x, self.lo, self.hi), float(best.fun)

    # ---- dynamics ----
    def reset(self, box_pos, q=None):
        mujoco.mj_resetData(self.m, self.d)
        q = np.zeros(self.n) if q is None else q
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.oq:self.oq + 3] = box_pos
        self.d.qpos[self.oq + 3:self.oq + 7] = (1, 0, 0, 0)
        self.d.ctrl[self.aidx] = q
        mujoco.mj_forward(self.m, self.d)

    def pin_here(self):
        self._pq = self.d.qpos[self.oq:self.oq + 7].copy()

    def step(self, q, n, pin=False):
        for _ in range(n):
            self.d.ctrl[self.aidx] = q
            mujoco.mj_step(self.m, self.d)
            if pin:
                self.d.qpos[self.oq:self.oq + 7] = self._pq
                self.d.qvel[self.ov:self.ov + 6] = 0.0

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.ogid, self.obid)

    def seat(self, q_open, q_close, f_target=2.0, n_coarse=50, settle=8):
        """Ramp closed on a PINNED object, stop at first contact force >=
        f_target, then back the command off by bisection so the hold is a small
        position offset past contact rather than a command deep inside it.

        The object must be pinned here. It is unsupported -- that is the whole
        test -- so without pinning it free-falls during the ~0.8 s closure ramp
        and the fingers close on nothing. An earlier version omitted this and
        reported drops of 44-69 m, which is just 0.5*g*t^2."""
        lo, hi, found = 0.0, 1.0, False
        for k in range(1, n_coarse + 1):
            f = k / n_coarse
            self.step(q_open + f * (q_close - q_open), settle, pin=True)
            if self.metrics()["f_total"] >= f_target:
                lo, hi, found = (k - 1) / n_coarse, f, True
                break
        if not found:
            return q_close
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            self.step(q_open + mid * (q_close - q_open), settle, pin=True)
            if self.metrics()["f_total"] >= f_target:
                hi = mid
            else:
                lo = mid
        q = q_open + hi * (q_close - q_open)
        self.step(q, settle, pin=True)
        return q

    def trial(self, G, f_target=2.0, hold_s=3.0):
        """Close on the box at G, then hold it against gravity. Nothing beneath."""
        q_cl, fit = self.close_pose(G)
        self.reset(G)
        self.pin_here()
        self.step(np.zeros(self.n), 30, pin=True)
        z_start = float(self.d.qpos[self.oq + 2])
        q_hold = self.seat(np.zeros(self.n), q_cl, f_target)
        g = self.metrics()
        z0 = float(self.d.qpos[self.oq + 2])
        self.step(q_hold, int(hold_s / self.m.opt.timestep))   # RELEASE
        after = self.metrics()
        drop = z0 - float(self.d.qpos[self.oq + 2])
        held = bool(drop < 0.01 and after["n_contacts"] > 0)
        return dict(G=[float(v) for v in G], fit=fit, eps=g["epsilon"],
                    n=g["n_contacts"], f_total=g["f_total"], delta=g["delta"],
                    settle_drop=float(z_start - z0), drop_m=float(drop),
                    n_after=after["n_contacts"], eps_after=after["epsilon"],
                    held=held)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="leap")
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--kp", type=float, default=5.0)
    ap.add_argument("--f-target", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()

    h = Hand(a.hand, box_mass=a.mass, kp=a.kp)
    G0, spread = h.converge_point()
    print(f"{a.hand}: {h.n} actuated joints, {len(h.tips)} fingertips, "
          f"box {2*h.half*100} cm / {h.mass} kg  (w_req {h.mass*9.81:.3f} N)")
    print(f"  natural convergence point {np.round(G0,4)}  (spread rms "
          f"{np.sqrt(spread/len(h.tips))*100:.2f} cm)")

    rows, best = [], None
    print(f"\n{'dx':>6}{'dy':>6}{'dz':>6}{'fit':>9}{'ncon':>6}{'F(N)':>9}"
          f"{'eps':>8}{'drop(cm)':>10}{'n_end':>7}{'held':>6}")
    for dx in (-0.02, 0.0, 0.02):
        for dy in (-0.02, 0.0, 0.02):
            for dz in (-0.02, 0.0, 0.02, 0.04):
                G = G0 + np.array([dx, dy, dz])
                r = h.trial(G, f_target=a.f_target)
                r.update(dx=dx, dy=dy, dz=dz)
                rows.append(r)
                print(f"{dx:>6.2f}{dy:>6.2f}{dz:>6.2f}{r['fit']:>9.5f}{r['n']:>6}"
                      f"{r['f_total']:>9.2f}{r['eps']:>8.4f}{r['drop_m']*100:>10.2f}"
                      f"{r['n_after']:>7}{str(r['held']):>6}", flush=True)
                if best is None or (r["held"], -r["drop_m"]) > (best["held"], -best["drop_m"]):
                    best = r
    n_held = sum(r["held"] for r in rows)
    print(f"\n{a.hand}: held {n_held}/{len(rows)}   max eps "
          f"{max(r['eps'] for r in rows):.4f}   best drop {best['drop_m']*100:.2f} cm")
    out = a.out or f"legacy/results/leap_bench_{a.hand}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(dict(hand=a.hand, mass=a.mass, kp=a.kp,
                                         G0=[float(v) for v in G0], rows=rows,
                                         n_held=int(n_held)), indent=2))
    print(f"wrote {out} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
