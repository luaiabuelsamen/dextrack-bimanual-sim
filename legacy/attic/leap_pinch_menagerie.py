"""LEAP (Menagerie model) pinches a block and lifts it.

Geometry checked, not assumed: with the palm mounted as Menagerie ships it, the
fingers curl UPWARD (tips z 0.108 -> 0.183) and the thumb opposes them across x.
So the grasp region is above the palm, and a block on the floor is unreachable.
This places the block in the actual gap between thumb and fingers.

Planning: sweep the closure fraction, measure the thumb-tip to finger-mean gap
at each, and pick the fraction where the gap equals the block width. That
fraction's midpoint is the grasp centre and the gap direction is the opposition
axis. Close past it, release, then lift with the palm slide.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.epsilon import grasp_metrics
from leap_menagerie_pick import Pick, FINGERS, FLEX, THUMB


def quat_x_to(u):
    x = np.array([1.0, 0, 0]); u = u / np.linalg.norm(u)
    c = float(np.dot(x, u))
    if c > 1 - 1e-9: return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-9: return np.array([0.0, 0, 0, 1.0])
    ax = np.cross(x, u); ax /= np.linalg.norm(ax); ang = np.arccos(c)
    return np.concatenate([[np.cos(ang/2)], np.sin(ang/2)*ax])


class PinchPick(Pick):
    def qpos_for(self, frac, flex, thumb):
        """Kinematic pose at closure fraction `frac` (planning only)."""
        q = np.zeros(self.m.nq)
        q[self.oq:self.oq+3] = (0, 0, 5.0)          # park the block far away
        q[self.oq+3] = 1.0
        for f in FINGERS:
            for j, v in zip(FLEX, flex):
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{j}")
                if jid >= 0: q[self.m.jnt_qposadr[jid]] = frac * v
        for n, v in zip(THUMB, thumb):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0: q[self.m.jnt_qposadr[jid]] = frac * v
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        tips = self.d.xpos[self.tips].copy()
        return tips[:3].mean(0), tips[3]            # finger mean, thumb tip

    def plan(self, flex, thumb, width, margin=0.005, n=120):
        fr = np.linspace(0, 1, n)
        gaps, mids, axes = [], [], []
        for f in fr:
            fm, th = self.qpos_for(f, flex, thumb)
            gaps.append(float(np.linalg.norm(th - fm)))
            mids.append(0.5 * (th + fm))
            axes.append((th - fm) / (np.linalg.norm(th - fm) + 1e-12))
        gaps = np.array(gaps)
        def at(target):
            return int(np.argmin(np.abs(gaps - target)))
        io, ic = at(width + 2*margin), at(max(width - 2*margin, 0.004))
        return dict(f_open=float(fr[io]), f_close=float(fr[ic]),
                    gap_open=gaps[io], gap_close=gaps[ic],
                    centre=mids[io], axis=axes[io],
                    gap_min=float(gaps.min()), gap_max=float(gaps.max()))

    def pick(self, flex, thumb, pl, lift_h=0.15, n_way=25, hold_s=2.0):
        c, quat = pl["centre"], quat_x_to(pl["axis"])
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.oq:self.oq+3] = c
        self.d.qpos[self.oq+3:self.oq+7] = quat
        co = self.ctrl_from([pl["f_open"]*v for v in flex],
                            [pl["f_open"]*v for v in thumb], 0.0)
        cc = self.ctrl_from([pl["f_close"]*v for v in flex],
                            [pl["f_close"]*v for v in thumb], 0.0)
        self.d.qpos[self.lift_q] = 0.0
        self.d.ctrl[:] = co
        mujoco.mj_forward(self.m, self.d)
        pq = self.d.qpos[self.oq:self.oq+7].copy()
        def stp(ctl, n, pin):
            for _ in range(n):
                self.d.ctrl[:] = ctl; mujoco.mj_step(self.m, self.d)
                if pin:
                    self.d.qpos[self.oq:self.oq+7] = pq
                    self.d.qvel[self.ov:self.ov+6] = 0.0
        stp(co, 200, True)
        for i in range(1, 41):
            stp(co + (i/40)*(cc-co), 15, True)
        g = self.metrics()
        stp(cc, 500, False)                       # RELEASE
        z_after_release = self.obj_z()
        held = self.metrics()["n_contacts"] > 0 and abs(z_after_release-c[2]) < 0.015
        z0 = self.obj_z()
        for i in range(1, n_way+1):
            cl = cc.copy(); cl[self.lift_a] = -lift_h*i/n_way
            stp(cl, 30, False)
        cl = cc.copy(); cl[self.lift_a] = -lift_h
        stp(cl, int(hold_s/self.m.opt.timestep), False)
        zf = self.obj_z(); after = self.metrics()
        return dict(eps=g["epsilon"], n=g["n_contacts"], f_total=g["f_total"],
                    delta=g["delta"], held_static=bool(held),
                    release_drop_m=float(c[2]-z_after_release),
                    lift_actual=float(-self.d.qpos[self.lift_q]),
                    net_lift_m=float(zf-z0), n_after=after["n_contacts"],
                    eps_after=after["epsilon"],
                    success=bool(zf-z0 >= 0.10 and after["n_contacts"] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="legacy/results/leap_pinch_menagerie.json")
    a = ap.parse_args()
    t0=time.time(); rows=[]; best=None
    print(f"{'w(cm)':>7}{'marg':>6}{'fric':>6}{'f_o':>6}{'f_c':>6}{'gapO':>7}{'gapC':>7}"
          f"{'ncon':>6}{'F(N)':>8}{'eps':>8}{'static':>8}{'lift(cm)':>10}{'n_end':>7}{'ok':>6}")
    for w in (0.030, 0.040, 0.050):
        for fric in ("1.0 0.02 0.001", "2.0 0.05 0.002"):
            p = PinchPick(block_half=(w/2, 0.025, 0.025), block_mass=a.mass,
                          friction=fric)
            for flex, thumb in [([0.9,1.2,0.6],[1.6,0.9,1.0,0.6]),
                                ([1.2,1.4,0.8],[2.0,1.1,1.2,0.8])]:
                for margin in (0.004, 0.008):
                    pl = p.plan(flex, thumb, w, margin)
                    r = p.pick(flex, thumb, pl)
                    r.update(w=w, fric=fric, margin=margin, **{k: float(pl[k])
                             for k in ("f_open","f_close","gap_open","gap_close","gap_min","gap_max")})
                    rows.append(r)
                    print(f"{w*100:>7.1f}{margin*1000:>6.0f}{fric.split()[0]:>6}"
                          f"{pl['f_open']:>6.2f}{pl['f_close']:>6.2f}"
                          f"{pl['gap_open']*100:>7.2f}{pl['gap_close']*100:>7.2f}"
                          f"{r['n']:>6}{r['f_total']:>8.2f}{r['eps']:>8.4f}"
                          f"{str(r['held_static']):>8}{r['net_lift_m']*100:>10.2f}"
                          f"{r['n_after']:>7}{str(r['success']):>6}", flush=True)
                    if best is None or (r["success"], r["net_lift_m"]) > (best["success"], best["net_lift_m"]):
                        best = r
    Path(a.out).write_text(json.dumps(dict(rows=rows,best=best),indent=2,default=float))
    ok=[r for r in rows if r["success"]]
    print(f"\nsuccess {len(ok)}/{len(rows)}  ({time.time()-t0:.0f} s)")
    print("BEST:", json.dumps(best, indent=2, default=float))

if __name__ == "__main__":
    main()
