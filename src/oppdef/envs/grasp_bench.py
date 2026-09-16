"""Clean grasp bench: one hand, one block, a real pre-grasp -> grasp -> squeeze.

Replaces `leap_final.py`, whose result was correct but carried two artifacts:

  - a 4812 N transient during closure, because the fingers were driven from
    fully open onto a pinned block;
  - a saturated closure fraction (`f_touch = 1.0`), because for a 4.5 cm block
    the closure family's gap range is 4.92-19.26 cm -- no intermediate fraction
    reaches 4.5 cm, so it clamped to full flex and drove straight through.

Both are fixed the way Dexonomy's data is structured: three poses, not a ramp
from open.

  pre-grasp  gap = width + margin   fingers already around the block, not touching
  grasp      gap = width            first contact
  squeeze    gap = width - margin   the grip force

The simulation STARTS at the pre-grasp pose, so the fingers never travel through
the object, and feasibility is checked before anything runs: a block is only
tested if the closure family's gap actually spans its width.

Success requires all of: lift >= 10 cm, contacts at the top, epsilon > 0.01 at
the top (force closure, not the block resting on the hand), survival of a shake,
and a sustained grip force under 100 N.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco

from oppdef.paths import (MENAGERIE, VEGA_URDF,  # noqa: E402
                          compile_urdf, menagerie_xml)

from oppdef.grasping.epsilon import grasp_metrics, object_contacts    # noqa: E402

# flex, and a closure direction per joint. Everything else is derived.
HANDS = {
    "leap": dict(
        xml=MENAGERIE / "leap_hand/right_hand.xml",
        palm="palm",
        tips=["if_ds", "mf_ds", "rf_ds"], thumb_tip="th_ds",
        flex={"if_mcp": 0.9, "if_pip": 1.2, "if_dip": 0.6,
              "mf_mcp": 0.9, "mf_pip": 1.2, "mf_dip": 0.6,
              "rf_mcp": 0.9, "rf_pip": 1.2, "rf_dip": 0.6,
              "th_cmc": 1.6, "th_axl": 0.9, "th_mcp": 1.0, "th_ipl": 0.6},
        note="LEAP 16-DoF"),
}


def build(hand, half, mass, friction, lift_range=(-0.06, 0.45)):
    cfg = HANDS[hand]
    xml = cfg["xml"].read_text()
    xml = xml.replace('meshdir="./assets/"',
                      f'meshdir="{(cfg["xml"].parent / "assets").as_posix()}/"')
    import re
    mm = re.search(r'<body name="%s"[^>]*>' % cfg["palm"], xml)
    assert mm, "palm body not found"
    anchor = mm.group(0)
    xml = xml.replace(anchor, anchor + '\n      <joint name="lift" type="slide"'
                      f' axis="0 0 -1" range="{lift_range[0]} {lift_range[1]}"'
                      ' damping="3"/>')
    xml = xml.replace("</actuator>",
                      '  <position name="lift_act" joint="lift" kp="900" kv="70"'
                      f' ctrlrange="{lift_range[0]} {lift_range[1]}"/>\n  </actuator>')
    sz = " ".join(str(v) for v in half)
    xml = xml.replace("</mujoco>", f"""
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <body name="object" pos="0 0 0.3">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{sz}" rgba="0.85 0.3 0.2 1"
            mass="{mass}" friction="{friction}"/>
    </body>
  </worldbody>
</mujoco>""")
    return mujoco.MjModel.from_xml_string(xml)


class Bench:
    def __init__(self, hand, half, mass, friction="2.0 0.05 0.002"):
        self.hand, self.cfg = hand, HANDS[hand]
        self.m = build(hand, half, mass, friction)
        self.d = mujoco.MjData(self.m)
        self.half, self.mass = np.array(half, float), mass
        n2 = lambda t, s: mujoco.mj_name2id(self.m, t, s)
        self.ogid = n2(mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = n2(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.oq = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.ov = self.m.jnt_dofadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.lift_a = n2(mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
        self.lift_q = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "lift")]
        self.palm = n2(mujoco.mjtObj.mjOBJ_BODY, self.cfg["palm"])
        self.tips = [n2(mujoco.mjtObj.mjOBJ_BODY, b) for b in self.cfg["tips"]]
        self.thumb = n2(mujoco.mjtObj.mjOBJ_BODY, self.cfg["thumb_tip"])
        self.jnames = list(self.cfg["flex"])
        self.jq = np.array([self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, j)]
                            for j in self.jnames])
        self.ja = np.array([n2(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_act")
                            for j in self.jnames])
        self.amp = np.array([self.cfg["flex"][j] for j in self.jnames])

    # ---- planning (kinematic) ----
    def gap_at(self, frac):
        q = np.zeros(self.m.nq)
        q[self.oq + 2] = 5.0; q[self.oq + 3] = 1.0
        q[self.jq] = frac * self.amp
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        fm = self.d.xpos[self.tips].mean(0)
        th = self.d.xpos[self.thumb].copy()
        return float(np.linalg.norm(th - fm)), 0.5 * (fm + th)

    def gap_curve(self, n=200):
        fr = np.linspace(0, 1, n)
        g = np.array([self.gap_at(f)[0] for f in fr])
        return fr, g

    def plan(self, width, margin=0.005):
        """Pre-grasp / grasp / squeeze fractions, or None if infeasible."""
        fr, g = self.gap_curve()
        lo, hi = float(g.min()), float(g.max())
        if not (lo <= width <= hi):
            return None, dict(gap_min=lo, gap_max=hi, feasible=False)
        def at(t):
            return float(fr[int(np.argmin(np.abs(g - np.clip(t, lo, hi))))])
        f_pre, f_grasp = at(width + 2 * margin), at(width)
        f_sq = at(width - 2 * margin)
        _, centre = self.gap_at(f_pre)
        return dict(f_pre=f_pre, f_grasp=f_grasp, f_squeeze=f_sq,
                    centre=centre), dict(gap_min=lo, gap_max=hi, feasible=True)

    # ---- execution ----
    def ctrl(self, frac, lift):
        c = np.zeros(self.m.nu)
        c[self.ja] = frac * self.amp
        c[self.lift_a] = lift
        return c

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.ogid, self.obid)

    def force_only(self):
        """Summed contact normal force. Cheap -- no convex hull. Tracking the
        peak transient with full grasp_metrics ran a 6-D hull on every physics
        step and made the sweep unusably slow."""
        _, _, _, f = object_contacts(self.m, self.d, self.ogid)
        return float(f.sum()) if len(f) else 0.0

    def place(self, f_pre, centre):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.jq] = f_pre * self.amp
        self.d.qpos[self.oq:self.oq + 3] = centre
        self.d.qpos[self.oq + 3] = 1.0
        self.d.ctrl[:] = self.ctrl(f_pre, 0.0)
        mujoco.mj_forward(self.m, self.d)
        return self.d.ncon

    def open_until_clear(self, pl, steps=40):
        """Back the pre-grasp open until the block starts penetration-free.

        Placing the block at the planned gap midpoint can still overlap the
        fingers along the block's OTHER two dimensions, and MuJoCo resolves that
        initial penetration at 10^5 N. The lift and shake numbers survive it
        (the sustained force is ~14 N) but the transient is meaningless, so the
        start is moved open until reset has no contact at all."""
        f = pl["f_pre"]
        for k in range(steps):
            if self.place(f, pl["centre"]) == 0:
                return f, True
            f *= 0.94
        return f, False

    def run(self, pl, lift_h=0.15, n_way=25, settle_s=1.0, hold_s=1.5):
        f_pre, clear = self.open_until_clear(pl)
        pl = dict(pl); pl["f_pre"] = f_pre; pl["clear_start"] = clear
        self.place(f_pre, pl["centre"])
        pq = self.d.qpos[self.oq:self.oq + 7].copy()
        fmax = [0.0]

        def stp(frac, n, pin, lift=0.0):
            c = self.ctrl(frac, lift)
            for _ in range(n):
                self.d.ctrl[:] = c
                mujoco.mj_step(self.m, self.d)
                if pin:
                    self.d.qpos[self.oq:self.oq + 7] = pq
                    self.d.qvel[self.ov:self.ov + 6] = 0.0
                fmax[0] = max(fmax[0], self.force_only())

        stp(pl["f_pre"], 60, True)                       # settle at pre-grasp
        for i in range(1, 31):                           # -> grasp
            stp(pl["f_pre"] + (pl["f_grasp"] - pl["f_pre"]) * i / 30, 8, True)
        for i in range(1, 21):                           # -> squeeze
            stp(pl["f_grasp"] + (pl["f_squeeze"] - pl["f_grasp"]) * i / 20, 8, True)
        g = self.metrics()
        f_sq = pl["f_squeeze"]
        stp(f_sq, int(settle_s / self.m.opt.timestep), False)      # RELEASE
        z_rel = float(self.d.qpos[self.oq + 2])
        held = self.metrics()["n_contacts"] > 0 and abs(z_rel - pl["centre"][2]) < 0.02
        z0 = float(self.d.qpos[self.oq + 2])
        for i in range(1, n_way + 1):
            stp(f_sq, 30, False, lift=lift_h * i / n_way)
        stp(f_sq, int(hold_s / self.m.opt.timestep), False, lift=lift_h)
        zf = float(self.d.qpos[self.oq + 2]); top = self.metrics()
        z_pre_shake = zf
        for _ in range(4):
            stp(f_sq, 12, False, lift=lift_h - 0.035)
            stp(f_sq, 12, False, lift=lift_h)
        shook = self.metrics()
        shake_drop = z_pre_shake - float(self.d.qpos[self.oq + 2])
        return dict(f_pre=pl["f_pre"], clear_start=bool(pl["clear_start"]),
                    f_grasp=pl["f_grasp"], f_squeeze=f_sq,
                    eps=g["epsilon"], n=g["n_contacts"], f_grip=g["f_total"],
                    f_max_transient=float(fmax[0]), held_static=bool(held),
                    release_drop_m=float(pl["centre"][2] - z_rel),
                    net_lift_m=float(zf - z0), eps_top=top["epsilon"],
                    n_top=top["n_contacts"], f_top=top["f_total"],
                    shake_drop_m=float(shake_drop), n_shake=shook["n_contacts"],
                    eps_shake=shook["epsilon"],
                    success=bool(zf - z0 >= 0.10 and top["n_contacts"] > 0
                                 and top["epsilon"] > 0.01
                                 and abs(shake_drop) < 0.02
                                 and shook["n_contacts"] > 0
                                 and top["f_total"] < 100.0
                                 and pl["clear_start"]
                                 and fmax[0] < 500.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="leap")
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="results/grasp_bench_clean.json")
    a = ap.parse_args(); t0 = time.time()

    probe = Bench(a.hand, (0.02, 0.025, 0.025), a.mass)
    fr, g = probe.gap_curve()
    print(f"{a.hand}: closure gap spans {g.min()*100:.2f} - {g.max()*100:.2f} cm"
          f"  (block widths outside that range are infeasible and are skipped)")

    rows, best = [], None
    print(f"\n{'w(cm)':>7}{'marg':>6}{'f_pre':>7}{'f_gr':>7}{'f_sq':>7}{'ncon':>6}"
          f"{'Fgrip':>8}{'Fmax':>9}{'eps':>7}{'lift':>8}{'epsTop':>8}{'Ftop':>8}"
          f"{'shake':>8}{'ok':>5}")
    for w in (0.050, 0.055, 0.060, 0.065, 0.070, 0.080):
        for margin in (0.003, 0.006, 0.010):
            b = Bench(a.hand, (w / 2, 0.025, 0.025), a.mass)
            pl, info = b.plan(w, margin)
            if pl is None:
                continue
            r = b.run(pl)
            r.update(w=w, margin=margin, **info)
            rows.append(r)
            print(f"{w*100:>7.1f}{margin*1000:>6.0f}{r['f_pre']:>7.3f}"
                  f"{r['f_grasp']:>7.3f}{r['f_squeeze']:>7.3f}{r['n']:>6}"
                  f"{r['f_grip']:>8.2f}{r['f_max_transient']:>9.1f}{r['eps']:>7.3f}"
                  f"{r['net_lift_m']*100:>8.1f}{r['eps_top']:>8.3f}{r['f_top']:>8.2f}"
                  f"{r['shake_drop_m']*100:>8.2f}{str(r['success']):>5}", flush=True)
            if best is None or (r["success"], -r["f_max_transient"]) > \
                    (best["success"], -best["f_max_transient"]):
                best = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, best=best), indent=2, default=float))
    ok = [r for r in rows if r["success"]]
    print(f"\nsuccess {len(ok)}/{len(rows)}   ({time.time()-t0:.0f} s)")
    if ok:
        clean = min(ok, key=lambda r: r["f_max_transient"])
        print("CLEANEST SUCCESS (lowest transient force):")
        print(json.dumps({k: v for k, v in clean.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()
