"""LEAP grasps and lifts the 0.05 kg block. Mid-air bench, correct lift sign.

Two bugs fixed here, both found by instrumenting rather than sweeping:

1. LIFT SIGN. The slide's axis is given in the PALM's body frame, and the palm
   carries quat="0 1 0 0" (a 180 deg flip), so the world-frame sense is inverted:
   POSITIVE ctrl raises the hand. Every earlier run commanded negative and
   either clamped at the range edge (8.8 cm) or drove the hand DOWN 16 cm, which
   is why the "block fell" -- the block was fine, the hand was descending.
2. NO PEDESTAL. A support column under the block passes straight through the
   palm (45 contacts at reset: pedestal vs palm_collision_2, if_bs_collision_*),
   because this hand's grasp region is ABOVE its palm. The block is held in
   mid-air during closure instead, which is also the harder test: nothing ever
   supports it.

Grip force is kept low deliberately. A seated grip of 50-114 N on a 0.05 kg
block is ~1500 m/s^2 and ejects it the moment the pin is released; the useful
range with Menagerie's kp=3 actuators is a couple of newtons.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.epsilon import grasp_metrics

MEN = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie/leap_hand"
FINGERS = ["if", "mf", "rf"]
FLEX = ["mcp", "pip", "dip"]
THUMB = ["th_cmc", "th_axl", "th_mcp", "th_ipl"]


def build(half, mass, friction, lift_range=(-0.06, 0.45)):
    xml = (MEN / "right_hand.xml").read_text()
    xml = xml.replace('meshdir="./assets/"', f'meshdir="{(MEN/"assets").as_posix()}/"')
    anchor = '<body name="palm" pos="0 0 0.1" quat="0 1 0 0">'
    assert anchor in xml
    xml = xml.replace(anchor, anchor + '\n      <joint name="lift" type="slide"'
                      f' axis="0 0 -1" range="{lift_range[0]} {lift_range[1]}" damping="3"/>')
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


class Leap:
    def __init__(self, half, mass, friction="2.0 0.05 0.002"):
        self.m = build(half, mass, friction); self.d = mujoco.MjData(self.m)
        self.half = np.array(half, float); self.mass = mass
        n2 = lambda t, s: mujoco.mj_name2id(self.m, t, s)
        self.ogid = n2(mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = n2(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.oq = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.ov = self.m.jnt_dofadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.lift_a = n2(mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
        self.lift_q = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "lift")]
        self.act = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a
                    for a in range(self.m.nu)}
        self.tips = [n2(mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("if_ds", "mf_ds", "rf_ds", "th_ds")]
        self.palm = n2(mujoco.mjtObj.mjOBJ_BODY, "palm")

    def ctrl(self, flex, thumb, lift):
        c = np.zeros(self.m.nu)
        for f in FINGERS:
            for j, v in zip(FLEX, flex):
                c[self.act[f"{f}_{j}_act"]] = v
        for n, v in zip(THUMB, thumb):
            c[self.act[f"{n}_act"]] = v
        c[self.lift_a] = lift          # POSITIVE raises
        return c

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.ogid, self.obid)

    def grasp_gap(self, flex, thumb, frac):
        """Kinematic thumb-tip / finger-mean gap and midpoint at closure `frac`."""
        q = np.zeros(self.m.nq); q[self.oq + 2] = 5.0; q[self.oq + 3] = 1.0
        for f in FINGERS:
            for j, v in zip(FLEX, flex):
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{j}")
                q[self.m.jnt_qposadr[jid]] = frac * v
        for n, v in zip(THUMB, thumb):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            q[self.m.jnt_qposadr[jid]] = frac * v
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d)
        t = self.d.xpos[self.tips].copy()
        fm, th = t[:3].mean(0), t[3]
        return float(np.linalg.norm(th - fm)), 0.5 * (fm + th)

    def plan(self, flex, thumb, width, n=150):
        fr = np.linspace(0, 1, n)
        gm = [self.grasp_gap(flex, thumb, f) for f in fr]
        gaps = np.array([g for g, _ in gm])
        i = int(np.argmin(np.abs(gaps - width)))
        return float(fr[i]), gaps[i], gm[i][1]

    def run(self, flex, thumb, width, extra=0.05, lift_h=0.15, n_way=25,
            settle_s=1.0, hold_s=2.0):
        f_touch, gap, centre = self.plan(flex, thumb, width)
        f_grip = min(f_touch + extra, 1.0)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.oq:self.oq + 3] = centre
        self.d.qpos[self.oq + 3] = 1.0
        mujoco.mj_forward(self.m, self.d)
        pq = self.d.qpos[self.oq:self.oq + 7].copy()
        cur = [0.0, 0.0]

        def stp(fr, n, pin, lift=0.0):
            c = self.ctrl([fr * v for v in flex], [fr * v for v in thumb], lift)
            for _ in range(n):
                self.d.ctrl[:] = c; mujoco.mj_step(self.m, self.d)
                if pin:
                    self.d.qpos[self.oq:self.oq + 7] = pq
                    self.d.qvel[self.ov:self.ov + 6] = 0.0

        stp(0.0, 150, True)
        for i in range(1, 61):                      # close to touch
            stp(f_touch * i / 60, 8, True)
        for i in range(1, 21):                      # squeeze
            stp(f_touch + (f_grip - f_touch) * i / 20, 10, True)
        g = self.metrics()
        stp(f_grip, int(settle_s / self.m.opt.timestep), False)   # RELEASE
        z_rel = float(self.d.qpos[self.oq + 2])
        held = (self.metrics()["n_contacts"] > 0 and abs(z_rel - centre[2]) < 0.02)
        z0 = float(self.d.qpos[self.oq + 2])
        palm0 = float(self.d.xpos[self.palm][2])
        for i in range(1, n_way + 1):
            stp(f_grip, 30, False, lift=lift_h * i / n_way)
        stp(f_grip, int(hold_s / self.m.opt.timestep), False, lift=lift_h)
        zf = float(self.d.qpos[self.oq + 2]); after = self.metrics()
        palm_rise = float(self.d.xpos[self.palm][2]) - palm0
        # SHAKE. Lifting alone is degenerate: a block resting on top of the hand
        # rides up with it and scores a perfect lift with epsilon = 0. Shaking
        # separates carrying from grasping -- only a force-closed grasp survives
        # the hand accelerating downward faster than the block would fall.
        z_pre_shake = zf
        for cyc in range(4):
            stp(f_grip, 12, False, lift=lift_h - 0.035)
            stp(f_grip, 12, False, lift=lift_h)
        shook = self.metrics()
        z_shake = float(self.d.qpos[self.oq + 2])
        return dict(gap=float(gap), f_touch=f_touch, f_grip=f_grip,
                    eps=g["epsilon"], n=g["n_contacts"], f_total=g["f_total"],
                    delta=g["delta"], held_static=bool(held),
                    release_drop_m=float(centre[2] - z_rel),
                    palm_rise_m=palm_rise, net_lift_m=float(zf - z0),
                    n_after=after["n_contacts"], eps_after=after["epsilon"],
                    shake_drop_m=float(z_pre_shake - z_shake),
                    n_shake=shook["n_contacts"], eps_shake=shook["epsilon"],
                    f_end=float(after["f_total"]),
                    success=bool(zf - z0 >= 0.10 and after["n_contacts"] > 0
                                 and after["epsilon"] > 0.01
                                 and abs(z_pre_shake - z_shake) < 0.02
                                 and shook["n_contacts"] > 0
                                 and after["f_total"] < 100.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="results/leap_final.json")
    a = ap.parse_args(); t0 = time.time(); rows = []; best = None
    print("success now requires: lift>=10cm, contacts at top, EPSILON>0.01 at top,"
          "\n  shake drop<2cm, contacts after shake, and grip force <100 N\n")
    print(f"{'w(cm)':>7}{'extra':>7}{'gap':>7}{'f_t':>6}{'ncon':>6}"
          f"{'F(N)':>9}{'eps':>7}{'lift':>8}{'epsEnd':>8}{'shake':>8}{'nShk':>6}{'ok':>6}")
    for w in (0.030, 0.035, 0.040, 0.045, 0.050, 0.055):
        L = Leap((w / 2, 0.025, 0.025), a.mass)
        for flex, thumb in [([0.9, 1.2, 0.6], [1.6, 0.9, 1.0, 0.6]),
                            ([1.2, 1.4, 0.8], [2.0, 1.1, 1.2, 0.8])]:
            for extra in (0.03, 0.06, 0.10, 0.15):
                r = L.run(flex, thumb, w, extra=extra)
                r.update(w=w, extra=extra, flex=flex)
                rows.append(r)
                print(f"{w*100:>7.1f}{extra:>7.2f}{r['gap']*100:>7.2f}"
                      f"{r['f_touch']:>6.2f}{r['n']:>6}{r['f_total']:>9.1f}"
                      f"{r['eps']:>7.3f}{r['net_lift_m']*100:>8.1f}"
                      f"{r['eps_after']:>8.3f}{r['shake_drop_m']*100:>8.2f}"
                      f"{r['n_shake']:>6}{str(r['success']):>6}", flush=True)
                if best is None or (r["success"], r["net_lift_m"]) > \
                        (best["success"], best["net_lift_m"]):
                    best = r
    Path(a.out).write_text(json.dumps(dict(rows=rows, best=best), indent=2, default=float))
    ok = [r for r in rows if r["success"]]
    print(f"\nsuccess {len(ok)}/{len(rows)}  ({time.time()-t0:.0f} s)")
    print("BEST:", json.dumps(best, indent=2, default=float))


if __name__ == "__main__":
    main()
