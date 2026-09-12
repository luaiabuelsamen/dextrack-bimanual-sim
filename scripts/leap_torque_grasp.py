"""LEAP grasps by applying a constant closing TORQUE, not a position target.

Everything before this drove position servos to a target inside the object, so
the servo error kept growing and the contact force with it: 26 N, 114 N, 6836 N
on a 50 g block. That is the wrong actuator for grasping and it is why the
grasps kept exploding or ejecting the object.

Constant-torque closure is what adaptive grippers do and what grasp simulations
normally use: every flexion joint is given a fixed effort, the fingers wrap until
the object stops them, and the contact force self-limits at whatever balances the
applied torque. The Menagerie model's elliptic cone and impratio=100 are already
set up for this.

Mechanics: LEAP joints here are motor actuators with gear=1, so ctrl is joint
torque in N*m. A fingertip ~4 cm from the joint applying 0.05 N*m gives ~1.2 N of
normal force; holding a 0.05 kg block (0.49 N) across two opposed contacts at
mu=1 needs ~0.25 N each, so the range to sweep is small.
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


def build(half, mass, friction, lift_range=(-0.45, 0.06)):
    xml = (MEN / "right_hand.xml").read_text()
    xml = xml.replace('meshdir="./assets/"', f'meshdir="{(MEN/"assets").as_posix()}/"')
    anchor = '<body name="palm" pos="0 0 0.1" quat="0 1 0 0">'
    xml = xml.replace(anchor, anchor + '\n      <joint name="lift" type="slide"'
                      f' axis="0 0 -1" range="{lift_range[0]} {lift_range[1]}" damping="4"/>')
    # torque actuators on every hand joint, plus a position servo for the lift
    acts = []
    for f in FINGERS:
        for j in FLEX + ["rot"]:
            acts.append(f'    <motor name="{f}_{j}_m" joint="{f}_{j}" gear="1" ctrlrange="-1 1"/>')
    for n in THUMB:
        acts.append(f'    <motor name="{n}_m" joint="{n}" gear="1" ctrlrange="-1 1"/>')
    acts.append('    <position name="lift_act" joint="lift" kp="600" kv="60"'
                f' ctrlrange="{lift_range[0]} {lift_range[1]}"/>')
    i, j = xml.index("<actuator>"), xml.index("</actuator>") + len("</actuator>")
    xml = xml[:i] + "<actuator>\n" + "\n".join(acts) + "\n  </actuator>" + xml[j:]
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


class TorqueGrasp:
    def __init__(self, half=(0.02, 0.025, 0.025), mass=0.05,
                 friction="1.0 0.02 0.001"):
        self.m = build(half, mass, friction); self.d = mujoco.MjData(self.m)
        self.half = np.array(half, float); self.mass = mass
        n2i = lambda t, s: mujoco.mj_name2id(self.m, t, s)
        self.ogid = n2i(mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = n2i(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.oq = self.m.jnt_qposadr[n2i(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.ov = self.m.jnt_dofadr[n2i(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.lift_a = n2i(mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
        self.lift_q = self.m.jnt_qposadr[n2i(mujoco.mjtObj.mjOBJ_JOINT, "lift")]
        self.flex_a = [n2i(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}_{j}_m")
                       for f in FINGERS for j in FLEX]
        self.thumb_a = [n2i(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_m") for n in THUMB]
        self.tips = [n2i(mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("if_ds", "mf_ds", "rf_ds", "th_ds")]

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.ogid, self.obid)

    def grasp_centre(self, tau_f, tau_t, n=800):
        """Close in free space with the same torques; the gap the fingers settle
        into is where the object goes."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.oq:self.oq + 3] = (0, 0, 5.0)      # park it away
        c = np.zeros(self.m.nu)
        c[self.flex_a] = tau_f; c[self.thumb_a] = tau_t
        for _ in range(n):
            self.d.ctrl[:] = c; mujoco.mj_step(self.m, self.d)
        t = self.d.xpos[self.tips].copy()
        fm, th = t[:3].mean(0), t[3]
        return 0.5 * (fm + th), float(np.linalg.norm(th - fm)), (th - fm)

    def run(self, tau_f, tau_t, centre, lift_h=0.15, n_way=25, hold_s=2.0,
            settle_s=1.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.oq:self.oq + 3] = centre
        self.d.qpos[self.lift_q] = 0.0
        mujoco.mj_forward(self.m, self.d)
        pq = self.d.qpos[self.oq:self.oq + 7].copy()
        c = np.zeros(self.m.nu)

        def stp(n, pin, lift=0.0):
            c[self.lift_a] = lift
            for _ in range(n):
                self.d.ctrl[:] = c; mujoco.mj_step(self.m, self.d)
                if pin:
                    self.d.qpos[self.oq:self.oq + 7] = pq
                    self.d.qvel[self.ov:self.ov + 6] = 0.0

        stp(100, True)                                   # settle open
        for i in range(1, 51):                           # ramp torque on
            c[self.flex_a] = tau_f * i / 50
            c[self.thumb_a] = tau_t * i / 50
            stp(10, True)
        g = self.metrics()
        stp(int(settle_s / self.m.opt.timestep), False)  # RELEASE
        z_rel = float(self.d.qpos[self.oq + 2])
        held = self.metrics()["n_contacts"] > 0 and abs(z_rel - centre[2]) < 0.015
        z0 = float(self.d.qpos[self.oq + 2])
        for i in range(1, n_way + 1):
            stp(30, False, lift=-lift_h * i / n_way)
        stp(int(hold_s / self.m.opt.timestep), False, lift=-lift_h)
        zf = float(self.d.qpos[self.oq + 2]); after = self.metrics()
        return dict(eps=g["epsilon"], n=g["n_contacts"], f_total=g["f_total"],
                    delta=g["delta"], held_static=bool(held),
                    release_drop_m=float(centre[2] - z_rel),
                    lift_actual=float(-self.d.qpos[self.lift_q]),
                    net_lift_m=float(zf - z0), n_after=after["n_contacts"],
                    eps_after=after["epsilon"],
                    success=bool(zf - z0 >= 0.10 and after["n_contacts"] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="results/leap_torque_grasp.json")
    a = ap.parse_args(); t0 = time.time(); rows = []; best = None
    print(f"{'w(cm)':>7}{'fric':>6}{'tau_f':>7}{'tau_t':>7}{'freegap':>9}{'ncon':>6}"
          f"{'F(N)':>8}{'eps':>8}{'static':>8}{'lift(cm)':>10}{'n_end':>7}{'ok':>6}")
    for w in (0.03, 0.04, 0.05):
        for fric in ("1.0 0.02 0.001", "2.0 0.05 0.002"):
            g = TorqueGrasp(half=(w / 2, 0.025, 0.025), mass=a.mass, friction=fric)
            for tau_f in (0.03, 0.08, 0.15):
                for tau_t in (0.05, 0.12):
                    c, gap, ax = g.grasp_centre(tau_f, tau_t)
                    # place the block so its closed gap straddles the fingers
                    r = g.run(tau_f, tau_t, c)
                    r.update(w=w, fric=fric, tau_f=tau_f, tau_t=tau_t,
                             free_gap=gap)
                    rows.append(r)
                    print(f"{w*100:>7.1f}{fric.split()[0]:>6}{tau_f:>7.2f}{tau_t:>7.2f}"
                          f"{gap*100:>9.2f}{r['n']:>6}{r['f_total']:>8.2f}"
                          f"{r['eps']:>8.4f}{str(r['held_static']):>8}"
                          f"{r['net_lift_m']*100:>10.2f}{r['n_after']:>7}"
                          f"{str(r['success']):>6}", flush=True)
                    if best is None or (r["success"], r["net_lift_m"]) > \
                            (best["success"], best["net_lift_m"]):
                        best = r
    Path(a.out).write_text(json.dumps(dict(rows=rows, best=best), indent=2, default=float))
    ok = [r for r in rows if r["success"]]
    print(f"\nsuccess {len(ok)}/{len(rows)}  ({time.time()-t0:.0f} s)")
    print("BEST:", json.dumps(best, indent=2, default=float))


if __name__ == "__main__":
    main()
