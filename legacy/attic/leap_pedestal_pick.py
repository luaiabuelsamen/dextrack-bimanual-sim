"""LEAP picks a block off a pedestal. No pin, low squeeze.

Two fixes over every earlier attempt, both from what the failures measured:

1. NO PIN. Holding the block in place during closure and then releasing it is
   what kept ejecting it: the seated grip measured 50-114 N on a 0.05 kg block,
   which is ~1500 m/s^2, so any imbalance at release throws it (release_drop was
   16 cm before the lift even started). Here the block rests on a physical
   pedestal at the height the fingers actually close, so it is supported by
   contact throughout and the grasp forms against a real support.

2. LOW SQUEEZE. The closure stops a small distance past first contact instead of
   driving the position targets deep. With Menagerie's kp=3 actuators, a 0.2 rad
   overshoot at a 4 cm lever is already ~37 N; the useful range is far smaller.

The pedestal is thin and sits below the block, so it obstructs only the underside
- the fingers close on the sides, which is what a pick off a table does anyway.
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


def build(half, mass, block_xyz, ped_top, friction, lift_range=(-0.45, 0.06)):
    xml = (MEN / "right_hand.xml").read_text()
    xml = xml.replace('meshdir="./assets/"', f'meshdir="{(MEN/"assets").as_posix()}/"')
    anchor = '<body name="palm" pos="0 0 0.1" quat="0 1 0 0">'
    xml = xml.replace(anchor, anchor + '\n      <joint name="lift" type="slide"'
                      f' axis="0 0 -1" range="{lift_range[0]} {lift_range[1]}" damping="3"/>')
    xml = xml.replace("</actuator>",
                      '  <position name="lift_act" joint="lift" kp="900" kv="70"'
                      f' ctrlrange="{lift_range[0]} {lift_range[1]}"/>\n  </actuator>')
    x, y, z = block_xyz
    sz = " ".join(str(v) for v in half)
    xml = xml.replace("</mujoco>", f"""
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <geom name="pedestal" type="box" size="0.012 0.012 {ped_top/2}"
          pos="{x} {y} {ped_top/2}" rgba="0.4 0.42 0.45 1" friction="1 0.02 0.001"/>
    <body name="object" pos="{x} {y} {z}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{sz}" rgba="0.85 0.3 0.2 1"
            mass="{mass}" friction="{friction}"/>
    </body>
  </worldbody>
</mujoco>""")
    return mujoco.MjModel.from_xml_string(xml)


class Ped:
    def __init__(self, half, mass, block_xyz, friction):
        ped_top = max(block_xyz[2] - half[2], 0.005)
        self.m = build(half, mass, block_xyz, ped_top, friction)
        self.d = mujoco.MjData(self.m)
        n2 = lambda t, s: mujoco.mj_name2id(self.m, t, s)
        self.ogid = n2(mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = n2(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.pgid = n2(mujoco.mjtObj.mjOBJ_GEOM, "pedestal")
        self.oq = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.lift_a = n2(mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
        self.lift_q = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "lift")]
        self.act = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a
                    for a in range(self.m.nu)}
        self.tips = [n2(mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("if_ds", "mf_ds", "rf_ds", "th_ds")]

    def ctrl(self, flex, thumb, lift):
        c = np.zeros(self.m.nu)
        for f in FINGERS:
            for j, v in zip(FLEX, flex):
                c[self.act[f"{f}_{j}_act"]] = v
        for n, v in zip(THUMB, thumb):
            c[self.act[f"{n}_act"]] = v
        c[self.lift_a] = lift
        return c

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.ogid, self.obid, (self.pgid,))

    def step(self, c, n):
        for _ in range(n):
            self.d.ctrl[:] = c
            mujoco.mj_step(self.m, self.d)

    def run(self, flex, thumb, f_target=1.5, extra=0.06, lift_h=0.15,
            n_way=25, hold_s=2.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.ctrl[:] = self.ctrl([0, 0, 0], [0, 0, 0, 0], 0.0)
        mujoco.mj_forward(self.m, self.d)
        self.step(self.ctrl([0, 0, 0], [0, 0, 0, 0], 0.0), 300)
        z_rest = float(self.d.qpos[self.oq + 2])
        # close until first contact force, then a small fixed overshoot
        hit = 1.0
        for i in range(1, 81):
            f = i / 80
            self.step(self.ctrl([f*v for v in flex], [f*v for v in thumb], 0.0), 12)
            if self.metrics()["f_total"] >= f_target:
                hit = f
                break
        fin = min(hit + extra, 1.0)
        self.step(self.ctrl([fin*v for v in flex], [fin*v for v in thumb], 0.0), 250)
        g = self.metrics()
        z0 = float(self.d.qpos[self.oq + 2])
        cf = [fin*v for v in flex]; ct = [fin*v for v in thumb]
        for i in range(1, n_way + 1):
            self.step(self.ctrl(cf, ct, -lift_h*i/n_way), 30)
        self.step(self.ctrl(cf, ct, -lift_h), int(hold_s/self.m.opt.timestep))
        zf = float(self.d.qpos[self.oq + 2]); after = self.metrics()
        return dict(hit_frac=float(hit), fin_frac=float(fin), eps=g["epsilon"],
                    n=g["n_contacts"], f_total=g["f_total"], delta=g["delta"],
                    settle_z=float(z_rest), lift_actual=float(-self.d.qpos[self.lift_q]),
                    net_lift_m=float(zf - z0), n_after=after["n_contacts"],
                    eps_after=after["epsilon"],
                    success=bool(zf - z0 >= 0.10 and after["n_contacts"] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="legacy/results/leap_pedestal.json")
    a = ap.parse_args(); t0 = time.time(); rows = []; best = None
    # grasp region measured earlier: fingers curl to x~0.008, thumb to x~-0.046,
    # z ~0.17. Put the block in that gap, on a pedestal.
    print(f"{'bx':>7}{'bz':>7}{'w(cm)':>7}{'ftgt':>6}{'extra':>7}{'hit':>6}"
          f"{'ncon':>6}{'F(N)':>8}{'eps':>8}{'lift(cm)':>10}{'n_end':>7}{'ok':>6}")
    for bx in (-0.02, 0.0, 0.02):
        for bz in (0.15, 0.17):
            for w in (0.035, 0.045):
                p = Ped((w/2, 0.025, 0.025), a.mass, (bx, 0.02, bz),
                        "2.0 0.05 0.002")
                for ftgt in (1.0, 2.5):
                    for extra in (0.03, 0.08):
                        r = p.run([0.9, 1.2, 0.6], [1.6, 0.9, 1.0, 0.6],
                                  f_target=ftgt, extra=extra)
                        r.update(bx=bx, bz=bz, w=w, ftgt=ftgt, extra=extra)
                        rows.append(r)
                        if r["n"] > 0:
                            print(f"{bx:>7.2f}{bz:>7.2f}{w*100:>7.1f}{ftgt:>6.1f}"
                                  f"{extra:>7.2f}{r['hit_frac']:>6.2f}{r['n']:>6}"
                                  f"{r['f_total']:>8.2f}{r['eps']:>8.4f}"
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
