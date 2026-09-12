"""LEAP picks up a block, using DeepMind's Menagerie model instead of a
hand-compiled URDF.

Why this replaces everything before it. The previous bench compiled the raw
DexTrack URDF, invented actuator gains, and guessed the closure direction by
driving joints toward range limits. That produced a hand balled into a fist,
sunk into the table, with the block outside it. All of that is solved upstream:

  - `mujoco_menagerie/leap_hand/right_hand.xml` is tuned by DeepMind: named
    joints with real semantics (if_mcp / if_pip / if_dip per finger, th_cmc /
    th_axl / th_mcp / th_ipl for the thumb), position actuators at kp=3, an
    elliptic friction cone with impratio=100, simplified fingertip collision
    meshes, and per-class joint ranges.
  - `scene_right.xml` already mounts the palm face-DOWN at z=0.1 over a ground
    plane, which is the top-down pick configuration.

So the only things added here are a vertical slide on the palm (to lift with)
and the block. The closure is the obvious one, now that the joints have
meanings: flex mcp/pip/dip on the three fingers, and swing the thumb across to
oppose them.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.epsilon import grasp_metrics    # noqa: E402

MEN = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie/leap_hand"

FINGERS = ["if", "mf", "rf"]
FLEX = ["mcp", "pip", "dip"]          # positive = curl toward the palm
THUMB = ["th_cmc", "th_axl", "th_mcp", "th_ipl"]


def build(block_half, block_mass, block_pos, friction="1.0 0.02 0.001",
          lift_range=(-0.06, 0.40)):
    """Menagerie right hand + a vertical slide on the palm + a free block."""
    xml = (MEN / "right_hand.xml").read_text()
    xml = xml.replace('meshdir="./assets/"', f'meshdir="{(MEN / "assets").as_posix()}/"')
    # vertical slide on the palm so the hand can actually lift
    anchor = '<body name="palm" pos="0 0 0.1" quat="0 1 0 0">'
    assert anchor in xml
    xml = xml.replace(anchor, anchor + f'\n      <joint name="lift" type="slide"'
                      f' axis="0 0 -1" range="{lift_range[0]} {lift_range[1]}"'
                      f' damping="2"/>')
    # the palm quat is a 180-deg flip, so the palm's local -z is world +z
    xml = xml.replace("</actuator>",
                      '  <position name="lift_act" joint="lift" kp="800" kv="60"'
                      f' ctrlrange="{lift_range[0]} {lift_range[1]}"/>\n  </actuator>')
    bx, by, bz = block_pos
    sz = " ".join(str(v) for v in block_half)
    obj = f"""
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" rgba="0.5 0.55 0.6 1"/>
    <body name="object" pos="{bx} {by} {bz}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{sz}" rgba="0.85 0.3 0.2 1"
            mass="{block_mass}" friction="{friction}"/>
    </body>
  </worldbody>
</mujoco>"""
    xml = xml.replace("</mujoco>", obj)
    return mujoco.MjModel.from_xml_string(xml)


class Pick:
    def __init__(self, block_half=(0.022, 0.022, 0.022), block_mass=0.05,
                 block_xy=(-0.05, -0.03), friction="1.0 0.02 0.001"):
        self.half = np.array(block_half, float)
        self.m = build(block_half, block_mass,
                       (block_xy[0], block_xy[1], float(block_half[2])), friction)
        self.d = mujoco.MjData(self.m)
        self.mass = block_mass
        self.ogid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.fgid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.oq = self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.ov = self.m.jnt_dofadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.lift_a = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
        self.lift_q = self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "lift")]
        self.act = {}
        for a in range(self.m.nu):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            self.act[n] = a
        self.tips = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("if_ds", "mf_ds", "rf_ds", "th_ds")
                     if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
        self.palm = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "palm")

    def ctrl_from(self, flex, thumb, lift):
        c = np.zeros(self.m.nu)
        for f in FINGERS:
            for j, v in zip(FLEX, flex):
                a = self.act.get(f"{f}_{j}_act")
                if a is not None:
                    c[a] = v
        for n, v in zip(THUMB, thumb):
            a = self.act.get(f"{n}_act")
            if a is not None:
                c[a] = v
        c[self.lift_a] = lift
        return c

    def reset(self, lift=0.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.lift_q] = lift
        self.d.ctrl[:] = self.ctrl_from([0, 0, 0], [0, 0, 0, 0], lift)
        mujoco.mj_forward(self.m, self.d)

    def step(self, c, n):
        for _ in range(n):
            self.d.ctrl[:] = c
            mujoco.mj_step(self.m, self.d)

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.ogid, self.obid, (self.fgid,))

    def obj_z(self):
        return float(self.d.qpos[self.oq + 2])

    def run(self, approach, flex, thumb, lift_h=0.15, n_way=25, hold_s=2.0,
            close_steps=600):
        """Lower to `approach`, close, lift by `lift_h`, hold."""
        self.reset(lift=0.0)
        # descend over the block
        for i in range(1, 21):
            self.step(self.ctrl_from([0, 0, 0], [0, 0, 0, 0], approach * i / 20), 20)
        pre = self.metrics()
        # close
        for i in range(1, 41):
            f = i / 40
            self.step(self.ctrl_from([f * v for v in flex],
                                     [f * v for v in thumb], approach),
                      close_steps // 40)
        g = self.metrics()
        z0 = self.obj_z()
        # lift
        for i in range(1, n_way + 1):
            self.step(self.ctrl_from(flex, thumb, approach - lift_h * i / n_way), 30)
        self.step(self.ctrl_from(flex, thumb, approach - lift_h),
                  int(hold_s / self.m.opt.timestep))
        zf = self.obj_z()
        after = self.metrics()
        return dict(pre_contacts=pre["n_contacts"], eps=g["epsilon"],
                    n=g["n_contacts"], f_total=g["f_total"], delta=g["delta"],
                    lift_cmd=float(lift_h), lift_actual=float(-self.d.qpos[self.lift_q]),
                    net_lift_m=float(zf - z0), n_after=after["n_contacts"],
                    eps_after=after["epsilon"],
                    success=bool(zf - z0 >= 0.10 and after["n_contacts"] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="results/leap_menagerie_pick.json")
    a = ap.parse_args()
    t0 = time.time()

    p = Pick(block_mass=a.mass)
    p.reset()
    print(f"actuators: {sorted(p.act)}")
    print(f"palm at {np.round(p.d.xpos[p.palm],4)}   block at "
          f"{np.round(p.d.qpos[p.oq:p.oq+3],4)}   mass {a.mass} kg")

    rows, best = [], None
    print(f"\n{'appr(cm)':>9}{'mcp':>6}{'pip':>6}{'dip':>6}{'th':>18}{'pre':>5}"
          f"{'ncon':>6}{'F(N)':>8}{'eps':>8}{'lift(cm)':>10}{'n_end':>7}{'ok':>6}")
    for approach in (0.02, 0.035, 0.05):
        for mcp in (0.6, 0.9):
            for pip in (0.8, 1.2):
                for th in ([1.2, 0.6, 0.8, 0.4], [1.6, 0.9, 1.0, 0.6],
                           [2.0, 1.2, 1.2, 0.8]):
                    flex = [mcp, pip, 0.6]
                    r = p.run(approach, flex, th)
                    r.update(approach=approach, flex=flex, thumb=th)
                    rows.append(r)
                    print(f"{approach*100:>9.1f}{mcp:>6.1f}{pip:>6.1f}{0.6:>6.1f}"
                          f"{str(th):>18}{r['pre_contacts']:>5}{r['n']:>6}"
                          f"{r['f_total']:>8.2f}{r['eps']:>8.4f}"
                          f"{r['net_lift_m']*100:>10.2f}{r['n_after']:>7}"
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
