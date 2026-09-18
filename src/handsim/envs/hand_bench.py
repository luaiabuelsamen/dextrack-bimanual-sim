"""M2, second dimension: what each hand can HOLD, not just what it can reach.

`opposition_axis.py` measures reach -- the closest the thumb can come to the
fingers. That is a necessary condition for a pinch, not a sufficient one. This
adds the sufficient half: across a shared set of block widths, which ones can
each hand actually close on and hold against gravity?

Everything mechanical is derived per hand by the same procedure, never
hard-coded (see `hand_specs.derive_flex`): the closure pose is solved for over
all joints at once, and the closure fraction that matches a given block width is
found from the hand's own gap curve. A width outside a hand's gap range is
reported INFEASIBLE rather than attempted -- that distinction is what the
original bench missed when it drove LEAP's fingers through a 4.4 cm block and
recorded it as a failure to grasp.

Protocol per (hand, width), identical for all:
  1. solve the closure, find the fraction whose thumb-to-finger gap == width
  2. place the block at the grasp centre, open the pre-grasp until the start is
     penetration-free
  3. close to the matched fraction, then squeeze past it
  4. release: hold against gravity for 2 s with NOTHING underneath
  held = object moves < 1 cm AND is still in contact
"""
from __future__ import annotations

import argparse, json, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

from handsim.paths import (MENAGERIE, VEGA_URDF,  # noqa: E402
                          compile_urdf, menagerie_xml)

from handsim.grasping.epsilon import grasp_metrics, object_contacts   # noqa: E402
from handsim.hands.specs import SPECS, derive_flex                      # noqa: E402


def build(key, flex, half, mass, friction="1.0 0.02 0.001"):
    """Hand + a free block, with position actuators on the derived flex joints."""
    cfg = SPECS[key]
    if cfg["kind"] == "mjcf":
        xml = cfg["path"].read_text()
        assets = (cfg["path"].parent / "assets").as_posix()
        xml = xml.replace('meshdir="./assets/"', f'meshdir="{assets}/"')
        xml = xml.replace('meshdir="assets"', f'meshdir="{assets}"')
    else:
        xml = compile_urdf(cfg['path'])
    root = ET.fromstring(xml)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    existing = {a.get("joint") for a in act}
    for j in flex:
        if j not in existing:
            ET.SubElement(act, "position",
                          dict(name=f"act_{j}", joint=j, kp="3.0", kv="0.05"))
    wb = root.find("worldbody")
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
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class HandBench:
    def __init__(self, key, flex, tip_names, half, mass):
        self.key, self.flex = key, flex
        self.m = build(key, flex, half, mass)
        self.d = mujoco.MjData(self.m)
        n2 = lambda t, s: mujoco.mj_name2id(self.m, t, s)
        self.tips = [n2(mujoco.mjtObj.mjOBJ_BODY, t) for t in tip_names]
        self.ogid = n2(mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obid = n2(mujoco.mjtObj.mjOBJ_BODY, "object")
        ofj = n2(mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.oq = self.m.jnt_qposadr[ofj]
        self.ov = self.m.jnt_dofadr[ofj]
        # Map each flex joint to whatever actuator drives it. The Menagerie
        # hands already ship actuators under their own names (if_mcp_act, not
        # act_if_mcp), so looking only for "act_<joint>" found none and left the
        # hand completely unactuated -- which is why the gap curve came back as
        # 10.8-12.0 cm for a hand whose closed gap is 0.25 cm.
        jnt2act = {}
        for aid in range(self.m.nu):
            if self.m.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = int(self.m.actuator_trnid[aid, 0])
                jnt2act[mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid)] = aid
        jq, ja, amp = [], [], []
        for j, v in flex.items():
            aid = jnt2act.get(j)
            if aid is None:
                continue
            jq.append(self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, j)])
            ja.append(aid); amp.append(v)
        self.jq = np.array(jq, int); self.ja = np.array(ja, int)
        self.amp = np.array(amp, float)
        assert len(self.ja) >= 4, f"{key}: only {len(self.ja)} flex joints actuated"

    # ---- kinematic gap curve ----
    def gap_at(self, frac):
        self.d.qpos[:] = 0.0
        self.d.qpos[self.oq + 2] = 5.0
        self.d.qpos[self.oq + 3] = 1.0
        self.d.qpos[self.jq] = frac * self.amp
        mujoco.mj_kinematics(self.m, self.d)
        t = self.d.xpos[self.tips]
        fm, th = t[:-1].mean(0), t[-1]
        return float(np.linalg.norm(th - fm)), 0.5 * (fm + th)

    def plan(self, width, margin=0.005, n=160):
        fr = np.linspace(0, 1, n)
        g = np.array([self.gap_at(f)[0] for f in fr])
        lo, hi = float(g.min()), float(g.max())
        if not (lo <= width <= hi):
            return None, dict(gap_min=lo, gap_max=hi, feasible=False)
        at = lambda t: float(fr[int(np.argmin(np.abs(g - np.clip(t, lo, hi))))])
        f_pre, f_gr = at(width + 2 * margin), at(width)
        f_sq = at(max(width - 2 * margin, lo + 0.002))
        _, centre = self.gap_at(f_pre)
        return dict(f_pre=f_pre, f_grasp=f_gr, f_squeeze=f_sq, centre=centre), \
            dict(gap_min=lo, gap_max=hi, feasible=True)

    # ---- dynamics ----
    def place(self, frac, centre):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.jq] = frac * self.amp
        self.d.qpos[self.oq:self.oq + 3] = centre
        self.d.qpos[self.oq + 3] = 1.0
        self.d.ctrl[self.ja] = frac * self.amp
        mujoco.mj_forward(self.m, self.d)
        return self.d.ncon

    def open_until_clear(self, pl, steps=40):
        f = pl["f_pre"]
        for _ in range(steps):
            if self.place(f, pl["centre"]) == 0:
                return f, True
            f *= 0.93
        return f, False

    def step(self, frac, n, pin=False, pq=None):
        c = frac * self.amp
        for _ in range(n):
            self.d.ctrl[self.ja] = c
            mujoco.mj_step(self.m, self.d)
            if pin:
                self.d.qpos[self.oq:self.oq + 7] = pq
                self.d.qvel[self.ov:self.ov + 6] = 0.0

    def trial(self, width, mass, hold_s=2.0):
        pl, info = self.plan(width)
        if pl is None:
            return dict(width=width, **info)
        f_pre, clear = self.open_until_clear(pl)
        pq = self.d.qpos[self.oq:self.oq + 7].copy()
        self.step(f_pre, 60, pin=True, pq=pq)
        for i in range(1, 31):
            self.step(f_pre + (pl["f_grasp"] - f_pre) * i / 30, 8, pin=True, pq=pq)
        for i in range(1, 21):
            self.step(pl["f_grasp"] + (pl["f_squeeze"] - pl["f_grasp"]) * i / 20,
                      8, pin=True, pq=pq)
        g = grasp_metrics(self.m, self.d, self.ogid, self.obid)
        z0 = float(self.d.qpos[self.oq + 2])
        self.step(pl["f_squeeze"], int(hold_s / self.m.opt.timestep))
        drop = z0 - float(self.d.qpos[self.oq + 2])
        after = grasp_metrics(self.m, self.d, self.ogid, self.obid)
        return dict(width=width, mass=mass, clear_start=bool(clear),
                    eps=g["epsilon"], n=g["n_contacts"], f_grip=g["f_total"],
                    drop_m=float(drop), n_after=after["n_contacts"],
                    held=bool(abs(drop) < 0.01 and after["n_contacts"] > 0),
                    **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="leap,allegro,shadow")
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="legacy/results/hand_axis.json")
    a = ap.parse_args()
    widths = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.11]
    t0 = time.time()
    table = {}
    for key in a.hands.split(","):
        m, cfg, flex, tips, closed_gap = derive_flex(key)
        print(f"\n=== {key}: {cfg['note']}  (closed gap {closed_gap*100:.2f} cm) ===",
              flush=True)
        print(f"{'width(cm)':>10}{'feasible':>10}{'ncon':>6}{'eps':>8}"
              f"{'F(N)':>8}{'drop(cm)':>10}{'held':>6}")
        rows = []
        for w in widths:
            b = HandBench(key, flex, tips, (w / 2, 0.025, 0.025), a.mass)
            r = b.trial(w, a.mass)
            rows.append(r)
            if not r.get("feasible", False):
                print(f"{w*100:>10.1f}{'no':>10}{'':>6}{'':>8}{'':>8}{'':>10}{'-':>6}"
                      f"   (hand gap {r['gap_min']*100:.1f}-{r['gap_max']*100:.1f} cm)")
            else:
                print(f"{w*100:>10.1f}{'yes':>10}{r['n']:>6}{r['eps']:>8.3f}"
                      f"{r['f_grip']:>8.2f}{r['drop_m']*100:>10.2f}"
                      f"{str(r['held']):>6}", flush=True)
        table[key] = dict(note=cfg["note"], closed_gap_m=closed_gap, rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(table, indent=2, default=float))
    print("\n" + "=" * 72)
    print(f"{'hand':10}{'closed gap':>12}{'feasible':>10}{'HELD':>8}"
          f"{'widths held (cm)':>28}")
    for k, v in table.items():
        feas = [r for r in v["rows"] if r.get("feasible")]
        held = [r for r in feas if r.get("held")]
        print(f"{k:10}{v['closed_gap_m']*100:>10.2f} cm{len(feas):>10}"
              f"{len(held):>8}{str([round(r['width']*100,1) for r in held]):>28}")
    print("=" * 72)
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
