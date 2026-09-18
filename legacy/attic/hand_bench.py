"""Floating-hand grasp bench: can this hand hold this object against gravity?

No arm. The palm is fixed in space, the object is placed at the grasp point, the
fingers close, and gravity is the only test. This is how grasp capability is
normally measured, and it is the only way to compare hands: arm reachability is
a property of the robot, not the hand, and on the Vega it is tight enough to
mask everything else.

Why this exists. Two separate things have to be true before any "f5d6 cannot do
X" sentence is allowed into findings (rule-null-results): a working reference
under the identical protocol, and that reference failing across the claimed
boundary. LEAP and Allegro are the working references -- 16-DoF hands with real
thumb opposition, URDFs already on disk in DexTrack/assets. If they hold the box
on this bench and f5d6 does not, the opposition deficit is a boundary. If they
also fail, the bench is broken and nothing measured with it means anything.

Usage:
    python scripts/hand_bench.py --hand leap --mass 0.05
    python scripts/hand_bench.py --all
"""
from __future__ import annotations

import argparse, json, re, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dextrack_vega"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.epsilon import grasp_metrics   # noqa: E402

DEX = Path("/home/jetson3/projects/DexTrack/assets")
VEGA_URDF = Path("/home/jetson3/projects/dexmate/dexmate-urdf/robots/humanoid/"
                 "vega_1u/vega_1u_f5d6-obj.urdf")

HANDS = {
    "leap": dict(urdf=DEX / "leap_hand/leap_hand_right.urdf", kp=3.0, kv=0.1,
                 dof=16, note="LEAP 16-DoF, real thumb opposition"),
    "allegro": dict(urdf=DEX / "allegro_hand_description/urdf/"
                            "allegro_hand_description_right.urdf",
                    kp=3.0, kv=0.1, dof=16, note="Allegro 16-DoF"),
    "f5d6": dict(urdf=VEGA_URDF, kp=8.0, kv=0.3, dof=11,
                 note="Dexmate f5d6, 11 joints, underactuated",
                 keep_prefix="R_"),
}


def compile_mjcf(urdf: Path) -> str:
    """URDF -> MJCF text with absolute mesh paths and .glb visuals stripped."""
    tree = ET.parse(urdf)
    root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("visual",):
                for m in child.iter("mesh"):
                    if str(m.get("filename", "")).lower().endswith(".glb"):
                        parent.remove(child)
                        break
    tmp = urdf.with_name("_bench_stripped.urdf")
    tmp.write_text(ET.tostring(root, encoding="unicode"))
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
        raw = tmp.with_name("_bench_raw.xml")
        mujoco.mj_saveLastXML(str(raw), model)
        mjcf = raw.read_text()
        raw.unlink()
    finally:
        tmp.unlink()
    d = urdf.parent

    def abso(m):
        rel = m.group(1)
        return m.group(0) if rel.startswith("/") else f'file="{(d / rel).resolve()}"'
    return re.sub(r'file="([^"]+)"', abso, mjcf)


def build_scene(hand_key, box_half, box_mass, box_pos, friction="2.0 0.05 0.002"):
    """Hand MJCF + a free box, with position actuators on every hand joint."""
    cfg = HANDS[hand_key]
    mjcf = compile_mjcf(cfg["urdf"])
    root = ET.fromstring(mjcf)

    # joints we actuate: every hinge/slide in the model (optionally name-filtered)
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
                                            kp=str(cfg["kp"]), kv=str(cfg["kv"])))
    # freeze everything we are not actuating (e.g. the Vega arm) by welding it
    # implicitly: unactuated joints just stay at qpos 0 under gravity, so give
    # the whole model a high-damping default rather than letting it flop.
    wb = root.find("worldbody")
    obj = ET.SubElement(wb, "body", dict(name="object",
                                         pos=" ".join(str(v) for v in box_pos)))
    ET.SubElement(obj, "freejoint", dict(name="object_free"))
    ET.SubElement(obj, "geom", dict(name="object_geom", type="box",
                                    size=" ".join(str(v) for v in box_half),
                                    rgba="0.8 0.3 0.2 1", mass=str(box_mass),
                                    friction=friction))
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")
    return ET.tostring(root, encoding="unicode"), names


class Bench:
    def __init__(self, hand_key, box_half=(0.04, 0.045, 0.07), box_mass=0.05,
                 box_pos=(0, 0, 0)):
        xml, self.act_joints = build_scene(hand_key, box_half, box_mass, box_pos)
        self.hand_key = hand_key
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.obj_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.obj_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_q = self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.obj_v = self.m.jnt_dofadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in self.act_joints]
        self.qadr = np.array([self.m.jnt_qposadr[i] for i in jids])
        rng = self.m.jnt_range[jids]
        self.lo, self.hi = rng[:, 0].copy(), rng[:, 1].copy()
        bad = self.hi <= self.lo
        self.lo[bad], self.hi[bad] = -np.pi, np.pi
        self.aidx = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                f"act_{n}") for n in self.act_joints])
        self.n_act = len(self.act_joints)
        # every geom that is not the object = the hand
        self.hand_geoms = [g for g in range(self.m.ngeom) if g != self.obj_gid]

    def reset(self, obj_pos, obj_quat=(1, 0, 0, 0), q=None):
        mujoco.mj_resetData(self.m, self.d)
        if q is not None:
            self.d.qpos[self.qadr] = q
        self.d.qpos[self.obj_q:self.obj_q + 3] = obj_pos
        self.d.qpos[self.obj_q + 3:self.obj_q + 7] = obj_quat
        self.d.ctrl[:] = 0
        if q is not None:
            self.d.ctrl[self.aidx] = q
        mujoco.mj_forward(self.m, self.d)

    def pin_obj(self):
        self._pq = self.d.qpos[self.obj_q:self.obj_q + 7].copy()

    def _repin(self):
        self.d.qpos[self.obj_q:self.obj_q + 7] = self._pq
        self.d.qvel[self.obj_v:self.obj_v + 6] = 0.0

    def step(self, ctrl, n, pin=False):
        for _ in range(n):
            self.d.ctrl[self.aidx] = ctrl
            mujoco.mj_step(self.m, self.d)
            if pin:
                self._repin()

    def metrics(self):
        return grasp_metrics(self.m, self.d, self.obj_gid, self.obj_bid)

    def palm_frame(self):
        """A crude palm centre: mean position of all hand body origins."""
        idx = [b for b in range(self.m.nbody) if b != self.obj_bid and b != 0]
        return self.d.xpos[idx].mean(0).copy()

    def seat_and_hold(self, q_close, f_target=3.0, n_coarse=40, settle=12,
                      hold_s=2.0):
        """Close on a pinned object until f_target, release, then hold against
        gravity. Returns epsilon at seating and the drop."""
        self.pin_obj()
        q_open = np.zeros(self.n_act)
        self.step(q_open, 40, pin=True)
        seated, f = q_close, 0.0
        for k in range(1, n_coarse + 1):
            q = q_open + (k / n_coarse) * (q_close - q_open)
            self.step(q, settle, pin=True)
            f = self.metrics()["f_total"]
            seated = q
            if f >= f_target:
                break
        g = self.metrics()
        z0 = float(self.d.qpos[self.obj_q + 2])
        self.step(seated, int(hold_s / self.m.opt.timestep))   # RELEASE
        drop = z0 - float(self.d.qpos[self.obj_q + 2])
        return dict(epsilon=g["epsilon"], n_contacts=g["n_contacts"],
                    f_seated=float(f), drop_m=float(drop),
                    held=bool(drop < 0.01), frac=float(np.max(np.abs(seated)) /
                                                       (np.max(np.abs(q_close)) + 1e-9)))


def close_pose(b, frac=0.75):
    """A generic power-grasp closure: drive every joint `frac` of the way to the
    limit whose sign curls the fingers inward. Crude but hand-agnostic, which is
    the point -- no per-hand tuning that could favour one hand."""
    mid = 0.5 * (b.lo + b.hi)
    span = b.hi - b.lo
    return np.clip(mid + frac * 0.5 * span, b.lo, b.hi)


def run_hand(key, mass=0.05, half=(0.04, 0.045, 0.07), n_pos=7, verbose=True):
    b = Bench(key, box_half=half, box_mass=mass)
    b.reset((0, 0, 0))
    palm = b.palm_frame()
    best = None
    rows = []
    rng = np.random.default_rng(0)
    # sweep the object's placement around the palm, and the closure depth
    for dz in (-0.04, -0.02, 0.0, 0.02):
        for dx in (0.0, 0.03, -0.03):
            for frac in (0.5, 0.75, 1.0):
                for sign in (+1, -1):
                    pos = palm + np.array([dx, 0.0, dz])
                    q = close_pose(b, sign * frac)
                    b.reset(pos, q=np.zeros(b.n_act))
                    r = b.seat_and_hold(q)
                    r.update(dx=dx, dz=dz, frac=frac, sign=sign)
                    rows.append(r)
                    if best is None or (r["held"], r["epsilon"]) > (best["held"], best["epsilon"]):
                        best = r
    if verbose:
        print(f"\n=== {key}: {HANDS[key]['note']}  ({b.n_act} actuated joints)")
        print(f"    candidates {len(rows)}   held {sum(r['held'] for r in rows)}"
              f"   max eps {max(r['epsilon'] for r in rows):.4f}")
        print(f"    best: eps={best['epsilon']:.4f} ncon={best['n_contacts']} "
              f"F={best['f_seated']:.2f} N drop={best['drop_m']*100:.1f} cm "
              f"held={best['held']}")
    return dict(hand=key, note=HANDS[key]["note"], n_act=b.n_act,
                n_candidates=len(rows), n_held=int(sum(r["held"] for r in rows)),
                max_eps=float(max(r["epsilon"] for r in rows)), best=best, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="legacy/results/hand_bench.json")
    a = ap.parse_args()
    keys = list(HANDS) if a.all else [a.hand or "leap"]
    t0 = time.time()
    out = {}
    for k in keys:
        try:
            out[k] = run_hand(k, mass=a.mass)
        except Exception as e:
            print(f"\n=== {k}: FAILED TO BENCH: {type(e).__name__}: {str(e)[:200]}")
            out[k] = dict(hand=k, error=f"{type(e).__name__}: {e}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n{'hand':<10}{'n_act':>7}{'held/n':>12}{'max eps':>10}{'best drop(cm)':>15}")
    for k, v in out.items():
        if "error" in v:
            print(f"{k:<10}{'-':>7}{'ERROR':>12}")
            continue
        print(f"{k:<10}{v['n_act']:>7}{str(v['n_held'])+'/'+str(v['n_candidates']):>12}"
              f"{v['max_eps']:>10.4f}{v['best']['drop_m']*100:>15.1f}")
    print(f"\nwrote {a.out} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
