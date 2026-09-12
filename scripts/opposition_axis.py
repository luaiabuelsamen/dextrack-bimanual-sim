"""M2: the opposition axis. How closely can each hand's thumb meet its fingers?

This is the project's spine. `dextrack_vega` measured one number the hard way --
the Dexmate f5d6's thumb cannot come closer than 3.1 cm to its finger mean, at
any pose, which is why it cannot grasp. That number only means something next to
other hands measured the same way.

The measurement is purely kinematic and hand-agnostic:

    opposition floor = min over joint configurations of
                       || thumb_tip(q) - mean(finger_tips(q)) ||

subject to joint limits, multi-start to avoid local minima. Also reported: the
maximum of the same quantity (the aperture, i.e. the largest object the hand can
straddle) and the actuated DoF count.

A hand whose floor is 0 can pinch anything down to zero width. A hand whose floor
is 3.1 cm cannot oppose at all -- it can only press objects against something
else. That is the deficit the project is named for, and this puts a number on it
for every hand on disk.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
from dextrack_vega.assets import _compile_to_mjcf          # noqa: E402

MEN = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie"
VEGA = Path("/home/jetson3/projects/dexmate/dexmate-urdf/robots/humanoid/"
            "vega_1u/vega_1u_f5d6-obj.urdf")

HANDS = {
    "leap": dict(kind="mjcf", path=MEN / "leap_hand/right_hand.xml",
                 thumb="th_ds", fingers=["if_ds", "mf_ds", "rf_ds"],
                 joints=None, note="LEAP, 16 DoF"),
    "allegro": dict(kind="mjcf", path=MEN / "wonik_allegro/right_hand.xml",
                    thumb="th_tip", fingers=["ff_tip", "mf_tip", "rf_tip"],
                    joints=None, note="Wonik Allegro, 16 DoF"),
    "shadow": dict(kind="mjcf", path=MEN / "shadow_hand/right_hand.xml",
                   thumb="rh_thdistal",
                   fingers=["rh_ffdistal", "rh_mfdistal", "rh_rfdistal",
                            "rh_lfdistal"],
                   joints=None, note="Shadow Hand, 24 DoF"),
    "f5d6": dict(kind="urdf", path=VEGA,
                 thumb="R_th_l2",
                 fingers=["R_ff_l2", "R_mf_l2", "R_rf_l2", "R_lf_l2"],
                 joints=["R_th_j0", "R_th_j1", "R_th_j2",
                         "R_ff_j1", "R_ff_j2", "R_mf_j1", "R_mf_j2",
                         "R_rf_j1", "R_rf_j2", "R_lf_j1", "R_lf_j2"],
                 note="Dexmate f5d6, 11 joints (underactuated)"),
}


def load(key):
    cfg = HANDS[key]
    if cfg["kind"] == "mjcf":
        xml = cfg["path"].read_text()
        xml = xml.replace('meshdir="./assets/"',
                          f'meshdir="{(cfg["path"].parent / "assets").as_posix()}/"')
        xml = xml.replace('meshdir="assets"',
                          f'meshdir="{(cfg["path"].parent / "assets").as_posix()}"')
        m = mujoco.MjModel.from_xml_string(xml)
    else:
        # dextrack_vega's compiler strips the .glb visual meshes MuJoCo cannot
        # decode; the local one only handled <visual> elements and choked.
        m = mujoco.MjModel.from_xml_string(_compile_to_mjcf(cfg["path"]))
    return m, cfg


def tip_ids(m, cfg):
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg["thumb"])
    fg = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in cfg["fingers"]]
    assert th >= 0 and all(f >= 0 for f in fg), f"missing tip bodies for {cfg}"
    return th, fg


def joint_set(m, cfg):
    if cfg["joints"] is not None:
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
               for j in cfg["joints"]]
        ids = [i for i in ids if i >= 0]
    else:
        ids = [j for j in range(m.njnt)
               if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                    mujoco.mjtJoint.mjJNT_SLIDE)]
    qadr = np.array([m.jnt_qposadr[j] for j in ids])
    rng = m.jnt_range[ids]
    lo, hi = rng[:, 0].copy(), rng[:, 1].copy()
    bad = hi <= lo
    lo[bad], hi[bad] = -np.pi, np.pi
    return ids, qadr, lo, hi


def extremes(key, restarts=24, seed=0):
    m, cfg = load(key)
    d = mujoco.MjData(m)
    th, fg = tip_ids(m, cfg)
    ids, qadr, lo, hi = joint_set(m, cfg)
    rng = np.random.default_rng(seed)

    def gap(x):
        q = np.zeros(m.nq)
        q[qadr] = x
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        return float(np.linalg.norm(d.xpos[th] - d.xpos[fg].mean(0)))

    def opt(sign):
        best = None
        for i in range(restarts):
            x0 = 0.5 * (lo + hi) if i == 0 else lo + rng.random(len(lo)) * (hi - lo)
            r = minimize(lambda x: sign * gap(x), x0, method="L-BFGS-B",
                         bounds=list(zip(lo, hi)), options=dict(maxiter=800))
            if best is None or r.fun < best.fun:
                best = r
        return sign * best.fun, best.x

    floor, q_lo = opt(+1.0)
    aper, q_hi = opt(-1.0)
    return dict(hand=key, note=cfg["note"], n_joints=len(ids),
                opposition_floor_m=float(floor), aperture_m=float(aper),
                span_m=float(aper - floor))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/opposition_axis.json")
    ap.add_argument("--restarts", type=int, default=24)
    a = ap.parse_args(); t0 = time.time()
    rows = []
    print(f"{'hand':<10}{'joints':>8}{'floor(cm)':>12}{'aperture(cm)':>14}"
          f"{'span(cm)':>11}   note")
    for k in ("shadow", "allegro", "leap", "f5d6"):
        try:
            r = extremes(k, restarts=a.restarts)
        except Exception as e:
            print(f"{k:<10}   FAILED: {type(e).__name__}: {str(e)[:80]}")
            continue
        rows.append(r)
        print(f"{k:<10}{r['n_joints']:>8}{r['opposition_floor_m']*100:>12.2f}"
              f"{r['aperture_m']*100:>14.2f}{r['span_m']*100:>11.2f}   {r['note']}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {a.out}  ({time.time()-t0:.0f} s)")
    if rows:
        f = {r["hand"]: r["opposition_floor_m"] * 100 for r in rows}
        print("\nOpposition floor is the smallest object the hand can pinch.")
        for k, v in sorted(f.items(), key=lambda kv: kv[1]):
            bar = "#" * int(round(v * 6))
            print(f"  {k:<9}{v:>6.2f} cm  {bar}")


if __name__ == "__main__":
    main()
