"""M1.5 probe, v2. Two changes, both forced by what v1 measured.

1. SCENE. v1 corrected the box spawn by raising the box to rest on the table
   (z 0.80 -> 0.8401). That removes the 4 cm penetration but moves the box 4 cm
   higher, into the part of the arm's workspace that dextrack_vega already
   measured as having no vertical headroom. Here the TABLE is lowered instead
   (table_z 0.75 -> 0.71, top 0.730) so the box rests at exactly its original
   centre height of 0.80 and every arm-reachability result still applies.

2. CLOSURE. v1 drove the fingers to a fixed closed pose. Against a position
   servo that commands a closure *inside* the box: 36-190 N of contact force on
   a 0.05 kg object, which then squirts out. Real grippers do not do this --
   they seat by contact and hold by a small position offset past it. `seat_to_force`
   ramps the closure and stops at a target contact force, which is the
   engineered-ceiling-first check that rule-null-results asks for before any
   0/N is believed.

Everything else -- epsilon, the three configurations, the hold and lift tests --
is unchanged from v1.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dextrack_vega"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dextrack_vega import config as C                        # noqa: E402
from dextrack_vega import assets                             # noqa: E402
from dextrack_vega.envs.tracking_env import VegaTrackingEnv  # noqa: E402
from analysis.epsilon import grasp_metrics                   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_m15 import (OPEN, SQUEEZE, PINCH, Probe, human_grasp,   # noqa: E402
                       keypoint_distance, retarget_keypoints, lerp_fingers)

# Table lowered so the box rests at its original, reachability-characterised
# centre height. top = 0.71 + 0.02 = 0.730 = box bottom at obj z 0.80.
TABLE_Z = 0.71
BOX = dict(obj_type="box", obj_dims=(0.04, 0.045, 0.07),
           obj_pos=(0.55, 0.0, 0.80), obj_mass=0.05)


def build_probe(seed=0, **box):
    """Write the scene with the lowered table, then build the Probe against it.

    VegaTrackingEnv would rebuild the scene with its own default table_z and
    overwrite ours, so it is constructed with rebuild=False for this one call.
    """
    assets.build_scene(sides=["R", "L"], table_z=TABLE_Z, **box)
    orig = VegaTrackingEnv.__init__

    def patched(self, *a, **kw):
        kw["rebuild"] = False
        return orig(self, *a, **kw)

    VegaTrackingEnv.__init__ = patched
    try:
        return Probe(seed=seed, **box)
    finally:
        VegaTrackingEnv.__init__ = orig


def _fnow(p):
    return grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid,
                         (p.table_gid, p.floor_gid))["f_total"]


def seat_to_force(p, q_lo, q_hi, f_target=2.0, n_coarse=30, settle=5, n_bisect=7):
    """Advance from q_lo toward q_hi until the object's contact force exceeds
    f_target, then BACK THE COMMAND OFF by bisection until it sits just above it.

    The back-off is the half that matters. A coarse ramp alone overshoots badly
    here -- one 2 mm increment of arm command took the measured force from under
    2 N to 110 N, because the kp=600 arm servo lags the command and arrives
    already deep inside the object. Seat by contact, hold by a small position
    offset past it: the standard gripper recipe.
    """
    lo_f, hi_f, found = 0.0, 1.0, False
    for k in range(1, n_coarse + 1):
        frac = k / n_coarse
        p.hold_ctrl(q_lo + frac * (q_hi - q_lo), settle)
        if _fnow(p) > f_target:
            lo_f, hi_f, found = (k - 1) / n_coarse, frac, True
            break
    if not found:
        return q_hi, _fnow(p), 1.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo_f + hi_f)
        p.hold_ctrl(q_lo + mid * (q_hi - q_lo), settle)
        if _fnow(p) > f_target:
            hi_f = mid
        else:
            lo_f = mid
    q = q_lo + hi_f * (q_hi - q_lo)
    p.hold_ctrl(q, settle)
    return q, _fnow(p), hi_f


def eval_config_v2(p, q_target, sides, standoff=0.08, f_target=2.0, settle=25):
    """Approach clear, descend, then close by FORCE, then settle. Fully physical
    -- no pinning, because a force-limited closure does not eject the object."""
    p.reset()
    centre = p.obj_pos()
    p.set_q36(q_target)
    goal = {s: p.ft(s).mean(0) for s in sides}
    away = {}
    for s in sides:
        v = goal[s] - centre; v[2] = 0.0
        nv = np.linalg.norm(v)
        away[s] = (v / nv) if nv > 1e-6 else np.array(
            [0.0, -1.0 if s == "R" else 1.0, 0.0])

    p.reset()
    q_app = q_target.copy()
    for s in sides:
        q_app[p.block[s]:p.block[s] + 7] = p.ik_side_to(s, goal[s] + standoff * away[s])
        q_app = p.fingers36(q_app, s, OPEN)
    p.set_q36(q_app)
    p.hold_ctrl(q_app, 10)

    q_desc = q_target.copy()
    for s in sides:
        q_desc = p.fingers36(q_desc, s, OPEN)
    # descend by force too: the arm pressing in is itself a closure on this hand
    q_mid, f_desc, frac_d = seat_to_force(p, q_app, q_desc, f_target)
    q_fin, f_close, frac_c = seat_to_force(p, q_mid, q_target, f_target)
    p.hold_ctrl(q_fin, settle)

    m = p.eps_now()
    m["f_seated"] = float(f_close)
    m["frac_descend"] = float(frac_d)
    m["frac_close"] = float(frac_c)
    m["displaced_m"] = float(np.linalg.norm(p.obj_pos() - centre))
    m["ejected"] = bool(m["displaced_m"] > 0.05)
    return m, q_fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="legacy/results/probe_v2.json")
    ap.add_argument("--f-target", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    p = build_probe(seed=a.seed, **BOX)
    p.reset()
    centre = p.obj_pos()
    half = np.array(p.m.geom_size[p.obj_gid][:3], dtype=float)
    hy = float(half[1])
    tb = mujoco.mj_name2id(p.m, mujoco.mjtObj.mjOBJ_BODY, "table")
    top = p.m.body_pos[tb][2] + p.m.geom_pos[p.table_gid][2] + p.m.geom_size[p.table_gid][2]
    print(f"table top {top:.4f}   box bottom {centre[2]-half[2]:.4f}   "
          f"penetration {top-(centre[2]-half[2]):+.4f} m   mass "
          f"{p.m.body_mass[p.obj_bid]:.3f} kg  w_req {p.m.body_mass[p.obj_bid]*9.81:.3f} N")
    assert abs(top - (centre[2] - half[2])) < 2e-3, "box is not resting on the table"

    # trivial baselines, on this scene, every time
    p.reset(); z0 = p.obj_pos()[2]
    p.d.ctrl[:] = 0
    for _ in range(400):
        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(p.m, p.d)
    dn = float(p.obj_pos()[2] - z0)
    print(f"baseline do-nothing net lift {dn*100:+.2f} cm")

    h_wrist, h_ft = human_grasp(centre, half)
    out = dict(scene=dict(table_top=float(top), centre=centre.tolist(),
                          half=half.tolist(), penetration=float(top - (centre[2] - half[2]))),
               baselines=dict(do_nothing_m=dn), f_target=a.f_target,
               configs={}, search={})

    # ---- A: pose retargeting ----
    print("\n[A] retargeting keypoints ...", flush=True)
    qA, fA = retarget_keypoints(p, "R", h_wrist, h_ft, seed=a.seed)
    mA, qA_fin = eval_config_v2(p, qA, ["R"], f_target=a.f_target)
    kA = keypoint_distance(p, "R", h_wrist, h_ft)
    holdA = p.hold_test(qA_fin)
    print(f"    eps={mA['epsilon']:.4f} ncon={mA['n_contacts']} F={mA['f_seated']:.2f} N"
          f"  kp={kA[0]*100:.2f} cm  disp={mA['displaced_m']*100:.1f} cm"
          f"  drop={holdA['drop_m']*100:.1f} cm")
    out["configs"]["A_pose_retarget"] = dict(kind="unimanual", metrics=mA,
        keypoint_vec_m=kA[0], hold=holdA, q=qA_fin.tolist())

    # ---- B: bimanual epsilon search ----
    print("\n[B] searching bimanual ...", flush=True)
    logB, best = [], None
    for inset in np.linspace(0.0, 0.025, 6):
        for dz in (-0.02, 0.0, 0.02):
            p.reset()
            gR = centre + np.array([0.0, -(hy - inset), dz])
            gL = centre + np.array([0.0, +(hy - inset), dz])
            aR, aL = p.ik_side_to("R", gR), p.ik_side_to("L", gL)
            for sq in (0.7, 0.85, 1.0):
                q = p.q36(); q[0:7], q[18:25] = aR, aL
                f = lerp_fingers(OPEN, SQUEEZE, sq)
                q = p.fingers36(q, "R", f); q = p.fingers36(q, "L", f)
                m, qf = eval_config_v2(p, q, ["R", "L"], f_target=a.f_target)
                rec = dict(inset=float(inset), dz=float(dz), sq=float(sq),
                           eps=m["epsilon"], n=m["n_contacts"], f=m["f_seated"],
                           disp=m["displaced_m"], ejected=m["ejected"])
                logB.append(rec)
                if not m["ejected"] and (best is None or rec["eps"] > best[0]["eps"]):
                    best = (rec, qf.copy(), {"R": gR, "L": gL}, f, m)
    out["search"]["bimanual"] = logB
    if best is None:
        print("    NO non-ejecting bimanual configuration found "
              f"({len(logB)} candidates)")
        out["configs"]["B_eps_bimanual"] = dict(kind="bimanual", found=False,
                                                n_candidates=len(logB))
    else:
        recB, qB, gB, fB, mB = best
        print(f"    best {recB}")
        holdB = p.hold_test(qB)
        print(f"    hold drop={holdB['drop_m']*100:.1f} cm")
        kB = min(keypoint_distance(p, s, h_wrist, h_ft) for s in ("R", "L"))
        eval_config_v2(p, qB, ["R", "L"], f_target=a.f_target)
        liftB = p.lift_test(["R", "L"], gB, {"R": fB, "L": fB})
        print(f"    lift {liftB}")
        out["configs"]["B_eps_bimanual"] = dict(kind="bimanual", found=True,
            best=recB, metrics=mB, keypoint_vec_m=kB[0], hold=holdB, lift=liftB,
            q=qB.tolist())

    out["wall_s"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out} ({out['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
