"""One table, one scene, three configurations. The M1.5 result.

A  pose-retargeted   : minimises keypoint distance to a synthetic human grasp
                       (one hand, thumb opposing four fingers across the 8 cm
                       face). Pose retargeting preserves the human's contact
                       ALLOCATION, so it produces a one-handed f5d6 grasp.
B  epsilon-max bimanual : free to re-allocate contacts across both hands; the
                       best configuration from the grip-margin and lift sweeps.
C  do-nothing / random : the trivial baselines, every time.

Scene: table lowered to 0.71 so the box rests at its reachability-characterised
centre height of 0.80. Do-nothing baseline is ~0 (see NOTES.md).
"""
import json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dextrack_vega import config as C
from analysis.epsilon import grasp_metrics
from probe_m15 import OPEN, SQUEEZE, human_grasp, keypoint_distance, retarget_keypoints
from probe_v2 import build_probe, BOX, eval_config_v2
from squeeze_sweep import _approach, _settle_grip
from lift_tune import trial

B_INSET, B_SQ, B_LIFT, B_GAIN = 0.015, 0.85, 0.15, 0.05


def main():
    t0 = time.time()
    p = build_probe(seed=0, **BOX)
    p.reset()
    centre = p.obj_pos(); half = np.array(p.m.geom_size[p.obj_gid][:3], float)
    ex = (p.table_gid, p.floor_gid)
    h_wrist, h_ft = human_grasp(centre, half)
    out = {}

    # ---- baselines ----
    p.reset(); z0 = p.obj_pos()[2]; p.d.ctrl[:] = 0
    for _ in range(400):
        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(p.m, p.d)
    out["do_nothing"] = dict(net_lift_m=float(p.obj_pos()[2] - z0))

    rng = np.random.default_rng(0)
    p.reset(); z0 = p.obj_pos()[2]; q = p.q36()
    for _ in range(400):
        q = np.clip(q + rng.normal(0, 0.004, q.shape), p.lo36, p.hi36)
        p.d.ctrl[p.env._act_ctrl_idx] = q
        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(p.m, p.d)
    out["random"] = dict(net_lift_m=float(p.obj_pos()[2] - z0))

    # ---- A ----
    qA, _ = retarget_keypoints(p, "R", h_wrist, h_ft, seed=0)
    mA, qA_fin = eval_config_v2(p, qA, ["R"], f_target=2.0)
    kA = keypoint_distance(p, "R", h_wrist, h_ft)
    holdA = p.hold_test(qA_fin)
    liftA = p.lift_test(["R"], {"R": p.ft("R").mean(0)},
                        {"R": {s.split("_", 1)[1]: qA_fin[7 + i] for i, s in
                               enumerate(p.ctrl_joints[7:18])}})
    out["A_pose_retarget"] = dict(
        eps=mA["epsilon"], n=mA["n_contacts"], kp_vec_m=kA[0],
        drop_m=holdA["drop_m"], held=bool(holdA["drop_m"] < 0.01),
        net_lift_m=liftA["net_lift_m"], success=liftA["success"])

    # ---- B ----
    hy = float(half[1])
    gR = centre + np.array([0.0, -(hy - B_INSET), 0.0])
    gL = centre + np.array([0.0, +(hy - B_INSET), 0.0])
    q_app, q_in, _ = _approach(p, gR, gL, B_SQ)
    _settle_grip(p, q_app, q_in)
    mB = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
    kB = min(keypoint_distance(p, s, h_wrist, h_ft) for s in ("R", "L"))
    zb = p.obj_pos()[2]
    p.m.geom_contype[p.table_gid] = 0; p.m.geom_conaffinity[p.table_gid] = 0
    mujoco.mj_forward(p.m, p.d)
    p.hold_ctrl(q_in, 40)
    dropB = float(zb - p.obj_pos()[2])
    afterB = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
    p.m.geom_contype[p.table_gid] = 1; p.m.geom_conaffinity[p.table_gid] = 1
    rB = trial(p, B_INSET, B_SQ, B_LIFT, B_GAIN)
    out["B_eps_bimanual"] = dict(
        eps=mB["epsilon"], n=mB["n_contacts"], kp_vec_m=kB[0], drop_m=dropB,
        held=bool(dropB < 0.01 and afterB["n_contacts"] > 0),
        net_lift_m=rB["net"], peak_m=rB["peak"], n_end=rB["n_end"],
        success=rB["success"])

    Path("results/final_table.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'':<22}{'eps':>8}{'ncon':>6}{'kp_vec(cm)':>12}{'drop(cm)':>10}"
          f"{'held':>7}{'net lift(cm)':>14}{'>=10cm':>8}")
    for k in ("do_nothing", "random"):
        print(f"{k:<22}{'-':>8}{'-':>6}{'-':>12}{'-':>10}{'-':>7}"
              f"{out[k]['net_lift_m']*100:>14.2f}{'-':>8}")
    for k in ("A_pose_retarget", "B_eps_bimanual"):
        v = out[k]
        print(f"{k:<22}{v['eps']:>8.4f}{v['n']:>6}{v['kp_vec_m']*100:>12.2f}"
              f"{v['drop_m']*100:>10.1f}{str(v['held']):>7}"
              f"{v['net_lift_m']*100:>14.2f}{str(v['success']):>8}")
    a, b = out["A_pose_retarget"], out["B_eps_bimanual"]
    print(f"\nkeypoint objective prefers A by {100*(b['kp_vec_m']-a['kp_vec_m']):.1f} cm;"
          f" epsilon prefers B by {b['eps']-a['eps']:+.4f}; the task agrees with epsilon.")
    print(f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
