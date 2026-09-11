"""Pairing protocol: run the KNOWN-GOOD scripted bimanual expert through this
project's own harness, and score it with this project's own epsilon.

dextrack_vega measured +21.3 cm for the open-loop replay of
BimanualDemoGen.generate_squeeze_lift on BIM_BOX. If this harness reproduces
that, the harness is sound and any worse number from the epsilon search is the
search's fault. If it does not, the harness is the bug and nothing measured with
it means anything yet. Either way this runs before any 0/N is believed.
"""
import sys, json
from pathlib import Path
import numpy as np, mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dextrack_vega"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dextrack_vega" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dextrack_vega import config as C
from dextrack_vega.envs.tracking_env import VegaTrackingEnv
from analysis.epsilon import grasp_metrics
import make_demo as MD


def main():
    # 1. generate the reference exactly as the validated demo does
    gen = MD.BimanualDemoGen(seed=0, **MD.BIM_BOX)
    lift_h = gen.generate_squeeze_lift()
    ref_ctrl = np.array(gen.rec_ctrl)
    print(f"reference: {len(ref_ctrl)} control steps, kinematic lift target {lift_h*100:.0f} cm")

    # 2. replay it OPEN-LOOP in a fresh env (zero residual, real physics)
    env = VegaTrackingEnv(sides=["R", "L"], seed=0, **MD.BIM_BOX)
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    for j, v in C.HOME_POSTURE.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    d.qpos[env._jnt_qposadr] = gen.rec_q[0]
    mujoco.mj_forward(m, d)

    oq = env._obj_qadr
    ogid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    obid = env._bid("object")
    tgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    fgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    z0 = float(d.qpos[oq + 2])

    trace, zmax = [], z0
    for k, ctrl in enumerate(ref_ctrl):
        d.ctrl[env._act_ctrl_idx] = ctrl
        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(m, d)
        z = float(d.qpos[oq + 2]); zmax = max(zmax, z)
        if k % 10 == 0:
            g = grasp_metrics(m, d, ogid, obid, (tgid, fgid))
            trace.append(dict(k=k, z_rel=z - z0, eps=g["epsilon"], n=g["n_contacts"],
                              f_total=g["f_total"], delta=g["delta"]))
    zf = float(d.qpos[oq + 2])
    print(f"\nOPEN-LOOP REPLAY: net lift {100*(zf-z0):+.1f} cm   max {100*(zmax-z0):+.1f} cm"
          f"   (dextrack_vega measured +21.3 cm)")
    print(f"\n{'step':>5}{'z_rel(cm)':>11}{'eps':>9}{'ncon':>6}{'F_tot(N)':>10}{'delta':>9}")
    for t in trace[::4]:
        dl = t["delta"]
        print(f"{t['k']:>5}{t['z_rel']*100:>11.2f}{t['eps']:>9.4f}{t['n']:>6}"
              f"{t['f_total']:>10.2f}{(dl if np.isfinite(dl) else 999):>9.2f}")
    held = [t for t in trace if t["z_rel"] > 0.05]
    if held:
        e = [t["eps"] for t in held]; dd = [t["delta"] for t in held if np.isfinite(t["delta"])]
        print(f"\nwhile airborne (>5 cm, n={len(held)}): eps median {np.median(e):.4f}"
              f"  min {min(e):.4f}   delta median {np.median(dd) if dd else float('nan'):.3f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/expert_baseline.json").write_text(json.dumps(
        dict(net_lift_m=zf - z0, max_lift_m=zmax - z0, trace=trace), indent=2))


if __name__ == "__main__":
    main()
