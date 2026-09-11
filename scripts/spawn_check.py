"""Corrected spawn height: does the scripted expert lift anything at all?

BIM_BOX spawns the object 4 cm inside the table (top 0.770, box bottom 0.730).
Everything that followed is measured against a do-nothing baseline of +3.99 cm.
This re-runs the trivial baselines and the scripted expert with the box RESTING
on the table (z = 0.840) instead of inside it.
"""
import sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega/scripts")
from dextrack_vega import config as C
from dextrack_vega.envs.tracking_env import VegaTrackingEnv
import make_demo as MD

FIXED = dict(MD.BIM_BOX); FIXED["obj_pos"] = (0.55, 0.0, 0.8401)


def run(box, label, ctrl_seq=None, n=400):
    env = VegaTrackingEnv(sides=["R", "L"], seed=0, **box)
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    for j, v in C.HOME_POSTURE.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    if ctrl_seq is not None:
        d.qpos[env._jnt_qposadr] = ctrl_seq[0]
    mujoco.mj_forward(m, d)
    oq = env._obj_qadr; z0 = float(d.qpos[oq + 2]); zmax = z0
    steps = ctrl_seq if ctrl_seq is not None else [None] * n
    for c in steps:
        if c is not None:
            d.ctrl[env._act_ctrl_idx] = c
        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(m, d)
        zmax = max(zmax, float(d.qpos[oq + 2]))
    zf = float(d.qpos[oq + 2])
    print(f"{label:<46} z {z0:.4f} -> {zf:.4f}   net {100*(zf-z0):+6.2f} cm   max {100*(zmax-z0):+6.2f} cm")
    return zf - z0


print("--- ORIGINAL spawn (box 4 cm inside the table) ---")
run(MD.BIM_BOX, "do nothing")
g = MD.BimanualDemoGen(seed=0, **MD.BIM_BOX); g.generate_squeeze_lift()
run(MD.BIM_BOX, "scripted expert (squeeze-lift), open loop", np.array(g.rec_ctrl))

print("\n--- CORRECTED spawn (box resting on the table, z=0.8401) ---")
run(FIXED, "do nothing")
rng = np.random.default_rng(0)
g2 = MD.BimanualDemoGen(seed=0, **FIXED); g2.generate_squeeze_lift()
seq = np.array(g2.rec_ctrl)
run(FIXED, "random action walk", seq[0] + np.cumsum(
    rng.normal(0, 0.01, size=seq.shape), axis=0))
run(FIXED, "scripted expert (squeeze-lift), open loop", seq)
