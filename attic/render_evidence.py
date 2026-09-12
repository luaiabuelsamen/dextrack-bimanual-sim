"""Render the three runs that show the spawn defect, side by side in time.

Each GIF overlays the object's height relative to its own start, so the number
in the caption is visible in the frame.
"""
import sys, argparse
from pathlib import Path
import numpy as np, mujoco
import imageio.v2 as imageio
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega/scripts")
from dextrack_vega import config as C
from dextrack_vega.envs.tracking_env import VegaTrackingEnv
import make_demo as MD

FIXED = dict(MD.BIM_BOX); FIXED["obj_pos"] = (0.55, 0.0, 0.8401)


def render(box, ctrl_seq, out, label, n=400, w=560, h=420, stride=3, fps=25):
    env = VegaTrackingEnv(sides=["R", "L"], seed=0, **box)
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    for j, v in C.HOME_POSTURE.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    if ctrl_seq is not None:
        d.qpos[env._jnt_qposadr] = ctrl_seq[0]
    mujoco.mj_forward(m, d)
    oq = env._obj_qadr; z0 = float(d.qpos[oq + 2])

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.0, 0.85]; cam.distance = 1.15
    cam.azimuth = 35; cam.elevation = -15
    ren = mujoco.Renderer(m, height=h, width=w)
    frames, zs = [], []
    steps = list(ctrl_seq) if ctrl_seq is not None else [None] * n
    for k, c in enumerate(steps):
        if c is not None:
            d.ctrl[env._act_ctrl_idx] = c
        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(m, d)
        z = float(d.qpos[oq + 2]); zs.append(z - z0)
        if k % stride == 0:
            ren.update_scene(d, camera=cam)
            img = ren.render().copy()
            bar = int(np.clip((z - z0) / 0.25, -1, 1) * (h // 2 - 10))
            cx = w - 26
            img[:, cx - 10:cx + 10] = (40, 40, 40)
            a, b = (h // 2 - bar, h // 2) if bar > 0 else (h // 2, h // 2 - bar)
            img[max(0, a):min(h, b), cx - 8:cx + 8] = (
                (60, 200, 90) if bar > 0 else (220, 70, 60))
            img[h // 2 - 1:h // 2 + 1, cx - 12:cx + 12] = (255, 255, 255)
            frames.append(img)
    ren.close()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"{label:<46} net {100*zs[-1]:+6.2f} cm  max {100*max(zs):+6.2f} cm  -> {out}")
    return zs


if __name__ == "__main__":
    g_orig = MD.BimanualDemoGen(seed=0, **MD.BIM_BOX); g_orig.generate_squeeze_lift()
    g_fix = MD.BimanualDemoGen(seed=0, **FIXED); g_fix.generate_squeeze_lift()
    render(MD.BIM_BOX, None, "figures/1_donothing_original_spawn.gif",
           "do nothing, ORIGINAL spawn", n=260)
    render(MD.BIM_BOX, np.array(g_orig.rec_ctrl),
           "figures/2_expert_original_spawn.gif", "scripted expert, ORIGINAL spawn")
    render(FIXED, np.array(g_fix.rec_ctrl),
           "figures/3_expert_corrected_spawn.gif", "scripted expert, CORRECTED spawn")
