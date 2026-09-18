"""RETRACTED. Kept for provenance; it does not import.

It draws poses from the withdrawn opposition axis, and its imports `tip_ids`
and `joint_set` were removed from `handsim.hands.axis` in 8cc4b63, when that
module stopped carrying its own hand table and loader. Moved here from
src/handsim/viz/ because the retraction README is explicit that nothing under
src/ belongs to this lineage.

Original docstring follows.

Render each hand AT its measured opposition floor -- the pose where the thumb
comes as close to the finger mean as the joint limits allow.

This is the visual behind the axis table. For f5d6 it is NOT a grasp and is not
claimed to be: the point is precisely that the gap does not close.

Blue sphere  = thumb tip.   Green sphere = mean of the finger tips.
Red capsule  = the measured opposition floor between them.
"""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
from scipy.optimize import minimize
from handsim.hands.axis import load, tip_ids, joint_set


def floor_pose(key, restarts=24, seed=0):
    m, cfg = load(key); d = mujoco.MjData(m)
    th, fg = tip_ids(m, cfg)
    ids, qadr, lo, hi = joint_set(m, cfg)
    rng = np.random.default_rng(seed)
    def gap(x):
        q = np.zeros(m.nq); q[qadr] = x
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        return float(np.linalg.norm(d.xpos[th] - d.xpos[fg].mean(0)))
    best = None
    for i in range(restarts):
        x0 = 0.5*(lo+hi) if i == 0 else lo + rng.random(len(lo))*(hi-lo)
        r = minimize(gap, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(maxiter=800))
        if best is None or r.fun < best.fun: best = r
    q = np.zeros(m.nq); q[qadr] = best.x
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    return m, d, th, fg, float(best.fun)


def _add(sc, gtype, size, pos, rgba, mat=None):
    if sc.ngeom >= sc.maxgeom:
        return None
    g = sc.geoms[sc.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float),
                        np.asarray(pos, float),
                        (np.eye(3).flatten() if mat is None else mat),
                        np.asarray(rgba, np.float32))
    sc.ngeom += 1
    return g


def render(key, out, dist=0.16, azim=135, elev=-10, w=640, h=480):
    m, d, th, fg, g = floor_pose(key)
    tp = d.xpos[th].copy(); fp = d.xpos[fg].mean(0)
    mid = 0.5 * (tp + fp)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = mid; cam.distance = dist
    cam.azimuth, cam.elevation = azim, elev
    # brighten via the model headlight; MjvLight has no `directional` field in
    # this MuJoCo build, so adding a scene light directly raises AttributeError
    m.vis.headlight.ambient[:] = (0.55, 0.55, 0.55)
    m.vis.headlight.diffuse[:] = (0.80, 0.80, 0.80)
    m.vis.headlight.specular[:] = (0.10, 0.10, 0.10)
    ren = mujoco.Renderer(m, height=h, width=w)
    opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt)
    ren.update_scene(d, camera=cam, scene_option=opt)
    sc = ren.scene
    sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    cap = _add(sc, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.0035, 0, 0], np.zeros(3),
               [1.0, 0.2, 0.15, 1.0])
    if cap is not None:
        mujoco.mjv_connector(cap, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0035, tp, fp)
    _add(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [0.007]*3, tp, [0.25, 0.5, 1.0, 1.0])
    _add(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [0.007]*3, fp, [0.3, 0.95, 0.4, 1.0])
    img = ren.render().copy(); ren.close()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, img)
    print(f"{key:<9} floor {g*100:5.2f} cm  -> {out}")
    return g


CFG = {"f5d6": dict(dist=0.17, azim=150, elev=-8),
       "leap": dict(dist=0.26, azim=95, elev=-18),
       "allegro": dict(dist=0.16, azim=120, elev=-10),
       "shadow": dict(dist=0.16, azim=120, elev=-10)}

if __name__ == "__main__":
    # one hand per process: creating a second EGL Renderer after the first is
    # closed raises EGL_NOT_INITIALIZED on this machine.
    key = sys.argv[1]
    render(key, f"legacy/figures/11_floor_{key}.png", **CFG[key])
