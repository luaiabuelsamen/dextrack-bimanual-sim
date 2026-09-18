"""Render the bimanual scene. Called with a stage name; renders a still."""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
from bimanual_env import BimanualBox

def main(out="legacy/figures/12_bimanual_scene.png", w=640, h=480,
         dist=0.62, azim=135, elev=-18, settle=400):
    e = BimanualBox(); m,d = e.m, e.d
    mujoco.mj_forward(m,d)
    for _ in range(settle):
        d.ctrl[:] = e.ctrl_vec(); mujoco.mj_step(m,d)
    m.vis.headlight.ambient[:] = (0.5,0.5,0.5)
    m.vis.headlight.diffuse[:] = (0.75,0.75,0.75)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.02, 0.0, 0.08]; cam.distance = dist
    cam.azimuth, cam.elevation = azim, elev
    ren = mujoco.Renderer(m, height=h, width=w)
    ren.update_scene(d, camera=cam)
    imageio.imwrite(out, ren.render()); ren.close()
    print(f"lid {np.degrees(e.lid_angle()):.1f} deg  box {np.round(e.box_pos(),4)} "
          f"tilt {e.box_tilt_deg():.2f}  ncon {d.ncon}  -> {out}")
    gn=lambda g:(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or f"g{g}")
    seen=set()
    for i in range(d.ncon):
        c=d.contact[i]; k=(gn(c.geom1),gn(c.geom2))
        if k not in seen: seen.add(k); print("   contact:",k[0],"<->",k[1])

if __name__ == "__main__":
    main()
