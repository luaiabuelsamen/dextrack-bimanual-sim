"""Render the scripted expert episode to a GIF so the phases can be inspected."""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
from handsim.control import expert as BE
from handsim.control.expert import Expert

def main(two=True, out=None, w=640, h=480, stride=16, fps=25):
    out = out or f"legacy/figures/13_expert_{'two' if two else 'one'}.gif"
    ex = Expert(two_handed=two); e = ex.e; m,d = e.m, e.d
    m.vis.headlight.ambient[:]=(0.5,0.5,0.5); m.vis.headlight.diffuse[:]=(0.8,0.8,0.8)
    cam = mujoco.MjvCamera(); cam.lookat[:]=[0.0,-0.02,0.08]
    cam.distance=0.72; cam.azimuth=128; cam.elevation=-14
    ren = mujoco.Renderer(m, height=h, width=w)
    frames=[]; k=[0]
    orig_step = mujoco.mj_step
    def shoot():
        if k[0] % stride == 0:
            ren.update_scene(d, camera=cam); img = ren.render().copy()
            out_cm = e.peg_out()*100
            bar = int(np.clip(out_cm/15.0,0,1)*(h-40))
            img[h-20-bar:h-20, 16:32] = (60,200,90)
            nk = sum(1 for i in range(d.ncon)
                     if e.knob_gid in (d.contact[i].geom1, d.contact[i].geom2))
            if nk: img[12:28,12:28]=(240,200,60)
            frames.append(img)
        k[0]+=1
    # monkeypatch mj_step so every integration step gets a chance to be captured
    def patched(mm, dd, nstep=1):
        orig_step(mm, dd, nstep); shoot()
    mujoco.mj_step = patched
    try:
        r = ex.run(verbose=False)
    finally:
        mujoco.mj_step = orig_step
    ren.close()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"{'two' if two else 'one'}-handed: peg out {r['peg_out_m']*100:.2f} cm  "
          f"base moved {r['box_disp_m']*100:.2f} cm (z {r['box_z_rise_m']*100:+.2f})  -> {out}")

if __name__ == "__main__":
    main(two=(sys.argv[1] != "one") if len(sys.argv)>1 else True)
