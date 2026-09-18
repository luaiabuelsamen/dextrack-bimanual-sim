import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
from bimanual_expert import Expert
two = (sys.argv[1] != "one") if len(sys.argv)>1 else True
ex=Expert(two_handed=two); e=ex.e; m,d=e.m,e.d
orig=mujoco.mj_step; shots={}
state={"n":0}
def patched(mm,dd,nstep=1):
    orig(mm,dd,nstep); state["n"]+=1
mujoco.mj_step=patched
try: r=ex.run(pull=0.14, verbose=False)
finally: mujoco.mj_step=orig
m.vis.headlight.ambient[:]=(0.5,)*3; m.vis.headlight.diffuse[:]=(0.8,)*3
cam=mujoco.MjvCamera(); cam.lookat[:]=[0.0,-0.02,0.12]
cam.distance=0.62; cam.azimuth=122; cam.elevation=-12
ren=mujoco.Renderer(m,height=480,width=640); ren.update_scene(d,camera=cam)
out=f"legacy/figures/15_final_{'two' if two else 'one'}.png"
imageio.imwrite(out, ren.render()); ren.close()
print(f"{'two' if two else 'one'}-handed END: peg out {r['peg_out_m']*100:.2f} cm  "
      f"base moved {r['box_disp_m']*100:.2f} cm (z {r['box_z_rise_m']*100:+.2f})  "
      f"tilt {r['box_tilt_deg']:.1f}  contacts {r['knob_contacts']}  ok={r['success']} -> {out}")
