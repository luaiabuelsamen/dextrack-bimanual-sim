"""Stills at each phase boundary of the expert, plus the numbers behind them."""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
from bimanual_env import BimanualBox, RH_HOME, LH_HOME, BASE_HALF
from bimanual_expert import Expert, palm_ctrl, OPEN, GRASP, FLAT, RH_GRASP_OFF

stage = sys.argv[1] if len(sys.argv)>1 else "grasp"
ex = Expert(two_handed=True); e = ex.e; m,d = e.m, e.d
rh_open = ex.rh_for_phi(0.0); rh_up = list(rh_open); rh_up[2] += 0.10
lh_press = palm_ctrl(LH_HOME, [0.0,-0.085,0.078]); lh_up=list(lh_press); lh_up[2]+=0.10

def cmd(rh,fing,lh,lfing,n):
    c=e.ctrl_vec(rh_pose=list(rh)+[0,0,0], lh_pose=list(lh)+[0,0,0],
                 rh_close=fing, lh_close=lfing)
    for _ in range(n): d.ctrl[:]=c; mujoco.mj_step(m,d)

cmd(rh_up,OPEN,lh_up,FLAT,250)
for i in range(1,61):
    a=i/60; cmd(rh_up,OPEN,[lh_up[j]+a*(lh_press[j]-lh_up[j]) for j in range(3)],FLAT,10)
cmd(rh_up,OPEN,lh_press,FLAT,200)
if stage in ("descend","grasp"):
    for i in range(1,61):
        a=i/60; cmd([rh_up[j]+a*(rh_open[j]-rh_up[j]) for j in range(3)],OPEN,lh_press,FLAT,10)
if stage=="grasp":
    for i in range(1,41):
        a=i/40; cmd(rh_open,[OPEN[j]+a*(GRASP[j]-OPEN[j]) for j in range(7)],lh_press,FLAT,10)
    cmd(rh_open,GRASP,lh_press,FLAT,200)

palm=d.xpos[e.palms["rh_"]]
tips=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f"rh_{b}") for b in ("if_ds","mf_ds","rf_ds","th_ds")]
tp=d.xpos[tips]
gc=0.5*(tp[:3].mean(0)+tp[3])
print(f"stage={stage}")
print(f"  knob world          {np.round(e.knob_pos(),4)}")
print(f"  rh palm             {np.round(palm,4)}   (commanded via ctrl {np.round(rh_open,4)})")
print(f"  rh grasp centre     {np.round(gc,4)}   miss = {np.linalg.norm(gc-e.knob_pos())*100:.2f} cm")
print(f"  rh tips z           {np.round(tp[:,2],4)}")
print(f"  lh palm             {np.round(d.xpos[e.palms['lh_']],4)}")
print(f"  lid {np.degrees(e.lid_angle()):.2f} deg  box {np.round(e.box_pos(),4)} tilt {e.box_tilt_deg():.2f}")
gn=lambda g:(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or f"g{g}")
print(f"  contacts ({d.ncon}):")
seen=set()
for i in range(d.ncon):
    c=d.contact[i]; k=(gn(c.geom1),gn(c.geom2))
    if k in seen: continue
    seen.add(k); print("     ",k[0],"<->",k[1])
m.vis.headlight.ambient[:]=(0.5,)*3; m.vis.headlight.diffuse[:]=(0.8,)*3
cam=mujoco.MjvCamera(); cam.lookat[:]=e.knob_pos(); cam.distance=0.34
cam.azimuth=128; cam.elevation=-14
ren=mujoco.Renderer(m,height=480,width=640); ren.update_scene(d,camera=cam)
out=f"figures/14_diag_{stage}.png"
imageio.imwrite(out, ren.render()); ren.close(); print("  ->",out)
