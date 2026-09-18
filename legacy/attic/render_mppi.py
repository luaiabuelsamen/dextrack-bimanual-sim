"""Render the seated-grasp + MPPI lift, with the force trace overlaid."""
import sys, json
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
from grasp_bench import Bench
from mppi_grasp import MPPI, FULL

W, MASS, LIFT_H, NC, SUB = 0.065, 0.05, 0.15, 60, 10

def main(plan=True, out=None, seed=1, w=640, h=480, stride=3, fps=25):
    out = out or ("legacy/figures/10_mppi_lift.gif" if plan else "legacy/figures/10_baseline_lift.gif")
    b = Bench("leap",(W/2,0.025,0.025),MASS)
    pl,info = b.plan(W, margin=0.006)
    f_pre,_ = b.open_until_clear(pl); b.place(f_pre, pl["centre"])
    pq = b.d.qpos[b.oq:b.oq+7].copy()
    def pinned(n, fr):
        for _ in range(n):
            b.d.ctrl[:] = b.ctrl(fr,0.0); mujoco.mj_step(b.m,b.d)
            b.d.qpos[b.oq:b.oq+7]=pq; b.d.qvel[b.ov:b.ov+6]=0.0
    pinned(150, f_pre)
    for i in range(1,31): pinned(8, f_pre+(pl["f_grasp"]-f_pre)*i/30)
    for i in range(1,21): pinned(8, pl["f_grasp"]+(pl["f_squeeze"]-pl["f_grasp"])*i/20)
    sf = pl["f_squeeze"]
    for _ in range(200):
        b.d.ctrl[:]=b.ctrl(sf,0.0); mujoco.mj_step(b.m,b.d)
    obj0 = b.d.qpos[b.oq:b.oq+3].copy(); quat0=b.d.qpos[b.oq+3:b.oq+7].copy()
    m = MPPI(b, substep=SUB, seed=seed); m.U = np.tile(sf*b.amp,(m.H,1))
    hold_n = NC//5
    sched = np.concatenate([np.zeros(hold_n), np.linspace(0,LIFT_H,NC-hold_n)])
    j0 = int(NC*0.75); sched[j0:j0+3] = sched[j0]-0.05; sched[j0+3:] = sched[j0]

    cam = mujoco.MjvCamera(); cam.distance = 0.40; cam.azimuth = 120; cam.elevation = -8
    ren = mujoco.Renderer(b.m, height=h, width=w)
    frames=[]; k=[0]; fmax=[0.0]; ftrace=[]
    def shoot():
        if k[0] % stride == 0:
            f = b.force_only(); fmax[0]=max(fmax[0],f); ftrace.append(f)
            cam.lookat[:] = b.d.qpos[b.oq:b.oq+3]
            ren.update_scene(b.d, camera=cam); img = ren.render().copy()
            z = float(b.d.qpos[b.oq+2]) - obj0[2]
            bar = int(np.clip(z/0.20,-1,1)*(h//2-12)); cx=w-26
            img[:, cx-10:cx+10]=(40,40,40)
            a1,b1=(h//2-bar,h//2) if bar>=0 else (h//2,h//2-bar)
            img[max(0,a1):min(h,b1), cx-8:cx+8]=(60,200,90) if bar>=0 else (220,70,60)
            img[h//2-1:h//2+1, cx-12:cx+12]=(255,255,255)
            # force bar, log scale, 1 N .. 2000 N
            fb = int(np.clip(np.log10(max(f,1.0))/np.log10(2000.0),0,1)*(h-40))
            img[h-20-fb:h-20, 16:30] = (240,170,60)
            if b.d.ncon: img[12:26,12:26]=(60,200,90)
            frames.append(img)
        k[0]+=1
    state0=np.empty(m.nstate); c=np.zeros(b.m.nu)
    for kk in range(NC):
        mujoco.mj_getState(b.m,b.d,state0,FULL)
        seg=sched[kk:kk+m.H]
        if len(seg)<m.H: seg=np.concatenate([seg,np.full(m.H-len(seg),sched[-1])])
        u = m.plan(state0, seg.repeat(SUB), obj0, quat0, b.m.opt.timestep) if plan else m.U[0].copy()
        c=np.zeros(b.m.nu); c[b.ja]=u; c[b.lift_a]=sched[kk]
        for _ in range(SUB):
            b.d.ctrl[:]=c; mujoco.mj_step(b.m,b.d); shoot()
        m.shift()
    lv0=float(sched[-1])
    n=int(0.06/b.m.opt.timestep)
    for i in range(n):
        c[b.lift_a]=lv0-0.06*(i+1)/n; b.d.ctrl[:]=c; mujoco.mj_step(b.m,b.d); shoot()
    for _ in range(4):
        for lv in (lv0-0.035, lv0):
            c[b.lift_a]=lv
            for _ in range(12):
                b.d.ctrl[:]=c; mujoco.mj_step(b.m,b.d); shoot()
    ren.close()
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    g=b.metrics()
    print(f"{'MPPI' if plan else 'baseline':>9}: lift {100*(float(b.d.qpos[b.oq+2])-obj0[2]):+.2f} cm  "
          f"Fpeak {fmax[0]:.1f} N  Fend {g['f_total']:.2f} N  ncon {g['n_contacts']}  "
          f"eps {g['epsilon']:.3f} -> {out}")

if __name__ == "__main__":
    main(plan=True)
    main(plan=False)
