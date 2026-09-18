"""Render the one physically defensible LEAP pick: 0.05 kg, +14.5 cm."""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
from leap_final import Leap, FINGERS, FLEX, THUMB
from analysis.epsilon import grasp_metrics

W, FLEXV, THUMBV, EXTRA, LIFT_H = 0.045, [0.9,1.2,0.6], [1.6,0.9,1.0,0.6], 0.03, 0.15

def main(out="legacy/figures/9_leap_grasp_005kg.gif", w=620, h=470, stride=4, fps=25):
    L = Leap((W/2, 0.025, 0.025), 0.05)
    f_touch, gap, centre = L.plan(FLEXV, THUMBV, W)
    f_grip = min(f_touch + EXTRA, 1.0)
    print(f"gap {gap*100:.2f} cm  f_touch {f_touch:.3f}  f_grip {f_grip:.3f}  centre {np.round(centre,4)}")
    mujoco.mj_resetData(L.m, L.d)
    L.d.qpos[L.oq:L.oq+3] = centre; L.d.qpos[L.oq+3] = 1.0
    mujoco.mj_forward(L.m, L.d)
    pq = L.d.qpos[L.oq:L.oq+7].copy()
    cam = mujoco.MjvCamera(); cam.lookat[:] = centre; cam.distance = 0.42
    cam.azimuth = 120; cam.elevation = -8
    ren = mujoco.Renderer(L.m, height=h, width=w)
    frames=[]; k=[0]; z0=[float(L.d.qpos[L.oq+2])]
    def shoot(pinned):
        if k[0] % stride == 0:
            g = grasp_metrics(L.m, L.d, L.ogid, L.obid)
            cam.lookat[:] = [centre[0], centre[1],
                             centre[2] + max(0.0, float(L.d.qpos[L.oq+2])-z0[0])*0.5]
            ren.update_scene(L.d, camera=cam); img = ren.render().copy()
            z = float(L.d.qpos[L.oq+2]) - z0[0]
            bar = int(np.clip(z/0.20, -1, 1)*(h//2-12)); cx = w-26
            img[:, cx-10:cx+10] = (40,40,40)
            a,b = (h//2-bar, h//2) if bar>=0 else (h//2, h//2-bar)
            img[max(0,a):min(h,b), cx-8:cx+8] = (60,200,90) if bar>=0 else (220,70,60)
            img[h//2-1:h//2+1, cx-12:cx+12] = (255,255,255)
            if g["n_contacts"]>0: img[12:26,12:26] = (60,200,90)
            if pinned: img[12:26,32:46] = (70,130,240)
            frames.append(img)
        k[0]+=1
    def stp(fr, n, pin, lift=0.0):
        c = L.ctrl([fr*v for v in FLEXV], [fr*v for v in THUMBV], lift)
        for _ in range(n):
            L.d.ctrl[:]=c; mujoco.mj_step(L.m,L.d)
            if pin:
                L.d.qpos[L.oq:L.oq+7]=pq; L.d.qvel[L.ov:L.ov+6]=0.0
            shoot(pin)
    stp(0.0,150,True)
    for i in range(1,61): stp(f_touch*i/60, 8, True)
    for i in range(1,21): stp(f_touch+(f_grip-f_touch)*i/20, 10, True)
    g = grasp_metrics(L.m,L.d,L.ogid,L.obid)
    print(f"at grasp: ncon={g['n_contacts']} F={g['f_total']:.1f} N eps={g['epsilon']:.3f}")
    stp(f_grip, 500, False)                      # RELEASE
    print(f"after release: z {float(L.d.qpos[L.oq+2])-z0[0]:+.4f} m  "
          f"ncon={grasp_metrics(L.m,L.d,L.ogid,L.obid)['n_contacts']}")
    zl = float(L.d.qpos[L.oq+2])
    for i in range(1,26): stp(f_grip, 30, False, lift=LIFT_H*i/25)
    stp(f_grip, 600, False, lift=LIFT_H)
    for cyc in range(4):                       # shake: carrying vs grasping
        stp(f_grip, 12, False, lift=LIFT_H-0.035)
        stp(f_grip, 12, False, lift=LIFT_H)
    stp(f_grip, 400, False, lift=LIFT_H)
    ren.close()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    gf = grasp_metrics(L.m,L.d,L.ogid,L.obid)
    print(f"NET LIFT {100*(float(L.d.qpos[L.oq+2])-zl):+.2f} cm   "
          f"contacts at top {gf['n_contacts']}   eps {gf['epsilon']:.3f} -> {out}")
    # a still at the top, to actually look at
    ren2 = mujoco.Renderer(L.m, height=480, width=640)
    ren2.update_scene(L.d, camera=cam)
    imageio.imwrite("legacy/figures/9_leap_grasp_top.png", ren2.render()); ren2.close()

if __name__ == "__main__":
    main()
