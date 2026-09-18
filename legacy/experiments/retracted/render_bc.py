"""Render the BC policy doing the task (seed 0, which reaches the expert)."""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
from handsim.learning import bc as BC
from handsim.control.expert import Expert, palm_ctrl
from handsim.envs.bimanual import LH_HOME, BASE_HALF


def main():
    z = np.load("legacy/results/bc_demos.npz")
    model = BC.train(z["obs"], z["act"], seed=0, epochs=240)
    fn = BC.policy_fn(model)

    SEED = 5000
    rng = np.random.default_rng(SEED)
    kw = dict(hinge_friction=float(rng.uniform(0.9,1.8)),
              base_mass=float(rng.uniform(0.06,0.12)))
    ex = Expert(two_handed=True, **kw); e = ex.e; m,d = e.m, e.d
    ex.pre_frac,_ = ex.start_at_pregrasp(palm_ctrl(LH_HOME,[0.0,-0.085,0.044]))
    out0 = ex.out0; box0 = np.array([0.,0.,BASE_HALF[2]])
    m.vis.headlight.ambient[:]=(0.5,)*3; m.vis.headlight.diffuse[:]=(0.8,)*3
    cam = mujoco.MjvCamera(); cam.lookat[:]=[0.0,-0.02,0.10]
    cam.distance=0.70; cam.azimuth=128; cam.elevation=-14
    ren = mujoco.Renderer(m, height=480, width=640)
    frames=[]; N=3470; stride=16
    lo=m.actuator_ctrlrange[:,0]; hi=m.actuator_ctrlrange[:,1]
    lim=m.actuator_ctrllimited.astype(bool)
    for k in range(N):
        c = fn(BC.observe(e, k/N))
        d.ctrl[:] = np.where(lim, np.clip(c,lo,hi), c)
        mujoco.mj_step(m,d)
        if k % stride == 0:
            ren.update_scene(d, camera=cam); img = ren.render().copy()
            out = (e.peg_out()-out0)*100
            bar = int(np.clip(out/15.0,0,1)*(480-40))
            img[480-20-bar:480-20, 16:32] = (42,120,214)
            frames.append(img)
    ren.close()
    Path("legacy/figures").mkdir(exist_ok=True)
    imageio.mimsave("legacy/figures/17_bc_policy.gif", frames, fps=25, loop=0)
    out=(e.peg_out()-out0)*100; lift=(e.box_pos()[2]-box0[2])*100
    print(f"BC policy: peg out {out:.2f} cm  base lift {lift:+.2f} cm  "
          f"ok={bool(out>=8 and lift<2 and e.box_tilt_deg()<15)} -> figures/17_bc_policy.gif")


if __name__ == "__main__":
    main()
