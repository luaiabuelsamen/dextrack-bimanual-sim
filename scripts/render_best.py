"""Render the best squeeze-lift configuration found by the sweep."""
import sys, json
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dextrack_vega import config as C
from analysis.epsilon import grasp_metrics
from probe_v2 import build_probe, BOX
from squeeze_sweep import _approach

INSET, SQ, DZ, LIFT_H = 0.012, 1.0, 0.0, 0.15


def main(out="figures/4_best_squeeze_lift.gif", w=600, h=460, stride=3, fps=25):
    p = build_probe(seed=0, **BOX)
    centre = np.array(BOX["obj_pos"], float)
    hy = float(p.m.geom_size[p.obj_gid][1])
    gR = centre + np.array([0.0, -(hy - INSET), DZ])
    gL = centre + np.array([0.0, +(hy - INSET), DZ])
    q_app, q_in, f = _approach(p, gR, gL, SQ)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.0, 0.84]; cam.distance = 1.0
    cam.azimuth = 35; cam.elevation = -12
    ren = mujoco.Renderer(p.m, height=h, width=w)
    frames, trace = [], []
    ex = (p.table_gid, p.floor_gid)

    p.reset(); p.set_q36(q_app)
    z0 = p.obj_pos()[2]
    k = [0]

    def shoot():
        z = p.obj_pos()[2] - z0
        g = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
        trace.append(dict(k=k[0], z=z, eps=g["epsilon"], n=g["n_contacts"]))
        if k[0] % stride == 0:
            ren.update_scene(p.d, camera=cam)
            img = ren.render().copy()
            bar = int(np.clip(z / 0.20, -1, 1) * (h // 2 - 12))
            cx = w - 26
            img[:, cx - 10:cx + 10] = (40, 40, 40)
            a, b = (h // 2 - bar, h // 2) if bar > 0 else (h // 2, h // 2 - bar)
            img[max(0, a):min(h, b), cx - 8:cx + 8] = (
                (60, 200, 90) if bar > 0 else (220, 70, 60))
            img[h // 2 - 1:h // 2 + 1, cx - 12:cx + 12] = (255, 255, 255)
            if g["n_contacts"] > 0:            # green dot = hands in contact
                img[12:26, 12:26] = (60, 200, 90)
            frames.append(img)
        k[0] += 1

    def hold(q, n):
        for _ in range(n):
            p.d.ctrl[p.env._act_ctrl_idx] = q
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(p.m, p.d)
            shoot()

    def ramp(q_to, n):
        start = p.q36()
        for i in range(n):
            a = (i + 1) / n; s = 3 * a ** 2 - 2 * a ** 3
            p.d.ctrl[p.env._act_ctrl_idx] = start + s * (q_to - start)
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(p.m, p.d)
            shoot()

    hold(q_app, 20)          # settle clear of the box
    ramp(q_in, 60)           # squeeze in
    hold(q_in, 30)
    up = np.array([0.0, 0.0, 1.0]); qq = q_in.copy()
    for i in range(1, 9):    # lift
        d = LIFT_H * i / 8
        qq[0:7] = p.ik_side_to("R", gR + d * up)
        qq[18:25] = p.ik_side_to("L", gL + d * up)
        ramp(qq, 26)
    hold(qq, 80)             # hold at the top -- this is where it is lost
    ren.close()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    zs = [t["z"] for t in trace]
    print(f"net {100*zs[-1]:+.1f} cm   max {100*max(zs):+.1f} cm -> {out}")
    lost = next((t["k"] for t in trace if t["k"] > 120 and t["n"] == 0), None)
    print(f"peak height {100*max(zs):.1f} cm at step {int(np.argmax(zs))};"
          f" contact lost at step {lost}")
    Path("results/best_trace.json").write_text(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
