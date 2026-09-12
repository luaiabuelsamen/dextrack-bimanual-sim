"""Render the sustained lift (config B) and the pose-retargeted failure (A)."""
import sys
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dextrack_vega import config as C
from analysis.epsilon import grasp_metrics
from probe_m15 import human_grasp, retarget_keypoints
from probe_v2 import build_probe, BOX, eval_config_v2
from squeeze_sweep import _approach

INSET, SQ, LIFT_H, GAIN = 0.015, 0.85, 0.15, 0.05


class Rec:
    def __init__(self, p, w=600, h=460, stride=3):
        self.p, self.w, self.h, self.stride, self.k = p, w, h, stride, 0
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.55, 0.0, 0.84]; self.cam.distance = 1.0
        self.cam.azimuth = 35; self.cam.elevation = -12
        self.ren = mujoco.Renderer(p.m, height=h, width=w)
        self.frames = []
        self.z0 = float(p.obj_pos()[2])

    def shoot(self):
        p, h, w = self.p, self.h, self.w
        if self.k % self.stride == 0:
            z = float(p.obj_pos()[2]) - self.z0
            g = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid,
                              (p.table_gid, p.floor_gid))
            self.ren.update_scene(p.d, camera=self.cam)
            img = self.ren.render().copy()
            bar = int(np.clip(z / 0.20, -1, 1) * (h // 2 - 12))
            cx = w - 26
            img[:, cx - 10:cx + 10] = (40, 40, 40)
            a, b = (h // 2 - bar, h // 2) if bar > 0 else (h // 2, h // 2 - bar)
            img[max(0, a):min(h, b), cx - 8:cx + 8] = (
                (60, 200, 90) if bar > 0 else (220, 70, 60))
            img[h // 2 - 1:h // 2 + 1, cx - 12:cx + 12] = (255, 255, 255)
            if g["n_contacts"] > 0:
                img[12:26, 12:26] = (60, 200, 90)
            self.frames.append(img)
        self.k += 1

    def hold(self, q, n):
        for _ in range(n):
            self.p.d.ctrl[self.p.env._act_ctrl_idx] = q
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(self.p.m, self.p.d)
            self.shoot()

    def ramp(self, q_to, n):
        start = self.p.q36()
        for i in range(n):
            a = (i + 1) / n; s = 3 * a ** 2 - 2 * a ** 3
            self.p.d.ctrl[self.p.env._act_ctrl_idx] = start + s * (q_to - start)
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(self.p.m, self.p.d)
            self.shoot()

    def save(self, out, fps=25):
        self.ren.close()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, self.frames, fps=fps, loop=0)
        return float(self.p.obj_pos()[2]) - self.z0


def render_B(p):
    centre = np.array(BOX["obj_pos"], float)
    hy = float(p.m.geom_size[p.obj_gid][1])
    gR = centre + np.array([0.0, -(hy - INSET), 0.0])
    gL = centre + np.array([0.0, +(hy - INSET), 0.0])
    q_app, q_in, _ = _approach(p, gR, gL, SQ)
    p.reset(); p.set_q36(q_app)
    r = Rec(p)
    r.hold(q_app, 20); r.ramp(q_in, 60); r.hold(q_in, 30)
    up = np.array([0.0, 0.0, 1.0]); qq = q_in.copy()
    for i in range(1, 11):
        d = LIFT_H * i / 10
        ins = INSET + GAIN * d
        qq[0:7] = p.ik_side_to("R", centre + np.array([0, -(hy - ins), 0]) + d * up)
        qq[18:25] = p.ik_side_to("L", centre + np.array([0, +(hy - ins), 0]) + d * up)
        r.ramp(qq, 26)
    r.hold(qq, 80)
    net = r.save("figures/5_B_eps_bimanual_lift.gif")
    print(f"B net {net*100:+.1f} cm -> figures/5_B_eps_bimanual_lift.gif")


def render_A(p):
    p.reset()
    centre = p.obj_pos(); half = np.array(p.m.geom_size[p.obj_gid][:3], float)
    hw, hf = human_grasp(centre, half)
    qA, _ = retarget_keypoints(p, "R", hw, hf, seed=0)
    _, qA_fin = eval_config_v2(p, qA, ["R"], f_target=2.0)
    # replay the same approach for the camera
    p.reset()
    r = Rec(p)
    q_open = qA_fin.copy()
    for i, j in enumerate(p.ctrl_joints[7:18]):
        q_open[7 + i] = 0.0
    p.set_q36(q_open); r.z0 = float(p.obj_pos()[2])
    r.hold(q_open, 20); r.ramp(qA_fin, 80); r.hold(qA_fin, 40)
    up = np.array([0.0, 0.0, 1.0]); qq = qA_fin.copy()
    g0 = p.ft("R").mean(0)
    for i in range(1, 11):
        qq[0:7] = p.ik_side_to("R", g0 + (LIFT_H * i / 10) * up)
        r.ramp(qq, 26)
    r.hold(qq, 60)
    net = r.save("figures/6_A_pose_retarget_fail.gif")
    print(f"A net {net*100:+.1f} cm -> figures/6_A_pose_retarget_fail.gif")


if __name__ == "__main__":
    p = build_probe(seed=0, **BOX)
    render_B(p)
    render_A(p)
