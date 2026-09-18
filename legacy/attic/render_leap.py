"""Render LEAP closing on the block and holding it against gravity.

Nothing is underneath the block. The hand base is welded to the world, so the
only thing holding it up is the grasp.
"""
import sys, argparse
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from leap_bench import Hand
from analysis.epsilon import grasp_metrics


def main(half=(0.02, 0.02, 0.02), mass=0.02, ftgt=8.0, kp=5.0,
         out="legacy/figures/7_leap_grasp.gif", w=600, h=460, stride=4, fps=25):
    hd = Hand("leap", box_half=half, box_mass=mass, kp=kp)
    G0, _ = hd.converge_point(restarts=6)
    best = None
    for dx in (-0.01, 0.0, 0.01, 0.02):
        for dz in (-0.02, -0.01, 0.0):
            G = G0 + np.array([dx, 0, dz])
            q, _ = hd.close_pose(G, restarts=3)
            hd.reset(G); hd.pin_here(); hd.step(np.zeros(hd.n), 20, pin=True)
            qs = hd.seat(np.zeros(hd.n), q, ftgt)
            g = hd.metrics(); z0 = float(hd.d.qpos[hd.oq + 2])
            hd.step(qs, 1500)
            drop = z0 - float(hd.d.qpos[hd.oq + 2])
            ok = abs(drop) < 0.01 and hd.metrics()["n_contacts"] > 0
            if best is None or (ok, g["epsilon"]) > (best[0], best[1]):
                best = (ok, g["epsilon"], G, qs, drop)
    ok, eps, G, qs, drop = best
    print(f"chosen G={np.round(G,4)}  eps={eps:.4f}  held={ok}  drop={drop*100:.2f} cm")

    cam = mujoco.MjvCamera()
    cam.lookat[:] = G; cam.distance = 0.36
    cam.azimuth = 130; cam.elevation = -12
    ren = mujoco.Renderer(hd.m, height=h, width=w)
    frames = []
    q_close, _ = hd.close_pose(G, restarts=3)
    hd.reset(G); hd.pin_here()
    z0 = float(hd.d.qpos[hd.oq + 2])
    k = [0]

    def shoot(pinned):
        if k[0] % stride == 0:
            g = grasp_metrics(hd.m, hd.d, hd.ogid, hd.obid)
            ren.update_scene(hd.d, camera=cam)
            img = ren.render().copy()
            z = float(hd.d.qpos[hd.oq + 2]) - z0
            bar = int(np.clip(z / 0.10, -1, 1) * (h // 2 - 12))
            cx = w - 26
            img[:, cx - 10:cx + 10] = (40, 40, 40)
            a, b = (h // 2 - bar, h // 2) if bar > 0 else (h // 2, h // 2 - bar)
            img[max(0, a):min(h, b), cx - 8:cx + 8] = (
                (60, 200, 90) if bar >= 0 else (220, 70, 60))
            img[h // 2 - 1:h // 2 + 1, cx - 12:cx + 12] = (255, 255, 255)
            if g["n_contacts"] > 0:
                img[12:26, 12:26] = (60, 200, 90)
            if pinned:                      # blue = object still held in place
                img[12:26, 32:46] = (70, 130, 240)
            frames.append(img)
        k[0] += 1

    def run(q, n, pin):
        for _ in range(n):
            hd.d.ctrl[hd.aidx] = q
            mujoco.mj_step(hd.m, hd.d)
            if pin:
                hd.d.qpos[hd.oq:hd.oq + 7] = hd._pq
                hd.d.qvel[hd.ov:hd.ov + 6] = 0.0
            shoot(pin)

    run(np.zeros(hd.n), 60, True)                 # open, object pinned
    steps = 40
    for i in range(1, steps + 1):
        run(np.zeros(hd.n) + (i / steps) * (qs - np.zeros(hd.n)), 10, True)
    run(qs, 60, True)
    run(qs, 1500, False)                          # RELEASE: gravity only
    ren.close()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    zf = float(hd.d.qpos[hd.oq + 2]) - z0
    print(f"after release, 3 s: object moved {zf*100:+.2f} cm -> {out}")


if __name__ == "__main__":
    main()
