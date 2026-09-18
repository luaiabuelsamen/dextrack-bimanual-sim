"""Render each live task, replaying grasps from their saved parameters.

Nothing here re-searches. Every grasp shown is read back out of a results file
by its saved placement and finger targets -- the provenance added after the
completion review -- so the figure and the number come from the same pose.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
import mujoco

from handsim.bench import Cell
from handsim.grasping.hold import DIRECTIONS

W, H = 460, 345   # README-sized; keeps the GIFs to ~1 MB


def _renderer(model):
    return mujoco.Renderer(model, H, W)


def _cam(model, lookat, dist=0.24, az=120, el=-22):
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, c)
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def save_gif(frames, path, fps=18):
    """Quantised to a shared palette -- a README GIF has to load, not just exist."""
    import imageio.v2 as imageio
    from PIL import Image
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f).quantize(colors=64, method=Image.MEDIANCUT)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  wrote {path}  ({len(frames)} frames)")


def render_grasp(row, out, probe_dirs=4, every=11):
    """Close on the object, release it, then push and twist it."""
    cell = Cell(row["hand"], row["width"], mass=row["mass"], kp=row["kp_finger"])
    sc = cell.scene
    frames = []
    orig_step = mujoco.mj_step

    def rec(m, d, n=1):
        orig_step(m, d, n)
        if rec.k % every == 0:
            r.update_scene(d, cam)
            frames.append(r.render())
        rec.k += 1
    rec.k = 0

    with _renderer(sc.m) as r:
        cam = _cam(sc.m, [0, 0, 0])
        mujoco.mj_step = rec
        try:
            sc.attempt(np.array(row["params"]), do_hold=False,
                       finger_target=row["finger_target"])
        finally:
            mujoco.mj_step = orig_step
        # the probe: push the object in a few directions and let it fail or hold
        snap = (sc.d.qpos.copy(), sc.d.qvel.copy(), sc.d.ctrl.copy())
        f = max(float(row["hold_N"]), 0.25) * 1.4
        for u in DIRECTIONS[:probe_dirs]:
            sc.d.qpos[:], sc.d.qvel[:] = snap[0].copy(), snap[1].copy()
            sc.d.ctrl[:] = snap[2]
            mujoco.mj_forward(sc.m, sc.d)
            for k in range(180):
                sc.d.xfrc_applied[sc.obj_bid, :3] = u * f
                orig_step(sc.m, sc.d)
                if k % every == 0:
                    r.update_scene(sc.d, cam)
                    frames.append(r.render())
            sc.d.xfrc_applied[sc.obj_bid, :] = 0.0
    save_gif(frames, out)


def pick(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            yield r


def render_bimanual(two_handed, out, every=22):
    """The peg-extraction task: the one bimanual result that survives.

    Socket friction (1.20 N) exceeds the base's weight (0.78 N), so a one-handed
    pull lifts the base instead of extracting anything. Nothing about this task
    depends on the retracted opposition axis -- it is a statement about the
    task's own force balance.
    """
    from handsim.control.expert import Expert
    ex = Expert(two_handed=two_handed)
    e = ex.e
    frames = []
    orig_step = mujoco.mj_step

    def rec(m, d, n=1):
        orig_step(m, d, n)
        if rec.k % every == 0:
            r.update_scene(d, cam)
            frames.append(r.render())
        rec.k += 1
    rec.k = 0

    with mujoco.Renderer(e.m, H, W) as r:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(e.m, cam)
        cam.lookat[:] = [0.0, 0.0, 0.10]
        cam.distance, cam.azimuth, cam.elevation = 0.85, 150, -14
        mujoco.mj_step = rec
        try:
            res = ex.run(verbose=False)
        finally:
            mujoco.mj_step = orig_step
    save_gif(frames, out, fps=30)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="allegro")
    ap.add_argument("--width", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("legacy/results/g1_*.json")):
        rows += json.loads(pathlib.Path(f).read_text())["rows"]
    for arm, name in (("wrench_refine", "wrench"), ("pose_squeeze", "pose")):
        cand = [r for r in pick(rows, hand=a.hand, arm=arm, seed=a.seed)
                if abs(r["width"] - a.width) < 1e-9 and r["params"]]
        if not cand:
            print(f"  no saved row for {a.hand} {arm} @ {a.width}")
            continue
        r = cand[0]
        print(f"{arm}: eps {r['epsilon']:.3f}  hold {r['hold_N']:.3f} N  "
              f"contacts {r['n_contacts']}")
        render_grasp(r, f"legacy/figures/task_grasp_{name}.gif")

    print("\nbimanual peg extraction:")
    for two, nm in ((True, "two_handed"), (False, "one_handed")):
        res = render_bimanual(two, f"legacy/figures/task_peg_{nm}.gif")
        print(f"  {nm}: peg out {res['peg_out_m']*100:.2f} cm  "
              f"base lift {res.get('box_z_rise_m', 0)*100:+.2f} cm  "
              f"success {res['success']}")


if __name__ == "__main__":
    main()
