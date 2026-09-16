"""Render the burial-to-grasp relaxation: reset, quarter, end.

The carriers all start buried and resolve into a contact set under motion. The
numbers say so -- stamp_lift falls from 14.23 mm and 878 N to 0.31 mm and 3.6 N
-- but a reader cannot see a relaxation in a table, and a peer session found the
mechanism only by rendering rather than trusting the endpoints.

Three grasp-class carriers, chosen for contrast: stamp_lift relaxes all the way
to a clean five-finger pinch, hammer_lift and banana_eat_1 relax by an order of
magnitude and stop short, ending 4-5 mm inside.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image

from oppdef.human import grab, track as T


def shot(rt, w, h, cam):
    """Framed on the OBJECT, following it. The default free camera puts the
    hand in a corner and, on a wide object, fills the frame with its face --
    a picture of nothing, which is worse than no picture."""
    mujoco.mj_forward(rt.sim.model, rt.sim.data)
    cam.lookat[:] = rt.sim.data.qpos[rt.obj_q:rt.obj_q + 3]
    with mujoco.Renderer(rt.sim.model, height=h, width=w) as r:
        r.update_scene(rt.sim.data, camera=cam)
        return r.render()


def label(img, text):
    from PIL import ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="stamp_lift,hammer_lift,banana_eat_1")
    ap.add_argument("--grips", default="results/stage2_grips_clean.json")
    ap.add_argument("--out", default="figures/burial_relaxation.png")
    ap.add_argument("--w", type=int, default=420)
    ap.add_argument("--h", type=int, default=420)
    ap.add_argument("--dist", type=float, default=0.30)
    ap.add_argument("--azim", type=float, default=135.0)
    a = ap.parse_args()

    # keyed on SUBJECT/SEQ, never the bare name (tests/test_seed_keys.py)
    grips = {f"{x['subject']}/{x['seq']}": x
             for x in json.loads(Path(a.grips).read_text())["rows"]}
    rows = []
    for name in a.seqs.split(","):
        hits = [k for k in grips if k == name or k.split("/", 1)[1] == name]
        if len(hits) != 1:
            raise SystemExit(f"{name!r} matches {len(hits)} grip rows "
                             f"({hits}); name it as SUBJECT/SEQ")
        g = grips[hits[0]]
        name = g["seq"]
        seq = grab.load(f"{g['subject']}/{name}.npz", verts=True, stride=8)
        rt = T.ReferenceTracker(seq, hand="shadow")
        rt.apply_wrist_offset(np.asarray(g["offset"], float))
        gf = rt.grasp_frames()
        if len(gf) == 0:
            print(f"{name}: no graspable frame"); continue
        k0 = int(gf[0]); span = rt.T - k0
        rt.reset_at(k0)

        def state():
            pen, _ = rt.sim.penetration()
            grip, n = T.total_grip(rt.sim)
            return pen * 1000, n, grip

        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = a.dist, a.azim, -12.0
        frames, marks = [], {0: "reset", int(span * 0.25): "25%"}
        for i in range(span):
            if i in marks:
                p, n, gr = state()
                frames.append(label(shot(rt, a.w, a.h, cam),
                                    f"{name}  {marks[i]}   "
                                    f"{p:.2f} mm  {n} con  {gr:.0f} N"))
            rt.apply(k0 + i, None)
            for _ in range(rt.ctrl_every):
                mujoco.mj_step(rt.sim.model, rt.sim.data)
        # AFTER the last step, so the end frame is the state the carry ended in
        p, n, gr = state()
        frames.append(label(shot(rt, a.w, a.h, cam),
                            f"{name}  end   {p:.2f} mm  {n} con  {gr:.0f} N"))
        rows.append(frames)
        print(f"{name}: {len(rows[-1])} frames", flush=True)

    if not rows:
        return
    cols = max(len(r) for r in rows)
    sheet = Image.new("RGB", (cols * a.w, len(rows) * a.h), (255, 255, 255))
    for j, r in enumerate(rows):
        for i, im in enumerate(r):
            sheet.paste(im, (i * a.w, j * a.h))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
