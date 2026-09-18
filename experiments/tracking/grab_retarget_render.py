"""Render a retargeted GRAB grasp in MuJoCo: robot hand on the real object.

The residuals say the fingertips are within ~13 mm of the human's contact
points and the rest of the hand is ~4 mm inside the object. Neither number
says whether the hand is holding the mug or passing through the handle at
right angles, which is what a picture says in one glance.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image

from handsim.human import grab, scene
from handsim.human.retarget import retarget_sequence


def shots(seq_name, hand="shadow", window=None, n=5, w=560, h=560,
          azim=140.0, elev=-14.0, out=None, stride=8, w_pen=2.0):
    s = grab.load(seq_name, verts=True, stride=stride)
    sc = scene.build(hand, s.obj)
    if window is None:                      # longest run of hand-object contact
        from experiments.tracking.grab_inventory import contact_mask, longest_run, _tree
        m = contact_mask(s, _tree(s, {}), "rhand")
        window = longest_run(m)
    tr = retarget_sequence(s, "rhand", hand, window=window, sc=sc, w_pen=w_pen)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance = 0.30
    cam.azimuth, cam.elevation = azim, elev

    ren = mujoco.Renderer(sc.model, height=h, width=w)
    picks = np.linspace(0, len(tr.frames) - 1, n).astype(int)
    tiles = []
    for t in picks:
        sc.set_q(tr.q[t])
        ren.update_scene(sc.data, camera=cam)
        tiles.append(Image.fromarray(ren.render()))
    ren.close()

    sheet = Image.new("RGB", (w * len(tiles), h), "white")
    for i, im in enumerate(tiles):
        sheet.paste(im, (i * w, 0))
    out = Path(out or f"figures/retarget_{hand}_{s.name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out, s, tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="s1/mug_drink_1.npz")
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--azim", type=float, default=140.0)
    ap.add_argument("--elev", type=float, default=-14.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out, s, tr = shots(a.seq, a.hand, azim=a.azim, elev=a.elev, out=a.out)
    print(f"{s.name} ({s.obj}, {s.intent}): {len(tr.frames)} frames, "
          f"contact err {np.nanmean(tr.contact_err)*1000:.1f} mm")
    print(f"wrote {out}")
