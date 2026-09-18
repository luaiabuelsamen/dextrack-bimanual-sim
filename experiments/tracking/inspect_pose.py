"""Render a hand/object configuration so it can be LOOKED AT.

Scalar contact metrics hide the thing that matters. A pose reported as "0.54 mm
penetration, 1 body in contact" reads like a near-miss grasp; rendered, it was a
hand sitting BESIDE the cup with its fingers closing on empty air. That
distinction cost several hours of sweeps to not notice and one picture to see.

Use this before believing any contact metric:

    from experiments.tracking.inspect_pose import look, contacts
    rt.reset_at(0)
    look(rt.sim, "baseline").save("/tmp/x.jpg")
    print(contacts(rt.sim))

`look` returns a two-viewpoint PIL image with a caption; `contacts` returns
(max penetration mm, n distinct hand bodies, one-sidedness, body names).
One-sidedness is |mean contact normal|: near 0 means opposed contacts on
different sides -- a wrap -- and 1.0 means every contact pushes the same way, or
there are none.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image, ImageDraw, ImageFont

_FONTS = Path("/usr/share/fonts/truetype/dejavu")


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(_FONTS / name), size)


def contacts(sc):
    """(max penetration mm, n hand bodies, one-sidedness, sorted body names)."""
    m, d = sc.model, sc.data
    mujoco.mj_forward(m, d)
    obj = set(sc.obj_gids)
    pen, bodies, normals = 0.0, set(), []
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if (g1 in obj) == (g2 in obj):
            continue
        hand_g = g2 if g1 in obj else g1
        bodies.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                     int(m.geom_bodyid[hand_g])) or "?")
        n = np.array(d.contact[i].frame[:3])
        normals.append(-n if g1 in obj else n)
        pen = max(pen, -float(d.contact[i].dist))
    one = float(np.linalg.norm(np.mean(normals, axis=0))) if normals else 1.0
    return pen * 1000, len(bodies), one, sorted(b.replace("hand_rh_", "") for b in bodies)


def look(sc, title="", subtitle=None, width=460, height=380,
         azimuth=135.0, elevation=-20.0, distance=0.42):
    """Two viewpoints of the CURRENT state, captioned. Returns a PIL image.

    Pass `subtitle=None` to have the contact summary filled in automatically --
    that pairing is the point, since the numbers are only trustworthy next to
    the picture.
    """
    m, d = sc.model, sc.data
    mujoco.mj_forward(m, d)
    if subtitle is None:
        pen, nb, one, names = contacts(sc)
        subtitle = (f"penetration {pen:.2f} mm | {nb} bodies | one-sided {one:.3f} | "
                    + (", ".join(names[:6]) if names else "NOTHING TOUCHING"))

    obj_b = [b for b in range(m.nbody)
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("obj_")]
    lookat = d.xipos[obj_b[0]].copy() if obj_b else np.zeros(3)

    saved = m.geom_rgba.copy()
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if body.startswith("obj_"):
            m.geom_rgba[g] = [0.98, 0.64, 0.24, 1.0]
        elif "forearm" in body:
            m.geom_rgba[g, 3] = 0.25          # visible but not occluding
        else:
            m.geom_rgba[g] = [0.78, 0.83, 0.90, 1.0]
    try:
        tiles = []
        for az in (azimuth, azimuth + 90.0):
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = distance, az, elevation
            cam.lookat[:] = lookat
            with mujoco.Renderer(m, height=height, width=width) as r:
                r.update_scene(d, camera=cam)
                tiles.append(Image.fromarray(r.render()))
    finally:
        m.geom_rgba[:] = saved

    out = Image.new("RGB", (width * 2, height + 52), "#101722")
    out.paste(tiles[0], (0, 52))
    out.paste(tiles[1], (width, 52))
    draw = ImageDraw.Draw(out)
    draw.text((14, 10), title, font=_font(19, True), fill="#f2f4f8")
    draw.text((14, 33), subtitle, font=_font(12), fill="#98adbf")
    return out


def sheet(images, path):
    """Stack captioned panels into one image and save."""
    images = list(images)
    out = Image.new("RGB", (images[0].width, sum(i.height for i in images)), "#101722")
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=92)
    return path


if __name__ == "__main__":
    import sys
    from handsim.human import grab, track

    seq_name = sys.argv[1] if len(sys.argv) > 1 else "cup_lift"
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seq = grab.load(f"s1/{seq_name}.npz", verts=True, stride=8)
    rt = track.ReferenceTracker(seq)
    rt.reset_at(frame)
    out = f"out/pose_{seq_name}_{frame}.jpg"
    sheet([look(rt.sim, f"{seq_name}  frame {frame}")], out)
    print(f"wrote {out}")
    print("contacts:", contacts(rt.sim))
