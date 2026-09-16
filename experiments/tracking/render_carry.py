"""Validate and RENDER the carry-scored stage-2 poses before counting them.

`stage2_carry.py` scores a wrist offset on the last frames of a 25-frame carry.
Two things a count from that file cannot tell you, and this does:

1. Whether the pose carries the object through the REST of the reference, not
   just the scored window. The offset was optimised on those frames and may
   have overfitted to them, so every row here is rolled from the same start
   frame to the end of the reference and reported the way `stage2_survives.py`
   reports: fraction of frames still in hand, end state read live.

2. What it looks like. Reset, end of the scored window, end of the reference,
   two viewpoints each, object translucent so a hand inside it is visible, and
   any hand link deeper than 2 mm painted red. A pose reported as "0.5 mm,
   6 bodies, opposed" has previously been a hand beside the object, and the
   only defence is to look.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from oppdef.human import grab, track as T
from experiments.tracking.stage2_carry import contact_state, LOST_M

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size):
    try:
        return ImageFont.truetype(_FONT, size)
    except Exception:                                    # noqa: BLE001
        return ImageFont.load_default()


def buried_links(sc, depth=0.002):
    """Hand bodies with any contact deeper than `depth` against the object."""
    m, d = sc.model, sc.data
    objs, hands = set(sc.obj_gids), set(sc.hand_gids)
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ({g1, g2} & objs and {g1, g2} & hands):
            continue
        if -float(c.dist) > depth:
            out.add(int(m.geom_bodyid[g1 if g1 in hands else g2]))
    return out


def shot(rt, w, h, azim, dist, elev=-15.0):
    """Two viewpoints framed on the object. Object translucent, buried links red."""
    m, d = rt.sim.model, rt.sim.data
    mujoco.mj_forward(m, d)
    saved = m.geom_rgba.copy()
    red = buried_links(rt.sim)
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if b == rt.sim.obj_bid or name.startswith("obj_"):
            m.geom_rgba[g] = [0.98, 0.64, 0.24, 0.45]
        elif b in red:
            m.geom_rgba[g] = [0.90, 0.15, 0.15, 1.0]
        elif "forearm" in name:
            m.geom_rgba[g] = [0.78, 0.83, 0.90, 0.25]
        else:
            m.geom_rgba[g] = [0.78, 0.83, 0.90, 1.0]
    tiles = []
    try:
        for az in (azim, azim + 90.0):
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = dist, az, elev
            cam.lookat[:] = d.qpos[rt.obj_q:rt.obj_q + 3]
            with mujoco.Renderer(m, height=h, width=w) as r:
                r.update_scene(d, camera=cam)
                tiles.append(Image.fromarray(r.render()))
    finally:
        m.geom_rgba[:] = saved
    im = Image.new("RGB", (2 * w, h))
    im.paste(tiles[0], (0, 0)); im.paste(tiles[1], (w, 0))
    return im


def panel(im, title, sub):
    out = Image.new("RGB", (im.width, im.height + 46), "#101722")
    out.paste(im, (0, 46))
    dr = ImageDraw.Draw(out)
    dr.text((10, 6), title, font=_font(17), fill="#f2f4f8")
    dr.text((10, 27), sub, font=_font(12), fill="#98adbf")
    return out


def caption(rt, k, err_mm=None):
    pen, n, nb, sided, grip = contact_state(rt.sim)
    s = (f"{pen*1000:.2f} mm pen | {n} contacts on {nb} links | "
         f"one-sided {sided:.2f} | {grip:.0f} N")
    if err_mm is not None:
        s += f" | err {err_mm:.1f} mm"
    return s


def validate(row, a):
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = T.ReferenceTracker(seq, hand="shadow")
    rt.apply_wrist_offset(np.asarray(row["offset"], float))
    k0, n_sc = int(row["start"]), int(row["frames"])
    span = rt.T - k0
    m, d = rt.sim.model, rt.sim.data
    rt.reset_at(k0)
    mujoco.mj_forward(m, d)
    panels = [panel(shot(rt, a.w, a.h, a.azim, a.dist),
                    f"{row['seq']}   reset (frame {k0})", caption(rt, k0))]
    errs = np.empty(span)
    for i in range(span):
        rt.apply(k0 + i, None)
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(m, d)
        errs[i], _ = rt.error(k0 + i)
        if i + 1 == n_sc and n_sc < span:
            panels.append(panel(shot(rt, a.w, a.h, a.azim, a.dist),
                                f"{row['seq']}   end of scored carry (+{n_sc})",
                                caption(rt, k0 + i, errs[i] * 1000)))
    panels.append(panel(shot(rt, a.w, a.h, a.azim, a.dist),
                        f"{row['seq']}   end of reference (+{span})",
                        caption(rt, rt.T - 1, errs[-1] * 1000)))
    pen, n, nb, sided, grip = contact_state(rt.sim)
    ok = np.flatnonzero(errs <= LOST_M)
    held_to = int(ok[-1] + 1) if len(ok) else 0
    sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y)); y += p.height
    out = Path(a.outdir) / f"{row['seq']}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return {
        "seq": row["seq"], "subject": row["subject"], "object": row["object"],
        "start": k0, "scored_frames": n_sc, "span": int(span),
        "carried_frames": held_to, "carried_frac": float(held_to / max(span, 1)),
        "mean_mm": float(errs.mean() * 1000),
        "ref_end_pen_mm": pen * 1000, "ref_end_contacts": n, "ref_end_bodies": nb,
        "ref_end_sided": sided, "ref_end_grip_n": grip,
        "ref_end_held": bool(errs[-1] <= LOST_M), "figure": str(out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry", default="results/stage2_carry.json")
    ap.add_argument("--out", default="results/stage2_carry_validate.json")
    ap.add_argument("--outdir", default="figures/carry")
    ap.add_argument("--seqs", default="", help="subset; default: every CLEAN row")
    ap.add_argument("--all", action="store_true", help="every row, not only clean")
    ap.add_argument("--w", type=int, default=400)
    ap.add_argument("--h", type=int, default=340)
    ap.add_argument("--azim", type=float, default=135.0)
    ap.add_argument("--dist", type=float, default=0.32)
    a = ap.parse_args()

    rows = json.loads(Path(a.carry).read_text())["rows"]
    if a.seqs:
        want = set(x.strip() for x in a.seqs.split(","))
        rows = [r for r in rows if r["seq"] in want]
    elif not a.all:
        rows = [r for r in rows if r["end"]["clean"]]
    out = []
    for r in rows:
        try:
            v = validate(r, a)
        except Exception as e:                            # noqa: BLE001
            print(f"  {r['seq']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        out.append(v)
        print(f"{v['seq']:24s} carried {v['carried_frames']:3d}/{v['span']:3d} "
              f"({v['carried_frac']*100:3.0f}%)  mean {v['mean_mm']:7.1f} mm  "
              f"ref end {v['ref_end_pen_mm']:5.2f} mm, {v['ref_end_contacts']:2d} con "
              f"on {v['ref_end_bodies']} links, sided {v['ref_end_sided']:.2f}, "
              f"{v['ref_end_grip_n']:6.0f} N, held={int(v['ref_end_held'])}  "
              f"-> {v['figure']}", flush=True)
        Path(a.out).write_text(json.dumps({"carry": a.carry, "rows": out}, indent=1))


if __name__ == "__main__":
    main()
