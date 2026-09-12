"""Does keypoint retargeting transfer a grasp, or only a posture?

One demonstration, several hands, three objectives, and a swept object width.
Each hand is asked to hold the same box its own opposition axis is centred on,
and the demonstration is scaled to that box, so nothing in the comparison is
fitted to make a hand look good.

Every row reports the DEMONSTRATION's own epsilon next to the fits, so a hand
that scores badly can be told apart from a reference that was never a grasp --
the first version of this script could not, and reported zeros for everything.

    python experiments/retarget_compare.py --out out/retarget

Writes a JSON record and one render per (hand, objective) so the poses get
looked at rather than believed. Renders need MUJOCO_GL=egl on this machine.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import mujoco

from oppdef.retarget import (retargeter_for, transform_ref, geometric_epsilon,
                             KEYPOINT, EPSILON, BLEND)
from oppdef.data import SyntheticSource

HANDS = ("shadow", "leap", "allegro", "f5d6")
OBJECTIVES = (KEYPOINT, EPSILON, BLEND)


def render(rt, q, obj_pos, obj_half, path, tips=None):
    """Render the fitted pose with the object drawn in, as a sanity image.

    The object is added to the SPEC and recompiled rather than drawn as an
    overlay, so what is rendered is the geometry the epsilon was computed
    against -- an overlay can agree with a number that is wrong.
    """
    import mujoco as mj
    if rt.spec_fn is None:
        print("    (no spec_fn: cannot render)")
        return None
    spec = rt.spec_fn()
    b = spec.worldbody.add_body(name="probe_obj", pos=[float(x) for x in obj_pos])
    b.add_geom(type=mj.mjtGeom.mjGEOM_BOX,
               size=[float(x) for x in obj_half], rgba=[0.85, 0.3, 0.2, 0.55])
    m = spec.compile()
    d = mj.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[rt.qadr] = q
    mj.mj_forward(m, d)
    cam = mj.MjvCamera()
    mj.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = obj_pos
    cam.distance = 0.45
    cam.azimuth, cam.elevation = 135, -20
    try:
        with mj.Renderer(m, 480, 640) as r:
            r.update_scene(d, cam)
            px = r.render()
    except Exception as e:
        print(f"    (render unavailable: {type(e).__name__}: {e})")
        return None
    try:
        from PIL import Image
        Image.fromarray(px).save(path)
    except Exception:
        import imageio.v2 as imageio
        imageio.imwrite(path, px)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/retarget")
    ap.add_argument("--widths", type=float, nargs="+", default=[0.05])
    ap.add_argument("--n-per", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--hands", nargs="+", default=list(HANDS))
    ap.add_argument("--box-widths", type=float, nargs="+",
                    default=[0.03, 0.05, 0.07, 0.09])
    ap.add_argument("--render-width", type=float, default=0.05)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = list(SyntheticSource(widths=tuple(a.widths), n_per=a.n_per))
    print(f"{len(refs)} demonstrations from SyntheticSource "
          f"(NOT MANO -- a stand-in; see oppdef.data)\n")

    hdr = (f"{'hand':8s} {'w_cm':>5s} {'objective':10s} "
           f"{'kp_err_cm':>9s} {'eps':>7s} {'cts':>4s} {'s':>6s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for hk in a.hands:
        rt = retargeter_for(hk)
        for w in a.box_widths:
            for ri_, ref in enumerate(refs):
                V, obj, half, sc, demo, wrist = transform_ref(rt, ref, width=w)
                e_ref, n_ref = geometric_epsilon(demo, half, obj_pos=obj)
                print(f"{hk:8s} {w*100:5.1f} {'[demo]':10s} {'--':>9s} "
                      f"{e_ref:7.4f} {n_ref:4d} {'--':>6s}", flush=True)
                rows.append(dict(hand=hk, ref=ref.label, objective="demo",
                                 width=w, scale=sc, keypoint_err_m=0.0,
                                 epsilon=e_ref, n_contacts=n_ref,
                                 obj_pos=obj.tolist(), obj_half=half.tolist()))
                for o in OBJECTIVES:
                    t0 = time.time()
                    # KEYPOINT and BLEND are told where the demonstration put
                    # the palm -- that IS the pipeline being tested, and without
                    # it an origin-free finger-shape match floats anywhere.
                    # EPSILON is given no such hint; it may place the hand
                    # however it likes, which only makes it the stronger
                    # baseline to beat.
                    r = rt.fit(V, half, objective=o, obj_pos=obj,
                               restarts=a.restarts, scale=sc,
                               wrist_target=None if o == EPSILON else wrist)
                    dt = time.time() - t0
                    print(f"{'':8s} {'':5s} {o:10s} {r.keypoint_err_m*100:9.2f} "
                          f"{r.epsilon:7.4f} {r.n_contacts:4d} {dt:6.1f}",
                          flush=True)
                    rows.append(dict(hand=hk, ref=ref.label, objective=o,
                                     width=w, scale=sc,
                                     keypoint_err_m=r.keypoint_err_m,
                                     epsilon=r.epsilon,
                                     n_contacts=r.n_contacts, seconds=dt,
                                     q=r.q.tolist()))
                    if not a.no_render and ri_ == 0 and abs(w - a.render_width) < 1e-9:
                        p = render(rt, r.q, obj, half,
                                   out / f"{hk}_{o}_{int(w*1000)}mm.png")
                        if p:
                            rows[-1]["render"] = p
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out/'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
