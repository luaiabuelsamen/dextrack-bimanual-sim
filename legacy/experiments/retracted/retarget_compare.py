"""Does keypoint retargeting transfer a grasp, or only a posture?

One demonstration, several hands, four conditions, and a swept object width.
Each hand is asked to hold the same box its own opposition axis is centred on,
and the demonstration is scaled to that box, so nothing in the comparison is
fitted to make a hand look good.

The PALM IS FIXED, identically, for every condition -- at the hand's own natural
pose, with the object at the midpoint of its opposition axis. Only the fingers
differ. An earlier version let the palm float and placed it at the human's wrist
offset rescaled by finger length; that offset is a human anatomical fact, it
landed a couple of centimetres out, and a couple of centimetres is the
difference between every contact and none -- the comparison then measured my
wrist placement rather than the objective. Fixing the palm removes the confound:
whatever epsilon gains, it gains by choosing finger angles, not by moving the
hand somewhere the keypoint condition was not allowed to go.

Conditions:

    keypoint    match the human's inter-fingertip geometry
    +squeeze    then close the fingers until they grip -- what practitioners
                actually ship, and what Dexonomy stores as its third element
    epsilon     choose finger angles to maximise Ferrari-Canny epsilon
    blend       both terms at once

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

from handsim.grasping.retarget_pose import (retargeter_for, transform_ref, geometric_epsilon,
                             squeeze, KEYPOINT, EPSILON, BLEND)
from handsim.data import SyntheticSource

HANDS = ("shadow", "leap", "allegro", "f5d6")
# Order matters: each condition seeds the next, and EPSILON runs LAST so it
# starts from every pose already found. Run first, its search kept losing to
# BLEND -- which optimises epsilon with a competing term attached, so beating it
# is arithmetically impossible for a converged epsilon search. Losing was the
# signal that it was not converged, not that blending helps.
OBJECTIVES = (KEYPOINT, BLEND, EPSILON)


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
          f"(NOT MANO -- a stand-in; see handsim.data)\n")

    hdr = (f"{'hand':8s} {'w_cm':>5s} {'objective':10s} "
           f"{'kp_err_cm':>9s} {'eps':>7s} {'cts':>4s} {'s':>6s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for hk in a.hands:
        rt = retargeter_for(hk, free_base=False)
        for w in a.box_widths:
            for ri_, ref in enumerate(refs):
                V, obj, half, sc, demo, _wrist = transform_ref(rt, ref, width=w)
                e_ref, n_ref = geometric_epsilon(demo, half, obj_pos=obj)
                print(f"{hk:8s} {w*100:5.1f} {'[demo]':10s} {'--':>9s} "
                      f"{e_ref:7.4f} {n_ref:4d} {'--':>6s}", flush=True)
                rows.append(dict(hand=hk, ref=ref.label, objective="demo",
                                 width=w, scale=sc, keypoint_err_m=0.0,
                                 epsilon=e_ref, n_contacts=n_ref,
                                 obj_pos=obj.tolist(), obj_half=half.tolist()))
                seeds = []
                kp_q = None
                for o in OBJECTIVES:
                    t0 = time.time()
                    # KEYPOINT and BLEND are told where the demonstration put
                    # the palm -- that IS the pipeline being tested, and without
                    # it an origin-free finger-shape match floats anywhere.
                    # EPSILON is given no such hint; it may place the hand
                    # however it likes, which only makes it the stronger
                    # baseline to beat.
                    # KEYPOINT runs first and its solution seeds the others:
                    # epsilon refining a retargeted grasp is both the question
                    # worth asking and the only way its search reliably beats
                    # BLEND, which optimises the same quantity with an extra
                    # term attached.
                    r = rt.fit(V, half, objective=o, obj_pos=obj,
                               restarts=a.restarts, scale=sc,
                               seed_q=np.array(seeds) if seeds else None)
                    seeds.append(r.q)
                    if o == KEYPOINT:
                        kp_q = r.q
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
                    if o == KEYPOINT:
                        t0 = time.time()
                        qs, es, ns = squeeze(rt, r.q, half, obj)
                        seeds.append(qs)
                        dt = time.time() - t0
                        kerr = rt.keypoint_error(qs, V, sc)
                        print(f"{'':8s} {'':5s} {'+squeeze':10s} "
                              f"{kerr*100:9.2f} {es:7.4f} {ns:4d} {dt:6.1f}",
                              flush=True)
                        rows.append(dict(hand=hk, ref=ref.label,
                                         objective="keypoint+squeeze", width=w,
                                         scale=sc, keypoint_err_m=kerr,
                                         epsilon=es, n_contacts=ns,
                                         seconds=dt, q=qs.tolist()))
                        if not a.no_render and ri_ == 0 and abs(w - a.render_width) < 1e-9:
                            render(rt, qs, obj, half,
                                   out / f"{hk}_squeeze_{int(w*1000)}mm.png")
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out/'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
