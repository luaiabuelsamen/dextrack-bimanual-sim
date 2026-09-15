"""Render the whole pipeline on one clip: human clip -> robot hand carrying it.

Kinematic playback, and labelled as such. The object is placed at the reference
pose and the hand at the pose the SE(3) feedforward produces, frame by frame.
No physics is stepped, so this shows what the RETARGET produces and makes no
claim that the grasp survives -- that is what G5 and the tracking stage measure.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image

from oppdef.human import grab, scene, track
from oppdef.human.retarget import retarget_sequence
from experiments.grab_inventory import contact_mask, longest_run, _tree


def render(seq_name, hand="shadow", stride=8, w=620, h=620, fps=12,
           out=None, follow=True):
    s = grab.load(seq_name, verts=True, stride=stride)
    mk = contact_mask(s, _tree(s, {}), "rhand")
    st, ln = longest_run(mk)
    if ln < 5:
        raise SystemExit(f"{s.name}: no right-hand hold window")

    fit = scene.build(hand, s.obj, obj_static=True)
    tr = retarget_sequence(s, "rhand", hand, window=(st, ln), sc=fit)

    pos, quat = track.reference_in_first_frame(s, st)
    pos, quat = pos[tr.frames], quat[tr.frames]
    P, Q, vals = track.feedforward_se3(fit, tr.q, pos, quat)

    sim = scene.build(hand, s.obj, base="mocap", obj_static=False)
    mh = track.MocapHand(sim)
    m, d = sim.model, sim.data
    oq = int(m.jnt_qposadr[[j for j in range(m.njnt)
                            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                            and int(m.jnt_bodyid[j]) == sim.obj_bid][0]])

    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.42, 135.0, -12.0
    ren = mujoco.Renderer(m, height=h, width=w)

    frames = []
    for k in range(len(P)):
        mh.place(P[k], Q[k], vals[k])
        d.qpos[oq:oq + 3] = pos[k]
        d.qpos[oq + 3:oq + 7] = quat[k]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = pos[k] if follow else pos[0]
        ren.update_scene(d, camera=cam)
        frames.append(Image.fromarray(ren.render()))
    ren.close()

    out = Path(out or f"figures/track_{hand}_{s.name}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return out, s, tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="s1/mug_drink_1.npz")
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out, s, tr = render(a.seq, a.hand, stride=a.stride, out=a.out)
    print(f"{s.name} ({s.obj}, {s.intent}): {len(tr.frames)} frames, "
          f"contact err {np.nanmean(tr.contact_err)*1000:.1f} mm")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
