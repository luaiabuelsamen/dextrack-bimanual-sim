"""Render a bimanual GRAB sequence retargeted onto two robot hands.

Kinematic playback, and labelled as such: both hands are placed at the poses
the retarget produces and the object at the reference pose. It shows that the
bimanual reference and the bimanual retarget exist and agree; it makes no claim
that the two-handed grasp survives physics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image

from handsim.human import grab, scene, track
from handsim.human.retarget import retarget_sequence


def _place(sc, sd, palm_p, palm_q, q):
    """Put side `sd`'s palm at (palm_p, palm_q) with joint values `q`."""
    m, d = sc.model, sc.data
    fq = sc.free_q[sd]
    d.qpos[sc.qadr[sd]] = q
    d.qpos[fq:fq + 3] = 0.0
    d.qpos[fq + 3:fq + 7] = [1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)
    p0 = d.xpos[sc.palm_bid[sd]].copy()
    q0 = np.zeros(4)
    mujoco.mju_mat2Quat(q0, d.xmat[sc.palm_bid[sd]].copy())
    pi, qi = track._inv(p0, q0)
    bp, bq = track._mul(np.asarray(palm_p, float), np.asarray(palm_q, float), pi, qi)
    d.qpos[fq:fq + 3] = bp
    d.qpos[fq + 3:fq + 7] = bq
    d.mocap_pos[m.body_mocapid[sc.mocap_bid[sd]]] = bp
    d.mocap_quat[m.body_mocapid[sc.mocap_bid[sd]]] = bq


def render(row, hand_r="shadow", hand_l="shadow_left", stride=8,
           w=640, h=640, fps=12, out=None):
    s = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=stride)
    st, ln = row["both_hold_start"], row["both_hold_len"]

    fits, trs = {}, {}
    for sd, side, hk in (("r", "rhand", hand_r), ("l", "lhand", hand_l)):
        fits[sd] = scene.build(hk, s.obj, obj_static=True)
        trs[sd] = retarget_sequence(s, side, hk, window=(st, ln), sc=fits[sd])

    pos, quat = track.reference_in_first_frame(s, st)
    frames_idx = trs["r"].frames
    pos, quat = pos[frames_idx], quat[frames_idx]

    ff = {}
    for sd in ("r", "l"):
        P, Q, vals = track.feedforward_se3(fits[sd], trs[sd].q, pos, quat)
        ff[sd] = (P, Q, trs[sd].q)

    sim = scene.build_bimanual(hand_r, hand_l, s.obj, obj_static=True)
    m, d = sim.model, sim.data
    # Shadow's model carries a full forearm, which fills the frame and hides
    # the thing worth looking at. Faded, not deleted -- it is still simulated.
    for gi in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               int(m.geom_bodyid[gi])) or ""
        if "forearm" in bn:
            m.geom_rgba[gi] = [0, 0, 0, 0]
    for gi in (g for sd in ("r", "l") for g in []):
        pass

    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.62, 145.0, -20.0
    ren = mujoco.Renderer(m, height=h, width=w)

    # joint-name mapping from each fitting scene to the bimanual scene
    maps = {}
    for sd in ("r", "l"):
        fs = fits[sd]
        names_fit = [mujoco.mj_id2name(fs.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                     for j in fs.jids]
        names_sim = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                     for j in sim.jids[sd]]
        # The bimanual scene prefixes each side ("r_rh_FFJ3"); the fitting
        # scene, built as a lone hand, does not ("rh_FFJ3"). Stripping one
        # underscore-segment from BOTH matched nothing, every lookup returned
        # -1, and the hands rendered at their zero pose.
        pre = f"{sd}_"
        idx = {n: i for i, n in enumerate(names_fit)}
        maps[sd] = [idx.get(n[len(pre):] if n and n.startswith(pre) else n, -1)
                    for n in names_sim]
        miss = sum(1 for i in maps[sd] if i < 0)
        if miss:
            raise SystemExit(f"side {sd}: {miss}/{len(names_sim)} joints unmapped; "
                             f"sim={names_sim[:3]} fit={names_fit[:3]}")

    pics = []
    for k in range(len(pos)):
        for sd in ("r", "l"):
            P, Q, qfit = ff[sd]
            q = np.array([qfit[k][i] if i >= 0 else 0.0 for i in maps[sd]])
            _place(sim, sd, P[k], Q[k], q)
        mujoco.mj_forward(m, d)
        cam.lookat[:] = pos[k]
        ren.update_scene(d, camera=cam)
        pics.append(Image.fromarray(ren.render()))
    ren.close()

    out = Path(out or f"figures/bimanual_{s.name}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    pics[0].save(out, save_all=True, append_images=pics[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return out, s, trs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    inv = json.loads(Path("results/grab_inventory.json").read_text())["rows"]
    cand = sorted([r for r in inv if r.get("bimanual")],
                  key=lambda r: -r["both_hold_len"])
    row = next((r for r in cand if r["seq"] == a.seq), cand[0])
    out, s, trs = render(row)
    for sd in ("r", "l"):
        print(f"  {sd}: contact err {np.nanmean(trs[sd].contact_err)*1000:.1f} mm")
    print(f"{s.name} ({s.obj}, {s.intent}) -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)")
