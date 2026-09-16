"""GIFs of the carry-scored seeds: feedforward and PPO side by side.

Every frame is read LIVE after that control frame's last physics step and
carries its own numbers on its face -- penetration, contacts and links, grip
force, tracking error -- so the picture and the metric cannot disagree
silently. Object translucent, any hand link deeper than 2 mm painted red: a
burial is visible as red inside the object, a grasp as white on its surface.
A per-frame JSON manifest is written beside each GIF.

Left column: the retargeted trajectory with no controller. Right column: the
PPO policy trained from this start (`stage3_carry.py`), if its checkpoint
exists. Both start from the identical state.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import torch
from PIL import Image, ImageDraw, ImageFont

from oppdef.human import grab, track as T, rl
from experiments.tracking.stage2_carry import contact_state
from experiments.tracking.render_carry import buried_links

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size, bold=False):
    try:
        return ImageFont.truetype(_BOLD if bold else _FONT, size)
    except Exception:                                    # noqa: BLE001
        return ImageFont.load_default()


def paint(rt):
    m = rt.sim.model
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


def frame(rt, renderer, cam, azim, dist, lookat):
    m, d = rt.sim.model, rt.sim.data
    mujoco.mj_forward(m, d)
    saved = m.geom_rgba.copy()
    paint(rt)
    try:
        cam.distance, cam.azimuth, cam.elevation = dist, azim, -15.0
        cam.lookat[:] = lookat
        renderer.update_scene(d, camera=cam)
        return Image.fromarray(renderer.render())
    finally:
        m.geom_rgba[:] = saved


def rollouts(rt, k0, net):
    """Yield per-frame (label, state-capture) for FF and, if net, PPO. States
    are captured as (qpos, qvel, mocap_pos, mocap_quat, ctrl) so both columns
    render from the same MjData afterwards."""
    cfg = rl.RLConfig()
    scale = np.concatenate([np.full(3, cfg.a_pos), np.full(3, cfg.a_rot),
                            np.full(rt.n_action - 6, cfg.a_fin)])
    out = {}
    for name in ("feedforward", "ppo") if net is not None else ("feedforward",):
        rt.reset_at(k0)
        states, rows = [rt.save()], [(0.0,) + contact_state(rt.sim)]
        for k in range(k0, rt.T):
            if name == "ppo":
                x = torch.as_tensor(rt.observe(k), dtype=torch.float32)[None]
                with torch.no_grad():
                    a = net.dist(x).mean.numpy()[0]
                rt.apply(k, np.clip(a, -1, 1) * scale)
            else:
                rt.apply(k, None)
            for _ in range(rt.ctrl_every):
                mujoco.mj_step(rt.sim.model, rt.sim.data)
            states.append(rt.save())
            rows.append((rt.error(k)[0],) + contact_state(rt.sim))
        out[name] = (states, rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="results/stage2_carry_robust_seeds.json")
    ap.add_argument("--seqs", default="camera_browse_1,cubemedium_inspect_1,"
                    "doorknob_use_1,flashlight_lift,pyramidlarge_inspect_1")
    ap.add_argument("--ckpt", default="results/ppo_carry")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--w", type=int, default=360)
    ap.add_argument("--h", type=int, default=300)
    ap.add_argument("--azim", type=float, default=135.0)
    ap.add_argument("--dist", type=float, default=0.30)
    ap.add_argument("--fps", type=float, default=7.5)
    a = ap.parse_args()

    # keyed on SUBJECT/SEQ: GRAB sequence names collide across subjects, and
    # a bare-name dict silently keeps one recording (tests/test_seed_keys.py)
    rows = {f"{r['subject']}/{r['seq']}": r
            for r in json.loads(Path(a.seeds).read_text())["rows"]}
    for name in [x.strip() for x in a.seqs.split(",") if x.strip()]:
        hits = [k for k in rows if k == name or k.split("/", 1)[1] == name]
        if len(hits) != 1:
            raise SystemExit(f"{name!r} matches {len(hits)} seed rows "
                             f"({hits}); name it as SUBJECT/SEQ")
        row = rows[hits[0]]
        name = row["seq"]
        seq = grab.load(f"{row['subject']}/{name}.npz", verts=True, stride=8)
        rt = T.ReferenceTracker(seq, hand="shadow")
        rt.apply_wrist_offset(np.asarray(row["offset"], float))
        k0 = int(row["start"])
        ck = Path(a.ckpt) / f"ppo_{name}_carry.pt"
        net = None
        if ck.exists():
            net = rl.make_policy(rt.n_obs, rt.n_action)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net.eval()
        runs = rollouts(rt, k0, net)
        cols = list(runs)
        # the camera FOLLOWS each column's own object: a fixed camera lost the
        # camera and doorknob clips off the frame within twenty frames, and
        # the thing to look at is the hand on the object, not the arc
        cam = mujoco.MjvCamera()
        frames, manifest = [], []
        bar = 44
        with mujoco.Renderer(rt.sim.model, height=a.h, width=a.w) as ren:
            n = len(runs["feedforward"][0])
            for i in range(n):
                sheet = Image.new("RGB", (a.w * len(cols), a.h + bar), "#101722")
                dr = ImageDraw.Draw(sheet)
                rec = {"frame": k0 + i}
                for j, c in enumerate(cols):
                    states, rws = runs[c]
                    rt.restore(states[i])
                    lookat = rt.sim.data.qpos[rt.obj_q:rt.obj_q + 3].copy()
                    im = frame(rt, ren, cam, a.azim, a.dist, lookat)
                    sheet.paste(im, (j * a.w, bar))
                    err, pen, ncon, nb, sided, grip = rws[i]
                    title = ("no controller" if c == "feedforward" else "PPO") + \
                        f"   {name}   frame {k0 + i}" if j == 0 else \
                        ("no controller" if c == "feedforward" else "PPO")
                    dr.text((j * a.w + 8, 5), title,
                            font=_font(14 if len(title) < 36 else 11, True),
                            fill="#f2f4f8")
                    dr.text((j * a.w + 8, 24),
                            f"{pen*1000:5.2f} mm in | {ncon:2d} con / {nb} links | "
                            f"{grip:6.0f} N | err {err*1000:5.1f} mm",
                            font=_font(11), fill="#98adbf")
                    rec[c] = {"err_mm": err * 1000, "pen_mm": pen * 1000,
                              "contacts": ncon, "links": nb, "sided": sided,
                              "grip_n": grip}
                frames.append(sheet)
                manifest.append(rec)
        out = Path(a.outdir) / f"carry_{name}.gif"
        out.parent.mkdir(parents=True, exist_ok=True)
        dur = int(1000 / a.fps)
        durations = [dur] * len(frames)
        durations[0] = 1200; durations[-1] = 1500       # hold reset and end
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=False)
        Path(str(out)[:-4] + ".json").write_text(json.dumps(
            {"seq": name, "subject": row["subject"], "start": k0,
             "offset": row["offset"], "search_seed": row.get("search_seed"),
             "checkpoint": str(ck) if net is not None else None,
             "frames": manifest}, indent=1))
        last = manifest[-1]
        print(f"{name:24s} {len(frames)} frames -> {out}   end: " +
              "   ".join(f"{c} {last[c]['err_mm']:.1f} mm / {last[c]['pen_mm']:.2f} mm in / "
                         f"{last[c]['links']} links" for c in cols), flush=True)


if __name__ == "__main__":
    main()
