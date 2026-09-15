"""Render measured GRAB tracking rollouts for the README.

Capture steps the free object in MuJoCo. Rendering uses a separate MjData and
the captured states, so camera work and overlays cannot change the rollout.
The local cache includes the compiled model; --render-only changes presentation
without repeating fitting/search. Source GRAB/MANO assets are required to capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEMOS = {
    "mug": ("mug_drink_1", "ppo", "MUG / DRINK"),
    "bowl": ("bowl_drink_1", "bimanual", "BOWL / DRINK"),
    "binoculars": ("binoculars_see_1", "bimanual", "BINOCULARS / SEE"),
    "camera": ("camera_takepicture_2", "bimanual", "CAMERA / TAKE A PICTURE"),
    "gamecontroller": ("gamecontroller_play_1", "bimanual", "GAME CONTROLLER / PLAY"),
}
CHECKPOINT = Path("results/ppo_mug_drink_1_h160.pt")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation_error(q, target):
    return 2 * np.arccos(np.clip(abs(np.dot(q, target)), 0, 1))


def capture(name, cache, seed):
    from oppdef.human import grab, track

    seq_name, mode, _title = DEMOS[name]
    print(f"{name}: fitting s1/{seq_name}", flush=True)
    seq = grab.load(f"s1/{seq_name}.npz", verts=True, stride=8)
    metadata = {
        "sequence": f"s1/{seq_name}", "mode": mode, "seed": seed,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "mujoco": mujoco.__version__, "numpy": np.__version__,
        "physics_source_sha256": {str(p): digest(p)
                                  for p in sorted(Path("src/oppdef").rglob("*.py"))},
        "reference_stride": 8,
        "scope": "Selected per-reference simulation; assisted initial grasp; "
                 "mocap-driven wrists and actuated fingers; free object under gravity.",
    }
    if mode == "ppo":
        import torch
        from oppdef.human import rl

        torch.set_num_threads(1)
        rt = track.ReferenceTracker(seq)
        if seed != 0:
            raise ValueError("The saved mug checkpoint uses grasp-search seed 0")
        # The checkpoint predates the switch to tracking-scored grasp search.
        # Its hold-scored initial grasp is part of the evaluation protocol.
        fit = rt.synthesize_grasp(objective="hold", seed=0)
        net = rl.make_policy(rt.n_obs, rt.n_action)
        net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        net.eval()
        cfg = rl.RLConfig()
        scale = np.r_[np.full(3, cfg.a_pos), np.full(3, cfg.a_rot),
                      np.full(rt.n_action - 6, cfg.a_fin)]
        start = 5  # The saved 111-frame evaluation window.
        metadata.update(checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
                        controller="Per-reference PPO residual + retargeted feedforward",
                        grasp_search={"objective": "hold", "seed": 0, "samples": 12,
                                      "rounds": 2, "offset": fit.offset.tolist(),
                                      "accepted": rt.grasp_frames_after >= rt.grasp_frames_before
                                                  and bool(fit.held)},
                        observation="Simulator state and reference lookahead")

        def command(k):
            with torch.no_grad():
                obs = torch.as_tensor(rt.observe(k), dtype=torch.float32)[None]
                action = net.dist(obs).mean.numpy()[0]
            rt.apply(k, np.clip(action, -1, 1) * scale)
    else:
        rt = track.BimanualTracker(seq)
        start = 0
        print(f"{name}: searching wrist offsets over all {rt.T} frames", flush=True)
        score, offsets = rt.synthesize_grasp(samples=10, rounds=2, seed=seed,
                                             objective="track", track_steps=None)
        metadata.update(controller="Joint retarget + searched wrist offsets + feedforward",
                        grasp_search={"samples": 10, "rounds": 2, "seed": seed,
                                      "objective": "track", "track_steps": rt.T,
                                      "score_unitless": score,
                                      "offsets": {sd: v.tolist() for sd, v in offsets.items()}},
                        observation="Reference commands; no learned bimanual policy")

        def command(k):
            for side in ("r", "l"):
                rt.command(side, k)
            rt.set_ctrl(k)

    m, d = rt.sim.model, rt.sim.data
    object_joints = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == rt.sim.obj_bid]
    assert any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in object_joints)
    assert not any(rt.sim.obj_bid in (int(m.eq_obj1id[e]), int(m.eq_obj2id[e]))
                   for e in range(m.neq)
                   if m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD)
    assert m.opt.gravity[2] < 0
    rt.reset_at(start)
    initial = d.qpos.copy()
    states, controls, positions, orientations, errors, contacts = [], [], [], [], [], []
    for k in range(start, rt.T):
        command(k)
        mujoco.mj_step(m, d, nstep=rt.ctrl_every)
        p = d.qpos[rt.obj_q:rt.obj_q + 3].copy()
        q = d.qpos[rt.obj_q + 3:rt.obj_q + 7].copy()
        states.append(d.qpos.copy())
        controls.append(d.ctrl.copy())
        positions.append(p)
        orientations.append(q)
        errors.append((np.linalg.norm(p - rt.ref_pos[k]),
                       rotation_error(q, rt.ref_quat[k])))
        object_geoms = set(rt.sim.obj_gids)
        contacts.append(sum((int(c.geom1) in object_geoms) !=
                            (int(c.geom2) in object_geoms) for c in d.contact))
    errors = np.asarray(errors)
    if not np.all(np.isfinite(errors)):
        raise RuntimeError(f"{name}: non-finite rollout")
    metadata.update(start_frame=start, frames=len(errors), reference_window=list(rt.window),
                    reference_dt_s=seq.dt, physics_dt_s=float(m.opt.timestep),
                    substeps=rt.ctrl_every,
                    object_mass_kg=float(m.body_mass[rt.sim.obj_bid]),
                    mean_position_error_mm=float(errors[:, 0].mean() * 1000),
                    max_position_error_mm=float(errors[:, 0].max() * 1000),
                    final_position_error_mm=float(errors[-1, 0] * 1000),
                    mean_orientation_error_deg=float(np.rad2deg(errors[:, 1]).mean()),
                    frames_within_50mm=int((errors[:, 0] < 0.05).sum()),
                    frames_above_100mm=int((errors[:, 0] > 0.10).sum()),
                    min_object_contacts=int(min(contacts)),
                    position_error_mm=(errors[:, 0] * 1000).tolist(),
                    orientation_error_deg=np.rad2deg(errors[:, 1]).tolist(),
                    object_contacts=contacts)
    # Replay through the existing evaluator from the same reset. This checks
    # that the movie loop did not change the experiment's control protocol.
    if mode == "ppo":
        replay, _ = rl.evaluate(rt, net, start=start)
    else:
        rt.reset_at(start)
        replay = rt.rollout(start=start).pos_err
    delta = float(np.max(np.abs(replay - errors[:, 0])))
    if delta > 1e-8:
        raise RuntimeError(f"{name}: capture/evaluator mismatch {delta} m")
    metadata["evaluator_replay_max_difference_m"] = delta
    cache.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m, str(cache / "scene.mjb"), None)
    np.savez_compressed(cache / "states.npz", qpos=states, ctrl=controls,
                        object_pos=positions, object_quat=orientations,
                        ref_pos=rt.ref_pos[start:], ref_quat=rt.ref_quat[start:],
                        initial_qpos=initial)
    metadata["model_sha256"] = digest(cache / "scene.mjb")
    (cache / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{name}: {metadata['mean_position_error_mm']:.1f} mm mean, "
          f"{metadata['frames_within_50mm']}/{len(errors)} within 50 mm; "
          f"replay delta {delta:.2g} m", flush=True)


def font(size, bold=False):
    path = Path("/usr/share/fonts/truetype/dejavu") / (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def line(scene, p, q, color, width=0.001):
    if np.linalg.norm(q - p) < 1e-8 or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                       np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p, q)
    scene.ngeom += 1


def render(name, cache, out):
    meta = json.loads((cache / "metrics.json").read_text())
    z = np.load(cache / "states.npz")
    m = mujoco.MjModel.from_binary_path(str(cache / "scene.mjb"))
    d = mujoco.MjData(m)
    # Display-only materials: physics was captured before these edits.
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if body.startswith("obj_"):
            m.geom_rgba[g] = [0.98, 0.64, 0.24, 1.0]
        elif "forearm" in body:
            m.geom_rgba[g, 3] = 0.0
        elif body.startswith("l_") and m.geom_rgba[g, 3] > 0:
            m.geom_rgba[g] = [0.27, 0.72, 0.75, 1.0]
        elif m.geom_rgba[g, 3] > 0:
            m.geom_rgba[g] = [0.73, 0.79, 0.88, 1.0]
    m.light_ambient[:] = 0.10
    m.light_diffuse[:] = 0.45
    m.light_specular[:] = 0.12
    m.vis.headlight.ambient[:] = [0.20, 0.20, 0.20]
    m.vis.headlight.diffuse[:] = [0.40, 0.40, 0.40]
    m.vis.headlight.specular[:] = [0.12, 0.12, 0.12]
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.62, 135.0, -22.0
    if name == "mug":
        cam.distance = 0.44
    width, height = 640, 480
    frames = []
    pos, ref = z["object_pos"], z["ref_pos"]
    mode_label = "PER-CLIP PPO" if meta["mode"] == "ppo" else "BIMANUAL / GRASP SEARCH"
    title_font, body_font, small_font = font(20, True), font(14), font(12)
    with mujoco.Renderer(m, height=height, width=width) as renderer:
        for k, state in enumerate(z["qpos"]):
            d.qpos[:] = state
            d.ctrl[:] = z["ctrl"][k]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = ref[k]
            renderer.update_scene(d, camera=cam)
            # Local portions of reference/actual paths remain in view as the
            # camera follows the reference, not the measured object position.
            for j in range(max(1, k - 24), min(len(ref), k + 12)):
                if j % 2 == 0:
                    line(renderer.scene, ref[j - 1], ref[j], [0.22, 0.79, 0.90, 0.65])
            for j in range(max(1, k - 24), k + 1):
                line(renderer.scene, pos[j - 1], pos[j], [1.0, 0.68, 0.27, 0.7], .0008)
            pic = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(pic)
            draw.rectangle((0, 0, width, 65), fill="#101722")
            draw.text((20, 12), DEMOS[name][2], font=title_font, fill="#f2f4f8")
            draw.text((20, 40), mode_label + "   |   MUJOCO PHYSICS", font=small_font, fill="#98adbf")
            draw.rectangle((0, height - 75, width, height), fill="#101722")
            e = meta["position_error_mm"][k]
            draw.text((20, height - 64), f"Position error  {e:5.1f} mm", font=body_font, fill="#f2f4f8")
            draw.text((247, height - 64), f"Angle error {meta['orientation_error_deg'][k]:.1f}°",
                      font=body_font, fill="#f2f4f8")
            draw.text((width - 180, height - 64), f"Mean  {meta['mean_position_error_mm']:.1f} mm",
                      font=body_font, fill="#f2f4f8")
            draw.text((20, height - 40), "REFERENCE", font=small_font, fill="#38cae6")
            draw.text((114, height - 40), "OBJECT", font=small_font, fill="#ffad45")
            draw.text((width - 222, height - 40), f"Frame {k+1:03d}/{len(pos):03d}  |  {k*meta['reference_dt_s']:.1f}s",
                      font=small_font, fill="#98adbf")
            draw.rectangle((20, height - 13, width - 20, height - 10), fill="#293548")
            draw.rectangle((20, height - 13, 20 + (width - 40) * (k + 1) / len(pos), height - 10),
                           fill="#38cae6")
            frames.append(pic)
    # Shared palette prevents colour flicker between GIF frames.
    sample = Image.new("RGB", (80 * 8, 60 * ((len(frames) + 7) // 8)))
    for i, f in enumerate(frames):
        sample.paste(f.resize((80, 60)), (80 * (i % 8), 60 * (i // 8)))
    palette = sample.quantize(colors=192)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    # GIF timing has 10 ms granularity; distribute rounding across frames.
    ticks = np.rint(np.arange(len(frames) + 1) * meta["reference_dt_s"] * 100).astype(int)
    durations = (np.diff(ticks) * 10).tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(out, save_all=True, append_images=quantized[1:],
                      duration=durations, loop=0, optimize=True, disposal=2)
    picks = np.linspace(0, len(frames) - 1, 4).astype(int)
    sheet = Image.new("RGB", (width * 2, height * 2))
    for i, k in enumerate(picks):
        sheet.paste(frames[k], ((i % 2) * width, (i // 2) * height))
    sheet.save(cache / "preview.jpg", quality=90)
    meta.update(gif=str(out), gif_sha256=digest(out), gif_frames=len(frames),
                gif_duration_ms=sum(durations), renderer_sha256=digest(__file__),
                camera="Follows the reference object position",
                presentation="Display-only hand/object recolouring; forearms hidden; "
                             "cyan reference path, amber measured object path")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("demos", nargs="+", choices=DEMOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--capture-only", action="store_true")
    a = ap.parse_args()
    if a.render_only and a.capture_only:
        ap.error("choose one of --render-only and --capture-only")
    for name in a.demos:
        cache = Path("out/tracking_gifs") / name
        if not a.render_only:
            capture(name, cache, a.seed)
        if not a.capture_only:
            render(name, cache, Path(f"figures/physics_{name}.gif"))


if __name__ == "__main__":
    main()
