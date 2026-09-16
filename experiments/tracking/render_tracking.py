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

# Carry-scored seeds (stage2_carry.py, the five that stay clean under all eight
# wrist perturbations), each rendered twice: the retargeted trajectory with no
# controller, and the PPO policy stage3_carry.py trained from that start.
# The subject, start frame and wrist offset come from the seeds file, keyed on
# SUBJECT/SEQ; the bare name here is only a label.
CARRY_SEEDS = Path("results/stage2_carry_robust_seeds.json")
CARRY_CKPT = Path("results/ppo_carry")
for _key, _seq, _title in (
        ("flashlight", "flashlight_lift", "FLASHLIGHT / LIFT"),
        ("camera1", "camera_browse_1", "CAMERA / BROWSE"),
        ("cube", "cubemedium_inspect_1", "CUBE / INSPECT"),
        ("doorknob", "doorknob_use_1", "DOORKNOB / USE"),
        ("pyramid", "pyramidlarge_inspect_1", "PYRAMID / INSPECT")):
    DEMOS[f"carry_{_key}"] = (_seq, "carry", _title)
    DEMOS[f"carryppo_{_key}"] = (_seq, "carry_ppo", _title)


def carry_seed(seq_name):
    rows = json.loads(CARRY_SEEDS.read_text())["rows"]
    by = {f"{r['subject']}/{r['seq']}": r for r in rows}
    hits = [k for k in by if k.split("/", 1)[1] == seq_name]
    if len(hits) != 1:
        raise SystemExit(f"{seq_name!r} matches {len(hits)} carry seeds: {hits}")
    return by[hits[0]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation_error(q, target):
    return 2 * np.arccos(np.clip(abs(np.dot(q, target)), 0, 1))


def contact_diagnostics(m, d, obj_gids, obj_bid):
    """Penetration, grip force and equilibrium residual at the current state.

    These are the quantities that distinguish a grasp from a hand buried in the
    object. Omitting them is how a gallery of contact artifacts was published
    with only a position-error column: the artifact does not show up there, and
    is unmissable here. `residual` is the net unbalanced wrench on the object
    including gravity, in multiples of object weight. The scale is: 0x is
    equilibrium (contacts balance gravity and nothing else), 1x is freefall
    (nothing supports the object), and 100x+ is a FRESHLY PLACED penetrated
    configuration. Rejecting both failure ends at placement is the property
    depth alone does not have -- but a burial that has settled reads ~0x like
    a real grasp, so this discriminates how a pose was placed, not its steady
    state. Read it at reset, not after stepping.
    """
    obj = set(obj_gids)
    weight = float(m.body_mass[obj_bid]) * 9.81
    com = d.xipos[obj_bid]
    pen = force = 0.0
    n = 0
    net = np.zeros(3)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if (g1 in obj) == (g2 in obj):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fw = d.contact[i].frame.reshape(3, 3).T @ f6[:3]
        net += -fw if g1 in obj else fw
        pen = max(pen, -d.contact[i].dist)
        force += abs(f6[0])
        n += 1
    net[2] -= weight
    return pen * 1000, force, n, float(np.linalg.norm(net) / weight)


def capture(name, cache, seed):
    from oppdef.human import grab, track

    seq_name, mode, _title = DEMOS[name]
    subject = carry_seed(seq_name)["subject"] if mode.startswith("carry") else "s1"
    print(f"{name}: fitting {subject}/{seq_name}", flush=True)
    seq = grab.load(f"{subject}/{seq_name}.npz", verts=True, stride=8)
    metadata = {
        "sequence": f"{subject}/{seq_name}", "mode": mode, "seed": seed,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "mujoco": mujoco.__version__, "numpy": np.__version__,
        "physics_source_sha256": {str(p): digest(p)
                                  for p in sorted(Path("src/oppdef").rglob("*.py"))},
        "reference_stride": 8,
        "scope": "Selected per-reference simulation; assisted initial grasp; "
                 "mocap-driven wrists and actuated fingers; free object under gravity.",
    }
    if mode.startswith("carry"):
        # The wrist offset the carry-scored search accepted, applied to the
        # whole trajectory; nothing is searched or closed here. Rolled from
        # the frame the carry was scored from, exactly as stage3_carry.py
        # evaluates it.
        row = carry_seed(seq_name)
        rt = track.ReferenceTracker(seq)
        rt.apply_wrist_offset(np.asarray(row["offset"], float))
        start = int(row["start"])
        metadata.update(grasp_search={
            "objective": "carry", "search_seed": row.get("search_seed"),
            "offset": row["offset"], "robust_clean_of_8": row.get("robust_clean"),
            "seeds_file": str(CARRY_SEEDS), "seeds_sha256": digest(CARRY_SEEDS)})
        if mode == "carry_ppo":
            import torch
            from oppdef.human import rl

            torch.set_num_threads(1)
            ck = CARRY_CKPT / f"ppo_{seq_name}_carry.pt"
            net = rl.make_policy(rt.n_obs, rt.n_action)
            net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
            net.eval()
            cfg = rl.RLConfig()
            scale = np.r_[np.full(3, cfg.a_pos), np.full(3, cfg.a_rot),
                          np.full(rt.n_action - 6, cfg.a_fin)]
            metadata.update(checkpoint=str(ck), checkpoint_sha256=digest(ck),
                            controller="Per-reference PPO residual + retargeted feedforward",
                            observation="Simulator state and reference lookahead")

            def command(k):
                with torch.no_grad():
                    obs = torch.as_tensor(rt.observe(k), dtype=torch.float32)[None]
                    action = net.dist(obs).mean.numpy()[0]
                rt.apply(k, np.clip(action, -1, 1) * scale)
        else:
            metadata.update(controller="Retargeted feedforward only; no controller",
                            observation="None")

            def command(k):
                rt.apply(k, None)
    elif mode == "ppo":
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
    pens, forces, residuals = [], [], []
    lead = 0
    if mode.startswith("carry"):
        # Show the RESET state as the first frame. The carry seeds start as
        # burials -- 10 to 32 mm inside, hundreds of newtons -- and relax out
        # in the first control frame; a movie that starts one step in never
        # shows what the search accepted. Zero error by construction.
        mujoco.mj_forward(m, d)
        states.append(d.qpos.copy()); controls.append(d.ctrl.copy())
        positions.append(d.qpos[rt.obj_q:rt.obj_q + 3].copy())
        orientations.append(d.qpos[rt.obj_q + 3:rt.obj_q + 7].copy())
        errors.append((0.0, 0.0))
        pen, force, ncon, residual = contact_diagnostics(
            m, d, rt.sim.obj_gids, rt.sim.obj_bid)
        contacts.append(ncon); pens.append(pen); forces.append(force)
        residuals.append(residual)
        lead = 1
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
        pen, force, ncon, residual = contact_diagnostics(
            m, d, rt.sim.obj_gids, rt.sim.obj_bid)
        contacts.append(ncon)
        pens.append(pen)
        forces.append(force)
        residuals.append(residual)
    errors = np.asarray(errors)
    if not np.all(np.isfinite(errors)):
        raise RuntimeError(f"{name}: non-finite rollout")
    # Statistics over the ROLLED frames only; the leading reset frame (carry
    # demos) has zero error by construction and would flatter the mean.
    rolled = errors[lead:]
    metadata.update(start_frame=start, frames=len(errors), reference_window=list(rt.window),
                    reference_dt_s=seq.dt, physics_dt_s=float(m.opt.timestep),
                    leading_reset_frame=lead,
                    substeps=rt.ctrl_every,
                    object_mass_kg=float(m.body_mass[rt.sim.obj_bid]),
                    mean_position_error_mm=float(rolled[:, 0].mean() * 1000),
                    max_position_error_mm=float(rolled[:, 0].max() * 1000),
                    final_position_error_mm=float(rolled[-1, 0] * 1000),
                    mean_orientation_error_deg=float(np.rad2deg(rolled[:, 1]).mean()),
                    frames_within_50mm=int((rolled[:, 0] < 0.05).sum()),
                    frames_above_100mm=int((rolled[:, 0] > 0.10).sum()),
                    min_object_contacts=int(min(contacts[lead:])),
                    max_penetration_mm=float(max(pens[lead:])),
                    mean_penetration_mm=float(np.mean(pens[lead:])),
                    mean_contact_force_N=float(np.mean(forces[lead:])),
                    max_contact_force_N=float(max(forces[lead:])),
                    mean_equilibrium_residual_x_weight=float(np.mean(residuals[lead:])),
                    reset_penetration_mm=float(pens[0]) if lead else None,
                    reset_contact_force_N=float(forces[0]) if lead else None,
                    penetration_mm=pens, contact_force_N=forces,
                    equilibrium_residual_x_weight=residuals,
                    live_diagnostics=True,
                    position_error_mm=(errors[:, 0] * 1000).tolist(),
                    orientation_error_deg=np.rad2deg(errors[:, 1]).tolist(),
                    object_contacts=contacts)
    # Replay through the existing evaluator from the same reset. This checks
    # that the movie loop did not change the experiment's control protocol.
    if mode in ("ppo", "carry_ppo"):
        replay, _ = rl.evaluate(rt, net, start=start)
    elif mode == "carry":
        rt.reset_at(start)
        replay = rt.rollout(start=start, steps=rt.T - start).pos_err
    else:
        rt.reset_at(start)
        replay = rt.rollout(start=start).pos_err
    delta = float(np.max(np.abs(replay - errors[lead:, 0])))
    if delta > 1e-8:
        raise RuntimeError(f"{name}: capture/evaluator mismatch {delta} m")
    metadata["evaluator_replay_max_difference_m"] = delta
    cache.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m, str(cache / "scene.mjb"), None)
    np.savez_compressed(cache / "states.npz", qpos=states, ctrl=controls,
                        object_pos=positions, object_quat=orientations,
                        ref_pos=rt.ref_pos[start - lead:], ref_quat=rt.ref_quat[start - lead:],
                        initial_qpos=initial)
    metadata["model_sha256"] = digest(cache / "scene.mjb")
    (cache / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{name}: {metadata['mean_position_error_mm']:.1f} mm mean, "
          f"{metadata['frames_within_50mm']}/{len(rolled)} within 50 mm; "
          f"replay delta {delta:.2g} m", flush=True)
    print(f"{name}: penetration {np.mean(pens[lead:]):.2f} mm mean / {max(pens[lead:]):.2f} mm max, "
          f"grip {np.mean(forces[lead:]):.1f} N "
          f"({np.mean(forces[lead:]) / (metadata['object_mass_kg'] * 9.81):.0f}x weight), "
          f"equilibrium residual {np.mean(residuals[lead:]):.1f}x weight"
          + (f"; at reset {pens[0]:.2f} mm / {forces[0]:.0f} N" if lead else ""), flush=True)


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
    # The object is TRANSLUCENT on purpose. Opaque, a hand buried 13 mm inside
    # it looks exactly like a hand touching it, which is how a gallery of
    # contact artifacts got published with a caption that said otherwise.
    # Through it, the fingers inside the mug are simply visible.
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if body.startswith("obj_"):
            m.geom_rgba[g] = [0.98, 0.64, 0.24, 0.42]
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
    obj_bid = [b for b in range(m.nbody)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("obj_")][0]
    obj_gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == obj_bid]
    weight = float(m.body_mass[obj_bid]) * 9.81
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.62, 135.0, -22.0
    if name == "mug":
        cam.distance = 0.44
    if meta["mode"].startswith("carry"):
        cam.distance = 0.40           # hand-sized objects; the mug framing
    obj_set = set(obj_gids)
    hand_reset = {g: m.geom_rgba[g].copy() for g in range(m.ngeom)
                  if int(m.geom_bodyid[g]) != obj_bid and m.geom_rgba[g, 3] > 0}
    width, height = 640, 480
    frames = []
    # REPLAYED diagnostics. Every frame below is a fresh placement -- qpos
    # written, mj_forward, no velocity, no warm-started constraint state -- so
    # what contact_diagnostics reads here is the state's geometry (penetration,
    # contact count) and the force a placement needs to resolve it, NOT the
    # force the rollout carried. The equilibrium residual of a placement is a
    # legitimate number -- it is what a reset state reads -- but it is hundreds
    # of x by construction and it is NOT the live residual: a replayed column
    # once wore the live name and put a 300-600 N "net force on the hand" into
    # STAGE2 that the live value (0.4x) refuted. So it is kept, under a name
    # that says what it is. Live values, when capture() recorded them, stay
    # under their plain names and are never overwritten here.
    diag = {"penetration_mm_replay": [], "contact_force_N_replay": [],
            "object_contacts_replay": [], "placement_residual_x_weight_replay": []}
    pos, ref = z["object_pos"], z["ref_pos"]
    mode_label = {"ppo": "PER-CLIP PPO",
                  "carry": "CARRY-SCORED SEED  |  NO CONTROLLER",
                  "carry_ppo": "CARRY-SCORED SEED  |  PER-CLIP PPO"}.get(
                      meta["mode"], "BIMANUAL / GRASP SEARCH")
    # Carry demos captured live per-frame diagnostics; show those, labelled,
    # rather than the force a fresh placement of the state would need.
    live = meta["mode"].startswith("carry") and meta.get("live_diagnostics")
    title_font, body_font, small_font = font(20, True), font(14), font(12)
    with mujoco.Renderer(m, height=height, width=width) as renderer:
        for k, state in enumerate(z["qpos"]):
            d.qpos[:] = state
            d.ctrl[:] = z["ctrl"][k]
            mujoco.mj_forward(m, d)
            # Paint the hand links that are INSIDE the object, before the scene
            # is built so it affects THIS frame. Attribution beats assertion:
            # "ffproximal is 13 mm in" is a claim a reader can check against the
            # picture, where "penetration 13.00 mm" is one they cannot.
            buried = set()
            for i in range(d.ncon):
                g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
                in1, in2 = g1 in obj_set, g2 in obj_set
                if in1 == in2 or -float(d.contact[i].dist) <= 0.002:
                    continue
                buried.add(int(m.geom_bodyid[g2 if in1 else g1]))
            for g, rgba in hand_reset.items():
                m.geom_rgba[g] = ([0.95, 0.30, 0.30, 1.0]
                                  if int(m.geom_bodyid[g]) in buried else rgba)
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
            draw.rectangle((0, height - 97, width, height), fill="#101722")
            pen, force, ncon, placement_residual = contact_diagnostics(m, d, obj_gids, obj_bid)
            diag["penetration_mm_replay"].append(pen)
            diag["contact_force_N_replay"].append(force)
            diag["object_contacts_replay"].append(ncon)
            diag["placement_residual_x_weight_replay"].append(placement_residual)
            e = meta["position_error_mm"][k]
            draw.text((20, height - 86), f"Position error  {e:5.1f} mm", font=body_font, fill="#f2f4f8")
            draw.text((247, height - 86), f"Angle error {meta['orientation_error_deg'][k]:.1f}°",
                      font=body_font, fill="#f2f4f8")
            draw.text((width - 180, height - 86), f"Mean  {meta['mean_position_error_mm']:.1f} mm",
                      font=body_font, fill="#f2f4f8")
            # A grasp and a hand buried in the object look identical above and
            # are told apart here. Red is not decoration: it is the threshold
            # past which the contact set is manufactured rather than measured.
            if live:
                pen, force = meta["penetration_mm"][k], meta["contact_force_N"][k]
            bad = pen > 2.0 or force > 40 * weight
            draw.text((20, height - 62), f"Penetration {pen:5.2f} mm", font=body_font,
                      fill="#ff6b6b" if pen > 2.0 else "#7ddc8a")
            draw.text((205, height - 62),
                      f"Grip {force:7.1f} N ({force / weight:4.0f}x weight) "
                      + ("live" if live else "replayed"),
                      font=body_font, fill="#ff6b6b" if force > 40 * weight else "#7ddc8a")
            draw.text((width - 152, height - 62), f"{ncon:3d} contacts", font=body_font,
                      fill="#ff6b6b" if bad else "#98adbf")
            draw.text((20, height - 40), "REFERENCE", font=small_font, fill="#38cae6")
            draw.text((114, height - 40), "OBJECT", font=small_font, fill="#ffad45")
            if bad:
                draw.text((width - 232, 14), "CONTACT SET IS AN ARTIFACT",
                          font=small_font, fill="#ff6b6b")
                draw.text((width - 232, 30), "NOT A GRASP", font=title_font, fill="#ff6b6b")
            elif live and ncon > 0 and e < 100:
                draw.text((width - 232, 14), "OUT OF THE OBJECT, HELD",
                          font=small_font, fill="#7ddc8a")
                draw.text((width - 232, 30), "A GRASP", font=title_font, fill="#7ddc8a")
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
    # Replayed columns never overwrite live ones, and a stale live residual
    # from an earlier replay is removed rather than left to be read again.
    for stale in ("equilibrium_residual_x_weight", "mean_equilibrium_residual_x_weight"):
        if stale in meta and "live_diagnostics" not in meta:
            meta.pop(stale)
    for stale in ("penetration_mm", "contact_force_N", "object_contacts",
                  "max_penetration_mm", "mean_penetration_mm",
                  "mean_contact_force_N", "max_contact_force_N"):
        if stale in meta and "live_diagnostics" not in meta:
            meta.pop(stale)
    meta.update({k: v for k, v in diag.items()},
                max_penetration_mm_replay=float(max(diag["penetration_mm_replay"])),
                mean_penetration_mm_replay=float(np.mean(diag["penetration_mm_replay"])),
                mean_contact_force_N_replay=float(np.mean(diag["contact_force_N_replay"])),
                max_contact_force_N_replay=float(max(diag["contact_force_N_replay"])),
                diagnostics_note=("*_replay columns are read from a fresh placement of each saved "
                                  "state (qpos + mj_forward): penetration and contact count are "
                                  "geometric (fixed by qpos, restored exactly) and survive replay "
                                  "unchanged; force depends on velocity and the warm-started "
                                  "constraint solution, which replay does not reconstruct, so it is the "
                                  "placement's resolving force, and placement_residual is the "
                                  "equilibrium residual OF A FRESH PLACEMENT (hundreds of x by "
                                  "construction; the live residual of a tracking rollout is ~0.4x). "
                                  "Live per-frame values exist only if capture() wrote them."))
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
