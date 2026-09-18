"""Render a one- or two-hand DexTrack reference, with the object and the numbers.

`render.py` replays poses a simulator logged. This replays a reference before
any simulator has seen it: the hand poses come from forward kinematics on
their own URDF, so what you are looking at is exactly what their trainer
would be asked to track. Nothing is simulated, and the frame says so.

Each hand is drawn from its own URDF as one mocap body per link, the object
translucent from its convex decomposition, any link deeper than 2 mm red, and
per-frame penetration for each hand measured with the same probe the training
runs use.

    python dextrack/render_reference.py --ref out/refs/bimanual_binoculars_see_1.npy \\
        --obj-dir out/dextrack_assets/obj/ori_grab_s1_binoculars_see_1 \\
        --out figures/bimanual_binoculars_reference.gif
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from dextrack.left_hand import mujoco_copy, DOF_ORDER
from dextrack.render import urdf_links, font

ASSETS = Path("out/dextrack_assets")
HANDS = {"right": ASSETS / "allegro_hand_description_right_fly_v2.urdf",
         "left": ASSETS / "allegro_hand_description_left_fly_v2.urdf"}


class Kinematics:
    """Forward kinematics on one hand, from their 22-vector to world link poses."""

    def __init__(self, urdf: Path, meshdir: Path = ASSETS / "meshes"):
        self.model = mujoco.MjModel.from_xml_path(str(mujoco_copy(urdf, Path("out/urdf_work"), meshdir)))
        self.data = mujoco.MjData(self.model)
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        self.qadr = [self.model.jnt_qposadr[names.index(n)] for n in DOF_ORDER]
        self.links = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                      for i in range(self.model.nbody)]

    def __call__(self, q22: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self.data.qpos[:] = 0
        self.data.qpos[self.qadr] = q22
        mujoco.mj_kinematics(self.model, self.data)
        out = {}
        for i, n in enumerate(self.links):
            if n in (None, "world"):
                continue
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.xmat[i])
            out[n] = (self.data.xpos[i].copy(), quat.copy())          # wxyz
        return out


def build_scene(hand_urdfs: dict[str, Path], obj_dir: Path, meshes: Path = ASSETS / "meshes"):
    """One mocap body per link of every hand, plus the object."""
    import trimesh

    spec = mujoco.MjSpec()
    spec.compiler.meshdir = "."
    w = spec.worldbody
    for side, urdf in hand_urdfs.items():
        colour = [0.78, 0.83, 0.90, 1.0] if side == "right" else [0.80, 0.86, 0.80, 1.0]
        for name, (vis, _) in urdf_links(urdf).items():
            if vis is None:
                continue
            mf = meshes / Path(vis).name
            if not mf.exists():
                continue
            b = w.add_body(name=f"{side}/{name}", mocap=True)
            spec.add_mesh(name=f"m_{side}_{name}", file=str(mf.resolve()))
            b.add_geom(name=f"v_{side}_{name}", type=mujoco.mjtGeom.mjGEOM_MESH,
                       meshname=f"m_{side}_{name}", contype=0, conaffinity=0, rgba=colour)
    ob = w.add_body(name="object", mocap=True)
    merged = trimesh.load(str(obj_dir / "decomposed.obj"), force="mesh", process=False)
    stl = obj_dir / "decomposed_merged.stl"
    merged.export(str(stl))
    spec.add_mesh(name="m_obj", file=str(stl.resolve()))
    ob.add_geom(name="v_obj", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m_obj",
                contype=0, conaffinity=0, rgba=[0.98, 0.64, 0.24, 0.45])
    spec.visual.headlight.ambient[:] = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    return spec.compile()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--obj-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--w", type=int, default=460)
    ap.add_argument("--h", type=int, default=380)
    ap.add_argument("--dist", type=float, default=0.62)
    ap.add_argument("--azim", type=float, default=135.0)
    ap.add_argument("--fps", type=float, default=12.0)
    a = ap.parse_args()

    ref = np.load(a.ref, allow_pickle=True).item()
    q_r = np.asarray(ref["robot_delta_states_weights_np"], float)
    q_l = ref.get("robot_delta_states_weights_np_left")
    q_l = None if q_l is None else np.asarray(q_l, float)
    obj_pos = np.asarray(ref["object_transl"], float)
    obj_quat = np.asarray(ref["object_rot_quat"], float)           # xyzw

    urdfs = {"right": HANDS["right"]} | ({"left": HANDS["left"]} if q_l is not None else {})
    fk = {s: Kinematics(u) for s, u in urdfs.items()}
    model = build_scene(urdfs, a.obj_dir)
    data = mujoco.MjData(model)
    mocap = {model.body(i).name: model.body_mocapid[i] for i in range(model.nbody)
             if model.body_mocapid[i] >= 0}

    from dextrack.penetration_torch import PenetrationProbe
    import torch
    probes, body_names = {}, {}
    for s, u in urdfs.items():
        names = [n for n in fk[s].links if n not in (None, "world")]
        body_names[s] = names
        probes[s] = PenetrationProbe(str(u), str(a.obj_dir), names, "cpu", spacing=0.002)

    frames, cam = [], mujoco.MjvCamera()
    idx = range(0, len(q_r), a.stride)
    with mujoco.Renderer(model, height=a.h, width=a.w) as ren:
        for k in idx:
            poses = {}
            for s in urdfs:
                q = q_r[k] if s == "right" else q_l[k]
                poses[s] = fk[s](q)
                for n, (p, quat) in poses[s].items():
                    key = f"{s}/{n}"
                    if key in mocap:
                        data.mocap_pos[mocap[key]] = p
                        data.mocap_quat[mocap[key]] = quat
            data.mocap_pos[mocap["object"]] = obj_pos[k]
            data.mocap_quat[mocap["object"]] = [obj_quat[k][3], *obj_quat[k][:3]]
            mujoco.mj_forward(model, data)

            pen = {}
            for s in urdfs:
                rb = torch.zeros(1, len(body_names[s]), 13)
                for i, n in enumerate(body_names[s]):
                    p, quat = poses[s][n]
                    rb[0, i, :3] = torch.tensor(p)
                    rb[0, i, 3:7] = torch.tensor([quat[1], quat[2], quat[3], quat[0]])
                op = torch.tensor(np.concatenate([obj_pos[k], obj_quat[k]]), dtype=torch.float32)[None]
                depth, _, per_link = probes[s](rb, op)
                # the probe reports only links that carry collision primitives,
                # in the order of its own link_idx, so key the depths by name
                by_name = {body_names[s][j]: float(per_link[0, i])
                           for i, j in enumerate(probes[s].link_idx)}
                pen[s] = (float(depth) * 1000, by_name)

            for gi in range(model.ngeom):
                gname = model.geom(gi).name
                if not gname.startswith("v_") or gname == "v_obj":
                    continue
                side = "right" if gname.startswith("v_right_") else "left"
                ln = gname[len(f"v_{side}_"):]
                deep = pen[side][1].get(ln, -1.0) > 0.002
                base = [0.78, 0.83, 0.90, 1.0] if side == "right" else [0.80, 0.86, 0.80, 1.0]
                model.geom_rgba[gi] = [0.90, 0.15, 0.15, 1.0] if deep else base

            tiles = []
            for az in (a.azim, a.azim + 90):
                cam.distance, cam.azimuth, cam.elevation = a.dist, az, -15.0
                cam.lookat[:] = obj_pos[k]
                ren.update_scene(data, camera=cam)
                tiles.append(Image.fromarray(ren.render()))
            im = Image.new("RGB", (2 * a.w, a.h + 50), "#101722")
            im.paste(tiles[0], (0, 50)); im.paste(tiles[1], (a.w, 50))
            dr = ImageDraw.Draw(im)
            dr.text((10, 6), a.title or a.ref.stem, font=font(15, True), fill="#f2f4f8")
            line = f"frame {k:3d} | right {pen['right'][0]:5.2f} mm"
            if "left" in pen:
                line += f" | left {pen['left'][0]:5.2f} mm"
            bad = max(v[0] for v in pen.values()) > 2.0
            dr.text((10, 28), line + "   kinematic reference, nothing simulated",
                    font=font(12), fill="#ff6b6b" if bad else "#7ddc8a")
            frames.append(im)

    dur = [int(1000 / a.fps)] * len(frames); dur[0] = 1000; dur[-1] = 1500
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=False)
    picks = np.linspace(0, len(frames) - 1, 4).astype(int)
    sheet = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for j, i in enumerate(picks):
        sheet.paste(frames[i], (0, j * frames[0].height))
    sheet.save(str(a.out.with_suffix(".preview.jpg")), quality=85)
    print(f"wrote {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
