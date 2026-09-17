"""Render a DexTrack rollout, logged by the audit hook, in MuJoCo with our
instruments: translucent object, any link deeper than 2 mm painted red, and
the live numbers on every frame.

Isaac Gym cannot render headless, so the scene is rebuilt here: one mocap
body per Allegro link carrying its visual mesh and its URDF collision
primitive, and one mocap body for the object carrying DexTrack's convex
decomposition. Every frame's poses come from the log; nothing is simulated.
The penetration and grip figures come from the audit JSON for the same log,
so the picture and the number are the same frame of the same rollout.
"""
from __future__ import annotations

import argparse, json, os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(_BOLD if bold else _FONT, size)
    except Exception:                                    # noqa: BLE001
        return ImageFont.load_default()


def urdf_links(urdf):
    """{link: (visual mesh file or None, [(kind, size, xyz, quat_wxyz)])}."""
    root = ET.parse(urdf).getroot()
    out = {}
    for link in root.findall("link"):
        vis = None
        v = link.find("visual")
        if v is not None and v.find("geometry") is not None and v.find("geometry").find("mesh") is not None:
            vis = v.find("geometry").find("mesh").get("filename")
        cols = []
        for c in link.findall("collision"):
            o = c.find("origin")
            xyz = [float(x) for x in o.get("xyz").split()] if o is not None and o.get("xyz") else [0, 0, 0]
            rpy = [float(x) for x in o.get("rpy").split()] if o is not None and o.get("rpy") else [0, 0, 0]
            q = R.from_euler("xyz", rpy).as_quat()        # xyzw
            g = c.find("geometry")
            if g is None:
                continue
            if g.find("box") is not None:
                cols.append(("box", [float(x) / 2 for x in g.find("box").get("size").split()], xyz, [q[3], q[0], q[1], q[2]]))
            elif g.find("sphere") is not None:
                cols.append(("sphere", [float(g.find("sphere").get("radius"))], xyz, [1, 0, 0, 0]))
        out[link.get("name")] = (vis, cols)
    return out


def build_model(urdf, meshes_dir, obj_dir, link_names, show_collision):
    links = urdf_links(urdf)
    spec = mujoco.MjSpec()
    spec.compiler.meshdir = "."
    m = spec.worldbody
    for n in link_names:
        vis, cols = links.get(n, (None, []))
        if vis is None and not cols:
            continue
        b = m.add_body(name=n, mocap=True)
        if vis is not None:
            mf = Path(meshes_dir) / Path(vis).name
            if mf.exists():
                mesh = spec.add_mesh(name=f"m_{n}", file=str(mf.resolve()))
                g = b.add_geom(name=f"v_{n}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m_{n}",
                               contype=0, conaffinity=0, rgba=[0.78, 0.83, 0.90, 1.0])
        for k, (kind, size, xyz, q) in enumerate(cols):
            if not show_collision:
                continue
            gt = mujoco.mjtGeom.mjGEOM_BOX if kind == "box" else mujoco.mjtGeom.mjGEOM_SPHERE
            b.add_geom(name=f"c_{n}_{k}", type=gt, size=size + [0] * (3 - len(size)), pos=xyz, quat=q,
                       contype=0, conaffinity=0, rgba=[0.3, 0.6, 1.0, 0.25])
    ob = m.add_body(name="object", mocap=True)
    # MuJoCo reads one object per OBJ; the decomposition has one per convex
    # part, so merge them into a single STL first
    import trimesh
    merged = trimesh.load(str(Path(obj_dir) / "decomposed.obj"), force="mesh", process=False)
    stl = Path(obj_dir) / "decomposed_merged.stl"
    merged.export(str(stl))
    mesh = spec.add_mesh(name="m_obj", file=str(stl.resolve()))
    ob.add_geom(name="v_obj", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m_obj", contype=0, conaffinity=0,
                rgba=[0.98, 0.64, 0.24, 0.45])
    spec.visual.headlight.ambient[:] = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    model = spec.compile()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--urdf", default="out/dextrack_assets/allegro_hand_description_right_fly_v2.urdf")
    ap.add_argument("--meshes", default="out/dextrack_assets/meshes")
    ap.add_argument("--obj-dir", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=420)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--azim", type=float, default=135.0)
    ap.add_argument("--dist", type=float, default=0.34)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--collision", action="store_true", help="also draw the collision primitives")
    a = ap.parse_args()

    rows = np.load(a.log, allow_pickle=True)[1:]           # drop the pre-pin first row
    audit = json.loads(Path(a.audit).read_text())["frames"][1:]
    # drop the trailing reset frame(s): their test mode resets when the clip
    # ends and the hook logs that state too
    rows, audit = rows[:-1], audit[:-1]
    names = list(rows[0]["rb_names"])
    model = build_model(a.urdf, a.meshes, a.obj_dir, names, a.collision)
    data = mujoco.MjData(model)
    mocap_of = {model.body(i).name: model.body_mocapid[i] for i in range(model.nbody) if model.body_mocapid[i] >= 0}
    weight = json.loads(Path(a.audit).read_text())["summary"]["object_weight_n"]
    frames = []
    cam = mujoco.MjvCamera()
    with mujoco.Renderer(model, height=a.h, width=a.w) as ren:
        for r, f in zip(rows, audit):
            rb = r["rb_states"]
            for i, n in enumerate(names):
                if n in mocap_of:
                    q = rb[i, 3:7]
                    data.mocap_pos[mocap_of[n]] = rb[i, :3]
                    data.mocap_quat[mocap_of[n]] = [q[3], q[0], q[1], q[2]]
            op = r["object_pose"]
            data.mocap_pos[mocap_of["object"]] = op[:3]
            data.mocap_quat[mocap_of["object"]] = [op[6], op[3], op[4], op[5]]
            mujoco.mj_forward(model, data)
            red = set(f["links"]) if f["pen_mm"] > 2.0 else set()
            for gi in range(model.ngeom):
                gname = model.geom(gi).name
                if gname.startswith("v_") and gname != "v_obj":
                    ln = gname[2:]
                    model.geom_rgba[gi] = [0.90, 0.15, 0.15, 1.0] if ln in red else [0.78, 0.83, 0.90, 1.0]
            tiles = []
            for az in (a.azim, a.azim + 90):
                cam.distance, cam.azimuth, cam.elevation = a.dist, az, -18.0
                cam.lookat[:] = op[:3]
                ren.update_scene(data, camera=cam)
                tiles.append(Image.fromarray(ren.render()))
            im = Image.new("RGB", (2 * a.w, a.h + 48), "#101722")
            im.paste(tiles[0], (0, 48)); im.paste(tiles[1], (a.w, 48))
            dr = ImageDraw.Draw(im)
            bad = f["pen_mm"] > 2.0 or f["grip_touch_x"] > 40
            dr.text((10, 6), a.title or Path(a.log).stem, font=font(16, True), fill="#f2f4f8")
            dr.text((10, 27), f"frame {f['ref_ts']:3d} | pen {f['pen_mm']:5.2f} mm | {f['n_links']} links | "
                              f"grip {f['grip_touch_n']:5.1f} N ({f['grip_touch_x']:4.0f}x) | err {f['pos_err_m']*100:4.1f} cm",
                    font=font(12), fill="#ff6b6b" if bad else "#7ddc8a")
            dr.text((2 * a.w - 150, 6), "DexTrack policy", font=font(12), fill="#98adbf")
            dr.text((2 * a.w - 150, 24), "MuJoCo replay of Isaac poses", font=font(10), fill="#98adbf")
            frames.append(im)
    dur = [int(1000 / a.fps)] * len(frames); dur[0] = 1000; dur[-1] = 1500
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=False)
    picks = np.linspace(0, len(frames) - 1, 4).astype(int)
    sheet = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for k, i in enumerate(picks):
        sheet.paste(frames[i], (0, k * frames[0].height))
    sheet.save(str(Path(a.out).with_suffix(".preview.jpg")), quality=85)
    print(f"wrote {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
