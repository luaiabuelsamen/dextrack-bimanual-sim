"""Burial audit of a DexTrack rollout, from the per-step log the AUDIT_LOG hook writes.

The hook (a 30-line env-var-gated addition to DexTrack's task, see
docs/STAGE2.md) records, for environment 0 at every control step: the object
pose, every hand rigid body's pose, and PhysX's net contact force on every
rigid body. Penetration is then measured GEOMETRICALLY here, with the same
shapes PhysX collided:

  hand   the URDF's collision primitives (boxes and spheres on the Allegro
         links, a box on the palm), sampled densely on their surfaces and
         placed by the logged body poses;
  object the convex decomposition DexTrack loads (`coacd/decomposed.obj`),
         one convex hull per connected part, placed by the logged object pose.

penetration = the deepest sampled hand-surface point inside any convex part
(signed distance into the hull, metres). A link "touches" if any of its
sampled points is within 1 mm of a part. Grip is the sum over hand links of
|net contact force| from PhysX, live, which double-counts nothing because it
is per body, not per contact, and is therefore comparable to our `total_grip`
only in order of magnitude. Both are reported per frame, and the verdict
columns use the README renderer's thresholds: red past 2 mm or 40x weight.

Object mass is what DexTrack sets: density 500 kg/m^3 times the hull volume.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def _rpy_xyz(el):
    o = el.find("origin")
    xyz = np.zeros(3); rpy = np.zeros(3)
    if o is not None:
        if o.get("xyz"): xyz = np.array([float(x) for x in o.get("xyz").split()])
        if o.get("rpy"): rpy = np.array([float(x) for x in o.get("rpy").split()])
    return xyz, R.from_euler("xyz", rpy).as_matrix()


def hand_primitives(urdf_path, n_box=400, n_sph=200):
    """{link name: [(points in link frame, label)]} from the URDF collisions."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        pts_all = []
        for col in link.findall("collision"):
            xyz, Rm = _rpy_xyz(col)
            g = col.find("geometry")
            if g is None:
                continue
            if g.find("box") is not None:
                s = np.array([float(x) for x in g.find("box").get("size").split()])
                m = trimesh.creation.box(extents=s)
                p, _ = trimesh.sample.sample_surface(m, n_box)
            elif g.find("sphere") is not None:
                r = float(g.find("sphere").get("radius"))
                m = trimesh.creation.icosphere(subdivisions=2, radius=r)
                p = m.vertices
            elif g.find("cylinder") is not None:
                r = float(g.find("cylinder").get("radius")); h = float(g.find("cylinder").get("length"))
                m = trimesh.creation.cylinder(radius=r, height=h)
                p, _ = trimesh.sample.sample_surface(m, n_box)
            elif g.find("mesh") is not None:
                fn = (Path(urdf_path).parent / g.find("mesh").get("filename")).resolve()
                m = trimesh.load(str(fn), force="mesh")
                sc = g.find("mesh").get("scale")
                if sc:
                    m.apply_scale([float(x) for x in sc.split()])
                p, _ = trimesh.sample.sample_surface(m, n_box)
            else:
                continue
            pts_all.append(np.asarray(p) @ Rm.T + xyz)
        if pts_all:
            out[link.get("name")] = np.concatenate(pts_all)
    return out


def object_hulls(obj_dir):
    """Convex hulls of the parts of DexTrack's decomposition, plus total volume."""
    m = trimesh.load(str(Path(obj_dir) / "decomposed.obj"), force="mesh", process=False)
    parts = m.split(only_watertight=False)
    hulls = [p.convex_hull for p in parts if len(p.vertices) >= 4]
    vol = float(sum(h.volume for h in hulls))
    return hulls, vol


def depth_into_hulls(points, hulls):
    """Max signed penetration (m) of points into any hull, and which are inside."""
    best = np.full(len(points), -np.inf)
    for h in hulls:
        # distance to each face plane; inside a convex hull iff all <= 0, and the
        # depth is the least-negative plane distance
        n, d = h.face_normals, -np.einsum("ij,ij->i", h.face_normals, h.triangles[:, 0])
        s = points @ n.T + d                       # (P, F), >0 outside that plane
        inside = np.all(s <= 1e-6, axis=1)
        depth = np.where(inside, -np.max(s, axis=1), -np.inf)
        best = np.maximum(best, depth)
    return best


def quat_xyzw_to_R(q):
    return R.from_quat(q).as_matrix()          # scipy takes xyzw, as Isaac Gym stores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="AUDIT_LOG .npy from the hook")
    ap.add_argument("--hand-urdf", required=True)
    ap.add_argument("--obj-dir", required=True, help=".../sem/<inst>/coacd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--density", type=float, default=500.0)
    ap.add_argument("--touch-mm", type=float, default=1.0)
    a = ap.parse_args()

    rows = np.load(a.log, allow_pickle=True)
    prims = hand_primitives(a.hand_urdf)
    hulls, vol = object_hulls(a.obj_dir)
    weight = a.density * vol * 9.81
    names = list(rows[0]["rb_names"])
    hand_idx = [i for i, n in enumerate(names) if n in prims]
    out = []
    for r in rows:
        op, oq = r["object_pose"][:3], r["object_pose"][3:7]
        Ro = quat_xyzw_to_R(oq)
        pen = 0.0; links = []; n_in = 0
        for i in hand_idx:
            st = r["rb_states"][i]
            p, q = st[:3], st[3:7]
            pts_w = prims[names[i]] @ quat_xyzw_to_R(q).T + p
            pts_o = (pts_w - op) @ Ro                 # into the object frame
            d = depth_into_hulls(pts_o, hulls)
            m = float(np.max(d))
            if m > -a.touch_mm / 1000:
                links.append(names[i]); n_in += int(np.sum(d > 0))
            pen = max(pen, m)
        cf = r["net_contact_force"]
        grip = float(np.sum(np.linalg.norm(cf[hand_idx], axis=1)))
        # Only links that geometrically touch the object. The hand collides
        # with itself at its reference pose (palm against thumb, index
        # against middle) and those pairs carry 10-70 N whether or not the
        # object is anywhere near, so the all-links sum overstates the squeeze.
        touch_idx = [i for i in hand_idx if names[i] in links]
        grip_touch = float(np.sum(np.linalg.norm(cf[touch_idx], axis=1))) if touch_idx else 0.0
        gp, gq = r.get("goal_pos"), r.get("goal_rot")
        err = float(np.linalg.norm(op - gp)) if gp is not None else float("nan")
        rot = float("nan")
        if gq is not None:
            dq = R.from_quat(oq) * R.from_quat(gq).inv()
            rot = float(dq.magnitude())
        out.append({"ref_ts": int(r["ref_ts"]), "pen_mm": pen * 1000,
                    "links": links, "n_links": len(links), "points_inside": n_in,
                    "grip_n": grip, "grip_x": grip / weight if weight else float("nan"),
                    "grip_touch_n": grip_touch, "grip_touch_x": grip_touch / weight if weight else float("nan"),
                    "pos_err_m": err, "rot_err_rad": rot})
    pen = np.array([o["pen_mm"] for o in out]); gx = np.array([o["grip_x"] for o in out])
    gtx = np.array([o["grip_touch_x"] for o in out])
    err = np.array([o["pos_err_m"] for o in out]); rot = np.array([o["rot_err_rad"] for o in out])
    summ = {
        "frames": len(out), "object_weight_n": weight, "hull_parts": len(hulls),
        "pen_mm_mean": float(pen.mean()), "pen_mm_max": float(pen.max()),
        "pen_mm_median": float(np.median(pen)),
        "frames_over_2mm": int(np.sum(pen > 2.0)), "frames_over_5mm": int(np.sum(pen > 5.0)),
        "grip_x_mean": float(np.nanmean(gx)), "grip_x_max": float(np.nanmax(gx)),
        "frames_over_40x": int(np.nansum(gx > 40)),
        "grip_touch_x_mean": float(np.nanmean(gtx)), "grip_touch_x_median": float(np.nanmedian(gtx)),
        "grip_touch_x_max": float(np.nanmax(gtx)), "frames_touch_over_40x": int(np.nansum(gtx > 40)),
        "frames_touching": int(np.sum([o["n_links"] > 0 for o in out])),
        "frames_red": int(np.sum((pen > 2.0) | (gx > 40))),
        "pos_err_cm_mean": float(np.nanmean(err) * 100), "rot_err_deg_mean": float(np.degrees(np.nanmean(rot))),
        "dextrack_obj_strict": bool(np.nanmean(err) <= 0.10 and np.nanmean(rot) <= np.radians(20)),
        "dextrack_obj_loose": bool(np.nanmean(err) <= 0.10 and np.nanmean(rot) <= np.radians(40)),
        "end": out[-1],
    }
    Path(a.out).write_text(json.dumps({"log": a.log, "summary": summ, "frames": out}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
