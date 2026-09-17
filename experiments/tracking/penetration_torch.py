"""Batched hand-object penetration depth on the GPU, for use as a reward term.

The same geometry the audit uses offline, made cheap enough to evaluate every
control step for thousands of environments: each hand link's collision
primitive is represented by a fixed set of surface sample points in the link
frame; the object's collision geometry by the face planes of the convex parts
DexTrack loads (one hull per connected part of `coacd/decomposed.obj`). A
point is inside a convex part iff it is behind every face plane, and its depth
is the least-negative plane distance. Per environment we report the deepest
sampled point over all links and parts, in metres, plus the count of links
that touch (within `touch`).

Memory: envs x links*points x parts x max_faces floats. With 21 links x 32
points, 64 parts of at most 40 faces and 4096 envs that is 7e9 elements in
float32 if done naively, so parts are looped and faces are padded per part.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R


def link_sample_points(urdf_path, n_box=12, n_sph=12):
    """{link name: (P, 3) float32 points in the link frame} from URDF collisions."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(x) for x in o.get("xyz").split()]) if o is not None and o.get("xyz") else np.zeros(3)
            rpy = np.array([float(x) for x in o.get("rpy").split()]) if o is not None and o.get("rpy") else np.zeros(3)
            Rm = R.from_euler("xyz", rpy).as_matrix()
            g = col.find("geometry")
            if g is None:
                continue
            if g.find("box") is not None:
                s = np.array([float(x) for x in g.find("box").get("size").split()])
                m = trimesh.creation.box(extents=s)
                p, _ = trimesh.sample.sample_surface(m, n_box, seed=0)
                p = np.concatenate([np.asarray(p), m.vertices])          # corners matter most
            elif g.find("sphere") is not None:
                r = float(g.find("sphere").get("radius"))
                p = trimesh.creation.icosphere(subdivisions=1, radius=r).vertices[:n_sph * 3]
            elif g.find("cylinder") is not None:
                r = float(g.find("cylinder").get("radius")); h = float(g.find("cylinder").get("length"))
                p, _ = trimesh.sample.sample_surface(trimesh.creation.cylinder(radius=r, height=h), n_box, seed=0)
            else:
                continue
            pts.append(np.asarray(p) @ Rm.T + xyz)
        if pts:
            out[link.get("name")] = np.concatenate(pts).astype(np.float32)
    return out


def hull_planes(obj_dir, max_parts=64):
    """(parts, max_faces, 4) padded plane tensor n.x + d <= 0 inside; padding is
    a plane every point is behind, so it never blocks the inside test."""
    m = trimesh.load(str(Path(obj_dir) / "decomposed.obj"), force="mesh", process=False)
    parts = [p.convex_hull for p in m.split(only_watertight=False) if len(p.vertices) >= 4][:max_parts]
    planes = []
    for h in parts:
        n = h.face_normals
        d = -np.einsum("ij,ij->i", n, h.triangles[:, 0])
        planes.append(np.concatenate([n, d[:, None]], 1))
    F = max(len(p) for p in planes)
    out = np.zeros((len(planes), F, 4), np.float32)
    out[:, :, 3] = -1e3                       # padding: n = 0, d = -1000 -> always "behind"
    for i, p in enumerate(planes):
        out[i, :len(p)] = p
    vol = float(sum(h.volume for h in parts))
    return torch.from_numpy(out), vol


def quat_apply(q, v):
    """Rotate vectors v (..., 3) by quaternions q (..., 4), xyzw, same leading shape."""
    qv = q[..., :3]
    w = q[..., 3:4]
    uv = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v + 2 * (w * uv + uuv)


def quat_inv_apply(q, v):
    qi = q * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=q.device)
    return quat_apply(qi, v)


class PenetrationProbe:
    """Holds link points and object planes on the device; call per step."""

    def __init__(self, urdf_path, obj_dir, body_names, device, touch=0.001):
        pts = link_sample_points(urdf_path)
        self.link_idx = [i for i, n in enumerate(body_names) if n in pts]
        P = max(len(pts[body_names[i]]) for i in self.link_idx)
        arr = np.zeros((len(self.link_idx), P, 3), np.float32)
        mask = np.zeros((len(self.link_idx), P), bool)
        for k, i in enumerate(self.link_idx):
            p = pts[body_names[i]]
            arr[k, :len(p)] = p; mask[k, :len(p)] = True
        self.points = torch.from_numpy(arr).to(device)            # (L, P, 3)
        self.pmask = torch.from_numpy(mask).to(device)            # (L, P)
        planes, self.volume = hull_planes(obj_dir)
        self.planes = planes.to(device)                            # (K, F, 4)
        # bounding spheres per hull, for culling: only points within radius +
        # margin of a hull centre are tested against its planes
        m = trimesh.load(str(Path(obj_dir) / "decomposed.obj"), force="mesh", process=False)
        parts = [p.convex_hull for p in m.split(only_watertight=False) if len(p.vertices) >= 4][:planes.shape[0]]
        c = np.stack([h.bounding_sphere.primitive.center for h in parts]).astype(np.float32)
        r = np.array([h.bounding_sphere.primitive.radius for h in parts], np.float32)
        self.centers = torch.from_numpy(c).to(device)              # (K, 3)
        self.radii = torch.from_numpy(r).to(device) + 0.01          # (K,) + 1 cm margin
        self.touch = touch
        self.device = device

    @torch.no_grad()
    def __call__(self, rb_states, obj_pose):
        """rb_states: (N, B, 13) rigid body states of the hand's actor; obj_pose:
        (N, 7) xyz + xyzw. Returns depth (N,) in metres (0 if nothing inside),
        n_touch (N,) links within `touch`, and per-link depth (N, L)."""
        N = rb_states.shape[0]
        lp = rb_states[:, self.link_idx, :3]                       # (N, L, 3)
        lq = rb_states[:, self.link_idx, 3:7]                      # (N, L, 4)
        pts = quat_apply(lq.unsqueeze(2).expand(-1, -1, self.points.shape[1], -1).reshape(N, -1, 4),
                         self.points.unsqueeze(0).expand(N, -1, -1, -1).reshape(N, -1, 3)) \
            + lp.unsqueeze(2).expand(-1, -1, self.points.shape[1], -1).reshape(N, -1, 3)
        pts = quat_inv_apply(obj_pose[:, 3:7].unsqueeze(1).expand(-1, pts.shape[1], -1),
                             pts - obj_pose[:, :3].unsqueeze(1))   # into the object frame
        # cull: which (env, point, hull) triples are within the hull's bounding
        # sphere. Then test planes only for hulls that have any candidate in
        # any env, restricted to the envs where they do.
        K = self.planes.shape[0]
        Q = pts.shape[1]
        d2 = ((pts.unsqueeze(2) - self.centers.view(1, 1, K, 3)) ** 2).sum(-1)   # (N, Q, K)
        cand = d2 <= (self.radii ** 2).view(1, 1, K)
        best = torch.full((N, Q), -1e3, device=self.device)
        hull_hit = cand.any(1)                                      # (N, K)
        for k in torch.nonzero(hull_hit.any(0)).flatten().tolist():
            envs = torch.nonzero(hull_hit[:, k]).flatten()
            pl = self.planes[k]
            s = pts[envs] @ pl[:, :3].T + pl[:, 3]                  # (n, Q, F)
            depth = -s.max(-1).values                               # (n, Q)
            depth = torch.where(cand[envs, :, k], depth, torch.full_like(depth, -1e3))
            best[envs] = torch.maximum(best[envs], depth)
        best = best.view(N, len(self.link_idx), -1)
        best = torch.where(self.pmask.unsqueeze(0), best, torch.full_like(best, -1e3))
        per_link = best.max(-1).values                             # (N, L), >0 = inside
        depth = per_link.max(-1).values.clamp(min=0.0)
        n_touch = (per_link > -self.touch).sum(-1)
        return depth, n_touch, per_link
