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


def _box_grid(half, spacing):
    """Every face of a box gridded at `spacing` (corners and edges included)."""
    axes = [np.linspace(-half[i], half[i], max(2, int(np.ceil(2 * half[i] / spacing)) + 1)) for i in range(3)]
    p = []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        A, B = np.meshgrid(axes[a], axes[b], indexing="ij")
        for sign in (-1, 1):
            q = np.zeros((A.size, 3)); q[:, a] = A.ravel(); q[:, b] = B.ravel(); q[:, ax] = sign * half[ax]
            p.append(q)
    return np.concatenate(p)


def link_sample_points(urdf_path, n_box=12, n_sph=12, spacing=0.0):
    """{link name: (P, 3) float32 points in the link frame} from URDF collisions.

    spacing == 0: the sparse set (12 random surface points + 8 corners per
    box, 36 icosphere vertices per sphere; ~20 points per link). spacing > 0:
    a deterministic grid over every box face at that spacing (metres) and a
    162-vertex icosphere per sphere.

    The sparse set is NOT safe as a training signal. The depth of a convex
    link into a convex part is the maximum of a concave function over the
    link's surface, so it can sit in the interior of a face or an edge, away
    from every corner. A policy rewarded on the sparse set learns to keep the
    sampled points shallow while the surface between them goes deeper: on the
    cube, fine-tunes that halved the sparse depth raised the dense depth
    (docs/STAGE2.md). Use spacing ~2 mm for anything a policy optimises."""
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
                if spacing > 0:
                    p = _box_grid(s / 2, spacing)
                else:
                    m = trimesh.creation.box(extents=s)
                    p, _ = trimesh.sample.sample_surface(m, n_box, seed=0)
                    p = np.concatenate([np.asarray(p), m.vertices])      # corners matter most
            elif g.find("sphere") is not None:
                r = float(g.find("sphere").get("radius"))
                if spacing > 0:
                    p = trimesh.creation.icosphere(subdivisions=2, radius=r).vertices
                else:
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


def quat_to_mat(q):
    """(..., 4) xyzw -> (..., 3, 3) rotation matrices."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def quat_inv_apply(q, v):
    qi = q * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=q.device)
    return quat_apply(qi, v)


class PenetrationProbe:
    """Holds link points and object planes on the device; call per step."""

    def __init__(self, urdf_path, obj_dir, body_names, device, touch=0.001, spacing=0.0, sdf_res=0.0, sdf_margin=0.03):
        """spacing: hand-surface sample grid (0 = sparse set). sdf_res > 0:
        precompute the object's signed depth on a voxel grid of that
        resolution (metres, object frame, `sdf_margin` around the hulls) and
        read every sampled point by trilinear interpolation instead of
        testing it against hull planes. Same convention (positive inside, the
        least plane violation), same result to interpolation error, and the
        per-step cost no longer depends on the number of hulls: the exact
        plane test with the 2 mm grid takes 9 s per step at 1024 envs on a
        64-hull object, the volume a few ms."""
        pts = link_sample_points(urdf_path, spacing=spacing)
        self.spacing = spacing
        self.n_points = int(sum(len(v) for v in pts.values()))
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
        self.sdf = None
        if sdf_res > 0:
            self._build_sdf(parts, sdf_res, sdf_margin)

    def _build_sdf(self, parts, res, margin):
        lo = np.min([h.vertices.min(0) for h in parts], 0) - margin
        hi = np.max([h.vertices.max(0) for h in parts], 0) + margin
        n = np.ceil((hi - lo) / res).astype(int) + 1
        axes = [torch.linspace(float(lo[i]), float(lo[i] + (n[i] - 1) * res), int(n[i]), device=self.device) for i in range(3)]
        Z, Y, X = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")          # volume indexed [z, y, x]
        G = torch.stack([X, Y, Z], -1).reshape(-1, 3)
        depth = torch.full((G.shape[0],), -1e3, device=self.device)
        for k in range(self.planes.shape[0]):
            pl = self.planes[k]
            for i in range(0, G.shape[0], 1 << 20):
                s = G[i:i + (1 << 20)] @ pl[:, :3].T + pl[:, 3]
                depth[i:i + (1 << 20)] = torch.maximum(depth[i:i + (1 << 20)], -s.max(-1).values)
        self.sdf = depth.view(int(n[2]), int(n[1]), int(n[0]))[None, None]         # (1, 1, D, H, W)
        self.sdf_lo = torch.tensor(lo, dtype=torch.float32, device=self.device)
        self.sdf_hi = torch.tensor(lo + (n - 1) * res, dtype=torch.float32, device=self.device)
        self.sdf_res = res
        allv = np.concatenate([h.vertices for h in parts])
        c = 0.5 * (allv.min(0) + allv.max(0))
        self.obj_center = torch.tensor(c, dtype=torch.float32, device=self.device)
        self.obj_radius = float(np.linalg.norm(allv - c, axis=1).max()) + 0.01

    def _sdf_depth(self, pts):
        """pts (N, Q, 3) in the object frame -> depth (N, Q), trilinear."""
        import torch.nn.functional as F
        g = 2 * (pts - self.sdf_lo) / (self.sdf_hi - self.sdf_lo) - 1                # (N, Q, 3) in [-1, 1], xyz order
        v = F.grid_sample(self.sdf, g.view(1, -1, 1, 1, 3), mode="bilinear", padding_mode="border", align_corners=True)
        return v.view(pts.shape[0], pts.shape[1])

    @torch.no_grad()
    def __call__(self, rb_states, obj_pose, chunk=None):
        """rb_states: (N, B, 13) rigid body states of the hand's actor; obj_pose:
        (N, 7) xyz + xyzw. Returns depth (N,) in metres (0 if nothing inside),
        n_touch (N,) links within `touch`, and per-link depth (N, L). Envs are
        processed `chunk` at a time: the dense point set is ~33k points per
        env, and the (envs, points, faces) tensor at 1024 envs would not fit
        beside a training run."""
        N = rb_states.shape[0]
        if chunk is None:      # keep the (envs, points, hulls) cull tensor near 2e8 elements
            Q = self.points.shape[0] * self.points.shape[1]
            chunk = max(1, min(int(2e8 // (Q * self.planes.shape[0])), int(2e7 // Q)))
        if N > chunk:
            outs = [self._call(rb_states[i:i + chunk], obj_pose[i:i + chunk]) for i in range(0, N, chunk)]
            return tuple(torch.cat([o[k] for o in outs], 0) for k in range(3))
        return self._call(rb_states, obj_pose)

    def _call(self, rb_states, obj_pose):
        N = rb_states.shape[0]
        lp = rb_states[:, self.link_idx, :3]                       # (N, L, 3)
        lq = rb_states[:, self.link_idx, 3:7]                      # (N, L, 4)
        # one object<-link transform per (env, link), then a single einsum
        # over the shared link point sets: the points are the bulk of the
        # work (33k per hand on the dense grid) and this touches each once
        Rl = quat_to_mat(lq)                                       # (N, L, 3, 3)
        Ro = quat_to_mat(obj_pose[:, 3:7])                         # (N, 3, 3)
        RoT = Ro.transpose(1, 2).unsqueeze(1)                      # (N, 1, 3, 3)
        R = RoT @ Rl                                               # (N, L, 3, 3)
        t = (RoT @ (lp - obj_pose[:, None, :3]).unsqueeze(-1)).squeeze(-1)   # (N, L, 3)
        pts = (torch.einsum("nlij,lpj->nlpi", R, self.points) + t.unsqueeze(2)).reshape(N, -1, 3)
        K = self.planes.shape[0]
        Q = pts.shape[1]
        if self.sdf is not None:
            # only points inside the object's bounding sphere (+ margin) are
            # looked up; the rest of the hand is far and reads -1e3
            flat = pts.reshape(-1, 3)
            near = ((flat - self.obj_center) ** 2).sum(-1) <= self.obj_radius ** 2
            idx = torch.nonzero(near).flatten()
            best = torch.full((N * Q,), -1e3, device=self.device)
            if idx.numel():
                best[idx] = self._sdf_depth(flat[idx].view(1, -1, 3)).reshape(-1)
            return self._finish(best, N)
        # cull: which (env, point, hull) triples are within the hull's bounding
        # sphere. Then test planes only for hulls that have any candidate in
        # any env, restricted to the envs where they do.
        d2 = ((pts.unsqueeze(2) - self.centers.view(1, 1, K, 3)) ** 2).sum(-1)   # (N, Q, K)
        cand = d2 <= (self.radii ** 2).view(1, 1, K)
        best = torch.full((N * Q,), -1e3, device=self.device)
        # only the candidate (env, point) pairs of each hull are tested, so the
        # work is proportional to how many points are near the object rather
        # than envs x points x hulls; with the dense grid on a 64-hull object
        # the full product would not fit
        for k in torch.nonzero(cand.any(1).any(0)).flatten().tolist():
            idx = torch.nonzero(cand[:, :, k].reshape(-1)).flatten()      # flat (env, point) ids
            pl = self.planes[k]
            s = pts.reshape(-1, 3)[idx] @ pl[:, :3].T + pl[:, 3]          # (n, F)
            best[idx] = torch.maximum(best[idx], -s.max(-1).values)
        return self._finish(best, N)

    def _finish(self, best, N):
        best = best.view(N, len(self.link_idx), -1)
        best = torch.where(self.pmask.unsqueeze(0), best, torch.full_like(best, -1e3))
        per_link = best.max(-1).values                             # (N, L), >0 = inside
        depth = per_link.max(-1).values.clamp(min=0.0)
        n_touch = (per_link > -self.touch).sum(-1)
        return depth, n_touch, per_link
