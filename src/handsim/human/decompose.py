"""Convex decomposition of the GRAB object meshes.

MuJoCo collides a mesh geom as its CONVEX HULL.  For GRAB that is not a small
approximation: the mug's hull is 3.52x the mug's own volume and the airplane's
is 3.51x, because the hull fills the cup's cavity and the handle's hole.  Every
grasp GRAB records through a handle would be physically impossible against the
hull, and a hand correctly placed by retargeting reads as 20-40 mm of
penetration that is not there.

So each object is decomposed into convex parts once, cached, and added to the
scene as several mesh geoms in one body -- the approach DexGraspNet and
DexGraspBench both take.  Nearly convex objects (the apple, at 1.05x) come back
as a single part and cost nothing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from handsim import paths

#: CoACD's concavity threshold. Lower means more parts and a tighter fit; 0.05
#: is CoACD's own default and keeps these objects in the 4-30 part range, which
#: MuJoCo handles without the contact count exploding.
THRESHOLD = 0.05
MAX_HULL_VERTS = 64          # per part, after decomposition


def cache_dir() -> Path:
    d = paths.DATA / "grab_convex"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mesh_path(obj: str) -> Path:
    return paths.GRAB / "tools" / "object_meshes" / "contact_meshes" / f"{obj}.ply"


def _key(obj: str, threshold: float) -> str:
    h = hashlib.sha256()
    h.update(mesh_path(obj).read_bytes())
    h.update(f"{threshold}:{MAX_HULL_VERTS}".encode())
    return h.hexdigest()[:16]


def mesh_volume(v: np.ndarray, f: np.ndarray) -> float:
    t = v[f]
    return float(abs(np.einsum("ij,ij->i", t[:, 0],
                               np.cross(t[:, 1], t[:, 2])).sum() / 6.0))


def decompose(obj: str, threshold: float = THRESHOLD,
              force: bool = False) -> list[np.ndarray]:
    """Convex parts of `obj`, as a list of (n_i, 3) vertex arrays.

    Cached on the mesh's own hash, so changing the threshold or the mesh
    invalidates it rather than silently reusing a stale decomposition.
    """
    from handsim.human import grab as grab_mod

    out = cache_dir() / f"{obj}_{_key(obj, threshold)}.npz"
    if out.exists() and not force:
        z = np.load(out)
        return [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]

    import coacd
    coacd.set_log_level("error")
    v, f = grab_mod._ply(mesh_path(obj))
    parts = coacd.run_coacd(coacd.Mesh(v, f), threshold=threshold)

    verts = []
    for pv, _pf in parts:
        pv = np.asarray(pv, float)
        if len(pv) > MAX_HULL_VERTS:
            from scipy.spatial import ConvexHull
            idx = np.unique(ConvexHull(pv).vertices)
            if len(idx) > MAX_HULL_VERTS:
                idx = idx[np.linspace(0, len(idx) - 1, MAX_HULL_VERTS).astype(int)]
            pv = pv[idx]
        verts.append(pv)

    np.savez_compressed(out, **{f"part_{i}": p for i, p in enumerate(verts)})
    return verts


def report(obj: str, threshold: float = THRESHOLD) -> dict:
    """How much of the hull's overreach the decomposition actually removes."""
    from scipy.spatial import ConvexHull
    from handsim.human import grab as grab_mod

    v, f = grab_mod._ply(mesh_path(obj))
    parts = decompose(obj, threshold)
    vol = mesh_volume(v, f)
    hull = ConvexHull(v).volume
    # parts overlap slightly, so this is an upper bound on the covered volume
    pv = sum(ConvexHull(p).volume for p in parts if len(p) >= 4)
    return {"object": obj, "verts": int(len(v)), "parts": len(parts),
            "mesh_cm3": round(vol * 1e6, 1), "hull_cm3": round(hull * 1e6, 1),
            "parts_cm3": round(pv * 1e6, 1),
            "hull_ratio": round(hull / vol, 2),
            "parts_ratio": round(pv / vol, 2)}
