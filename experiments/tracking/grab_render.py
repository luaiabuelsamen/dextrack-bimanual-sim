"""Render a GRAB sequence: human hands + object, as a GIF.

The point is to look at the reference before building anything on top of it.
Every serious error in this repository so far was found by rendering a frame.

The projection is done by hand rather than with mpl_toolkits.mplot3d: this
machine has a stale system copy of that package shadowing the installed one,
and a painter's-algorithm triangle sort gives correct occlusion, which
mplot3d's per-collection z-ordering does not.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.collections import PolyCollection, LineCollection

from handsim.human import grab

# MANO chains for drawing: wrist -> proximal -> middle -> distal -> tip vertex
CHAINS = [[0, 13, 14, 15, 16], [0, 1, 2, 3, 17], [0, 4, 5, 6, 18],
          [0, 10, 11, 12, 19], [0, 7, 8, 9, 20]]
COLOR = {"rhand": "#d62728", "lhand": "#1f77b4"}
HAND_FACES = None      # filled from the MANO model on first render


def camera(elev_deg: float, azim_deg: float) -> np.ndarray:
    """World -> camera basis; rows are (right, up, forward)."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.stack([right, up, fwd])


def project(P: np.ndarray, M: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    """(N,3) world -> (N,3) screen, third column is depth (larger = nearer)."""
    return (np.asarray(P) - ctr) @ M.T


def render(seq, out: Path, face_stride: int = 6, fps: int = 10, spin: float = 0.4,
           mesh: bool = True):
    """Draw the hands as their MANO SURFACE, not as a stick skeleton.

    A skeleton cannot show whether a hand is curled: five polylines radiating
    from a wrist look like a flat fan whether the fingers are wrapped round a
    mug or splayed flat, and reading one as the other is exactly the mistake a
    picture is supposed to prevent. The mesh shows the grasp.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=95)
    ov, of = seq.obj_mesh
    of = of[::face_stride]

    # Frame on the object, not on the whole reach: the reach spans ~1.5 m
    # while the grasp -- the only part that carries information -- spans ~0.2 m,
    # so a global bounding box renders the interesting part a few pixels wide.
    obj_rad = float(np.abs(ov).max())
    rad = max(0.16, obj_rad * 1.6)

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out), dpi=95):
        for k in range(seq.T):
            ax.clear()
            M = camera(16.0, -60.0 + spin * k)
            ctr = seq.obj_pos[k]

            w = project(seq.object_world(k), M, ctr)
            tri = w[of]                                   # (F, 3, 3)
            order = np.argsort(tri[:, :, 2].mean(1))      # far -> near
            tri = tri[order]
            # cheap lambertian from the triangle normal
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = np.clip(0.45 + 0.55 * np.abs(n[:, 2]), 0, 1)
            ax.add_collection(PolyCollection(
                tri[:, :, :2], facecolors=plt.cm.gray(shade),
                edgecolors="none", zorder=1))

            for side, h in seq.hands.items():
                if mesh and h.verts is not None:
                    hv = project(h.verts[k], M, ctr)
                    hf = HAND_FACES
                    ht = hv[hf]
                    ho = np.argsort(ht[:, :, 2].mean(1))
                    ht = ht[ho]
                    hn = np.cross(ht[:, 1] - ht[:, 0], ht[:, 2] - ht[:, 0])
                    hn /= np.linalg.norm(hn, axis=1, keepdims=True) + 1e-12
                    sh = np.clip(0.35 + 0.65 * np.abs(hn[:, 2]), 0, 1)
                    cmap = (plt.cm.Reds if side == "rhand" else plt.cm.Blues)
                    ax.add_collection(PolyCollection(
                        ht[:, :, :2], facecolors=cmap(0.35 + 0.45 * sh),
                        edgecolors="none", zorder=5))
                else:
                    J = project(h.joints[k], M, ctr)
                    segs = [J[c[i:i + 2], :2] for c in CHAINS for i in range(4)]
                    ax.add_collection(LineCollection(
                        segs, colors=COLOR[side], linewidths=2.0, zorder=3))
                    ax.scatter(J[:, 0], J[:, 1], s=11, c=COLOR[side], zorder=4,
                               edgecolors="white", linewidths=0.4)

            ax.set_xlim(-rad, rad); ax.set_ylim(-rad, rad)
            ax.set_aspect("equal"); ax.set_axis_off()
            ax.set_title(f"GRAB  {seq.name}   ({seq.subject} · {seq.obj} · "
                         f"{seq.intent})   frame {k+1}/{seq.T}", fontsize=9)
            writer.grab_frame()
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="s1/mug_drink_1.npz")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--faces", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = grab.load(a.seq, stride=a.stride, verts=True)
    from handsim.human import mano as _mano
    globals()["HAND_FACES"] = _mano.load("right").faces
    out = Path(a.out or f"figures/grab_{s.name}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    render(s, out, face_stride=a.faces)
    print(f"wrote {out}  ({s.T} frames, {out.stat().st_size/1e6:.1f} MB)")
