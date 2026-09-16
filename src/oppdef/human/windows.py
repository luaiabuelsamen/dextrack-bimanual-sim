"""Where a hand is in contact with the object, and for how long.

These three routines decide which frames of a GRAB sequence every later stage
operates on -- the retarget's window, the tracker's window, the frames a grasp
search is allowed to start from. That makes them library code, and they lived in
`experiments/tracking/grab_inventory.py` until `oppdef.human.track` was importing
them back out of `experiments/`. That inverts the dependency: the package worked
only because the Makefile puts the repo root on PYTHONPATH, and a plain
`pip install -e .` would have failed on it.

The experiment script now imports them from here, which is the direction the
dependency should run.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: 5 mm, which is GRAB's own contact threshold.
CONTACT_M = 0.005
#: A contact shorter than this is a brush, not a grasp.
MIN_FRAMES = 5


def object_tree(seq, cache: dict | None = None) -> cKDTree:
    """A KD-tree over the object's CANONICAL vertices, cached per object.

    Cached because the alternative -- rebuilding a 30k-point tree per frame --
    is about 200x slower and answers the same question. The tree is built in the
    object's own frame, so the hand is transformed into that frame rather than
    the mesh into the world's.
    """
    if cache is None:
        cache = {}
    if seq.obj not in cache:
        cache[seq.obj] = cKDTree(seq.obj_mesh[0])
    return cache[seq.obj]


def contact_mask(seq, tree, side: str, thresh: float = CONTACT_M) -> np.ndarray:
    """(T,) bool -- is this hand within `thresh` of the object surface."""
    h = seq.hands.get(side)
    if h is None or h.verts is None:
        return np.zeros(seq.T, bool)
    out = np.zeros(seq.T, bool)
    for k in range(seq.T):
        local = seq.to_object(h.verts[k], k)           # world -> object frame
        out[k] = tree.query(local, k=1)[0].min() < thresh
    return out


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest True run."""
    best = (0, 0)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i > best[1]:
                best = (i, j - i)
            i = j
        else:
            i += 1
    return best


#: Kept so the experiment scripts that spell it this way keep working.
_tree = object_tree
