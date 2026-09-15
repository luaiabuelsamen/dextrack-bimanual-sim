"""Survey GRAB: which sequences are usable references, and which are bimanual.

This runs before any retargeting because two facts decide the whole plan and
neither is documented in the GRAB release:

  1. how many sequences have BOTH hands on the object at once -- the bimanual
     goal is only as real as this number, and "bimanual dataset" in a paper
     abstract usually means "both hands were tracked", not "both hands were
     used";
  2. how long the object is actually held, which is the window a tracking
     controller is asked to reproduce.

Contact is measured in the OBJECT frame: the hand vertices are pushed through
the inverse object transform so one static KD-tree per object serves every
frame of every sequence using it.  Measuring in the world frame instead would
mean rebuilding a 30k-point tree per frame, which is ~200x slower and answers
the same question.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from oppdef import paths
from oppdef.human import grab

CONTACT_M = 0.005     # 5 mm: GRAB's own contact threshold
MIN_FRAMES = 5        # a contact shorter than this is a brush, not a grasp


def _tree(seq, cache: dict) -> cKDTree:
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


def survey(subjects, stride: int, limit: int | None, thresh: float):
    cache: dict = {}
    rows = []
    for subj in subjects:
        for p in grab.sequences(subj)[:limit]:
            t0 = time.time()
            try:
                s = grab.load(p, verts=True, stride=stride)
            except Exception as e:                       # noqa: BLE001
                rows.append({"seq": p.stem, "subject": subj, "error": repr(e)})
                continue
            tree = _tree(s, cache)
            rec = {
                "seq": s.name, "subject": s.subject, "object": s.obj,
                "intent": s.intent, "frames": s.T, "dt": s.dt,
                "duration_s": round(s.T * s.dt, 3),
            }
            masks = {}
            for side in ("rhand", "lhand"):
                m = contact_mask(s, tree, side, thresh)
                masks[side] = m
                st, ln = longest_run(m)
                rec[f"{side}_frames"] = int(m.sum())
                rec[f"{side}_hold_start"] = int(st)
                rec[f"{side}_hold_len"] = int(ln)
            both = masks["rhand"] & masks["lhand"]
            bst, bln = longest_run(both)
            rec["both_frames"] = int(both.sum())
            rec["both_hold_start"] = int(bst)
            rec["both_hold_len"] = int(bln)
            rec["bimanual"] = bool(bln >= MIN_FRAMES)
            # how far the object travels while held by either hand
            held = masks["rhand"] | masks["lhand"]
            rec["obj_travel_m"] = round(float(
                np.linalg.norm(np.ptp(s.obj_pos[held], axis=0))) if held.any() else 0.0, 4)
            rec["secs"] = round(time.time() - t0, 2)
            rows.append(rec)
            print(f"{s.subject}/{s.name:34s} {s.obj:14s} {s.intent:10s} "
                  f"T={s.T:4d}  R={rec['rhand_hold_len']:4d} "
                  f"L={rec['lhand_hold_len']:4d} BOTH={bln:4d} "
                  f"{'<< BIMANUAL' if rec['bimanual'] else ''}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="s1,s2")
    ap.add_argument("--stride", type=int, default=8)     # 120 Hz -> 15 Hz
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=CONTACT_M)
    ap.add_argument("--out", default="results/grab_inventory.json")
    a = ap.parse_args()

    rows = survey(a.subjects.split(","), a.stride, a.limit, a.thresh)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"contact_m": a.thresh, "stride": a.stride,
         "min_frames": MIN_FRAMES, "rows": rows}, indent=1))

    ok = [r for r in rows if "error" not in r]
    bi = [r for r in ok if r["bimanual"]]
    print(f"\n{len(ok)}/{len(rows)} sequences read")
    print(f"bimanual (both hands in contact >= {MIN_FRAMES} frames): "
          f"{len(bi)} ({100*len(bi)/max(len(ok),1):.1f}%)")
    print(f"wrote {out}")
