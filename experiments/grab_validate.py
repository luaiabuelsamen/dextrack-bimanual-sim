"""Validate the reference pipeline against GRAB's OWN contact labels.

Every check this pipeline had before was self-referential: it measured the
reconstructed hand against the reconstructed object and asked whether they were
close. They always were, and they were close in the wrong place. The mug in
`mug_drink_1` is grasped through its HANDLE -- GRAB's labels put 41-100% of its
contacts there -- while the reconstruction had the hand wrapped round the body,
0-18% on the handle, and passed a 0.1 mm minimum-distance check the whole time.

GRAB ships `contact['object']`: per frame, per object vertex, which body part
touches it. That is an external ground truth for the one thing the pipeline must
get right, and it is what this script scores against:

    recall  fraction of GRAB's contacted vertices that the reconstruction also
            marks as contacted -- the number that matters, because a miss here
            means the hand is in the wrong place
    IoU     agreement on the contact SET, which additionally penalises a hand
            that smears contact over the whole object

The bug this was written for: GRAB's ObjectModel computes `matmul(v, R)`, which
is `R.T @ v`, the transpose of the usual convention. Under the wrong one, recall
was 0.158; under the right one, 0.847.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from oppdef import paths
from oppdef.human import grab

NEAR_M = 0.005          # a hand vertex this close counts as touching


def score(seq_path, stride=8, near=NEAR_M, limit_frames=14):
    raw = np.load(paths.GRAB / "grab" / seq_path, allow_pickle=True)
    co = np.asarray(raw["contact"].item()["object"])
    s = grab.load(seq_path, verts=True, stride=stride)
    v, _f = s.obj_mesh

    ious, recs, ns = [], [], 0
    ks = np.linspace(0, s.T - 1, limit_frames).astype(int)
    for t in ks:
        k = int(t) * stride
        if k >= len(co):
            continue
        gt = co[k] > 0
        if gt.sum() == 0:
            continue
        for side in ("rhand", "lhand"):
            h = s.hands.get(side)
            if h is None or h.verts is None:
                continue
            d, _ = cKDTree(h.verts[t]).query(s.object_world(int(t)), k=1)
            mine = d < near
            if mine.sum() == 0:
                continue
            inter = int((gt & mine).sum())
            ious.append(inter / max(int((gt | mine).sum()), 1))
            recs.append(inter / max(int(gt.sum()), 1))
        ns += 1
    if not recs:
        return None
    return {"seq": s.name, "subject": s.subject, "object": s.obj,
            "intent": s.intent, "frames": ns,
            "recall": float(np.mean(recs)), "iou": float(np.mean(ious))}


def main(subjects, limit, stride, out):
    rows = []
    for subj in subjects:
        for p in grab.sequences(subj)[:limit]:
            try:
                r = score(f"{subj}/{p.name}", stride=stride)
            except Exception as e:                        # noqa: BLE001
                print(f"{p.stem}: FAILED {e!r}", flush=True)
                continue
            if r is None:
                continue
            rows.append(r)
            print(f"{r['subject']}/{r['seq']:30s} {r['object']:12s} "
                  f"recall {r['recall']:.3f}  IoU {r['iou']:.3f}", flush=True)
    if rows:
        rec = np.array([r["recall"] for r in rows])
        iou = np.array([r["iou"] for r in rows])
        print(f"\n{len(rows)} sequences")
        print(f"  recall  mean {rec.mean():.3f}  median {np.median(rec):.3f}  "
              f"min {rec.min():.3f}   >=0.5: {(rec >= 0.5).sum()}/{len(rec)}")
        print(f"  IoU     mean {iou.mean():.3f}  median {np.median(iou):.3f}")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(
            {"near_m": NEAR_M, "stride": stride, "rows": rows}, indent=1))
        print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="s1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="results/grab_validation.json")
    a = ap.parse_args()
    main(a.subjects.split(","), a.limit, a.stride, a.out)
