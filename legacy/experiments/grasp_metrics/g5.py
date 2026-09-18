"""G5 -- does a retargeted human grasp hold the real object under gravity?

See docs/G5_PREREGISTRATION.md. Written after the pre-registration, which fixes
the hypotheses, the sampling and the decision rule.

One scene is compiled per (hand, object) and reused across every sequence that
uses that object: compiling the model and loading the convex decomposition cost
far more than a 1 s rollout, and there are only 51 objects behind 291 sequences.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from handsim import paths
from handsim.human import grab, scene, track
from handsim.human.decompose import cache_dir
from handsim.human.retarget import retarget_sequence

N_FRAMES = 10          # per sequence, evenly spaced across the hold window
MIN_HOLD = 20          # frames; shorter windows are not a grasp
HOLD_S = 1.0
SETTLE_S = 0.25
DROP_M = 0.05


def cached_objects() -> set[str]:
    return {p.name.rsplit("_", 1)[0] for p in cache_dir().glob("*.npz")}


def run(hands, subjects, stride, limit, out_path):
    inv = json.loads(Path("legacy/results/grab_inventory.json").read_text())
    have = cached_objects()
    rows = [r for r in inv["rows"]
            if "error" not in r and r["rhand_hold_len"] >= MIN_HOLD
            and r["object"] in have and r["subject"] in subjects]
    rows.sort(key=lambda r: (r["object"], r["seq"]))
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} sequences, {len({r['object'] for r in rows})} objects")

    out = []
    scenes: dict[tuple, object] = {}
    for hand in hands:
        for i, r in enumerate(rows):
            t0 = time.time()
            try:
                s = grab.load(f"{r['subject']}/{r['seq']}.npz", verts=True,
                              stride=stride)
                key_s = (hand, s.obj, True)
                key_d = (hand, s.obj, False)
                if key_s not in scenes:
                    scenes[key_s] = scene.build(hand, s.obj, obj_static=True)
                    scenes[key_d] = scene.build(hand, s.obj, obj_static=False)
                st, ln = r["rhand_hold_start"], r["rhand_hold_len"]
                tr = retarget_sequence(s, "rhand", hand, window=(st, ln),
                                       sc=scenes[key_s])
                env = track.GrabTrackEnv(s, hand, sc=scenes[key_d])

                picks = np.linspace(0, len(tr.frames) - 1, N_FRAMES).astype(int)
                for t in picks:
                    h = env.hold(tr.q[t], seconds=HOLD_S, settle=SETTLE_S)
                    out.append({
                        "hand": hand, "seq": s.name, "subject": s.subject,
                        "object": s.obj, "intent": s.intent,
                        "frame": int(t), "window_len": int(len(tr.frames)),
                        "window_frac": float(t / max(len(tr.frames) - 1, 1)),
                        "held": bool(h.held), "drop_m": float(h.drop_m),
                        "settle_m": float(h.settle_m),
                        "n_contact": int(h.n_contact), "pen_mm": float(h.pen_mm),
                        "tip_err_mm": float(tr.tip_err[t] * 1000),
                        "contact_err_mm": float(np.nan_to_num(
                            tr.contact_err[t], nan=-1.0) * 1000),
                        "n_tip_contact": int(tr.n_contact[t]),
                    })
                held = sum(o["held"] for o in out[-N_FRAMES:])
                print(f"[{hand} {i+1}/{len(rows)}] {s.name:32s} {s.obj:12s} "
                      f"held {held}/{N_FRAMES}  ({time.time()-t0:.1f}s)",
                      flush=True)
            except Exception as e:                        # noqa: BLE001
                print(f"[{hand} {i+1}/{len(rows)}] {r['seq']}: FAILED {e!r}",
                      flush=True)
                out.append({"hand": hand, "seq": r["seq"], "error": repr(e)})

            if (i + 1) % 10 == 0:
                Path(out_path).write_text(json.dumps(
                    {"config": CONFIG, "rows": out}, indent=1))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"config": CONFIG, "rows": out}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="shadow")
    ap.add_argument("--subjects", default="s1,s2")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="legacy/results/g5_hold.json")
    a = ap.parse_args()

    CONFIG = dict(n_frames=N_FRAMES, min_hold=MIN_HOLD, hold_s=HOLD_S,
                  settle_s=SETTLE_S, drop_m=DROP_M, stride=a.stride,
                  hands=a.hands.split(","), subjects=a.subjects.split(","))
    rows = run(a.hands.split(","), set(a.subjects.split(",")), a.stride,
               a.limit, a.out)
    ok = [r for r in rows if "error" not in r]
    print(f"\n{len(ok)} observations, "
          f"hold rate {100*np.mean([r['held'] for r in ok]):.1f}%")
    print(f"wrote {a.out}")
