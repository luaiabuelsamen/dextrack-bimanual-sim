"""Stage 6: two-handed tracking on GRAB's bimanual references.

Runs each reference BOTH ways -- two hands, and the same reference with the
left hand removed -- because "the bimanual pipeline runs" is not a result. The
claim this project exists to test is that some manipulations need a second hand
for reasons of wrench geometry rather than payload, and that only shows up as a
gap between the two conditions on the SAME reference.

`hold_test` and the tracking rollout are reported separately. A two-handed
grasp that holds but tracks badly and one that never holds are different
failures, and stage 2 spent a long time learning that conflating them hides
which one you have.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

from handsim.human import grab, track as T


def one(row, hand_r="shadow", hand_l="shadow_left", stride=8, grip=8.0,
        seconds=0.8, samples=10, rounds=2, steps=None):
    s = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=stride)
    bt = T.BimanualTracker(s, hand_r=hand_r, hand_l=hand_l)
    # NOT frame 0. The window opens where the hands first come within 5 mm of
    # the object, which is the approach; started there the arm is still
    # sweeping through and the rollout diverges before anything is tracked.
    gf = bt.grasp_frames()
    if len(gf) == 0:
        return {"seq": row["seq"], "subject": row["subject"],
                "object": row["object"], "intent": row["intent"],
                "window": list(map(int, bt.window)), "no_grasp_frame": True}
    k = int(gf[0])

    raw_drop = bt.hold_test(k, seconds=seconds, grip=grip)
    bt.synthesize_grasp(k=k, samples=samples, rounds=rounds)
    drop = bt.hold_test(k, seconds=seconds, grip=grip)

    ro = bt.rollout(start=k, steps=steps)
    err = float(np.nanmean(ro.pos_err)) * 1000
    span = bt.T - k

    # The one-handed control: same reference, same grasp search, left hand
    # parked away from the object.
    bt.park("l")
    solo_drop = bt.hold_test(k, seconds=seconds, grip=grip)
    ro1 = bt.rollout(start=k, steps=steps)
    solo_err = float(np.nanmean(ro1.pos_err)) * 1000
    bt.unpark()

    return {
        "seq": row["seq"], "subject": row["subject"], "object": row["object"],
        "intent": row["intent"], "window": list(map(int, bt.window)),
        "start_frame": k, "n_grasp_frames": int(len(gf)),
        "raw_drop_m": float(raw_drop), "drop_m": float(drop),
        "held": bool(drop < 0.05), "track_mm": err,
        "tracked_steps": int(ro.steps), "span": int(span),
        "tracked_frac": float(ro.steps / max(span, 1)),
        "solo_drop_m": float(solo_drop), "solo_track_mm": solo_err,
        "solo_tracked_steps": int(ro1.steps),
        "solo_tracked_frac": float(ro1.steps / max(span, 1)),
        "two_handed_helps": bool(np.isfinite(solo_drop) and drop < solo_drop),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="results/stage6_bimanual.json")
    a = ap.parse_args()

    rows = json.load(open("results/grab_inventory.json"))["rows"]
    cand = [r for r in rows
            if r.get("bimanual") and (r.get("both_hold_len") or 0) >= 15]
    seen, picked = set(), []
    for r in sorted(cand, key=lambda x: -(x.get("both_hold_len") or 0)):
        if r["object"] in seen:
            continue
        seen.add(r["object"]); picked.append(r)
        if len(picked) >= a.n:
            break

    out, t0 = [], time.time()
    for i, r in enumerate(picked):
        try:
            rec = one(r, stride=a.stride, samples=a.samples, rounds=a.rounds)
        except Exception as e:
            print(f"  {r['seq']}: FAILED {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        out.append(rec)
        if rec.get("no_grasp_frame"):
            print(f"[{i+1}/{len(picked)}] {rec['seq']:26s} {rec['object']:14s} "
                  f"no frame holds on its own", file=sys.stderr, flush=True)
            Path(a.out).write_text(json.dumps({"rows": out}, indent=1))
            continue
        print(f"[{i+1}/{len(picked)}] {rec['seq']:26s} {rec['object']:14s} "
              f"drop {rec['raw_drop_m']*1000:7.1f} -> {rec['drop_m']*1000:7.1f} mm "
              f"held={rec['held']}  track {rec['track_mm']:7.1f} mm over "
              f"{rec['tracked_frac']*100:3.0f}% of the clip  (solo "
              f"{rec['solo_track_mm']:7.1f} mm / {rec['solo_tracked_frac']*100:3.0f}%)"
              f"  ({time.time()-t0:5.0f}s)", file=sys.stderr, flush=True)
        Path(a.out).write_text(json.dumps({"rows": out}, indent=1))

    if out:
        h = [r["held"] for r in out if "held" in r]
        print(f"\nstage 6: {sum(h)}/{len(h)} two-handed grasps hold",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
