"""Does a stage-2 grasp survive being CARRIED along the reference?

This is the question every learning stage turns out to be downstream of. Stage 2
scores a grasp by whether the object stays put while the hand is held still; the
tracker then asks the hand to move along the demonstration. Nothing in the stage-2
score tests that transition, and the corrected stage-3 run says the initial
condition determines the outcome while the controller barely moves it -- so if
stage-2 grasps do not survive the carry, no amount of work on stages 3-5 can
help, and if they do, the tracking stage has something to hold on to.

Feedforward only, deliberately. No policy, no search: the hand is commanded along
the retargeted trajectory and we record how far the object gets before it is
lost. That isolates the grasp from everything downstream of it.

Reported by seed class, because the one thing the corrected stage 3 established
is that class predicts the outcome and nothing else measured does.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco

from handsim.human import grab, track as T

W = 0.2 * 9.81          # the object weight this pipeline standardises on


def seed_class(r):
    if not r.get("held"):
        return "failed"
    n, g = r["n_contact"], r["grip_n"]
    if n > 30 or g > 200 * W:
        return "burial"
    if g < W:
        return "thin"
    if n <= 12 and g <= 50 * W:
        return "grasp"
    return "mixed"


def one(row, lost=0.10):
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = T.ReferenceTracker(seq, hand="shadow")
    rt.apply_wrist_offset(np.asarray(row["offset"], float))
    gf = rt.grasp_frames()
    if len(gf) == 0:
        return {"seq": row["seq"], "subject": row["subject"],
                "object": row["object"], "kind": seed_class(row),
                "no_grasp_frame": True}
    k0 = int(gf[0])
    rt.reset_at(k0)
    # RESET state, before any motion. The carriers all start buried and relax
    # under motion -- stamp_lift falls 46x in depth and 244x in force between
    # here and the end -- so the initial contact set is the precondition for the
    # carry rather than a defect in it, and its grip force is the candidate
    # discriminator this sweep exists to test.
    mujoco.mj_forward(rt.sim.model, rt.sim.data)
    r_pen, _ = rt.sim.penetration()
    r_grip, r_ncon = T.total_grip(rt.sim)
    # Equilibrium alongside the other three, so "grip force separates and depth
    # does not" is shown on the same rows rather than asserted across them.
    r_eq = T.equilibrium_residual(rt.sim, rt.sim.obj_bid)
    ro = rt.rollout(start=k0, steps=rt.T - k0)

    # how far along the reference the object was still being carried
    ok = np.flatnonzero(ro.pos_err <= lost)
    held_to = int(ok[-1] + 1) if len(ok) else 0
    span = int(rt.T - k0)
    pen_mm, _ = rt.sim.penetration()
    grip_n, ncon = T.total_grip(rt.sim)
    return {
        "seq": row["seq"], "subject": row["subject"], "object": row["object"],
        "kind": seed_class(row), "start": k0, "span": span,
        "grasp_frames": int(len(gf)),
        "carried_frames": held_to, "carried_frac": float(held_to / max(span, 1)),
        "mean_mm": float(np.nanmean(ro.pos_err) * 1000),
        "end_pen_mm": float(pen_mm * 1000), "end_contacts": int(ncon),
        "end_grip_n": float(grip_n),
        "seed_contacts": int(row["n_contact"]), "seed_grip_n": float(row["grip_n"]),
        "reset_pen_mm": float(r_pen * 1000), "reset_contacts": int(r_ncon),
        "reset_grip_n": float(r_grip), "reset_eq": float(r_eq),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grips", default="results/stage2_grips_clean.json")
    ap.add_argument("--out", default="results/stage2_survives.json")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()

    rows = [r for r in json.loads(Path(a.grips).read_text())["rows"] if r["held"]]
    if a.n:
        rows = rows[:a.n]
    print(f"{len(rows)} held grasps from {a.grips}", flush=True)

    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        try:
            rec = one(r)
        except Exception as e:                      # a bad mesh must not stop it
            print(f"  {r['seq']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        out.append(rec)
        if rec.get("no_grasp_frame"):
            print(f"[{i+1}/{len(rows)}] {rec['seq']:24s} {rec['kind']:7s} "
                  f"no graspable frame", flush=True)
        else:
            print(f"[{i+1}/{len(rows)}] {rec['seq']:24s} {rec['kind']:7s} "
                  f"reset {rec['reset_pen_mm']:5.1f}mm/{rec['reset_grip_n']:7.0f}N  "
                  f"carried {rec['carried_frames']:3d}/{rec['span']:3d} frames "
                  f"({rec['carried_frac']*100:3.0f}%)  mean {rec['mean_mm']:8.1f} mm  "
                  f"ends {rec['end_pen_mm']:5.2f} mm, {rec['end_contacts']:3d} con, "
                  f"{rec['end_grip_n']:8.0f} N  ({time.time()-t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps({"grips": a.grips, "rows": out}, indent=1))

    done = [r for r in out if not r.get("no_grasp_frame")]
    if done:
        print(f"\ncarried to the end of the reference: "
              f"{sum(r['carried_frac'] > 0.95 for r in done)}/{len(done)}")
        car = [r for r in done if r["carried_frac"] > 0.95]
        fail = [r for r in done if r["carried_frac"] <= 0.95]
        if car and fail:
            print("\n  at RESET, carried to the end vs not:")
            for lab, key, fmt in (("pen mm", "reset_pen_mm", "8.2f"),
                                  ("contacts", "reset_contacts", "8.0f"),
                                  ("grip N", "reset_grip_n", "8.0f"),
                                  ("equilib", "reset_eq", "8.2f")):
                c = [r[key] for r in car]; f = [r[key] for r in fail]
                print(f"    {lab:9s} carried median {np.median(c):{fmt}}  "
                      f"[{min(c):{fmt}}, {max(c):{fmt}}]   "
                      f"not median {np.median(f):{fmt}}  "
                      f"[{min(f):{fmt}}, {max(f):{fmt}}]")
            print(f"\n  reset grip N -- carried to the end: median "
                  f"{np.median([r['reset_grip_n'] for r in car]):8.0f}  "
                  f"min {min(r['reset_grip_n'] for r in car):8.0f}")
            print(f"                  did not:            median "
                  f"{np.median([r['reset_grip_n'] for r in fail]):8.0f}  "
                  f"max {max(r['reset_grip_n'] for r in fail):8.0f}")
        for k in ("grasp", "mixed", "thin", "burial"):
            sel = [r for r in done if r["kind"] == k]
            if sel:
                print(f"  {k:7s} n={len(sel):2d}  median carried "
                      f"{np.median([r['carried_frac'] for r in sel])*100:3.0f}%  "
                      f"median end pen {np.median([r['end_pen_mm'] for r in sel]):5.2f} mm")


if __name__ == "__main__":
    main()
