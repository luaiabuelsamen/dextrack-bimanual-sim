"""Is a carry-scored pose a grasp, or a knife-edge?

`hand_inspect_1` scored clean over all 50 frames inside the search process and
dropped at frame 22 when the SAME offset was rebuilt and rolled by
`render_carry.py`. The two trajectories differ by the round-off of adding and
subtracting the rejected perturbations, so the pose that was accepted holds on
one floating-point path and not on another. A hold that depends on the last
bits of the wrist pose is not a grasp; it is a coincidence of the contact
solver.

So every accepted pose is re-carried under `--k` small random wrist
perturbations -- half a millimetre and half a degree by default, well under
the retarget's own error -- and the count that stays clean is what gets
reported. A pose that survives all of them is a basin; one that survives none
is the search having found a seam in the contact model.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from handsim.human import grab, track as T
from experiments.tracking.stage2_carry import carry, score_of


def one(row, a, rng):
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = T.ReferenceTracker(seq, hand="shadow")
    rt.apply_wrist_offset(np.asarray(row["offset"], float))
    k0 = int(row["start"])
    n = rt.T - k0
    _, base = score_of(carry(rt, k0, n))
    sig = np.array([a.sigma_pos] * 3 + [a.sigma_rot] * 3)
    trials = []
    for _ in range(a.k):
        dlt = rng.normal(size=6) * sig
        rt.apply_wrist_offset(dlt)
        _, sm = score_of(carry(rt, k0, n))
        rt.apply_wrist_offset(-dlt)
        trials.append({k: sm[k] for k in ("held", "clean", "relaxed", "mean_mm",
                                          "tail_pen_mm", "tail_bodies",
                                          "tail_sided", "tail_grip_n")})
    return {"seq": row["seq"], "subject": row["subject"], "object": row["object"],
            "start": k0, "span": int(n),
            "rebuilt": {k: base[k] for k in ("held", "clean", "mean_mm",
                                            "tail_pen_mm", "tail_bodies")},
            "k": a.k, "n_held": sum(t["held"] for t in trials),
            "n_clean": sum(t["clean"] for t in trials), "trials": trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry", default="results/stage2_carry_full.json")
    ap.add_argument("--out", default="results/stage2_carry_robust.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sigma-pos", type=float, default=0.0005)
    ap.add_argument("--sigma-rot", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqs", default="")
    a = ap.parse_args()

    rows = json.loads(Path(a.carry).read_text())["rows"]
    if a.seqs:
        want = set(x.strip() for x in a.seqs.split(","))
        rows = [r for r in rows if r["seq"] in want]
    else:
        rows = [r for r in rows if r["end"]["clean"]]
    rng = np.random.default_rng(a.seed)
    out = []
    print(f"{len(rows)} poses x {a.k} perturbations of "
          f"{a.sigma_pos*1000:.1f} mm / {np.degrees(a.sigma_rot):.1f} deg", flush=True)
    for r in rows:
        try:
            rec = one(r, a, rng)
        except Exception as e:                            # noqa: BLE001
            print(f"  {r['seq']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        out.append(rec)
        b = rec["rebuilt"]
        print(f"{rec['seq']:24s} rebuilt clean={int(b['clean'])} "
              f"({b['tail_pen_mm']:5.2f} mm, {b['mean_mm']:7.1f} mm)   "
              f"perturbed: held {rec['n_held']}/{a.k}  clean {rec['n_clean']}/{a.k}",
              flush=True)
        Path(a.out).write_text(json.dumps(
            {"carry": a.carry, "k": a.k, "sigma_pos": a.sigma_pos,
             "sigma_rot": a.sigma_rot, "seed": a.seed, "rows": out}, indent=1))
    if out:
        print(f"\nall {a.k} clean: {sum(r['n_clean'] == a.k for r in out)}/{len(out)}; "
              f">= half clean: {sum(2 * r['n_clean'] >= a.k for r in out)}/{len(out)}; "
              f"none clean: {sum(r['n_clean'] == 0 for r in out)}/{len(out)}")


if __name__ == "__main__":
    main()
