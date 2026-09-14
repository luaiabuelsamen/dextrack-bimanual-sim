"""G2 analysis, exactly as pre-registered in docs/G2_PREREGISTRATION.md.

Committed before any G2 result existed. Two co-primaries: task success (P1) and
the static probe magnitude of the selected grasp (P2). McNemar exact on P1,
Wilcoxon on P2, seed-stratified alongside pooled, and the pre-declared decision
rule printed from the result rather than chosen after it.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
from scipy.stats import binomtest, wilcoxon

POSE, GENERIC, DEMO = "pose_squeeze", "task_generic", "task_demo"
ARMS = (POSE, GENERIC, DEMO)


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        blob = json.loads(pathlib.Path(f).read_text())
        rows += blob["rows"] if isinstance(blob, dict) else blob
    cells = {}
    for r in rows:
        cells.setdefault((r["hand"], r["width"], r["seed"]), {})[r["arm"]] = r
    return {k: v for k, v in cells.items() if len(v) == len(ARMS)}


def mcnemar(a, b):
    bo = int((b & ~a).sum()); ao = int((a & ~b).sum())
    return bo, ao, binomtest(bo, max(bo + ao, 1), 0.5).pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/g2*.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    cells = load(a.glob)
    keys = sorted(cells)
    if not keys:
        raise SystemExit("no complete cells")
    S = {x: np.array([cells[k][x]["task_success"] for k in keys]) for x in ARMS}
    P2 = {x: np.array([cells[k][x]["hold_N"] for k in keys]) for x in ARMS}
    MG = {x: np.array([cells[k][x]["margin"] for k in keys]) for x in ARMS}
    EP = {x: np.array([cells[k][x]["epsilon"] for k in keys]) for x in ARMS}
    V = {x: np.array([cells[k][x]["rejections"].get("valid", 0) for k in keys])
         for x in ARMS}
    SD = np.array([k[2] for k in keys]); H = np.array([k[0] for k in keys])
    n = len(keys)

    print(f"G2 -- pre-registered analysis.  n = {n} cells "
          f"({len(set((k[0],k[1]) for k in keys))} configurations x "
          f"{len(set(SD))} optimiser seeds)\n")
    print(f"{'arm':14s} {'P1 task':>9s} {'P2 hold_N':>10s} {'margin':>9s} "
          f"{'eps':>7s} {'valid/budget':>13s}")
    for x in ARMS:
        print(f"{x:14s} {S[x].sum():4d}/{n:<4d} {P2[x].mean():10.3f} "
              f"{MG[x].mean():9.2f} {EP[x].mean():7.4f} {V[x].mean():13.1f}")

    print("\n--- H1: task_demo > pose_squeeze ---")
    b1, a1, p1 = mcnemar(S[POSE], S[DEMO])
    print(f"  P1  discordant {b1} (demo only) vs {a1} (pose only)   "
          f"McNemar exact p = {p1:.5f}")
    d = P2[DEMO] - P2[POSE]; nz = d != 0
    if nz.sum() >= 5:
        print(f"  P2  Wilcoxon p = {wilcoxon(P2[DEMO][nz], P2[POSE][nz]).pvalue:.5f}")

    print("\n--- H2 (the question): task_demo > task_generic ---")
    b2, a2, p2v = mcnemar(S[GENERIC], S[DEMO])
    print(f"  P1  discordant {b2} (demo only) vs {a2} (generic only)  "
          f"McNemar exact p = {p2v:.5f}")
    d2 = P2[DEMO] - P2[GENERIC]; nz2 = d2 != 0
    p2_p = float("nan")
    if nz2.sum() >= 5:
        p2_p = wilcoxon(P2[DEMO][nz2], P2[GENERIC][nz2]).pvalue
        print(f"  P2  demo {P2[DEMO].mean():.3f} N vs generic "
              f"{P2[GENERIC].mean():.3f} N   Wilcoxon p = {p2_p:.5f}")

    print("\nseed-stratified task success (pre-registered):")
    for sd in sorted(set(SD)):
        m = SD == sd
        bb, aa, pp = mcnemar(S[GENERIC][m], S[DEMO][m])
        print(f"  seed {sd}: pose {S[POSE][m].sum():2d}/{m.sum()}  "
              f"generic {S[GENERIC][m].sum():2d}/{m.sum()}  "
              f"demo {S[DEMO][m].sum():2d}/{m.sum()}   "
              f"H2 discordant {bb}v{aa} p={pp:.4f}")
    print("\nper hand task success:")
    for h in sorted(set(H)):
        m = H == h
        print(f"  {h:8s} n={m.sum():2d}  pose {S[POSE][m].sum():2d}  "
              f"generic {S[GENERIC][m].sum():2d}  demo {S[DEMO][m].sum():2d}")

    print("\n=== PRE-DECLARED DECISION RULE ===")
    h2_p1 = p2v < a.alpha and S[DEMO].sum() > S[GENERIC].sum()
    h2_p2 = (p2_p == p2_p) and p2_p < a.alpha and P2[DEMO].mean() > P2[GENERIC].mean()
    if p1 >= a.alpha and S[DEMO].sum() <= S[POSE].sum():
        print("  H1 NULL -> neither object-side arm beats the shipped pipeline")
        print("  on a task. This contradicts G1 and must be explained before")
        print("  anything else is claimed.")
    elif h2_p1:
        print("  H2 SIGNIFICANT on P1 -> human data specifies the TASK, not the")
        print("  grasp. The demonstration is necessary rather than decorative.")
    elif h2_p2:
        print("  H2 null on P1, significant on P2 -> the demonstration buys")
        print("  grasp STRENGTH but not task RELIABILITY, confirming G1's")
        print("  exploratory finding under pre-registration. Narrower claim.")
    else:
        print("  H2 NULL on both -> object-side wrench specification is")
        print("  sufficient; human hand-object data is dispensable for this")
        print("  class of problem. Two independent pre-registered nulls. This")
        print("  is the headline, not a footnote.")


if __name__ == "__main__":
    main()
