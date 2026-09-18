"""G1 analysis, exactly as pre-registered in docs/G1_PREREGISTRATION.md.

Written to run without having seen the numbers. Primary outcome is binary
survival of the sampled probe; McNemar exact on discordant pairs; Wilcoxon on
non-tied magnitudes as the secondary; seed-stratified alongside pooled. Two
declared comparisons, H1 and H2, and the pre-declared decision rule is printed
from the result rather than chosen after it.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
from scipy.stats import binomtest, wilcoxon

POSE_SQUEEZE, EPS_SYNTH, WRENCH_REFINE = "pose_squeeze", "eps_synth", "wrench_refine"
ARMS = (POSE_SQUEEZE, EPS_SYNTH, WRENCH_REFINE)


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
    """Two-sided exact test on discordant pairs; returns (b_only, a_only, p)."""
    bo = int((b & ~a).sum()); ao = int((a & ~b).sum())
    return bo, ao, binomtest(bo, max(bo + ao, 1), 0.5).pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="legacy/results/g1_*.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    cells = load(a.glob)
    keys = sorted(cells)
    if not keys:
        raise SystemExit("no complete cells")
    S = {arm: np.array([cells[k][arm]["hold_N"] > 0 for k in keys]) for arm in ARMS}
    M = {arm: np.array([cells[k][arm]["hold_N"] for k in keys]) for arm in ARMS}
    V = {arm: np.array([cells[k][arm]["rejections"].get("valid", 0) for k in keys])
         for arm in ARMS}
    E = {arm: np.array([cells[k][arm]["epsilon"] for k in keys]) for arm in ARMS}
    K = {arm: np.array([cells[k][arm]["keypoint_err_m"] for k in keys]) for arm in ARMS}
    SD = np.array([k[2] for k in keys])
    H = np.array([k[0] for k in keys])
    n = len(keys)

    print(f"G1 -- pre-registered analysis.  n = {n} cells "
          f"({len(set((k[0],k[1]) for k in keys))} configurations x "
          f"{len(set(SD))} optimiser seeds)\n")
    print(f"{'arm':15s} {'survives':>10s} {'hold_N':>9s} {'eps':>8s} "
          f"{'kp_err_cm':>10s} {'valid/144':>10s}")
    for arm in ARMS:
        print(f"{arm:15s} {S[arm].sum():5d}/{n:<4d} {M[arm].mean():9.3f} "
              f"{E[arm].mean():8.4f} {np.nanmean(K[arm])*100:10.2f} "
              f"{V[arm].mean():10.1f}")

    print("\n--- H1 (the thesis): wrench_refine > pose_squeeze ---")
    bo, ao, p1 = mcnemar(S[POSE_SQUEEZE], S[WRENCH_REFINE])
    print(f"  survival  discordant {bo} (wrench only) vs {ao} (pose only)   "
          f"McNemar exact p = {p1:.5f}")
    d = M[WRENCH_REFINE] - M[POSE_SQUEEZE]; nz = d != 0
    if nz.sum() >= 5:
        print(f"  magnitude wrench wins {(d>0).sum()}, pose wins {(d<0).sum()} "
              f"of {nz.sum()} non-tied   Wilcoxon p = "
              f"{wilcoxon(M[WRENCH_REFINE][nz], M[POSE_SQUEEZE][nz]).pvalue:.5f}")

    print("\n--- H2 (does the demonstration help?): wrench_refine vs eps_synth ---")
    bo2, ao2, p2 = mcnemar(S[EPS_SYNTH], S[WRENCH_REFINE])
    print(f"  survival  discordant {bo2} (refine only) vs {ao2} (synth only)  "
          f"McNemar exact p = {p2:.5f}")

    print("\nseed-stratified survival (pre-registered):")
    for sd in sorted(set(SD)):
        m = SD == sd
        b1, a1, pp = mcnemar(S[POSE_SQUEEZE][m], S[WRENCH_REFINE][m])
        print(f"  seed {sd}: pose {S[POSE_SQUEEZE][m].sum():2d}/{m.sum()}  "
              f"synth {S[EPS_SYNTH][m].sum():2d}/{m.sum()}  "
              f"refine {S[WRENCH_REFINE][m].sum():2d}/{m.sum()}   "
              f"H1 discordant {b1}v{a1} p={pp:.4f}")
    print("\nper hand survival:")
    for h in sorted(set(H)):
        m = H == h
        print(f"  {h:8s} n={m.sum():2d}  pose {S[POSE_SQUEEZE][m].sum():2d}  "
              f"synth {S[EPS_SYNTH][m].sum():2d}  refine {S[WRENCH_REFINE][m].sum():2d}")

    print("\n=== PRE-DECLARED DECISION RULE ===")
    h1 = p1 < a.alpha
    better_than_synth = S[WRENCH_REFINE].sum() > S[EPS_SYNTH].sum() and p2 < a.alpha
    if not h1:
        print("  H1 NOT significant -> the wrench objective does not beat the")
        print("  shipped pipeline. The retargeting framing is NOT supported;")
        print("  the surviving contribution is about objectives for grasp")
        print("  SYNTHESIS and the project should be reframed and renamed.")
    elif h1 and not better_than_synth:
        print("  H1 significant, H2 not -> a wrench objective beats the shipped")
        print("  pipeline, but the demonstration contributes nothing beyond")
        print("  initialisation. The contribution is the OBJECTIVE, not the")
        print("  retargeting.")
    else:
        print("  H1 significant AND wrench_refine > eps_synth -> the")
        print("  demonstration carries information a wrench objective alone")
        print("  does not. This is the only outcome supporting the project as")
        print("  originally framed.")


if __name__ == "__main__":
    main()
