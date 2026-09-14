"""G3 Part 2 analysis, exactly as pre-registered in docs/G3_PREREGISTRATION.md.

Committed before any G3 result existed. Seven declared predictors, Spearman and
ROC AUC as co-primaries, Holm correction applied because reporting the best of
seven without it is a multiple-comparisons error, per-hand and per-shape
breakdowns alongside the pooled figure, and the pre-declared decision rule
printed from the data.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

METRICS = ("epsilon", "hold_N", "margin", "margin_per_N", "n_contacts",
           "f_total", "penetration_mm")
RHO_THRESHOLD = 0.3


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        blob = json.loads(pathlib.Path(f).read_text())
        rows += blob["rows"] if isinstance(blob, dict) else blob
    return rows


def auc(x, y):
    """ROC AUC of score x against binary label y, by rank (Mann-Whitney)."""
    x, y = np.asarray(x, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(x)
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def holm(pvals):
    """Holm-Bonferroni corrected p-values, same order as the input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        out[idx] = min(running, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/g3*.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    rows = load(a.glob)
    if not rows:
        raise SystemExit("no rows")

    # pooled over both tasks: each grasp contributes two observations
    X = {m: np.array([r[m] for r in rows] + [r[m] for r in rows], float)
         for m in METRICS}
    Y = np.array([r["task_a"] for r in rows] + [r["task_b"] for r in rows], bool)
    n = len(Y)
    print(f"G3 Part 2 -- pre-registered analysis")
    print(f"{len(rows)} grasps x 2 tasks = {n} observations, "
          f"{int(Y.sum())} successes ({100*Y.mean():.1f}%)\n")

    rhos, ps, aucs = [], [], []
    for m in METRICS:
        r = spearmanr(X[m], Y)
        rhos.append(r.correlation); ps.append(r.pvalue); aucs.append(auc(X[m], Y))
    hp = holm(ps)
    print(f"{'metric':16s} {'Spearman':>9s} {'p':>9s} {'p(Holm)':>9s} {'AUC':>7s}")
    for m, r_, p_, h_, u_ in zip(METRICS, rhos, ps, hp, aucs):
        flag = "  *" if (h_ < a.alpha and abs(r_) >= RHO_THRESHOLD) else ""
        print(f"{m:16s} {r_:+9.3f} {p_:9.4f} {h_:9.4f} {u_:7.3f}{flag}")

    print("\nper hand (Spearman with task success, pooled over both tasks):")
    hands = sorted(set(r["hand"] for r in rows))
    print(f"{'metric':16s} " + "".join(f"{h:>10s}" for h in hands))
    for m in METRICS[:4]:
        cells = []
        for h in hands:
            sel = [r for r in rows if r["hand"] == h]
            xs = [r[m] for r in sel] * 2
            ys = [r["task_a"] for r in sel] + [r["task_b"] for r in sel]
            cells.append(spearmanr(xs, ys).correlation if len(set(ys)) > 1
                         else float("nan"))
        print(f"{m:16s} " + "".join(f"{c:+10.3f}" for c in cells))

    print("\nper shape:")
    shapes = sorted(set(r["shape"] for r in rows))
    print(f"{'metric':16s} " + "".join(f"{s:>10s}" for s in shapes))
    for m in METRICS[:4]:
        cells = []
        for s in shapes:
            sel = [r for r in rows if r["shape"] == s]
            xs = [r[m] for r in sel] * 2
            ys = [r["task_a"] for r in sel] + [r["task_b"] for r in sel]
            cells.append(spearmanr(xs, ys).correlation if len(set(ys)) > 1
                         else float("nan"))
        print(f"{m:16s} " + "".join(f"{c:+10.3f}" for c in cells))

    print("\n=== PRE-DECLARED DECISION RULE ===")
    winners = [(m, r_, h_) for m, r_, h_ in zip(METRICS, rhos, hp)
               if h_ < a.alpha and abs(r_) >= RHO_THRESHOLD]
    eps_rho = rhos[METRICS.index("epsilon")]
    if winners:
        best = max(winners, key=lambda t: abs(t[1]))
        print(f"  {best[0]} predicts task success (rho {best[1]:+.3f}, "
              f"Holm p {best[2]:.4f}).")
        print("  That is what the project should optimise, and G1's result is")
        print("  meaningful to the extent it used that metric.")
    elif eps_rho <= 0:
        print(f"  NO metric reaches rho 0.3 with Holm p < 0.05, AND epsilon is")
        print(f"  non-positive (rho {eps_rho:+.3f}). The field's standard grasp")
        print("  metric is uninformative or inverted for task outcomes here.")
        print("  Say this plainly and carefully.")
    else:
        print("  NO metric reaches rho 0.3 with Holm p < 0.05. The standard")
        print("  grasp metrics do not predict task success in this setting.")
        print("  This is the headline: it explains the G1/G2 disagreement and")
        print("  is a benchmark critique, not a failure of the project.")


if __name__ == "__main__":
    main()
