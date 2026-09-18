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


def cluster_bootstrap(rows, metric, n_boot=1500, seed=0):
    """Spearman rho and AUC with a CI that respects clustering by GRASP.

    Each grasp contributes two observations -- task A and task B -- which share
    the grasp and are not independent. Treating them as 2N independent points
    understates the variance and inflates significance. The resample unit is
    the grasp.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    rhos, aucs = [], []
    for _ in range(n_boot):
        sel = [rows[i] for i in rng.integers(0, n, n)]
        x = np.array([r[metric] for r in sel] * 2, float)
        y = np.array([r["task_a"] for r in sel] + [r["task_b"] for r in sel], bool)
        if y.all() or not y.any() or np.ptp(x) == 0:
            continue
        rhos.append(spearmanr(x, y).correlation)
        aucs.append(auc(x, y))
    if not rhos:
        return dict(rho_lo=np.nan, rho_hi=np.nan, auc_lo=np.nan,
                    auc_hi=np.nan, p_boot=1.0)
    rhos, aucs = np.array(rhos), np.array(aucs)
    frac = float(min((rhos <= 0).mean(), (rhos >= 0).mean()))
    return dict(rho_lo=float(np.percentile(rhos, 2.5)),
                rho_hi=float(np.percentile(rhos, 97.5)),
                auc_lo=float(np.percentile(aucs, 2.5)),
                auc_hi=float(np.percentile(aucs, 97.5)),
                p_boot=float(min(2 * frac, 1.0)))


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
    ap.add_argument("--glob", default="legacy/results/g3*.json")
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
    boots = {m: cluster_bootstrap(rows, m) for m in METRICS}
    hp_boot = holm([boots[m]["p_boot"] for m in METRICS])
    print("inference clustered BY GRASP: task A and task B share a grasp, so")
    print("2N independent observations would understate the variance.\n")
    print(f"{'metric':16s} {'rho':>8s} {'rho 95% CI':>20s} {'p(Holm)':>9s} "
          f"{'AUC':>7s} {'AUC 95% CI':>16s}")
    for i, m in enumerate(METRICS):
        b = boots[m]
        flag = " *" if (hp_boot[i] < a.alpha and abs(rhos[i]) >= RHO_THRESHOLD) else ""
        print(f"{m:16s} {rhos[i]:+8.3f} "
              f"{'[%+.3f,%+.3f]' % (b['rho_lo'], b['rho_hi']):>20s} "
              f"{hp_boot[i]:9.4f} {aucs[i]:7.3f} "
              f"{'[%.3f,%.3f]' % (b['auc_lo'], b['auc_hi']):>16s}{flag}")
    print(f"\n  naive per-observation Holm p (NOT used for the decision): "
          f"{', '.join('%s %.3f' % (m, h) for m, h in zip(METRICS, hp))}")

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
    winners = [(m, r_, h_) for m, r_, h_ in zip(METRICS, rhos, hp_boot)
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
