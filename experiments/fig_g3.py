"""Figure: do standard grasp metrics predict task success? Statistics derived."""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from experiments.g3_analysis import load, METRICS, auc, holm

LABEL = {"epsilon": "Ferrari–Canny $\\epsilon$", "hold_N": "static probe hold",
         "margin": "task margin", "margin_per_N": "task margin / N",
         "n_contacts": "contact count", "f_total": "contact force",
         "penetration_mm": "penetration (mm)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/g3*.json")
    ap.add_argument("--out", default="figures/g3_metrics.png")
    a = ap.parse_args()
    rows = load(a.glob)
    X = {m: np.array([r[m] for r in rows] * 2, float) for m in METRICS}
    Y = np.array([r["task_a"] for r in rows] + [r["task_b"] for r in rows], bool)
    rho = [spearmanr(X[m], Y).correlation for m in METRICS]
    pv = [spearmanr(X[m], Y).pvalue for m in METRICS]
    hp = holm(pv)
    au = [auc(X[m], Y) for m in METRICS]
    order = np.argsort(np.abs(rho))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    names = [LABEL[METRICS[i]] for i in order]
    vals = [rho[i] for i in order]
    cols = ["#0072B2" if hp[i] < 0.05 else "#BBBBBB" for i in order]
    ax.barh(range(len(vals))[::-1], vals, color=cols)
    ax.axvline(0.3, ls="--", color="#CC79A7", lw=1.3)
    ax.axvline(-0.3, ls="--", color="#CC79A7", lw=1.3)
    ax.axvline(0, color="#555555", lw=0.9)
    ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Spearman $\\rho$ with task success", fontsize=9)
    ax.set_title("no metric reaches the pre-declared $\\rho=0.3$", fontsize=9.5)
    ax.text(0.31, len(vals) - 1.4, "pre-declared\nthreshold", fontsize=7.5,
            color="#CC79A7")

    ax = axes[1]
    ax.barh(range(len(vals))[::-1], [au[i] for i in order], color=cols)
    ax.axvline(0.5, color="#555555", lw=1.1)
    ax.set_xlim(0.25, 0.75)
    ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels([])
    ax.set_xlabel("ROC AUC (0.5 = coin flip)", fontsize=9)
    e = METRICS.index("epsilon")
    ax.set_title(f"$\\epsilon$ AUC = {au[e]:.3f}", fontsize=9.5)
    for a_ in axes:
        a_.grid(alpha=0.25, axis="x", linewidth=0.6); a_.set_axisbelow(True)
        a_.tick_params(labelsize=8)
        for sp in ("top", "right"): a_.spines[sp].set_visible(False)
    fig.suptitle(f"{len(rows)} sampled grasps x 2 carry tasks = {len(Y)} "
                 f"observations, {int(Y.sum())} successes", fontsize=10, y=1.03)
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"wrote {a.out}  (blue = Holm p < 0.05)")


if __name__ == "__main__":
    main()
