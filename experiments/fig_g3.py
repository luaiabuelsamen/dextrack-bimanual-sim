"""Figure: which grasp metrics predict task success? Statistics derived, clustered."""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from experiments.g3_analysis import load, METRICS, auc, cluster_bootstrap

LABEL = {"epsilon": "Ferrari–Canny $\\epsilon$", "hold_N": "static probe hold",
         "margin": "task margin", "margin_per_N": "task margin / N",
         "n_contacts": "contact count", "f_total": "contact force",
         "penetration_mm": "penetration"}
THRESH = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/g3_*.json")
    ap.add_argument("--out", default="figures/g3_metrics.png")
    a = ap.parse_args()
    rows = load(a.glob)
    X = {m: np.array([r[m] for r in rows] * 2, float) for m in METRICS}
    Y = np.array([r["task_a"] for r in rows] + [r["task_b"] for r in rows], bool)
    rho = {m: spearmanr(X[m], Y).correlation for m in METRICS}
    au = {m: auc(X[m], Y) for m in METRICS}
    bs = {m: cluster_bootstrap(rows, m) for m in METRICS}
    order = sorted(METRICS, key=lambda m: au[m], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    vals = [au[m] for m in order]
    err = np.array([[au[m] - bs[m]["auc_lo"] for m in order],
                    [bs[m]["auc_hi"] - au[m] for m in order]])
    cols = ["#0072B2" if abs(rho[m]) >= THRESH else "#7FB3D5" for m in order]
    ax.barh(y, vals, color=cols, xerr=err, error_kw=dict(lw=1.1, ecolor="#444444"))
    ax.axvline(0.5, color="#555555", lw=1.2)
    ax.set_xlim(0.2, 0.95)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in order], fontsize=8.5)
    ax.set_xlabel("ROC AUC predicting task success  (0.5 = coin flip)", fontsize=9)
    ax.set_title("contact FORCE predicts best; $\\epsilon$ is weaker", fontsize=9.5)
    for i, m in enumerate(order):
        ax.text(vals[i] + err[1][i] + 0.012, y[i], f"{vals[i]:.3f}",
                va="center", fontsize=7.5)

    ax = axes[1]
    shapes = sorted(set(r["shape"] for r in rows))
    show = ["f_total", "margin", "epsilon"]
    w = 0.26
    for j, m in enumerate(show):
        vs = []
        for s in shapes:
            sel = [r for r in rows if r["shape"] == s]
            xs = [r[m] for r in sel] * 2
            ys = [r["task_a"] for r in sel] + [r["task_b"] for r in sel]
            vs.append(spearmanr(xs, ys).correlation if len(set(ys)) > 1 else np.nan)
        ax.bar(np.arange(len(shapes)) + (j - 1) * w, vs, w,
               label=LABEL[m], color=["#0072B2", "#009E73", "#E69F00"][j])
    ax.axhline(0, color="#555555", lw=1.0)
    ax.set_xticks(np.arange(len(shapes))); ax.set_xticklabels(shapes, fontsize=8.5)
    ax.set_ylabel("Spearman $\\rho$ with task success", fontsize=9)
    ax.set_title("every metric inverts on capsules", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    for a_ in axes:
        a_.grid(alpha=0.25, axis="x" if a_ is axes[0] else "y", linewidth=0.6)
        a_.set_axisbelow(True); a_.tick_params(labelsize=8)
        for sp in ("top", "right"): a_.spines[sp].set_visible(False)
    fig.suptitle(f"{len(rows)} sampled grasps x 2 carry tasks "
                 f"({int(Y.sum())} successes); CIs bootstrapped by grasp",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
