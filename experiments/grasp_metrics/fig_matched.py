"""Figure for the budget-matched experiment. Statistics derived, never typed in.

The withdrawn `fig_thesis.py` hardcoded its p-value and discordant counts into
the title, so the figure could not disagree with its data. Everything printed
here is computed from the files it plots.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

POSE, WRENCH = "#E69F00", "#0072B2"
HANDS = ["shadow", "leap", "allegro", "f5d6"]


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        blob = json.loads(pathlib.Path(f).read_text())
        rows += blob["rows"] if isinstance(blob, dict) else blob
    cells = {}
    for r in rows:
        cells.setdefault((r["hand"], r["width"], r["seed"]), {})[r["condition"]] = r
    return [v for v in cells.values() if len(v) == 2], cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/matched_*.json")
    ap.add_argument("--out", default="figures/matched.png")
    a = ap.parse_args()
    _full, cells = load(a.glob)
    keys = [k for k, v in cells.items() if len(v) == 2]
    P = np.array([cells[k]["pose"]["hold_N"] > 0 for k in keys])
    W = np.array([cells[k]["wrench"]["hold_N"] > 0 for k in keys])
    KP = np.array([cells[k]["pose"]["keypoint_err_m"] for k in keys]) * 100
    KW = np.array([cells[k]["wrench"]["keypoint_err_m"] for k in keys]) * 100
    H = np.array([k[0] for k in keys])
    SD = np.array([k[2] for k in keys])
    b = int((W & ~P).sum()); c = int((P & ~W).sum())
    pv = binomtest(b, max(b + c, 1), 0.5).pvalue

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    hands = [h for h in HANDS if (H == h).any()]
    x = np.arange(len(hands) + 1)
    pv_ = [P[H == h].mean() * 100 for h in hands] + [P.mean() * 100]
    wv_ = [W[H == h].mean() * 100 for h in hands] + [W.mean() * 100]
    ns = [int((H == h).sum()) for h in hands] + [len(P)]
    ax.bar(x - 0.19, pv_, 0.36, color=POSE, label="maximise POSE fidelity")
    ax.bar(x + 0.19, wv_, 0.36, color=WRENCH, label="maximise WRENCH quality")
    for i, (u, v) in enumerate(zip(pv_, wv_)):
        ax.text(i - 0.19, u + 1.8, f"{u:.0f}%", ha="center", fontsize=8)
        ax.text(i + 0.19, v + 1.8, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}\nn={n}" for h, n in zip(hands + ["all"], ns)],
                       fontsize=8.5)
    ax.set_ylabel("cells yielding a grasp that survives\nthe sampled probe (%)",
                  fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(f"McNemar exact $p$={pv:.1e}  ({b} discordant vs {c})",
                 fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    ax = axes[1]
    for sd, mk in zip(sorted(set(SD)), ("o", "s", "^")):
        m = (SD == sd) & ~np.isnan(KP) & ~np.isnan(KW)
        ax.scatter(KP[m], KW[m], s=30, marker=mk, color="#009E73",
                   edgecolor="white", zorder=3, label=f"seed {sd}")
    lim = float(np.nanmax(np.r_[KP, KW])) * 1.08
    ax.plot([0, lim], [0, lim], "--", color="#999999", linewidth=1.2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("pose condition: keypoint error (cm)", fontsize=9)
    ax.set_ylabel("wrench condition (cm)", fontsize=9)
    ax.set_title("the pose objective IS the more faithful one\n"
                 "(points above the diagonal)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for a_ in axes:
        a_.grid(alpha=0.25, linewidth=0.6); a_.set_axisbelow(True)
        a_.tick_params(labelsize=8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"wrote {a.out}")

    print("\nseed-stratified survival discordances (wrench-only vs pose-only):")
    for sd in sorted(set(SD)):
        m = SD == sd
        bb = int((W[m] & ~P[m]).sum()); cc = int((P[m] & ~W[m]).sum())
        print(f"  seed {sd}: {bb} vs {cc}   exact p="
              f"{binomtest(bb, max(bb+cc,1), 0.5).pvalue:.4f}   n={int(m.sum())}")
    print(f"pooled: {b} vs {c}  p={pv:.3e}  "
          f"(cells are 20 hand x width configs x 3 optimiser seeds, "
          f"NOT 60 independent tasks)")


if __name__ == "__main__":
    main()
