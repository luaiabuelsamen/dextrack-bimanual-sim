"""Figure: what each retargeting objective achieves, per hand and box width.

Small multiples rather than one crowded axis -- the comparison that matters is
WITHIN a hand (same kinematics, same object, different objective), and putting
four hands on one axis invites reading across hands, which is not the claim.

Colours are the Okabe-Ito set, designed for colour-vision deficiency; the
skill's palette validator is not installed on this machine, so a palette with a
published CVD guarantee is used rather than one checked by eye. Identity is
never carried by colour alone: every series is also directly labelled in the
first panel and distinguished by marker.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, in fixed order. Never cycled, never reassigned by rank.
COLORS = {
    "demo": "#666666",
    "keypoint": "#E69F00",          # orange
    "keypoint+squeeze": "#CC79A7",  # reddish purple
    "epsilon": "#0072B2",           # blue
}
MARKERS = {"demo": "s", "keypoint": "o", "keypoint+squeeze": "^", "epsilon": "D"}
LABELS = {"demo": "human demo", "keypoint": "keypoint",
          "keypoint+squeeze": "keypoint + squeeze", "epsilon": "$\\epsilon$-optimised"}
ORDER = ["demo", "keypoint", "keypoint+squeeze", "epsilon"]
HANDS = ["shadow", "leap", "allegro", "f5d6"]


def load(root):
    rows = []
    for h in HANDS:
        f = pathlib.Path(root) / h / "results.json"
        if f.exists():
            rows += json.loads(f.read_text())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/retarget")
    ap.add_argument("--out", default="figures/retarget_epsilon.png")
    a = ap.parse_args()

    rows = load(a.root)
    if not rows:
        raise SystemExit(f"no results under {a.root}")
    hands = [h for h in HANDS if any(r["hand"] == h for r in rows)]

    fig, axes = plt.subplots(1, len(hands), figsize=(3.1 * len(hands), 3.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, hand in zip(axes, hands):
        for cond in ORDER:
            pts = sorted([(r["width"] * 100, r["epsilon"]) for r in rows
                          if r["hand"] == hand and r["objective"] == cond])
            if not pts:
                continue
            x, y = zip(*pts)
            # keypoint and +squeeze sit almost on top of each other -- that
            # overlap IS the result (squeezing a keypoint pose rarely rescues
            # it), so squeeze is drawn as a thinner line with HOLLOW markers
            # and keypoint stays visible underneath rather than being buried.
            hollow = cond == "keypoint+squeeze"
            ax.plot(x, y, marker=MARKERS[cond], color=COLORS[cond],
                    linewidth=1.6 if hollow else 2.2,
                    markersize=6.5 if cond == "keypoint" else 5.0,
                    markerfacecolor="none" if hollow else COLORS[cond],
                    markeredgewidth=1.6 if hollow else 1.0,
                    linestyle="--" if cond == "demo" else "-",
                    label=LABELS[cond],
                    zorder={"epsilon": 4, "keypoint+squeeze": 3,
                            "keypoint": 2, "demo": 1}[cond])
        ax.set_title(hand, fontsize=11)
        ax.set_xlabel("box width (cm)", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Ferrari-Canny $\\epsilon$", fontsize=9)
    axes[0].set_ylim(-0.02, None)

    # A zero here is not a small number: it means no force closure at all.
    for ax in axes:
        ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.13))
    n_zero = sum(1 for r in rows
                 if r["objective"] == "keypoint" and r["epsilon"] == 0)
    n_kp = sum(1 for r in rows if r["objective"] == "keypoint")
    fig.suptitle(f"Matching the human's fingertips does not produce a grasp: "
                 f"$\\epsilon=0$ in {n_zero} of {n_kp} hand x width cases",
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"wrote {a.out}")

    print("\nmean epsilon by condition:")
    for cond in ORDER:
        v = [r["epsilon"] for r in rows if r["objective"] == cond]
        z = sum(1 for e in v if e == 0)
        print(f"  {LABELS[cond]:22s} {np.mean(v):.4f}   zero in {z}/{len(v)}")


if __name__ == "__main__":
    main()
