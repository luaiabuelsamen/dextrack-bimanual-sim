"""Figure: what a second hand buys, in newtons the grasp actually survives.

Two panels per claim. Left, the wrench the best grasp resists in its WORST
direction -- the physical quantity epsilon is a prediction of. Right, epsilon
itself, from real MuJoCo contacts. One line per hand, solid for two hands and
dashed for one, so the gap between the pair IS the result.

Okabe-Ito colours (published CVD guarantee; the skill's validator is not
installed here). Hands are also distinguished by marker, so identity is never
carried by colour alone.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HANDS = ["shadow", "leap", "allegro", "f5d6"]
COLORS = {"shadow": "#0072B2", "leap": "#009E73",
          "allegro": "#E69F00", "f5d6": "#CC79A7"}
MARKERS = {"shadow": "o", "leap": "s", "allegro": "^", "f5d6": "D"}


def load(pattern):
    rows = []
    for f in sorted(pathlib.Path(".").glob(pattern)):
        rows += json.loads(f.read_text())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="legacy/results/deficit_*.json")
    ap.add_argument("--out", default="legacy/figures/deficit_repair.png")
    a = ap.parse_args()
    rows = load(a.glob)
    if not rows:
        raise SystemExit("no results")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for hand in HANDS:
        for nh, ls, alpha in ((1, "--", 0.75), (2, "-", 1.0)):
            pts = sorted((r["width"] * 100, r["hold_N"], r["epsilon"])
                         for r in rows
                         if r["hand"] == hand and r["n_hands"] == nh)
            if not pts:
                continue
            w, hold, eps = zip(*pts)
            for ax, y in ((axes[0], hold), (axes[1], eps)):
                ax.plot(w, y, ls, color=COLORS[hand], marker=MARKERS[hand],
                        markersize=5, linewidth=2.0 if nh == 2 else 1.5,
                        alpha=alpha, markerfacecolor=(COLORS[hand] if nh == 2
                                                      else "none"),
                        markeredgewidth=1.4,
                        label=f"{hand}" if nh == 2 else None)
    axes[0].set_ylabel("force sustained in the worst\ndirection (N)", fontsize=9)
    axes[1].set_ylabel("Ferrari-Canny $\\epsilon$\n(real MuJoCo contacts)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("box width (cm)", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    h, l = axes[0].get_legend_handles_labels()
    style = [plt.Line2D([], [], color="#555555", linestyle="--", linewidth=1.5,
                        label="one hand"),
             plt.Line2D([], [], color="#555555", linestyle="-", linewidth=2.0,
                        label="two hands")]
    fig.legend(handles=h + style, loc="lower center", ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.13))

    both = {}
    for r in rows:
        both.setdefault((r["hand"], r["width"]), {})[r["n_hands"]] = r
    pairs = [(v[1], v[2]) for v in both.values() if 1 in v and 2 in v]
    gain = [(b["hold_N"], o["hold_N"]) for o, b in pairs
            if o["hold_N"] > 0 or b["hold_N"] > 0]
    better = sum(1 for two, one in gain if two > one)
    fig.suptitle(f"A second hand raises the sustained wrench in "
                 f"{better} of {len(gain)} hand x width cases", fontsize=11,
                 y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"wrote {a.out}")

    print(f"\n{'hand':8s} {'w':>5s} {'eps1':>7s} {'eps2':>7s} "
          f"{'hold1':>6s} {'hold2':>6s}")
    for (hand, w), v in sorted(both.items()):
        if 1 in v and 2 in v:
            print(f"{hand:8s} {w*100:5.1f} {v[1]['epsilon']:7.4f} "
                  f"{v[2]['epsilon']:7.4f} {v[1]['hold_N']:6.2f} "
                  f"{v[2]['hold_N']:6.2f}")
    one = np.array([o["hold_N"] for o, b in pairs])
    two = np.array([b["hold_N"] for o, b in pairs])
    print(f"\nmean sustained force: one hand {one.mean():.2f} N, "
          f"two hands {two.mean():.2f} N  ({two.mean()/max(one.mean(),1e-9):.2f}x)")
    print(f"two hands >= one in {int((two >= one).sum())}/{len(one)} cells, "
          f"strictly greater in {int((two > one).sum())}")


if __name__ == "__main__":
    main()
