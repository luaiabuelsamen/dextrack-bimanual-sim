"""Figure: the claim, on an outcome epsilon did not choose.

Left, how often each retargeting objective yields a grasp that survives ANY
six-dimensional wrench disturbance -- the pre-specified outcome. Right, the
second hand's gain per newton it actually applies, which is the control that
separates allocation from budget.
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POSE, WRENCH = "#E69F00", "#0072B2"


def main():
    rows = []
    for f in sorted(glob.glob("results/conf_*.json")):
        rows += json.loads(pathlib.Path(f).read_text())
    cells = {}
    for r in rows:
        cells.setdefault((r["hand"], r["width"], r["mass"]), {})[r["condition"]] = r
    keys = [k for k, v in cells.items() if len(v) == 2]
    P = np.array([cells[k]["pose"]["hold_N"] > 0 for k in keys])
    W = np.array([cells[k]["wrench"]["hold_N"] > 0 for k in keys])
    H = np.array([k[0] for k in keys])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    hands = ["allegro", "f5d6"]
    labels = [f"{h}\nfloor {'2.41' if h=='allegro' else '3.40'} cm" for h in hands]
    x = np.arange(len(hands) + 1)
    pv = [P[H == h].mean() * 100 for h in hands] + [P.mean() * 100]
    wv = [W[H == h].mean() * 100 for h in hands] + [W.mean() * 100]
    ns = [int((H == h).sum()) for h in hands] + [len(P)]
    ax.bar(x - 0.19, pv, 0.36, color=POSE, label="retarget the POSE")
    ax.bar(x + 0.19, wv, 0.36, color=WRENCH, label="retarget the WRENCH")
    for i, (a, b, n) in enumerate(zip(pv, wv, ns)):
        ax.text(i - 0.19, a + 2, f"{a:.0f}%", ha="center", fontsize=8)
        ax.text(i + 0.19, b + 2, f"{b:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn={n}" for l, n in
                        zip(labels + ["both (pooled)"], ns)], fontsize=8.5)
    ax.set_ylabel("cells yielding a grasp that survives\nany 6-D wrench (%)",
                  fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_title("McNemar exact $p=0.0010$  (14 discordant vs 1)", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    ax = axes[1]
    drows = []
    for f in sorted(glob.glob("results/deficit_*.json")):
        drows += json.loads(pathlib.Path(f).read_text())
    d = {}
    for r in drows:
        d.setdefault((r["hand"], r["width"]), {})[r["n_hands"]] = r
    one, two = [], []
    for v in d.values():
        if 1 in v and 2 in v and v[1]["f_total"] > 0 and v[2]["f_total"] > 0:
            one.append(v[1]["hold_N"] / v[1]["f_total"])
            two.append(v[2]["hold_N"] / v[2]["f_total"])
    one, two = np.array(one), np.array(two)
    ax.scatter(one, two, s=34, color="#009E73", edgecolor="white", zorder=3)
    lim = max(one.max(), two.max()) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="#999999", linewidth=1.2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("one hand: wrench held per newton applied", fontsize=9)
    ax.set_ylabel("two hands", fontsize=9)
    ax.set_title(f"second hand: {two.mean()/one.mean():.2f}x per newton "
                 f"({int((two>one).sum())}/{len(one)} cells)", fontsize=9.5)
    for a in axes:
        a.grid(alpha=0.25, linewidth=0.6); a.set_axisbelow(True)
        a.tick_params(labelsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    pathlib.Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/thesis.png", dpi=170, bbox_inches="tight")
    print("wrote figures/thesis.png")


if __name__ == "__main__":
    main()
