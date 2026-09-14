"""The opposition axis figure: what each hand can reach, and what it can hold."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8b8a85"
HELD      = "#2a78d6"   # series 1
DROPPED   = "#eb6834"   # series 2  (validated vs series 1: CVD dE 24.7, normal 33.6)
INFEAS    = "#e4e3df"

from oppdef.paths import RESULTS, FIGURES
a = json.load(open(RESULTS / "hand_axis.json"))
a.update(json.load(open(RESULTS / "hand_axis_f5d6.json")))
order = sorted(a, key=lambda k: a[k]["closed_gap_m"])
widths = [r["width"] * 100 for r in a[order[0]]["rows"]]
LABEL = {"shadow": "Shadow\n24 DoF", "leap": "LEAP\n16 DoF",
         "allegro": "Allegro\n16 DoF", "f5d6": "Dexmate f5d6\n11 DoF"}
FLOOR = {"shadow": 0.00, "leap": 0.00, "allegro": 0.00, "f5d6": 3.08}

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(11.2, 3.9), gridspec_kw=dict(width_ratios=[1, 1.85], wspace=0.42))
fig.patch.set_facecolor(SURFACE)
for ax in (axL, axR):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)

# ---- left: how far the hand can close -------------------------------------
y = np.arange(len(order))[::-1]
gaps = [a[k]["closed_gap_m"] * 100 for k in order]
axL.barh(y, gaps, height=0.52, color=HELD, zorder=3)
for yi, g, k in zip(y, gaps, order):
    axL.text(g + 0.12, yi, f"{g:.2f}", va="center", ha="left",
             fontsize=10, color=INK, zorder=4)
    if FLOOR[k] > 0:
        axL.text(g + 0.95, yi, f"(floor {FLOOR[k]:.2f})", va="center",
                 ha="left", fontsize=8.5, color=INK_MUTED, zorder=4)
axL.set_yticks(y)
axL.set_yticklabels([LABEL[k] for k in order], fontsize=9.5, color=INK)
axL.set_xlim(0, 5.0)
axL.set_xticks([0, 1, 2, 3, 4])
axL.tick_params(axis="x", colors=INK_2, labelsize=9, length=0)
axL.tick_params(axis="y", length=0)
axL.grid(axis="x", color="#e8e7e3", lw=0.8, zorder=0)
axL.set_axisbelow(True)
axL.set_xlabel("closest thumb-to-finger gap  (cm)", fontsize=9.5, color=INK_2)
axL.set_title("How far the hand can close", fontsize=11, color=INK,
              loc="left", pad=10)

# ---- right: which block widths it actually holds ---------------------------
for row, k in enumerate(order):
    yy = len(order) - 1 - row
    for col, r in enumerate(a[k]["rows"]):
        if not r.get("feasible"):
            c, glyph, gc = INFEAS, "–", INK_MUTED
        elif r.get("held"):
            c, glyph, gc = HELD, "✓", SURFACE
        else:
            c, glyph, gc = DROPPED, "✗", SURFACE
        axR.add_patch(plt.Rectangle((col + 0.06, yy - 0.30), 0.88, 0.60,
                                    facecolor=c, edgecolor=SURFACE, lw=1.4))
        axR.text(col + 0.5, yy, glyph, ha="center", va="center",
                 fontsize=11, color=gc, zorder=5)
    n_held = sum(1 for r in a[k]["rows"] if r.get("held"))
    axR.text(len(widths) + 0.28, yy, f"{n_held}/8", ha="left", va="center",
             fontsize=10.5, color=INK)

axR.set_xlim(0, len(widths) + 1.1)
axR.set_ylim(-0.6, len(order) - 0.4)
axR.set_xticks(np.arange(len(widths)) + 0.5)
axR.set_xticklabels([f"{w:.0f}" for w in widths], fontsize=9, color=INK_2)
axR.set_yticks([])
axR.tick_params(length=0)
axR.set_xlabel("block width  (cm)", fontsize=9.5, color=INK_2)
axR.set_title("Which blocks it holds against gravity, nothing underneath",
              fontsize=11, color=INK, loc="left", pad=10)
axR.legend(handles=[Patch(facecolor=HELD, label="held  ✓"),
                    Patch(facecolor=DROPPED, label="dropped  ✗"),
                    Patch(facecolor=INFEAS, label="cannot close that far  –")],
           loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=3, frameon=False,
           fontsize=9, labelcolor=INK_2, handlelength=1.1, columnspacing=1.6)

fig.text(0.008, 0.965,
         "The opposition floor predicts what a hand can hold",
         fontsize=13.5, color=INK, weight="bold", va="top")
fig.text(0.008, 0.90,
         "Four hands, one identical derived closure and one shared set of block widths. "
         "Ordering is the same on both panels.",
         fontsize=9.5, color=INK_2, va="top")
fig.subplots_adjust(left=0.105, right=0.975, top=0.70, bottom=0.235)
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "16_opposition_axis.png", dpi=200, facecolor=SURFACE)
print(f"wrote {FIGURES / '16_opposition_axis.png'}")
