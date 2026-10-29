#!/usr/bin/env python3
"""Re-plot Table II of the paper (exact reported numbers) as a four-quadrant outcome chart.

    python3 tools/plot_paper_results.py --out assets/readme/table2_outcomes.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight_safety.paper_results import TABLE_II  # noqa: E402

QUADS = [("SSR", "Safe success", "#2a78d6"), ("USR", "Unsafe success", "#eb6834"),
         ("SFR", "Safe failure", "#1baf7a"), ("UFR", "Unsafe failure", "#eda100")]
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/readme/table2_outcomes.png")
    args = ap.parse_args()
    models = list(TABLE_II)[::-1]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for y, m in enumerate(models):
        left = 0.0
        for key, _, color in QUADS:
            v = TABLE_II[m][key]
            ax.barh(y, v, left=left, height=0.62, color=color, edgecolor="white", linewidth=2)
            if v >= 0.06:
                ax.text(left + v / 2, y, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color=INK)
            left += v
        ax.text(1.03, y, f"{TABLE_II[m]['CC']:.2f}", ha="center", va="center", fontsize=10, color=INK,
                transform=ax.get_yaxis_transform())
        ax.text(1.12, y, f"{TABLE_II[m]['SASR']:.2f}", ha="center", va="center", fontsize=10, color=INK,
                transform=ax.get_yaxis_transform())
    top = len(models) - 0.35
    ax.text(1.03, top, "CC", ha="center", va="bottom", fontsize=10, color=MUTED, transform=ax.get_yaxis_transform())
    ax.text(1.12, top, "SASR", ha="center", va="bottom", fontsize=10, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(models)), models, fontsize=10.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"], fontsize=9, color=MUTED)
    ax.set_xlabel("Share of episodes", fontsize=9.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in QUADS]
    ax.legend(handles, [f"{lab} ({k})" for k, lab, _ in QUADS], ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02),
              frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.6)
    ax.set_title("Paper Table II: four-quadrant outcomes on the 66 scenarios (L0/W0/V0), 50 episodes per cell",
                 fontsize=11, color=INK, loc="left", pad=30)
    fig.tight_layout()
    fig.savefig(args.out, facecolor="white", bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
