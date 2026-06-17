#!/usr/bin/env python3
"""Render the cohort-composition figure for a Canva presentation slide.

Produces:
    notebooks/figs/cohort_breakdown_white.png  (white text — for dark slides)
    notebooks/figs/cohort_breakdown_black.png  (black text — for light slides)

Both PNGs have transparent backgrounds at 600 DPI.  The chart uses the
project's canonical condition colour palette (COND_COLORS) and orders
cohorts along the DoC severity axis: COMA < UWS < MCS < EMCS < CNT.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COHORT_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
COND_COLORS = {
    "COMA": "#3B4A6B",
    "UWS":  "#8B6B8B",
    "MCS":  "#C5622F",
    "EMCS": "#E8B56D",
    "CNT":  "#5B8A72",
}

# (cohort, n_non_sedated, n_sedated)  from data/doc_patients_new_data
COHORT_DATA = {
    "COMA": (6,  4),
    "UWS":  (29, 22),
    "MCS":  (29, 46),
    "EMCS": (7,  11),
    "CNT":  (35, 0),
}


def _lighten(hex_color: str, amount: float = 0.55) -> str:
    """Return a lighter version of `hex_color` by blending it toward white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def render(out_path: Path, fg: str) -> None:
    rc = {
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":            13,
        "axes.labelsize":       13,
        "axes.titlesize":       15,
        "xtick.labelsize":      13,
        "ytick.labelsize":      11,
        "axes.linewidth":       1.0,
        "xtick.major.size":     0,
        "ytick.major.size":     3,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "savefig.transparent":  True,
        "savefig.dpi":          600,
    }

    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)

        x = np.arange(len(COHORT_ORDER))
        width = 0.62

        for i, coh in enumerate(COHORT_ORDER):
            n_non, n_sed = COHORT_DATA[coh]
            color = COND_COLORS[coh]
            light = _lighten(color, 0.55)

            # non-sedated stack (bottom, full saturation)
            ax.bar(x[i], n_non, width=width, color=color,
                   edgecolor=fg, linewidth=0.6, zorder=3,
                   label="non-sedated" if i == 0 else None)
            # sedated stack (top, lighter)
            if n_sed > 0:
                ax.bar(x[i], n_sed, width=width, bottom=n_non, color=light,
                       edgecolor=fg, linewidth=0.6, zorder=3,
                       label="sedated" if i == 0 else None)

            # Per-segment counts inside bars (white text for contrast)
            if n_non >= 4:
                ax.text(x[i], n_non / 2, str(n_non),
                        ha="center", va="center", fontsize=12,
                        color="white", fontweight="bold", zorder=5)
            if n_sed >= 4:
                ax.text(x[i], n_non + n_sed / 2, str(n_sed),
                        ha="center", va="center", fontsize=12,
                        color=color, fontweight="bold", zorder=5)

            # Total label above bar
            total = n_non + n_sed
            ax.text(x[i], total + 2.0, f"n={total}",
                    ha="center", va="bottom", fontsize=12,
                    color=fg, fontweight="bold")

        ax.set_xticks(x, COHORT_ORDER)
        ax.set_ylabel("Number of subjects")
        ax.set_ylim(0, max(sum(d) for d in COHORT_DATA.values()) + 12)
        ax.set_title("Cohort composition  ·  N = 189", pad=12)

        # Custom legend (variant-free patches in neutral grey)
        from matplotlib.patches import Patch
        leg_handles = [
            Patch(facecolor="#888888", edgecolor=fg, linewidth=0.6, label="non-sedated"),
            Patch(facecolor="#cccccc", edgecolor=fg, linewidth=0.6, label="sedated"),
        ]
        legend = ax.legend(
            handles=leg_handles, loc="upper right",
            frameon=False, fontsize=11, handlelength=1.2, handleheight=1.2,
        )
        for text in legend.get_texts():
            text.set_color(fg)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=600, transparent=True,
                    bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
    print(f"saved → {out_path}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figs"
    render(out_dir / "cohort_breakdown_white.png", fg="white")
    render(out_dir / "cohort_breakdown_black.png", fg="black")


if __name__ == "__main__":
    main()
