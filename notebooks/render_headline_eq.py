#!/usr/bin/env python3
r"""Render the noise-mechanism headline equation to transparent PNGs.

Produces:
    notebooks/figs/headline_eq_white.png   (white text — for dark slides)
    notebooks/figs/headline_eq_black.png   (black text — for light slides)

Both have transparent backgrounds and are saved at 600 DPI for crisp
projection at any size.  Uses real LaTeX (usetex=True) so \underbrace
renders cleanly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


HEADLINE = (
    r"\nu_i^{\mathrm{drive}}(t) \;=\; "
    r"\underbrace{K\sum_{j} C_{ij}\,\nu_j^{E}(t - \tau_{ij})}_{"
    r"\text{TVB long-range coupling}} "
    r"\;+\; w_\eta\,"
    r"\underbrace{\!\Bigl[\sqrt{1-\alpha}\,\eta_i(t) + "
    r"\sqrt{\alpha}\,(\mathbf{M}\boldsymbol{\eta})_i(t)\Bigr]\!}_{"
    r"\text{private + shared OU drive (this work)}}"
)

OU = (
    r"\tau_{\mathrm{OU}}\, d\eta_i \;=\; -\eta_i\, dt \;+\; "
    r"\sigma\sqrt{2\tau_{\mathrm{OU}}}\, dW_i(t), "
    r"\qquad i = 1, \dots, N."
)


def render(out_path: Path, color: str) -> None:
    rc = {
        "text.usetex":            True,
        "text.latex.preamble":    r"\usepackage{amsmath}\usepackage{bm}",
        "font.family":            "serif",
        "font.serif":             ["Computer Modern Roman"],
        "font.size":              20,
        "savefig.transparent":    True,
        "savefig.dpi":            600,
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(14.0, 4.0))
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_facecolor("none")

        # Headline equation, with a thin frame to mimic the boxed look
        ax.text(
            0.5, 0.66,
            rf"${HEADLINE}$",
            ha="center", va="center",
            color=color, fontsize=20,
            transform=ax.transAxes,
            bbox=dict(
                facecolor="none",
                edgecolor=color,
                linewidth=0.8,
                boxstyle="round,pad=0.45",
            ),
        )

        # "with the OU process" caption + OU equation
        ax.text(
            0.05, 0.28,
            r"with the OU process",
            ha="left", va="center",
            color=color, fontsize=14,
            transform=ax.transAxes,
        )
        ax.text(
            0.5, 0.10,
            rf"${OU}$",
            ha="center", va="center",
            color=color, fontsize=18,
            transform=ax.transAxes,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=600,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close(fig)
    print(f"saved → {out_path}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figs"
    render(out_dir / "headline_eq_white.png", color="white")
    render(out_dir / "headline_eq_black.png", color="black")


if __name__ == "__main__":
    main()
