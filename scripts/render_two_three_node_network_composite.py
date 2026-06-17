#!/usr/bin/env python3
"""Render a composite schematic comparing the 2-node and 3-node AdEx motifs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


W2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
W3 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)


def _motif_panel(ax: plt.Axes, *, n_nodes: int) -> None:
    node_face = "#F6F1E8"
    node_edge = "#2D3748"
    coupled_edge = "#4A6C8C"
    uncoupled_edge = "#CFC7B8"

    if n_nodes == 2:
        coords = {
            1: np.array([0.24, 0.50]),
            2: np.array([0.76, 0.50]),
        }
        ax.text(0.50, 0.82, "Bidirectional structural coupling", ha="center", va="bottom", fontsize=11.0, color=coupled_edge)
        ax.text(0.50, 0.95, r"Effective coupling = $G \times W_{ij}$", ha="center", va="center", fontsize=12.0, color="#2D3748")
    else:
        coords = {
            1: np.array([0.20, 0.64]),
            2: np.array([0.80, 0.64]),
            3: np.array([0.50, 0.16]),
        }
        for a, b in [(1, 3), (2, 3)]:
            p0, p1 = coords[a], coords[b]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                linestyle=(0, (2, 4)),
                color=uncoupled_edge,
                linewidth=1.6,
                alpha=0.85,
                zorder=1,
            )
        ax.text(0.50, 0.75, "Bidirectional structural coupling", ha="center", va="bottom", fontsize=11.0, color=coupled_edge)
        ax.text(0.50, 0.02, "Node 3 structurally uncoupled", ha="center", va="bottom", fontsize=10.8, color="#7A6F63")
        ax.text(0.50, 0.92, r"Effective coupling = $G \times W_{ij}$", ha="center", va="center", fontsize=12.0, color="#2D3748")

    arrow = FancyArrowPatch(
        posA=coords[1],
        posB=coords[2],
        arrowstyle="<->",
        mutation_scale=18,
        linewidth=4.0,
        color=coupled_edge,
        shrinkA=22,
        shrinkB=22,
        zorder=2,
    )
    ax.add_patch(arrow)

    for idx, center in coords.items():
        circ = Circle(center, 0.09, facecolor=node_face, edgecolor=node_edge, linewidth=2.0, zorder=3)
        ax.add_patch(circ)
        ax.text(center[0], center[1], str(idx), ha="center", va="center", fontsize=16, fontweight="bold", color=node_edge)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def _matrix_panel(ax: plt.Axes, weights: np.ndarray) -> None:
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "motif_weights", ["#F8F4EC", "#DCE5EB", "#7A98AF", "#35556C"]
    )
    im = ax.imshow(weights, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
    n = weights.shape[0]
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i) for i in range(1, n + 1)])
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i) for i in range(1, n + 1)])
    ax.set_xlabel("Target node")
    ax.set_ylabel("Source node")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#1F2937")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{weights[i, j]:.0f}", ha="center", va="center", fontsize=13, color="#1F2937")
    return im


def render(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(10.8, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.07], hspace=0.36, wspace=0.18)

    motif2 = fig.add_subplot(gs[0, 0])
    mat2 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[0, 2])
    motif3 = fig.add_subplot(gs[1, 0])
    mat3 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[1, 2])

    _motif_panel(motif2, n_nodes=2)
    _motif_panel(motif3, n_nodes=3)
    im2 = _matrix_panel(mat2, W2)
    im3 = _matrix_panel(mat3, W3)

    motif2.set_title("Two-node AdEx motif", pad=18)
    mat2.set_title("Structural weight matrix $W$", pad=18)
    motif3.set_title("Three-node AdEx motif", pad=18)
    mat3.set_title("Structural weight matrix $W$", pad=18)

    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label("Weight")
    cbar3 = fig.colorbar(im3, cax=cax3)
    cbar3.set_label("Weight")

    motif2.text(-0.18, 0.5, "Two-node", transform=motif2.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold", color="#2B2B2B")
    motif3.text(-0.18, 0.5, "Three-node", transform=motif3.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold", color="#2B2B2B")

    fig.subplots_adjust(top=0.94, left=0.10, right=0.97, bottom=0.07, hspace=0.34, wspace=0.22)

    for suffix, transparent in [("png", False), ("pdf", False), ("svg", False), ("transparent.png", True)]:
        fig.savefig(out_dir / f"two_three_node_network_composite.{suffix}", dpi=300, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / "three_node_adex_phiid_g_noise_sweep_30x30" / "figures"
    render(out_dir)


if __name__ == "__main__":
    main()
