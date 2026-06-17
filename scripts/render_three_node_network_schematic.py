#!/usr/bin/env python3
"""Render the three-node AdEx motif and structural connectivity matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


WEIGHTS = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)


def render(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(10.6, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)

    ax_net = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[0, 1])

    # Left: motif schematic
    coords = {
        1: np.array([0.20, 0.64]),
        2: np.array([0.80, 0.64]),
        3: np.array([0.50, 0.16]),
    }
    node_face = "#F6F1E8"
    node_edge = "#2D3748"
    coupled_edge = "#4A6C8C"
    uncoupled_edge = "#CFC7B8"

    # faint absent links for visual contrast
    for a, b in [(1, 3), (2, 3)]:
        p0 = coords[a]
        p1 = coords[b]
        ax_net.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            linestyle=(0, (2, 4)),
            color=uncoupled_edge,
            linewidth=1.6,
            alpha=0.8,
            zorder=1,
        )

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
    ax_net.add_patch(arrow)

    for idx, center in coords.items():
        circ = Circle(center, 0.09, facecolor=node_face, edgecolor=node_edge, linewidth=2.0, zorder=3)
        ax_net.add_patch(circ)
        ax_net.text(center[0], center[1], str(idx), ha="center", va="center", fontsize=16, fontweight="bold", color=node_edge)

    ax_net.text(0.50, 0.75, "Bidirectional structural coupling", ha="center", va="bottom", fontsize=11.5, color=coupled_edge)
    ax_net.text(0.50, 0.02, "Node 3 structurally uncoupled", ha="center", va="bottom", fontsize=11.5, color="#7A6F63")
    ax_net.text(0.50, 0.92, r"Effective coupling = $G \times W_{ij}$", ha="center", va="center", fontsize=12.0, color="#2D3748")
    ax_net.set_title("Three-node AdEx motif", pad=24)
    ax_net.set_xlim(0.0, 1.0)
    ax_net.set_ylim(0.0, 1.0)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    # Right: exact structural weights
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "motif_weights", ["#F8F4EC", "#DCE5EB", "#7A98AF", "#35556C"]
    )
    im = ax_mat.imshow(WEIGHTS, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
    ax_mat.set_title("Structural weight matrix $W$", pad=24)
    ax_mat.set_xticks([0, 1, 2])
    ax_mat.set_xticklabels(["1", "2", "3"])
    ax_mat.set_yticks([0, 1, 2])
    ax_mat.set_yticklabels(["1", "2", "3"])
    ax_mat.set_xlabel("Target node")
    ax_mat.set_ylabel("Source node")
    for spine in ax_mat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#1F2937")
    for i in range(3):
        for j in range(3):
            val = WEIGHTS[i, j]
            ax_mat.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=13, color="#1F2937")
    cbar = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    cbar.set_label("Weight")

    fig.subplots_adjust(top=0.83, left=0.06, right=0.97, bottom=0.12, wspace=0.20)

    for suffix, transparent in [("png", False), ("pdf", False), ("svg", False), ("transparent.png", True)]:
        fig.savefig(out_dir / f"three_node_network_schematic.{suffix}", dpi=300, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / "three_node_adex_phiid_g_noise_sweep_30x30" / "figures"
    render(out_dir)


if __name__ == "__main__":
    main()
