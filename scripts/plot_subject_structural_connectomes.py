#!/usr/bin/env python3
"""Plot every subject connectome, one publication-style figure per cohort."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COHORT_ORDER = ("control", "emcs", "mcs", "uws", "coma")
def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _positive_limits(
    dataset_root: Path,
    cohorts: list[str],
    data_key: str,
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    positive = []
    for cohort in cohorts:
        with np.load(dataset_root / f"subjects_{cohort}.npz") as data:
            x = np.asarray(data[data_key], dtype=float)
        positive.append(x[x > 0.0])
    pooled = np.concatenate(positive)
    # Robust display range only. The underlying matrices are not transformed.
    return (
        float(np.percentile(pooled, lower_percentile)),
        float(np.percentile(pooled, upper_percentile)),
    )


def plot_cohort(
    dataset_root: Path,
    output_dir: Path,
    cohort: str,
    *,
    data_key: str,
    cmap: str,
    colorbar_label: str,
    filename_prefix: str,
    logarithmic: bool,
    vmin: float,
    vmax: float,
) -> None:
    with np.load(dataset_root / f"subjects_{cohort}.npz") as data:
        ids = np.asarray(data["subject_ids"]).astype(str)
        matrices = np.asarray(data[data_key], dtype=float)

    n = len(ids)
    ncols = min(10, max(4, math.ceil(math.sqrt(n * 1.25))))
    nrows = math.ceil(n / ncols)
    figure_height = 1.48 * nrows + 1.05
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.42 * ncols, figure_height),
        constrained_layout=False,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    norm = (
        mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        if logarithmic
        else mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    )
    image = None
    for ax, sid, matrix in zip(axes, ids, matrices):
        masked = np.ma.masked_less_equal(matrix, 0.0)
        image = ax.imshow(
            masked,
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        ax.axhline(44.5, color="white", lw=0.35, alpha=0.85)
        ax.axvline(44.5, color="white", lw=0.35, alpha=0.85)
        subject_number = str(int("".join(ch for ch in str(sid) if ch.isdigit())))
        ax.set_title(subject_number, fontsize=8, pad=2, color="#111111")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")

    # Reserve a fixed amount of physical space for the colour bar. Using fixed
    # figure fractions clipped the label for small cohorts such as COMA.
    colourbar_bottom = 0.62 / figure_height
    colourbar_height = 0.18 / figure_height
    matrix_bottom = 1.25 / figure_height
    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        bottom=matrix_bottom,
        top=0.965,
        wspace=0.18,
        hspace=0.24,
    )
    bar_width = min(0.46, max(0.28, 0.052 * ncols))
    cax = fig.add_axes(
        [0.5 - bar_width / 2.0, colourbar_bottom, bar_width, colourbar_height]
    )
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.set_label(colorbar_label, fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9, width=0.7, length=3, pad=2)
    cb.outline.set_linewidth(0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{filename_prefix}_{cohort}.{suffix}",
            dpi=300 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.10,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    scheme = index.get("connectivity_normalization", {}).get("scheme")
    if scheme != "native_invnodevol":
        raise ValueError(f"Expected native_invnodevol dataset, found {scheme!r}.")
    cohorts = [c for c in COHORT_ORDER if c in index["cohorts"]]
    sc_vmin, sc_vmax = _positive_limits(
        root,
        cohorts,
        "connectivity",
        lower_percentile=1.0,
        upper_percentile=99.5,
    )
    tl_vmin, tl_vmax = _positive_limits(
        root,
        cohorts,
        "tract_lengths",
        lower_percentile=0.0,
        upper_percentile=99.5,
    )
    for cohort in cohorts:
        plot_cohort(
            root,
            args.output_dir,
            cohort,
            data_key="connectivity",
            cmap="YlOrRd",
            colorbar_label="Inverse-node-volume SC weight",
            filename_prefix="structural_connectomes",
            logarithmic=True,
            vmin=sc_vmin,
            vmax=sc_vmax,
        )
        plot_cohort(
            root,
            args.output_dir,
            cohort,
            data_key="tract_lengths",
            cmap="bone_r",
            colorbar_label="Tract length (mm)",
            filename_prefix="tract_lengths",
            logarithmic=False,
            vmin=tl_vmin,
            vmax=tl_vmax,
        )
    print(f"Wrote {2 * len(cohorts)} cohort figures to {args.output_dir}")
    print(f"SC shared display limits: {sc_vmin:.8g} to {sc_vmax:.8g}")
    print(f"Tract-length shared display limits: {tl_vmin:.8g} to {tl_vmax:.8g} mm")


if __name__ == "__main__":
    _style()
    main()
