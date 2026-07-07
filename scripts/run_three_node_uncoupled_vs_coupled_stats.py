#!/usr/bin/env python3
"""Plane-level tests for uncoupled STS > coupled STS in the three-node sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from tvbtoolkit.analysis import plot_three_node_hypothesis_clean
from tvbtoolkit.core.paths import legacy_results

NATIVE_NOISE_RANGE = (5e-5, 3e-4)


def _build_contrast_frame(avg_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sts = avg_df.loc[avg_df["atom"] == "sts"].copy()
    for _, row in sts.iterrows():
        mat = np.asarray(row["matrix"], dtype=float)
        rows.append(
            {
                "measure": str(row["measure"]),
                "g_value": float(row["g_value"]),
                "noise_value": float(row["noise_value"]),
                "coupled_sts": float(mat[0, 1]),
                "uncoupled_13_sts": float(mat[0, 2]),
                "uncoupled_23_sts": float(mat[1, 2]),
                "uncoupled_mean_sts": float((mat[0, 2] + mat[1, 2]) / 2.0),
                "contrast_uncoupled_minus_coupled": float((mat[0, 2] + mat[1, 2]) / 2.0 - mat[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def _sign_flip_pvalue(values: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = float(np.mean(values))
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, values.size))
    null = np.mean(flips * values[None, :], axis=1)
    return float(((null >= obs).sum() + 1) / (n_perm + 1))


def _summarize_tests(contrast_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for measure in ["mmi", "ccs"]:
        for subset_name, sub in [
            ("all_parameter_cells", contrast_df.loc[contrast_df["measure"] == measure].copy()),
            (
                "native_noise_band",
                contrast_df.loc[
                    (contrast_df["measure"] == measure)
                    & (contrast_df["noise_value"] >= NATIVE_NOISE_RANGE[0])
                    & (contrast_df["noise_value"] <= NATIVE_NOISE_RANGE[1])
                ].copy(),
            ),
        ]:
            vals = sub["contrast_uncoupled_minus_coupled"].to_numpy(dtype=float)
            positive = int((vals > 0).sum())
            n = int(vals.size)
            rows.append(
                {
                    "measure": measure,
                    "subset": subset_name,
                    "n_cells": n,
                    "n_positive_cells": positive,
                    "positive_fraction": float(positive / n),
                    "mean_contrast": float(vals.mean()),
                    "median_contrast": float(np.median(vals)),
                    "binom_p_greater_than_half": float(binomtest(positive, n, p=0.5, alternative="greater").pvalue),
                    "signflip_perm_p_mean_greater_than_zero": _sign_flip_pvalue(vals, n_perm=20000, seed=0),
                }
            )
    return pd.DataFrame(rows)


def _annotate_figure(fig: plt.Figure, axes: np.ndarray, stats_df: pd.DataFrame) -> None:
    for row_idx, measure in enumerate(["mmi", "ccs"]):
        all_row = stats_df.loc[(stats_df["measure"] == measure) & (stats_df["subset"] == "all_parameter_cells")].iloc[0]
        native_row = stats_df.loc[(stats_df["measure"] == measure) & (stats_df["subset"] == "native_noise_band")].iloc[0]
        text = (
            f"All cells: {all_row['n_positive_cells']}/{all_row['n_cells']} positive "
            f"({100*all_row['positive_fraction']:.1f}%)\n"
            f"Binomial p = {all_row['binom_p_greater_than_half']:.2e}\n"
            f"Mean contrast p = {all_row['signflip_perm_p_mean_greater_than_zero']:.2e}\n"
            f"Native band: {native_row['n_positive_cells']}/{native_row['n_cells']} positive "
            f"({100*native_row['positive_fraction']:.1f}%)"
        )
        ax = axes[row_idx, 3]
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="#1F2937",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": (1, 1, 1, 0.84), "edgecolor": "#D0C7B7", "linewidth": 0.8},
        )


def main() -> None:
    root = legacy_results("results", "three_node_adex_phiid_g_noise_sweep_30x30")
    avg_df = pd.read_pickle(root / "tables" / "averaged_matrices.pkl")
    contrast_df = _build_contrast_frame(avg_df)
    stats_df = _summarize_tests(contrast_df)

    out_tables = root / "stats"
    out_tables.mkdir(parents=True, exist_ok=True)
    contrast_df.to_csv(out_tables / "uncoupled_vs_coupled_sts_cellwise.csv", index=False)
    stats_df.to_csv(out_tables / "uncoupled_vs_coupled_sts_plane_tests.csv", index=False)

    fig, axes = plot_three_node_hypothesis_clean(avg_df, normalize_sts=True)
    _annotate_figure(fig, axes, stats_df)
    out_figs = root / "figures"
    for suffix, transparent in [("png", False), ("pdf", False), ("svg", False), ("transparent.png", True)]:
        fig.savefig(out_figs / f"hypothesis_clean_summary_stats.{suffix}", dpi=300, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


if __name__ == "__main__":
    main()
