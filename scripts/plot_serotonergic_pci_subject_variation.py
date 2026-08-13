#!/usr/bin/env python3
"""Plot condition-average and subject-level serotonergic PCI trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
CONDITION_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
    "EMCS": "#E8B56D",
    "CNT": "#5B8A72",
}
REQUIRED_COLUMNS = {"condition", "subject_id", "occupancy", "pci_mean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/serotonergic_pci_3per_condition_10trials/"
            "tables/serotonergic_pci_subject_metrics.csv"
        ),
        help="Subject-level PCI CSV produced by the serotonergic PCI workflow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/serotonergic_pci_3per_condition_10trials/figures/subject_variation"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="serotonergic_pci_subject_variation",
    )
    return parser.parse_args()


def _sem(values: pd.Series) -> float:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def load_subject_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["condition"] = df["condition"].astype(str).str.upper()
    unknown_conditions = sorted(
        set(df["condition"].dropna()).difference(CONDITION_ORDER)
    )
    if unknown_conditions:
        raise ValueError(
            f"Unrecognized conditions in subject metrics: {unknown_conditions}"
        )

    df["occupancy"] = pd.to_numeric(df["occupancy"], errors="raise")
    df["pci_mean"] = pd.to_numeric(df["pci_mean"], errors="raise")
    if not np.isfinite(df[["occupancy", "pci_mean"]].to_numpy(float)).all():
        raise ValueError("occupancy and pci_mean must contain only finite values.")

    duplicate = df.duplicated(
        ["condition", "subject_id", "occupancy"],
        keep=False,
    )
    if duplicate.any():
        example = df.loc[
            duplicate,
            ["condition", "subject_id", "occupancy"],
        ].iloc[0]
        raise ValueError(
            f"Duplicate subject/condition/occupancy row: {example.to_dict()}"
        )

    df["condition"] = pd.Categorical(
        df["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    return df.sort_values(["condition", "subject_id", "occupancy"]).reset_index(
        drop=True
    )


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["condition", "occupancy"], observed=True)
        .agg(
            pci_mean=("pci_mean", "mean"),
            pci_sem=("pci_mean", _sem),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    summary["condition"] = pd.Categorical(
        summary["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    return summary.sort_values(["condition", "occupancy"]).reset_index(drop=True)


def _padded_y_limits(values: pd.Series) -> tuple[float, float]:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    span = max(data_max - data_min, 0.1)
    lower = np.floor((data_min - 0.07 * span) / 0.05) * 0.05
    upper = np.ceil((data_max + 0.07 * span) / 0.05) * 0.05
    return max(0.0, float(lower)), min(1.0, float(upper))


def _format_occupancy(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    if np.isclose(value, 0.25):
        return "0.25"
    if np.isclose(value, 0.5):
        return "0.50"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.55)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.4, alpha=0.72)


def _draw_group_mean(
    ax: mpl.axes.Axes,
    group: pd.DataFrame,
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 1.2,
    markersize: float = 3.0,
) -> None:
    x = group["occupancy"].to_numpy(float)
    mean = group["pci_mean"].to_numpy(float)
    error = group["pci_sem"].to_numpy(float)
    ax.fill_between(
        x,
        mean - error,
        mean + error,
        color=color,
        alpha=0.15,
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        x,
        mean,
        color=color,
        linewidth=linewidth,
        marker="o",
        markersize=markersize,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label=label,
        zorder=5,
    )


def figure_note(df: pd.DataFrame) -> str:
    counts = df.groupby("condition", observed=True)["subject_id"].nunique()
    count_text = ", ".join(
        f"{condition} n={int(counts.get(condition, 0))}"
        for condition in CONDITION_ORDER
    )
    if "n_trials" in df.columns:
        trials = sorted({int(value) for value in df["n_trials"].dropna().unique()})
        if len(trials) == 1:
            trial_text = f" PCI estimates use {trials[0]} aligned trials."
        elif trials:
            trial_text = (
                f" PCI estimates use {min(trials)}-{max(trials)} aligned trials."
            )
        else:
            trial_text = ""
    else:
        trial_text = ""
    return f"{count_text}.{trial_text}"


def build_subject_variation_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    occupancies = np.array(sorted(df["occupancy"].unique()), dtype=float)
    occupancy_labels = [_format_occupancy(value) for value in occupancies]
    subject_y_limits = _padded_y_limits(df["pci_mean"])
    mean_extent = pd.Series(
        np.concatenate(
            [
                (summary["pci_mean"] - summary["pci_sem"]).to_numpy(float),
                (summary["pci_mean"] + summary["pci_sem"]).to_numpy(float),
            ]
        )
    )
    mean_y_limits = _padded_y_limits(mean_extent)

    figure_width = 183.0 / 25.4
    figure_height = 5.35
    fig = plt.figure(
        figsize=(figure_width, figure_height),
        constrained_layout=False,
    )
    grid = fig.add_gridspec(
        2,
        len(CONDITION_ORDER),
        height_ratios=[1.15, 1.0],
        left=0.080,
        right=0.985,
        bottom=0.105,
        top=0.955,
        hspace=0.46,
        wspace=0.24,
    )

    mean_ax = fig.add_subplot(grid[0, :])
    for condition in CONDITION_ORDER:
        group = summary[summary["condition"].eq(condition)]
        _draw_group_mean(
            mean_ax,
            group,
            color=CONDITION_COLORS[condition],
            label=condition,
        )

    mean_ax.set_title("Condition means", loc="left", pad=4)
    mean_ax.set_ylabel("PCI")
    mean_ax.set_xlim(float(occupancies.min()) - 0.025, float(occupancies.max()) + 0.025)
    mean_ax.set_ylim(*mean_y_limits)
    mean_ax.set_xticks(occupancies, occupancy_labels)
    mean_ax.set_xlabel(r"5-HT$_{2A}$ occupancy")
    mean_ax.legend(
        frameon=False,
        ncol=len(CONDITION_ORDER),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.03),
        handlelength=1.8,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    _style_axis(mean_ax)
    mean_ax.text(
        -0.065,
        1.045,
        "a",
        transform=mean_ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )

    subject_axes: list[mpl.axes.Axes] = []
    for index, condition in enumerate(CONDITION_ORDER):
        if index == 0:
            ax = fig.add_subplot(grid[1, index])
        else:
            ax = fig.add_subplot(
                grid[1, index],
                sharex=subject_axes[0],
                sharey=subject_axes[0],
            )
        subject_axes.append(ax)

        color = CONDITION_COLORS[condition]
        condition_df = df[df["condition"].eq(condition)]
        for _, subject in condition_df.groupby("subject_id", observed=True):
            ax.plot(
                subject["occupancy"].to_numpy(float),
                subject["pci_mean"].to_numpy(float),
                color=color,
                alpha=0.16,
                linewidth=0.45,
                linestyle=(0, (1.4, 2.2)),
                marker="o",
                markersize=1.4,
                markeredgewidth=0,
                zorder=1,
            )

        group = summary[summary["condition"].eq(condition)]
        _draw_group_mean(
            ax,
            group,
            color=color,
            linewidth=1.15,
            markersize=2.8,
        )
        n_subjects = int(condition_df["subject_id"].nunique())
        ax.set_title(
            f"{condition}  (n={n_subjects})",
            color="#222222",
            fontweight="normal",
            pad=3,
        )
        ax.set_facecolor("white")
        ax.set_xlim(
            float(occupancies.min()) - 0.035,
            float(occupancies.max()) + 0.035,
        )
        ax.set_ylim(*subject_y_limits)
        ax.set_xticks(occupancies, occupancy_labels, rotation=45, ha="right")
        if index == 0:
            ax.set_ylabel("PCI")
        else:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)
        _style_axis(ax)

    fig.text(
        0.080,
        0.485,
        "b",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="center",
    )
    fig.text(
        0.101,
        0.485,
        "Individual trajectories",
        fontsize=6.5,
        ha="left",
        va="center",
    )
    fig.text(
        0.535,
        0.035,
        r"5-HT$_{2A}$ occupancy",
        fontsize=6.2,
        ha="center",
        va="center",
    )
    return fig


def save_outputs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_subject_variation_figure(df, summary)
    outputs: list[Path] = []
    for extension in ("pdf", "svg"):
        path = output_dir / f"{prefix}.{extension}"
        figure.savefig(path)
        outputs.append(path)
    png_path = output_dir / f"{prefix}.png"
    figure.savefig(png_path, dpi=600)
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
    outputs.append(png_path)
    tiff_path = output_dir / f"{prefix}.tiff"
    figure.savefig(
        tiff_path,
        dpi=300,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(tiff_path) as image:
        image.convert("RGB").save(
            tiff_path,
            dpi=(300, 300),
            compression="tiff_lzw",
        )
    outputs.append(tiff_path)
    plt.close(figure)

    summary_path = output_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)
    legend_path = output_dir / f"{prefix}_legend.md"
    legend_path.write_text(
        "# Proposed figure legend\n\n"
        "**Diagnosis-level and individual serotonergic PCI dose responses.** "
        "**a,** Mean PCI at each 5-HT2A occupancy; shaded bands show s.e.m. "
        "**b,** Individual subject trajectories within each diagnosis. Thin "
        "dotted lines show subjects; thick lines and shaded bands show the "
        f"diagnosis mean and s.e.m. {figure_note(df)}\n",
        encoding="utf-8",
    )
    outputs.append(legend_path)
    manifest_path = output_dir / f"{prefix}_format_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "figure_width_mm": 183,
                "figure_height_mm": 5.35 * 25.4,
                "font": "Arial",
                "font_size_pt_range": [5.5, 8.0],
                "line_width_pt_range": [0.4, 1.2],
                "vector_exports": ["pdf", "svg"],
                "raster_exports": {"png_dpi": 600, "tiff_dpi": 300},
                "colour_space": "RGB",
                "diagnosis_colours": CONDITION_COLORS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    metrics = load_subject_metrics(args.input)
    summary = condition_summary(metrics)
    outputs = save_outputs(
        metrics,
        summary,
        args.output_dir,
        args.prefix,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
