#!/usr/bin/env python3
"""Plot thresholded subject delta-PCI trajectories with responders highlighted."""

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
import seaborn as sns
from matplotlib.lines import Line2D
from PIL import Image

DISPLAY_CONDITIONS = ["COMA", "UWS", "MCS"]
CONDITION_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
}
MEAN_COLOR = "#25282B"
LEGEND_STYLE_COLOR = "#62676D"
GRID_COLOR = "#E7E4E0"
NATURE_DOUBLE_COLUMN_IN = 183.0 / 25.4
NATURE_SINGLE_COLUMN_IN = 89.0 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing personalized PCI analysis outputs.",
    )
    parser.add_argument(
        "--analysis-prefix",
        required=True,
        help="Prefix used by analyze_serotonergic_pci_personalized.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <analysis-dir>/nature_figures/trajectories.",
    )
    parser.add_argument(
        "--figure-prefix",
        default="serotonergic_pci_responder_trajectories",
    )
    return parser.parse_args()


def load_tables(
    directory: Path,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data_path = directory / f"{prefix}_analysis_input.csv"
    subject_path = directory / f"{prefix}_subject_responder_table.csv"
    manifest_path = directory / f"{prefix}_analysis_manifest.json"
    for path in [data_path, subject_path, manifest_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required analysis output not found: {path}")

    data = pd.read_csv(data_path)
    subjects = pd.read_csv(subject_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_columns = {"condition", "subject_id", "occupancy", "pci_mean"}
    subject_columns = {
        "condition",
        "subject_id",
        "max_dose_delta",
        "control_referenced_response",
    }
    missing_data = data_columns.difference(data.columns)
    missing_subjects = subject_columns.difference(subjects.columns)
    if missing_data:
        raise ValueError(f"Analysis input is missing columns: {sorted(missing_data)}")
    if missing_subjects:
        raise ValueError(
            f"Subject table is missing columns: {sorted(missing_subjects)}"
        )
    if data.duplicated(["condition", "subject_id", "occupancy"]).any():
        raise ValueError("Duplicate subject-dose observations were found.")

    selected_subjects = subjects.loc[
        subjects["condition"].isin(DISPLAY_CONDITIONS)
    ].copy()
    selected_data = data.loc[data["condition"].isin(DISPLAY_CONDITIONS)].copy()
    selected_data = selected_data.merge(
        selected_subjects[
            [
                "condition",
                "subject_id",
                "max_dose_delta",
                "control_referenced_response",
            ]
        ],
        on=["condition", "subject_id"],
        how="left",
        validate="many_to_one",
    )
    if selected_data["control_referenced_response"].isna().any():
        raise ValueError("Not every trajectory matched a responder classification.")
    selected_data["control_referenced_response"] = selected_data[
        "control_referenced_response"
    ].astype(bool)
    baseline_rows = selected_data.loc[
        np.isclose(selected_data["occupancy"], 0.0),
        ["condition", "subject_id", "pci_mean"],
    ].rename(columns={"pci_mean": "baseline_pci"})
    if baseline_rows.duplicated(["condition", "subject_id"]).any():
        raise ValueError("Multiple baseline PCI observations were found for a subject.")
    selected_data = selected_data.merge(
        baseline_rows,
        on=["condition", "subject_id"],
        how="left",
        validate="many_to_one",
    )
    if selected_data["baseline_pci"].isna().any():
        raise ValueError("Not every subject has a zero-occupancy baseline PCI value.")
    selected_data["pci_delta"] = (
        selected_data["pci_mean"] - selected_data["baseline_pci"]
    )
    return selected_data, selected_subjects, manifest


def _set_aesthetic() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Arial",
        rc={
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#242424",
            "text.color": "#242424",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.45,
            "axes.linewidth": 0.6,
        },
    )
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.8,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
        }
    )


def _style_axis(axis: mpl.axes.Axes) -> None:
    axis.grid(axis="x", visible=False)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.92)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.set_axisbelow(True)


def validate_response_classification(
    data: pd.DataFrame,
    subjects: pd.DataFrame,
    response_threshold: float,
) -> None:
    maximum_dose = float(data["occupancy"].max())
    endpoints = data.loc[
        np.isclose(data["occupancy"], maximum_dose),
        ["condition", "subject_id", "pci_delta"],
    ].copy()
    endpoints["computed_response"] = endpoints["pci_delta"].gt(response_threshold)
    comparison = endpoints.merge(
        subjects[
            [
                "condition",
                "subject_id",
                "control_referenced_response",
            ]
        ],
        on=["condition", "subject_id"],
        how="left",
        validate="one_to_one",
    )
    stored_response = comparison["control_referenced_response"].astype(bool)
    if not comparison["computed_response"].eq(stored_response).all():
        mismatches = comparison.loc[
            ~comparison["computed_response"].eq(stored_response),
            ["condition", "subject_id", "pci_delta"],
        ]
        raise ValueError(
            "Stored responder classifications do not match the plotted "
            f"maximum-dose threshold rule:\n{mismatches.to_string(index=False)}"
        )


def draw_trajectory_panel(
    axis: mpl.axes.Axes,
    data: pd.DataFrame,
    subjects: pd.DataFrame,
    *,
    condition: str,
    response_threshold: float,
    panel_label: str | None,
    show_x_label: bool,
    show_y_label: bool,
    show_x_tick_labels: bool,
    show_threshold_label: bool,
) -> pd.DataFrame:
    group = data.loc[data["condition"].eq(condition)].copy()
    condition_subjects = subjects.loc[subjects["condition"].eq(condition)].copy()
    condition_color = CONDITION_COLORS[condition]

    nonresponders = group.loc[~group["control_referenced_response"]]
    responders = group.loc[group["control_referenced_response"]]
    for _, trajectory in nonresponders.groupby("subject_id"):
        trajectory = trajectory.sort_values("occupancy")
        axis.plot(
            trajectory["occupancy"],
            trajectory["pci_delta"],
            color=condition_color,
            linewidth=0.45,
            alpha=0.13,
            zorder=1,
        )
    for _, trajectory in responders.groupby("subject_id"):
        trajectory = trajectory.sort_values("occupancy")
        axis.plot(
            trajectory["occupancy"],
            trajectory["pci_delta"],
            color=condition_color,
            linewidth=0.80,
            alpha=0.78,
            zorder=3,
        )

    axis.axhline(
        0.0,
        color="#777777",
        linewidth=0.55,
        alpha=0.70,
        zorder=0,
    )
    axis.axhline(
        response_threshold,
        color="#777777",
        linewidth=0.75,
        linestyle=(0, (3.0, 2.2)),
        zorder=2,
    )
    if show_threshold_label:
        axis.text(
            0.018,
            response_threshold + 0.012,
            f"Control cutoff = {response_threshold:.3f}",
            ha="left",
            va="bottom",
            fontsize=5.0,
            color="#595959",
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    sns.lineplot(
        data=group,
        x="occupancy",
        y="pci_delta",
        estimator="mean",
        errorbar="se",
        sort=True,
        color=MEAN_COLOR,
        linewidth=1.25,
        marker="o",
        markersize=3.0,
        markeredgecolor="white",
        markeredgewidth=0.45,
        err_style="band",
        err_kws={"alpha": 0.10, "linewidth": 0},
        legend=False,
        ax=axis,
        zorder=5,
    )

    maximum_dose = float(group["occupancy"].max())
    responder_endpoints = responders.loc[
        np.isclose(responders["occupancy"], maximum_dose)
    ]
    axis.scatter(
        responder_endpoints["occupancy"],
        responder_endpoints["pci_delta"],
        s=11,
        facecolor=condition_color,
        edgecolor="white",
        linewidth=0.45,
        zorder=6,
    )

    n_responders = int(
        condition_subjects["control_referenced_response"].astype(bool).sum()
    )
    n_subjects = len(condition_subjects)
    axis.set_title(condition, loc="left", fontweight="normal", color="#222222", pad=4.0)
    axis.text(
        1.0,
        1.025,
        f"{n_responders}/{n_subjects} above cutoff",
        transform=axis.transAxes,
        fontsize=5.5,
        color="#222222",
        ha="right",
        va="bottom",
    )
    if panel_label:
        axis.text(
            -0.075,
            1.03,
            panel_label,
            transform=axis.transAxes,
            fontsize=8.0,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    occupancies = np.sort(group["occupancy"].unique())
    axis.set_xlim(-0.025, 0.855)
    axis.set_ylim(-0.40, 0.47)
    axis.set_xticks(occupancies)
    axis.set_xticklabels(["0", "0.25", "0.50", "0.766"])
    axis.set_xlabel(r"5-HT$_{2A}$ occupancy" if show_x_label else "")
    axis.set_ylabel("ΔPCI from baseline" if show_y_label else "")
    axis.tick_params(labelbottom=show_x_tick_labels)
    _style_axis(axis)
    return condition_subjects.loc[
        condition_subjects["control_referenced_response"].astype(bool)
    ].copy()


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=LEGEND_STYLE_COLOR,
            linewidth=0.7,
            alpha=0.25,
            label="Below cutoff",
        ),
        Line2D(
            [0],
            [0],
            color=LEGEND_STYLE_COLOR,
            linewidth=1.1,
            marker="o",
            markersize=2.6,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label="Above cutoff at 0.766",
        ),
        Line2D(
            [0],
            [0],
            color=MEAN_COLOR,
            linewidth=1.3,
            marker="o",
            markersize=2.8,
            label="Mean ± s.e.m.",
        ),
        Line2D(
            [0],
            [0],
            color="#777777",
            linewidth=0.8,
            linestyle=(0, (3.0, 2.2)),
            label="Control cutoff",
        ),
    ]


def _save_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    *,
    png_dpi: int = 600,
    tiff_dpi: int = 300,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg"]:
        figure.savefig(output_dir / f"{stem}.{extension}")
    png_path = output_dir / f"{stem}.png"
    figure.savefig(png_path, dpi=png_dpi)
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(png_dpi, png_dpi))
    tiff_path = output_dir / f"{stem}.tiff"
    figure.savefig(
        tiff_path,
        dpi=tiff_dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(tiff_path) as image:
        image.convert("RGB").save(
            tiff_path,
            dpi=(tiff_dpi, tiff_dpi),
            compression="tiff_lzw",
        )


def plot_combined(
    data: pd.DataFrame,
    subjects: pd.DataFrame,
    *,
    response_threshold: float,
    output_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(NATURE_DOUBLE_COLUMN_IN, 5.75),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    responder_tables: list[pd.DataFrame] = []
    for index, (axis, condition) in enumerate(
        zip(axes, DISPLAY_CONDITIONS, strict=True)
    ):
        responders = draw_trajectory_panel(
            axis,
            data,
            subjects,
            condition=condition,
            response_threshold=response_threshold,
            panel_label=chr(ord("a") + index),
            show_x_label=index == len(axes) - 1,
            show_y_label=index == 1,
            show_x_tick_labels=index == len(axes) - 1,
            show_threshold_label=False,
        )
        responder_tables.append(responders)
    figure.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.61, 1.006),
        ncol=4,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.9,
        handletextpad=0.45,
    )
    _save_figure(
        figure,
        output_dir,
        f"{prefix}_combined",
    )
    plt.close(figure)
    return pd.concat(responder_tables, ignore_index=True)


def plot_individual_conditions(
    data: pd.DataFrame,
    subjects: pd.DataFrame,
    *,
    response_threshold: float,
    output_dir: Path,
    prefix: str,
) -> None:
    for condition in DISPLAY_CONDITIONS:
        figure, axis = plt.subplots(
            figsize=(NATURE_SINGLE_COLUMN_IN, 2.55),
            layout="constrained",
        )
        draw_trajectory_panel(
            axis,
            data,
            subjects,
            condition=condition,
            response_threshold=response_threshold,
            panel_label=None,
            show_x_label=True,
            show_y_label=True,
            show_x_tick_labels=True,
            show_threshold_label=True,
        )
        _save_figure(
            figure,
            output_dir,
            f"{prefix}_{condition.lower()}",
        )
        plt.close(figure)


def write_legend(
    path: Path,
    subjects: pd.DataFrame,
    manifest: dict[str, object],
    *,
    prefix: str,
) -> None:
    counts = []
    for condition in DISPLAY_CONDITIONS:
        group = subjects.loc[subjects["condition"].eq(condition)]
        n_responders = int(group["control_referenced_response"].astype(bool).sum())
        counts.append(f"{condition}: {n_responders}/{len(group)}")
    threshold = float(manifest["control_reference_delta_threshold"])
    text = f"""# Proposed figure legend

## {prefix}_combined

**Individual serotonergic changes in PCI in COMA, UWS and MCS.**
Trajectories show each subject's PCI change from their own zero-occupancy baseline
and use the established diagnosis colours (COMA, navy; UWS, mauve; MCS, burnt
sienna). The dashed line marks the 95th percentile of the CNT maximum-dose change
(delta PCI = {threshold:.6f}). Saturated trajectories and filled maximum-dose
endpoints identify every subject whose delta PCI at 0.766 exceeded this cutoff;
other subjects are shown as thin translucent lines. The black line and light band
show the diagnosis mean and s.e.m., respectively. {", ".join(counts)}. PCI was
computed once per subject and occupancy from the time-locked average of 100
stimulation trials. Responder status is descriptive and does not constitute an
individual statistical or clinical treatment-response classification.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir or args.analysis_dir / "nature_figures" / "trajectories"
    )
    data, subjects, manifest = load_tables(
        args.analysis_dir,
        args.analysis_prefix,
    )
    _set_aesthetic()
    response_threshold = float(manifest["control_reference_delta_threshold"])
    validate_response_classification(
        data,
        subjects,
        response_threshold,
    )
    highlighted_responders = plot_combined(
        data,
        subjects,
        response_threshold=response_threshold,
        output_dir=output_dir,
        prefix=args.figure_prefix,
    )
    plot_individual_conditions(
        data,
        subjects,
        response_threshold=response_threshold,
        output_dir=output_dir,
        prefix=args.figure_prefix,
    )
    highlighted_responders.to_csv(
        output_dir / f"{args.figure_prefix}_highlighted_responders.csv",
        index=False,
    )
    legacy_labelled_path = output_dir / f"{args.figure_prefix}_labelled_subjects.csv"
    if legacy_labelled_path.exists():
        legacy_labelled_path.unlink()
    write_legend(
        output_dir / f"{args.figure_prefix}_legend.md",
        subjects,
        manifest,
        prefix=args.figure_prefix,
    )
    format_manifest = {
        "combined_width_mm": 183,
        "combined_height_mm": 5.75 * 25.4,
        "individual_width_mm": 89,
        "individual_height_mm": 2.55 * 25.4,
        "font": "Arial",
        "font_size_pt_range": [5.0, 8.0],
        "plotting_libraries": ["matplotlib", "seaborn"],
        "vector_exports": ["pdf", "svg"],
        "raster_exports": {"png_dpi": 600, "tiff_dpi": 300},
        "colour_space": "RGB",
        "diagnosis_colours": CONDITION_COLORS,
        "trajectory_metric": "PCI minus each subject's zero-occupancy PCI",
        "responder_rule": (
            "maximum-dose delta PCI exceeds the CNT 95th-percentile "
            f"threshold ({response_threshold:.12g})"
        ),
        "analysis_directory": str(args.analysis_dir.resolve()),
        "analysis_prefix": args.analysis_prefix,
    }
    (output_dir / f"{args.figure_prefix}_format_manifest.json").write_text(
        json.dumps(format_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote polished responder-trajectory figures to {output_dir}")


if __name__ == "__main__":
    main()
