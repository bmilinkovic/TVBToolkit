#!/usr/bin/env python3
"""Create Nature-format statistical figures for serotonergic PCI responses."""

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
from scipy import stats

CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
RESPONDER_ORDER = ["UWS", "MCS", "COMA", "EMCS", "CNT"]
CONDITION_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
    "EMCS": "#E8B56D",
    "CNT": "#5B8A72",
}
CONDITION_LINESTYLES = {
    "COMA": "-",
    "UWS": (0, (4, 1.5)),
    "MCS": (0, (1.5, 1.2)),
    "EMCS": (0, (5, 1.5, 1.5, 1.5)),
    "CNT": "-.",
}
NATURE_DOUBLE_COLUMN_IN = 183.0 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing the personalized-analysis CSV outputs.",
    )
    parser.add_argument(
        "--analysis-prefix",
        required=True,
        help="Filename prefix used by analyze_serotonergic_pci_personalized.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <analysis-dir>/nature_figures.",
    )
    parser.add_argument(
        "--figure-prefix",
        default="serotonergic_pci_statistical_response",
    )
    parser.add_argument("--jitter-seed", type=int, default=20260730)
    return parser.parse_args()


def _read_table(directory: Path, prefix: str, suffix: str) -> pd.DataFrame:
    path = directory / f"{prefix}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required analysis table not found: {path}")
    return pd.read_csv(path)


def load_analysis_tables(
    directory: Path,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    groups = _read_table(directory, prefix, "group_summary")
    pairwise = _read_table(directory, prefix, "pairwise_slope_tests")
    subjects = _read_table(directory, prefix, "subject_responder_table")
    manifest_path = directory / f"{prefix}_analysis_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Required analysis manifest not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    group_columns = {
        "condition",
        "n_subjects",
        "control_referenced_response_n",
        "control_referenced_response_proportion",
        "control_referenced_response_ci95_low",
        "control_referenced_response_ci95_high",
    }
    pairwise_columns = {
        "condition_a",
        "condition_b",
        "mean_slope_difference_a_minus_b",
        "difference_ci95_low",
        "difference_ci95_high",
        "p_holm_all_pairs",
        "significant_holm_0_05",
    }
    subject_columns = {
        "condition",
        "subject_id",
        "linear_slope_per_occupancy",
        "max_dose_delta",
    }
    for table, required, name in [
        (groups, group_columns, "group summary"),
        (pairwise, pairwise_columns, "pairwise tests"),
        (subjects, subject_columns, "subject responder table"),
    ]:
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    expected = set(CONDITION_ORDER)
    for table, name in [(groups, "group summary"), (subjects, "subject table")]:
        observed = set(table["condition"])
        if observed != expected:
            raise ValueError(
                f"{name} diagnoses differ from expected: {sorted(observed)}"
            )
    if len(pairwise) != 10:
        raise ValueError("Expected all ten pairwise diagnosis comparisons.")
    if groups["n_subjects"].sum() != len(subjects):
        raise ValueError("Group and subject-table sample sizes do not agree.")
    return groups, pairwise, subjects, manifest


def _nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 0.8,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _style_axis(axis: mpl.axes.Axes, *, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    if grid_axis:
        axis.grid(
            axis=grid_axis,
            color="#D7D7D7",
            linewidth=0.4,
            alpha=0.72,
            zorder=0,
        )


def _panel_label(axis: mpl.axes.Axes, label: str, title: str) -> None:
    axis.text(
        -0.12,
        1.045,
        label,
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    axis.set_title(title, loc="left", pad=4.0, fontweight="normal")


def _format_adjusted_p(p_value: float) -> str:
    if p_value < 0.001:
        exponent = int(np.floor(np.log10(p_value)))
        coefficient = p_value / (10**exponent)
        return f"{coefficient:.2f}e{exponent}"
    if p_value < 0.01:
        return f"{p_value:.4f}"
    return f"{p_value:.3f}"


def _save_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:
        figure.savefig(output_dir / f"{stem}.{extension}")
    png_path = output_dir / f"{stem}.png"
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
    tiff_path = output_dir / f"{stem}.tiff"
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


def plot_main_statistical_figure(
    groups: pd.DataFrame,
    pairwise: pd.DataFrame,
    manifest: dict[str, object],
    *,
    output_dir: Path,
    prefix: str,
) -> None:
    figure = plt.figure(
        figsize=(NATURE_DOUBLE_COLUMN_IN, 4.0),
        layout="constrained",
    )
    layout = figure.add_gridspec(1, 2, width_ratios=[0.82, 1.45], wspace=0.08)
    responder_axis = figure.add_subplot(layout[0, 0])
    pairwise_axis = figure.add_subplot(layout[0, 1])

    group_indexed = groups.set_index("condition").loc[RESPONDER_ORDER]
    y_positions = np.arange(len(RESPONDER_ORDER))
    proportions = group_indexed["control_referenced_response_proportion"].to_numpy(
        float
    )
    ci_low = group_indexed["control_referenced_response_ci95_low"].to_numpy(float)
    ci_high = group_indexed["control_referenced_response_ci95_high"].to_numpy(float)
    colors = [CONDITION_COLORS[condition] for condition in RESPONDER_ORDER]

    for index, condition in enumerate(RESPONDER_ORDER):
        responder_axis.errorbar(
            proportions[index],
            y_positions[index],
            xerr=[
                [proportions[index] - ci_low[index]],
                [ci_high[index] - proportions[index]],
            ],
            fmt="o",
            markersize=4.0,
            markerfacecolor=colors[index],
            markeredgecolor="#222222",
            markeredgewidth=0.45,
            ecolor=colors[index],
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.7,
            zorder=3,
        )
        count = int(group_indexed.loc[condition, "control_referenced_response_n"])
        total = int(group_indexed.loc[condition, "n_subjects"])
        responder_axis.text(
            min(ci_high[index] + 0.025, 0.70),
            y_positions[index],
            f"{count}/{total} ({proportions[index]:.0%})",
            ha="left",
            va="center",
            fontsize=5.7,
            color="#222222",
        )

    control_fraction = float(
        group_indexed.loc["CNT", "control_referenced_response_proportion"]
    )
    responder_axis.axvline(
        control_fraction,
        color="#666666",
        linewidth=0.65,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    responder_axis.set_yticks(y_positions, RESPONDER_ORDER)
    responder_axis.invert_yaxis()
    responder_axis.set_xlim(0, 0.75)
    responder_axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    responder_axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    responder_axis.set_xlabel("Fraction above control cutoff")
    _panel_label(
        responder_axis,
        "a",
        "High-dose responder fraction",
    )
    _style_axis(responder_axis, grid_axis="x")

    y_pairwise = np.arange(len(pairwise))
    significant = pairwise["significant_holm_0_05"].astype(bool).to_numpy()
    effect_colors = np.where(significant, "#146B73", "#8C8C8C")
    pair_labels = [
        f"{row.condition_a} \N{MINUS SIGN} {row.condition_b}"
        for row in pairwise.itertuples(index=False)
    ]
    differences = pairwise["mean_slope_difference_a_minus_b"].to_numpy(float)
    pair_ci_low = pairwise["difference_ci95_low"].to_numpy(float)
    pair_ci_high = pairwise["difference_ci95_high"].to_numpy(float)

    for index, row in pairwise.iterrows():
        is_significant = bool(row["significant_holm_0_05"])
        pairwise_axis.errorbar(
            differences[index],
            y_pairwise[index],
            xerr=[
                [differences[index] - pair_ci_low[index]],
                [pair_ci_high[index] - differences[index]],
            ],
            fmt="o",
            markersize=3.8,
            markerfacecolor=(effect_colors[index] if is_significant else "white"),
            markeredgecolor=effect_colors[index],
            markeredgewidth=0.7,
            ecolor=effect_colors[index],
            elinewidth=0.85,
            capsize=1.9,
            capthick=0.65,
            zorder=3,
        )
        pairwise_axis.text(
            0.307,
            y_pairwise[index],
            ("* " if is_significant else "")
            + _format_adjusted_p(float(row["p_holm_all_pairs"])),
            ha="left",
            va="center",
            fontsize=5.2,
            fontweight="bold" if is_significant else "normal",
            color="#111111" if is_significant else "#666666",
        )

    pairwise_axis.axvline(0, color="#222222", linewidth=0.65, zorder=1)
    pairwise_axis.axvline(
        0.292,
        color="#BDBDBD",
        linewidth=0.45,
        zorder=1,
    )
    pairwise_axis.text(
        0.307,
        -0.74,
        "Holm-adjusted P",
        ha="left",
        va="bottom",
        fontsize=5.3,
        color="#444444",
    )
    pairwise_axis.set_yticks(y_pairwise, pair_labels)
    pairwise_axis.invert_yaxis()
    pairwise_axis.set_xlim(-0.31, 0.43)
    pairwise_axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    pairwise_axis.set_xlabel("Mean slope difference (first \N{MINUS SIGN} second)")
    _panel_label(
        pairwise_axis,
        "b",
        "Pairwise slope differences",
    )
    _style_axis(pairwise_axis, grid_axis="x")

    _save_figure(
        figure,
        output_dir,
        f"{prefix}_main",
    )
    plt.close(figure)


def plot_heterogeneity_figure(
    subjects: pd.DataFrame,
    manifest: dict[str, object],
    *,
    output_dir: Path,
    prefix: str,
    jitter_seed: int,
) -> None:
    rng = np.random.default_rng(jitter_seed)
    figure = plt.figure(
        figsize=(NATURE_DOUBLE_COLUMN_IN, 3.05),
        layout="constrained",
    )
    layout = figure.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.08)
    slope_axis = figure.add_subplot(layout[0, 0])
    survival_axis = figure.add_subplot(layout[0, 1])

    positions = np.arange(len(CONDITION_ORDER))
    distributions = [
        subjects.loc[
            subjects["condition"].eq(condition),
            "linear_slope_per_occupancy",
        ].to_numpy(float)
        for condition in CONDITION_ORDER
    ]
    violins = slope_axis.violinplot(
        distributions,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, condition in zip(violins["bodies"], CONDITION_ORDER, strict=True):
        body.set_facecolor(CONDITION_COLORS[condition])
        body.set_edgecolor("none")
        body.set_alpha(0.16)

    for position, condition, values in zip(
        positions,
        CONDITION_ORDER,
        distributions,
        strict=True,
    ):
        jitter = rng.normal(0, 0.055, len(values))
        slope_axis.scatter(
            position + jitter,
            values,
            s=4.5,
            color=CONDITION_COLORS[condition],
            alpha=0.38,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        ci_half_width = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
        slope_axis.errorbar(
            position,
            mean,
            yerr=ci_half_width,
            fmt="o",
            markersize=3.8,
            markerfacecolor=CONDITION_COLORS[condition],
            markeredgecolor="#111111",
            markeredgewidth=0.55,
            color="#111111",
            elinewidth=0.85,
            capsize=2,
            zorder=4,
        )

    slope_axis.axhline(0, color="#222222", linewidth=0.65)
    slope_axis.set_xticks(positions, CONDITION_ORDER)
    slope_axis.set_ylabel("Linear PCI slope")
    _panel_label(slope_axis, "a", "Linear slope distributions")
    _style_axis(slope_axis, grid_axis="y")

    threshold = float(manifest["control_reference_delta_threshold"])
    for condition in CONDITION_ORDER:
        values = np.sort(
            subjects.loc[
                subjects["condition"].eq(condition),
                "max_dose_delta",
            ].to_numpy(float)
        )
        x_values = np.r_[values[0] - 0.02, values, values[-1] + 0.02]
        survival = np.r_[1.0, 1.0 - np.arange(1, len(values) + 1) / len(values), 0.0]
        survival_axis.step(
            x_values,
            survival,
            where="post",
            color=CONDITION_COLORS[condition],
            linestyle=CONDITION_LINESTYLES[condition],
            linewidth=1.0,
            label=condition,
        )
    survival_axis.axvline(
        threshold,
        color="#333333",
        linewidth=0.65,
        linestyle=(0, (3, 2)),
    )
    survival_axis.text(
        threshold + 0.008,
        0.97,
        "Control cutoff",
        fontsize=5.3,
        rotation=90,
        va="top",
        ha="left",
        color="#444444",
    )
    for condition, y_offset in [("UWS", 0.025), ("MCS", -0.035)]:
        values = subjects.loc[
            subjects["condition"].eq(condition),
            "max_dose_delta",
        ]
        proportion = float((values > threshold).mean())
        survival_axis.scatter(
            [threshold],
            [proportion],
            s=11,
            color="#222222",
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )
        survival_axis.text(
            threshold + 0.025,
            proportion + y_offset,
            f"{condition}: {proportion:.0%}",
            fontsize=5.4,
            color="#222222",
            va="center",
        )
    survival_axis.set_xlim(-0.39, 0.47)
    survival_axis.set_ylim(0, 1)
    survival_axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    survival_axis.set_xlabel("\N{GREEK CAPITAL LETTER DELTA}PCI at 0.766 occupancy")
    survival_axis.set_ylabel(
        "Fraction with \N{GREEK CAPITAL LETTER DELTA}PCI \N{GREATER-THAN OR EQUAL TO} x"
    )
    _panel_label(
        survival_axis,
        "b",
        "Maximum-dose \N{GREEK CAPITAL LETTER DELTA}PCI distributions",
    )
    _style_axis(survival_axis, grid_axis="y")
    survival_axis.legend(
        loc="upper right",
        frameon=False,
        ncol=1,
        handlelength=2.0,
        handletextpad=0.5,
    )

    _save_figure(
        figure,
        output_dir,
        f"{prefix}_extended_data",
    )
    plt.close(figure)


def write_figure_legends(
    path: Path,
    groups: pd.DataFrame,
    manifest: dict[str, object],
    *,
    prefix: str,
) -> None:
    sample_sizes = ", ".join(
        f"{condition} n={int(groups.set_index('condition').loc[condition, 'n_subjects'])}"
        for condition in CONDITION_ORDER
    )
    threshold = float(manifest["control_reference_delta_threshold"])
    text = f"""# Proposed figure legends

## {prefix}_main

**Control-referenced serotonergic PCI response and pairwise diagnosis contrasts.**
**a,** Proportion of subjects whose maximum-dose PCI change exceeded the 95th
percentile of the CNT change distribution (delta PCI > {threshold:.6f}). Points
show observed proportions, horizontal lines show Wilson 95% confidence intervals,
and labels give responder count/total count. The dashed line denotes the observed
CNT reference fraction. **b,** Pairwise differences in mean subject-level linear
PCI dose-response slopes. Points show the first diagnosis minus the second and
horizontal lines show two-sided Welch 95% confidence intervals. Filled teal points
denote comparisons with Holm-adjusted P < 0.05; exact adjusted P values are shown
for all ten comparisons. {sample_sizes}. Each subject slope was fitted across
occupancies 0, 0.25, 0.50 and 0.766.

## {prefix}_extended_data

**Heterogeneity of individual serotonergic PCI responses.** **a,** Distribution
of subject-level linear PCI dose-response slopes. Small points represent subjects,
violin envelopes show the empirical density, and large points and error bars show
the mean and two-sided t-based 95% confidence interval. **b,** Empirical survival
distributions of maximum-dose PCI change relative to baseline. The dashed vertical
line marks the CNT 95th-percentile threshold; labels identify the fractions of UWS
and MCS subjects exceeding it. {sample_sizes}. PCI was computed once per subject
and occupancy from the time-locked average of 100 stimulation trials.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.analysis_dir / "nature_figures"
    groups, pairwise, subjects, manifest = load_analysis_tables(
        args.analysis_dir,
        args.analysis_prefix,
    )
    _nature_style()
    plot_main_statistical_figure(
        groups,
        pairwise,
        manifest,
        output_dir=output_dir,
        prefix=args.figure_prefix,
    )
    plot_heterogeneity_figure(
        subjects,
        manifest,
        output_dir=output_dir,
        prefix=args.figure_prefix,
        jitter_seed=args.jitter_seed,
    )
    write_figure_legends(
        output_dir / f"{args.figure_prefix}_legends.md",
        groups,
        manifest,
        prefix=args.figure_prefix,
    )
    format_manifest = {
        "figure_width_mm": 183,
        "main_figure_height_mm": 4.0 * 25.4,
        "extended_data_height_mm": 3.05 * 25.4,
        "font": "Arial",
        "font_size_pt_range": [5.2, 8.0],
        "line_width_pt_range": [0.4, 1.0],
        "vector_exports": ["pdf", "svg"],
        "raster_exports": {"png_dpi": 600, "tiff_dpi": 300},
        "colour_space": "RGB",
        "analysis_directory": str(args.analysis_dir.resolve()),
        "analysis_prefix": args.analysis_prefix,
    }
    (output_dir / f"{args.figure_prefix}_format_manifest.json").write_text(
        json.dumps(format_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Nature-format statistical figures to {output_dir}")


if __name__ == "__main__":
    main()
