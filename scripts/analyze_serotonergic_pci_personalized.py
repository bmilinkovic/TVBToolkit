#!/usr/bin/env python3
"""Analyse group and subject-level serotonergic PCI dose responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
CONDITION_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
    "EMCS": "#E8B56D",
    "CNT": "#5B8A72",
}
REQUIRED_COLUMNS = {
    "condition",
    "subject_id",
    "occupancy",
    "pci_mean",
    "n_trials",
    "pci_estimator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Corrected time-locked subject PCI metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serotonergic_pci_personalized_analysis"),
    )
    parser.add_argument(
        "--prefix",
        default="serotonergic_pci_personalized",
    )
    parser.add_argument(
        "--large-change",
        type=float,
        default=0.05,
        help="Descriptive absolute PCI-change threshold; not a significance cutoff.",
    )
    parser.add_argument(
        "--control-quantile",
        type=float,
        default=0.95,
        help="Upper control max-dose-change quantile for control-referenced response.",
    )
    parser.add_argument("--jitter-seed", type=int, default=20260730)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _holm_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted
    ordered = finite_indices[np.argsort(values[finite_indices])]
    n_tests = ordered.size
    running_max = 0.0
    for rank, index in enumerate(ordered):
        candidate = (n_tests - rank) * values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    z_value = stats.norm.ppf(0.975)
    denominator = 1.0 + z_value**2 / total
    centre = (proportion + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / total + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(centre - half_width), float(centre + half_width)


def _mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        value = float(array[0]) if array.size else np.nan
        return value, value
    mean = float(np.mean(array))
    half_width = float(
        stats.t.ppf(0.975, array.size - 1) * np.std(array, ddof=1) / np.sqrt(array.size)
    )
    return mean - half_width, mean + half_width


def _welch_anova(groups: list[np.ndarray]) -> tuple[float, float, float, float]:
    arrays = [np.asarray(group, dtype=float) for group in groups]
    arrays = [group[np.isfinite(group)] for group in arrays]
    if any(group.size < 2 for group in arrays):
        return np.nan, np.nan, np.nan, np.nan

    n_obs = np.asarray([group.size for group in arrays], dtype=float)
    means = np.asarray([np.mean(group) for group in arrays], dtype=float)
    variances = np.asarray([np.var(group, ddof=1) for group in arrays], dtype=float)
    if np.any(variances <= 0):
        return np.nan, np.nan, np.nan, np.nan

    weights = n_obs / variances
    weight_sum = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / weight_sum)
    n_groups = len(arrays)
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / (n_groups - 1))
    correction_term = float(np.sum((1.0 - weights / weight_sum) ** 2 / (n_obs - 1.0)))
    denominator = 1.0 + (2.0 * (n_groups - 2.0) / (n_groups**2 - 1.0)) * correction_term
    statistic = numerator / denominator
    df_num = float(n_groups - 1)
    df_denom = float((n_groups**2 - 1.0) / (3.0 * correction_term))
    p_value = float(stats.f.sf(statistic, df_num, df_denom))
    return statistic, p_value, df_num, df_denom


def load_data(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = REQUIRED_COLUMNS.difference(header.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    columns = [
        column
        for column in [
            "cohort",
            "condition",
            "subject_id",
            "scenario",
            "occupancy",
            "n_trials",
            "pci_estimator",
            "pci_mean",
        ]
        if column in header.columns
    ]
    data = pd.read_csv(path, usecols=columns)
    data["condition"] = data["condition"].astype(str).str.upper()
    data["subject_id"] = data["subject_id"].astype(str)
    data["occupancy"] = pd.to_numeric(data["occupancy"], errors="raise")
    data["pci_mean"] = pd.to_numeric(data["pci_mean"], errors="raise")

    unknown = sorted(set(data["condition"]).difference(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"Unknown diagnoses: {unknown}")
    if not np.isfinite(data[["occupancy", "pci_mean"]].to_numpy()).all():
        raise ValueError("occupancy and pci_mean must be finite.")
    if data.duplicated(["condition", "subject_id", "occupancy"]).any():
        raise ValueError("Duplicate condition/subject/dose records were found.")
    if data["n_trials"].nunique() != 1:
        raise ValueError("All rows must use the same number of trials.")
    if data["pci_estimator"].nunique() != 1:
        raise ValueError("All rows must use the same PCI estimator.")

    data["subject_uid"] = data["condition"] + ":" + data["subject_id"]
    dose_sets = data.groupby("subject_uid")["occupancy"].apply(
        lambda values: tuple(sorted(values))
    )
    if dose_sets.nunique() != 1:
        raise ValueError("Subjects do not share one complete occupancy grid.")
    occupancies = np.asarray(dose_sets.iloc[0], dtype=float)
    if occupancies.size < 3 or not np.isclose(occupancies[0], 0.0):
        raise ValueError(
            "The analysis requires baseline plus at least two positive doses."
        )

    data["condition"] = pd.Categorical(
        data["condition"],
        CONDITION_ORDER,
        ordered=True,
    )
    return data.sort_values(["condition", "subject_id", "occupancy"]).reset_index(
        drop=True
    )


def build_subject_metrics(
    data: pd.DataFrame,
    *,
    large_change: float,
    control_quantile: float,
) -> tuple[pd.DataFrame, float]:
    occupancies = np.sort(data["occupancy"].unique())
    maximum_dose = float(occupancies[-1])
    rows: list[dict[str, object]] = []
    for (condition, subject_id), group in data.groupby(
        ["condition", "subject_id"],
        observed=True,
        sort=False,
    ):
        group = group.sort_values("occupancy")
        dose = group["occupancy"].to_numpy(float)
        pci = group["pci_mean"].to_numpy(float)
        fit = stats.linregress(dose, pci)
        differences = np.diff(pci)
        rows.append(
            {
                "condition": str(condition),
                "subject_id": str(subject_id),
                "subject_uid": f"{condition}:{subject_id}",
                "baseline_pci": float(pci[0]),
                "max_dose_occupancy": maximum_dose,
                "max_dose_pci": float(pci[-1]),
                "max_dose_delta": float(pci[-1] - pci[0]),
                "relative_max_dose_change": float((pci[-1] - pci[0]) / pci[0]),
                "linear_slope_per_occupancy": float(fit.slope),
                "linear_r_squared": float(fit.rvalue**2),
                "positive_dose_steps": int(np.sum(differences > 0)),
                "negative_dose_steps": int(np.sum(differences < 0)),
                "strictly_monotonic_increase": bool(np.all(differences > 0)),
                "strictly_monotonic_decrease": bool(np.all(differences < 0)),
                "peak_occupancy": float(dose[np.argmax(pci)]),
                "trough_occupancy": float(dose[np.argmin(pci)]),
                "mean_positive_dose_change": float(np.mean(pci[1:] - pci[0])),
            }
        )

    subjects = pd.DataFrame(rows)
    control_deltas = subjects.loc[
        subjects["condition"].eq("CNT"),
        "max_dose_delta",
    ]
    if control_deltas.empty:
        raise ValueError(
            "CNT subjects are required for the control-referenced threshold."
        )
    control_threshold = float(control_deltas.quantile(control_quantile))

    subjects["positive_max_dose_response"] = subjects["max_dose_delta"] > 0
    subjects["large_absolute_response"] = subjects["max_dose_delta"] >= large_change
    subjects["control_referenced_response"] = (
        subjects["max_dose_delta"] > control_threshold
    )
    subjects["high_dose_candidate"] = (
        subjects["control_referenced_response"]
        & subjects["linear_slope_per_occupancy"].gt(0)
        & np.isclose(subjects["peak_occupancy"], maximum_dose)
    )
    subjects["consistent_high_dose_candidate"] = subjects[
        "high_dose_candidate"
    ] & subjects["positive_dose_steps"].ge(len(occupancies) - 2)
    subjects["individual_inference_status"] = (
        "descriptive_only_no_subject_dose_uncertainty"
    )
    subjects["rank_within_condition_by_max_dose_delta"] = (
        subjects.groupby("condition")["max_dose_delta"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    subjects["condition"] = pd.Categorical(
        subjects["condition"],
        CONDITION_ORDER,
        ordered=True,
    )
    subjects = subjects.sort_values(
        ["condition", "rank_within_condition_by_max_dose_delta"]
    ).reset_index(drop=True)
    return subjects, control_threshold


def build_group_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    response_flags = [
        "positive_max_dose_response",
        "large_absolute_response",
        "control_referenced_response",
        "high_dose_candidate",
        "consistent_high_dose_candidate",
        "strictly_monotonic_increase",
    ]
    rows: list[dict[str, object]] = []
    for condition in CONDITION_ORDER:
        group = subjects.loc[subjects["condition"].eq(condition)]
        slope_ci = _mean_ci(group["linear_slope_per_occupancy"])
        delta_ci = _mean_ci(group["max_dose_delta"])
        result: dict[str, object] = {
            "condition": condition,
            "n_subjects": len(group),
            "baseline_pci_mean": float(group["baseline_pci"].mean()),
            "max_dose_pci_mean": float(group["max_dose_pci"].mean()),
            "max_dose_delta_mean": float(group["max_dose_delta"].mean()),
            "max_dose_delta_ci95_low": delta_ci[0],
            "max_dose_delta_ci95_high": delta_ci[1],
            "linear_slope_mean": float(group["linear_slope_per_occupancy"].mean()),
            "linear_slope_ci95_low": slope_ci[0],
            "linear_slope_ci95_high": slope_ci[1],
            "linear_slope_sd": float(group["linear_slope_per_occupancy"].std(ddof=1)),
        }
        for flag in response_flags:
            count = int(group[flag].sum())
            ci_low, ci_high = _wilson_interval(count, len(group))
            result[f"{flag}_n"] = count
            result[f"{flag}_proportion"] = count / len(group)
            result[f"{flag}_ci95_low"] = ci_low
            result[f"{flag}_ci95_high"] = ci_high
        rows.append(result)
    return pd.DataFrame(rows)


def build_inferential_tests(
    subjects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slope_groups = [
        subjects.loc[
            subjects["condition"].eq(condition),
            "linear_slope_per_occupancy",
        ].to_numpy()
        for condition in CONDITION_ORDER
    ]
    welch_f, welch_p, welch_df1, welch_df2 = _welch_anova(slope_groups)
    kruskal = stats.kruskal(*slope_groups)
    tests: list[dict[str, object]] = [
        {
            "family": "global_linear_dose_by_diagnosis",
            "condition": "ALL",
            "test": "Welch_ANOVA_of_subject_linear_slopes",
            "statistic": welch_f,
            "df1": welch_df1,
            "df2": welch_df2,
            "p_value": welch_p,
            "interpretation": (
                "Tests whether the within-subject linear occupancy slope differs "
                "among diagnoses."
            ),
        },
        {
            "family": "global_linear_dose_by_diagnosis",
            "condition": "ALL",
            "test": "Kruskal_Wallis_of_subject_linear_slopes",
            "statistic": float(kruskal.statistic),
            "df1": len(CONDITION_ORDER) - 1,
            "df2": np.nan,
            "p_value": float(kruskal.pvalue),
            "interpretation": (
                "Rank-based sensitivity test for different slope distributions."
            ),
        },
    ]

    for condition in CONDITION_ORDER:
        group = subjects.loc[
            subjects["condition"].eq(condition),
            "linear_slope_per_occupancy",
        ]
        t_test = stats.ttest_1samp(group, 0.0)
        try:
            wilcoxon = stats.wilcoxon(group, alternative="two-sided")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
        n_positive = int(
            subjects.loc[
                subjects["condition"].eq(condition),
                "positive_max_dose_response",
            ].sum()
        )
        binomial = stats.binomtest(n_positive, len(group), p=0.5)
        tests.extend(
            [
                {
                    "family": "slope_vs_zero_t",
                    "condition": condition,
                    "test": "one_sample_t_test_of_subject_slopes",
                    "statistic": float(t_test.statistic),
                    "df1": float(t_test.df),
                    "df2": np.nan,
                    "p_value": float(t_test.pvalue),
                    "interpretation": (
                        "Directional linear dose trend; two-sided test against zero."
                    ),
                },
                {
                    "family": "slope_vs_zero_wilcoxon",
                    "condition": condition,
                    "test": "Wilcoxon_signed_rank_of_subject_slopes",
                    "statistic": wilcoxon_stat,
                    "df1": np.nan,
                    "df2": np.nan,
                    "p_value": wilcoxon_p,
                    "interpretation": (
                        "Rank-based sensitivity test for a non-zero typical slope."
                    ),
                },
                {
                    "family": "positive_fraction_binomial",
                    "condition": condition,
                    "test": "exact_binomial_positive_max_dose_change",
                    "statistic": n_positive,
                    "df1": len(group),
                    "df2": np.nan,
                    "p_value": float(binomial.pvalue),
                    "interpretation": (
                        "Tests whether the positive-response fraction differs from 50%."
                    ),
                },
            ]
        )

    test_table = pd.DataFrame(tests)
    test_table["p_holm_within_family"] = test_table["p_value"]
    for family in [
        "slope_vs_zero_t",
        "slope_vs_zero_wilcoxon",
        "positive_fraction_binomial",
    ]:
        mask = test_table["family"].eq(family)
        test_table.loc[mask, "p_holm_within_family"] = _holm_adjust(
            test_table.loc[mask, "p_value"]
        )

    pairwise: list[dict[str, object]] = []
    for condition_a, condition_b in combinations(CONDITION_ORDER, 2):
        values_a = subjects.loc[
            subjects["condition"].eq(condition_a),
            "linear_slope_per_occupancy",
        ]
        values_b = subjects.loc[
            subjects["condition"].eq(condition_b),
            "linear_slope_per_occupancy",
        ]
        test = stats.ttest_ind(values_a, values_b, equal_var=False)
        variance_a = float(values_a.var(ddof=1))
        variance_b = float(values_b.var(ddof=1))
        component_a = variance_a / len(values_a)
        component_b = variance_b / len(values_b)
        standard_error = float(np.sqrt(component_a + component_b))
        welch_df = float(
            (component_a + component_b) ** 2
            / (
                component_a**2 / (len(values_a) - 1)
                + component_b**2 / (len(values_b) - 1)
            )
        )
        mean_difference = float(values_a.mean() - values_b.mean())
        ci_half_width = float(stats.t.ppf(0.975, welch_df) * standard_error)
        pairwise.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_a": len(values_a),
                "n_b": len(values_b),
                "mean_slope_difference_a_minus_b": mean_difference,
                "difference_ci95_low": mean_difference - ci_half_width,
                "difference_ci95_high": mean_difference + ci_half_width,
                "difference_standard_error": standard_error,
                "welch_df": welch_df,
                "welch_t": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    pairwise_table = pd.DataFrame(pairwise)
    pairwise_table["p_holm_all_pairs"] = _holm_adjust(pairwise_table["p_value"])
    pairwise_table["significant_holm_0_05"] = pairwise_table["p_holm_all_pairs"].lt(
        0.05
    )
    return test_table, pairwise_table


def build_baseline_associations(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in [*CONDITION_ORDER, "ALL"]:
        group = (
            subjects
            if condition == "ALL"
            else subjects.loc[subjects["condition"].eq(condition)]
        )
        for outcome in ["max_dose_delta", "linear_slope_per_occupancy"]:
            result = stats.spearmanr(group["baseline_pci"], group[outcome])
            rows.append(
                {
                    "condition": condition,
                    "predictor": "baseline_pci",
                    "outcome": outcome,
                    "n_subjects": len(group),
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "caution": (
                        "Exploratory only: baseline PCI is mathematically included "
                        "in change scores, so regression-to-the-mean coupling is possible."
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table["p_holm_within_outcome"] = np.nan
    for outcome in table["outcome"].unique():
        mask = table["outcome"].eq(outcome) & table["condition"].ne("ALL")
        table.loc[mask, "p_holm_within_outcome"] = _holm_adjust(
            table.loc[mask, "p_value"]
        )
    return table


def _style_axis(axis: mpl.axes.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(length=2.5, width=0.55)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.65)


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


def plot_personalized_response(
    data: pd.DataFrame,
    subjects: pd.DataFrame,
    group_summary: pd.DataFrame,
    *,
    control_threshold: float,
    output_dir: Path,
    prefix: str,
    jitter_seed: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.3,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    rng = np.random.default_rng(jitter_seed)
    figure = plt.figure(
        figsize=(183.0 / 25.4, 5.35),
        layout="constrained",
    )
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    axis_distribution = figure.add_subplot(grid[0, 0])
    axis_proportions = figure.add_subplot(grid[0, 1])
    axis_coma_rank = figure.add_subplot(grid[1, 0])
    axis_coma_dose = figure.add_subplot(grid[1, 1])

    positions = np.arange(len(CONDITION_ORDER))
    distributions = [
        subjects.loc[
            subjects["condition"].eq(condition),
            "max_dose_delta",
        ].to_numpy()
        for condition in CONDITION_ORDER
    ]
    violins = axis_distribution.violinplot(
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
        body.set_alpha(0.15)
    for position, condition, values in zip(
        positions,
        CONDITION_ORDER,
        distributions,
        strict=True,
    ):
        jitter = rng.normal(0.0, 0.06, size=len(values))
        axis_distribution.scatter(
            position + jitter,
            values,
            s=5,
            alpha=0.42,
            color=CONDITION_COLORS[condition],
            edgecolors="none",
            zorder=2,
        )
        mean = float(np.mean(values))
        ci_low, ci_high = _mean_ci(values)
        axis_distribution.errorbar(
            position,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="o",
            markersize=3.5,
            color="#111111",
            markerfacecolor=CONDITION_COLORS[condition],
            markeredgewidth=0.8,
            capsize=2,
            linewidth=0.85,
            zorder=4,
        )
    axis_distribution.axhline(0, color="#222222", linewidth=0.65)
    axis_distribution.axhline(
        control_threshold,
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=0.7,
        label=f"Control cutoff = {control_threshold:.3f}",
    )
    axis_distribution.set_xticks(positions, CONDITION_ORDER)
    axis_distribution.set_ylabel("\N{GREEK CAPITAL LETTER DELTA}PCI at 0.766")
    _panel_label(axis_distribution, "a", "Maximum-dose responses")
    axis_distribution.legend(frameon=False, loc="upper right")
    _style_axis(axis_distribution)

    proportion_specs = [
        (
            "positive_max_dose_response",
            "Positive change",
            "#8397B4",
        ),
        (
            "control_referenced_response",
            "Above CNT 95th percentile",
            "#D57A45",
        ),
        (
            "high_dose_candidate",
            "High-dose candidate",
            "#4F8C6A",
        ),
    ]
    width = 0.23
    for offset_index, (flag, label, color) in enumerate(proportion_specs):
        x_values = positions + (offset_index - 1) * width
        proportions = group_summary[f"{flag}_proportion"].to_numpy()
        lower = group_summary[f"{flag}_ci95_low"].to_numpy()
        upper = group_summary[f"{flag}_ci95_high"].to_numpy()
        axis_proportions.bar(
            x_values,
            proportions,
            width=width * 0.9,
            color=color,
            alpha=0.85,
            label=label,
            zorder=2,
        )
        axis_proportions.errorbar(
            x_values,
            proportions,
            yerr=[proportions - lower, upper - proportions],
            fmt="none",
            color="#333333",
            linewidth=0.8,
            capsize=2,
            zorder=3,
        )
    axis_proportions.set_xticks(positions, CONDITION_ORDER)
    axis_proportions.set_ylim(0, 1)
    axis_proportions.set_ylabel("Proportion of subjects")
    axis_proportions.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    _panel_label(axis_proportions, "b", "Descriptive response fractions")
    axis_proportions.legend(frameon=False, loc="upper right")
    _style_axis(axis_proportions)

    coma = subjects.loc[subjects["condition"].eq("COMA")].sort_values("max_dose_delta")
    bar_colors = np.where(
        coma["max_dose_delta"].gt(0),
        CONDITION_COLORS["COMA"],
        "#AEB5C1",
    )
    axis_coma_rank.barh(
        coma["subject_id"],
        coma["max_dose_delta"],
        color=bar_colors,
        alpha=0.88,
        height=0.72,
    )
    axis_coma_rank.axvline(0, color="#222222", linewidth=0.65)
    axis_coma_rank.axvline(
        control_threshold,
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=0.7,
    )
    axis_coma_rank.set_xlabel("\N{GREEK CAPITAL LETTER DELTA}PCI at 0.766")
    axis_coma_rank.set_ylabel("COMA subject")
    _panel_label(axis_coma_rank, "c", "COMA subjects ranked by response")
    _style_axis(axis_coma_rank)
    axis_coma_rank.grid(axis="x", color="#D9D9D9", linewidth=0.4, alpha=0.65)
    axis_coma_rank.grid(axis="y", visible=False)

    coma_data = data.loc[data["condition"].eq("COMA")]
    highlighted = set(
        coma.loc[coma["control_referenced_response"], "subject_id"].astype(str)
    )
    for subject_id, group in coma_data.groupby("subject_id"):
        group = group.sort_values("occupancy")
        is_highlighted = str(subject_id) in highlighted
        axis_coma_dose.plot(
            group["occupancy"],
            group["pci_mean"],
            marker="o",
            markersize=2.5 if is_highlighted else 1.7,
            linewidth=1.1 if is_highlighted else 0.45,
            alpha=0.92 if is_highlighted else 0.20,
            color=CONDITION_COLORS["COMA"],
            zorder=3 if is_highlighted else 1,
        )
        if is_highlighted:
            final = group.iloc[-1]
            axis_coma_dose.annotate(
                str(subject_id),
                (final["occupancy"], final["pci_mean"]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=5,
                color="#222222",
            )
    occupancies = np.sort(data["occupancy"].unique())
    axis_coma_dose.set_xticks(occupancies)
    axis_coma_dose.set_xlabel(r"5-HT$_{2A}$ occupancy")
    axis_coma_dose.set_ylabel("PCI")
    _panel_label(axis_coma_dose, "d", "COMA dose trajectories")
    _style_axis(axis_coma_dose)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{prefix}_responder_figure"
    for extension in ["pdf", "svg"]:
        figure.savefig(stem.with_suffix(f".{extension}"))
    png_path = stem.with_suffix(".png")
    figure.savefig(png_path, dpi=600)
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
    tiff_path = stem.with_suffix(".tiff")
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
    plt.close(figure)


def write_readme(
    path: Path,
    *,
    input_path: Path,
    subjects: pd.DataFrame,
    group_summary: pd.DataFrame,
    inferential_tests: pd.DataFrame,
    control_threshold: float,
    large_change: float,
    control_quantile: float,
) -> None:
    global_row = inferential_tests.loc[
        inferential_tests["test"].eq("Welch_ANOVA_of_subject_linear_slopes")
    ].iloc[0]
    lines = [
        "# Personalized serotonergic PCI analysis",
        "",
        f"Source: `{input_path}`",
        "",
        "## Primary result",
        "",
        (
            "Diagnosis altered the within-subject linear occupancy response "
            f"(Welch F({global_row['df1']:.0f}, {global_row['df2']:.2f})="
            f"{global_row['statistic']:.3f}, p={global_row['p_value']:.3g})."
        ),
        "",
        "## Group summary",
        "",
        (
            "| Group | n | Mean slope | 95% CI | Positive max-dose change | "
            "Control-referenced response |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_summary.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.n_subjects} | {row.linear_slope_mean:.3f} | "
            f"[{row.linear_slope_ci95_low:.3f}, "
            f"{row.linear_slope_ci95_high:.3f}] | "
            f"{row.positive_max_dose_response_n}/{row.n_subjects} "
            f"({row.positive_max_dose_response_proportion:.1%}) | "
            f"{row.control_referenced_response_n}/{row.n_subjects} "
            f"({row.control_referenced_response_proportion:.1%}) |"
        )
    lines.extend(
        [
            "",
            "## Responder definitions",
            "",
            "- `positive_max_dose_response`: PCI at maximum occupancy is above baseline.",
            (
                f"- `large_absolute_response`: maximum-dose PCI increase is at least "
                f"{large_change:.3f}. This is a descriptive threshold."
            ),
            (
                f"- `control_referenced_response`: increase exceeds the "
                f"{control_quantile:.0%} quantile of CNT changes "
                f"({control_threshold:.6f}). This is a reference distribution, not "
                "an individual significance test."
            ),
            (
                "- `high_dose_candidate`: control-referenced response, positive "
                "linear slope, and the subject's largest PCI occurs at maximum dose."
            ),
            (
                "- `consistent_high_dose_candidate`: high-dose candidate with at "
                "least two of the three consecutive-dose increments positive."
            ),
            "",
            "## Interpretive limits",
            "",
            (
                "Each subject-dose cell contains one PCI computed from the "
                "time-locked average of all trials. The table has no subject-dose "
                "sampling uncertainty, so individual labels are descriptive."
            ),
            (
                "The analysis identifies simulated PCI response, not emergence from "
                "coma or treatment benefit. Recovery outcomes, etiology, time since "
                "injury, lesion features, structural-connectome features, age, and "
                "medication are absent and therefore cannot explain or validate "
                "personalized clinical response."
            ),
            (
                "Baseline-versus-change associations can be inflated because "
                "baseline is part of the change score. Treat those correlations as "
                "hypothesis-generating only."
            ),
            "",
            f"Total subjects: {subjects['subject_uid'].nunique()}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 < args.control_quantile < 1:
        raise ValueError("--control-quantile must be strictly between zero and one.")
    if args.large_change <= 0:
        raise ValueError("--large-change must be positive.")

    data = load_data(args.input)
    subjects, control_threshold = build_subject_metrics(
        data,
        large_change=args.large_change,
        control_quantile=args.control_quantile,
    )
    group_summary = build_group_summary(subjects)
    inferential_tests, pairwise_tests = build_inferential_tests(subjects)
    baseline_associations = build_baseline_associations(subjects)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / f"{args.prefix}_analysis_input.csv", index=False)
    subjects.to_csv(
        args.output_dir / f"{args.prefix}_subject_responder_table.csv",
        index=False,
    )
    group_summary.to_csv(
        args.output_dir / f"{args.prefix}_group_summary.csv",
        index=False,
    )
    inferential_tests.to_csv(
        args.output_dir / f"{args.prefix}_inferential_tests.csv",
        index=False,
    )
    pairwise_tests.to_csv(
        args.output_dir / f"{args.prefix}_pairwise_slope_tests.csv",
        index=False,
    )
    baseline_associations.to_csv(
        args.output_dir / f"{args.prefix}_baseline_associations.csv",
        index=False,
    )
    subjects.loc[subjects["condition"].eq("COMA")].to_csv(
        args.output_dir / f"{args.prefix}_coma_subjects_ranked.csv",
        index=False,
    )

    plot_personalized_response(
        data,
        subjects,
        group_summary,
        control_threshold=control_threshold,
        output_dir=args.output_dir,
        prefix=args.prefix,
        jitter_seed=args.jitter_seed,
    )
    write_readme(
        args.output_dir / "README.md",
        input_path=args.input,
        subjects=subjects,
        group_summary=group_summary,
        inferential_tests=inferential_tests,
        control_threshold=control_threshold,
        large_change=args.large_change,
        control_quantile=args.control_quantile,
    )

    manifest = {
        "input_path": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "n_rows": len(data),
        "n_subjects": int(subjects["subject_uid"].nunique()),
        "conditions": CONDITION_ORDER,
        "occupancies": sorted(data["occupancy"].unique().tolist()),
        "n_trials_per_subject_dose": int(data["n_trials"].iloc[0]),
        "pci_estimator": str(data["pci_estimator"].iloc[0]),
        "large_change_threshold": args.large_change,
        "control_reference_quantile": args.control_quantile,
        "control_reference_delta_threshold": control_threshold,
        "individual_inference": (
            "descriptive_only; one PCI value and no uncertainty per subject-dose cell"
        ),
    }
    (args.output_dir / f"{args.prefix}_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(group_summary.to_string(index=False))
    print()
    print(inferential_tests.to_string(index=False))
    print(f"\nWrote personalized PCI analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
