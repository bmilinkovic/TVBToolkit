#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "notebooks"
    / "outputs"
    / "structural_zero_edges"
    / "brain_act_structural_zero_edges_by_subject.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "outputs" / "structural_zero_edges"

COHORT_ORDER = ["control", "emcs", "mcs", "uws", "coma"]
COHORT_LABELS = {
    "control": "Control",
    "emcs": "EMCS",
    "mcs": "MCS",
    "uws": "UWS",
    "coma": "Coma",
}
COHORT_COLORS = {
    "control": "#5B8A72",
    "emcs": "#E8B56D",
    "mcs": "#C5622F",
    "uws": "#8B6B8B",
    "coma": "#3B4A6B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run nonparametric statistics on the BrainAct structural zero-edge "
            "subject table and save a separate summary figure."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = float(p_values[idx])
        adj = (m - rank) * raw
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = 0
    lt = 0
    for xv in x:
        gt += int(np.sum(xv > y))
        lt += int(np.sum(xv < y))
    return float((gt - lt) / (x.size * y.size))


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v, v
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    medians = np.median(draws, axis=1)
    lo = float(np.quantile(medians, alpha / 2.0))
    hi = float(np.quantile(medians, 1.0 - alpha / 2.0))
    return median, lo, hi


def build_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    grouped = {
        cohort: df.loc[df["cohort"] == cohort, "pct_zero_edges"].to_numpy(dtype=float)
        for cohort in COHORT_ORDER
    }

    summary = (
        df.groupby(["cohort", "condition"], observed=True)["pct_zero_edges"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    kw_stat, kw_p = kruskal(*(grouped[c] for c in COHORT_ORDER))
    n_total = int(df.shape[0])
    k = len(COHORT_ORDER)
    kw_eps_sq = float(max((kw_stat - k + 1) / (n_total - k), 0.0))

    rank_map = {cohort: idx for idx, cohort in enumerate(COHORT_ORDER)}
    rank_values = df["cohort"].map(rank_map).to_numpy(dtype=float)
    rho, rho_p = spearmanr(rank_values, df["pct_zero_edges"].to_numpy(dtype=float))

    pairwise_rows: list[dict[str, object]] = []
    raw_ps: list[float] = []
    pairs = list(combinations(COHORT_ORDER, 2))
    for left, right in pairs:
        x = grouped[left]
        y = grouped[right]
        u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        raw_ps.append(float(p_val))
        pairwise_rows.append(
            {
                "group_a": COHORT_LABELS[left],
                "group_b": COHORT_LABELS[right],
                "n_a": int(x.size),
                "n_b": int(y.size),
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
                "mannwhitney_u": float(u_stat),
                "p_raw": float(p_val),
                "cliffs_delta": float(cliffs_delta(x, y)),
            }
        )

    adj_ps = holm_adjust(raw_ps)
    for row, adj in zip(pairwise_rows, adj_ps, strict=True):
        row["p_holm"] = float(adj)
        row["significant_holm_0p05"] = bool(adj < 0.05)

    omnibus = {
        "n_subjects_total": n_total,
        "condition_order": [COHORT_LABELS[c] for c in COHORT_ORDER],
        "kruskal_wallis": {
            "statistic_h": float(kw_stat),
            "p_value": float(kw_p),
            "epsilon_squared": kw_eps_sq,
        },
        "spearman_trend": {
            "rho": float(rho),
            "p_value": float(rho_p),
            "rank_definition": {
                COHORT_LABELS[c]: idx for idx, c in enumerate(COHORT_ORDER)
            },
        },
    }

    pairwise = pd.DataFrame(pairwise_rows)
    return summary, pairwise, omnibus


def make_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    omnibus: dict[str, object],
    output_path: Path,
    *,
    seed: int,
    n_boot: int,
) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.2),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    ax0, ax1 = axes

    data = [
        df.loc[df["cohort"] == cohort, "pct_zero_edges"].to_numpy(dtype=float)
        for cohort in COHORT_ORDER
    ]
    positions = np.arange(1, len(COHORT_ORDER) + 1)

    violin = ax0.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, cohort in zip(violin["bodies"], COHORT_ORDER, strict=True):
        body.set_facecolor(COHORT_COLORS[cohort])
        body.set_edgecolor(COHORT_COLORS[cohort])
        body.set_alpha(0.28)

    for xpos, cohort in zip(positions, COHORT_ORDER, strict=True):
        vals = df.loc[df["cohort"] == cohort, "pct_zero_edges"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.14, 0.14, size=vals.size)
        ax0.scatter(
            np.full(vals.shape, xpos, dtype=float) + jitter,
            vals,
            s=24,
            alpha=0.82,
            color=COHORT_COLORS[cohort],
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        median = float(np.median(vals))
        ax0.hlines(median, xpos - 0.19, xpos + 0.19, colors="black", linewidth=1.1, zorder=4)

    kw = omnibus["kruskal_wallis"]
    trend = omnibus["spearman_trend"]
    stat_text = (
        f"Kruskal-Wallis: H = {kw['statistic_h']:.2f}, p = {kw['p_value']:.2e}\n"
        f"Spearman trend: rho = {trend['rho']:.3f}, p = {trend['p_value']:.2e}"
    )
    ax0.text(
        0.02,
        0.98,
        stat_text,
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.95, "boxstyle": "round,pad=0.35"},
    )
    ax0.set_xticks(positions, [COHORT_LABELS[c] for c in COHORT_ORDER])
    ax0.set_xlabel("Condition")
    ax0.set_ylabel("Zero structural edges (%)")
    ax0.set_title("Subject-level distributions")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.grid(axis="y", alpha=0.2, linewidth=0.6)

    med_rows = []
    for cohort in COHORT_ORDER:
        vals = df.loc[df["cohort"] == cohort, "pct_zero_edges"].to_numpy(dtype=float)
        median, lo, hi = bootstrap_median_ci(vals, rng=rng, n_boot=n_boot)
        med_rows.append((cohort, median, lo, hi))

    y = np.arange(len(COHORT_ORDER))
    for idx, (cohort, median, lo, hi) in enumerate(med_rows):
        ax1.hlines(idx, lo, hi, color=COHORT_COLORS[cohort], linewidth=2.2)
        ax1.scatter(
            median,
            idx,
            s=58,
            color=COHORT_COLORS[cohort],
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax1.set_yticks(y, [COHORT_LABELS[c] for c in COHORT_ORDER])
    ax1.invert_yaxis()
    ax1.set_xlabel("Median zero-edge percentage (%)")
    ax1.set_title("Median with 95% bootstrap CI")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.2, linewidth=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv.resolve())
    df["cohort"] = pd.Categorical(df["cohort"], categories=COHORT_ORDER, ordered=True)
    df = df.sort_values(["cohort", "subject_id"]).reset_index(drop=True)

    summary, pairwise, omnibus = build_stats(df)

    summary_csv = output_dir / "brain_act_structural_zero_edges_summary_stats.csv"
    pairwise_csv = output_dir / "brain_act_structural_zero_edges_pairwise_stats.csv"
    omnibus_json = output_dir / "brain_act_structural_zero_edges_omnibus_stats.json"
    fig_png = output_dir / "brain_act_structural_zero_edges_stats.png"
    fig_pdf = output_dir / "brain_act_structural_zero_edges_stats.pdf"

    summary.to_csv(summary_csv, index=False)
    pairwise.to_csv(pairwise_csv, index=False)
    omnibus_json.write_text(json.dumps(omnibus, indent=2), encoding="utf-8")
    make_figure(df, summary, omnibus, fig_png, seed=args.seed, n_boot=args.bootstrap_iterations)
    make_figure(df, summary, omnibus, fig_pdf, seed=args.seed, n_boot=args.bootstrap_iterations)

    print(f"Saved summary stats to: {summary_csv}")
    print(f"Saved pairwise stats to: {pairwise_csv}")
    print(f"Saved omnibus stats to: {omnibus_json}")
    print(f"Saved stats figure to: {fig_png}")
    print(f"Saved stats figure to: {fig_pdf}")


if __name__ == "__main__":
    main()
