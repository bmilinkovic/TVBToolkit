#!/usr/bin/env python3
"""Plot + stats pass from saved Brain-Act CSV outputs (no re-simulation).

If BOLD columns are present and finite, figures are generated in dual-domain layout:
- Figure 1: row 1 rates, row 2 BOLD (LZc/PCI columns)
- Figure 2: row 1 rates, row 2 BOLD (SCFC vs occupancy)
- Figure 3: row 1 rates, row 2 BOLD (LZc/PCI columns with sedation split)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, spearmanr, linregress


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.linewidth": 1.2,
        }
    )


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def holm_correct(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    out = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        adj = max(adj, prev)
        out[idx] = min(adj, 1.0)
        prev = out[idx]
    return out


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def normalize_sedation_group(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "unknown", "na", "n/a"}:
        return "unknown"
    if "non_sedated" in s or "non-sedated" in s or "non sedated" in s or "unsedated" in s or "awake" in s:
        return "non_sedated"
    return "sedated"


def safe_linreg(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return None
    return linregress(x, y)


def has_finite(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(x).any())


def compute_pairwise_cohort_stats(
    metrics: pd.DataFrame,
    scenario_order: list[str],
    cohorts_present: list[str],
    metrics_to_test: list[str],
) -> pd.DataFrame:
    pair_rows = []
    pairwise = [(a, b) for i, a in enumerate(cohorts_present) for b in cohorts_present[i + 1 :]]

    for metric in metrics_to_test:
        for scenario in scenario_order:
            sub_s = metrics[metrics["scenario"] == scenario]
            raw, tmp = [], []
            for a, b in pairwise:
                xa = sub_s[sub_s["cohort"] == a][metric].to_numpy(dtype=float)
                xb = sub_s[sub_s["cohort"] == b][metric].to_numpy(dtype=float)
                xa = xa[np.isfinite(xa)]
                xb = xb[np.isfinite(xb)]
                if xa.size == 0 or xb.size == 0:
                    continue
                U, p = mannwhitneyu(xa, xb, alternative="two-sided")
                raw.append(float(p))
                tmp.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "contrast": f"{a} vs {b}",
                        "U": float(U),
                        "p_raw": float(p),
                        "median_a": float(np.median(xa)),
                        "median_b": float(np.median(xb)),
                        "n_a": int(xa.size),
                        "n_b": int(xb.size),
                    }
                )
            if tmp:
                adj = holm_correct(np.asarray(raw, dtype=float))
                for row, p_adj in zip(tmp, adj):
                    row["p_holm"] = float(p_adj)
                    row["stars"] = p_to_stars(float(p_adj))
                    pair_rows.append(row)

    return pd.DataFrame(pair_rows)


def compute_pairwise_sedation_stats(
    metrics: pd.DataFrame,
    scenario_order: list[str],
    cohorts_present: list[str],
    metrics_to_test: list[str],
) -> pd.DataFrame:
    rows = []
    for metric in metrics_to_test:
        raw = []
        tmp = []
        for scenario in scenario_order:
            sub_s = metrics[metrics["scenario"] == scenario]
            for cohort in cohorts_present:
                sub = sub_s[sub_s["cohort"] == cohort]
                xa = sub[sub["sedation_group_fixed"] == "sedated"][metric].to_numpy(dtype=float)
                xb = sub[sub["sedation_group_fixed"] == "non_sedated"][metric].to_numpy(dtype=float)
                xa = xa[np.isfinite(xa)]
                xb = xb[np.isfinite(xb)]
                if xa.size == 0 or xb.size == 0:
                    continue
                U, p = mannwhitneyu(xa, xb, alternative="two-sided")
                raw.append(float(p))
                tmp.append(
                    {
                        "scenario": scenario,
                        "cohort": cohort,
                        "metric": metric,
                        "contrast": "sedated vs non_sedated",
                        "U": float(U),
                        "p_raw": float(p),
                        "median_sedated": float(np.median(xa)),
                        "median_non_sedated": float(np.median(xb)),
                        "n_sedated": int(xa.size),
                        "n_non_sedated": int(xb.size),
                    }
                )
        if tmp:
            adj = holm_correct(np.asarray(raw, dtype=float))
            for row, p_adj in zip(tmp, adj):
                row["p_holm"] = float(p_adj)
                row["stars"] = p_to_stars(float(p_adj))
                rows.append(row)
    return pd.DataFrame(rows)


def compute_pairwise_sedation_by_scenario(
    metrics: pd.DataFrame,
    scenario_order: list[str],
    metrics_to_test: list[str],
) -> pd.DataFrame:
    rows = []
    for metric in metrics_to_test:
        raw = []
        tmp = []
        for scenario in scenario_order:
            sub_s = metrics[metrics["scenario"] == scenario]
            xa = sub_s[sub_s["sedation_group_fixed"] == "sedated"][metric].to_numpy(dtype=float)
            xb = sub_s[sub_s["sedation_group_fixed"] == "non_sedated"][metric].to_numpy(dtype=float)
            xa = xa[np.isfinite(xa)]
            xb = xb[np.isfinite(xb)]
            if xa.size == 0 or xb.size == 0:
                continue
            U, p = mannwhitneyu(xa, xb, alternative="two-sided")
            raw.append(float(p))
            tmp.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "contrast": "sedated vs non_sedated",
                    "U": float(U),
                    "p_raw": float(p),
                    "median_sedated": float(np.median(xa)),
                    "median_non_sedated": float(np.median(xb)),
                    "n_sedated": int(xa.size),
                    "n_non_sedated": int(xb.size),
                }
            )
        if tmp:
            adj = holm_correct(np.asarray(raw, dtype=float))
            for row, p_adj in zip(tmp, adj):
                row["p_holm"] = float(p_adj)
                row["stars"] = p_to_stars(float(p_adj))
                rows.append(row)
    return pd.DataFrame(rows)


def draw_brackets(
    ax: plt.Axes,
    x_pairs: list[tuple[float, float]],
    stars: list[str],
    y_start: float,
    y_step: float,
    lw: float = 2.8,
    fs: float = 20,
) -> float:
    y = y_start
    for (x1, x2), s in zip(x_pairs, stars):
        if not s or s == "ns":
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        tick = 0.28 * y_step
        ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="black", lw=lw, zorder=6)
        ax.text((x1 + x2) / 2.0, y + 0.08 * y_step, s, ha="center", va="bottom", fontsize=fs, fontweight="bold")
        y += y_step
    return y


def as_2d_axes(axes, n_rows: int, n_cols: int):
    arr = np.array(axes, dtype=object)
    if n_rows == 1 and n_cols == 1:
        return arr.reshape(1, 1)
    if n_rows == 1:
        return arr.reshape(1, n_cols)
    if n_cols == 1:
        return arr.reshape(n_rows, 1)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="notebooks/outputs/ba_rates")
    args = ap.parse_args()

    set_publication_style()

    root = Path(args.root).resolve()
    res = root / "res"
    fig = root / "figs"
    fig.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(res / "dual_domain_metrics.csv")
    states = pd.read_csv(res / "dual_domain_state_rows.csv")

    metrics["sedation_group_fixed"] = metrics["sedation"].map(normalize_sedation_group)
    states["sedation_group_fixed"] = states["sedation"].map(normalize_sedation_group)

    scenario_order = [
        "private_alpha0",
        "global_alpha_low",
        "global_alpha_med",
        "global_alpha_high",
        "sc_alpha_med",
    ]
    scenario_order = [s for s in scenario_order if s in set(metrics["scenario"])]
    scenario_labels = {
        "private_alpha0": "No correlated noise",
        "global_alpha_low": "Low correlated noise",
        "global_alpha_med": "Medium correlated noise",
        "global_alpha_high": "High correlated noise",
        "sc_alpha_med": "SC correlated noise",
    }

    preferred_cohorts = ["coma", "uws", "mcs", "emcs", "control"]
    cohorts_present = [c for c in preferred_cohorts if c in set(metrics["cohort"])]

    colors = {
        "control": "#5FA7C6",
        "emcs": "#8EA65E",
        "mcs": "#E1A84A",
        "uws": "#C2543D",
        "coma": "#6B5876",
    }
    labels = {"coma": "COMA", "uws": "UWS", "mcs": "MCS", "emcs": "EMCS", "control": "CONTROL"}

    # pci_bold is intentionally excluded: PCI does not apply to BOLD signals.
    # BOLD rows are drawn whenever lzc_bold and the SCFC/occupancy columns are available.
    bold_available = (
        has_finite(metrics, "lzc_bold")
        and has_finite(states, "sfc_bold")
        and has_finite(states, "occ_bold")
    )

    domains = [
        {
            "name": "rates",
            "label": "Firing rates",
            "lzc": "lzc_rate",
            "pci": "pci_rate",
            "sfc": "sfc_rate",
            "occ": "occ_rate",
        }
    ]
    if bold_available:
        domains.append(
            {
                "name": "bold",
                "label": "BOLD",
                "lzc": "lzc_bold",
                "pci": "pci_bold",
                "sfc": "sfc_bold",
                "occ": "occ_bold",
            }
        )

    metrics_to_test = []
    for d in domains:
        metrics_to_test.extend([d["lzc"], d["pci"]])

    pair_df = compute_pairwise_cohort_stats(metrics, scenario_order, cohorts_present, metrics_to_test)
    pair_df.to_csv(res / "stats_pairwise_mannwhitney_holm_with_stars.csv", index=False)

    sed_pair_df = compute_pairwise_sedation_stats(metrics, scenario_order, cohorts_present, metrics_to_test)
    sed_pair_df.to_csv(res / "stats_pairwise_sedation_within_cohort_holm_with_stars.csv", index=False)

    sed_scenario_df = compute_pairwise_sedation_by_scenario(metrics, scenario_order, metrics_to_test)
    sed_scenario_df.to_csv(res / "stats_pairwise_sedation_by_scenario_holm_with_stars.csv", index=False)

    n_rows = len(domains)
    n_cols = 2
    cohort_offsets = np.linspace(-0.34, 0.34, max(1, len(cohorts_present)))
    bar_w = 0.13

    # Figure 1: complexity across scenarios (rates row + optional BOLD row)
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(24, 7.2 * n_rows), sharex=True)
    axes1 = as_2d_axes(axes1, n_rows, n_cols)

    for r, dom in enumerate(domains):
        for c, metric in enumerate([dom["lzc"], dom["pci"]]):
            ax = axes1[r, c]
            if not has_finite(metrics, metric):
                ax.set_visible(False)
                continue
            y_min, y_max = np.inf, -np.inf

            for s_idx, scenario in enumerate(scenario_order):
                sub_s = metrics[metrics["scenario"] == scenario]
                heights = {}
                tops = {}
                for c_idx, cohort in enumerate(cohorts_present):
                    sub = sub_s[sub_s["cohort"] == cohort]
                    y = sub[metric].to_numpy(dtype=float)
                    y = y[np.isfinite(y)]
                    if y.size == 0:
                        continue
                    x = s_idx + cohort_offsets[c_idx]
                    m = float(np.mean(y))
                    e = sem(y)
                    heights[cohort] = m
                    tops[cohort] = m + e
                    y_min = min(y_min, float(np.min(y)))
                    y_max = max(y_max, float(np.max(y)), m + e)

                    ax.bar(
                        x,
                        m,
                        width=bar_w,
                        color=colors[cohort],
                        alpha=0.32,
                        edgecolor=colors[cohort],
                        linewidth=1.6,
                        zorder=2,
                    )
                    if e > 0:
                        ax.errorbar(x, m, yerr=e, color=colors[cohort], lw=1.8, capsize=4, zorder=3)

                    jitter = np.linspace(-0.04, 0.04, max(1, y.size))[: y.size]
                    ax.scatter(x + jitter, y, s=11, alpha=0.16, color=colors[cohort], edgecolors="none", zorder=1)

                hit = pair_df[(pair_df["scenario"] == scenario) & (pair_df["metric"] == metric) & (pair_df["p_holm"] < 0.05)]
                hit = hit.sort_values("p_holm", ascending=True)
                if not hit.empty and heights:
                    local_top = max(tops.values())
                    local_span = max(1e-6, local_top - min(heights.values()))
                    y_start = local_top + 0.48 * local_span
                    y_step = max(0.18 * local_span, 0.018)
                    x_pairs, stars = [], []
                    for _, row in hit.iterrows():
                        a, b = str(row["contrast"]).split(" vs ")
                        if a not in cohorts_present or b not in cohorts_present:
                            continue
                        x1 = s_idx + cohort_offsets[cohorts_present.index(a)]
                        x2 = s_idx + cohort_offsets[cohorts_present.index(b)]
                        x_pairs.append((x1, x2))
                        stars.append(str(row["stars"]))
                    if x_pairs:
                        y_used = draw_brackets(ax, x_pairs, stars, y_start=y_start, y_step=y_step, lw=2.8, fs=20)
                        y_max = max(y_max, y_used + 0.12 * local_span)

            if np.isfinite(y_min) and np.isfinite(y_max):
                span = max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - 0.08 * span, y_max + 0.08 * span)

            ax.set_ylabel("Lempel-Ziv Complexity" if c == 0 else "PCI")
            ax.set_xticks(np.arange(len(scenario_order)))
            ax.set_xticklabels([scenario_labels.get(s, s) for s in scenario_order], rotation=20, ha="right")
            ax.grid(alpha=0.28, axis="y")

            if c == 0:
                ax.text(-0.28, 1.03, dom["label"], transform=ax.transAxes, fontsize=18, fontweight="bold", ha="left", va="bottom")

    cohort_handles = [Patch(facecolor=colors[c], edgecolor=colors[c], alpha=0.42, label=labels.get(c, c.upper())) for c in cohorts_present]
    fig1.legend(handles=cohort_handles, loc="upper center", ncol=min(5, len(cohort_handles)), bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=20)
    fig1.suptitle("Complexity Across Correlated Noise Scenarios", y=1.04, fontsize=24)
    fig1.tight_layout()
    fig1.savefig(fig / "fig01_complexity_dual_domain_annotated.svg", bbox_inches="tight")
    fig1.savefig(fig / "fig01_complexity_dual_domain_annotated.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: occupancy vs SCFC coupling, faceted by scenario (rates row + optional BOLD row)
    fig2, axes2 = plt.subplots(
        n_rows,
        len(scenario_order),
        figsize=(5.8 * len(scenario_order), 6.0 * n_rows),
        sharex=False,
        sharey=False,
    )
    axes2 = as_2d_axes(axes2, n_rows, len(scenario_order))

    reg_rows = []
    for r, dom in enumerate(domains):
        for s_idx, scenario in enumerate(scenario_order):
            ax2 = axes2[r, s_idx]
            sub_sc = states[states["scenario"] == scenario]
            for cohort in cohorts_present:
                sub = sub_sc[sub_sc["cohort"] == cohort]
                x = sub[dom["sfc"]].to_numpy(dtype=float)
                y = sub[dom["occ"]].to_numpy(dtype=float)
                m = np.isfinite(x) & np.isfinite(y)
                x = x[m]
                y = y[m]
                if x.size == 0:
                    continue

                ax2.scatter(x, y, s=11, alpha=0.14, color=colors[cohort], edgecolors="none")

                rho, p_rho = (float("nan"), float("nan"))
                if x.size >= 3:
                    rho, p_rho = spearmanr(x, y)

                lr = safe_linreg(x, y)
                slope = intercept = p_slope = r_lin = float("nan")
                if lr is not None:
                    slope = float(lr.slope)
                    intercept = float(lr.intercept)
                    p_slope = float(lr.pvalue)
                    r_lin = float(lr.rvalue)
                    xx = np.linspace(float(np.min(x)), float(np.max(x)), 140)
                    yy = slope * xx + intercept
                    ax2.plot(xx, yy, color=colors[cohort], lw=4.2, alpha=0.74)

                reg_rows.append(
                    {
                        "domain": dom["name"],
                        "scenario": scenario,
                        "scenario_label": scenario_labels.get(scenario, scenario),
                        "group": cohort,
                        "n_points": int(x.size),
                        "spearman_rho": float(rho),
                        "spearman_p": float(p_rho),
                        "spearman_stars": p_to_stars(float(p_rho)),
                        "slope": slope,
                        "slope_p": p_slope,
                        "slope_stars": p_to_stars(float(p_slope)),
                        "lin_r": r_lin,
                    }
                )

            y_all = sub_sc[dom["occ"]].to_numpy(dtype=float)
            y_all = y_all[np.isfinite(y_all)]
            if y_all.size > 5:
                lo = float(np.quantile(y_all, 0.01))
                hi = float(np.quantile(y_all, 0.98))
                span = max(1e-6, hi - lo)
                ax2.set_ylim(max(0.0, lo - 0.05 * span), hi + 0.08 * span)

            if r == n_rows - 1:
                ax2.set_xlabel("SCFC coupling")
            if s_idx == 0:
                ax2.set_ylabel("Occupancy probability")
            ax2.grid(alpha=0.30)
            if r == 0:
                ax2.set_title(scenario_labels.get(scenario, scenario), fontsize=15, pad=10)
            if s_idx == 0:
                ax2.text(0.01, 0.98, dom["label"], transform=ax2.transAxes, fontsize=18, fontweight="bold", ha="left", va="top")

    handles = [Line2D([0], [0], marker="o", color="w", label=labels[c], markerfacecolor=colors[c], markersize=9) for c in cohorts_present]
    axes2[0, 0].legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=13)
    fig2.suptitle("Subject level occupancy versus SCFC coupling by scenario", y=1.01, fontsize=24)
    fig2.tight_layout()
    fig2.savefig(fig / "fig02_sfc_vs_occupancy_dual_domain_annotated.svg", bbox_inches="tight")
    fig2.savefig(fig / "fig02_sfc_vs_occupancy_dual_domain_annotated.pdf", bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: complexity with sedation split (rates row + optional BOLD row)
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(24, 7.2 * n_rows), sharex=True)
    axes3 = as_2d_axes(axes3, n_rows, n_cols)
    sed_groups = ["non_sedated", "sedated"]
    sed_offsets = {"non_sedated": -0.035, "sedated": 0.035}
    sed_hatch = {"non_sedated": "....", "sedated": "xxxx"}
    sed_alpha = {"non_sedated": 0.22, "sedated": 0.42}

    for r, dom in enumerate(domains):
        for c, metric in enumerate([dom["lzc"], dom["pci"]]):
            ax = axes3[r, c]
            if not has_finite(metrics, metric):
                ax.set_visible(False)
                continue
            y_min, y_max = np.inf, -np.inf

            for s_idx, scenario in enumerate(scenario_order):
                sub_s = metrics[metrics["scenario"] == scenario]
                local_tops = {}

                for c_idx, cohort in enumerate(cohorts_present):
                    c_off = cohort_offsets[c_idx]
                    sub_c = sub_s[sub_s["cohort"] == cohort]
                    for sed in sed_groups:
                        y = sub_c[sub_c["sedation_group_fixed"] == sed][metric].to_numpy(dtype=float)
                        y = y[np.isfinite(y)]
                        if y.size == 0:
                            continue
                        x = s_idx + c_off + sed_offsets[sed]
                        m = float(np.mean(y))
                        e = sem(y)
                        local_tops[(cohort, sed)] = m + e
                        y_min = min(y_min, float(np.min(y)))
                        y_max = max(y_max, float(np.max(y)), m + e)

                        ax.bar(
                            x,
                            m,
                            width=0.062,
                            color=colors[cohort],
                            alpha=sed_alpha[sed],
                            edgecolor="#2F2F2F",
                            linewidth=1.9,
                            hatch=sed_hatch[sed],
                            zorder=2,
                        )
                        if e > 0:
                            ax.errorbar(x, m, yerr=e, color=colors[cohort], lw=1.6, capsize=3.5, zorder=3)

                        jitter = np.linspace(-0.018, 0.018, max(1, y.size))[: y.size]
                        ax.scatter(x + jitter, y, s=8, alpha=0.12, color=colors[cohort], edgecolors="none", zorder=1)

                hits = sed_pair_df[
                    (sed_pair_df["scenario"] == scenario)
                    & (sed_pair_df["metric"] == metric)
                    & (sed_pair_df["p_holm"] < 0.05)
                ].sort_values("p_holm", ascending=True)

                if not hits.empty and local_tops:
                    local_max = max(local_tops.values())
                    local_min = min(v for v in local_tops.values())
                    span = max(1e-6, local_max - local_min)
                    y_step = max(0.18 * span, 0.018)
                    y_cursor = local_max + 0.48 * span

                    x_pairs, stars = [], []
                    for _, row in hits.iterrows():
                        cohort = str(row["cohort"])
                        if (cohort, "non_sedated") not in local_tops or (cohort, "sedated") not in local_tops:
                            continue
                        c_off = cohort_offsets[cohorts_present.index(cohort)]
                        x1 = s_idx + c_off + sed_offsets["non_sedated"]
                        x2 = s_idx + c_off + sed_offsets["sedated"]
                        x_pairs.append((x1, x2))
                        stars.append(str(row["stars"]))
                    if x_pairs:
                        y_used = draw_brackets(ax, x_pairs, stars, y_start=y_cursor, y_step=y_step, lw=2.8, fs=20)
                        y_max = max(y_max, y_used + 0.10 * span)

                hit_global = sed_scenario_df[
                    (sed_scenario_df["scenario"] == scenario)
                    & (sed_scenario_df["metric"] == metric)
                    & (sed_scenario_df["p_holm"] < 0.05)
                ]
                if not hit_global.empty and local_tops:
                    gstars = str(hit_global.iloc[0]["stars"])
                    if gstars and gstars != "ns":
                        local_max = max(local_tops.values())
                        local_min = min(v for v in local_tops.values())
                        span = max(1e-6, local_max - local_min)
                        y_text = local_max + 0.95 * span
                        ax.text(
                            s_idx,
                            y_text,
                            f"S vs NS: {gstars}",
                            ha="center",
                            va="bottom",
                            fontsize=18,
                            fontweight="bold",
                            color="black",
                        )
                        y_max = max(y_max, y_text + 0.12 * span)

            if np.isfinite(y_min) and np.isfinite(y_max):
                span = max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - 0.08 * span, y_max + 0.10 * span)

            ax.set_ylabel("Lempel-Ziv Complexity" if c == 0 else "PCI")
            ax.set_xticks(np.arange(len(scenario_order)))
            ax.set_xticklabels([scenario_labels.get(s, s) for s in scenario_order], rotation=20, ha="right")
            ax.grid(alpha=0.28, axis="y")

            if c == 0:
                ax.text(-0.28, 1.03, dom["label"], transform=ax.transAxes, fontsize=18, fontweight="bold", ha="left", va="bottom")

    cohort_handles = [Patch(facecolor=colors[c], edgecolor=colors[c], alpha=0.40, label=labels.get(c, c.upper())) for c in cohorts_present]
    sed_handles = [
        Patch(facecolor="#BBBBBB", edgecolor="#666666", alpha=0.22, hatch="....", label="Non-sedated"),
        Patch(facecolor="#BBBBBB", edgecolor="#666666", alpha=0.42, hatch="xxxx", label="Sedated"),
    ]
    fig3.legend(
        handles=cohort_handles + sed_handles,
        loc="upper center",
        ncol=min(7, len(cohort_handles) + len(sed_handles)),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=18,
    )
    fig3.suptitle("Complexity by Scenario with Sedation Split", y=1.04, fontsize=24)
    fig3.tight_layout()
    fig3.savefig(fig / "fig03_complexity_by_sedation_facets_annotated.svg", bbox_inches="tight")
    fig3.savefig(fig / "fig03_complexity_by_sedation_facets_annotated.pdf", bbox_inches="tight")
    plt.close(fig3)

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(res / "stats_regression_sfc_occupancy_fig02_with_stars.csv", index=False)

    metrics.to_csv(res / "dual_domain_metrics_corrected_sedation_group.csv", index=False)
    states.to_csv(res / "dual_domain_state_rows_corrected_sedation_group.csv", index=False)

    print("Wrote annotated figures and stats to:", res, fig)
    print({"bold_available": bold_available, "domains": [d["name"] for d in domains]})


if __name__ == "__main__":
    main()
