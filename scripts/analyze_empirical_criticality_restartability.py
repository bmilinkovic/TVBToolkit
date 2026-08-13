#!/usr/bin/env python3
"""Empirical BOLD criticality proxies and simulated serotonergic restartability.

This analysis uses empirical AAL90 BOLD time series to estimate subject-level
criticality proxies, then asks whether those empirical dynamics predict the
simulated serotonergic PCI dose-response.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
from scipy import signal
from scipy.stats import kruskal, mannwhitneyu, spearmanr, zscore
from statsmodels.stats.multitest import multipletests

from brain_states_new_doc_bold_audited import (  # noqa: E402
    _maybe_apply_roi_reordering,
    load_new_doc_subjects,
)


CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
COND_COLORS = {
    "COMA": "#3E4C6D",
    "UWS": "#7B5E89",
    "MCS": "#C66A3D",
    "EMCS": "#D9A441",
    "CNT": "#2F6F73",
}
TEXT = "#1F2933"
GRID = "#D9DEE7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/Volumes/ex_data/cnrs/data_doc_liege/raw/doc_data"))
    p.add_argument(
        "--source-map",
        type=Path,
        default=Path("/Volumes/ex_data/cnrs/data_doc_liege/raw/doc_data/converted_structural/source_subject_map.csv"),
    )
    p.add_argument(
        "--restartability-csv",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/restartability/tables/subject_restartability_rankings.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/outputs/empirical_criticality_restartability"),
    )
    p.add_argument("--tr-seconds", type=float, default=2.0)
    p.add_argument("--bandpass-low-hz", type=float, default=0.01)
    p.add_argument("--bandpass-high-hz", type=float, default=0.08)
    p.add_argument("--filter-order", type=int, default=3)
    p.add_argument("--event-threshold-z", type=float, default=1.0)
    p.add_argument("--roi-reorder-mode", type=str, default="apply", choices=["auto", "apply", "none", "aal90_fc"])
    p.add_argument("--max-subjects-per-group", type=int, default=None)
    return p.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": TEXT,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def preprocess_bold(
    x: np.ndarray,
    tr_seconds: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    filter_order: int,
) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    y = signal.detrend(y, axis=0, type="linear")
    if bandpass_low_hz > 0.0 and bandpass_high_hz > bandpass_low_hz:
        nyq = 0.5 / float(tr_seconds)
        low = float(bandpass_low_hz) / nyq
        high = float(bandpass_high_hz) / nyq
        b, a = signal.butter(int(filter_order), [low, high], btype="band")
        padlen = min(3 * (max(len(a), len(b)) - 1), y.shape[0] - 1)
        y = signal.filtfilt(b, a, y, axis=0, padtype="odd", padlen=padlen)
    y = zscore(y, axis=0, ddof=0, nan_policy="omit")
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def avalanches_from_events(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activity = np.asarray(events, dtype=int).sum(axis=1)
    active = activity > 0
    sizes: list[int] = []
    durations: list[int] = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            stop = i
            sizes.append(int(activity[start:stop].sum()))
            durations.append(int(stop - start))
            start = None
    if start is not None:
        sizes.append(int(activity[start:].sum()))
        durations.append(int(len(active) - start))
    return np.asarray(sizes, dtype=float), np.asarray(durations, dtype=float), activity.astype(float)


def power_law_alpha(values: np.ndarray, xmin: float = 1.0) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= xmin)]
    if x.size < 8:
        return float("nan")
    denom = float(np.sum(np.log(x / (xmin - 0.5)))) if xmin > 0.5 else float(np.sum(np.log(x / xmin)))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 + x.size / denom)


def mean_size_duration_slope(sizes: np.ndarray, durations: np.ndarray) -> tuple[float, int]:
    if sizes.size < 8 or durations.size < 8:
        return float("nan"), 0
    rows = []
    for d in np.unique(durations.astype(int)):
        mask = durations == d
        if np.count_nonzero(mask) >= 3:
            rows.append((float(d), float(np.mean(sizes[mask]))))
    if len(rows) < 3:
        return float("nan"), len(rows)
    arr = np.asarray(rows, dtype=float)
    slope = float(np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 1]), 1)[0])
    return slope, len(rows)


def lag1_and_timescale(y: np.ndarray, tr_seconds: float) -> tuple[float, float]:
    vals = []
    for k in range(y.shape[1]):
        s = y[:, k]
        if np.std(s[:-1]) < 1e-12 or np.std(s[1:]) < 1e-12:
            continue
        r = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    lag1 = float(np.mean(vals))
    clipped = float(np.clip(lag1, 1e-6, 0.999999))
    tau = float(-float(tr_seconds) / np.log(clipped))
    return lag1, tau


def dfa_alpha(x: np.ndarray) -> float:
    s = np.asarray(x, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 64:
        return float("nan")
    y = np.cumsum(s - np.mean(s))
    scales = np.asarray([4, 6, 8, 12, 16, 24, 32, 48, 64], dtype=int)
    fluct = []
    used = []
    for scale in scales:
        n = len(y) // int(scale)
        if n < 3:
            continue
        rms = []
        for i in range(n):
            seg = y[i * scale : (i + 1) * scale]
            t = np.arange(seg.size, dtype=float)
            fit = np.polyval(np.polyfit(t, seg, 1), t)
            rms.append(float(np.sqrt(np.mean((seg - fit) ** 2))))
        f = float(np.sqrt(np.mean(np.square(rms)))) if rms else float("nan")
        if np.isfinite(f) and f > 0.0:
            used.append(scale)
            fluct.append(f)
    if len(used) < 4:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(fluct), 1)[0])


def subject_criticality(y: np.ndarray, tr_seconds: float, threshold_z: float) -> dict[str, float]:
    events = y > float(threshold_z)
    sizes, durations, activity = avalanches_from_events(events)

    valid = activity[:-1] > 0
    denom = float(np.sum(activity[:-1][valid]))
    branching = float(np.sum(activity[1:][valid]) / denom) if denom > 0 else float("nan")
    branch_distance = float(abs(np.log(branching))) if np.isfinite(branching) and branching > 0 else float("nan")

    alpha_size = power_law_alpha(sizes, xmin=1.0)
    alpha_duration = power_law_alpha(durations, xmin=1.0)
    gamma_pred = (
        float((alpha_duration - 1.0) / (alpha_size - 1.0))
        if np.isfinite(alpha_size) and np.isfinite(alpha_duration) and abs(alpha_size - 1.0) > 1e-9
        else float("nan")
    )
    gamma_fit, n_duration_bins = mean_size_duration_slope(sizes, durations)
    crackling_error = float(abs(gamma_fit - gamma_pred)) if np.isfinite(gamma_fit) and np.isfinite(gamma_pred) else float("nan")
    lag1, tau = lag1_and_timescale(y, tr_seconds=tr_seconds)
    global_signal = np.mean(y, axis=1)
    dfa = dfa_alpha(global_signal)

    return {
        "event_threshold_z": float(threshold_z),
        "event_rate": float(np.mean(events)),
        "n_avalanches": int(sizes.size),
        "mean_avalanche_size": float(np.mean(sizes)) if sizes.size else float("nan"),
        "median_avalanche_size": float(np.median(sizes)) if sizes.size else float("nan"),
        "max_avalanche_size": float(np.max(sizes)) if sizes.size else float("nan"),
        "mean_avalanche_duration_bins": float(np.mean(durations)) if durations.size else float("nan"),
        "median_avalanche_duration_bins": float(np.median(durations)) if durations.size else float("nan"),
        "max_avalanche_duration_bins": float(np.max(durations)) if durations.size else float("nan"),
        "branching_ratio": branching,
        "branching_distance_log": branch_distance,
        "avalanche_size_alpha": alpha_size,
        "avalanche_duration_alpha": alpha_duration,
        "size_duration_gamma_fit": gamma_fit,
        "size_duration_gamma_predicted": gamma_pred,
        "size_duration_crackling_error": crackling_error,
        "n_duration_bins_for_gamma": int(n_duration_bins),
        "mean_lag1_autocorrelation": lag1,
        "intrinsic_timescale_seconds": tau,
        "dfa_alpha_global": dfa,
    }


def compact_condition(cohort: str) -> str:
    return "CNT" if str(cohort).lower() == "control" else str(cohort).upper()


def map_records_to_compact_ids(qc: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["cohort", "stage", "sedation", "source_sc_file", "source_subject_index"]
    q = qc.copy()
    sm = source_map.copy()
    q["source_subject_index"] = q["source_subject_index"].astype(int)
    sm["source_subject_index"] = sm["source_subject_index"].astype(int)
    merged = q.merge(
        sm[["subject_id", *key_cols]].rename(columns={"subject_id": "compact_subject_id"}),
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
    missing = merged[merged["compact_subject_id"].isna()]
    if not missing.empty:
        raise RuntimeError(f"Could not map {missing.shape[0]} empirical records to compact simulation IDs.")
    return merged


def compute_subject_table(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, qc = load_new_doc_subjects(args.data_root, max_subjects_per_group=args.max_subjects_per_group)
    records, reorder_qc, reorder_meta = _maybe_apply_roi_reordering(records, args.roi_reorder_mode)
    source_map = pd.read_csv(args.source_map)
    mapped_qc = map_records_to_compact_ids(qc, source_map)
    compact_by_empirical_id = dict(zip(mapped_qc["subject_id"], mapped_qc["compact_subject_id"], strict=True))

    rows = []
    excluded_rows = []
    for rec in records:
        compact_id = compact_by_empirical_id[rec.subject_id]
        finite_fraction = float(np.isfinite(np.asarray(rec.timeseries, dtype=float)).mean())
        if finite_fraction < 1.0:
            excluded_rows.append(
                {
                    "cohort": rec.cohort,
                    "condition": compact_condition(rec.cohort),
                    "empirical_subject_id": rec.subject_id,
                    "subject_id": compact_id,
                    "stage": rec.stage,
                    "sedation": rec.sedation,
                    "source_subject_label": rec.source_subject_label,
                    "source_subject_index": int(rec.source_subject_index),
                    "finite_fraction": finite_fraction,
                    "reason": "nonfinite_empirical_bold",
                }
            )
            continue
        y = preprocess_bold(
            rec.timeseries,
            tr_seconds=float(args.tr_seconds),
            bandpass_low_hz=float(args.bandpass_low_hz),
            bandpass_high_hz=float(args.bandpass_high_hz),
            filter_order=int(args.filter_order),
        )
        metrics = subject_criticality(y, tr_seconds=float(args.tr_seconds), threshold_z=float(args.event_threshold_z))
        rows.append(
            {
                "cohort": rec.cohort,
                "condition": compact_condition(rec.cohort),
                "empirical_subject_id": rec.subject_id,
                "subject_id": compact_id,
                "stage": rec.stage,
                "sedation": rec.sedation,
                "source_subject_label": rec.source_subject_label,
                "source_subject_index": int(rec.source_subject_index),
                "n_timepoints": int(rec.timeseries.shape[0]),
                "n_regions": int(rec.timeseries.shape[1]),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)

    restart = pd.read_csv(args.restartability_csv)
    joined = out.merge(
        restart,
        on=["cohort", "condition", "subject_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_restart"),
    )
    doc_mask = ~joined["condition"].eq("CNT")
    positive = doc_mask & joined["positive_dose_slope"].gt(0.0)
    threshold = float(joined.loc[positive, "positive_dose_slope"].quantile(0.75)) if bool(positive.any()) else float("nan")
    joined["positive_dose_response"] = positive
    joined["strong_positive_responder"] = positive & joined["positive_dose_slope"].ge(threshold)
    joined["strong_positive_slope_threshold"] = threshold
    joined["pci_response_group"] = np.where(
        joined["strong_positive_responder"],
        "Strong positive",
        np.where(joined["positive_dose_response"], "Positive", np.where(doc_mask, "PCI decrease", "Wake control")),
    )

    meta = pd.DataFrame(
        [
            {
                "roi_reorder_mode": args.roi_reorder_mode,
                "n_reorder_qc_rows": int(reorder_qc.shape[0]),
                **{f"reorder_{k}": json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in reorder_meta.items()},
            }
        ]
    )
    return joined, meta, pd.DataFrame(excluded_rows)


def finite_series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)


def group_stats(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    desc_rows = []
    omnibus_rows = []
    pair_rows = []
    for metric in metrics:
        for condition in CONDITION_ORDER:
            vals = finite_series(df[df["condition"].eq(condition)], metric)
            desc_rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "n": int(vals.size),
                    "mean": float(np.mean(vals)) if vals.size else np.nan,
                    "median": float(np.median(vals)) if vals.size else np.nan,
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                }
            )
        groups = [finite_series(df[df["condition"].eq(c)], metric) for c in CONDITION_ORDER]
        groups = [g for g in groups if g.size > 1]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
        else:
            stat, p = np.nan, np.nan
        omnibus_rows.append({"metric": metric, "test": "Kruskal-Wallis across conditions", "statistic": stat, "p": p})

        for condition in [c for c in CONDITION_ORDER if c != "CNT"]:
            a = finite_series(df[df["condition"].eq(condition)], metric)
            b = finite_series(df[df["condition"].eq("CNT")], metric)
            if a.size > 1 and b.size > 1:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            else:
                stat, p = np.nan, np.nan
            pair_rows.append({"metric": metric, "comparison": f"{condition} vs CNT", "statistic": stat, "p": p})
    omnibus = pd.DataFrame(omnibus_rows)
    if omnibus["p"].notna().any():
        omnibus.loc[omnibus["p"].notna(), "q"] = multipletests(omnibus.loc[omnibus["p"].notna(), "p"], method="fdr_bh")[1]
    pairwise = pd.DataFrame(pair_rows)
    if pairwise["p"].notna().any():
        pairwise.loc[pairwise["p"].notna(), "q"] = multipletests(pairwise.loc[pairwise["p"].notna(), "p"], method="fdr_bh")[1]
    return pd.DataFrame(desc_rows), omnibus, pairwise


def association_stats(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = ["positive_dose_slope", "max_delta_pci", "restartability_score", "baseline_pci"]
    rows = []
    lm_rows = []
    for subset_name, subset in [("all_subjects", df), ("doc_only", df[~df["condition"].eq("CNT")])]:
        for metric in metrics:
            for target in targets:
                work = subset[[metric, target, "condition"]].replace([np.inf, -np.inf], np.nan).dropna()
                if work.shape[0] < 8:
                    rho, p = np.nan, np.nan
                else:
                    rho, p = spearmanr(work[metric].to_numpy(float), work[target].to_numpy(float))
                rows.append({"subset": subset_name, "metric": metric, "target": target, "n": int(work.shape[0]), "spearman_rho": rho, "p": p})

            work = subset[[metric, "positive_dose_slope", "condition"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
            if work.shape[0] >= 20 and work["condition"].nunique() > 1:
                work["metric_z"] = zscore(work[metric].to_numpy(float), ddof=0)
                try:
                    model = smf.ols("positive_dose_slope ~ metric_z + C(condition)", data=work).fit()
                    lm_rows.append(
                        {
                            "subset": subset_name,
                            "metric": metric,
                            "n": int(work.shape[0]),
                            "model": "positive_dose_slope ~ metric_z + condition",
                            "beta_metric_z": float(model.params.get("metric_z", np.nan)),
                            "t_metric_z": float(model.tvalues.get("metric_z", np.nan)),
                            "p_metric_z": float(model.pvalues.get("metric_z", np.nan)),
                            "r2": float(model.rsquared),
                        }
                    )
                except Exception as exc:
                    lm_rows.append({"subset": subset_name, "metric": metric, "n": int(work.shape[0]), "model": "failed", "error": str(exc)})
    corr = pd.DataFrame(rows)
    if corr["p"].notna().any():
        corr.loc[corr["p"].notna(), "q"] = multipletests(corr.loc[corr["p"].notna(), "p"], method="fdr_bh")[1]
    lm = pd.DataFrame(lm_rows)
    if not lm.empty and "p_metric_z" in lm and lm["p_metric_z"].notna().any():
        lm.loc[lm["p_metric_z"].notna(), "q_metric_z"] = multipletests(lm.loc[lm["p_metric_z"].notna(), "p_metric_z"], method="fdr_bh")[1]
    return corr, lm


def jittered_positions(n: int, width: float = 0.14) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def draw_condition_distribution(ax: Any, df: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    vals_all = [finite_series(df[df["condition"].eq(c)], metric) for c in CONDITION_ORDER]
    positions_all = np.arange(len(CONDITION_ORDER), dtype=float)
    nonempty = [(pos, condition, arr) for pos, condition, arr in zip(positions_all, CONDITION_ORDER, vals_all, strict=True) if arr.size > 0]
    if not nonempty:
        ax.set_axis_off()
        return
    positions = np.asarray([x[0] for x in nonempty], dtype=float)
    vals = [x[2] for x in nonempty]
    conditions = [x[1] for x in nonempty]
    vp = ax.violinplot(vals, positions=positions, showmeans=False, showmedians=False, showextrema=False, widths=0.78)
    for body, condition in zip(vp["bodies"], conditions, strict=True):
        body.set_facecolor(COND_COLORS[condition])
        body.set_edgecolor("none")
        body.set_alpha(0.28)
    for pos, condition, arr in nonempty:
        if arr.size:
            ax.scatter(pos + jittered_positions(arr.size), arr, s=8, color=COND_COLORS[condition], alpha=0.72, linewidth=0)
            ax.plot([pos - 0.22, pos + 0.22], [np.median(arr), np.median(arr)], color=TEXT, linewidth=0.8)
    ax.set_xticks(positions_all, CONDITION_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.75)
    if metric == "branching_ratio":
        ax.axhline(1.0, color=TEXT, linewidth=0.65, linestyle="--", alpha=0.75)
        ax.text(0.02, 0.96, "critical branching", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=TEXT)


def plot_condition_summary(df: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.6), constrained_layout=True)
    specs = [
        ("branching_ratio", "Branching ratio", "Avalanche branching"),
        ("branching_distance_log", "Distance from branching ratio 1", "Branching distance"),
        ("intrinsic_timescale_seconds", "Intrinsic timescale (s)", "Slow temporal persistence"),
        ("dfa_alpha_global", "DFA exponent", "Long-range temporal structure"),
    ]
    for ax, (metric, ylabel, title) in zip(axes.flat, specs, strict=True):
        draw_condition_distribution(ax, df, metric, ylabel, title)
    save_figure(fig, out_dir, "fig1_empirical_criticality_by_condition")


def scatter_with_fit(ax: Any, df: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str, subset_label: str) -> None:
    work = df[[xcol, ycol, "condition", "pci_response_group"]].replace([np.inf, -np.inf], np.nan).dropna()
    for condition in CONDITION_ORDER:
        d = work[work["condition"].eq(condition)]
        if d.empty:
            continue
        sizes = np.where(d["pci_response_group"].eq("Strong positive"), 34, np.where(d["pci_response_group"].eq("Positive"), 25, 16))
        ax.scatter(d[xcol], d[ycol], s=sizes, color=COND_COLORS[condition], alpha=0.76, edgecolor="white", linewidth=0.35, label=condition)
    if work.shape[0] >= 5:
        x = work[xcol].to_numpy(float)
        y = work[ycol].to_numpy(float)
        fit = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, np.polyval(fit, xx), color=TEXT, linewidth=0.85)
        rho, p = spearmanr(x, y)
        ax.text(
            0.03,
            0.97,
            f"{subset_label}\nSpearman rho={rho:.2f}, p={p:.2g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            color=TEXT,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(color=GRID, linewidth=0.45, alpha=0.75)


def plot_restartability_associations(df: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    doc = df[~df["condition"].eq("CNT")]
    specs = [
        ("branching_distance_log", "Branching distance", "Near-critical branching"),
        ("intrinsic_timescale_seconds", "Intrinsic timescale (s)", "Temporal persistence"),
        ("dfa_alpha_global", "DFA exponent", "Long-range structure"),
    ]
    for ax, (xcol, xlabel, title) in zip(axes, specs, strict=True):
        scatter_with_fit(
            ax,
            doc,
            xcol=xcol,
            ycol="positive_dose_slope",
            xlabel=xlabel,
            ylabel="Simulated PCI dose-response slope",
            title=title,
            subset_label="DoC only",
        )
    handles = [Patch(color=COND_COLORS[c], label=c) for c in CONDITION_ORDER if c != "CNT"]
    axes[-1].legend(handles=handles, frameon=False, loc="best", fontsize=6.4)
    save_figure(fig, out_dir, "fig2_empirical_criticality_predicts_restartability")


def pooled_avalanche_scaling(records_df: pd.DataFrame, raw_records: list[Any], args: argparse.Namespace) -> pd.DataFrame:
    empirical_to_condition = dict(zip(records_df["empirical_subject_id"], records_df["condition"], strict=True))
    rows = []
    for rec in raw_records:
        condition = empirical_to_condition.get(rec.subject_id)
        if condition is None:
            continue
        y = preprocess_bold(
            rec.timeseries,
            tr_seconds=float(args.tr_seconds),
            bandpass_low_hz=float(args.bandpass_low_hz),
            bandpass_high_hz=float(args.bandpass_high_hz),
            filter_order=int(args.filter_order),
        )
        sizes, durations, _ = avalanches_from_events(y > float(args.event_threshold_z))
        for size, duration in zip(sizes, durations, strict=False):
            rows.append({"condition": condition, "size": float(size), "duration": float(duration)})
    pooled = pd.DataFrame(rows)
    out_rows = []
    for (condition, duration), g in pooled.groupby(["condition", "duration"], observed=True):
        if g.shape[0] >= 5:
            out_rows.append(
                {
                    "condition": condition,
                    "duration": float(duration),
                    "mean_size": float(g["size"].mean()),
                    "n_avalanches": int(g.shape[0]),
                }
            )
    return pd.DataFrame(out_rows)


def plot_avalanche_scaling(scaling: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    for condition in CONDITION_ORDER:
        d = scaling[scaling["condition"].eq(condition)].sort_values("duration")
        if d.empty:
            continue
        ax.plot(d["duration"], d["mean_size"], marker="o", markersize=2.8, linewidth=0.9, color=COND_COLORS[condition], label=condition)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avalanche duration (TR bins)")
    ax.set_ylabel("Mean avalanche size")
    ax.set_title("Empirical BOLD avalanche scaling")
    ax.grid(color=GRID, linewidth=0.45, alpha=0.75, which="both")
    ax.legend(frameon=False, fontsize=6.5)
    save_figure(fig, out_dir, "fig3_empirical_avalanche_scaling")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    records, qc = load_new_doc_subjects(args.data_root, max_subjects_per_group=args.max_subjects_per_group)
    records, _, _ = _maybe_apply_roi_reordering(records, args.roi_reorder_mode)
    subject_df, reorder_meta, excluded_df = compute_subject_table(args)
    metrics = [
        "branching_distance_log",
        "branching_ratio",
        "size_duration_crackling_error",
        "intrinsic_timescale_seconds",
        "dfa_alpha_global",
        "mean_avalanche_size",
        "event_rate",
    ]
    desc, omnibus, pairwise = group_stats(subject_df, metrics)
    corr, lm = association_stats(subject_df, metrics)
    retained_empirical_ids = set(subject_df["empirical_subject_id"].astype(str))
    retained_records = [rec for rec in records if rec.subject_id in retained_empirical_ids]
    scaling = pooled_avalanche_scaling(subject_df, retained_records, args)

    subject_df.to_csv(tables_dir / "empirical_criticality_subjects_with_restartability.csv", index=False)
    excluded_df.to_csv(tables_dir / "empirical_criticality_excluded_subjects.csv", index=False)
    desc.to_csv(tables_dir / "empirical_criticality_condition_summary.csv", index=False)
    omnibus.to_csv(tables_dir / "empirical_criticality_condition_omnibus_tests.csv", index=False)
    pairwise.to_csv(tables_dir / "empirical_criticality_condition_pairwise_vs_wake.csv", index=False)
    corr.to_csv(tables_dir / "empirical_criticality_restartability_spearman.csv", index=False)
    lm.to_csv(tables_dir / "empirical_criticality_restartability_condition_adjusted_models.csv", index=False)
    scaling.to_csv(tables_dir / "empirical_avalanche_size_duration_scaling.csv", index=False)
    reorder_meta.to_csv(tables_dir / "empirical_criticality_roi_reordering_qc.csv", index=False)

    plot_condition_summary(subject_df, figures_dir)
    plot_restartability_associations(subject_df, figures_dir)
    plot_avalanche_scaling(scaling, figures_dir)

    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_subjects": int(subject_df.shape[0]),
        "n_excluded_subjects": int(excluded_df.shape[0]),
        "n_subjects_by_condition": subject_df["condition"].value_counts().to_dict(),
        "n_positive_dose_response_doc": int(subject_df["positive_dose_response"].sum()),
        "n_strong_positive_responders": int(subject_df["strong_positive_responder"].sum()),
        "outputs": {
            "tables": str(tables_dir),
            "figures": str(figures_dir),
        },
    }
    (logs_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
