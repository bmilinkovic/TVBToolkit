#!/usr/bin/env python3
"""LZC preprocessing sensitivity + structure-association figures/tables.

Computes subject-level LZC from:
1) raw empirical BOLD
2) preprocessed BOLD (detrend + bandpass + z-score)

Also quantifies whether LZC tracks reduced temporal/spatial structure
(mean |FC| and lag-1 autocorrelation) and saves publication-ready figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import mannwhitneyu, spearmanr, zscore

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))
os.environ.setdefault("TVB_USER_HOME", str((_REPO_ROOT / ".tvb-temp").resolve()))

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402
from tvbtoolkit.complexity.measures import lzc_multichannel, lzc_single_channel  # noqa: E402

from brain_states_new_doc_bold_audited import (  # noqa: E402
    COHORTS,
    PALETTE,
    _maybe_apply_roi_reordering,
    load_new_doc_subjects,
    set_publication_style,
)


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _preprocess_bold(
    x: np.ndarray,
    tr_seconds: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    filter_order: int,
) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    y = signal.detrend(y, axis=0, type="linear")
    if bandpass_low_hz > 0.0 and bandpass_high_hz > bandpass_low_hz:
        nyq = 0.5 / float(tr_seconds)
        low = float(bandpass_low_hz) / nyq
        high = float(bandpass_high_hz) / nyq
        b, a = signal.butter(int(filter_order), [low, high], btype="band")
        padlen = min(3 * (max(len(a), len(b)) - 1), y.shape[0] - 1)
        y = signal.filtfilt(b, a, y, axis=0, padtype="odd", padlen=padlen)
    y = zscore(y, axis=0, ddof=0, nan_policy="omit")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _structure_metrics(x: np.ndarray) -> tuple[float, float]:
    y = np.asarray(x, dtype=float)
    yz = zscore(y, axis=0, ddof=0, nan_policy="omit")
    yz = np.nan_to_num(yz, nan=0.0, posinf=0.0, neginf=0.0)
    fc = np.corrcoef(yz, rowvar=False)
    iu = np.triu_indices(fc.shape[0], k=1)
    mean_abs_fc = float(np.mean(np.abs(fc[iu])))

    lag_vals = []
    for k in range(yz.shape[1]):
        s = yz[:, k]
        if s.size < 3:
            continue
        if np.std(s[:-1]) < 1e-12 or np.std(s[1:]) < 1e-12:
            continue
        lag_vals.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
    mean_lag1 = float(np.nanmean(np.asarray(lag_vals, dtype=float))) if lag_vals else float("nan")
    return mean_abs_fc, mean_lag1


def _cohort_violin(ax: Any, df: pd.DataFrame, value_col: str, title: str) -> None:
    cohorts = [c for c in COHORTS if c in set(df["cohort"].astype(str))]
    vals = []
    labels = []
    for c in cohorts:
        x = df.loc[df["cohort"] == c, value_col].to_numpy(dtype=float)
        if x.size == 0:
            continue
        vals.append(x)
        labels.append(c)
    if vals:
        vp = ax.violinplot(vals, showmeans=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            c = labels[i]
            body.set_facecolor(PALETTE.get(c, "#666666"))
            body.set_edgecolor("#222222")
            body.set_alpha(0.58)
        for i, x in enumerate(vals, start=1):
            j = np.linspace(-0.08, 0.08, x.size)
            ax.scatter(np.full(x.size, i) + j, x, s=14, color="#101010", alpha=0.45)
        ax.set_xticks(np.arange(1, len(labels) + 1), [c.upper() for c in labels])
    ax.set_ylabel("LZC multichannel")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def _plot_scatter_structure(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6), constrained_layout=True)
    specs = [
        ("mean_abs_fc_raw", "lzc_multichannel_raw", "Raw: mean |FC|", "Raw LZC"),
        ("mean_lag1_raw", "lzc_multichannel_raw", "Raw: mean lag-1 autocorr", "Raw LZC"),
        ("mean_abs_fc_preproc", "lzc_multichannel_preproc", "Preprocessed: mean |FC|", "Preprocessed LZC"),
        ("mean_lag1_preproc", "lzc_multichannel_preproc", "Preprocessed: mean lag-1 autocorr", "Preprocessed LZC"),
    ]
    rows = []
    for ax, (xc, yc, xl, yl) in zip(axes.flatten(), specs):
        for cohort in COHORTS:
            d = df[df["cohort"] == cohort]
            if d.empty:
                continue
            ax.scatter(
                d[xc].to_numpy(dtype=float),
                d[yc].to_numpy(dtype=float),
                s=24,
                color=PALETTE.get(cohort, "#777777"),
                alpha=0.65,
                edgecolor="none",
                label=cohort.upper(),
            )
        xr = df[xc].to_numpy(dtype=float)
        yr = df[yc].to_numpy(dtype=float)
        good = np.isfinite(xr) & np.isfinite(yr)
        rho, p = spearmanr(xr[good], yr[good], nan_policy="omit")
        rows.append({"x_metric": xc, "y_metric": yc, "rho_spearman": float(rho), "p_value": float(p), "n": int(np.sum(good))})
        ax.text(
            0.03,
            0.97,
            f"rho={rho:.3f}, p={p:.2e}, n={int(np.sum(good))}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("LZC tracks reduced structure (FC / autocorrelation)", y=1.03)
    _save_figure(fig, out_dir, "fig_lzc_vs_structure_raw_preproc")
    return pd.DataFrame(rows)


def _plot_violin_raw_vs_preproc(df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), constrained_layout=True)
    all_df = df.copy()
    ns_df = df[df["sedation"] == "non_sedated"].copy()
    _cohort_violin(axes[0, 0], all_df, "lzc_multichannel_raw", "All subjects: raw")
    _cohort_violin(axes[0, 1], all_df, "lzc_multichannel_preproc", "All subjects: preprocessed")
    _cohort_violin(axes[1, 0], ns_df, "lzc_multichannel_raw", "Non-sedated only: raw")
    _cohort_violin(axes[1, 1], ns_df, "lzc_multichannel_preproc", "Non-sedated only: preprocessed")
    fig.suptitle("LZC cohort distributions with and without preprocessing", y=1.02)
    _save_figure(fig, out_dir, "fig_lzc_multichannel_raw_vs_preproc_violin")


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    log_dir = out_root / "logs"
    for p in (fig_dir, tab_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    print("Loading subjects...")
    records, _ = load_new_doc_subjects(data_root, max_subjects_per_group=args.max_subjects_per_group)
    recs, reorder_qc_df, reorder_decision = _maybe_apply_roi_reordering(records, mode=args.roi_reorder_mode)
    reorder_qc_df.to_csv(tab_dir / "roi_order_qc.csv", index=False)

    rows = []
    excluded = []
    n_total = len(recs)
    for i, rec in enumerate(recs, start=1):
        x_raw = np.asarray(rec.timeseries, dtype=float)
        if not np.all(np.isfinite(x_raw)):
            bad = ~np.isfinite(x_raw)
            excluded.append(
                {
                    "subject_id": str(rec.subject_id),
                    "cohort": str(rec.cohort),
                    "reason": "non_finite_timeseries",
                    "n_nonfinite_values": int(np.sum(bad)),
                }
            )
            continue

        x_pre = _preprocess_bold(
            x_raw,
            tr_seconds=float(args.tr_seconds),
            bandpass_low_hz=float(args.bandpass_low_hz),
            bandpass_high_hz=float(args.bandpass_high_hz),
            filter_order=int(args.filter_order),
        )

        fc_raw, lag_raw = _structure_metrics(x_raw)
        fc_pre, lag_pre = _structure_metrics(x_pre)

        row = {
            "subject_id": str(rec.subject_id),
            "cohort": str(rec.cohort),
            "stage": str(rec.stage),
            "sedation": str(rec.sedation),
            "n_timepoints": int(x_raw.shape[0]),
            "n_regions": int(x_raw.shape[1]),
            "mean_abs_fc_raw": float(fc_raw),
            "mean_lag1_raw": float(lag_raw),
            "mean_abs_fc_preproc": float(fc_pre),
            "mean_lag1_preproc": float(lag_pre),
            "lzc_multichannel_raw": float(lzc_multichannel(x_raw)),
            "lzc_multichannel_preproc": float(lzc_multichannel(x_pre)),
        }
        if bool(args.compute_single_channel):
            row["lzc_single_channel_raw"] = float(lzc_single_channel(x_raw))
            row["lzc_single_channel_preproc"] = float(lzc_single_channel(x_pre))
        rows.append(row)

        if i % 10 == 0 or i == n_total:
            print(f"Processed {i}/{n_total} subjects.")

    df = pd.DataFrame(rows).sort_values(["cohort", "subject_id"]).reset_index(drop=True)
    exc_df = pd.DataFrame(excluded)
    df.to_csv(tab_dir / "lzc_subjects_raw_vs_preproc.csv", index=False)
    exc_df.to_csv(tab_dir / "excluded_subjects.csv", index=False)

    # Save preprocessed-only table in same schema as empirical_lzc_subjects.csv.
    cols_pre = ["subject_id", "cohort", "stage", "sedation", "n_timepoints", "n_regions", "lzc_multichannel_preproc"]
    if "lzc_single_channel_preproc" in df.columns:
        cols_pre.append("lzc_single_channel_preproc")
    pre = df[cols_pre].rename(
        columns={
            "lzc_multichannel_preproc": "lzc_multichannel",
            "lzc_single_channel_preproc": "lzc_single_channel",
        }
    )
    pre.to_csv(tab_dir / "empirical_lzc_subjects_preprocessed.csv", index=False)

    cols_raw = ["subject_id", "cohort", "stage", "sedation", "n_timepoints", "n_regions", "lzc_multichannel_raw"]
    if "lzc_single_channel_raw" in df.columns:
        cols_raw.append("lzc_single_channel_raw")
    raw = df[cols_raw].rename(
        columns={
            "lzc_multichannel_raw": "lzc_multichannel",
            "lzc_single_channel_raw": "lzc_single_channel",
        }
    )
    raw.to_csv(tab_dir / "empirical_lzc_subjects_raw_recomputed.csv", index=False)

    assoc = _plot_scatter_structure(df, fig_dir)
    assoc.to_csv(tab_dir / "lzc_structure_association_spearman.csv", index=False)
    _plot_violin_raw_vs_preproc(df, fig_dir)

    # Control vs pooled coma in non-sedated subset, raw vs preprocessed.
    stat_rows = []
    ns = df[df["sedation"] == "non_sedated"].copy()
    metric_pairs = [("raw", "lzc_multichannel_raw"), ("preprocessed", "lzc_multichannel_preproc")]
    if "lzc_single_channel_raw" in df.columns and "lzc_single_channel_preproc" in df.columns:
        metric_pairs.extend([("raw", "lzc_single_channel_raw"), ("preprocessed", "lzc_single_channel_preproc")])
    for mode, col in metric_pairs:
        ctrl = ns.loc[ns["cohort"] == "control", col].to_numpy(dtype=float)
        coma = ns.loc[ns["cohort"].isin(["emcs", "mcs", "uws"]), col].to_numpy(dtype=float)
        if ctrl.size and coma.size:
            _, p = mannwhitneyu(ctrl, coma, alternative="two-sided")
            stat_rows.append(
                {
                    "metric": col.replace("_raw", "").replace("_preproc", ""),
                    "mode": mode,
                    "n_control": int(ctrl.size),
                    "n_coma_non_sedated": int(coma.size),
                    "mean_control": float(np.mean(ctrl)),
                    "mean_coma_non_sedated": float(np.mean(coma)),
                    "mean_diff_coma_minus_control": float(np.mean(coma) - np.mean(ctrl)),
                    "median_control": float(np.median(ctrl)),
                    "median_coma_non_sedated": float(np.median(coma)),
                    "median_diff_coma_minus_control": float(np.median(coma) - np.median(ctrl)),
                    "p_mannwhitney": float(p),
                }
            )
    pd.DataFrame(stat_rows).to_csv(tab_dir / "lzc_control_vs_pooled_coma_non_sedated_raw_vs_preproc.csv", index=False)

    meta = {
        "data_root": str(data_root),
        "output_root": str(out_root),
        "n_subjects_in": int(n_total),
        "n_subjects_out": int(df.shape[0]),
        "n_excluded": int(exc_df.shape[0]),
        "compute_single_channel": bool(args.compute_single_channel),
        "preprocessing": {
            "detrend": True,
            "bandpass_low_hz": float(args.bandpass_low_hz),
            "bandpass_high_hz": float(args.bandpass_high_hz),
            "filter_order": int(args.filter_order),
            "zscore": True,
            "tr_seconds": float(args.tr_seconds),
        },
        "roi_reorder_requested": str(args.roi_reorder_mode),
        "roi_reorder_applied": str(reorder_decision["applied_mode"]),
    }
    (log_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved preprocessing sensitivity outputs to: {out_root}")
    print(f"Subjects analyzed: {df.shape[0]} (excluded: {exc_df.shape[0]})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(doc_liege_raw("doc_data")))
    p.add_argument(
        "--output-root",
        type=str,
        default=str(
            doc_liege_results(
                "doc_patients_new_bold_brain_states_audited",
                "empirical_markov_lzc",
                "preprocessing_sensitivity",
            )
        ),
    )
    p.add_argument("--tr-seconds", type=float, default=2.0)
    p.add_argument("--bandpass-low-hz", type=float, default=0.01)
    p.add_argument("--bandpass-high-hz", type=float, default=0.08)
    p.add_argument("--filter-order", type=int, default=3)
    p.add_argument("--compute-single-channel", action="store_true")
    p.add_argument("--max-subjects-per-group", type=int, default=None)
    p.add_argument("--roi-reorder-mode", type=str, default="apply", choices=["auto", "apply", "none"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
