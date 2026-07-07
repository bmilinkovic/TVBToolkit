#!/usr/bin/env python3
"""Noise-control analysis for empirical BOLD LZC.

Per subject:
1) compute original LZC on BOLD (time x ROI)
2) independently shuffle time within each ROI
3) recompute LZC on shuffled data

Outputs:
- subject-level original vs shuffled LZC table
- cohort summaries/statistics
- figures: original-vs-shuffled scatter and cohort boxplots
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
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))
os.environ.setdefault("TVB_USER_HOME", str((_REPO_ROOT / ".tvb-temp").resolve()))

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402
from tvbtoolkit.complexity.measures import lzc_multichannel  # noqa: E402

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


def _shuffle_time_per_roi(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    t, r = y.shape
    out = np.empty_like(y)
    for k in range(r):
        idx = rng.permutation(t)
        out[:, k] = y[idx, k]
    return out


def _holm_correct(pvals: list[float]) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    if arr.size == 0:
        return arr
    m = arr.size
    order = np.argsort(arr)
    out = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * arr[idx]
        adj = max(adj, prev)
        out[idx] = min(adj, 1.0)
        prev = out[idx]
    return out


def _cohort_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    cohorts = [c for c in COHORTS if c in set(df["cohort"].astype(str))]
    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    x = np.arange(len(cohorts), dtype=float)
    width = 0.34

    orig_data = [df.loc[df["cohort"] == c, "lzc_original"].to_numpy(dtype=float) for c in cohorts]
    shuf_data = [df.loc[df["cohort"] == c, "lzc_shuffled_mean"].to_numpy(dtype=float) for c in cohorts]
    pos_orig = x - width / 2.0
    pos_shuf = x + width / 2.0

    bp_orig = ax.boxplot(orig_data, positions=pos_orig, widths=0.26, patch_artist=True, showfliers=False)
    bp_shuf = ax.boxplot(shuf_data, positions=pos_shuf, widths=0.26, patch_artist=True, showfliers=False)
    for patch, cohort in zip(bp_orig["boxes"], cohorts):
        patch.set_facecolor(PALETTE.get(cohort, "#666666"))
        patch.set_alpha(0.72)
        patch.set_edgecolor("#222222")
    for patch, cohort in zip(bp_shuf["boxes"], cohorts):
        patch.set_facecolor(PALETTE.get(cohort, "#666666"))
        patch.set_alpha(0.30)
        patch.set_edgecolor("#222222")

    for i, cohort in enumerate(cohorts):
        o = orig_data[i]
        s = shuf_data[i]
        jo = np.linspace(-0.05, 0.05, o.size) if o.size else np.asarray([])
        js = np.linspace(-0.05, 0.05, s.size) if s.size else np.asarray([])
        if o.size:
            ax.scatter(np.full(o.size, pos_orig[i]) + jo, o, s=11, color="#111111", alpha=0.35, zorder=3)
        if s.size:
            ax.scatter(np.full(s.size, pos_shuf[i]) + js, s, s=11, color="#111111", alpha=0.28, zorder=3)

    ax.set_xticks(x, [c.upper() for c in cohorts])
    ax.set_ylabel("LZC multichannel")
    ax.set_title("Original vs time-shuffled LZC by cohort")
    ax.grid(alpha=0.25)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#555555", alpha=0.72, ec="#222222", label="original"),
        plt.Rectangle((0, 0), 1, 1, fc="#555555", alpha=0.30, ec="#222222", label="time-shuffled"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    _save_figure(fig, out_dir, "fig_lzc_original_vs_shuffled_boxplots_by_cohort")


def _scatter(df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.6, 6.1))
    for cohort in COHORTS:
        d = df[df["cohort"] == cohort]
        if d.empty:
            continue
        ax.scatter(
            d["lzc_original"].to_numpy(dtype=float),
            d["lzc_shuffled_mean"].to_numpy(dtype=float),
            s=28,
            color=PALETTE.get(cohort, "#777777"),
            alpha=0.70,
            edgecolor="none",
            label=cohort.upper(),
        )
    lo = float(np.nanmin(np.concatenate([df["lzc_original"].to_numpy(dtype=float), df["lzc_shuffled_mean"].to_numpy(dtype=float)])))
    hi = float(np.nanmax(np.concatenate([df["lzc_original"].to_numpy(dtype=float), df["lzc_shuffled_mean"].to_numpy(dtype=float)])))
    pad = 0.02 * max(1e-6, hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.2, color="black", alpha=0.7)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Original LZC")
    ax.set_ylabel("Time-shuffled LZC")
    ax.set_title("Subject-level original vs shuffled LZC")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    _save_figure(fig, out_dir, "fig_lzc_original_vs_shuffled_scatter")


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
        x = np.asarray(rec.timeseries, dtype=float)
        if not np.all(np.isfinite(x)):
            bad = ~np.isfinite(x)
            excluded.append(
                {
                    "subject_id": str(rec.subject_id),
                    "cohort": str(rec.cohort),
                    "reason": "non_finite_timeseries",
                    "n_nonfinite_values": int(np.sum(bad)),
                }
            )
            continue

        lzc_orig = float(lzc_multichannel(x))
        sh_vals = []
        for s in range(int(args.n_shuffles)):
            seed = int(args.seed) + 7919 * i + s
            rng = np.random.default_rng(seed)
            x_sh = _shuffle_time_per_roi(x, rng)
            sh_vals.append(float(lzc_multichannel(x_sh)))
        sh_arr = np.asarray(sh_vals, dtype=float)
        rows.append(
            {
                "subject_id": str(rec.subject_id),
                "cohort": str(rec.cohort),
                "stage": str(rec.stage),
                "sedation": str(rec.sedation),
                "n_timepoints": int(x.shape[0]),
                "n_regions": int(x.shape[1]),
                "n_shuffles": int(args.n_shuffles),
                "lzc_original": float(lzc_orig),
                "lzc_shuffled_mean": float(np.mean(sh_arr)),
                "lzc_shuffled_std": float(np.std(sh_arr, ddof=1)) if sh_arr.size > 1 else 0.0,
                "lzc_delta_original_minus_shuffled": float(lzc_orig - np.mean(sh_arr)),
                "lzc_ratio_original_over_shuffled": float(lzc_orig / max(1e-12, np.mean(sh_arr))),
            }
        )
        if i % 10 == 0 or i == n_total:
            print(f"Processed {i}/{n_total} subjects.")

    df = pd.DataFrame(rows).sort_values(["cohort", "subject_id"]).reset_index(drop=True)
    exc_df = pd.DataFrame(excluded)
    df.to_csv(tab_dir / "lzc_original_vs_shuffled_subjects.csv", index=False)
    exc_df.to_csv(tab_dir / "excluded_subjects.csv", index=False)

    # Group summary table.
    summary = (
        df.groupby("cohort", as_index=False)
        .agg(
            n=("subject_id", "size"),
            lzc_original_mean=("lzc_original", "mean"),
            lzc_original_std=("lzc_original", "std"),
            lzc_original_median=("lzc_original", "median"),
            lzc_shuffled_mean=("lzc_shuffled_mean", "mean"),
            lzc_shuffled_std=("lzc_shuffled_mean", "std"),
            lzc_shuffled_median=("lzc_shuffled_mean", "median"),
            delta_mean=("lzc_delta_original_minus_shuffled", "mean"),
            delta_std=("lzc_delta_original_minus_shuffled", "std"),
            delta_median=("lzc_delta_original_minus_shuffled", "median"),
        )
        .sort_values("cohort")
    )
    summary.to_csv(tab_dir / "lzc_original_vs_shuffled_group_summary.csv", index=False)

    # Paired within-subject original vs shuffled (overall + per cohort).
    paired_rows = []
    d_all = df[["lzc_original", "lzc_shuffled_mean"]].dropna()
    if not d_all.empty:
        w, p = wilcoxon(d_all["lzc_original"].to_numpy(dtype=float), d_all["lzc_shuffled_mean"].to_numpy(dtype=float), alternative="two-sided")
        paired_rows.append(
            {
                "scope": "all_subjects",
                "cohort": "all",
                "n": int(d_all.shape[0]),
                "median_delta_original_minus_shuffled": float(np.median(d_all["lzc_original"] - d_all["lzc_shuffled_mean"])),
                "wilcoxon_W": float(w),
                "p_raw": float(p),
            }
        )
    for c in COHORTS:
        dc = df[df["cohort"] == c][["lzc_original", "lzc_shuffled_mean"]].dropna()
        if dc.empty:
            continue
        w, p = wilcoxon(dc["lzc_original"].to_numpy(dtype=float), dc["lzc_shuffled_mean"].to_numpy(dtype=float), alternative="two-sided")
        paired_rows.append(
            {
                "scope": "per_cohort",
                "cohort": c,
                "n": int(dc.shape[0]),
                "median_delta_original_minus_shuffled": float(np.median(dc["lzc_original"] - dc["lzc_shuffled_mean"])),
                "wilcoxon_W": float(w),
                "p_raw": float(p),
            }
        )
    p_raw = [float(r["p_raw"]) for r in paired_rows]
    p_h = _holm_correct(p_raw)
    for r, pa in zip(paired_rows, p_h):
        r["p_holm"] = float(pa)
    pd.DataFrame(paired_rows).to_csv(tab_dir / "lzc_original_vs_shuffled_paired_stats.csv", index=False)

    # Between-cohort distribution comparisons (original and shuffled separately).
    dist_rows = []
    for cond, col in [("original", "lzc_original"), ("shuffled", "lzc_shuffled_mean")]:
        vals = [df.loc[df["cohort"] == c, col].to_numpy(dtype=float) for c in COHORTS if c in set(df["cohort"])]
        vals = [v for v in vals if v.size > 0]
        if len(vals) >= 2:
            h, p = kruskal(*vals)
            dist_rows.append({"condition": cond, "metric": col, "test": "kruskal", "H": float(h), "p_raw": float(p)})
    pd.DataFrame(dist_rows).to_csv(tab_dir / "lzc_group_distribution_omnibus.csv", index=False)

    pair_rows = []
    for cond, col in [("original", "lzc_original"), ("shuffled", "lzc_shuffled_mean")]:
        control = df.loc[df["cohort"] == "control", col].to_numpy(dtype=float)
        for cmp_c in ("emcs", "mcs", "uws"):
            cmp_v = df.loc[df["cohort"] == cmp_c, col].to_numpy(dtype=float)
            if control.size == 0 or cmp_v.size == 0:
                continue
            u, p = mannwhitneyu(control, cmp_v, alternative="two-sided")
            pair_rows.append(
                {
                    "condition": cond,
                    "contrast": f"control vs {cmp_c}",
                    "n_control": int(control.size),
                    "n_other": int(cmp_v.size),
                    "median_control": float(np.median(control)),
                    "median_other": float(np.median(cmp_v)),
                    "U": float(u),
                    "p_raw": float(p),
                }
            )
    p_raw = [float(r["p_raw"]) for r in pair_rows]
    p_h = _holm_correct(p_raw)
    for r, pa in zip(pair_rows, p_h):
        r["p_holm"] = float(pa)
    pd.DataFrame(pair_rows).to_csv(tab_dir / "lzc_group_distribution_pairwise_control_vs_doc.csv", index=False)

    _scatter(df, fig_dir)
    _cohort_boxplot(df, fig_dir)

    meta = {
        "data_root": str(data_root),
        "output_root": str(out_root),
        "n_subjects_in": int(n_total),
        "n_subjects_out": int(df.shape[0]),
        "n_excluded": int(exc_df.shape[0]),
        "n_shuffles": int(args.n_shuffles),
        "seed": int(args.seed),
        "lzc_impl": "tvbtoolkit.complexity.measures.lzc_multichannel",
        "roi_reorder_requested": str(args.roi_reorder_mode),
        "roi_reorder_applied": str(reorder_decision["applied_mode"]),
    }
    (log_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved LZC noise-control outputs to: {out_root}")
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
                "noise_control_lzc",
            )
        ),
    )
    p.add_argument("--n-shuffles", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-subjects-per-group", type=int, default=None)
    p.add_argument("--roi-reorder-mode", type=str, default="apply", choices=["auto", "apply", "none"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
