#!/usr/bin/env python3
"""Sedation-stratified LZC analysis and state/cohort difference statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


COHORTS = ("control", "emcs", "mcs", "uws")
PALETTE = {
    "control": "#2E86AB",
    "emcs": "#4DAF4A",
    "mcs": "#E67E22",
    "uws": "#C0392B",
}
SEDATION_ORDER = ("non_sedated", "sedated")


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F9FC",
            "axes.edgecolor": "#2A2A2A",
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.dpi": 240,
        }
    )


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


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


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    if xa.size == 0 or ya.size == 0:
        return float("nan")
    gt = np.sum(xa[:, None] > ya[None, :])
    lt = np.sum(xa[:, None] < ya[None, :])
    return float((gt - lt) / float(xa.size * ya.size))


def _bootstrap_median_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 3000, seed: int = 7) -> tuple[float, float, float]:
    """Median(y - x) and bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    if xa.size == 0 or ya.size == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(np.median(ya) - np.median(xa))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xs = xa[rng.integers(0, xa.size, size=xa.size)]
        ys = ya[rng.integers(0, ya.size, size=ya.size)]
        boots[i] = float(np.median(ys) - np.median(xs))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def _pairwise_by_group(df: pd.DataFrame, metric: str, groups: list[str], group_col: str = "cohort") -> pd.DataFrame:
    rows = []
    p_raw = []
    idxs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = groups[i]
            b = groups[j]
            xa = df.loc[df[group_col] == a, metric].to_numpy(dtype=float)
            xb = df.loc[df[group_col] == b, metric].to_numpy(dtype=float)
            if xa.size == 0 or xb.size == 0:
                u, p = np.nan, np.nan
            else:
                u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            p_raw.append(1.0 if not np.isfinite(p) else float(p))
            idxs.append(len(rows))
            rows.append(
                {
                    "metric": metric,
                    "contrast": f"{a} vs {b}",
                    "U": float(u) if np.isfinite(u) else np.nan,
                    "p_raw": float(p) if np.isfinite(p) else np.nan,
                    "n_a": int(xa.size),
                    "n_b": int(xb.size),
                    "median_a": float(np.median(xa)) if xa.size else np.nan,
                    "median_b": float(np.median(xb)) if xb.size else np.nan,
                    "cliffs_delta": _cliffs_delta(xa, xb),
                }
            )
    p_adj = _holm_correct(p_raw)
    for irow, pa in zip(idxs, p_adj):
        rows[irow]["p_holm"] = float(pa)
    return pd.DataFrame(rows)


def _run_omnibus(df: pd.DataFrame, metric: str, groups: list[str], group_col: str = "cohort") -> dict[str, Any]:
    vals = [df.loc[df[group_col] == g, metric].to_numpy(dtype=float) for g in groups]
    vals = [v for v in vals if v.size > 0]
    if len(vals) < 2:
        return {"metric": metric, "H": np.nan, "p_kw": np.nan}
    H, p = kruskal(*vals)
    return {"metric": metric, "H": float(H), "p_kw": float(p)}


def _plot_lzc_split(df: pd.DataFrame, metric: str, ylabel: str, out_dir: Path, stem: str) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    cohorts = [c for c in COHORTS if c in set(df["cohort"].astype(str))]
    x = np.arange(len(cohorts), dtype=float)
    width = 0.34

    # Build side-by-side violin plots: one for non_sedated, one for sedated.
    for si, sed in enumerate(SEDATION_ORDER):
        offset = (-0.5 + si) * width
        vals_all = []
        pos_all = []
        cohorts_present = []
        for i, cohort in enumerate(cohorts):
            vals = df.loc[(df["cohort"] == cohort) & (df["sedation"] == sed), metric].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            xc = x[i] + offset
            vals_all.append(vals)
            pos_all.append(xc)
            cohorts_present.append(cohort)

        if vals_all:
            parts = ax.violinplot(
                vals_all,
                positions=pos_all,
                widths=0.28,
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            for body, cohort in zip(parts["bodies"], cohorts_present):
                body.set_facecolor(PALETTE.get(cohort, "#777777"))
                body.set_edgecolor(PALETTE.get(cohort, "#777777"))
                body.set_alpha(0.65 if sed == "non_sedated" else 0.42)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_colors("black")
                    parts[key].set_linewidths(0.9)

            # Overlay light points for transparency/QC.
            for vals, xc, cohort in zip(vals_all, pos_all, cohorts_present):
                j = np.linspace(-0.045, 0.045, vals.size)
                ax.scatter(
                    np.full(vals.size, xc) + j,
                    vals,
                    s=10,
                    color=PALETTE.get(cohort, "#555555"),
                    alpha=0.30,
                    edgecolor="none",
                    zorder=3,
                )

    ax.set_xticks(x, [c.upper() for c in cohorts])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: cohort split by sedation status")
    ax.grid(alpha=0.25)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="gray", ec="gray", alpha=0.65, label="non_sedated"),
        plt.Rectangle((0, 0), 1, 1, fc="gray", ec="gray", alpha=0.42, label="sedated"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    _save_figure(fig, out_dir, stem)


def _plot_sedation_delta(delta_df: pd.DataFrame, metric: str, out_dir: Path, stem: str) -> None:
    _set_style()
    dd = delta_df[delta_df["metric"] == metric].copy()
    if dd.empty:
        return
    dd = dd.sort_values("cohort")
    y = np.arange(dd.shape[0], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.8 + 0.35 * dd.shape[0]))
    for i, row in enumerate(dd.itertuples(index=False)):
        ax.hlines(y=i, xmin=row.ci_low, xmax=row.ci_high, color=PALETTE.get(row.cohort, "#444444"), lw=2.2)
        ax.scatter(row.median_diff_sed_minus_non, i, s=58, color=PALETTE.get(row.cohort, "#444444"), edgecolor="black", linewidth=0.35)
    ax.axvline(0.0, color="black", lw=1.0, linestyle="--")
    ax.set_yticks(y, [str(c).upper() for c in dd["cohort"].tolist()])
    ax.set_xlabel("Median difference (sedated - non_sedated)")
    ax.set_title(f"Sedation effect on {metric} (bootstrap 95% CI)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_dir, stem)


def run(args: argparse.Namespace) -> None:
    in_csv = Path(args.input_csv).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    log_dir = out_root / "logs"
    for p in (fig_dir, tab_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    req = {"subject_id", "cohort", "stage", "sedation", "lzc_multichannel"}
    miss = req.difference(df.columns)
    if miss:
        raise RuntimeError(f"Missing required columns in {in_csv}: {sorted(miss)}")
    df = df[df["cohort"].isin(COHORTS)].copy()
    metrics = ["lzc_multichannel"]
    if "lzc_single_channel" in df.columns:
        metrics.append("lzc_single_channel")

    # Descriptives
    agg_spec: dict[str, tuple[str, str]] = {
        "n": ("subject_id", "size"),
        "lzc_multichannel_mean": ("lzc_multichannel", "mean"),
        "lzc_multichannel_std": ("lzc_multichannel", "std"),
        "lzc_multichannel_median": ("lzc_multichannel", "median"),
    }
    if "lzc_single_channel" in metrics:
        agg_spec.update(
            {
                "lzc_single_channel_mean": ("lzc_single_channel", "mean"),
                "lzc_single_channel_std": ("lzc_single_channel", "std"),
                "lzc_single_channel_median": ("lzc_single_channel", "median"),
            }
        )
    desc = df.groupby(["cohort", "sedation"], as_index=False).agg(**agg_spec).sort_values(["cohort", "sedation"])
    desc.to_csv(tab_dir / "lzc_descriptives_by_cohort_sedation.csv", index=False)

    # Sedation effect within each cohort (where both groups exist), per metric.
    sed_rows = []
    p_collect = []
    idx_collect = []
    for metric in metrics:
        for cohort in COHORTS:
            x_non = df.loc[(df["cohort"] == cohort) & (df["sedation"] == "non_sedated"), metric].to_numpy(dtype=float)
            x_sed = df.loc[(df["cohort"] == cohort) & (df["sedation"] == "sedated"), metric].to_numpy(dtype=float)
            if x_non.size == 0 or x_sed.size == 0:
                continue
            u, p = mannwhitneyu(x_non, x_sed, alternative="two-sided")
            d = _cliffs_delta(x_non, x_sed)
            md, ci_lo, ci_hi = _bootstrap_median_diff(x_non, x_sed, n_boot=int(args.n_boot), seed=7)
            idx_collect.append(len(sed_rows))
            p_collect.append(float(p))
            sed_rows.append(
                {
                    "metric": metric,
                    "cohort": cohort,
                    "contrast": "non_sedated vs sedated",
                    "n_non_sedated": int(x_non.size),
                    "n_sedated": int(x_sed.size),
                    "U": float(u),
                    "p_raw": float(p),
                    "cliffs_delta_non_vs_sed": float(d),
                    "median_non_sedated": float(np.median(x_non)),
                    "median_sedated": float(np.median(x_sed)),
                    "median_diff_sed_minus_non": float(md),
                    "ci_low": float(ci_lo),
                    "ci_high": float(ci_hi),
                }
            )
    p_adj = _holm_correct(p_collect)
    for irow, pa in zip(idx_collect, p_adj):
        sed_rows[irow]["p_holm"] = float(pa)
    sed_df = pd.DataFrame(sed_rows).sort_values(["metric", "cohort"])
    sed_df.to_csv(tab_dir / "lzc_stats_sedated_vs_non_sedated_within_cohort.csv", index=False)

    # State/cohort differences: all subjects.
    all_omni = []
    all_pair = []
    cohorts_all = [c for c in COHORTS if c in set(df["cohort"])]
    for metric in metrics:
        all_omni.append(_run_omnibus(df, metric, groups=cohorts_all, group_col="cohort"))
        pp = _pairwise_by_group(df, metric, groups=cohorts_all, group_col="cohort")
        pp["subset"] = "all_subjects"
        all_pair.append(pp)
    pd.DataFrame(all_omni).to_csv(tab_dir / "lzc_stats_condition_omnibus_all.csv", index=False)
    pd.concat(all_pair, ignore_index=True).to_csv(tab_dir / "lzc_stats_condition_pairwise_all.csv", index=False)

    # State/cohort differences: non-sedated subset only.
    ns = df[df["sedation"] == "non_sedated"].copy()
    ns_omni = []
    ns_pair = []
    cohorts_ns = [c for c in COHORTS if c in set(ns["cohort"])]
    for metric in metrics:
        ns_omni.append(_run_omnibus(ns, metric, groups=cohorts_ns, group_col="cohort"))
        pp = _pairwise_by_group(ns, metric, groups=cohorts_ns, group_col="cohort")
        pp["subset"] = "non_sedated_only"
        ns_pair.append(pp)
    pd.DataFrame(ns_omni).to_csv(tab_dir / "lzc_stats_condition_omnibus_non_sedated.csv", index=False)
    pd.concat(ns_pair, ignore_index=True).to_csv(tab_dir / "lzc_stats_condition_pairwise_non_sedated.csv", index=False)

    # Sedated-only patient cohort differences (control absent by design).
    sed = df[df["sedation"] == "sedated"].copy()
    sed = sed[sed["cohort"].isin(("emcs", "mcs", "uws"))].copy()
    sed_omni = []
    sed_pair = []
    cohorts_sed = [c for c in ("emcs", "mcs", "uws") if c in set(sed["cohort"])]
    for metric in metrics:
        if len(cohorts_sed) >= 2:
            sed_omni.append(_run_omnibus(sed, metric, groups=cohorts_sed, group_col="cohort"))
            pp = _pairwise_by_group(sed, metric, groups=cohorts_sed, group_col="cohort")
            pp["subset"] = "sedated_patients_only"
            sed_pair.append(pp)
    pd.DataFrame(sed_omni).to_csv(tab_dir / "lzc_stats_condition_omnibus_sedated_patients.csv", index=False)
    if sed_pair:
        pd.concat(sed_pair, ignore_index=True).to_csv(tab_dir / "lzc_stats_condition_pairwise_sedated_patients.csv", index=False)

    # Figures
    _plot_lzc_split(df, "lzc_multichannel", "LZC multichannel", fig_dir, "lzc_multichannel_by_cohort_sedation")
    _plot_sedation_delta(sed_df, "lzc_multichannel", fig_dir, "lzc_multichannel_sedation_effect")
    if "lzc_single_channel" in metrics:
        _plot_lzc_split(df, "lzc_single_channel", "LZC single-channel mean", fig_dir, "lzc_single_channel_by_cohort_sedation")
        _plot_sedation_delta(sed_df, "lzc_single_channel", fig_dir, "lzc_single_channel_sedation_effect")

    meta = {
        "input_csv": str(in_csv),
        "output_root": str(out_root),
        "n_subjects": int(df.shape[0]),
        "counts_by_cohort": df["cohort"].value_counts().to_dict(),
        "metrics": metrics,
        "counts_by_cohort_sedation": (
            df.groupby(["cohort", "sedation"]).size().rename("n").reset_index().to_dict(orient="records")
        ),
        "n_boot": int(args.n_boot),
    }
    (log_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved sedation/state LZC analysis to: {out_root}")
    print("Counts by cohort:", df["cohort"].value_counts().to_dict())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/empirical_markov_lzc/tables/empirical_lzc_subjects.csv",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/empirical_markov_lzc/sedation_state_stats",
    )
    p.add_argument("--n-boot", type=int, default=3000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
