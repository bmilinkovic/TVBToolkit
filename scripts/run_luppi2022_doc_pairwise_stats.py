#!/usr/bin/env python3
"""Run paired subject-level inferential statistics for MMI/CCS comparison figures.

This script focuses on comparisons that are valid at the subject level:

1. FC/SC similarity:
   Within each cohort and redundancy function, compare subject-level
   ``RTR`` vs ``STS`` similarity values with paired sign-flip permutation tests.

2. Graph metrics:
   Recompute subject-level global efficiency and modularity from the subject
   ``STS`` and ``RTR`` matrices, then compare ``RTR`` vs ``STS`` within each
   cohort and redundancy function using paired sign-flip permutation tests.

3. Surface gradient:
   Compute parcel-wise one-sample sign-flip permutation tests of the nodal
   synergy-redundancy gradient against zero. These are saved as a table and
   summarized by counts of FDR-significant cortical parcels per panel. Because
   the surface figure already represents a difference map, these tests are
   conceptually distinct from the paired RTR-vs-STS tests above.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.analysis import PUBLICATION_COHORT_ORDER, load_phiid_index, load_phiid_matrix  # noqa: E402
from tvbtoolkit.analysis.luppi2022 import weighted_global_efficiency, weighted_modularity  # noqa: E402


COHORT_ORDER = list(PUBLICATION_COHORT_ORDER)
COHORT_DISPLAY = {
    "control": "CNTL",
    "emcs": "EMCS",
    "mcs": "MCS",
    "uws": "UWS",
    "coma": "COMA",
}
METHOD_ORDER = ["mmi", "ccs"]
METHOD_LABELS = {"mmi": "MMI", "ccs": "CCS"}
ATOM_COLORS = {"rtr": "#2E5EAA", "sts": "#D43D2A"}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12.0,
            "axes.titlesize": 21.0,
            "axes.labelsize": 18.0,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    ok = np.isfinite(arr)
    if not np.any(ok):
        return out
    vals = arr[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = ranked.size
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    back = np.empty_like(q)
    back[order] = q
    out[ok] = back
    return out


def _star(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def _perm_test_paired(a: np.ndarray, b: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        return float("nan"), float("nan")
    diff = y - x
    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size), replace=True)
    perm = (signs * diff[None, :]).mean(axis=1)
    p = (1.0 + np.sum(np.abs(perm) >= abs(obs))) / (n_perm + 1.0)
    return obs, float(p)


def _perm_test_one_sample(x: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, arr.size), replace=True)
    perm = (signs * arr[None, :]).mean(axis=1)
    p = (1.0 + np.sum(np.abs(perm) >= abs(obs))) / (n_perm + 1.0)
    return obs, float(p)


def _safe_sem(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _subject_similarity_tests(
    subject_df: pd.DataFrame,
    *,
    measure_name: str,
    rtr_col: str,
    sts_col: str,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    i = 0
    for method, sub_method in subject_df.groupby("method", sort=False):
        for cohort, sub_cohort in sub_method.groupby("cohort", sort=False):
            a = sub_cohort[rtr_col].to_numpy(dtype=float)
            b = sub_cohort[sts_col].to_numpy(dtype=float)
            diff, p = _perm_test_paired(a, b, n_perm=n_perm, seed=seed + i)
            rows.append(
                {
                    "measure": measure_name,
                    "method": method,
                    "cohort": cohort,
                    "n_subjects": int(np.sum(np.isfinite(a) & np.isfinite(b))),
                    "mean_rtr": float(np.nanmean(a)),
                    "mean_sts": float(np.nanmean(b)),
                    "mean_diff_sts_minus_rtr": diff,
                    "perm_p": p,
                }
            )
            i += 1
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = _bh_fdr(out["perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out


def _subject_graph_metrics(
    *,
    phiid_root: Path,
    manifest_path: Path,
    methods: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        index_df = load_phiid_index(phiid_root / method, manifest_path=manifest_path)
        sub = index_df.loc[index_df["atom"].isin(["sts", "rtr"])].copy()
        for _, row in sub.iterrows():
            mat = np.asarray(load_phiid_matrix(row["path"], atom=row["atom"]), dtype=float)
            rows.append(
                {
                    "method": method,
                    "subject_id": row["subject_id"],
                    "subject_stub": row["subject_stub"],
                    "cohort": row["cohort"],
                    "stage": row["stage"],
                    "sedation": row["sedation"],
                    "atom": row["atom"],
                    "global_efficiency": float(weighted_global_efficiency(mat)),
                    "modularity": float(weighted_modularity(mat)),
                }
            )
    return pd.DataFrame(rows)


def _subject_graph_tests(
    graph_df: pd.DataFrame,
    *,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["global_efficiency", "modularity"]
    i = 0
    for method, sub_method in graph_df.groupby("method", sort=False):
        for cohort, sub_cohort in sub_method.groupby("cohort", sort=False):
            piv = sub_cohort.pivot_table(index="subject_id", columns="atom", values=metrics)
            for metric in metrics:
                a = piv[(metric, "rtr")].to_numpy(dtype=float)
                b = piv[(metric, "sts")].to_numpy(dtype=float)
                diff, p = _perm_test_paired(a, b, n_perm=n_perm, seed=seed + i)
                rows.append(
                    {
                        "measure": metric,
                        "method": method,
                        "cohort": cohort,
                        "n_subjects": int(np.sum(np.isfinite(a) & np.isfinite(b))),
                        "mean_rtr": float(np.nanmean(a)),
                        "mean_sts": float(np.nanmean(b)),
                        "mean_diff_sts_minus_rtr": diff,
                        "perm_p": p,
                    }
                )
                i += 1
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = _bh_fdr(out["perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out


def _surface_gradient_roi_tests(
    gradient_df: pd.DataFrame,
    *,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    sub = gradient_df.loc[gradient_df["include_in_cortical_only"] == 1].copy()
    rows: list[dict[str, Any]] = []
    i = 0
    for (method, cohort, roi_index, roi_label), g in sub.groupby(
        ["method", "cohort", "roi_index", "roi_label"], sort=False
    ):
        vals = g["gradient_value"].to_numpy(dtype=float)
        mean_val, p = _perm_test_one_sample(vals, n_perm=n_perm, seed=seed + i)
        rows.append(
            {
                "method": method,
                "cohort": cohort,
                "roi_index": int(roi_index),
                "roi_label": str(roi_label),
                "n_subjects": int(np.isfinite(vals).sum()),
                "mean_gradient": mean_val,
                "perm_p": p,
            }
        )
        i += 1
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = _bh_fdr(out["perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out


def _add_bracket(ax: plt.Axes, x0: float, x1: float, y: float, text: str, color: str = "#1F2430") -> None:
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = 0.025 * yr
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=1.2, clip_on=False)
    ax.text((x0 + x1) / 2.0, y + h + 0.008 * yr, text, ha="center", va="bottom", fontsize=14, fontweight="bold", color=color)


def _plot_similarity_paired(
    subject_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    rtr_col: str,
    sts_col: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
) -> None:
    _set_style()
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    offsets = {"rtr": -0.18, "sts": 0.18}
    rng = np.random.default_rng(123)
    for row_idx, method in enumerate(METHOD_ORDER):
        ax = axes[row_idx]
        sub_method = subject_df.loc[subject_df["method"] == method].copy()
        yvals = []
        for cidx, cohort in enumerate(COHORT_ORDER):
            sub = sub_method.loc[sub_method["cohort"] == cohort].copy()
            for atom, col in [("rtr", rtr_col), ("sts", sts_col)]:
                vals = sub[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                x = cidx + offsets[atom]
                jitter = rng.normal(0.0, 0.05, size=vals.size)
                ax.scatter(np.full(vals.size, x) + jitter, vals, s=42, color=ATOM_COLORS[atom], alpha=0.28, edgecolors="none", zorder=2)
                mean = float(np.nanmean(vals)) if vals.size else np.nan
                sem = _safe_sem(vals)
                ax.errorbar(x, mean, yerr=sem, color="#1E1E1E", capsize=3, lw=1.2, zorder=4)
                ax.scatter(x, mean, s=64, color=ATOM_COLORS[atom], edgecolor="black", linewidth=0.4, zorder=5)
                yvals.extend(list(vals))

            hit = stats_df.loc[(stats_df["method"] == method) & (stats_df["cohort"] == cohort)]
            if not hit.empty:
                stars = _star(float(hit["perm_q_fdr"].iloc[0]))
                if stars:
                    max_y = np.nanmax(
                        np.concatenate(
                            [
                                sub[rtr_col].to_numpy(dtype=float),
                                sub[sts_col].to_numpy(dtype=float),
                            ]
                        )
                    )
                    _add_bracket(ax, cidx + offsets["rtr"], cidx + offsets["sts"], max_y + 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9), stars)

        ax.axhline(0.0, color="#BEB7A7", lw=0.9, ls="--", zorder=0)
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 0.5, METHOD_LABELS[method], rotation=90, va="center", ha="center", transform=ax.transAxes, fontsize=22, color="#1F2430")
        pad = 0.18 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9)
        ax.set_ylim(np.nanmin(yvals) - 0.08 * pad, np.nanmax(yvals) + pad)
    axes[-1].set_xticks(range(len(COHORT_ORDER)))
    axes[-1].set_xticklabels([COHORT_DISPLAY[c] for c in COHORT_ORDER])
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["rtr"], markersize=10, label="Redundancy (RTR)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["sts"], markersize=10, label="Synergy (STS)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=2, fontsize=16)
    fig.suptitle(title, fontsize=24, y=0.992)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.11, hspace=0.16)
    _save_figure(fig, out_dir, stem)


def _plot_graph_paired(
    graph_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str,
) -> None:
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.8), squeeze=False)
    offsets = {"rtr": -0.18, "sts": 0.18}
    rng = np.random.default_rng(321)
    metric_titles = {"global_efficiency": "Global efficiency", "modularity": "Modularity"}
    for row_idx, method in enumerate(METHOD_ORDER):
        sub_method = graph_df.loc[graph_df["method"] == method].copy()
        for col_idx, metric in enumerate(["global_efficiency", "modularity"]):
            ax = axes[row_idx, col_idx]
            yvals = []
            for cidx, cohort in enumerate(COHORT_ORDER):
                sub = sub_method.loc[sub_method["cohort"] == cohort].copy()
                for atom in ["rtr", "sts"]:
                    vals = sub.loc[sub["atom"] == atom, metric].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    x = cidx + offsets[atom]
                    jitter = rng.normal(0.0, 0.05, size=vals.size)
                    ax.scatter(np.full(vals.size, x) + jitter, vals, s=40, color=ATOM_COLORS[atom], alpha=0.28, edgecolors="none", zorder=2)
                    mean = float(np.nanmean(vals)) if vals.size else np.nan
                    sem = _safe_sem(vals)
                    ax.errorbar(x, mean, yerr=sem, color="#1E1E1E", capsize=3, lw=1.2, zorder=4)
                    ax.scatter(x, mean, s=64, color=ATOM_COLORS[atom], edgecolor="black", linewidth=0.4, zorder=5)
                    yvals.extend(list(vals))
                hit = stats_df.loc[(stats_df["method"] == method) & (stats_df["cohort"] == cohort) & (stats_df["measure"] == metric)]
                if not hit.empty:
                    stars = _star(float(hit["perm_q_fdr"].iloc[0]))
                    if stars:
                        max_y = np.nanmax(np.concatenate([sub.loc[sub["atom"] == "rtr", metric].to_numpy(dtype=float), sub.loc[sub["atom"] == "sts", metric].to_numpy(dtype=float)]))
                        _add_bracket(ax, cidx + offsets["rtr"], cidx + offsets["sts"], max_y + 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9), stars)
            ax.set_xticks(range(len(COHORT_ORDER)))
            if row_idx == 1:
                ax.set_xticklabels([COHORT_DISPLAY[c] for c in COHORT_ORDER])
            else:
                ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(metric_titles[metric])
            if col_idx == 0:
                ax.set_ylabel("Value")
                ax.text(-0.20, 0.5, METHOD_LABELS[method], rotation=90, va="center", ha="center", transform=ax.transAxes, fontsize=22, color="#1F2430")
            ax.grid(axis="y", color="#E8E1D3", linewidth=0.7, alpha=0.7)
            ax.set_axisbelow(True)
            pad = 0.18 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9)
            ax.set_ylim(np.nanmin(yvals) - 0.08 * pad, np.nanmax(yvals) + pad)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["rtr"], markersize=10, label="Redundancy (RTR)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["sts"], markersize=10, label="Synergy (STS)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.11, wspace=0.18, hspace=0.18)
    _save_figure(fig, out_dir, stem)


def run(args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root).expanduser().resolve()
    phiid_root = Path(args.phiid_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    fig_root = Path(args.figures_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    table_root = out_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)

    mmi_subject = pd.read_csv(results_root / "mmi" / "tables" / "subject_similarity_metrics.csv")
    ccs_subject = pd.read_csv(results_root / "ccs" / "tables" / "subject_similarity_metrics.csv")
    mmi_subject["method"] = "mmi"
    ccs_subject["method"] = "ccs"
    subject_df = pd.concat([mmi_subject, ccs_subject], ignore_index=True)

    fc_tests = _subject_similarity_tests(
        subject_df,
        measure_name="fc_similarity",
        rtr_col="fc_vs_rtr_rho",
        sts_col="fc_vs_sts_rho",
        n_perm=args.n_perm,
        seed=args.seed,
    )
    sc_tests = _subject_similarity_tests(
        subject_df,
        measure_name="sc_similarity",
        rtr_col="sc_vs_rtr_rho",
        sts_col="sc_vs_sts_rho",
        n_perm=args.n_perm,
        seed=args.seed + 1000,
    )
    fc_tests.to_csv(table_root / "fc_similarity_paired_tests.csv", index=False)
    sc_tests.to_csv(table_root / "sc_similarity_paired_tests.csv", index=False)

    graph_df = _subject_graph_metrics(phiid_root=phiid_root, manifest_path=manifest_path, methods=METHOD_ORDER)
    graph_df.to_csv(table_root / "subject_graph_metrics.csv", index=False)
    graph_tests = _subject_graph_tests(graph_df, n_perm=args.n_perm, seed=args.seed + 2000)
    graph_tests.to_csv(table_root / "graph_metric_paired_tests.csv", index=False)

    gradient_df = pd.read_csv(
        results_root / "mmi_ccs_comparison" / "gradient_stats" / "tables" / "subject_gradient_values.csv"
    )
    surface_tests = _surface_gradient_roi_tests(gradient_df, n_perm=args.n_perm, seed=args.seed + 3000)
    surface_tests.to_csv(table_root / "surface_gradient_roi_one_sample_tests.csv", index=False)
    surface_counts = (
        surface_tests.groupby(["method", "cohort"], as_index=False)["significant_fdr"].sum().rename(columns={"significant_fdr": "n_sig_cortical_rois"})
    )
    surface_counts.to_csv(table_root / "surface_gradient_significant_counts.csv", index=False)

    _plot_similarity_paired(
        subject_df,
        fc_tests,
        rtr_col="fc_vs_rtr_rho",
        sts_col="fc_vs_sts_rho",
        title="FC similarity",
        ylabel="Spearman rho",
        out_dir=fig_root / "fc_similarity",
        stem="subject_fc_similarity_mmi_vs_ccs_by_cohort_paired_stats",
    )
    _plot_similarity_paired(
        subject_df,
        sc_tests,
        rtr_col="sc_vs_rtr_rho",
        sts_col="sc_vs_sts_rho",
        title="SC similarity",
        ylabel="Spearman rho",
        out_dir=fig_root / "sc_similarity",
        stem="subject_sc_similarity_mmi_vs_ccs_by_cohort_paired_stats",
    )
    _plot_graph_paired(
        graph_df,
        graph_tests,
        out_dir=fig_root / "graph_metrics",
        stem="subject_graph_metrics_mmi_vs_ccs_paired_stats",
    )

    summary = {
        "n_perm": int(args.n_perm),
        "fc_tests": int(fc_tests.shape[0]),
        "sc_tests": int(sc_tests.shape[0]),
        "graph_tests": int(graph_tests.shape[0]),
        "surface_roi_tests": int(surface_tests.shape[0]),
    }
    (out_root / "summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=str, default="results/phiid_empirical_bold/downstream_luppi2022")
    p.add_argument("--phiid-root", type=str, default="results/phiid_empirical_bold/phiid")
    p.add_argument("--manifest", type=str, default="results/phiid_empirical_bold/inputs/manifest.csv")
    p.add_argument("--figures-root", type=str, default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/figures")
    p.add_argument("--output-root", type=str, default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/pairwise_stats")
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=11)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
