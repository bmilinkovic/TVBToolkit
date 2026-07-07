#!/usr/bin/env python3
"""Run Luppi-style downstream analyses on completed empirical BOLD PhiID outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scipy.io

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.analysis import (  # noqa: E402
    PUBLICATION_COHORT_ORDER,
    load_phiid_index,
    load_phiid_matrix,
    sanitize_subject_stub,
)
from tvbtoolkit.analysis.luppi2022 import (  # noqa: E402
    compute_fc_matrix,
    edge_rank_gradient,
    matrix_spearman_similarity,
    redundancy_synergy_rank_gradient,
    summarize_within_between,
    threshold_top_density,
    upper_triangle_values,
    weighted_global_efficiency,
    weighted_modularity,
)
from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results, project_root  # noqa: E402
from brain_states_new_doc_bold_audited import (  # noqa: E402
    _maybe_apply_roi_reordering,
    build_roi_order_reference,
    load_new_doc_subjects,
    resolve_roi_order_names,
    validate_final_roi_order_or_raise,
)


COHORT_ORDER = list(PUBLICATION_COHORT_ORDER)
COHORT_DISPLAY = {
    "control": "CNTL",
    "emcs": "EMCS",
    "mcs": "MCS",
    "uws": "UWS",
    "coma": "COMA",
}
COHORT_PALETTE = {
    "control": "#5B8A72",
    "emcs": "#E8B56D",
    "mcs": "#C5622F",
    "uws": "#8B6B8B",
    "coma": "#3B4A6B",
}
ATOM_ORDER = ["rtr", "sts"]
ATOM_LABELS = {"rtr": "Redundancy (RTR)", "sts": "Synergy (STS)"}
ATOM_COLORS = {"rtr": "#4C6A78", "sts": "#B65D4A"}
STAGE_ORDER = {"control": 0, "acute": 1, "chronic": 2}
SEDATION_ORDER = {"non_sedated": 0, "sedated": 1}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: Any, out_dir: Path, stem: str, *, transparent_copy: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    if transparent_copy:
        fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
        fig.savefig(out_dir / f"{stem}_transparent.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def _safe_sem(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _safe_modularity(matrix: np.ndarray) -> float:
    try:
        return float(weighted_modularity(matrix))
    except Exception:
        return float("nan")


def _safe_similarity(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(matrix_spearman_similarity(a, b, k=1))
    except Exception:
        return float("nan")


def _group_label(row: pd.Series, cols: list[str]) -> str:
    if cols == ["cohort"]:
        return str(row["cohort"]).upper()
    return f"{row['cohort'].upper()} | {row['stage']} | {row['sedation']}"


def _group_stem(row: pd.Series, cols: list[str]) -> str:
    parts: list[str] = []
    for col in cols:
        parts.append(f"{col}-{sanitize_subject_stub(str(row[col]))}")
    return "__".join(parts)


def _sort_group_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "cohort" in out.columns:
        out["_cohort_order"] = out["cohort"].map({c: i for i, c in enumerate(COHORT_ORDER)}).fillna(999)
    if "stage" in out.columns:
        out["_stage_order"] = out["stage"].map(STAGE_ORDER).fillna(999)
    if "sedation" in out.columns:
        out["_sed_order"] = out["sedation"].map(SEDATION_ORDER).fillna(999)
    sort_cols = [c for c in ["_cohort_order", "_stage_order", "_sed_order"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols + cols).reset_index(drop=True)
    drop_cols = [c for c in ["_cohort_order", "_stage_order", "_sed_order"] if c in out.columns]
    return out.drop(columns=drop_cols)


def _heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "luppi_metrics",
        ["#FBF7EF", "#E8D8BE", "#B7C2C8", "#6E8797", "#324B5A"],
    )


def _gradient_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "syn_red_gradient",
        ["#3E5C67", "#D9D4C7", "#B65D4A"],
    )


def _load_records(data_root: Path) -> tuple[list[Any], list[str], dict[str, Any]]:
    records, _ = load_new_doc_subjects(data_root)
    records_use, _, reorder_decision = _maybe_apply_roi_reordering(records, mode="aal90_fc")
    roi_ref = build_roi_order_reference(data_root)
    validate_final_roi_order_or_raise(roi_ref, applied_mode=str(reorder_decision["applied_mode"]))
    roi_labels, _ = resolve_roi_order_names(roi_ref, mode=str(reorder_decision["applied_mode"]))
    return records_use, list(roi_labels), reorder_decision


def _record_lookup(records: list[Any]) -> dict[str, Any]:
    return {sanitize_subject_stub(str(rec.subject_id)): rec for rec in records}


def _subject_atom_paths(index_df: pd.DataFrame) -> pd.DataFrame:
    sub = index_df.loc[index_df["atom"].isin(ATOM_ORDER), ["subject_stub", "atom", "path"]].copy()
    wide = sub.pivot(index="subject_stub", columns="atom", values="path").reset_index()
    return wide.dropna(subset=ATOM_ORDER)


def _subject_metrics(index_df: pd.DataFrame, records: list[Any]) -> pd.DataFrame:
    rec_map = _record_lookup(records)
    paths = _subject_atom_paths(index_df)
    rows: list[dict[str, Any]] = []
    for row in paths.to_dict(orient="records"):
        stub = str(row["subject_stub"])
        rec = rec_map.get(stub)
        if rec is None:
            continue
        sts = load_phiid_matrix(row["sts"], atom="sts")
        rtr = load_phiid_matrix(row["rtr"], atom="rtr")
        fc = compute_fc_matrix(rec.timeseries)
        sc = np.asarray(rec.sc_matrix, dtype=float)
        sc_density = float(np.mean(upper_triangle_values((sc > 0).astype(float), k=1)))
        sts_thr = threshold_top_density(sts, sc_density)
        rtr_thr = threshold_top_density(rtr, sc_density)
        sc_mask = upper_triangle_values(sc, k=1) > 0
        sts_vec = upper_triangle_values(sts, k=1)
        rtr_vec = upper_triangle_values(rtr, k=1)
        rows.append(
            {
                "subject_id": str(rec.subject_id),
                "subject_stub": stub,
                "cohort": str(rec.cohort),
                "stage": str(rec.stage),
                "sedation": str(rec.sedation),
                "fc_vs_sts_rho": _safe_similarity(fc, sts),
                "fc_vs_rtr_rho": _safe_similarity(fc, rtr),
                "sc_density": sc_density,
                "sc_vs_sts_rho": _safe_similarity(sc, sts_thr),
                "sc_vs_rtr_rho": _safe_similarity(sc, rtr_thr),
                "sts_connected_mean": _safe_mean(sts_vec[sc_mask]),
                "sts_unconnected_mean": _safe_mean(sts_vec[~sc_mask]),
                "rtr_connected_mean": _safe_mean(rtr_vec[sc_mask]),
                "rtr_unconnected_mean": _safe_mean(rtr_vec[~sc_mask]),
                "sts_connected_minus_unconnected": _safe_mean(sts_vec[sc_mask]) - _safe_mean(sts_vec[~sc_mask]),
                "rtr_connected_minus_unconnected": _safe_mean(rtr_vec[sc_mask]) - _safe_mean(rtr_vec[~sc_mask]),
            }
        )
    return pd.DataFrame(rows)


def _group_mean_connectomes(records: list[Any], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(
        {
            "subject_id": [str(r.subject_id) for r in records],
            "cohort": [str(r.cohort) for r in records],
            "stage": [str(r.stage) for r in records],
            "sedation": [str(r.sedation) for r in records],
        }
    )
    rec_map = {str(r.subject_id): r for r in records}
    for group_key, group in frame.groupby(group_cols, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        mats_fc = [compute_fc_matrix(rec_map[sid].timeseries) for sid in group["subject_id"]]
        mats_sc = [np.asarray(rec_map[sid].sc_matrix, dtype=float) for sid in group["subject_id"]]
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_subjects"] = int(group.shape[0])
        row["fc_matrix"] = np.mean(np.stack(mats_fc, axis=0), axis=0)
        row["sc_matrix"] = np.mean(np.stack(mats_sc, axis=0), axis=0)
        rows.append(row)
    return pd.DataFrame(rows)


def _group_pair_tables(
    avgs: pd.DataFrame,
    mean_connectomes: pd.DataFrame,
    group_cols: list[str],
    roi_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    piv = {
        atom: avgs.loc[avgs["atom"] == atom].set_index(group_cols)
        for atom in ATOM_ORDER
    }
    conn = mean_connectomes.set_index(group_cols)
    metric_rows: list[dict[str, Any]] = []
    nodal_rows: list[dict[str, Any]] = []
    within_between_rows: list[dict[str, Any]] = []
    for key in piv["sts"].index:
        sts = np.asarray(piv["sts"].loc[key, "matrix"], dtype=float)
        rtr = np.asarray(piv["rtr"].loc[key, "matrix"], dtype=float)
        fc = np.asarray(conn.loc[key, "fc_matrix"], dtype=float)
        sc = np.asarray(conn.loc[key, "sc_matrix"], dtype=float)
        grad = redundancy_synergy_rank_gradient(sts, rtr)
        edge_grad = edge_rank_gradient(sts, rtr)
        sc_density = float(np.mean(upper_triangle_values((sc > 0).astype(float), k=1)))
        sts_thr = threshold_top_density(sts, sc_density)
        rtr_thr = threshold_top_density(rtr, sc_density)
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = {col: val for col, val in zip(group_cols, key_tuple)}
        metric_rows.extend(
            [
                {
                    **base,
                    "atom": "sts",
                    "n_subjects": int(conn.loc[key, "n_subjects"]),
                    "global_efficiency": weighted_global_efficiency(sts),
                    "modularity": _safe_modularity(sts),
                    "fc_similarity_rho": _safe_similarity(fc, sts),
                    "sc_similarity_rho": _safe_similarity(sc, sts_thr),
                },
                {
                    **base,
                    "atom": "rtr",
                    "n_subjects": int(conn.loc[key, "n_subjects"]),
                    "global_efficiency": weighted_global_efficiency(rtr),
                    "modularity": _safe_modularity(rtr),
                    "fc_similarity_rho": _safe_similarity(fc, rtr),
                    "sc_similarity_rho": _safe_similarity(sc, rtr_thr),
                },
            ]
        )
        for roi_idx, value in enumerate(grad, start=1):
            nodal_rows.append(
                {
                    **base,
                    "roi_index": int(roi_idx),
                    "roi_label": str(roi_labels[roi_idx - 1]) if roi_idx - 1 < len(roi_labels) else "",
                    "gradient_value": float(value),
                }
            )
        labels = ["synergy_pref" if g > 0 else "redundancy_pref" if g < 0 else "balanced" for g in grad]
        wb = summarize_within_between(edge_grad, labels)
        within_between_rows.append(
            {
                **base,
                "within_mean": float(wb["within_mean"]),
                "between_mean": float(wb["between_mean"]),
                "within_minus_between": float(wb["within_minus_between"]),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(nodal_rows), pd.DataFrame(within_between_rows)


def _save_gradient_matrices(avgs: pd.DataFrame, roi_labels: list[str], group_cols: list[str], out_dir: Path) -> pd.DataFrame:
    piv = {
        atom: avgs.loc[avgs["atom"] == atom].set_index(group_cols)
        for atom in ATOM_ORDER
    }
    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in piv["sts"].index:
        sts = np.asarray(piv["sts"].loc[key, "matrix"], dtype=float)
        rtr = np.asarray(piv["rtr"].loc[key, "matrix"], dtype=float)
        nodal = redundancy_synergy_rank_gradient(sts, rtr)
        edge = edge_rank_gradient(sts, rtr)
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = {col: val for col, val in zip(group_cols, key_tuple)}
        row_series = pd.Series(base)
        stem = _group_stem(row_series, group_cols)
        np.save(out_dir / f"{stem}__nodal_gradient.npy", nodal)
        np.save(out_dir / f"{stem}__edge_gradient.npy", edge)
        scipy.io.savemat(
            out_dir / f"{stem}__gradients.mat",
            {
                "nodal_gradient": nodal,
                "edge_gradient": edge,
                "roi_labels": np.asarray(roi_labels, dtype=object).reshape(1, -1),
            },
            do_compression=True,
        )
        rows.append({**base, "stem": stem})
    return pd.DataFrame(rows)


def _plot_subject_similarity_by_cohort(subject_df: pd.DataFrame, value_cols: list[str], titles: list[str], out_dir: Path, stem: str) -> None:
    _set_style()
    fig, axes = plt.subplots(1, len(value_cols), figsize=(12.5, 4.2), squeeze=False)
    cohort_order = [c for c in COHORT_ORDER if c in set(subject_df["cohort"])]
    for ax, value_col, title in zip(axes[0], value_cols, titles, strict=True):
        for x_idx, cohort in enumerate(cohort_order):
            sub = subject_df.loc[subject_df["cohort"] == cohort, value_col].to_numpy(dtype=float)
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                continue
            jitter = np.linspace(-0.12, 0.12, sub.size) if sub.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(sub.size, x_idx) + jitter,
                sub,
                s=20,
                alpha=0.35,
                color=COHORT_PALETTE.get(cohort, "#666666"),
                edgecolors="none",
            )
            m = _safe_mean(sub)
            se = _safe_sem(sub)
            ax.errorbar(x_idx, m, yerr=se, color="#1E1E1E", capsize=3, lw=1.2, zorder=4)
            ax.scatter(x_idx, m, s=42, color=COHORT_PALETTE.get(cohort, "#666666"), edgecolor="black", linewidth=0.35, zorder=5)
        ax.axhline(0.0, color="#BEB7A7", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(range(len(cohort_order)))
        ax.set_xticklabels([c.upper() for c in cohort_order], rotation=0)
        ax.set_title(title)
        ax.set_ylabel("Spearman rho")
    fig.suptitle("Subject-Level Similarity Summaries", y=1.02, fontsize=11)
    _save_figure(fig, out_dir, stem)


def _plot_similarity_comparison_composite(
    mmi_subject_df: pd.DataFrame,
    ccs_subject_df: pd.DataFrame,
    *,
    value_cols: list[str],
    titles: list[str],
    out_dir: Path,
    stem: str,
    figure_title: str,
) -> None:
    _set_style()
    fig, axes = plt.subplots(2, len(value_cols), figsize=(12.8, 7.4), squeeze=False)
    method_frames = [("MMI", mmi_subject_df), ("CCS", ccs_subject_df)]
    plt.rcParams.update(
        {
            "font.size": 15.0,
            "axes.titlesize": 19.0,
            "axes.labelsize": 18.0,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
        }
    )

    for row_idx, (method_label, subject_df) in enumerate(method_frames):
        cohort_order = [c for c in COHORT_ORDER if c in set(subject_df["cohort"])]
        for col_idx, (value_col, title) in enumerate(zip(value_cols, titles, strict=True)):
            ax = axes[row_idx, col_idx]
            for x_idx, cohort in enumerate(cohort_order):
                sub = subject_df.loc[subject_df["cohort"] == cohort, value_col].to_numpy(dtype=float)
                sub = sub[np.isfinite(sub)]
                if sub.size == 0:
                    continue
                jitter = np.linspace(-0.12, 0.12, sub.size) if sub.size > 1 else np.array([0.0])
                ax.scatter(
                    np.full(sub.size, x_idx) + jitter,
                    sub,
                    s=20,
                    alpha=0.35,
                    color=COHORT_PALETTE.get(cohort, "#666666"),
                    edgecolors="none",
                )
                m = _safe_mean(sub)
                se = _safe_sem(sub)
                ax.errorbar(x_idx, m, yerr=se, color="#1E1E1E", capsize=3, lw=1.2, zorder=4)
                ax.scatter(
                    x_idx,
                    m,
                    s=42,
                    color=COHORT_PALETTE.get(cohort, "#666666"),
                    edgecolor="black",
                    linewidth=0.35,
                    zorder=5,
                )
            ax.axhline(0.0, color="#BEB7A7", lw=0.8, ls="--", zorder=0)
            ax.set_xticks(range(len(cohort_order)))
            if row_idx == len(method_frames) - 1:
                ax.set_xticklabels([COHORT_DISPLAY.get(c, c.upper()) for c in cohort_order], rotation=0)
            else:
                ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel("Spearman rho")
            else:
                ax.set_ylabel("")
            if col_idx == 0:
                ax.text(
                    -0.23,
                    0.5,
                    method_label,
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=18.0,
                    color="#1F2430",
                )
            ax.tick_params(axis="both", labelsize=16)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)
            ax.title.set_size(19)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.10, wspace=0.18, hspace=0.16)
    _save_figure(fig, out_dir, stem)


def _plot_graph_metrics_comparison_composite(
    mmi_df: pd.DataFrame,
    ccs_df: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str,
) -> None:
    _set_style()
    method_frames = [("MMI", mmi_df), ("CCS", ccs_df)]
    cohort_order = [c for c in COHORT_ORDER if c in set(mmi_df["cohort"]).union(set(ccs_df["cohort"]))]
    metric_specs = [("global_efficiency", "Global efficiency"), ("modularity", "Modularity")]
    atom_colors = {"rtr": "#2E5EAA", "sts": "#D43D2A"}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.8), squeeze=False)
    plt.rcParams.update(
        {
            "font.size": 15.0,
            "axes.titlesize": 19.0,
            "axes.labelsize": 18.0,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
        }
    )
    for row_idx, (method_label, df) in enumerate(method_frames):
        x = np.arange(len(cohort_order), dtype=float)
        width = 0.34
        for col_idx, (metric, title) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for off, atom in [(-width / 2, "rtr"), (width / 2, "sts")]:
                vals = []
                for cohort in cohort_order:
                    hit = df.loc[(df["cohort"] == cohort) & (df["atom"] == atom), metric]
                    vals.append(float(hit.iloc[0]) if not hit.empty else np.nan)
                ax.bar(
                    x + off,
                    vals,
                    width=width,
                    color=atom_colors[atom],
                    alpha=0.95,
                    edgecolor="white",
                    linewidth=0.6,
                    label=ATOM_LABELS[atom] if row_idx == 0 and col_idx == 1 else None,
                )
            ax.set_xticks(x)
            if row_idx == len(method_frames) - 1:
                ax.set_xticklabels([COHORT_DISPLAY.get(c, c.upper()) for c in cohort_order])
            else:
                ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel("Value")
                ax.text(
                    -0.24,
                    0.5,
                    method_label,
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=18.0,
                    color="#1F2430",
                )
            ax.grid(axis="y", color="#E8E1D3", linewidth=0.7, alpha=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=16)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)
            ax.title.set_size(19)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=15.0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.11, wspace=0.18, hspace=0.16)
    _save_figure(fig, out_dir, stem)


def _plot_metric_heatmap(df: pd.DataFrame, row_cols: list[str], metric_cols: list[str], out_dir: Path, stem: str, title: str) -> None:
    _set_style()
    ordered = _sort_group_df(df, row_cols)
    labels = [_group_label(row, row_cols) for _, row in ordered.iterrows()]
    arr = ordered.loc[:, metric_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.25 * len(metric_cols) + 3.8, 0.42 * max(len(labels), 3) + 1.8))
    im = ax.imshow(arr, aspect="auto", cmap=_heatmap_cmap(), origin="upper")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=35, ha="right")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Value")
    _save_figure(fig, out_dir, stem)


def _plot_graph_metrics_by_cohort(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    _set_style()
    cohort_order = [c for c in COHORT_ORDER if c in set(df["cohort"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), squeeze=False)
    metrics = ["global_efficiency", "modularity"]
    for ax, metric in zip(axes[0], metrics, strict=True):
        x = np.arange(len(cohort_order), dtype=float)
        width = 0.32
        for off, atom in [(-width / 2, "rtr"), (width / 2, "sts")]:
            vals = []
            for cohort in cohort_order:
                hit = df.loc[(df["cohort"] == cohort) & (df["atom"] == atom), metric]
                vals.append(float(hit.iloc[0]) if not hit.empty else np.nan)
            ax.bar(
                x + off,
                vals,
                width=width,
                color=ATOM_COLORS[atom],
                alpha=0.9,
                label=ATOM_LABELS[atom],
                edgecolor="none",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in cohort_order])
        ax.set_title(metric.replace("_", " ").title())
    axes[0, 0].set_ylabel("Value")
    axes[0, 1].legend(frameon=False, loc="upper right")
    fig.suptitle("Cohort-Level Graph Metrics", y=1.02, fontsize=11)
    _save_figure(fig, out_dir, stem)


def _plot_nodal_gradient_heatmap(nodal_df: pd.DataFrame, roi_labels: list[str], group_cols: list[str], out_dir: Path, stem: str, title: str) -> None:
    _set_style()
    ordered = _sort_group_df(nodal_df[group_cols].drop_duplicates(), group_cols)
    grad_rows = []
    row_labels = []
    for _, grow in ordered.iterrows():
        mask = np.ones(len(nodal_df), dtype=bool)
        for col in group_cols:
            mask &= nodal_df[col].astype(str) == str(grow[col])
        sub = nodal_df.loc[mask].sort_values("roi_index")
        grad_rows.append(sub["gradient_value"].to_numpy(dtype=float))
        row_labels.append(_group_label(grow, group_cols))
    arr = np.vstack(grad_rows)
    sort_idx = np.argsort(np.nanmean(arr, axis=0))
    arr = arr[:, sort_idx]
    roi_sorted = [roi_labels[i] for i in sort_idx]
    fig, ax = plt.subplots(figsize=(18.0, 0.45 * max(arr.shape[0], 3) + 2.8))
    vmax = np.nanmax(np.abs(arr))
    im = ax.imshow(arr, aspect="auto", cmap=_gradient_cmap(), vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    tick_idx = np.arange(0, len(roi_sorted), 5, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([roi_sorted[i] for i in tick_idx], rotation=90)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Synergy rank - Redundancy rank")
    _save_figure(fig, out_dir, stem)


def _plot_edge_gradient_grid(avgs: pd.DataFrame, group_cols: list[str], out_dir: Path, stem: str, title: str, ncols: int) -> None:
    _set_style()
    piv = {
        atom: avgs.loc[avgs["atom"] == atom].set_index(group_cols)
        for atom in ATOM_ORDER
    }
    ordered = _sort_group_df(avgs[group_cols].drop_duplicates(), group_cols)
    matrices: list[np.ndarray] = []
    labels: list[str] = []
    for _, grow in ordered.iterrows():
        key = tuple(grow[col] for col in group_cols) if len(group_cols) > 1 else grow[group_cols[0]]
        sts = np.asarray(piv["sts"].loc[key, "matrix"], dtype=float)
        rtr = np.asarray(piv["rtr"].loc[key, "matrix"], dtype=float)
        matrices.append(edge_rank_gradient(sts, rtr))
        labels.append(_group_label(grow, group_cols))
    n = len(matrices)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.9 * nrows), squeeze=False)
    vmax = max(float(np.nanmax(np.abs(m))) for m in matrices)
    for ax, mat, label in zip(axes.reshape(-1), matrices, labels, strict=False):
        mm = np.asarray(mat, dtype=float).copy()
        np.fill_diagonal(mm, np.nan)
        im = ax.imshow(mm, cmap=_gradient_cmap(), origin="lower", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(label, fontsize=8)
        tick_idx = np.arange(0, mm.shape[0], 15, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_yticks(tick_idx)
        ax.set_xticklabels([str(i + 1) for i in tick_idx], rotation=90)
        ax.set_yticklabels([str(i + 1) for i in tick_idx])
    for ax in axes.reshape(-1)[n:]:
        ax.axis("off")
    fig.suptitle(title, y=0.995, fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="Edge rank gradient")
    _save_figure(fig, out_dir, stem)


def _plot_annotation_pending(out_dir: Path, stem: str) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(9.0, 2.4))
    ax.axis("off")
    msg = (
        "Figure 2 / RSN-pair summaries are pending because\n"
        "AAL90 annotation columns `rsn_7` and `cyto_class` are still empty."
    )
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#4C4C4C")
    _save_figure(fig, out_dir, stem)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    phiid_root = Path(args.phiid_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    averages_root = Path(args.averages_root).expanduser().resolve()
    annotation_path = Path(args.annotation_csv).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    for path in [out_root, out_root / "tables", out_root / "figures", out_root / "matrices", out_root / "logs"]:
        path.mkdir(parents=True, exist_ok=True)

    index_df = load_phiid_index(phiid_root, manifest_path=manifest_path)
    records, roi_labels, reorder_decision = _load_records(data_root)

    subject_df = _subject_metrics(index_df, records)
    subject_df.to_csv(out_root / "tables" / "subject_similarity_metrics.csv", index=False)

    cohort_avgs = pd.read_pickle(averages_root / "cohort_averages.pkl")
    condition_avgs = pd.read_pickle(averages_root / "condition_averages.pkl")
    mean_cohort_conn = _group_mean_connectomes(records, ["cohort"])
    mean_condition_conn = _group_mean_connectomes(records, ["cohort", "stage", "sedation"])

    cohort_metrics, cohort_nodal, cohort_within_between = _group_pair_tables(
        cohort_avgs,
        mean_cohort_conn,
        ["cohort"],
        roi_labels,
    )
    condition_metrics, condition_nodal, condition_within_between = _group_pair_tables(
        condition_avgs,
        mean_condition_conn,
        ["cohort", "stage", "sedation"],
        roi_labels,
    )

    for df, name in [
        (cohort_metrics, "cohort_metrics.csv"),
        (condition_metrics, "condition_metrics.csv"),
        (cohort_nodal, "cohort_nodal_gradients.csv"),
        (condition_nodal, "condition_nodal_gradients.csv"),
        (cohort_within_between, "cohort_within_between_gradient.csv"),
        (condition_within_between, "condition_within_between_gradient.csv"),
    ]:
        df.to_csv(out_root / "tables" / name, index=False)

    _save_gradient_matrices(cohort_avgs, roi_labels, ["cohort"], out_root / "matrices" / "cohort_gradients")
    _save_gradient_matrices(condition_avgs, roi_labels, ["cohort", "stage", "sedation"], out_root / "matrices" / "condition_gradients")

    subject_cohort_summary = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "fc_vs_sts_mean": _safe_mean(sub["fc_vs_sts_rho"]),
                "fc_vs_rtr_mean": _safe_mean(sub["fc_vs_rtr_rho"]),
                "sc_vs_sts_mean": _safe_mean(sub["sc_vs_sts_rho"]),
                "sc_vs_rtr_mean": _safe_mean(sub["sc_vs_rtr_rho"]),
                "n_subjects": int(sub.shape[0]),
            }
            for cohort, sub in subject_df.groupby("cohort")
        ]
    )
    subject_cohort_summary.to_csv(out_root / "tables" / "subject_similarity_by_cohort.csv", index=False)

    condition_mean_similarity = pd.DataFrame(
        [
            {
                "cohort": keys[0],
                "stage": keys[1],
                "sedation": keys[2],
                "fc_vs_sts_mean": _safe_mean(sub["fc_vs_sts_rho"]),
                "fc_vs_rtr_mean": _safe_mean(sub["fc_vs_rtr_rho"]),
                "sc_vs_sts_mean": _safe_mean(sub["sc_vs_sts_rho"]),
                "sc_vs_rtr_mean": _safe_mean(sub["sc_vs_rtr_rho"]),
                "n_subjects": int(sub.shape[0]),
            }
            for keys, sub in subject_df.groupby(["cohort", "stage", "sedation"])
        ]
    )
    condition_mean_similarity.to_csv(out_root / "tables" / "subject_similarity_by_condition.csv", index=False)

    fig_root = out_root / "figures"
    _plot_subject_similarity_by_cohort(
        subject_df,
        ["fc_vs_rtr_rho", "fc_vs_sts_rho"],
        ["FC vs RTR", "FC vs STS"],
        fig_root / "fc_similarity",
        "subject_fc_similarity_by_cohort",
    )
    _plot_subject_similarity_by_cohort(
        subject_df,
        ["sc_vs_rtr_rho", "sc_vs_sts_rho"],
        ["SC vs RTR (density-matched)", "SC vs STS (density-matched)"],
        fig_root / "sc_similarity",
        "subject_sc_similarity_by_cohort",
    )
    _plot_metric_heatmap(
        _sort_group_df(condition_mean_similarity, ["cohort", "stage", "sedation"]),
        ["cohort", "stage", "sedation"],
        ["fc_vs_rtr_mean", "fc_vs_sts_mean", "sc_vs_rtr_mean", "sc_vs_sts_mean"],
        fig_root / "similarity_heatmaps",
        "condition_similarity_heatmap",
        "Condition-Level Mean FC/SC Similarity",
    )
    _plot_metric_heatmap(
        _sort_group_df(cohort_metrics, ["cohort"]),
        ["cohort"],
        ["global_efficiency", "modularity", "fc_similarity_rho", "sc_similarity_rho"],
        fig_root / "graph_metrics",
        "cohort_graph_metric_heatmap",
        "Cohort-Level PhiID Graph Metrics",
    )
    _plot_metric_heatmap(
        _sort_group_df(condition_metrics, ["cohort", "stage", "sedation"]),
        ["cohort", "stage", "sedation"],
        ["global_efficiency", "modularity", "fc_similarity_rho", "sc_similarity_rho"],
        fig_root / "graph_metrics",
        "condition_graph_metric_heatmap",
        "Condition-Level PhiID Graph Metrics",
    )
    _plot_graph_metrics_by_cohort(cohort_metrics, fig_root / "graph_metrics", "cohort_graph_metrics_bar")
    _plot_nodal_gradient_heatmap(
        cohort_nodal,
        roi_labels,
        ["cohort"],
        fig_root / "gradients",
        "cohort_nodal_gradient_heatmap",
        "Cohort Nodal Synergy-Redundancy Rank Gradient",
    )
    _plot_nodal_gradient_heatmap(
        condition_nodal,
        roi_labels,
        ["cohort", "stage", "sedation"],
        fig_root / "gradients",
        "condition_nodal_gradient_heatmap",
        "Condition Nodal Synergy-Redundancy Rank Gradient",
    )
    _plot_edge_gradient_grid(
        cohort_avgs,
        ["cohort"],
        fig_root / "gradients",
        "cohort_edge_gradient_grid",
        "Cohort Edge Synergy-Redundancy Rank Gradient",
        ncols=5,
    )
    _plot_edge_gradient_grid(
        condition_avgs,
        ["cohort", "stage", "sedation"],
        fig_root / "gradients",
        "condition_edge_gradient_grid",
        "Condition Edge Synergy-Redundancy Rank Gradient",
        ncols=4,
    )

    annotation_ready = False
    if annotation_path.exists():
        ann = pd.read_csv(annotation_path)
        rsn_filled = ann.get("rsn_7", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
        cyto_filled = ann.get("cyto_class", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
        annotation_ready = bool(rsn_filled and cyto_filled)
    if not annotation_ready:
        _plot_annotation_pending(fig_root / "annotation_pending", "annotation_required_for_figure2_rsn_summaries")

    summary = {
        "phiid_root": str(phiid_root),
        "manifest_path": str(manifest_path),
        "n_subjects_with_similarity_metrics": int(subject_df.shape[0]),
        "n_cohort_groups": int(cohort_metrics["cohort"].nunique()),
        "n_condition_groups": int(condition_metrics[["cohort", "stage", "sedation"]].drop_duplicates().shape[0]),
        "roi_reorder_mode": str(reorder_decision["applied_mode"]),
        "annotation_ready": bool(annotation_ready),
        "annotation_csv": str(annotation_path),
    }
    (out_root / "logs" / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(doc_liege_raw("doc_data")))
    p.add_argument("--phiid-root", type=str, default=str(doc_liege_results("phiid_empirical_bold", "phiid", "mmi")))
    p.add_argument("--manifest", type=str, default=str(doc_liege_results("phiid_empirical_bold", "inputs", "manifest.csv")))
    p.add_argument("--averages-root", type=str, default=str(doc_liege_results("phiid_empirical_bold", "averages", "mmi")))
    p.add_argument("--annotation-csv", type=str, default=str(project_root() / "data" / "reference" / "aal90_luppi2022_annotations_template.csv"))
    p.add_argument("--output-root", type=str, default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022", "mmi")))
    return p


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
