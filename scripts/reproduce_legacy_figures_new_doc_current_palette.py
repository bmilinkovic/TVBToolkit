#!/usr/bin/env python3
"""Reproduce legacy-style Fig1/Fig7 with the current BrainAct palette.

This is a palette-updated, coma-capable companion to
`scripts/reproduce_legacy_figures_new_doc.py`.

Outputs:
- fig1_k_vs_ipvc.(pdf/png/svg)
- fig7_kstar_distributions_and_sfc.(pdf/png/svg)
- fig7_kstar_distributions_and_sfc_pooledref.(pdf/png/svg)
- companion tables/json/npy for auditability
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans

from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402

from brain_states_new_doc_bold_audited import (
    SubjectRecord,
    _decode_subject_names,
    _extract_phase_patterns_clean,
    _full_matrix_vector,
    _load_mat_mapping,
    _maybe_apply_roi_reordering,
    _safe_pearson,
    _select_3d_numeric,
    _to_subject_roi_roi,
    _to_subject_roi_time,
    _upper_triangle_vector,
    set_publication_style,
)


CURRENT_PALETTE = {
    "control": "#5B8A72",
    "emcs": "#E8B56D",
    "mcs": "#C5622F",
    "uws": "#8B6B8B",
    "coma": "#3B4A6B",
}

BASE_COHORTS = ("control", "emcs", "mcs", "uws")
COMA_COHORTS = ("control", "emcs", "mcs", "uws", "coma")

BASE_FILE_SPECS = (
    {
        "cohort": "control",
        "stage": "control",
        "sedation": "non_sedated",
        "fc_path": "CNT_send/FC/DoC_CNT.mat",
        "fc_var": "DoC_CNT",
        "sc_path": "CNT_send/SC/CNT_SC.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "emcs",
        "stage": "acute",
        "sedation": "non_sedated",
        "fc_path": "FC_send/DoC_acute_EMCS_matched.mat",
        "fc_var": "DoC_acute_EMCS",
        "sc_path": "SC_send/norm/anon/DoC_acute_EMCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "emcs",
        "stage": "chronic",
        "sedation": "non_sedated",
        "fc_path": "FC_send/DoC_chronic_EMCS_matched.mat",
        "fc_var": "DoC_chronic_EMCS",
        "sc_path": "SC_send/norm/anon/DoC_EMCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "emcs",
        "stage": "chronic",
        "sedation": "sedated",
        "fc_path": "FC_send/DoC_chronic_sedated_EMCS_matched.mat",
        "fc_var": "DoC_chronic_sedated_EMCS",
        "sc_path": "SC_send/norm/anon/DoC_EMCS_sedated_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "stage": "acute",
        "sedation": "non_sedated",
        "fc_path": "FC_send/DoC_acute_MCS_matched.mat",
        "fc_var": "DoC_acute_MCS",
        "sc_path": "SC_send/norm/anon/DoC_acute_MCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "stage": "acute",
        "sedation": "sedated",
        "fc_path": "FC_send/DoC_acute_sedated_MCS_matched.mat",
        "fc_var": "DoC_acute_sedated_MCS",
        "sc_path": "SC_send/norm/anon/DoC_acute_sedated_MCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "stage": "chronic",
        "sedation": "non_sedated",
        "fc_path": "FC_send/DoC_chronic_MCS_matched.mat",
        "fc_var": "DoC_chronic_MCS",
        "sc_path": "SC_send/norm/anon/DoC_MCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "stage": "chronic",
        "sedation": "sedated",
        "fc_path": "FC_send/DoC_chronic_sedated_MCS_matched.mat",
        "fc_var": "DoC_chronic_sedated_MCS",
        "sc_path": "SC_send/norm/anon/DoC_MCS_sedated_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "stage": "acute",
        "sedation": "non_sedated",
        "fc_path": "FC_send/DoC_acute_UWS_matched.mat",
        "fc_var": "DoC_acute_UWS",
        "sc_path": "SC_send/norm/anon/DoC_acute_UWS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "stage": "acute",
        "sedation": "sedated",
        "fc_path": "FC_send/DoC_acute_sedated_UWS_matched.mat",
        "fc_var": "DoC_acute_sedated_UWS",
        "sc_path": "SC_send/norm/anon/DoC_acute_sedated_UWS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "stage": "chronic",
        "sedation": "non_sedated",
        "fc_path": "FC_send/DoC_chronic_UWS_matched.mat",
        "fc_var": "DoC_chronic_UWS",
        "sc_path": "SC_send/norm/anon/DoC_UWS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "stage": "chronic",
        "sedation": "sedated",
        "fc_path": "FC_send/DoC_chronic_sedated_UWS_matched.mat",
        "fc_var": "DoC_chronic_sedated_UWS",
        "sc_path": "SC_send/norm/anon/DoC_UWS_sedated_SC_matched.mat",
        "sc_var": "SC",
    },
)

OPTIONAL_COMA_SPECS = (
    {
        "cohort": "coma",
        "stage": "acute",
        "sedation": "non_sedated",
        "fc_path": "FC_send/DoC_acute_coma_matched.mat",
        "fc_var": "DoC_acute_coma",
        "sc_path": "SC_send/norm/anon/DoC_acute_COMA_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "coma",
        "stage": "acute",
        "sedation": "sedated",
        "fc_path": "FC_send/DoC_acute_sedated_COMA_matched.mat",
        "fc_var": "DoC_acute_sedated_COMA",
        "sc_path": "SC_send/norm/anon/DoC_acute_sedated_COMA_SC_matched.mat",
        "sc_var": "SC",
    },
)


def _set_axes_transparent(fig: Any) -> None:
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("none")


def _save_figure(fig: Any, out_dir: Path, stem: str, *, save_transparent_copy: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    if save_transparent_copy:
        _set_axes_transparent(fig)
        fig.savefig(out_dir / f"{stem}_transparent.pdf", bbox_inches="tight", transparent=True)
        fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
        fig.savefig(out_dir / f"{stem}_transparent.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def _inter_pattern_corr_var(squareforms: np.ndarray) -> float:
    arr = np.asarray(squareforms, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected [k,N,N], got {arr.shape}")
    flat = arr.reshape(arr.shape[0], -1)
    corr = np.corrcoef(flat)
    return float(np.var(corr))


def _state_sfc_sorted(
    centers: np.ndarray,
    sc_ref: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(centers, dtype=float)
    if mode == "upper":
        sc_vec = _upper_triangle_vector(sc_ref)
        sfc = np.asarray([_safe_pearson(row, sc_vec) for row in c], dtype=float)
    elif mode == "full":
        sc_full = _full_matrix_vector(sc_ref)
        mats = np.asarray([squareform(row) for row in c], dtype=float)
        sfc = np.asarray([_safe_pearson(_full_matrix_vector(m), sc_full) for m in mats], dtype=float)
    else:
        raise ValueError(f"Unknown SFC mode: {mode}")
    order = np.argsort(np.nan_to_num(sfc, nan=np.inf))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return c[order], sfc[order], inv


def _bh_fdr(pvals: list[float], alpha: float) -> list[tuple[float, bool]]:
    p = np.array([1.0 if (x is None or not np.isfinite(x)) else float(x) for x in pvals], dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    p_adj_rank = np.empty(m, dtype=float)
    cmin = 1.0
    for i in range(m - 1, -1, -1):
        cmin = min(cmin, ranked[i] * m / float(i + 1))
        p_adj_rank[i] = cmin
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = np.clip(p_adj_rank, 0.0, 1.0)
    sig = p_adj < float(alpha)
    return [(float(pa), bool(ss)) for pa, ss in zip(p_adj, sig)]


def _occ_rates_for_span(labels: np.ndarray, span: tuple[int, int], k: int) -> np.ndarray:
    a, b = int(span[0]), int(span[1])
    seg = np.asarray(labels[a:b], dtype=int)
    if seg.size == 0:
        return np.zeros(k, dtype=float)
    counts = np.bincount(seg, minlength=k).astype(float)
    return counts / float(seg.size)


@dataclass
class PooledData:
    x: np.ndarray
    subject_splits: list[tuple[int, int]]
    subject_ids: list[str]
    subject_cohorts: list[str]
    subject_stages: list[str]
    subject_sedation: list[str]
    subject_sc_mats: list[np.ndarray]
    condition_splits: dict[str, tuple[int, int]]
    excluded_df: pd.DataFrame


def _effective_specs(data_root: Path, include_coma: bool) -> tuple[dict[str, Any], ...]:
    specs = list(BASE_FILE_SPECS)
    if include_coma:
        for spec in OPTIONAL_COMA_SPECS:
            if (data_root / str(spec["fc_path"])).exists() and (data_root / str(spec["sc_path"])).exists():
                specs.append(spec)
    return tuple(specs)


def load_new_doc_subjects_current_palette(
    data_root: Path,
    *,
    include_coma: bool,
    max_subjects_per_group: int | None = None,
) -> tuple[list[SubjectRecord], pd.DataFrame]:
    records: list[SubjectRecord] = []
    counter_by_cohort: dict[str, int] = {}
    qc_rows: list[dict[str, Any]] = []

    for spec in _effective_specs(data_root, include_coma=include_coma):
        cohort = str(spec["cohort"])
        counter_by_cohort.setdefault(cohort, 0)
        if max_subjects_per_group is not None and counter_by_cohort[cohort] >= max_subjects_per_group:
            continue

        fc_path = data_root / str(spec["fc_path"])
        sc_path = data_root / str(spec["sc_path"])
        fc_mapping = _load_mat_mapping(fc_path)
        sc_mapping = _load_mat_mapping(sc_path)

        fc_var, fc_arr = _select_3d_numeric(fc_mapping, variable_hint=str(spec["fc_var"]))
        sc_var, sc_arr = _select_3d_numeric(sc_mapping, variable_hint=str(spec["sc_var"]))

        fc_srt = _to_subject_roi_time(fc_arr)
        sc_srr = _to_subject_roi_roi(sc_arr)

        n_subjects_fc = int(fc_srt.shape[0])
        n_subjects_sc = int(sc_srr.shape[0])
        if n_subjects_fc != n_subjects_sc:
            raise RuntimeError(
                f"FC/SC subject count mismatch in {fc_path.name} vs {sc_path.name}: "
                f"{n_subjects_fc} vs {n_subjects_sc}"
            )

        fc_subject_names = _decode_subject_names(fc_mapping.get("subj_names"))
        sc_subject_names = _decode_subject_names(sc_mapping.get("subj_names"))
        if fc_subject_names and len(fc_subject_names) != n_subjects_fc:
            fc_subject_names = []
        if sc_subject_names and len(sc_subject_names) != n_subjects_sc:
            sc_subject_names = []

        for i in range(n_subjects_fc):
            if max_subjects_per_group is not None and counter_by_cohort[cohort] >= max_subjects_per_group:
                break

            counter_by_cohort[cohort] += 1
            stage = str(spec["stage"])
            sedation = str(spec["sedation"])
            if fc_subject_names and sc_subject_names:
                if str(fc_subject_names[i]) != str(sc_subject_names[i]):
                    raise RuntimeError(
                        "FC/SC subject name mismatch before analysis: "
                        f"fc_file={fc_path.name}, sc_file={sc_path.name}, index={i}, "
                        f"fc_name={fc_subject_names[i]}, sc_name={sc_subject_names[i]}."
                    )

            source_label = fc_subject_names[i] if fc_subject_names else f"idx{i + 1:03d}"
            sid = f"{cohort}_{stage}_{sedation}_{source_label}"

            timeseries = np.asarray(fc_srt[i].T, dtype=float)
            sc_matrix = np.asarray(sc_srr[i], dtype=float)
            if timeseries.ndim != 2 or timeseries.shape[1] != 90:
                raise RuntimeError(f"Unexpected timeseries shape for {sid}: {timeseries.shape}")
            if sc_matrix.shape != (90, 90):
                raise RuntimeError(f"Unexpected SC shape for {sid}: {sc_matrix.shape}")

            records.append(
                SubjectRecord(
                    cohort=cohort,
                    subject_id=sid,
                    stage=stage,
                    sedation=sedation,
                    source_fc_file=str(fc_path.relative_to(data_root)),
                    source_sc_file=str(sc_path.relative_to(data_root)),
                    source_subject_index=i,
                    source_subject_label=source_label,
                    timeseries=timeseries,
                    sc_matrix=sc_matrix,
                )
            )

            qc_rows.append(
                {
                    "cohort": cohort,
                    "stage": stage,
                    "sedation": sedation,
                    "subject_id": sid,
                    "source_subject_index": int(i),
                    "source_subject_label": source_label,
                    "source_fc_file": str(fc_path.relative_to(data_root)),
                    "source_sc_file": str(sc_path.relative_to(data_root)),
                    "fc_var": fc_var,
                    "sc_var": sc_var,
                    "n_timepoints": int(timeseries.shape[0]),
                    "n_regions": int(timeseries.shape[1]),
                    "sc_trace": float(np.trace(sc_matrix)),
                    "sc_symmetry_l1": float(np.mean(np.abs(sc_matrix - sc_matrix.T))),
                    "has_fc_subject_names": bool(fc_subject_names),
                    "has_sc_subject_names": bool(sc_subject_names),
                    "fc_subject_name": fc_subject_names[i] if fc_subject_names else "",
                    "sc_subject_name": sc_subject_names[i] if sc_subject_names else "",
                    "subject_name_match": (
                        bool(fc_subject_names)
                        and bool(sc_subject_names)
                        and str(fc_subject_names[i]) == str(sc_subject_names[i])
                    ),
                }
            )

    subject_ids = [r.subject_id for r in records]
    dup_ids = pd.Series(subject_ids).duplicated().sum()
    if dup_ids > 0:
        raise RuntimeError(f"Detected {dup_ids} duplicated subject_id values; subject identifiers must be unique.")
    return records, pd.DataFrame(qc_rows)


def _cohorts(include_coma: bool) -> tuple[str, ...]:
    return COMA_COHORTS if include_coma else BASE_COHORTS


def _build_pooled_patterns(args: argparse.Namespace, out_dir: Path) -> PooledData:
    data_root = Path(args.data_root)
    cohorts = _cohorts(include_coma=bool(args.include_coma))
    records, _ = load_new_doc_subjects_current_palette(
        data_root,
        include_coma=bool(args.include_coma),
        max_subjects_per_group=args.max_subjects_per_group,
    )
    recs, reorder_qc, reorder_decision = _maybe_apply_roi_reordering(records, mode=args.roi_reorder_mode)
    reorder_qc.to_csv(out_dir / "tables" / "roi_order_qc_coupling_checks.csv", index=False)
    (out_dir / "logs" / "roi_reorder_decision.json").write_text(json.dumps(reorder_decision, indent=2))

    x_blocks: list[np.ndarray] = []
    subject_splits: list[tuple[int, int]] = []
    subject_ids: list[str] = []
    subject_cohorts: list[str] = []
    subject_stages: list[str] = []
    subject_sedation: list[str] = []
    subject_sc_mats: list[np.ndarray] = []
    excluded_rows: list[dict[str, Any]] = []

    offset = 0
    for i, rec in enumerate(recs, start=1):
        bad = ~np.isfinite(np.asarray(rec.timeseries, dtype=float))
        n_bad = int(np.sum(bad))
        if n_bad > 0:
            excluded_rows.append(
                {
                    "subject_id": rec.subject_id,
                    "cohort": rec.cohort,
                    "stage": rec.stage,
                    "sedation": rec.sedation,
                    "n_nonfinite_values": n_bad,
                    "n_bad_timepoints": int(np.sum(np.any(bad, axis=1))),
                    "n_bad_rois": int(np.sum(np.any(bad, axis=0))),
                    "reason": "excluded_nonfinite_bold",
                }
            )
            continue

        patt, _, n_raw, n_kept = _extract_phase_patterns_clean(rec.timeseries, args)
        if patt.size == 0:
            excluded_rows.append(
                {
                    "subject_id": rec.subject_id,
                    "cohort": rec.cohort,
                    "stage": rec.stage,
                    "sedation": rec.sedation,
                    "n_nonfinite_values": 0,
                    "n_bad_timepoints": 0,
                    "n_bad_rois": 0,
                    "reason": f"no_phase_patterns_after_qc_raw={n_raw}_kept={n_kept}",
                }
            )
            continue

        start = offset
        end = offset + int(patt.shape[0])
        offset = end
        subject_splits.append((start, end))
        subject_ids.append(str(rec.subject_id))
        subject_cohorts.append(str(rec.cohort))
        subject_stages.append(str(rec.stage))
        subject_sedation.append(str(rec.sedation))
        subject_sc_mats.append(np.asarray(rec.sc_matrix, dtype=float))
        x_blocks.append(np.asarray(patt, dtype=np.float32))

        if args.progress_every > 0 and (i % args.progress_every == 0):
            print(f"[load {i:03d}/{len(recs):03d}] {rec.subject_id} -> rows={patt.shape[0]}")

    if not x_blocks:
        raise RuntimeError("No valid pooled phase-pattern rows available.")

    x = np.vstack(x_blocks).astype(np.float32, copy=False)
    excluded_df = pd.DataFrame(excluded_rows)

    condition_splits: dict[str, tuple[int, int]] = {}
    for cohort in cohorts:
        idx = [ii for ii, c in enumerate(subject_cohorts) if c == cohort]
        if not idx:
            continue
        a = subject_splits[idx[0]][0]
        b = subject_splits[idx[-1]][1]
        condition_splits[cohort] = (int(a), int(b))

    return PooledData(
        x=x,
        subject_splits=subject_splits,
        subject_ids=subject_ids,
        subject_cohorts=subject_cohorts,
        subject_stages=subject_stages,
        subject_sedation=subject_sedation,
        subject_sc_mats=subject_sc_mats,
        condition_splits=condition_splits,
        excluded_df=excluded_df,
    )


def _build_sc_reference(args: argparse.Namespace, pooled: PooledData) -> np.ndarray:
    mats_all = [np.asarray(m, dtype=float) for m in pooled.subject_sc_mats]
    if not mats_all:
        raise RuntimeError("Cannot build SC reference: no SC matrices found for retained subjects.")
    mats_ctrl = [m for m, c in zip(mats_all, pooled.subject_cohorts) if str(c) == "control"]
    if args.pooled_sc_reference == "control" and mats_ctrl:
        ref = np.mean(np.stack(mats_ctrl, axis=0), axis=0)
    else:
        ref = np.mean(np.stack(mats_all, axis=0), axis=0)
    ref = 0.5 * (ref + ref.T)
    np.fill_diagonal(ref, 0.0)
    return np.asarray(ref, dtype=float)


def _subject_specific_sfc_by_state(centers_sorted: np.ndarray, pooled: PooledData, mode: str) -> np.ndarray:
    centers = np.asarray(centers_sorted, dtype=float)
    s = len(pooled.subject_ids)
    k = int(centers.shape[0])
    out = np.full((s, k), np.nan, dtype=float)

    if mode == "upper":
        for si, sc in enumerate(pooled.subject_sc_mats):
            sc_vec = _upper_triangle_vector(np.asarray(sc, dtype=float))
            out[si, :] = np.asarray([_safe_pearson(centers[ki], sc_vec) for ki in range(k)], dtype=float)
        return out

    if mode == "full":
        mats = np.asarray([squareform(row) for row in centers], dtype=float)
        for si, sc in enumerate(pooled.subject_sc_mats):
            sc_full = _full_matrix_vector(np.asarray(sc, dtype=float))
            out[si, :] = np.asarray([_safe_pearson(_full_matrix_vector(mats[ki]), sc_full) for ki in range(k)], dtype=float)
        return out

    raise ValueError(f"Unknown SFC mode: {mode}")


def _load_cached_pooled_data(args: argparse.Namespace, out_root: Path) -> PooledData:
    table_dir = out_root / "tables"
    splits_path = table_dir / "subject_splits.csv"
    if not splits_path.exists():
        raise RuntimeError(f"--reuse-existing requested but missing: {splits_path}")

    df = pd.read_csv(splits_path)
    need_cols = {"subject_id", "cohort", "stage", "sedation", "start", "end"}
    if not need_cols.issubset(set(df.columns)):
        raise RuntimeError(f"subject_splits.csv missing required columns: {sorted(need_cols)}")
    df = df.sort_values("start").reset_index(drop=True)

    records, _ = load_new_doc_subjects_current_palette(
        Path(args.data_root),
        include_coma=bool(args.include_coma),
        max_subjects_per_group=args.max_subjects_per_group,
    )
    recs, _, _ = _maybe_apply_roi_reordering(records, mode=args.roi_reorder_mode)
    sc_map = {str(r.subject_id): np.asarray(r.sc_matrix, dtype=float) for r in recs}

    subject_ids = df["subject_id"].astype(str).tolist()
    missing = [sid for sid in subject_ids if sid not in sc_map]
    if missing:
        raise RuntimeError(f"Missing SC matrices for cached subject IDs (first 5): {missing[:5]}")

    subject_splits = [(int(a), int(b)) for a, b in zip(df["start"].to_numpy(), df["end"].to_numpy())]
    subject_cohorts = df["cohort"].astype(str).tolist()
    subject_stages = df["stage"].astype(str).tolist()
    subject_sedation = df["sedation"].astype(str).tolist()
    subject_sc_mats = [sc_map[sid] for sid in subject_ids]

    condition_splits: dict[str, tuple[int, int]] = {}
    for cohort in _cohorts(include_coma=bool(args.include_coma)):
        idx = [ii for ii, c in enumerate(subject_cohorts) if c == cohort]
        if idx:
            condition_splits[cohort] = (subject_splits[idx[0]][0], subject_splits[idx[-1]][1])

    excluded_path = table_dir / "excluded_subjects_nonfinite_bold.csv"
    excluded_df = pd.read_csv(excluded_path) if excluded_path.exists() else pd.DataFrame()

    return PooledData(
        x=np.empty((0, 0), dtype=np.float32),
        subject_splits=subject_splits,
        subject_ids=subject_ids,
        subject_cohorts=subject_cohorts,
        subject_stages=subject_stages,
        subject_sedation=subject_sedation,
        subject_sc_mats=subject_sc_mats,
        condition_splits=condition_splits,
        excluded_df=excluded_df,
    )


def _plot_fig1_ipvc(
    k_vals: np.ndarray,
    ipvc_vals: np.ndarray,
    k_star: int,
    fig_dir: Path,
    *,
    save_transparent_copy: bool,
) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax.plot(
        k_vals,
        ipvc_vals,
        marker="o",
        linewidth=3.0,
        markersize=8.5,
        color="#6a4c93",
        alpha=0.78,
        markerfacecolor="#6a4c93",
        markeredgecolor="#6a4c93",
        markeredgewidth=1.2,
        zorder=2,
    )
    ax.axvline(int(k_star), linestyle="--", color="#d7191c", linewidth=2.0, alpha=0.9, zorder=1)
    ax.set_xlabel("Distinct patterns (k)")
    ax.set_ylabel("IPVC")
    ax.set_title("Pooled phase-pattern complexity across k")
    fig.tight_layout()
    _save_figure(fig, fig_dir, "fig1_k_vs_ipvc", save_transparent_copy=save_transparent_copy)


def _sig_stars(p_adj: float, alpha: float = 0.05) -> str:
    if (not np.isfinite(p_adj)) or p_adj >= alpha:
        return ""
    if p_adj < 1e-4:
        return "****"
    if p_adj < 1e-3:
        return "***"
    if p_adj < 1e-2:
        return "**"
    return "*"


def _plot_fig7_for_k(
    *,
    labels: np.ndarray,
    x_per_subject_state: np.ndarray,
    display_state_index: np.ndarray,
    pooled: PooledData,
    cohorts: tuple[str, ...],
    k: int,
    alpha_fdr: float,
    fig_dir: Path,
    table_dir: Path,
    stem: str,
    pvalues_filename: str,
    points_filename: str,
    x_label: str,
    save_transparent_copy: bool,
) -> None:
    set_publication_style()
    cohorts_list = list(cohorts)
    try:
        ctrl_idx = int(cohorts_list.index("control"))
    except ValueError as exc:
        raise RuntimeError("Expected 'control' cohort in cohorts for control-only comparisons.") from exc
    target_idx = [i for i in range(len(cohorts_list)) if i != ctrl_idx]
    pair_idx = [(ctrl_idx, j) for j in target_idx]
    pair_lbl = [f"{cohorts_list[ctrl_idx]} vs {cohorts_list[j]}" for j in target_idx]

    subj_rates = [_occ_rates_for_span(labels, span, k) for span in pooled.subject_splits]
    disp_idx = np.asarray(display_state_index, dtype=int)
    if disp_idx.shape != (len(subj_rates), int(k)):
        raise ValueError(f"display_state_index must be [subjects,k], got {disp_idx.shape}")

    per_centroid_data: list[dict[str, list[float]]] = []
    for c in range(k):
        by_cond = {cond: [] for cond in cohorts_list}
        for si, (rv, cond) in enumerate(zip(subj_rates, pooled.subject_cohorts)):
            st = int(disp_idx[si, c])
            by_cond[str(cond)].append(float(rv[st]))
        per_centroid_data.append(by_cond)

    ncols = int(k + 1)
    fig = plt.figure(figsize=(3.7 * ncols, 5.3), dpi=300)
    gs = fig.add_gridspec(1, ncols, wspace=0.28)

    rng = np.random.default_rng(42)
    colors = [CURRENT_PALETTE[c] for c in cohorts_list]
    x_positions = np.arange(len(cohorts_list), dtype=float)
    p_rows: list[dict[str, Any]] = []

    for c in range(k):
        ax = fig.add_subplot(gs[0, c])
        arrays = [np.asarray(per_centroid_data[c][cond], dtype=float) for cond in cohorts_list]

        vp = ax.violinplot(
            arrays,
            positions=x_positions,
            widths=0.62,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body, clr in zip(vp["bodies"], colors):
            body.set_facecolor(clr)
            body.set_alpha(0.50)
            body.set_edgecolor(clr)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in vp:
                vp[key].set_colors("black")
                vp[key].set_linewidths(1.0)

        for xi, arr, clr in zip(x_positions, arrays, colors):
            if arr.size == 0:
                continue
            jx = xi + rng.uniform(-0.085, 0.085, size=arr.size)
            ax.scatter(jx, arr, s=18, color=clr, alpha=0.72, edgecolor="none", zorder=3)

        raw_p: list[float] = []
        for i, j in pair_idx:
            if arrays[i].size == 0 or arrays[j].size == 0:
                raw_p.append(np.nan)
                continue
            _, p = ttest_ind(arrays[i], arrays[j], equal_var=False, nan_policy="omit")
            raw_p.append(float(p))
        adj = _bh_fdr(raw_p, alpha=alpha_fdr)
        for lbl, pr, (pa, sig) in zip(pair_lbl, raw_p, adj):
            p_rows.append(
                {
                    "state": int(c) + 1,
                    "comparison": lbl,
                    "p_raw": float(pr) if np.isfinite(pr) else np.nan,
                    "p_adj_bh": float(pa),
                    "significant_0.05": bool(sig),
                }
            )
        ymax = float(max([arr.max() if arr.size else 0.0 for arr in arrays] + [0.0]))
        star_base = max(1.08, min(1.12, ymax + 0.08))
        star_step = 0.035
        for rank, ((i, j), (pa, _)) in enumerate(zip(pair_idx, adj)):
            stars = _sig_stars(pa, alpha=alpha_fdr)
            if stars:
                y_star = min(1.24, star_base + rank * star_step)
                ax.text(
                    x_positions[j],
                    y_star,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="bold",
                    color="black",
                )

        ax.set_xticks(x_positions, [c.upper() for c in cohorts_list], rotation=25, ha="right")
        ax.set_ylim(0.0, 1.24)
        ax.set_title(f"Pattern {c + 1}")
        ax.set_ylabel("Occupancy rate" if c == 0 else "")
        ax.grid(alpha=0.20)

    ax = fig.add_subplot(gs[0, k])
    x_all: list[float] = []
    y_all: list[float] = []
    point_rows: list[dict[str, Any]] = []

    for cohort, clr in zip(cohorts_list, colors):
        xs: list[float] = []
        ys: list[float] = []
        subj_idx = [ii for ii, cc in enumerate(pooled.subject_cohorts) if str(cc) == cohort]
        for si in subj_idx:
            sid = str(pooled.subject_ids[si])
            stage = str(pooled.subject_stages[si])
            sedation = str(pooled.subject_sedation[si])
            rv = np.asarray(subj_rates[si], dtype=float)
            xv = np.asarray(x_per_subject_state[si], dtype=float)
            for c in range(k):
                st = int(disp_idx[si, c])
                if not np.isfinite(xv[st]) or not np.isfinite(rv[st]):
                    continue
                xs.append(float(xv[st]))
                ys.append(float(rv[st]))
                point_rows.append(
                    {
                        "subject_id": sid,
                        "cohort": cohort,
                        "stage": stage,
                        "sedation": sedation,
                        "display_pattern": int(c) + 1,
                        "underlying_state": int(st) + 1,
                        "sfc": float(xv[st]),
                        "occupancy": float(rv[st]),
                    }
                )
        if not xs:
            continue
        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        x_all.extend(x_arr.tolist())
        y_all.extend(y_arr.tolist())

        ax.scatter(x_arr, y_arr, color=clr, alpha=0.42, s=18, label=cohort.upper(), edgecolors="none")
        if x_arr.size >= 2 and float(np.ptp(x_arr)) > 0.0:
            p = np.polyfit(x_arr, y_arr, 1)
            xx = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 100)
            yy = p[0] * xx + p[1]
            good = yy >= 0.0
            if np.any(good):
                ax.plot(xx[good], yy[good], color=clr, linewidth=2.2, alpha=0.9)

    xa = np.asarray(x_all, dtype=float)
    ya = np.asarray(y_all, dtype=float)
    if ya.size:
        ymax = float(np.max(ya))
        ax.set_ylim(0.0, min(1.0, ymax + 0.05 * max(ymax, 1e-6)))
    else:
        ax.set_ylim(0.0, 1.0)
    if xa.size:
        x_margin = 0.05 * max(float(np.ptp(xa)), 1e-6)
        ax.set_xlim(float(np.min(xa)) - x_margin, float(np.max(xa)) + x_margin)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.set_title("SFC vs occupancy")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    _save_figure(fig, fig_dir, stem, save_transparent_copy=save_transparent_copy)
    pd.DataFrame(p_rows).to_csv(table_dir / pvalues_filename, sep="\t", index=False)
    pd.DataFrame(point_rows).to_csv(table_dir / points_filename, index=False)


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.output_root).expanduser().resolve()
    fig_dir = out_root / "figs"
    table_dir = out_root / "tables"
    npy_dir = out_root / "arrays"
    log_dir = out_root / "logs"
    for p in (fig_dir, table_dir, npy_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    cohorts = _cohorts(include_coma=bool(args.include_coma))

    if args.reuse_existing:
        print(f"Reusing cached pooled artifacts from: {out_root}")
        pooled = _load_cached_pooled_data(args=args, out_root=out_root)
        ipvc_path = out_root / "ipvc_per_k.json"
        if not ipvc_path.exists():
            raise RuntimeError(f"--reuse-existing requested but missing: {ipvc_path}")
        ipvc_meta = json.loads(ipvc_path.read_text())
        k_vals = np.asarray(ipvc_meta["k_range"], dtype=int)
        ipvc_arr = np.asarray(ipvc_meta["ipvc"], dtype=float)
        k_star = int(ipvc_meta.get("k_star", int(k_vals[int(np.argmax(ipvc_arr))])))
    else:
        print(f"Building pooled phase patterns from: {Path(args.data_root).expanduser().resolve()}")
        pooled = _build_pooled_patterns(args=args, out_dir=out_root)
        pooled.excluded_df.to_csv(table_dir / "excluded_subjects_nonfinite_bold.csv", index=False)
        print(f"Excluded non-finite subjects: {int(pooled.excluded_df.shape[0])}")
        print(f"Retained subjects: {len(pooled.subject_ids)}")
        print(f"Pooled rows x features: {pooled.x.shape}")
        print("Retained cohort counts:", pd.Series(pooled.subject_cohorts).value_counts().to_dict())

        subject_span_rows = []
        for sid, coh, sta, sed, span in zip(
            pooled.subject_ids,
            pooled.subject_cohorts,
            pooled.subject_stages,
            pooled.subject_sedation,
            pooled.subject_splits,
        ):
            subject_span_rows.append(
                {
                    "subject_id": sid,
                    "cohort": coh,
                    "stage": sta,
                    "sedation": sed,
                    "start": int(span[0]),
                    "end": int(span[1]),
                    "n_rows": int(span[1] - span[0]),
                }
            )
        pd.DataFrame(subject_span_rows).to_csv(table_dir / "subject_splits.csv", index=False)
        (out_root / "subject_splits.json").write_text(json.dumps(pooled.subject_splits, indent=2))
        (out_root / f"condition_splits_grouped_{'_'.join(c.upper() for c in cohorts)}.json").write_text(
            json.dumps({k: [int(v[0]), int(v[1])] for k, v in pooled.condition_splits.items()}, indent=2)
        )

        sc_ref = _build_sc_reference(args=args, pooled=pooled)
        np.save(npy_dir / "SC_pooled_reference.npy", sc_ref)

        k_vals = np.arange(int(args.k_min), int(args.k_max) + 1, dtype=int)
        ipvc_vals: list[float] = []
        x = pooled.x
        print(f"K sweep: {k_vals.tolist()}")
        for k in k_vals:
            km = KMeans(
                n_clusters=int(k),
                random_state=int(args.random_seed),
                n_init=int(args.n_init),
                max_iter=int(args.max_iter),
            )
            labels = km.fit_predict(x)
            centers = np.asarray(km.cluster_centers_, dtype=float)
            sq = np.asarray([squareform(row) for row in centers], dtype=float)
            ipvc = _inter_pattern_corr_var(sq)
            centers_sorted, sfc_sorted, inv = _state_sfc_sorted(centers, sc_ref=sc_ref, mode=args.state_sc_correlation_mode)
            labels_sorted = np.asarray(inv[np.asarray(labels, dtype=int)], dtype=int)
            ipvc_vals.append(float(ipvc))

            np.save(npy_dir / f"labels_k{k}_sfcSorted.npy", labels_sorted)
            np.save(npy_dir / f"centroids_k{k}_vec_sfcSorted.npy", centers_sorted)
            np.save(npy_dir / f"squareforms_k{k}_sfcSorted.npy", np.asarray([squareform(v) for v in centers_sorted], dtype=float))
            (out_root / f"sfc_k{k}_sorted_GLOBALmean.json").write_text(json.dumps(sfc_sorted.tolist(), indent=2))
            print(f"[k={k}] ipvc={ipvc:.6f} rows={labels_sorted.size}")

        ipvc_arr = np.asarray(ipvc_vals, dtype=float)
        k_star = int(k_vals[int(np.argmax(ipvc_arr))])
        ipvc_meta = {"k_range": k_vals.tolist(), "ipvc": ipvc_arr.tolist(), "k_star": int(k_star)}
        (out_root / "ipvc_per_k.json").write_text(json.dumps(ipvc_meta, indent=2))
        pd.DataFrame({"k": k_vals, "ipvc": ipvc_arr}).to_csv(table_dir / "ipvc_per_k.csv", index=False)

    _plot_fig1_ipvc(
        k_vals=k_vals,
        ipvc_vals=ipvc_arr,
        k_star=k_star,
        fig_dir=fig_dir,
        save_transparent_copy=bool(args.transparent_figures),
    )

    labels_star = np.load(npy_dir / f"labels_k{k_star}_sfcSorted.npy")
    centers_star = np.load(npy_dir / f"centroids_k{k_star}_vec_sfcSorted.npy")
    sfc_path = out_root / f"sfc_k{k_star}_sorted_GLOBALmean.json"
    sfc_sorted = np.asarray(json.loads(sfc_path.read_text()), dtype=float) if sfc_path.exists() else np.full(k_star, np.nan)

    x_subject = _subject_specific_sfc_by_state(centers_star, pooled=pooled, mode=args.state_sc_correlation_mode)
    x_pooled_ref = np.tile(np.asarray(sfc_sorted, dtype=float).reshape(1, -1), (len(pooled.subject_ids), 1))
    display_identity = np.tile(np.arange(int(k_star), dtype=int).reshape(1, -1), (len(pooled.subject_ids), 1))
    display_subject_rank = np.argsort(np.nan_to_num(x_subject, nan=np.inf), axis=1)

    order_rows: list[dict[str, Any]] = []
    for si, sid in enumerate(pooled.subject_ids):
        for rank in range(int(k_star)):
            order_rows.append(
                {
                    "subject_id": sid,
                    "cohort": pooled.subject_cohorts[si],
                    "display_pattern": int(rank) + 1,
                    "state_subject_rank_order": int(display_subject_rank[si, rank]) + 1,
                    "state_pooled_identity_order": int(display_identity[si, rank]) + 1,
                }
            )
    pd.DataFrame(order_rows).to_csv(table_dir / "fig7_kstar_subject_state_order_maps.csv", index=False)

    if args.fig7_x_mode in {"subject_specific", "both"}:
        _plot_fig7_for_k(
            labels=np.asarray(labels_star, dtype=int),
            x_per_subject_state=np.asarray(x_subject, dtype=float),
            display_state_index=np.asarray(display_subject_rank, dtype=int),
            pooled=pooled,
            cohorts=cohorts,
            k=int(k_star),
            alpha_fdr=float(args.alpha_fdr),
            fig_dir=fig_dir,
            table_dir=table_dir,
            stem="fig7_kstar_distributions_and_sfc",
            pvalues_filename="fig7_kstar_pairwise_pvalues.tsv",
            points_filename="fig7_kstar_subject_points_subject_specific.csv",
            x_label="SC-FC coupling (subject-specific SC)",
            save_transparent_copy=bool(args.transparent_figures),
        )

    if args.fig7_x_mode in {"pooled_ref", "both"}:
        _plot_fig7_for_k(
            labels=np.asarray(labels_star, dtype=int),
            x_per_subject_state=np.asarray(x_pooled_ref, dtype=float),
            display_state_index=np.asarray(display_identity, dtype=int),
            pooled=pooled,
            cohorts=cohorts,
            k=int(k_star),
            alpha_fdr=float(args.alpha_fdr),
            fig_dir=fig_dir,
            table_dir=table_dir,
            stem="fig7_kstar_distributions_and_sfc_pooledref",
            pvalues_filename="fig7_kstar_pairwise_pvalues_pooledref.tsv",
            points_filename="fig7_kstar_subject_points_pooledref.csv",
            x_label="SC-FC coupling (pooled SC reference)",
            save_transparent_copy=bool(args.transparent_figures),
        )

    run_meta = {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "output_root": str(out_root),
        "include_coma": bool(args.include_coma),
        "cohorts": list(cohorts),
        "palette": {k: v for k, v in CURRENT_PALETTE.items() if k in cohorts},
        "n_subjects_retained": int(len(pooled.subject_ids)),
        "k_range": np.asarray(k_vals, dtype=int).tolist(),
        "k_star": int(k_star),
        "n_rows_total": int(labels_star.size),
        "n_features": int(centers_star.shape[1]),
        "state_sc_correlation_mode": str(args.state_sc_correlation_mode),
        "pooled_sc_reference": str(args.pooled_sc_reference),
        "pipeline": str(args.pipeline),
        "trim_edge_samples": int(args.trim_edge_samples),
        "tr_seconds": float(args.tr_seconds),
        "bandpass_low_hz": float(args.bandpass_low_hz),
        "bandpass_high_hz": float(args.bandpass_high_hz),
        "filter_order": int(args.filter_order),
        "n_init": int(args.n_init),
        "max_iter": int(args.max_iter),
        "random_seed": int(args.random_seed),
        "reuse_existing": bool(args.reuse_existing),
        "fig7_x_mode": str(args.fig7_x_mode),
        "transparent_figures": bool(args.transparent_figures),
    }
    (log_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    print(f"k* (max IPVC): {k_star}")
    print(f"Saved fig1: {fig_dir / 'fig1_k_vs_ipvc.pdf'}")
    if args.fig7_x_mode in {"subject_specific", "both"}:
        print(f"Saved fig7 (subject-specific x): {fig_dir / 'fig7_kstar_distributions_and_sfc.pdf'}")
    if args.fig7_x_mode in {"pooled_ref", "both"}:
        print(f"Saved fig7 (pooled-ref x): {fig_dir / 'fig7_kstar_distributions_and_sfc_pooledref.pdf'}")
    print(f"Saved outputs root: {out_root}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(doc_liege_raw("doc_data")))
    p.add_argument(
        "--output-root",
        type=str,
        default=str(doc_liege_results("doc_patients_new_bold_brain_states_audited", "legacy_style_repro_current_palette")),
    )
    p.add_argument("--include-coma", action="store_true", help="Include acute coma and acute sedated coma cohorts.")

    p.add_argument("--pipeline", type=str, default="brain_act_legacy", choices=["standard", "brain_act_legacy"])
    p.add_argument("--trim-edge-samples", type=int, default=9)
    p.add_argument("--tr-seconds", type=float, default=2.4)
    p.add_argument("--bandpass-low-hz", type=float, default=0.01)
    p.add_argument("--bandpass-high-hz", type=float, default=0.20)
    p.add_argument("--filter-order", type=int, default=3)

    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--n-init", type=int, default=40)
    p.add_argument("--max-iter", type=int, default=260)
    p.add_argument("--random-seed", type=int, default=11)
    p.add_argument("--alpha-fdr", type=float, default=0.05)
    p.add_argument("--reuse-existing", action="store_true")

    p.add_argument("--roi-reorder-mode", type=str, default="aal90_fc", choices=["auto", "none", "aal90_fc", "aal90_sc", "aal90_both"])
    p.add_argument("--max-subjects-per-group", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=10)

    p.add_argument("--pooled-sc-reference", type=str, default="control", choices=["control", "all"])
    p.add_argument(
        "--state-sc-correlation-mode",
        type=str,
        default="upper",
        choices=["upper", "full"],
        help="Method for centroid-vs-SC correlation used for SFC sorting.",
    )
    p.add_argument(
        "--fig7-x-mode",
        type=str,
        default="both",
        choices=["subject_specific", "pooled_ref", "both"],
        help="X-axis coupling source for Fig7 scatter.",
    )
    p.add_argument(
        "--transparent-figures",
        action="store_true",
        help="Also save transparent-background copies with a _transparent suffix.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
