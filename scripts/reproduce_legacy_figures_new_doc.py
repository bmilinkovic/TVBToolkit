#!/usr/bin/env python3
"""Reproduce legacy-style Fig1/Fig7 on new DoC BOLD data.

Outputs:
- fig1_k_vs_ipvc.(pdf/png/svg): IPVC across k and selected k* (argmax IPVC).
- fig7_kstar_distributions_and_sfc.(pdf/png/svg): Occupancy distributions by cohort
  + pooled-state SFC vs occupancy scatter for chosen k*.
- Companion tables/json/npy for auditability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans

from brain_states_new_doc_bold_audited import (
    COHORTS,
    PALETTE,
    _extract_phase_patterns_clean,
    _full_matrix_vector,
    _maybe_apply_roi_reordering,
    _safe_pearson,
    _upper_triangle_vector,
    load_new_doc_subjects,
    set_publication_style,
)


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _inter_pattern_corr_var(squareforms: np.ndarray) -> float:
    """Legacy IPVC: variance of centroid correlation matrix after full flatten."""
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
    """Sort pooled centroids by SFC ascending and relabel labels via returned inverse map."""
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


def _build_pooled_patterns(args: argparse.Namespace, out_dir: Path) -> PooledData:
    records, _ = load_new_doc_subjects(Path(args.data_root), max_subjects_per_group=args.max_subjects_per_group)
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

    # Cohort blocks in concatenated pooled time index.
    condition_splits: dict[str, tuple[int, int]] = {}
    for cohort in COHORTS:
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

    records, _ = load_new_doc_subjects(Path(args.data_root), max_subjects_per_group=args.max_subjects_per_group)
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
    for cohort in COHORTS:
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


def _plot_fig1_ipvc(k_vals: np.ndarray, ipvc_vals: np.ndarray, k_star: int, fig_dir: Path) -> None:
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
    _save_figure(fig, fig_dir, "fig1_k_vs_ipvc")


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
    k: int,
    alpha_fdr: float,
    fig_dir: Path,
    table_dir: Path,
    stem: str,
    pvalues_filename: str,
    points_filename: str,
    x_label: str,
) -> None:
    set_publication_style()
    cohorts = list(COHORTS)
    try:
        ctrl_idx = int(cohorts.index("control"))
    except ValueError as exc:
        raise RuntimeError("Expected 'control' cohort in COHORTS for control-only comparisons.") from exc
    target_idx = [i for i in range(len(cohorts)) if i != ctrl_idx]
    pair_idx = [(ctrl_idx, j) for j in target_idx]
    pair_lbl = [f"{cohorts[ctrl_idx]} vs {cohorts[j]}" for j in target_idx]

    subj_rates = [_occ_rates_for_span(labels, span, k) for span in pooled.subject_splits]
    disp_idx = np.asarray(display_state_index, dtype=int)
    if disp_idx.shape != (len(subj_rates), int(k)):
        raise ValueError(f"display_state_index must be [subjects,k], got {disp_idx.shape}")

    per_centroid_data: list[dict[str, list[float]]] = []
    for c in range(k):
        by_cond = {cond: [] for cond in cohorts}
        for si, (rv, cond) in enumerate(zip(subj_rates, pooled.subject_cohorts)):
            st = int(disp_idx[si, c])
            by_cond[str(cond)].append(float(rv[st]))
        per_centroid_data.append(by_cond)

    ncols = int(k + 1)
    fig = plt.figure(figsize=(3.7 * ncols, 5.3), dpi=300)
    gs = fig.add_gridspec(1, ncols, wspace=0.28)

    rng = np.random.default_rng(42)
    colors = [PALETTE[c] for c in cohorts]
    x_positions = np.arange(len(cohorts), dtype=float)
    p_rows: list[dict[str, Any]] = []

    for c in range(k):
        ax = fig.add_subplot(gs[0, c])
        arrays = [np.asarray(per_centroid_data[c][cond], dtype=float) for cond in cohorts]

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
        y_star = min(1.18, ymax + 0.08)
        for (i, j), (pa, _) in zip(pair_idx, adj):
            stars = _sig_stars(pa, alpha=alpha_fdr)
            if stars:
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

        ax.set_xticks(x_positions, [c.upper() for c in cohorts], rotation=25, ha="right")
        ax.set_ylim(0.0, 1.24)
        ax.set_title(f"Pattern {c + 1}")
        ax.set_ylabel("Occupancy rate" if c == 0 else "")
        ax.grid(alpha=0.20)

    ax = fig.add_subplot(gs[0, k])
    x_all: list[float] = []
    y_all: list[float] = []
    point_rows: list[dict[str, Any]] = []

    for cohort, clr in zip(cohorts, colors):
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
    _save_figure(fig, fig_dir, stem)
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
        (out_root / "condition_splits_grouped_control_EMCS_MCS_UWS.json").write_text(
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

    # Always re-render fig1 from current IPVC metadata.
    _plot_fig1_ipvc(k_vals=k_vals, ipvc_vals=ipvc_arr, k_star=k_star, fig_dir=fig_dir)

    # Load k* artifacts.
    labels_star = np.load(npy_dir / f"labels_k{k_star}_sfcSorted.npy")
    centers_star = np.load(npy_dir / f"centroids_k{k_star}_vec_sfcSorted.npy")
    sfc_path = out_root / f"sfc_k{k_star}_sorted_GLOBALmean.json"
    sfc_sorted = np.asarray(json.loads(sfc_path.read_text()), dtype=float) if sfc_path.exists() else np.full(k_star, np.nan)

    x_subject = _subject_specific_sfc_by_state(centers_star, pooled=pooled, mode=args.state_sc_correlation_mode)
    x_pooled_ref = np.tile(np.asarray(sfc_sorted, dtype=float).reshape(1, -1), (len(pooled.subject_ids), 1))
    display_identity = np.tile(np.arange(int(k_star), dtype=int).reshape(1, -1), (len(pooled.subject_ids), 1))
    display_subject_rank = np.argsort(np.nan_to_num(x_subject, nan=np.inf), axis=1)

    # Save ordering maps for auditability.
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
            k=int(k_star),
            alpha_fdr=float(args.alpha_fdr),
            fig_dir=fig_dir,
            table_dir=table_dir,
            stem="fig7_kstar_distributions_and_sfc",
            pvalues_filename="fig7_kstar_pairwise_pvalues.tsv",
            points_filename="fig7_kstar_subject_points_subject_specific.csv",
            x_label="SC-FC coupling (subject-specific SC)",
        )

    if args.fig7_x_mode in {"pooled_ref", "both"}:
        _plot_fig7_for_k(
            labels=np.asarray(labels_star, dtype=int),
            x_per_subject_state=np.asarray(x_pooled_ref, dtype=float),
            display_state_index=np.asarray(display_identity, dtype=int),
            pooled=pooled,
            k=int(k_star),
            alpha_fdr=float(args.alpha_fdr),
            fig_dir=fig_dir,
            table_dir=table_dir,
            stem="fig7_kstar_distributions_and_sfc_pooledref",
            pvalues_filename="fig7_kstar_pairwise_pvalues_pooledref.tsv",
            points_filename="fig7_kstar_subject_points_pooledref.csv",
            x_label="SC-FC coupling (pooled SC reference)",
        )

    run_meta = {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "output_root": str(out_root),
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
    p.add_argument("--data-root", type=str, default="data/doc_patients_new_data")
    p.add_argument(
        "--output-root",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/legacy_style_repro",
    )

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
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing saved k-sweep artifacts in output-root (skip re-fitting KMeans).",
    )

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
        default="subject_specific",
        choices=["subject_specific", "pooled_ref", "both"],
        help="X-axis coupling source for Fig7 scatter.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
