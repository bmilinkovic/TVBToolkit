#!/usr/bin/env python3
"""Empirical baseline analyses from current pooled-BOLD implementation.

Produces:
1) Markov transition summaries/figures for pooled BOLD states (k*=argmax IPVC).
2) Empirical LZC summaries/figures on subject BOLD timeseries.
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
from scipy.stats import kruskal, mannwhitneyu

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Keep caches/logs writable in sandboxed runs.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))
os.environ.setdefault("TVB_USER_HOME", str((_REPO_ROOT / ".tvb-temp").resolve()))

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

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


def _collapse_runs(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    if arr.size == 0:
        return arr
    out = [int(arr[0])]
    for x in arr[1:]:
        xi = int(x)
        if xi != out[-1]:
            out.append(xi)
    return np.asarray(out, dtype=int)


def _transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    tm = np.zeros((n_states, n_states), dtype=float)
    arr = np.asarray(labels, dtype=int).reshape(-1)
    if arr.size < 2:
        return tm
    for a, b in zip(arr[:-1], arr[1:]):
        ia = int(a)
        ib = int(b)
        if 0 <= ia < n_states and 0 <= ib < n_states:
            tm[ia, ib] += 1.0
    rs = tm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        tm = np.divide(tm, rs, out=np.zeros_like(tm), where=rs > 0)
    return tm


def _transition_no_self(labels: np.ndarray, n_states: int, collapse_runs: bool = True) -> np.ndarray:
    seq = _collapse_runs(labels) if collapse_runs else np.asarray(labels, dtype=int).reshape(-1)
    tm = np.zeros((n_states, n_states), dtype=float)
    if seq.size < 2:
        return tm
    for a, b in zip(seq[:-1], seq[1:]):
        ia = int(a)
        ib = int(b)
        if ia == ib:
            continue
        if 0 <= ia < n_states and 0 <= ib < n_states:
            tm[ia, ib] += 1.0
    rs = tm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        tm = np.divide(tm, rs, out=np.zeros_like(tm), where=rs > 0)
    np.fill_diagonal(tm, 0.0)
    return tm


def _stationary_distribution(P: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    k = int(P.shape[0])
    mu = np.ones(k, dtype=float) / float(max(k, 1))
    for _ in range(max_iter):
        nxt = mu @ P
        if np.allclose(nxt, mu, atol=tol, rtol=0.0):
            break
        mu = nxt
    return mu


def _markov_entropy_bits(P: np.ndarray) -> float:
    mu = _stationary_distribution(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.where(P > 0, P * np.log2(P), 0.0)
    return float(-np.sum(mu[:, None] * inner))


def _markov_entropy_norm(P: np.ndarray) -> float:
    k = int(P.shape[0])
    if k <= 1:
        return 0.0
    return float(_markov_entropy_bits(P) / np.log2(k))


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


def _plot_transition_heatmaps(cohort_mats: dict[str, np.ndarray], n_states: int, out_dir: Path, stem: str, title: str) -> None:
    set_publication_style()
    vmax = 0.0
    for c in COHORTS:
        m = cohort_mats.get(c)
        if m is not None and m.size:
            vmax = max(vmax, float(np.max(m)))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.flatten()
    im = None
    for ax, cohort in zip(axes_flat, COHORTS):
        m = cohort_mats.get(cohort, np.zeros((n_states, n_states), dtype=float))
        im = ax.imshow(m, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(cohort.upper())
        ax.set_xticks(np.arange(n_states), [f"S{i}" for i in range(1, n_states + 1)])
        ax.set_yticks(np.arange(n_states), [f"S{i}" for i in range(1, n_states + 1)])
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")
    if im is not None:
        fig.colorbar(im, ax=axes_flat, fraction=0.025, pad=0.02, label="Transition probability")
    fig.suptitle(title, y=0.98)
    _save_figure(fig, out_dir, stem)


def _plot_entropy_violin(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True)
    metrics = [
        ("entropy_rate_bits", "Entropy rate (bits)"),
        ("entropy_rate_norm", "Normalized entropy rate"),
    ]
    for ax, (col, ylab) in zip(axes, metrics):
        vals = []
        labels = []
        for cohort in COHORTS:
            x = df.loc[df["cohort"] == cohort, col].to_numpy(dtype=float)
            if x.size == 0:
                continue
            vals.append(x)
            labels.append(cohort)
        if vals:
            vp = ax.violinplot(vals, showmeans=True, showextrema=False)
            for i, body in enumerate(vp["bodies"]):
                c = labels[i]
                body.set_facecolor(PALETTE[c])
                body.set_edgecolor("#111111")
                body.set_alpha(0.55)
            for i, x in enumerate(vals, start=1):
                j = np.linspace(-0.08, 0.08, x.size)
                ax.scatter(np.full(x.size, i) + j, x, s=16, color="#111111", alpha=0.58)
        ax.set_xticks(np.arange(1, len(labels) + 1), [x.upper() for x in labels])
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
    fig.suptitle("Empirical pooled-state Markov entropy by cohort", y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_dir, stem)


def _plot_lzc_by_cohort(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True)
    specs = [
        ("lzc_multichannel", "LZC multichannel"),
        ("lzc_single_channel", "LZC single-channel mean"),
    ]
    for ax, (col, ttl) in zip(axes, specs):
        vals = []
        labels = []
        for cohort in COHORTS:
            x = df.loc[df["cohort"] == cohort, col].to_numpy(dtype=float)
            if x.size == 0:
                continue
            vals.append(x)
            labels.append(cohort)
        if vals:
            vp = ax.violinplot(vals, showmeans=True, showextrema=False)
            for i, body in enumerate(vp["bodies"]):
                c = labels[i]
                body.set_facecolor(PALETTE[c])
                body.set_edgecolor("#111111")
                body.set_alpha(0.55)
            for i, x in enumerate(vals, start=1):
                j = np.linspace(-0.08, 0.08, x.size)
                ax.scatter(np.full(x.size, i) + j, x, s=16, color="#111111", alpha=0.58)
        ax.set_xticks(np.arange(1, len(labels) + 1), [x.upper() for x in labels])
        ax.set_ylabel(ttl)
        ax.set_title(ttl)
        ax.grid(alpha=0.25)
    fig.suptitle("Empirical BOLD Lempel-Ziv complexity by cohort", y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_dir, stem)


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    pooled_root = Path(args.pooled_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    npz_dir = out_root / "npz"
    log_dir = out_root / "logs"
    for p in (fig_dir, tab_dir, npz_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    ipvc_meta = json.loads((pooled_root / "ipvc_per_k.json").read_text())
    k_star = int(ipvc_meta["k_star"]) if args.k is None else int(args.k)
    labels = np.load(pooled_root / "arrays" / f"labels_k{k_star}_sfcSorted.npy").astype(int)
    splits_df = pd.read_csv(pooled_root / "tables" / "subject_splits.csv").sort_values("start").reset_index(drop=True)
    needed = {"subject_id", "cohort", "stage", "sedation", "start", "end", "n_rows"}
    if not needed.issubset(set(splits_df.columns)):
        raise RuntimeError(f"subject_splits.csv missing required columns: {sorted(needed)}")

    n_rows_total = int(splits_df["n_rows"].sum())
    if n_rows_total != int(labels.size):
        raise RuntimeError(f"Pooled label length mismatch: labels={labels.size} vs n_rows={n_rows_total}")

    # ------------------------
    # Markov from pooled labels
    # ------------------------
    sub_rows = []
    tr_rows = []
    tr_ns_rows = []
    occ_rows = []
    for row in splits_df.itertuples(index=False):
        start = int(row.start)
        end = int(row.end)
        seq = np.asarray(labels[start:end], dtype=int)
        if seq.size == 0:
            continue
        occ = np.bincount(seq, minlength=k_star).astype(float) / float(seq.size)
        tm = _transition_matrix(seq, n_states=k_star)
        tm_ns = _transition_no_self(seq, n_states=k_star, collapse_runs=True)
        e_bits = _markov_entropy_bits(tm_ns)
        e_norm = _markov_entropy_norm(tm_ns)

        sub_rows.append(
            {
                "subject_id": str(row.subject_id),
                "cohort": str(row.cohort),
                "stage": str(row.stage),
                "sedation": str(row.sedation),
                "start": start,
                "end": end,
                "n_rows": int(seq.size),
                "entropy_rate_bits": float(e_bits),
                "entropy_rate_norm": float(e_norm),
            }
        )
        for s in range(k_star):
            occ_rows.append(
                {
                    "subject_id": str(row.subject_id),
                    "cohort": str(row.cohort),
                    "state": int(s) + 1,
                    "occupancy": float(occ[s]),
                }
            )
            for t in range(k_star):
                tr_rows.append(
                    {
                        "subject_id": str(row.subject_id),
                        "cohort": str(row.cohort),
                        "from_state": int(s) + 1,
                        "to_state": int(t) + 1,
                        "probability": float(tm[s, t]),
                    }
                )
                tr_ns_rows.append(
                    {
                        "subject_id": str(row.subject_id),
                        "cohort": str(row.cohort),
                        "from_state": int(s) + 1,
                        "to_state": int(t) + 1,
                        "probability": float(tm_ns[s, t]),
                    }
                )

    markov_df = pd.DataFrame(sub_rows).sort_values(["cohort", "subject_id"]).reset_index(drop=True)
    occ_df = pd.DataFrame(occ_rows).sort_values(["cohort", "subject_id", "state"]).reset_index(drop=True)
    tr_df = pd.DataFrame(tr_rows)
    tr_ns_df = pd.DataFrame(tr_ns_rows)

    cohort_tm: dict[str, np.ndarray] = {}
    cohort_tm_ns: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        d = tr_df[tr_df["cohort"] == cohort]
        if not d.empty:
            piv = (
                d.pivot_table(index="from_state", columns="to_state", values="probability", aggfunc="mean")
                .reindex(index=np.arange(1, k_star + 1), columns=np.arange(1, k_star + 1), fill_value=0.0)
            )
            cohort_tm[cohort] = piv.to_numpy(dtype=float)
        d2 = tr_ns_df[tr_ns_df["cohort"] == cohort]
        if not d2.empty:
            piv2 = (
                d2.pivot_table(index="from_state", columns="to_state", values="probability", aggfunc="mean")
                .reindex(index=np.arange(1, k_star + 1), columns=np.arange(1, k_star + 1), fill_value=0.0)
            )
            cohort_tm_ns[cohort] = piv2.to_numpy(dtype=float)

    markov_df.to_csv(tab_dir / "empirical_pooled_markov_subject_summary.csv", index=False)
    occ_df.to_csv(tab_dir / "empirical_pooled_occupancy_long.csv", index=False)
    tr_df.to_csv(tab_dir / "empirical_pooled_transition_long.csv", index=False)
    tr_ns_df.to_csv(tab_dir / "empirical_pooled_transition_no_self_long.csv", index=False)

    _plot_transition_heatmaps(
        cohort_tm,
        n_states=k_star,
        out_dir=fig_dir,
        stem="empirical_pooled_transition_heatmaps",
        title=f"Empirical pooled-state transitions (k={k_star})",
    )
    _plot_transition_heatmaps(
        cohort_tm_ns,
        n_states=k_star,
        out_dir=fig_dir,
        stem="empirical_pooled_transition_heatmaps_no_self",
        title=f"Empirical pooled-state transitions no-self (k={k_star})",
    )
    _plot_entropy_violin(markov_df, fig_dir, "empirical_pooled_markov_entropy")

    np.savez_compressed(
        npz_dir / "empirical_pooled_markov_means.npz",
        cohorts=np.asarray([c for c in COHORTS if c in cohort_tm], dtype=object),
        transition_mean=np.asarray([cohort_tm[c] for c in COHORTS if c in cohort_tm], dtype=float),
        transition_no_self_mean=np.asarray([cohort_tm_ns[c] for c in COHORTS if c in cohort_tm_ns], dtype=float),
        k=np.asarray([k_star], dtype=int),
    )

    # ------------------------
    # LZC on empirical BOLD
    # ------------------------
    print("Loading empirical subject BOLD for LZC...")
    records, _ = load_new_doc_subjects(data_root, max_subjects_per_group=args.max_subjects_per_group)
    recs, reorder_qc_df, reorder_decision = _maybe_apply_roi_reordering(records, mode=args.roi_reorder_mode)
    reorder_qc_df.to_csv(tab_dir / "empirical_lzc_roi_order_qc.csv", index=False)

    id_keep = set(splits_df["subject_id"].astype(str).tolist())
    rec_map = {str(r.subject_id): r for r in recs}

    lzc_rows = []
    excluded_rows = []
    for sid in sorted(id_keep):
        rec = rec_map.get(sid)
        if rec is None:
            excluded_rows.append({"subject_id": sid, "reason": "not_found_in_loaded_records"})
            continue
        x = np.asarray(rec.timeseries, dtype=float)
        bad = ~np.isfinite(x)
        if np.any(bad):
            excluded_rows.append(
                {
                    "subject_id": sid,
                    "reason": "non_finite_timeseries",
                    "n_nonfinite_values": int(np.sum(bad)),
                    "n_bad_timepoints": int(np.sum(np.any(bad, axis=1))),
                    "n_bad_rois": int(np.sum(np.any(bad, axis=0))),
                }
            )
            continue
        lzc_m = float(lzc_multichannel(x))
        lzc_s = float(lzc_single_channel(x))
        lzc_rows.append(
            {
                "subject_id": sid,
                "cohort": str(rec.cohort),
                "stage": str(rec.stage),
                "sedation": str(rec.sedation),
                "n_timepoints": int(x.shape[0]),
                "n_regions": int(x.shape[1]),
                "lzc_multichannel": lzc_m,
                "lzc_single_channel": lzc_s,
            }
        )

    lzc_df = pd.DataFrame(lzc_rows).sort_values(["cohort", "subject_id"]).reset_index(drop=True)
    exc_df = pd.DataFrame(excluded_rows)
    lzc_df.to_csv(tab_dir / "empirical_lzc_subjects.csv", index=False)
    exc_df.to_csv(tab_dir / "empirical_lzc_excluded_subjects.csv", index=False)

    cohort_summary = (
        lzc_df.groupby("cohort", as_index=False)
        .agg(
            n_subjects=("subject_id", "size"),
            lzc_multichannel_mean=("lzc_multichannel", "mean"),
            lzc_multichannel_std=("lzc_multichannel", "std"),
            lzc_single_channel_mean=("lzc_single_channel", "mean"),
            lzc_single_channel_std=("lzc_single_channel", "std"),
        )
        .sort_values("cohort")
    )
    cohort_summary.to_csv(tab_dir / "empirical_lzc_cohort_summary.csv", index=False)
    _plot_lzc_by_cohort(lzc_df, fig_dir, "empirical_lzc_by_cohort")

    # Control-vs-other stats (Holm).
    ctrl = "control"
    stat_rows = []
    for metric in ("lzc_multichannel", "lzc_single_channel"):
        p_raw = []
        tmp = []
        x_ctrl = lzc_df.loc[lzc_df["cohort"] == ctrl, metric].to_numpy(dtype=float)
        for cohort in COHORTS:
            if cohort == ctrl:
                continue
            x_cmp = lzc_df.loc[lzc_df["cohort"] == cohort, metric].to_numpy(dtype=float)
            if x_ctrl.size == 0 or x_cmp.size == 0:
                p = np.nan
                u = np.nan
            else:
                u, p = mannwhitneyu(x_ctrl, x_cmp, alternative="two-sided")
            p_raw.append(1.0 if not np.isfinite(p) else float(p))
            tmp.append(
                {
                    "metric": metric,
                    "contrast": f"{ctrl} vs {cohort}",
                    "U": float(u) if np.isfinite(u) else np.nan,
                    "p_raw": float(p) if np.isfinite(p) else np.nan,
                    "n_control": int(x_ctrl.size),
                    "n_other": int(x_cmp.size),
                    "median_control": float(np.median(x_ctrl)) if x_ctrl.size else np.nan,
                    "median_other": float(np.median(x_cmp)) if x_cmp.size else np.nan,
                }
            )
        p_adj = _holm_correct(p_raw)
        for r, pa in zip(tmp, p_adj):
            r["p_holm"] = float(pa)
            stat_rows.append(r)

    # Omnibus per metric.
    omni_rows = []
    for metric in ("lzc_multichannel", "lzc_single_channel"):
        vals = [
            lzc_df.loc[lzc_df["cohort"] == c, metric].to_numpy(dtype=float)
            for c in COHORTS
            if not lzc_df.loc[lzc_df["cohort"] == c].empty
        ]
        if len(vals) >= 2:
            H, p = kruskal(*vals)
            omni_rows.append({"metric": metric, "H": float(H), "p_kw": float(p)})
    pd.DataFrame(omni_rows).to_csv(tab_dir / "empirical_lzc_stats_omnibus.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(tab_dir / "empirical_lzc_stats_control_vs_others.csv", index=False)

    # Link LZC with pooled Markov entropy.
    merged = markov_df.merge(lzc_df, on=["subject_id", "cohort", "stage", "sedation"], how="inner")
    merged.to_csv(tab_dir / "empirical_markov_lzc_joined_subjects.csv", index=False)

    run_meta = {
        "data_root": str(data_root),
        "pooled_root": str(pooled_root),
        "output_root": str(out_root),
        "k_star_used": int(k_star),
        "n_subjects_markov": int(markov_df.shape[0]),
        "n_subjects_lzc": int(lzc_df.shape[0]),
        "n_subjects_lzc_excluded": int(exc_df.shape[0]),
        "roi_reorder_requested": str(args.roi_reorder_mode),
        "roi_reorder_applied": str(reorder_decision["applied_mode"]),
        "lzc_metric_impl": "tvbtoolkit.complexity.measures.lzc_multichannel / lzc_single_channel",
    }
    (log_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    print(f"Saved outputs to: {out_root}")
    print(f"k*: {k_star}")
    print("Markov subjects:", int(markov_df.shape[0]))
    print("LZC subjects:", int(lzc_df.shape[0]), "excluded:", int(exc_df.shape[0]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default="data/doc_patients_new_data")
    p.add_argument(
        "--pooled-root",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/legacy_style_repro",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/empirical_markov_lzc",
    )
    p.add_argument("--k", type=int, default=None, help="Optional override for pooled k (defaults to k* from ipvc_per_k.json).")
    p.add_argument("--roi-reorder-mode", type=str, default="aal90_fc", choices=["auto", "none", "aal90_fc", "aal90_sc", "aal90_both"])
    p.add_argument("--max-subjects-per-group", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
