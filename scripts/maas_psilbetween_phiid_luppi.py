#!/usr/bin/env python3
"""Run true STS/RTR PhiID and Luppi-style downstream analysis for Maas psilocybin.

This is the companion PhiID workflow for
``scripts/maas_psilbetween_fc_brain_states.py``.  It mirrors the existing
coma/Luppi downstream analysis, but uses the Maas between-subject AAL116 data.

The expensive step is MATLAB ``PhiIDFull``.  By default the script only exports
inputs and renders downstream figures if PhiID outputs already exist.  Pass
``--run-matlab`` to launch the full pairwise STS/RTR atom computation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from maas_psilbetween_fc_brain_states import (  # noqa: E402
    CONDITION_LABEL,
    METADATA_FILE,
    PSIL_FILE,
    ROI_NAMES_FILE,
    fc_matrix,
    load_between_subject_scans,
    load_metadata,
    load_roi_names,
)
from tvbtoolkit.analysis import (  # noqa: E402
    average_atom_matrices_by_group,
    build_matlab_batch_command,
    export_phiid_subject_inputs,
    load_phiid_index,
    load_phiid_matrix,
    sanitize_subject_stub,
    save_group_average_outputs,
)
from tvbtoolkit.analysis.luppi2022 import (  # noqa: E402
    edge_rank_gradient,
    matrix_spearman_similarity,
    redundancy_synergy_rank_gradient,
    threshold_top_density,
    weighted_global_efficiency,
    weighted_modularity,
)


DEFAULT_OUT = _REPO_ROOT / "results" / "maas_psilbetween_phiid_luppi"
ATOM_ORDER = ["rtr", "sts"]
ATOM_LABEL = {"rtr": "Redundancy (RTR)", "sts": "Synergy (STS)"}
ATOM_COLOR = {"rtr": "#2F7D5A", "sts": "#C43C39"}
CONDITION_ORDER = ["pla", "psil"]
GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    "redundancy_green_synergy_red",
    ["#2F7D5A", "#F7F3E8", "#C43C39"],
)
RTR_CMAP = LinearSegmentedColormap.from_list("rtr_green", ["#F7F3E8", "#B8D8C6", "#2F7D5A"])
STS_CMAP = LinearSegmentedColormap.from_list("sts_red", ["#F7F3E8", "#E9B7A8", "#C43C39"])


@dataclass(frozen=True)
class MaasPhiidRecord:
    subject_id: str
    cohort: str
    stage: str
    sedation: str
    condition: str
    timeseries: np.ndarray
    source_fc_file: str = ""
    source_sc_file: str = ""
    source_subject_index: int = -1
    source_subject_label: str = ""


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return q
    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * ranked.size / (np.arange(ranked.size) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(adj, 0.0, 1.0)
    q[finite] = out
    return q


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#F7F3E8",
            "axes.facecolor": "#F7F3E8",
            "savefig.facecolor": "#F7F3E8",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=320 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def build_records(
    data_file: Path,
    metadata_file: Path,
    roi_names_file: Path,
) -> tuple[list[MaasPhiidRecord], pd.DataFrame, list[str]]:
    scans = load_between_subject_scans(data_file)
    metadata = load_metadata(metadata_file)
    roi_names = load_roi_names(roi_names_file)
    meta_cols = ["subject", "condition"]
    known = set(tuple(x) for x in metadata[meta_cols].astype(str).to_numpy())
    records: list[MaasPhiidRecord] = []
    for idx, scan in enumerate(scans):
        key = (str(scan.subject), str(scan.condition))
        if key not in known:
            raise ValueError(f"No metadata row for scan {key}.")
        subject_id = f"maas_psilbetween_{scan.condition}_{scan.subject}"
        records.append(
            MaasPhiidRecord(
                subject_id=subject_id,
                cohort=str(scan.condition),
                stage="between_subject",
                sedation="none",
                condition=str(scan.condition),
                timeseries=np.asarray(scan.timeseries, dtype=float),
                source_fc_file=str(data_file),
                source_subject_index=idx,
                source_subject_label=str(scan.subject),
            )
        )
    return records, metadata, roi_names


def condition_label(condition: str) -> str:
    return CONDITION_LABEL.get(condition, str(condition).upper())


def _record_fc_by_stub(records: list[MaasPhiidRecord]) -> dict[str, np.ndarray]:
    return {
        sanitize_subject_stub(rec.subject_id): fc_matrix(np.asarray(rec.timeseries, dtype=float))
        for rec in records
    }


def _subject_atom_paths(index_df: pd.DataFrame) -> pd.DataFrame:
    sub = index_df.loc[index_df["atom"].isin(ATOM_ORDER), ["subject_stub", "atom", "path"]].copy()
    wide = sub.pivot(index="subject_stub", columns="atom", values="path").reset_index()
    return wide.dropna(subset=ATOM_ORDER)


def _safe_similarity(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(matrix_spearman_similarity(a, b, k=1))
    except Exception:
        return float("nan")


def _safe_modularity(matrix: np.ndarray) -> float:
    try:
        return float(weighted_modularity(matrix))
    except Exception:
        return float("nan")


def compute_subject_tables(
    index_df: pd.DataFrame,
    records: list[MaasPhiidRecord],
    roi_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fc_by_stub = _record_fc_by_stub(records)
    meta = index_df.drop_duplicates("subject_stub").set_index("subject_stub")
    paths = _subject_atom_paths(index_df)
    metric_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []
    nodal_rows: list[dict[str, Any]] = []
    edge_stat_values: dict[tuple[str, str], list[np.ndarray]] = {}

    for row in paths.to_dict(orient="records"):
        stub = str(row["subject_stub"])
        if stub not in fc_by_stub or stub not in meta.index:
            continue
        condition = str(meta.loc[stub, "cohort"])
        fc = fc_by_stub[stub]
        sts = load_phiid_matrix(row["sts"], atom="sts")
        rtr = load_phiid_matrix(row["rtr"], atom="rtr")
        grad = redundancy_synergy_rank_gradient(sts, rtr)
        for atom, matrix in [("rtr", rtr), ("sts", sts)]:
            edge_stat_values.setdefault((condition, atom), []).append(np.asarray(matrix, dtype=float))
            sim_rows.append(
                {
                    "subject_stub": stub,
                    "condition": condition,
                    "atom": atom,
                    "fc_similarity_rho": _safe_similarity(fc, matrix),
                }
            )
            metric_rows.append(
                {
                    "subject_stub": stub,
                    "condition": condition,
                    "atom": atom,
                    "global_efficiency": float(weighted_global_efficiency(matrix)),
                    "modularity": _safe_modularity(matrix),
                }
            )
        for roi_idx, value in enumerate(grad, start=1):
            nodal_rows.append(
                {
                    "subject_stub": stub,
                    "condition": condition,
                    "roi_index": roi_idx,
                    "roi_name": roi_names[roi_idx - 1],
                    "gradient_value": float(value),
                }
            )

    metric_df = pd.DataFrame(metric_rows)
    similarity_df = pd.DataFrame(sim_rows)
    nodal_df = pd.DataFrame(nodal_rows)
    edge_stats_df = compute_edge_atom_stats(edge_stat_values)
    return metric_df, similarity_df, nodal_df, edge_stats_df


def compute_edge_atom_stats(edge_values: dict[tuple[str, str], list[np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for atom in ATOM_ORDER:
        pla = np.stack(edge_values.get(("pla", atom), []), axis=0)
        psil = np.stack(edge_values.get(("psil", atom), []), axis=0)
        n = pla.shape[1]
        tri = np.triu_indices(n, k=1)
        p_values = []
        raw_rows = []
        for i, j in zip(tri[0], tri[1], strict=True):
            test = stats.ttest_ind(psil[:, i, j], pla[:, i, j], equal_var=False, nan_policy="omit")
            raw_rows.append(
                {
                    "atom": atom,
                    "roi_i": int(i + 1),
                    "roi_j": int(j + 1),
                    "placebo_mean": float(np.nanmean(pla[:, i, j])),
                    "psilocybin_mean": float(np.nanmean(psil[:, i, j])),
                    "delta_mean": float(np.nanmean(psil[:, i, j]) - np.nanmean(pla[:, i, j])),
                    "t": float(test.statistic),
                    "p": float(test.pvalue),
                }
            )
            p_values.append(float(test.pvalue))
        q_values = bh_fdr(np.asarray(p_values, dtype=float))
        for raw, q in zip(raw_rows, q_values, strict=True):
            raw["q"] = float(q)
            rows.append(raw)
    return pd.DataFrame(rows)


def compare_metric_table(df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        base = {col: val for col, val in zip(group_cols, key_tuple)}
        for metric in metric_cols:
            pla = sub.loc[sub["condition"].eq("pla"), metric].to_numpy(float)
            psil = sub.loc[sub["condition"].eq("psil"), metric].to_numpy(float)
            test = stats.ttest_ind(psil, pla, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "placebo_mean": float(np.nanmean(pla)),
                    "psilocybin_mean": float(np.nanmean(psil)),
                    "delta_mean": float(np.nanmean(psil) - np.nanmean(pla)),
                    "t": float(test.statistic),
                    "p": float(test.pvalue),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_fdr(out["p"].to_numpy(float))
    return out


def compare_nodal_gradients(nodal_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for roi_idx, sub in nodal_df.groupby("roi_index"):
        pla = sub.loc[sub["condition"].eq("pla"), "gradient_value"].to_numpy(float)
        psil = sub.loc[sub["condition"].eq("psil"), "gradient_value"].to_numpy(float)
        test = stats.ttest_ind(psil, pla, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "roi_index": int(roi_idx),
                "roi_name": str(sub["roi_name"].iloc[0]),
                "placebo_mean": float(np.nanmean(pla)),
                "psilocybin_mean": float(np.nanmean(psil)),
                "delta_mean": float(np.nanmean(psil) - np.nanmean(pla)),
                "t": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"].to_numpy(float))
    return out


def condition_pair_matrices(averages_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for row in averages_df.to_dict(orient="records"):
        cond = str(row["cohort"])
        atom = str(row["atom"])
        out.setdefault(cond, {})[atom] = np.asarray(row["matrix"], dtype=float)
    return out


def plot_mean_atom_matrices(
    matrices: dict[str, dict[str, np.ndarray]],
    fig_dir: Path,
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6), constrained_layout=True)
    for row_idx, atom in enumerate(ATOM_ORDER):
        vals = [matrices[cond][atom] for cond in CONDITION_ORDER]
        vmax = max(float(np.nanpercentile(v, 99)) for v in vals)
        cmap = RTR_CMAP if atom == "rtr" else STS_CMAP
        for col_idx, cond in enumerate(CONDITION_ORDER):
            mat = matrices[cond][atom].copy()
            np.fill_diagonal(mat, np.nan)
            im = axes[row_idx, col_idx].imshow(mat, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
            axes[row_idx, col_idx].set_title(f"{condition_label(cond)} {ATOM_LABEL[atom]}")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
        cbar = fig.colorbar(im, ax=axes[row_idx, :].ravel().tolist(), fraction=0.025, pad=0.01)
        cbar.set_label("High redundancy" if atom == "rtr" else "High synergy")
    save_figure(fig, fig_dir, "fig01_condition_mean_sts_rtr_matrices")


def plot_edge_atom_deltas(
    matrices: dict[str, dict[str, np.ndarray]],
    edge_stats: pd.DataFrame,
    fig_dir: Path,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    for ax, atom in zip(axes, ATOM_ORDER, strict=True):
        delta = matrices["psil"][atom] - matrices["pla"][atom]
        np.fill_diagonal(delta, np.nan)
        vmax = float(np.nanmax(np.abs(delta)))
        im = ax.imshow(delta, origin="lower", cmap=GRADIENT_CMAP, vmin=-vmax, vmax=vmax)
        sig = np.zeros_like(delta, dtype=bool)
        sub = edge_stats[(edge_stats["atom"].eq(atom)) & (edge_stats["q"] < 0.05)]
        for row in sub.to_dict(orient="records"):
            i = int(row["roi_i"]) - 1
            j = int(row["roi_j"]) - 1
            sig[i, j] = True
            sig[j, i] = True
        if np.any(sig):
            ax.contour(sig.astype(float), levels=[0.5], colors="#1B1B1B", linewidths=0.35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ATOM_LABEL[atom]}: psilocybin - placebo")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Atom difference")
    cbar.set_ticks([im.norm.vmin, 0, im.norm.vmax])
    cbar.set_ticklabels(["more placebo", "0", "more psilocybin"])
    save_figure(fig, fig_dir, "fig02_condition_sts_rtr_delta_with_fdr_edges")


def plot_rank_gradients(
    matrices: dict[str, dict[str, np.ndarray]],
    nodal_stats: pd.DataFrame,
    roi_names: list[str],
    fig_dir: Path,
) -> None:
    setup_style()
    nodal_rows = []
    edge_mats = []
    for cond in CONDITION_ORDER:
        nodal = redundancy_synergy_rank_gradient(matrices[cond]["sts"], matrices[cond]["rtr"])
        edge = edge_rank_gradient(matrices[cond]["sts"], matrices[cond]["rtr"])
        nodal_rows.append(nodal)
        edge_mats.append(edge)
    nodal_arr = np.vstack(nodal_rows)
    sort_idx = np.argsort(np.nanmean(nodal_arr, axis=0))
    nodal_arr = nodal_arr[:, sort_idx]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1, 1]})
    vmax = float(np.nanmax(np.abs(nodal_arr)))
    im0 = axes[0].imshow(nodal_arr, aspect="auto", cmap=GRADIENT_CMAP, vmin=-vmax, vmax=vmax)
    axes[0].set_yticks([0, 1], [condition_label(c) for c in CONDITION_ORDER])
    tick_idx = np.arange(0, len(sort_idx), 8, dtype=int)
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels([roi_names[sort_idx[i]] for i in tick_idx], rotation=90)
    axes[0].set_title("Nodal STS-RTR rank gradient")
    sig_lookup = nodal_stats.set_index("roi_index")["q"].to_dict()
    for x_pos, roi_zero in enumerate(sort_idx):
        q = sig_lookup.get(int(roi_zero + 1), np.nan)
        if np.isfinite(q) and q < 0.05:
            axes[0].text(x_pos, -0.38, "*", ha="center", va="center", fontsize=6, color="#1B1B1B", clip_on=False)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.015)
    cb0.set_label("Synergy rank - redundancy rank")
    cb0.set_ticks([-vmax, 0, vmax])
    cb0.set_ticklabels(["high redundancy", "0", "high synergy"])

    vmax_edge = max(float(np.nanmax(np.abs(e))) for e in edge_mats)
    for ax, cond, edge in zip(axes[1:], CONDITION_ORDER, edge_mats, strict=True):
        mat = edge[np.ix_(sort_idx, sort_idx)].copy()
        np.fill_diagonal(mat, np.nan)
        im = ax.imshow(mat, origin="lower", cmap=GRADIENT_CMAP, vmin=-vmax_edge, vmax=vmax_edge)
        ax.set_title(f"{condition_label(cond)} edge gradient")
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[1:].ravel().tolist(), fraction=0.035, pad=0.015)
    cb.set_label("Edge rank gradient")
    cb.set_ticks([-vmax_edge, 0, vmax_edge])
    cb.set_ticklabels(["high redundancy", "0", "high synergy"])
    save_figure(fig, fig_dir, "fig03_condition_synergy_redundancy_rank_gradients")


def plot_metric_violins(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    value_cols: list[tuple[str, str]],
    fig_dir: Path,
    stem: str,
    title: str,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, len(value_cols), figsize=(3.4 * len(value_cols), 3.0), constrained_layout=True)
    if len(value_cols) == 1:
        axes = np.array([axes])
    for ax, (metric, ylabel) in zip(axes, value_cols, strict=True):
        positions = []
        values = []
        colors = []
        labels = []
        pos = 0
        for atom in ATOM_ORDER:
            for cond in CONDITION_ORDER:
                sub = df[(df["atom"].eq(atom)) & (df["condition"].eq(cond))]
                values.append(sub[metric].to_numpy(float))
                positions.append(pos)
                colors.append(ATOM_COLOR[atom])
                labels.append(f"{ATOM_LABEL[atom].split()[0]}\n{condition_label(cond)}")
                pos += 1
            pos += 0.45
        parts = ax.violinplot(values, positions=positions, widths=0.78, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors, strict=True):
            body.set_facecolor(color)
            body.set_edgecolor("#1B1B1B")
            body.set_alpha(0.62)
        rng = np.random.default_rng(808)
        for x, y, color in zip(positions, values, colors, strict=True):
            ax.scatter(x + rng.normal(0, 0.045, size=y.size), y, s=12, color=color, edgecolor="white", lw=0.35, zorder=3)
        for atom in ATOM_ORDER:
            row = stats_df[(stats_df["atom"].eq(atom)) & (stats_df["metric"].eq(metric))]
            if row.empty:
                continue
            q = float(row.iloc[0]["q"])
            atom_positions = [positions[i] for i, label in enumerate(labels) if label.startswith(ATOM_LABEL[atom].split()[0])]
            if len(atom_positions) == 2:
                yfinite = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
                y = float(np.nanmax(yfinite)) + 0.08 * max(float(np.nanmax(yfinite) - np.nanmin(yfinite)), 1e-6)
                ax.plot(atom_positions, [y, y], color="#1B1B1B", lw=0.8)
                ax.text(np.mean(atom_positions), y, f"{p_to_stars(q)} q={q:.2g}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(positions, labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color="#DED6C8", lw=0.6)
    fig.suptitle(title, y=1.05, fontsize=10)
    save_figure(fig, fig_dir, stem)


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Maas psilocybin PhiID/Luppi-style analysis

This folder contains the true STS/RTR downstream analysis for the Maas
between-subject psilocybin dataset.

Important distinction:

- STS/RTR rank gradients here are computed from PhiID atom matrices.
- Positive gradient means higher synergy rank than redundancy rank and is shown in red.
- Negative gradient means higher redundancy rank than synergy rank and is shown in green.
- The FC-only topology outputs in `results/maas_psilbetween_fc_brain_states` are not STS/RTR gradients.

Current status: `{summary["status"]}`.

To run the expensive PhiID computation:

```bash
python scripts/maas_psilbetween_phiid_luppi.py --run-matlab --matlab-parallel --matlab-workers 8
```

Primary figures, once PhiID outputs are available:

- `fig01_condition_mean_sts_rtr_matrices`: mean RTR and STS atom matrices per condition.
- `fig02_condition_sts_rtr_delta_with_fdr_edges`: psilocybin-placebo atom differences with FDR-significant edges outlined.
- `fig03_condition_synergy_redundancy_rank_gradients`: nodal and edge synergy-minus-redundancy rank gradients.
- `fig04_subject_fc_similarity_to_atoms`: subject FC similarity to RTR/STS.
- `fig05_subject_phiid_graph_metrics`: graph metrics of RTR/STS atom matrices.
"""
    (out_dir / "README.md").write_text(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.expanduser().resolve()
    input_dir = out_dir / "inputs"
    phiid_dir = out_dir / "phiid" / args.redundancy
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    avg_dir = out_dir / "averages" / args.redundancy
    log_dir = out_dir / "logs"
    for path in [input_dir, phiid_dir, table_dir, fig_dir, avg_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    records, _, roi_names = build_records(args.data_file, args.metadata_file, args.roi_names_file)
    manifest = export_phiid_subject_inputs(
        records,
        input_dir,
        roi_labels=roi_names,
        max_timepoints=args.max_timepoints,
        standardize=args.standardize,
        tr_seconds=args.tr_seconds,
    )
    manifest["condition"] = manifest["cohort"]
    manifest_path = input_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    matlab_cmd = build_matlab_batch_command(
        input_dir=input_dir,
        output_dir=phiid_dir,
        redundancy=args.redundancy,
        matlab_bin=args.matlab_bin,
        matlab_toolbox_root=args.matlab_toolbox_root,
        runner_path=args.matlab_runner,
        use_parallel=args.matlab_parallel,
        n_workers=args.matlab_workers,
    )
    (log_dir / "matlab_command.txt").write_text(matlab_cmd + "\n")

    if args.run_matlab:
        subprocess.run(matlab_cmd, shell=True, cwd=_REPO_ROOT, check=True)

    index_df = load_phiid_index(phiid_dir, manifest_path=manifest_path)
    if index_df.empty:
        summary = {
            "status": "exported_inputs_only",
            "out_dir": str(out_dir),
            "n_subjects_exported": int(manifest.shape[0]),
            "n_phiid_outputs_indexed": 0,
            "matlab_command": matlab_cmd,
        }
        (log_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        write_readme(out_dir, summary)
        return summary

    index_df.to_csv(log_dir / "phiid_output_index.csv", index=False)
    condition_avgs = pd.concat(
        [average_atom_matrices_by_group(index_df, atom=atom, group_cols=["cohort"]) for atom in ATOM_ORDER],
        ignore_index=True,
    )
    condition_avgs.to_pickle(avg_dir / "condition_averages.pkl")
    save_group_average_outputs(condition_avgs, avg_dir / "by_condition")

    metric_df, similarity_df, nodal_df, edge_stats_df = compute_subject_tables(index_df, records, roi_names)
    metric_stats_df = compare_metric_table(metric_df, ["global_efficiency", "modularity"], ["atom"])
    similarity_stats_df = compare_metric_table(similarity_df, ["fc_similarity_rho"], ["atom"])
    nodal_stats_df = compare_nodal_gradients(nodal_df)

    metric_df.to_csv(table_dir / "subject_phiid_graph_metrics.csv", index=False)
    metric_stats_df.to_csv(table_dir / "subject_phiid_graph_metric_stats.csv", index=False)
    similarity_df.to_csv(table_dir / "subject_fc_similarity_to_atoms.csv", index=False)
    similarity_stats_df.to_csv(table_dir / "subject_fc_similarity_to_atoms_stats.csv", index=False)
    nodal_df.to_csv(table_dir / "subject_synergy_redundancy_nodal_gradients.csv", index=False)
    nodal_stats_df.to_csv(table_dir / "subject_synergy_redundancy_nodal_gradient_stats.csv", index=False)
    edge_stats_df.to_csv(table_dir / "edgewise_sts_rtr_condition_stats.csv", index=False)

    matrices = condition_pair_matrices(condition_avgs)
    plot_mean_atom_matrices(matrices, fig_dir)
    plot_edge_atom_deltas(matrices, edge_stats_df, fig_dir)
    plot_rank_gradients(matrices, nodal_stats_df, roi_names, fig_dir)
    plot_metric_violins(
        similarity_df,
        similarity_stats_df,
        value_cols=[("fc_similarity_rho", "Spearman rho")],
        fig_dir=fig_dir,
        stem="fig04_subject_fc_similarity_to_atoms",
        title="Subject FC similarity to PhiID atom matrices",
    )
    plot_metric_violins(
        metric_df,
        metric_stats_df,
        value_cols=[("global_efficiency", "Global efficiency"), ("modularity", "Modularity")],
        fig_dir=fig_dir,
        stem="fig05_subject_phiid_graph_metrics",
        title="Subject PhiID atom graph metrics",
    )

    summary = {
        "status": "complete",
        "out_dir": str(out_dir),
        "n_subjects_exported": int(manifest.shape[0]),
        "n_phiid_outputs_indexed": int(index_df.shape[0]),
        "redundancy": args.redundancy,
        "matlab_command": matlab_cmd,
    }
    (log_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(out_dir, summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=PSIL_FILE)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)
    parser.add_argument("--roi-names-file", type=Path, default=ROI_NAMES_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--redundancy", type=str, default="mmi")
    parser.add_argument("--standardize", type=str, default=None)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--tr-seconds", type=float, default=2.4)
    parser.add_argument("--run-matlab", action="store_true")
    parser.add_argument("--matlab-parallel", action="store_true")
    parser.add_argument("--matlab-workers", type=int, default=0)
    parser.add_argument("--matlab-bin", type=str, default="/Applications/MATLAB_R2023b.app/bin/matlab")
    parser.add_argument("--matlab-toolbox-root", type=str, default="/Users/borjan/code/matlab/elph")
    parser.add_argument("--matlab-runner", type=Path, default=_REPO_ROOT / "scripts" / "phiid_empirical_bold_aal90.m")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
