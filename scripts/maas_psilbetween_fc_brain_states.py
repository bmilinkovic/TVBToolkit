#!/usr/bin/env python3
"""Publication-ready FC and brain-state analysis for Maas psilocybin between-subject data.

This workflow intentionally uses only the non-corrupted between-subject file:
``data/drugs_data/maas_psilbetween_ts_aal116_noGSR.mat``.

Outputs
-------
results/maas_psilbetween_fc_brain_states/
    tables/   CSV summaries and statistical results
    figures/  PDF/SVG/PNG publication figures
    npz/      Compact numerical arrays for reuse
    logs/     JSON run manifest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

try:
    from tvbtoolkit.analysis.brain_states import (
        _compute_occupancy,
        _compute_transition_matrix,
        cluster_brain_states,
        phase_patterns,
    )
    from tvbtoolkit.analysis.luppi2022 import (
        edge_rank_gradient,
        nodal_strength,
        summarize_within_between,
        threshold_top_density,
        weighted_global_efficiency,
        weighted_modularity,
    )
except ModuleNotFoundError:  # pragma: no cover
    src = _REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tvbtoolkit.analysis.brain_states import (
        _compute_occupancy,
        _compute_transition_matrix,
        cluster_brain_states,
        phase_patterns,
    )
    from tvbtoolkit.analysis.luppi2022 import (
        edge_rank_gradient,
        nodal_strength,
        summarize_within_between,
        threshold_top_density,
        weighted_global_efficiency,
        weighted_modularity,
    )


DATA_DIR = _REPO_ROOT / "data" / "drugs_data"
DEFAULT_OUT = _REPO_ROOT / "results" / "maas_psilbetween_fc_brain_states"
PSIL_FILE = DATA_DIR / "maas_psilbetween_ts_aal116_noGSR.mat"
METADATA_FILE = DATA_DIR / "metadata.xlsx"
ROI_NAMES_FILE = DATA_DIR / "aal116NodeNames.txt"

CONDITION_ORDER = ["pla", "psil"]
CONDITION_LABEL = {"pla": "Placebo", "psil": "Psilocybin"}

# Wes Anderson-inspired "Zissou" palette.
# Source palette family: sea blue, pale blue, yellow, ochre, red-orange.
WES_ZISSOU = {
    "deep_blue": "#3B9AB2",
    "pale_blue": "#78B7C5",
    "sun_yellow": "#EBCC2A",
    "ochre": "#E1AF00",
    "red_orange": "#F21A00",
    "cream": "#F7F3E8",
    "ink": "#2B2B2B",
    "muted": "#8A8F8D",
}
CONDITION_COLOR = {"pla": WES_ZISSOU["deep_blue"], "psil": WES_ZISSOU["ochre"]}
SIGNIFICANT_COLOR = WES_ZISSOU["red_orange"]
NEUTRAL_COLOR = WES_ZISSOU["muted"]
COMPARISON_LABEL = "more placebo  <-  psilocybin - placebo  ->  more psilocybin"
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "wes_placebo_psilocybin_diff",
    [CONDITION_COLOR["pla"], WES_ZISSOU["cream"], CONDITION_COLOR["psil"]],
)


@dataclass(frozen=True)
class ScanRecord:
    subject: str
    condition: str
    timeseries: np.ndarray  # regions x time


def decode_hdf5_cell(file: h5py.File, ref):
    """Decode a MATLAB v7.3 cell reference as a string or numeric array."""
    arr = file[ref][()]
    if arr.dtype == np.uint16:
        return "".join(map(chr, arr.flatten(order="F"))).strip("\x00")
    return np.asarray(arr, dtype=float)


def load_between_subject_scans(path: Path = PSIL_FILE) -> list[ScanRecord]:
    """Load the between-subject psilocybin AAL116 time series."""
    scans: list[ScanRecord] = []
    with h5py.File(path, "r") as file:
        cell = file["roi_data"]
        subjects = [decode_hdf5_cell(file, cell[0, col]) for col in range(1, cell.shape[1])]
        conditions = [decode_hdf5_cell(file, cell[1, col]) for col in range(1, cell.shape[1])]
        for col, (subject, condition) in enumerate(zip(subjects, conditions), start=1):
            ts = decode_hdf5_cell(file, cell[2, col])
            if ts.shape[0] != 116 and ts.shape[1] == 116:
                ts = ts.T
            scans.append(ScanRecord(subject=str(int(subject)), condition=str(condition), timeseries=ts))
    return scans


def load_metadata(path: Path = METADATA_FILE) -> pd.DataFrame:
    """Load and normalize the psilocybin between-subject metadata sheet."""
    meta = pd.read_excel(path, sheet_name="psil_between ").copy()
    meta = meta.rename(columns={"sub": "subject", "drug": "condition", "emax_psilocin_pk_ngml": "emax_pk_ng_ml"})
    meta["subject"] = meta["subject"].astype(int).astype(str)
    meta["condition"] = meta["condition"].astype(str)
    return meta


def load_roi_names(path: Path = ROI_NAMES_FILE) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(names) != 116:
        raise ValueError(f"Expected 116 ROI names, got {len(names)} from {path}")
    return names


def macro_system(name: str) -> str:
    """Small AAL116 macro-system map for block-level summaries."""
    base = name.replace("_L", "").replace("_R", "")
    if "Cerebelum" in base or "Vermis" in base:
        return "Cerebellum"
    if any(k in base for k in ["Caudate", "Putamen", "Pallidum", "Thalamus"]):
        return "Subcortical"
    if any(k in base for k in ["Hippocampus", "ParaHippocampal", "Amygdala", "Cingulum", "Olfactory", "Insula"]):
        return "Limbic"
    if "Temporal" in base or "Heschl" in base:
        return "Temporal"
    if "Occipital" in base or "Calcarine" in base or "Cuneus" in base or "Lingual" in base or "Fusiform" in base:
        return "Occipital"
    if "Parietal" in base or "Precuneus" in base or "SupraMarginal" in base or "Angular" in base or "Postcentral" in base:
        return "Parietal"
    return "Frontal"


def fisher_z(fc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.arctanh(np.clip(fc, -1.0 + eps, 1.0 - eps))


def fc_matrix(ts_regions_time: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts_regions_time, dtype=float)
    finite_var = np.nanstd(ts, axis=1) > 0
    fc = np.full((ts.shape[0], ts.shape[0]), np.nan, dtype=float)
    fc_sub = np.corrcoef(ts[finite_var])
    idx = np.flatnonzero(finite_var)
    fc[np.ix_(idx, idx)] = fc_sub
    np.fill_diagonal(fc, 1.0)
    return fc


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return q
    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(ranked.size)
    adj = ranked * m / (np.arange(ranked.size) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(adj, 0.0, 1.0)
    q[finite] = out
    return q


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def label_directional_colorbar(cbar, *, low_label: str = "more placebo", high_label: str = "more psilocybin", label: str = "Difference") -> None:
    ticks = cbar.get_ticks()
    if len(ticks) >= 2:
        vmin = float(ticks[0])
        vmax = float(ticks[-1])
    else:
        vmin, vmax = -1.0, 1.0
    cbar.set_label(label)
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_ticklabels([low_label, "0", high_label])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": WES_ZISSOU["cream"],
            "axes.edgecolor": WES_ZISSOU["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D0B8",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.45,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def condition_arrays(df: pd.DataFrame, value_col: str) -> list[np.ndarray]:
    return [df.loc[df["condition"].eq(cond), value_col].to_numpy(dtype=float) for cond in CONDITION_ORDER]


def draw_significance(ax: plt.Axes, x0: float, x1: float, y: float, text: str) -> None:
    ax.plot([x0, x0, x1, x1], [y, y * 1.015, y * 1.015, y], color=WES_ZISSOU["ink"], lw=0.8)
    ax.text((x0 + x1) / 2.0, y * 1.02, text, ha="center", va="bottom", fontsize=7)


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


def write_analysis_guide(out_dir: Path, args: argparse.Namespace) -> None:
    """Write a plain-language guide describing every analysis and figure."""
    text = f"""# Maas Psilocybin Between-Subject Analysis Guide

This folder contains the functional-connectivity and brain-state analysis for
the usable Maas psilocybin between-subject dataset.

## Why Only This Dataset Is Used

The within-subject psilocybin and THC `.mat` files were excluded because their
stored condition time series are duplicated within each subject. In those files,
subtracting placebo and drug arrays element by element gives a maximum absolute
difference of exactly `0.0`, so they cannot support drug-condition contrasts.

This analysis uses only:

`{args.data_file}`

## Input Data

- Design: between-subject placebo vs psilocybin.
- Scans: 54 total, 29 placebo and 25 psilocybin.
- Parcellation: AAL116.
- Time series: no-GSR regional BOLD time series.
- Metadata: dose, weight, psilocin Emax, scan completion, scrub percentage,
  and mean framewise displacement from `{args.metadata_file}`.

## Functional Connectivity Analysis

For each scan, the script computes a 116 x 116 Pearson correlation matrix across
regional BOLD time series. This is the static functional-connectivity matrix.

The analysis then computes:

1. **Subject-level FC summaries**
   - `mean_abs_fc`: average absolute FC across all 6,670 upper-triangle edges.
   - `mean_signed_fc`: average signed FC across all edges.
   - `mean_fisher_z`: average Fisher-z transformed FC across all edges.
   - Placebo and psilocybin groups are compared with Welch's t-test and a
     Mann-Whitney U test.

2. **Group-average FC matrices**
   - Mean placebo FC matrix.
   - Mean psilocybin FC matrix.
   - Difference matrix: psilocybin minus placebo.

3. **Edgewise FC statistics**
   - Each of the 6,670 AAL116 edges is tested with a Welch t-test on
     Fisher-z transformed FC values.
   - Benjamini-Hochberg FDR correction is applied across all edges.

4. **Macro-system summaries**
   - AAL regions are grouped into broad systems: Frontal, Limbic, Temporal,
     Parietal, Occipital, Subcortical, and Cerebellum.
   - The script summarizes mean edge effects and number of FDR-significant
     edges for each system-to-system block.
   - A scan-level macro-block table is also tested with Welch tests and FDR
     correction; significant blocks are marked in the macro-system figure.

5. **FC topology adaptation inspired by the Luppi downstream summaries**
   - The Luppi et al. code in this repository is primarily designed for PhiID
     synergy (`sts`) and redundancy (`rtr`) matrices.
   - Because this drug dataset does not yet have PhiID atom matrices, this
     section is explicitly an FC-only adaptation using positive FC matrices.
   - For each scan, positive FC is thresholded to top-edge densities of 10%,
     15%, and 20%.
   - The script computes weighted global efficiency, modularity, and
     within-minus-between macro-system connectivity.
   - It also computes drug-minus-placebo nodal and edge rank gradients from
     group-average positive FC matrices.
   - These outputs are labeled `luppi_style_fc_*` for continuity with the first
     draft, but they are FC topology/rank summaries, not STS/RTR
     synergy-minus-redundancy gradients.

## Brain-State Analysis

The brain-state analysis uses phase-coherence pattern clustering.

For each scan:

1. Regional BOLD time series are preprocessed with the TVBToolkit
   `brain_act_legacy` pipeline:
   - ROI-wise z-scoring.
   - ROI-mean removal at each time point.
   - Butterworth band-pass filtering from {args.bandpass_low_hz} to
     {args.bandpass_high_hz} Hz.
   - Hilbert transform to estimate instantaneous phase.

2. At each time point, the script computes cosine phase differences between
   every pair of AAL116 regions. Each time point is therefore represented as a
   6,670-edge phase-coherence pattern.

3. Phase-coherence patterns from all scans are pooled and clustered with
   k-means into `{args.n_states}` shared states.

4. States are sorted by increasing mean centroid phase coherence. In plain
   language, State 1 is the lowest mean phase-coherence pattern and State
   `{args.n_states}` is the highest mean phase-coherence pattern.

5. The pooled labels are split back into individual scans to calculate:
   - State occupancy: fraction of time spent in each state.
   - Dwell length: how long consecutive visits to the same state last.
   - No-self transition matrix: transition probabilities after collapsing
     repeated self-runs and excluding self-transitions.
   - Global synchrony: average instantaneous phase synchrony over time.

## Figure Guide

### `fig01_qc_metadata`

Shows motion and acquisition-related metadata by group:

- Mean framewise displacement.
- Percentage of scrubbed volumes.
- Psilocin Emax. Placebo is expected to be zero.

Purpose: verify that group-level effects are interpreted alongside motion and
quality-control variables.

### `fig02_fc_group_heatmaps`

Three matrices:

- Placebo mean FC.
- Psilocybin mean FC.
- Psilocybin minus placebo FC.

The white block boundaries mark broad macro-system groupings. This figure gives
the main whole-brain visual overview of FC organization and drug-associated
changes.

The difference panel uses a directional colorbar:

- negative values mean stronger FC in placebo;
- positive values mean stronger FC in psilocybin.

Red-orange contours mark edges with FDR q < 0.05 in the edgewise FC analysis.

### `fig03_fc_subject_summary`

Subject-level violin/point plots for:

- Mean absolute FC.
- Mean signed FC.

Purpose: show whether whole-brain FC summaries differ between groups and how
individual scans are distributed.

### `fig04_fc_edgewise_volcano`

Each point is one AAL116 edge.

- x-axis: psilocybin minus placebo Fisher-z FC effect.
- y-axis: statistical evidence, `-log10(p)`.
- Red-orange points are FDR-significant at q < 0.05.

Purpose: show which effects survive edgewise multiple-comparison correction.

### `fig05_fc_macro_system_delta`

System-to-system heatmap of mean FC effects.

Purpose: reduce 6,670 edgewise effects into interpretable anatomical blocks.

The colorbar is directional: negative values indicate more placebo-like FC and
positive values indicate more psilocybin-like FC. Asterisks mark macro-system
blocks that survive FDR q < 0.05 in scan-level Welch tests.

### `fig06_brain_state_centroids`

The five pooled phase-coherence state centroids.

Purpose: show the recurrent whole-brain phase-coherence patterns discovered by
k-means.

### `fig07_brain_state_occupancy`

Violin/point plots of state occupancy by condition.

The title annotations use FDR-corrected q-values across the five state-occupancy
tests.

Purpose: show whether psilocybin changes the fraction of time spent in each
recurrent brain state.

### `fig08_brain_state_transitions`

Mean no-self transition matrices for:

- Placebo.
- Psilocybin.
- Psilocybin minus placebo.

Purpose: show whether the dynamics of switching between states differs between
groups.

The difference panel is directional: negative values indicate transition
probabilities that are higher in placebo and positive values indicate transition
probabilities that are higher in psilocybin. Red-orange asterisks mark
transition cells with FDR q < 0.05.

### `fig09_brain_state_dwell_synchrony`

Subject-level distributions for:

- Mean dwell length in samples.
- Mean global synchrony.

Purpose: summarize temporal persistence and overall phase synchrony.

### `fig10_luppi_style_fc_topology`

FC graph-topology adaptation using positive FC thresholded at the top
15% edge density.

It shows:

- global efficiency;
- modularity;
- within-minus-between macro-system FC.

All panels include FDR-corrected q-value annotations across the three topology
tests at the displayed density.

### `fig11_luppi_style_fc_rank_gradients`

FC-only rank-gradient adaptation using group-average positive FC.

- The node panel shows regional strength rank under psilocybin minus regional
  strength rank under placebo.
- The edge panel shows the same rank-gradient idea for edges.

Blue means the node or edge is more highly ranked in placebo. Ochre/brown means
it is more highly ranked in psilocybin.

This is an FC topology adaptation, not a true synergy-minus-redundancy PhiID
gradient. In the true PhiID workflow, high synergy is red, high redundancy is
green, and the gradient is computed as STS rank minus RTR rank.

## Output Tables

- `scan_level_fc_and_metadata.csv`: one row per scan with metadata and FC
  summaries.
- `scan_level_fc_brain_state_metrics.csv`: one row per scan with FC and
  brain-state summaries.
- `fc_subject_summary_stats.csv`: group comparisons for whole-brain FC metrics.
- `fc_edgewise_welch_fdr.csv`: all edgewise FC tests and FDR q-values.
- `fc_macro_system_edge_summary.csv`: macro-system edge summaries.
- `fc_macro_system_scan_long.csv`: scan-level macro-system FC values used for
  macro-block significance tests.
- `brain_state_occupancy_long.csv`: subject x state occupancy table.
- `brain_state_occupancy_stats.csv`: group comparisons for each state's
  occupancy, including FDR q-values.
- `brain_state_transition_long.csv`: long-form transition probabilities.
- `brain_state_transition_stats.csv`: group tests and FDR q-values for every
  transition cell.
- `brain_state_scan_metric_stats.csv`: group comparisons for dwell and synchrony
  metrics.
- `brain_state_reference.csv`: state ordering and centroid coherence summaries.
- `luppi_style_fc_topology_scan_long.csv`: density-thresholded graph metrics
  for each scan.
- `luppi_style_fc_topology_stats.csv`: group tests and FDR q-values for the
  FC topology metrics.
- `luppi_style_fc_nodal_rank_gradient.csv`: drug-minus-placebo nodal rank
  gradient.

## Color Palette

Figures use a Wes Anderson-inspired Zissou palette:

- Placebo: deep sea blue `{CONDITION_COLOR['pla']}`.
- Psilocybin: ochre `{CONDITION_COLOR['psil']}`.
- FDR-significant edges: red-orange `{SIGNIFICANT_COLOR}`.
- Background: warm cream `{WES_ZISSOU['cream']}`.

## Reproducibility

Run from the repository root:

```bash
python scripts/maas_psilbetween_fc_brain_states.py
```

The exact run configuration is also stored in `logs/run_manifest.json`.
"""
    (out_dir / "ANALYSIS_GUIDE.md").write_text(text)


def plot_qc_summary(scan_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    metrics = [("fd_mean", "Mean FD"), ("scrub_perc", "Scrubbed volumes (%)"), ("emax_pk_ng_ml", "Psilocin Emax (ng/mL)")]
    for ax, (col, label) in zip(axes, metrics):
        data = condition_arrays(scan_df, col)
        parts = ax.violinplot(data, positions=[0, 1], widths=0.72, showmeans=False, showextrema=False)
        for body, cond in zip(parts["bodies"], CONDITION_ORDER):
            body.set_facecolor(CONDITION_COLOR[cond])
            body.set_edgecolor(WES_ZISSOU["ink"])
            body.set_alpha(0.62)
        for i, cond in enumerate(CONDITION_ORDER):
            y = scan_df.loc[scan_df["condition"].eq(cond), col].to_numpy(dtype=float)
            rng = np.random.default_rng(100 + i)
            ax.scatter(i + rng.normal(0, 0.045, size=y.size), y, s=15, color=CONDITION_COLOR[cond], edgecolor="white", lw=0.45, zorder=3)
        ax.set_xticks([0, 1], [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
    save_figure(fig, fig_dir, "fig01_qc_metadata")


def plot_fc_heatmaps(
    mean_fc: dict[str, np.ndarray],
    delta_fc: np.ndarray,
    roi_names: list[str],
    fig_dir: Path,
    *,
    edge_q_matrix: np.ndarray | None = None,
) -> None:
    macros = np.array([macro_system(n) for n in roi_names])
    macro_order = ["Frontal", "Limbic", "Temporal", "Parietal", "Occipital", "Subcortical", "Cerebellum"]
    order = np.argsort([macro_order.index(m) if m in macro_order else 99 for m in macros])
    bounds = []
    ordered_macros = macros[order]
    for macro in macro_order:
        idx = np.flatnonzero(ordered_macros == macro)
        if idx.size:
            bounds.append((idx[0], idx[-1] + 1, macro))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), constrained_layout=True)
    mats = [mean_fc["pla"], mean_fc["psil"], delta_fc]
    titles = ["Placebo mean FC", "Psilocybin mean FC", "Psilocybin - placebo"]
    cmaps = ["RdBu_r", "RdBu_r", DIFF_CMAP]
    ranges = [(-0.6, 0.6), (-0.6, 0.6), (-0.16, 0.16)]
    for panel_idx, (ax, mat, title, cmap, (vmin, vmax)) in enumerate(zip(axes, mats, titles, cmaps, ranges)):
        im = ax.imshow(mat[np.ix_(order, order)], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for start, stop, _ in bounds:
            ax.axhline(start - 0.5, color="white", lw=0.45)
            ax.axvline(start - 0.5, color="white", lw=0.45)
            ax.axhline(stop - 0.5, color="white", lw=0.45)
            ax.axvline(stop - 0.5, color="white", lw=0.45)
        if panel_idx == 2 and edge_q_matrix is not None:
            sig = np.asarray(edge_q_matrix[np.ix_(order, order)] < 0.05, dtype=float)
            sig[~np.isfinite(sig)] = 0.0
            ax.contour(sig, levels=[0.5], colors=[SIGNIFICANT_COLOR], linewidths=0.35)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if panel_idx == 2:
            label_directional_colorbar(cbar, label="Delta FC")
    save_figure(fig, fig_dir, "fig02_fc_group_heatmaps")


def plot_fc_summary(scan_df: pd.DataFrame, stats_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7), constrained_layout=True)
    for ax, col, ylabel in zip(axes, ["mean_abs_fc", "mean_signed_fc"], ["Mean |FC|", "Mean signed FC"]):
        data = condition_arrays(scan_df, col)
        parts = ax.violinplot(data, positions=[0, 1], widths=0.72, showmeans=True, showextrema=False)
        for body, cond in zip(parts["bodies"], CONDITION_ORDER):
            body.set_facecolor(CONDITION_COLOR[cond])
            body.set_edgecolor(WES_ZISSOU["ink"])
            body.set_alpha(0.62)
        for i, cond in enumerate(CONDITION_ORDER):
            y = scan_df.loc[scan_df["condition"].eq(cond), col].to_numpy(dtype=float)
            rng = np.random.default_rng(200 + i)
            ax.scatter(i + rng.normal(0, 0.045, size=y.size), y, s=16, color=CONDITION_COLOR[cond], edgecolor="white", lw=0.45, zorder=3)
        row = stats_df.loc[stats_df["metric"].eq(col)].iloc[0]
        ymax = np.nanmax(np.concatenate(data))
        draw_significance(ax, 0, 1, ymax * 1.04, f"{p_to_stars(row['p'])}, p={row['p']:.3g}")
        ax.set_xticks([0, 1], [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, 1.6)
    save_figure(fig, fig_dir, "fig03_fc_subject_summary")


def plot_edgewise(edge_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    x = edge_df["delta_z"].to_numpy(float)
    y = -np.log10(np.clip(edge_df["p"].to_numpy(float), 1e-300, 1.0))
    q = edge_df["q"].to_numpy(float)
    sig = q < 0.05
    ax.scatter(x[~sig], y[~sig], s=5, color=NEUTRAL_COLOR, alpha=0.35, linewidths=0)
    ax.scatter(x[sig], y[sig], s=9, color=SIGNIFICANT_COLOR, alpha=0.78, linewidths=0)
    ax.axvline(0, color=WES_ZISSOU["ink"], lw=0.7)
    ax.axhline(-np.log10(0.05), color=WES_ZISSOU["muted"], lw=0.7, ls="--")
    ax.set_xlabel("Edge effect: Fisher z(FC), psilocybin - placebo")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"Edgewise FC differences ({int(sig.sum())} FDR q<0.05)")
    save_figure(fig, fig_dir, "fig04_fc_edgewise_volcano")


def plot_macro_blocks(block_df: pd.DataFrame, fig_dir: Path) -> None:
    systems = ["Frontal", "Limbic", "Temporal", "Parietal", "Occipital", "Subcortical", "Cerebellum"]
    mat = np.full((len(systems), len(systems)), np.nan)
    for _, row in block_df.iterrows():
        i = systems.index(row["system_i"])
        j = systems.index(row["system_j"])
        mat[i, j] = mat[j, i] = row["delta_z_mean"]
    sig = np.zeros_like(mat, dtype=bool)
    if "q" in block_df.columns:
        for _, row in block_df.iterrows():
            i = systems.index(row["system_i"])
            j = systems.index(row["system_j"])
            is_sig = bool(np.isfinite(row["q"]) and float(row["q"]) < 0.05)
            sig[i, j] = sig[j, i] = is_sig
    fig, ax = plt.subplots(figsize=(4.1, 3.6), constrained_layout=True)
    im = ax.imshow(mat, cmap=DIFF_CMAP, vmin=-0.08, vmax=0.08)
    ax.set_xticks(np.arange(len(systems)), systems, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(systems)), systems)
    for i in range(len(systems)):
        for j in range(len(systems)):
            val = mat[i, j]
            if np.isfinite(val):
                label = f"{val:.2f}" + ("*" if sig[i, j] else "")
                txt_color = "white" if abs(val) >= 0.07 else WES_ZISSOU["ink"]
                ax.text(j, i, label, ha="center", va="center", fontsize=6, color=txt_color)
    ax.set_title("Macro-system FC effect")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    label_directional_colorbar(cbar, label="Delta Fisher z")
    save_figure(fig, fig_dir, "fig05_fc_macro_system_delta")


def plot_state_centroids(centers: np.ndarray, n_regions: int, fig_dir: Path) -> None:
    n_states = centers.shape[0]
    fig, axes = plt.subplots(1, n_states, figsize=(1.85 * n_states, 2.0), constrained_layout=True)
    if n_states == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        mat = np.zeros((n_regions, n_regions), dtype=float)
        iu, ju = np.triu_indices(n_regions, k=1)
        mat[iu, ju] = centers[k]
        mat[ju, iu] = centers[k]
        np.fill_diagonal(mat, 1.0)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"State {k + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.024, pad=0.015, label="cos(delta phase)")
    save_figure(fig, fig_dir, "fig06_brain_state_centroids")


def plot_state_occupancy(state_df: pd.DataFrame, state_stats: pd.DataFrame, fig_dir: Path) -> None:
    states = sorted(state_df["state"].unique())
    fig, axes = plt.subplots(1, len(states), figsize=(1.65 * len(states), 2.65), sharey=True, constrained_layout=True)
    if len(states) == 1:
        axes = [axes]
    for ax, state in zip(axes, states):
        sub = state_df[state_df["state"].eq(state)]
        data = condition_arrays(sub, "occupancy")
        parts = ax.violinplot(data, positions=[0, 1], widths=0.72, showmeans=True, showextrema=False)
        for body, cond in zip(parts["bodies"], CONDITION_ORDER):
            body.set_facecolor(CONDITION_COLOR[cond])
            body.set_edgecolor(WES_ZISSOU["ink"])
            body.set_alpha(0.62)
        for i, cond in enumerate(CONDITION_ORDER):
            y = sub.loc[sub["condition"].eq(cond), "occupancy"].to_numpy(float)
            rng = np.random.default_rng(300 + state * 10 + i)
            ax.scatter(i + rng.normal(0, 0.045, size=y.size), y, s=15, color=CONDITION_COLOR[cond], edgecolor="white", lw=0.45, zorder=3)
        row = state_stats.loc[state_stats["state"].eq(state)].iloc[0]
        ax.set_title(f"State {state}\n{p_to_stars(row['q'])} q={row['q']:.2g}")
        ax.set_xticks([0, 1], ["Pla", "Psil"], rotation=25, ha="right")
        ax.set_ylim(0, max(0.6, state_df["occupancy"].max() * 1.15))
    axes[0].set_ylabel("Occupancy")
    save_figure(fig, fig_dir, "fig07_brain_state_occupancy")


def plot_transition_matrices(
    transition_mean: dict[str, np.ndarray],
    fig_dir: Path,
    *,
    transition_stats: pd.DataFrame | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.35), constrained_layout=True)
    mats = [transition_mean["pla"], transition_mean["psil"], transition_mean["psil"] - transition_mean["pla"]]
    titles = ["Placebo transitions", "Psilocybin transitions", "Psilocybin - placebo"]
    cmaps = ["YlGnBu", "YlGnBu", DIFF_CMAP]
    ranges = [(0, 0.55), (0, 0.55), (-0.18, 0.18)]
    sig = np.zeros_like(mats[0], dtype=bool)
    if transition_stats is not None and not transition_stats.empty and "q" in transition_stats.columns:
        for _, row in transition_stats.iterrows():
            i = int(row["from_state"]) - 1
            j = int(row["to_state"]) - 1
            if 0 <= i < sig.shape[0] and 0 <= j < sig.shape[1]:
                sig[i, j] = bool(np.isfinite(row["q"]) and float(row["q"]) < 0.05)
    for panel_idx, (ax, mat, title, cmap, (vmin, vmax)) in enumerate(zip(axes, mats, titles, cmaps, ranges)):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")
        ax.set_xticks(np.arange(mat.shape[0]), np.arange(1, mat.shape[0] + 1))
        ax.set_yticks(np.arange(mat.shape[0]), np.arange(1, mat.shape[0] + 1))
        if panel_idx == 2:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if sig[i, j]:
                        ax.text(j, i, "*", ha="center", va="center", color=SIGNIFICANT_COLOR, fontsize=10, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if panel_idx == 2:
            label_directional_colorbar(cbar, label="Delta transition probability")
    save_figure(fig, fig_dir, "fig08_brain_state_transitions")


def plot_dwell_synchrony(scan_df: pd.DataFrame, stats_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.65), constrained_layout=True)
    for ax, col, ylabel in zip(axes, ["mean_dwell_samples", "global_synchrony_mean"], ["Mean dwell length (samples)", "Mean global synchrony"]):
        data = condition_arrays(scan_df, col)
        parts = ax.violinplot(data, positions=[0, 1], widths=0.72, showmeans=True, showextrema=False)
        for body, cond in zip(parts["bodies"], CONDITION_ORDER):
            body.set_facecolor(CONDITION_COLOR[cond])
            body.set_edgecolor(WES_ZISSOU["ink"])
            body.set_alpha(0.62)
        for i, cond in enumerate(CONDITION_ORDER):
            y = scan_df.loc[scan_df["condition"].eq(cond), col].to_numpy(float)
            rng = np.random.default_rng(400 + i)
            ax.scatter(i + rng.normal(0, 0.045, size=y.size), y, s=16, color=CONDITION_COLOR[cond], edgecolor="white", lw=0.45, zorder=3)
        row = stats_df.loc[stats_df["metric"].eq(col)].iloc[0]
        ax.set_title(f"{p_to_stars(row['p'])}, p={row['p']:.3g}")
        ax.set_xticks([0, 1], [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    save_figure(fig, fig_dir, "fig09_brain_state_dwell_synchrony")


def plot_luppi_fc_topology(
    topology_stats: pd.DataFrame,
    fig_dir: Path,
    *,
    density: float,
) -> None:
    sub = topology_stats[np.isclose(topology_stats["density"].astype(float), float(density))].copy()
    metrics = [
        ("global_efficiency", "Global efficiency"),
        ("modularity", "Modularity"),
        ("within_minus_between_macro", "Within - between macro FC"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.4, 2.75), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = [sub.loc[sub["condition"].eq(cond), metric].to_numpy(float) for cond in CONDITION_ORDER]
        parts = ax.violinplot(values, positions=[0, 1], widths=0.72, showmeans=True, showextrema=False)
        for body, cond in zip(parts["bodies"], CONDITION_ORDER):
            body.set_facecolor(CONDITION_COLOR[cond])
            body.set_edgecolor(WES_ZISSOU["ink"])
            body.set_alpha(0.62)
        for i, cond in enumerate(CONDITION_ORDER):
            y = sub.loc[sub["condition"].eq(cond), metric].to_numpy(float)
            rng = np.random.default_rng(600 + i)
            ax.scatter(i + rng.normal(0, 0.045, size=y.size), y, s=15, color=CONDITION_COLOR[cond], edgecolor="white", lw=0.45, zorder=3)
        if {"metric", "p", "q"}.issubset(sub.columns):
            # Stats live in a separate table, so this branch is kept for future joined inputs.
            pass
        ax.set_xticks([0, 1], [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    save_figure(fig, fig_dir, "fig10_luppi_style_fc_topology")


def plot_luppi_fc_topology_with_stats(
    topology_df: pd.DataFrame,
    topology_stats: pd.DataFrame,
    fig_dir: Path,
    *,
    density: float,
) -> None:
    sub = topology_df[np.isclose(topology_df["density"].astype(float), float(density))].copy()
    metrics = [
        ("global_efficiency", "Global efficiency"),
        ("modularity", "Modularity"),
        ("within_minus_between_macro", "Within - between macro FC"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.6, 2.75), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = [sub.loc[sub["condition"].eq(cond), metric].to_numpy(float) for cond in CONDITION_ORDER]
        parts = ax.violinplot(values, positions=[0, 1], widths=0.72, showmeans=True, showextrema=False)
        for body, cond in zip(parts["bodies"], CONDITION_ORDER):
            body.set_facecolor(CONDITION_COLOR[cond])
            body.set_edgecolor(WES_ZISSOU["ink"])
            body.set_alpha(0.62)
        for i, cond in enumerate(CONDITION_ORDER):
            y = sub.loc[sub["condition"].eq(cond), metric].to_numpy(float)
            rng = np.random.default_rng(650 + i)
            ax.scatter(i + rng.normal(0, 0.045, size=y.size), y, s=15, color=CONDITION_COLOR[cond], edgecolor="white", lw=0.45, zorder=3)
        row = topology_stats[
            np.isclose(topology_stats["density"].astype(float), float(density))
            & topology_stats["metric"].eq(metric)
        ]
        if not row.empty:
            q = float(row.iloc[0]["q"])
            ymax = np.nanmax(np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()]))
            ymin = np.nanmin(np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()]))
            y = ymax + 0.08 * max(ymax - ymin, 1e-6)
            draw_significance(ax, 0, 1, y, f"{p_to_stars(q)} q={q:.2g}")
            ax.set_ylim(top=y + 0.13 * max(ymax - ymin, 1e-6))
        ax.set_xticks([0, 1], [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    fig.suptitle(f"FC topology adaptation at top {density:.0%} positive-edge density", y=1.08, fontsize=10)
    save_figure(fig, fig_dir, "fig10_luppi_style_fc_topology")


def plot_luppi_rank_gradients(
    nodal_gradient_df: pd.DataFrame,
    edge_gradient: np.ndarray,
    roi_names: list[str],
    fig_dir: Path,
) -> None:
    systems = ["Frontal", "Limbic", "Temporal", "Parietal", "Occipital", "Subcortical", "Cerebellum"]
    macros = np.array([macro_system(n) for n in roi_names])
    order = np.argsort([systems.index(m) if m in systems else 99 for m in macros])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), constrained_layout=True)
    ordered = nodal_gradient_df.iloc[order].reset_index(drop=True)
    colors = [CONDITION_COLOR["psil"] if v > 0 else CONDITION_COLOR["pla"] for v in ordered["rank_gradient_psil_minus_placebo"]]
    axes[0].bar(np.arange(ordered.shape[0]), ordered["rank_gradient_psil_minus_placebo"], color=colors, width=0.85)
    axes[0].axhline(0, color=WES_ZISSOU["ink"], lw=0.8)
    axes[0].set_xticks([])
    axes[0].set_ylabel("Positive-FC strength rank gradient")
    axes[0].set_title("Nodes: psilocybin rank - placebo rank")
    axes[0].text(0.01, 0.98, "blue: more placebo-ranked\nochre: more psilocybin-ranked", transform=axes[0].transAxes, va="top", fontsize=6)

    im = axes[1].imshow(edge_gradient[np.ix_(order, order)], cmap=DIFF_CMAP, vmin=-np.nanmax(np.abs(edge_gradient)), vmax=np.nanmax(np.abs(edge_gradient)))
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Edges: psilocybin rank - placebo rank")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    label_directional_colorbar(cbar, low_label="more placebo-ranked", high_label="more psilocybin-ranked", label="Positive-FC edge rank gradient")
    save_figure(fig, fig_dir, "fig11_luppi_style_fc_rank_gradients")


def dwell_lengths(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return np.array([], dtype=int)
    lengths = []
    current = labels[0]
    run = 1
    for val in labels[1:]:
        if val == current:
            run += 1
        else:
            lengths.append(run)
            current = val
            run = 1
    lengths.append(run)
    return np.asarray(lengths, dtype=int)


def test_metric(df: pd.DataFrame, metric: str) -> dict[str, float | str | int]:
    pla = df.loc[df["condition"].eq("pla"), metric].to_numpy(float)
    psil = df.loc[df["condition"].eq("psil"), metric].to_numpy(float)
    test = stats.ttest_ind(psil, pla, equal_var=False)
    mw = stats.mannwhitneyu(psil, pla, alternative="two-sided")
    return {
        "metric": metric,
        "n_placebo": int(pla.size),
        "n_psilocybin": int(psil.size),
        "placebo_mean": float(np.nanmean(pla)),
        "psilocybin_mean": float(np.nanmean(psil)),
        "delta_mean": float(np.nanmean(psil) - np.nanmean(pla)),
        "t": float(test.statistic),
        "p": float(test.pvalue),
        "mannwhitney_u": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
    }


def _safe_weighted_modularity(matrix: np.ndarray) -> float:
    try:
        return float(weighted_modularity(matrix))
    except Exception:
        return float("nan")


def _positive_offdiag(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    np.fill_diagonal(arr, 0.0)
    return arr


def _edge_table_to_matrix(edge_df: pd.DataFrame, value_col: str, n_regions: int) -> np.ndarray:
    mat = np.full((n_regions, n_regions), np.nan, dtype=float)
    for _, row in edge_df.iterrows():
        i = int(row["roi_i_index"])
        j = int(row["roi_j_index"])
        value = float(row[value_col])
        mat[i, j] = value
        mat[j, i] = value
    return mat


def analyze(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    npz_dir = out_dir / "npz"
    log_dir = out_dir / "logs"
    for path in (table_dir, fig_dir, npz_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    setup_style()

    scans = load_between_subject_scans(args.data_file)
    metadata = load_metadata(args.metadata_file)
    roi_names = load_roi_names(args.roi_names_file)
    macros = np.array([macro_system(name) for name in roi_names])
    iu, ju = np.triu_indices(len(roi_names), k=1)

    fc_mats = []
    z_edges = []
    scan_rows = []
    for scan in scans:
        fc = fc_matrix(scan.timeseries)
        z = fisher_z(fc)
        upper_fc = fc[iu, ju]
        upper_z = z[iu, ju]
        fc_mats.append(fc)
        z_edges.append(upper_z)
        scan_rows.append(
            {
                "subject": scan.subject,
                "condition": scan.condition,
                "n_regions": int(scan.timeseries.shape[0]),
                "n_timepoints": int(scan.timeseries.shape[1]),
                "mean_abs_fc": float(np.nanmean(np.abs(upper_fc))),
                "mean_signed_fc": float(np.nanmean(upper_fc)),
                "mean_fisher_z": float(np.nanmean(upper_z)),
            }
        )

    fc_mats_arr = np.stack(fc_mats)
    z_edges_arr = np.stack(z_edges)
    scan_df = pd.DataFrame(scan_rows).merge(metadata, on=["subject", "condition"], how="left", validate="one_to_one")
    scan_df.to_csv(table_dir / "scan_level_fc_and_metadata.csv", index=False)

    fc_stats = pd.DataFrame([test_metric(scan_df, "mean_abs_fc"), test_metric(scan_df, "mean_signed_fc"), test_metric(scan_df, "mean_fisher_z")])
    fc_stats.to_csv(table_dir / "fc_subject_summary_stats.csv", index=False)

    pla_mask = scan_df["condition"].eq("pla").to_numpy()
    psil_mask = scan_df["condition"].eq("psil").to_numpy()
    mean_fc = {
        "pla": np.nanmean(fc_mats_arr[pla_mask], axis=0),
        "psil": np.nanmean(fc_mats_arr[psil_mask], axis=0),
    }
    mean_z_edges = {
        "pla": np.nanmean(z_edges_arr[pla_mask], axis=0),
        "psil": np.nanmean(z_edges_arr[psil_mask], axis=0),
    }
    delta_fc = mean_fc["psil"] - mean_fc["pla"]

    edge_rows = []
    p_vals = np.empty(iu.size, dtype=float)
    for e, (i, j) in enumerate(zip(iu, ju)):
        pla = z_edges_arr[pla_mask, e]
        psil = z_edges_arr[psil_mask, e]
        test = stats.ttest_ind(psil, pla, equal_var=False, nan_policy="omit")
        p_vals[e] = test.pvalue
        sys_i, sys_j = sorted([macros[i], macros[j]])
        edge_rows.append(
            {
                "edge": e,
                "roi_i_index": int(i),
                "roi_j_index": int(j),
                "roi_i": roi_names[i],
                "roi_j": roi_names[j],
                "system_i": sys_i,
                "system_j": sys_j,
                "placebo_z_mean": float(np.nanmean(pla)),
                "psilocybin_z_mean": float(np.nanmean(psil)),
                "delta_z": float(np.nanmean(psil) - np.nanmean(pla)),
                "t": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    edge_df = pd.DataFrame(edge_rows)
    edge_df["q"] = bh_fdr(p_vals)
    edge_df.sort_values(["q", "p"], inplace=True, ignore_index=True)
    edge_df.to_csv(table_dir / "fc_edgewise_welch_fdr.csv", index=False)

    block_summary_df = (
        edge_df.groupby(["system_i", "system_j"], as_index=False)
        .agg(
            n_edges=("edge", "count"),
            delta_z_mean=("delta_z", "mean"),
            delta_z_median=("delta_z", "median"),
            n_fdr_05=("q", lambda s: int(np.sum(np.asarray(s, dtype=float) < 0.05))),
        )
        .sort_values(["system_i", "system_j"])
    )
    macro_rows = []
    macro_stat_rows = []
    edge_meta = pd.DataFrame(edge_rows)
    for (system_i, system_j), meta_sub in edge_meta.groupby(["system_i", "system_j"]):
        edge_idx = meta_sub["edge"].to_numpy(dtype=int)
        for scan_idx, scan in enumerate(scans):
            macro_rows.append(
                {
                    "subject": scan.subject,
                    "condition": scan.condition,
                    "system_i": system_i,
                    "system_j": system_j,
                    "mean_fisher_z": float(np.nanmean(z_edges_arr[scan_idx, edge_idx])),
                }
            )
    macro_long_df = pd.DataFrame(macro_rows)
    for (system_i, system_j), sub in macro_long_df.groupby(["system_i", "system_j"]):
        pla = sub.loc[sub["condition"].eq("pla"), "mean_fisher_z"].to_numpy(float)
        psil = sub.loc[sub["condition"].eq("psil"), "mean_fisher_z"].to_numpy(float)
        test = stats.ttest_ind(psil, pla, equal_var=False, nan_policy="omit")
        macro_stat_rows.append(
            {
                "system_i": system_i,
                "system_j": system_j,
                "placebo_mean": float(np.nanmean(pla)),
                "psilocybin_mean": float(np.nanmean(psil)),
                "delta_z_mean": float(np.nanmean(psil) - np.nanmean(pla)),
                "t": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    macro_stats_df = pd.DataFrame(macro_stat_rows)
    macro_stats_df["q"] = bh_fdr(macro_stats_df["p"].to_numpy(float))
    block_df = block_summary_df.merge(
        macro_stats_df[["system_i", "system_j", "placebo_mean", "psilocybin_mean", "t", "p", "q"]],
        on=["system_i", "system_j"],
        how="left",
    )
    block_df.to_csv(table_dir / "fc_macro_system_edge_summary.csv", index=False)
    macro_long_df.to_csv(table_dir / "fc_macro_system_scan_long.csv", index=False)

    topology_rows = []
    topology_densities = [0.10, 0.15, 0.20]
    for scan_idx, scan in enumerate(scans):
        fc_pos = _positive_offdiag(fc_mats_arr[scan_idx])
        for density in topology_densities:
            graph = threshold_top_density(fc_pos, density)
            wb = summarize_within_between(graph, macros)
            topology_rows.append(
                {
                    "subject": scan.subject,
                    "condition": scan.condition,
                    "density": float(density),
                    "global_efficiency": float(weighted_global_efficiency(graph)),
                    "modularity": _safe_weighted_modularity(graph),
                    "within_macro_mean": wb["within_mean"],
                    "between_macro_mean": wb["between_mean"],
                    "within_minus_between_macro": wb["within_minus_between"],
                }
            )
    topology_df = pd.DataFrame(topology_rows).merge(metadata, on=["subject", "condition"], how="left", validate="many_to_one")
    topology_stat_rows = []
    for density in topology_densities:
        sub = topology_df[np.isclose(topology_df["density"].astype(float), density)]
        for metric in ["global_efficiency", "modularity", "within_minus_between_macro"]:
            row = test_metric(sub, metric)
            row["density"] = float(density)
            topology_stat_rows.append(row)
    topology_stats_df = pd.DataFrame(topology_stat_rows)
    topology_stats_df["q"] = np.nan
    for density in topology_densities:
        mask = np.isclose(topology_stats_df["density"].astype(float), density)
        topology_stats_df.loc[mask, "q"] = bh_fdr(topology_stats_df.loc[mask, "p"].to_numpy(float))
    topology_df.to_csv(table_dir / "luppi_style_fc_topology_scan_long.csv", index=False)
    topology_stats_df.to_csv(table_dir / "luppi_style_fc_topology_stats.csv", index=False)

    mean_fc_pos = {cond: _positive_offdiag(mean_fc[cond]) for cond in CONDITION_ORDER}
    nodal_placebo = nodal_strength(mean_fc_pos["pla"])
    nodal_psil = nodal_strength(mean_fc_pos["psil"])
    nodal_gradient = rankdata(nodal_psil, method="average") - rankdata(nodal_placebo, method="average")
    nodal_gradient_df = pd.DataFrame(
        {
            "roi_index": np.arange(len(roi_names)),
            "roi_name": roi_names,
            "macro_system": macros,
            "placebo_positive_fc_strength": nodal_placebo,
            "psilocybin_positive_fc_strength": nodal_psil,
            "strength_delta": nodal_psil - nodal_placebo,
            "rank_gradient_psil_minus_placebo": nodal_gradient,
        }
    ).sort_values("rank_gradient_psil_minus_placebo", ascending=False)
    edge_gradient = edge_rank_gradient(mean_fc_pos["psil"], mean_fc_pos["pla"])
    nodal_gradient_df.to_csv(table_dir / "luppi_style_fc_nodal_rank_gradient.csv", index=False)
    np.savez_compressed(
        npz_dir / "luppi_style_fc_rank_gradients.npz",
        nodal_gradient=nodal_gradient,
        edge_gradient=edge_gradient,
        nodal_placebo=nodal_placebo,
        nodal_psilocybin=nodal_psil,
    )

    patterns_by_scan: list[np.ndarray] = []
    sync_by_scan: list[np.ndarray] = []
    for scan in scans:
        patterns, global_sync, _, _ = phase_patterns(
            scan.timeseries.T,
            pipeline=args.brain_state_pipeline,
            tr_seconds=args.tr_seconds,
            bandpass_hz=(args.bandpass_low_hz, args.bandpass_high_hz),
            filter_order=args.filter_order,
            trim_edge_samples=args.trim_edge_samples,
        )
        patterns_by_scan.append(patterns.astype(np.float32, copy=False))
        sync_by_scan.append(global_sync.astype(np.float32, copy=False))

    split_lengths = [p.shape[0] for p in patterns_by_scan]
    pooled_patterns = np.vstack(patterns_by_scan)
    labels_raw, centers_raw = cluster_brain_states(
        pooled_patterns,
        n_states=args.n_states,
        random_seed=args.random_seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
        backend="sklearn",
    )

    state_strength_raw = np.nanmean(centers_raw, axis=1)
    order = np.argsort(state_strength_raw)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    labels_pooled = inv[labels_raw]
    centers = centers_raw[order]
    state_strength = state_strength_raw[order]

    state_rows = []
    transition_rows = []
    labels_by_scan: list[np.ndarray] = []
    offset = 0
    for scan, n, sync in zip(scans, split_lengths, sync_by_scan):
        labels = labels_pooled[offset : offset + n]
        labels_by_scan.append(labels)
        offset += n
        occ = _compute_occupancy(labels, args.n_states)
        trans = _compute_transition_matrix(labels, args.n_states, collapse_runs=True, exclude_self=True)
        dwell = dwell_lengths(labels)
        for state_idx, value in enumerate(occ, start=1):
            state_rows.append(
                {
                    "subject": scan.subject,
                    "condition": scan.condition,
                    "state": state_idx,
                    "occupancy": float(value),
                }
            )
        for i in range(args.n_states):
            for j in range(args.n_states):
                transition_rows.append(
                    {
                        "subject": scan.subject,
                        "condition": scan.condition,
                        "from_state": i + 1,
                        "to_state": j + 1,
                        "transition_probability": float(trans[i, j]),
                    }
                )
        scan_df.loc[scan_df["subject"].eq(scan.subject), "mean_dwell_samples"] = float(np.mean(dwell)) if dwell.size else np.nan
        scan_df.loc[scan_df["subject"].eq(scan.subject), "median_dwell_samples"] = float(np.median(dwell)) if dwell.size else np.nan
        scan_df.loc[scan_df["subject"].eq(scan.subject), "global_synchrony_mean"] = float(np.nanmean(sync))
        scan_df.loc[scan_df["subject"].eq(scan.subject), "global_synchrony_sd"] = float(np.nanstd(sync))

    state_df = pd.DataFrame(state_rows).merge(metadata, on=["subject", "condition"], how="left", validate="many_to_one")
    transition_df = pd.DataFrame(transition_rows)
    state_df.to_csv(table_dir / "brain_state_occupancy_long.csv", index=False)
    transition_df.to_csv(table_dir / "brain_state_transition_long.csv", index=False)
    scan_df.to_csv(table_dir / "scan_level_fc_brain_state_metrics.csv", index=False)

    state_stats_rows = []
    for state in range(1, args.n_states + 1):
        sub = state_df[state_df["state"].eq(state)]
        row = test_metric(sub, "occupancy")
        row["state"] = state
        state_stats_rows.append(row)
    state_stats = pd.DataFrame(state_stats_rows)
    state_stats["q"] = bh_fdr(state_stats["p"].to_numpy(float))
    state_stats.to_csv(table_dir / "brain_state_occupancy_stats.csv", index=False)

    brain_scan_stats = pd.DataFrame(
        [
            test_metric(scan_df, "mean_dwell_samples"),
            test_metric(scan_df, "median_dwell_samples"),
            test_metric(scan_df, "global_synchrony_mean"),
            test_metric(scan_df, "global_synchrony_sd"),
        ]
    )
    brain_scan_stats.to_csv(table_dir / "brain_state_scan_metric_stats.csv", index=False)

    transition_stat_rows = []
    for (from_state, to_state), sub in transition_df.groupby(["from_state", "to_state"]):
        pla = sub.loc[sub["condition"].eq("pla"), "transition_probability"].to_numpy(float)
        psil = sub.loc[sub["condition"].eq("psil"), "transition_probability"].to_numpy(float)
        test = stats.ttest_ind(psil, pla, equal_var=False, nan_policy="omit")
        transition_stat_rows.append(
            {
                "from_state": int(from_state),
                "to_state": int(to_state),
                "placebo_mean": float(np.nanmean(pla)),
                "psilocybin_mean": float(np.nanmean(psil)),
                "delta_mean": float(np.nanmean(psil) - np.nanmean(pla)),
                "t": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    transition_stats_df = pd.DataFrame(transition_stat_rows)
    transition_stats_df["q"] = bh_fdr(transition_stats_df["p"].to_numpy(float))
    transition_stats_df.to_csv(table_dir / "brain_state_transition_stats.csv", index=False)

    transition_mean = {}
    for cond in CONDITION_ORDER:
        arr = (
            transition_df[transition_df["condition"].eq(cond)]
            .groupby(["from_state", "to_state"])["transition_probability"]
            .mean()
            .unstack()
            .reindex(index=np.arange(1, args.n_states + 1), columns=np.arange(1, args.n_states + 1))
            .to_numpy(float)
        )
        transition_mean[cond] = arr

    state_reference = pd.DataFrame(
        {
            "state": np.arange(1, args.n_states + 1),
            "mean_centroid_phase_coherence": state_strength,
        }
    )
    state_reference.to_csv(table_dir / "brain_state_reference.csv", index=False)

    np.savez_compressed(
        npz_dir / "fc_and_brain_state_arrays.npz",
        fc_mats=fc_mats_arr,
        z_edges=z_edges_arr,
        pooled_patterns=pooled_patterns,
        pooled_labels=labels_pooled,
        centers=centers,
        split_lengths=np.asarray(split_lengths, dtype=int),
        mean_fc_placebo=mean_fc["pla"],
        mean_fc_psilocybin=mean_fc["psil"],
    )

    plot_qc_summary(scan_df, fig_dir)
    edge_q_matrix = _edge_table_to_matrix(edge_df, "q", len(roi_names))
    plot_fc_heatmaps(mean_fc, delta_fc, roi_names, fig_dir, edge_q_matrix=edge_q_matrix)
    plot_fc_summary(scan_df, fc_stats, fig_dir)
    plot_edgewise(edge_df, fig_dir)
    plot_macro_blocks(block_df, fig_dir)
    plot_state_centroids(centers, len(roi_names), fig_dir)
    plot_state_occupancy(state_df, state_stats, fig_dir)
    plot_transition_matrices(transition_mean, fig_dir, transition_stats=transition_stats_df)
    plot_dwell_synchrony(scan_df, brain_scan_stats, fig_dir)
    plot_luppi_fc_topology_with_stats(topology_df, topology_stats_df, fig_dir, density=0.15)
    plot_luppi_rank_gradients(nodal_gradient_df, edge_gradient, roi_names, fig_dir)

    manifest = {
        "data_file": str(args.data_file),
        "metadata_file": str(args.metadata_file),
        "roi_names_file": str(args.roi_names_file),
        "out_dir": str(out_dir),
        "n_scans": len(scans),
        "condition_counts": scan_df["condition"].value_counts().to_dict(),
        "n_regions": len(roi_names),
        "n_edges": int(iu.size),
        "brain_state_pipeline": args.brain_state_pipeline,
        "n_states": args.n_states,
        "tr_seconds": args.tr_seconds,
        "bandpass_hz": [args.bandpass_low_hz, args.bandpass_high_hz],
        "filter_order": args.filter_order,
        "random_seed": args.random_seed,
        "n_init": args.n_init,
        "fc_subject_stats_csv": str(table_dir / "fc_subject_summary_stats.csv"),
        "brain_state_occupancy_stats_csv": str(table_dir / "brain_state_occupancy_stats.csv"),
        "brain_state_transition_stats_csv": str(table_dir / "brain_state_transition_stats.csv"),
        "luppi_style_fc_topology_stats_csv": str(table_dir / "luppi_style_fc_topology_stats.csv"),
        "luppi_style_note": "These are FC-only topology/rank summaries inspired by the Luppi downstream analysis. They are not STS/RTR synergy-redundancy gradients; true PhiID STS/RTR requires a separate MATLAB PhiID run on this AAL116 dataset.",
    }
    (log_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_analysis_guide(out_dir, args)

    print(json.dumps(manifest, indent=2))
    print("\nKey FC stats")
    print(fc_stats.to_string(index=False))
    print("\nBrain-state occupancy stats")
    print(state_stats[["state", "placebo_mean", "psilocybin_mean", "delta_mean", "p", "q"]].to_string(index=False))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=PSIL_FILE)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)
    parser.add_argument("--roi-names-file", type=Path, default=ROI_NAMES_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-states", type=int, default=5)
    parser.add_argument("--brain-state-pipeline", choices=["standard", "brain_act_legacy"], default="brain_act_legacy")
    parser.add_argument("--tr-seconds", type=float, default=2.4)
    parser.add_argument("--bandpass-low-hz", type=float, default=0.01)
    parser.add_argument("--bandpass-high-hz", type=float, default=0.20)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--trim-edge-samples", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
