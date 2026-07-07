#!/usr/bin/env python3
"""Brain-state analysis for new DoC BOLD data (Control/EMCS/MCS/UWS).

This script reproduces the prior TVBToolkit subject-level brain-state settings
used in earlier Brain-Act work:
- n_states = 5
- trim_edge_samples = 9
- n_init (subject clustering) = 20
- default summarize_brain_states pipeline/backend

It loads BOLD time series from the external DOC raw tree, runs subject-level
brain-state extraction, aligns local states to shared templates, computes group
summaries/statistics, and writes publication-style figures + CSV outputs.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep matplotlib/fontconfig/TVB caches writable inside sandboxed runs.
_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))
os.environ.setdefault("TVB_USER_HOME", str((_REPO_ROOT / ".tvb-temp").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required to load MATLAB files.") from exc

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    from tvbtoolkit import align_states_to_templates, fit_state_templates, summarize_brain_states
except ModuleNotFoundError:  # pragma: no cover
    src = _REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tvbtoolkit import align_states_to_templates, fit_state_templates, summarize_brain_states

from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F9FC",
            "axes.edgecolor": "#2A2A2A",
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.dpi": 200,
        }
    )


PALETTE = {
    "control": "#2E86AB",
    "emcs": "#54A24B",
    "mcs": "#E67E22",
    "uws": "#C0392B",
}
COHORTS = ("control", "emcs", "mcs", "uws")

# Cohort-specific BOLD bundles in the new dataset.
FILE_SPECS = (
    # Control
    {
        "cohort": "control",
        "stage": "control",
        "sedation": "non_sedated",
        "fc_path": "CNT_send/FC/DoC_CNT.mat",
        "fc_var": "DoC_CNT",
        "sc_path": "CNT_send/SC/CNT_SC.mat",
        "sc_var": "SC",
    },
    # EMCS
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
    # MCS
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
    # UWS
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


@dataclass
class SubjectRecord:
    cohort: str
    subject_id: str
    stage: str
    sedation: str
    source_fc_file: str
    source_sc_file: str
    timeseries: np.ndarray  # (time, regions)
    sc_matrix: np.ndarray  # (regions, regions)


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_mat_mapping(path: Path) -> dict[str, np.ndarray]:
    try:
        raw = sio.loadmat(path)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        pass
    except ValueError:
        pass

    if h5py is None:
        raise RuntimeError(f"Cannot read {path}; h5py is required for MATLAB v7.3 files.")

    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                out[key] = obj[()]
    return out


def _decode_subject_names(value: np.ndarray | None) -> list[str]:
    if value is None:
        return []
    arr = np.asarray(value)
    names: list[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            txt = item.decode("utf-8", errors="ignore").strip()
        elif isinstance(item, str):
            txt = item.strip()
        elif isinstance(item, np.ndarray):
            if item.dtype.kind in {"U", "S"}:
                txt = "".join(np.asarray(item).astype(str).reshape(-1)).strip()
            elif item.size == 1 and isinstance(item.item(), (str, bytes)):
                cell = item.item()
                txt = cell.decode("utf-8", errors="ignore").strip() if isinstance(cell, bytes) else str(cell).strip()
            else:
                txt = str(item.squeeze()).strip()
        else:
            txt = str(item).strip()
        if txt:
            names.append(txt)
    return names


def _select_bold_array(mapping: dict[str, np.ndarray], variable_hint: str | None) -> tuple[str, np.ndarray]:
    if variable_hint is not None and variable_hint in mapping:
        arr = np.asarray(mapping[variable_hint], dtype=float)
        if arr.ndim == 3:
            return variable_hint, arr

    candidates: list[tuple[str, np.ndarray]] = []
    for key, val in mapping.items():
        arr = np.asarray(val)
        if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
            candidates.append((key, np.asarray(arr, dtype=float)))
    if not candidates:
        raise RuntimeError(f"No 3D numeric BOLD array found in variables: {list(mapping.keys())}")

    candidates.sort(key=lambda kv: (0 if kv[0].lower().startswith("doc") else 1, kv[0]))
    return candidates[0]


def _to_subject_roi_time(arr3d: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr3d, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")

    shape = arr.shape
    roi_axes = [i for i, s in enumerate(shape) if s == 90]
    if not roi_axes:
        raise ValueError(f"Cannot infer ROI axis (size 90) from shape {shape}")
    roi_axis = roi_axes[0]

    time_axes = [i for i, s in enumerate(shape) if s in (295, 297) and i != roi_axis]
    if time_axes:
        time_axis = time_axes[0]
    else:
        other = [i for i in range(3) if i != roi_axis]
        time_axis = max(other, key=lambda i: shape[i])

    subject_axis = [i for i in range(3) if i not in (roi_axis, time_axis)][0]
    return np.moveaxis(arr, (subject_axis, roi_axis, time_axis), (0, 1, 2))


def _to_subject_roi_roi(arr3d: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr3d, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D SC array, got {arr.shape}")
    shape = arr.shape
    non_90_axes = [i for i, s in enumerate(shape) if s != 90]
    if len(non_90_axes) == 1:
        subject_axis = non_90_axes[0]
    else:
        # Fallback for malformed arrays: assume trailing axis is subjects.
        subject_axis = 2
    out = np.moveaxis(arr, subject_axis, 0)
    if out.shape[1] != 90 or out.shape[2] != 90:
        raise ValueError(f"SC reshape failed for {shape} -> {out.shape}")
    return out


def _upper_triangle_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got {arr.shape}")
    iu = np.triu_indices(arr.shape[0], k=1)
    return np.asarray(arr[iu], dtype=float)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=float).reshape(-1)
    xb = np.asarray(b, dtype=float).reshape(-1)
    if xa.size != xb.size or xa.size == 0:
        return float("nan")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(xb)):
        return float("nan")
    if float(np.std(xa)) <= 0.0 or float(np.std(xb)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def load_new_doc_subjects(data_root: Path, max_subjects_per_group: int | None = None) -> list[SubjectRecord]:
    records: list[SubjectRecord] = []
    cohort_counter: dict[str, int] = defaultdict(int)

    for spec in FILE_SPECS:
        cohort = str(spec["cohort"])
        if max_subjects_per_group is not None and cohort_counter[cohort] >= max_subjects_per_group:
            continue

        fc_path = data_root / str(spec["fc_path"])
        sc_path = data_root / str(spec["sc_path"])
        fc_mapping = _load_mat_mapping(fc_path)
        sc_mapping = _load_mat_mapping(sc_path)
        _, fc_arr = _select_bold_array(fc_mapping, variable_hint=str(spec["fc_var"]))
        fc_srt = _to_subject_roi_time(fc_arr)  # (subjects, 90, time)
        _, sc_arr = _select_bold_array(sc_mapping, variable_hint=str(spec["sc_var"]))
        sc_srr = _to_subject_roi_roi(sc_arr)  # (subjects, 90, 90)

        n_subjects = int(fc_srt.shape[0])
        if int(sc_srr.shape[0]) != n_subjects:
            raise RuntimeError(
                f"FC/SC subject count mismatch for cohort={cohort} "
                f"({fc_path.name}: {n_subjects} vs {sc_path.name}: {int(sc_srr.shape[0])})"
            )

        names = _decode_subject_names(fc_mapping.get("subj_names"))
        if names and len(names) != n_subjects:
            names = []

        for i in range(n_subjects):
            if max_subjects_per_group is not None and cohort_counter[cohort] >= max_subjects_per_group:
                break

            cohort_counter[cohort] += 1
            if cohort == "control" and names:
                sid = names[i]
            else:
                sid = f"{cohort}-sub-{cohort_counter[cohort]:03d}"

            records.append(
                SubjectRecord(
                    cohort=cohort,
                    subject_id=sid,
                    stage=str(spec["stage"]),
                    sedation=str(spec["sedation"]),
                    source_fc_file=str(fc_path.relative_to(data_root)),
                    source_sc_file=str(sc_path.relative_to(data_root)),
                    timeseries=np.asarray(fc_srt[i].T, dtype=float),  # (time, 90)
                    sc_matrix=np.asarray(sc_srr[i], dtype=float),
                )
            )

    return records


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


def _compute_occupancy_stats(occupancy_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows: list[dict[str, float | str]] = []
    pair_rows: list[dict[str, float | str]] = []

    for t in sorted(occupancy_df["template_state"].unique()):
        dt = occupancy_df[occupancy_df["template_state"] == t]
        per_cohort = {
            c: dt.loc[dt["cohort"] == c, "occupancy"].to_numpy(dtype=float)
            for c in COHORTS
            if not dt.loc[dt["cohort"] == c].empty
        }
        if len(per_cohort) < 2:
            continue

        try:
            h, p_kw = kruskal(*per_cohort.values())
        except ValueError:
            continue

        omnibus_rows.append(
            {
                "template_state": int(t),
                "test": "Kruskal-Wallis",
                "H": float(h),
                "p": float(p_kw),
            }
        )

        contrasts = list(itertools.combinations(per_cohort.keys(), 2))
        p_raw: list[float] = []
        rows_idx: list[int] = []
        for a, b in contrasts:
            xa = per_cohort[a]
            xb = per_cohort[b]
            u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            p_raw.append(float(p))
            rows_idx.append(len(pair_rows))
            pair_rows.append(
                {
                    "template_state": int(t),
                    "contrast": f"{a} vs {b}",
                    "n_a": int(xa.size),
                    "n_b": int(xb.size),
                    "U": float(u),
                    "p_raw": float(p),
                    "p_holm": np.nan,
                    "median_a": float(np.median(xa)),
                    "median_b": float(np.median(xb)),
                }
            )

        p_holm = _holm_correct(p_raw)
        for idx, val in zip(rows_idx, p_holm):
            pair_rows[idx]["p_holm"] = float(val)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pair_rows)


def _plot_template_occupancy_distributions(occupancy_df: pd.DataFrame, n_states: int, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, n_states, figsize=(3.3 * n_states, 4.8), sharey=True)
    if n_states == 1:
        axes = [axes]

    for t in range(n_states):
        ax = axes[t]
        dt = occupancy_df[occupancy_df["template_state"] == t]
        for ci, cohort in enumerate(COHORTS):
            vals = dt.loc[dt["cohort"] == cohort, "occupancy"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            jitter = np.linspace(-0.08, 0.08, num=vals.size)
            ax.scatter(
                np.full(vals.size, ci, dtype=float) + jitter,
                vals,
                s=22,
                alpha=0.7,
                color=PALETTE[cohort],
                edgecolor="black",
                linewidth=0.25,
            )
            median = np.median(vals)
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            ax.plot([ci - 0.18, ci + 0.18], [median, median], color="black", lw=1.5)
            ax.vlines(ci, q1, q3, color="black", lw=1.0)

        ax.set_title(f"Template {t + 1}")
        ax.set_xticks(np.arange(len(COHORTS)), [c.upper() for c in COHORTS], rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        if t == 0:
            ax.set_ylabel("Occupancy")

    fig.suptitle("Subject-Level Template Occupancy by Cohort", y=1.03)
    fig.tight_layout()
    _save_figure(fig, out_dir, "template_occupancy_distributions")


def _plot_cohort_mean_occupancy(occupancy_df: pd.DataFrame, n_states: int, out_dir: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    x = np.arange(n_states)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(COHORTS))
    for off, cohort in zip(offsets, COHORTS):
        vals = []
        sems = []
        for t in range(n_states):
            vt = occupancy_df.loc[
                (occupancy_df["cohort"] == cohort) & (occupancy_df["template_state"] == t), "occupancy"
            ].to_numpy(dtype=float)
            vals.append(float(np.mean(vt)) if vt.size else np.nan)
            sems.append(float(np.std(vt, ddof=1) / np.sqrt(vt.size)) if vt.size > 1 else 0.0)
        ax.bar(x + off, vals, width=width, yerr=sems, capsize=3, color=PALETTE[cohort], alpha=0.85, label=cohort.upper())

    ax.set_xticks(x, [f"T{t + 1}" for t in range(n_states)])
    ax.set_xlabel("Template state")
    ax.set_ylabel("Mean occupancy")
    ax.set_title("Cohort Mean Occupancy by Canonical Template")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, out_dir, "cohort_mean_template_occupancy")


def _plot_transition_heatmaps(cohort_transition_means: dict[str, np.ndarray], n_states: int, out_dir: Path) -> None:
    set_publication_style()
    vmax = 0.0
    for cohort in COHORTS:
        arr = cohort_transition_means.get(cohort)
        if arr is not None and arr.size:
            vmax = max(vmax, float(np.max(arr)))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.flatten()
    im = None
    for ax, cohort in zip(axes_flat, COHORTS):
        arr = cohort_transition_means.get(cohort, np.zeros((n_states, n_states), dtype=float))
        im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(cohort.upper())
        ax.set_xticks(np.arange(n_states), [f"T{i + 1}" for i in range(n_states)])
        ax.set_yticks(np.arange(n_states), [f"T{i + 1}" for i in range(n_states)])
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")

    if im is not None:
        fig.colorbar(im, ax=axes_flat, fraction=0.025, pad=0.02, label="Transition probability")
    fig.suptitle("Cohort Mean Transition Matrices (Template-Aligned)", y=0.98)
    _save_figure(fig, out_dir, "cohort_transition_heatmaps")


def _plot_matched_similarity(subject_df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    vals = []
    labels = []
    for cohort in COHORTS:
        vc = subject_df.loc[subject_df["cohort"] == cohort, "matched_similarity_mean"].to_numpy(dtype=float)
        if vc.size == 0:
            continue
        vals.append(vc)
        labels.append(cohort.upper())

    vp = ax.violinplot(vals, showmeans=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        cohort_key = labels[i].lower()
        body.set_facecolor(PALETTE.get(cohort_key, "#666666"))
        body.set_edgecolor("#111111")
        body.set_alpha(0.55)

    for i, vc in enumerate(vals, start=1):
        jitter = np.linspace(-0.08, 0.08, num=vc.size)
        ax.scatter(np.full(vc.size, i, dtype=float) + jitter, vc, s=20, alpha=0.65, color="#111111")

    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel("Mean matched centroid similarity (Pearson r)")
    ax.set_title("Template-Matching Quality by Cohort")
    fig.tight_layout()
    _save_figure(fig, out_dir, "template_matching_quality")


def _plot_subject_occupancy_heatmap(subject_df: pd.DataFrame, n_states: int, out_dir: Path) -> None:
    set_publication_style()
    cols = [f"template_{i + 1}_occupancy" for i in range(n_states)]
    heat = (
        subject_df[["cohort", "subject_id", *cols]]
        .sort_values(["cohort", "subject_id"])
        .set_index(["cohort", "subject_id"])
    )
    arr = heat[cols].to_numpy(dtype=float)
    ylab = [f"{c}:{s}" for c, s in heat.index]

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.18 * arr.shape[0])))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.2, float(np.max(arr))))
    ax.set_xticks(np.arange(n_states), [f"T{i + 1}" for i in range(n_states)])
    ax.set_yticks(np.arange(arr.shape[0]), ylab)
    ax.set_xlabel("Template state")
    ax.set_ylabel("Subject")
    ax.set_title("Subject-Level Occupancy Heatmap (Template-Aligned)")
    fig.colorbar(im, ax=ax, label="Occupancy")
    fig.tight_layout()
    _save_figure(fig, out_dir, "subject_occupancy_heatmap")


def _plot_sfc_vs_occupancy_local(sfc_local_df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    for cohort in COHORTS:
        dc = sfc_local_df[sfc_local_df["cohort"] == cohort]
        if dc.empty:
            continue
        x = dc["sfc"].to_numpy(dtype=float)
        y = dc["occupancy_local"].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size == 0:
            continue

        ax.scatter(
            x,
            y,
            s=20,
            alpha=0.38,
            color=PALETTE[cohort],
            edgecolors="none",
            label=f"{cohort.upper()} states",
        )
        if x.size >= 3 and float(np.std(x)) > 0.0:
            p = np.polyfit(x, y, deg=1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            yy = p[0] * xx + p[1]
            ax.plot(xx, yy, color=PALETTE[cohort], lw=2.0, alpha=0.9)

    ax.set_xlabel("State FC-SC coupling (subject-level Pearson r)")
    ax.set_ylabel("State occupancy")
    ax.set_title("Subject-Level State Occupancy vs FC-SC Coupling")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "sfc_vs_occupancy_subject_level")


def run_analysis(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    fig_root = out_root / "figures"
    met_root = out_root / "metrics"
    fig_root.mkdir(parents=True, exist_ok=True)
    met_root.mkdir(parents=True, exist_ok=True)

    records = load_new_doc_subjects(data_root=data_root, max_subjects_per_group=args.max_subjects_per_group)
    if not records:
        raise RuntimeError("No subject records loaded from new DoC BOLD data.")

    cohort_counts = pd.Series([r.cohort for r in records]).value_counts().to_dict()
    print("Loaded subjects by cohort:", cohort_counts)

    subject_results: list[dict[str, Any]] = []
    all_centers: list[np.ndarray] = []
    for idx, rec in enumerate(records):
        bs = summarize_brain_states(
            rec.timeseries,
            n_states=args.n_states,
            trim_edge_samples=args.trim_edge_samples,
            random_seed=args.random_seed_subject + idx,
            n_init=args.n_init_subject,
            pipeline=args.pipeline,
            clustering_backend=args.clustering_backend,
            tr_seconds=args.tr_seconds,
            bandpass_hz=(args.bandpass_low_hz, args.bandpass_high_hz),
            filter_order=args.filter_order,
        )
        centers = np.asarray(bs.centers, dtype=float)
        occupancy = np.asarray(bs.occupancy, dtype=float)
        if centers.ndim != 2 or occupancy.ndim != 1 or centers.shape[0] != occupancy.shape[0]:
            continue
        if centers.shape[0] != args.n_states:
            # Keep script robust if effective K differs for very short traces.
            continue

        sc_vec = _upper_triangle_vector(rec.sc_matrix)
        sfc_local = np.asarray([_safe_pearson(row, sc_vec) for row in centers], dtype=float)
        sfc_order = np.argsort(np.nan_to_num(sfc_local, nan=np.inf))

        subject_results.append(
            {
                "cohort": rec.cohort,
                "subject_id": rec.subject_id,
                "stage": rec.stage,
                "sedation": rec.sedation,
                "source_fc_file": rec.source_fc_file,
                "source_sc_file": rec.source_sc_file,
                "centers": centers,
                "occupancy_local": occupancy,
                "transition_local": np.asarray(bs.transition_matrix, dtype=float),
                "sfc_local": sfc_local,
                "sfc_rank_order_local": sfc_order,
                "occupancy_sfc_sorted_local": occupancy[sfc_order],
                "sfc_sorted_local": sfc_local[sfc_order],
                "global_sync_mean": float(np.mean(np.asarray(bs.global_synchrony, dtype=float))),
                "global_sync_std": float(np.std(np.asarray(bs.global_synchrony, dtype=float))),
            }
        )
        all_centers.append(centers)

    if not subject_results:
        raise RuntimeError("Brain-state extraction produced no valid subject outputs.")

    if args.template_source == "control_only":
        template_pool = [r["centers"] for r in subject_results if r["cohort"] == "control"]
        if not template_pool:
            raise RuntimeError("template_source='control_only' but no control subjects were loaded.")
    else:
        template_pool = all_centers

    templates = fit_state_templates(
        np.vstack(template_pool),
        n_states=args.n_states,
        random_seed=args.random_seed_template,
        n_init=args.n_init_template,
    )

    for res in subject_results:
        aligned = align_states_to_templates(res["centers"], res["occupancy_local"], templates)
        assignment = np.asarray(aligned.assignment_local_to_template, dtype=int)
        occ_template = np.asarray(aligned.occupancy_aligned, dtype=float)
        sim_matched = np.asarray(aligned.matched_similarity, dtype=float)
        sfc_local = np.asarray(res["sfc_local"], dtype=float)
        sfc_template = np.full(args.n_states, np.nan, dtype=float)
        for i_local, i_template in enumerate(assignment):
            if int(i_template) >= 0:
                sfc_template[int(i_template)] = sfc_local[i_local]

        tm_local = np.asarray(res["transition_local"], dtype=float)
        tm_template = np.zeros((args.n_states, args.n_states), dtype=float)
        for i_local in range(tm_local.shape[0]):
            ti = int(assignment[i_local])
            if ti < 0:
                continue
            for j_local in range(tm_local.shape[1]):
                tj = int(assignment[j_local])
                if tj < 0:
                    continue
                tm_template[ti, tj] += tm_local[i_local, j_local]
        rs = tm_template.sum(axis=1, keepdims=True)
        rs[rs == 0.0] = 1.0
        tm_template = tm_template / rs

        res["assignment_local_to_template"] = assignment
        res["occupancy_template"] = occ_template
        res["transition_template"] = tm_template
        res["sfc_template"] = sfc_template
        res["matched_similarity_mean"] = float(np.mean(sim_matched))
        res["matched_similarity_min"] = float(np.min(sim_matched))

    subject_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    sfc_template_rows: list[dict[str, Any]] = []
    sfc_local_rows: list[dict[str, Any]] = []
    for res in subject_results:
        row: dict[str, Any] = {
            "cohort": res["cohort"],
            "subject_id": res["subject_id"],
            "stage": res["stage"],
            "sedation": res["sedation"],
            "source_fc_file": res["source_fc_file"],
            "source_sc_file": res["source_sc_file"],
            "global_sync_mean": res["global_sync_mean"],
            "global_sync_std": res["global_sync_std"],
            "matched_similarity_mean": res["matched_similarity_mean"],
            "matched_similarity_min": res["matched_similarity_min"],
        }
        occ = np.asarray(res["occupancy_template"], dtype=float)
        sfc_template = np.asarray(res["sfc_template"], dtype=float)
        for t in range(args.n_states):
            row[f"template_{t + 1}_occupancy"] = float(occ[t])
            row[f"template_{t + 1}_sfc"] = float(sfc_template[t])
            occupancy_rows.append(
                {
                    "cohort": res["cohort"],
                    "subject_id": res["subject_id"],
                    "stage": res["stage"],
                    "sedation": res["sedation"],
                    "template_state": t,
                    "occupancy": float(occ[t]),
                }
            )
            sfc_template_rows.append(
                {
                    "cohort": res["cohort"],
                    "subject_id": res["subject_id"],
                    "stage": res["stage"],
                    "sedation": res["sedation"],
                    "template_state": t,
                    "sfc": float(sfc_template[t]),
                }
            )

        tm = np.asarray(res["transition_template"], dtype=float)
        for i in range(args.n_states):
            for j in range(args.n_states):
                transition_rows.append(
                    {
                        "cohort": res["cohort"],
                        "subject_id": res["subject_id"],
                        "from_template": i,
                        "to_template": j,
                        "probability": float(tm[i, j]),
                    }
                )

        sfc_local = np.asarray(res["sfc_local"], dtype=float)
        occ_local = np.asarray(res["occupancy_local"], dtype=float)
        for i_local in range(args.n_states):
            sfc_local_rows.append(
                {
                    "cohort": res["cohort"],
                    "subject_id": res["subject_id"],
                    "stage": res["stage"],
                    "sedation": res["sedation"],
                    "template_state": np.nan,
                    "local_state": i_local,
                    "sfc": float(sfc_local[i_local]),
                    "occupancy_local": float(occ_local[i_local]),
                    "sfc_rank_local": int(np.where(res["sfc_rank_order_local"] == i_local)[0][0]),
                }
            )
        subject_rows.append(row)

    subject_df = pd.DataFrame(subject_rows).sort_values(["cohort", "subject_id"]).reset_index(drop=True)
    occupancy_df = pd.DataFrame(occupancy_rows)
    transition_df = pd.DataFrame(transition_rows)
    sfc_template_df = pd.DataFrame(sfc_template_rows)
    sfc_local_df = pd.DataFrame(sfc_local_rows)

    cohort_transition_means: dict[str, np.ndarray] = {}
    cohort_summary_rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        dsub = subject_df[subject_df["cohort"] == cohort]
        if dsub.empty:
            continue
        dtr = transition_df[transition_df["cohort"] == cohort]
        tmean = (
            dtr.pivot_table(index="from_template", columns="to_template", values="probability", aggfunc="mean")
            .reindex(index=np.arange(args.n_states), columns=np.arange(args.n_states), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        cohort_transition_means[cohort] = tmean
        cohort_summary_rows.append(
            {
                "cohort": cohort,
                "n_subjects": int(dsub.shape[0]),
                "global_sync_mean": float(dsub["global_sync_mean"].mean()),
                "global_sync_std": float(dsub["global_sync_mean"].std(ddof=1) if dsub.shape[0] > 1 else 0.0),
                "matched_similarity_mean": float(dsub["matched_similarity_mean"].mean()),
                "matched_similarity_std": float(
                    dsub["matched_similarity_mean"].std(ddof=1) if dsub.shape[0] > 1 else 0.0
                ),
            }
        )

    stats_omnibus_df, stats_pairwise_df = _compute_occupancy_stats(occupancy_df)

    subject_df.to_csv(met_root / "subject_level_template_occupancy.csv", index=False)
    occupancy_df.to_csv(met_root / "subject_level_template_occupancy_long.csv", index=False)
    transition_df.to_csv(met_root / "subject_level_template_transitions_long.csv", index=False)
    sfc_template_df.to_csv(met_root / "subject_level_template_sfc.csv", index=False)
    sfc_local_df.to_csv(met_root / "subject_level_local_state_sfc.csv", index=False)
    pd.DataFrame(cohort_summary_rows).to_csv(met_root / "cohort_summary.csv", index=False)
    stats_omnibus_df.to_csv(met_root / "stats_omnibus_template_occupancy.csv", index=False)
    stats_pairwise_df.to_csv(met_root / "stats_pairwise_template_occupancy.csv", index=False)
    np.savez_compressed(
        met_root / "state_templates.npz",
        templates=np.asarray(templates, dtype=float),
        template_source=np.asarray([args.template_source]),
        n_states=np.asarray([args.n_states], dtype=int),
    )

    _plot_template_occupancy_distributions(occupancy_df, n_states=args.n_states, out_dir=fig_root)
    _plot_cohort_mean_occupancy(occupancy_df, n_states=args.n_states, out_dir=fig_root)
    _plot_transition_heatmaps(cohort_transition_means, n_states=args.n_states, out_dir=fig_root)
    _plot_matched_similarity(subject_df, out_dir=fig_root)
    _plot_subject_occupancy_heatmap(subject_df, n_states=args.n_states, out_dir=fig_root)
    _plot_sfc_vs_occupancy_local(sfc_local_df, out_dir=fig_root)

    print(f"Saved outputs to: {out_root}")
    print(f"Subjects analyzed: {subject_df.shape[0]}")
    print("Cohort counts:", subject_df["cohort"].value_counts().to_dict())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(doc_liege_raw("doc_data")),
        help="Path to the new DoC dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(doc_liege_results("doc_patients_new_bold_brain_states")),
        help="Directory where metrics and figures are saved.",
    )
    parser.add_argument("--n-states", type=int, default=5, help="Number of brain states/templates.")
    parser.add_argument("--trim-edge-samples", type=int, default=9, help="Edge trimming used in prior notebook parity.")
    parser.add_argument("--n-init-subject", type=int, default=20, help="Subject-level clustering restarts.")
    parser.add_argument("--n-init-template", type=int, default=48, help="Template fitting restarts.")
    parser.add_argument(
        "--template-source",
        type=str,
        default="control_only",
        choices=["control_only", "all_cohorts"],
        help="Template basis source.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="standard",
        choices=["standard", "brain_act_legacy"],
        help="Brain-state preprocessing pipeline passed to summarize_brain_states.",
    )
    parser.add_argument(
        "--clustering-backend",
        type=str,
        default=None,
        choices=["scipy", "sklearn"],
        help="Optional clustering backend override.",
    )
    parser.add_argument("--tr-seconds", type=float, default=2.4, help="TR for legacy pipeline filtering.")
    parser.add_argument("--bandpass-low-hz", type=float, default=0.01, help="Legacy low cutoff (Hz).")
    parser.add_argument("--bandpass-high-hz", type=float, default=0.20, help="Legacy high cutoff (Hz).")
    parser.add_argument("--filter-order", type=int, default=3, help="Legacy filter order.")
    parser.add_argument("--random-seed-subject", type=int, default=0, help="Base random seed for subject clustering.")
    parser.add_argument("--random-seed-template", type=int, default=7, help="Random seed for template fitting.")
    parser.add_argument(
        "--max-subjects-per-group",
        type=int,
        default=None,
        help="Optional debug cap per cohort for quick smoke runs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
