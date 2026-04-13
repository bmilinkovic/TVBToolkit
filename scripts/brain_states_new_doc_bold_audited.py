#!/usr/bin/env python3
"""Audited brain-state analysis for the new DoC BOLD dataset.

Primary analysis design (default):
- Subject-local phase-state extraction (k=5) from each subject's BOLD timeseries.
- Brain-Act legacy-compatible preprocessing path for phase extraction.
- Subject-wise SC/FC coupling checks with explicit ROI-order diagnostics.
- Rank-based post-hoc alignment within subject (by local state SC-coupling), not
  canonical template enforcement during state fitting.

Optional (explicitly secondary):
- Canonical template fitting/alignment as a post-hoc comparison step.
"""

from __future__ import annotations

import argparse
import json
import math
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
from scipy.spatial.distance import squareform

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required to load MATLAB files.") from exc

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    from tvbtoolkit import (
        align_states_to_templates,
        cluster_brain_states,
        fit_state_templates,
        phase_patterns,
        summarize_brain_states,
    )
except ModuleNotFoundError:  # pragma: no cover
    src = _REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tvbtoolkit import (
        align_states_to_templates,
        cluster_brain_states,
        fit_state_templates,
        phase_patterns,
        summarize_brain_states,
    )


COHORTS = ("control", "emcs", "mcs", "uws")
PALETTE = {
    "control": "#2E86AB",
    "emcs": "#4DAF4A",
    "mcs": "#E67E22",
    "uws": "#C0392B",
}

FILE_SPECS = (
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


@dataclass
class SubjectRecord:
    cohort: str
    subject_id: str
    stage: str
    sedation: str
    source_fc_file: str
    source_sc_file: str
    source_subject_index: int
    source_subject_label: str
    timeseries: np.ndarray  # (time, regions)
    sc_matrix: np.ndarray  # (regions, regions)


@dataclass(frozen=True)
class ROIOrderReference:
    """Reference ROI ordering metadata for FC/SC alignment checks."""

    interleaved_names: tuple[str, ...]
    symmetric_names: tuple[str, ...]
    interleaved_to_symmetric_idx: np.ndarray
    interleaved_source: str
    symmetric_source: str


def set_publication_style() -> None:
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
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.dpi": 220,
        }
    )


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
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


def _select_3d_numeric(mapping: dict[str, np.ndarray], variable_hint: str | None) -> tuple[str, np.ndarray]:
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
        raise RuntimeError(f"No 3D numeric array found in variables: {list(mapping.keys())}")

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


def _full_matrix_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got {arr.shape}")
    return np.asarray(arr.reshape(-1), dtype=float)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=float).reshape(-1)
    xb = np.asarray(b, dtype=float).reshape(-1)
    if xa.size != xb.size or xa.size == 0:
        return float("nan")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(xb)):
        return float("nan")
    sa = float(np.std(xa))
    sb = float(np.std(xb))
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def _compute_static_fc(timeseries: np.ndarray) -> np.ndarray:
    x = np.asarray(timeseries, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D timeseries (T,R), got {x.shape}")
    fc = np.corrcoef(x, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    fc = 0.5 * (fc + fc.T)
    np.fill_diagonal(fc, 1.0)
    return fc


def _aal90_symmetry_reorder_index() -> np.ndarray:
    left = np.arange(0, 90, 2)
    right = np.arange(1, 90, 2)[::-1]
    idx = np.concatenate([left, right])
    if idx.size != 90:
        raise RuntimeError("AAL90 reorder index has invalid length.")
    return idx.astype(int)


def _parse_roi_mni_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(str(parts[1]).strip())
        else:
            parts_ws = line.split()
            if len(parts_ws) >= 2:
                names.append(str(parts_ws[1]).strip())
    if len(names) != 90:
        raise RuntimeError(f"Expected 90 ROI names in {path}, found {len(names)}.")
    return names


def _parse_symmetric_lookup_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(str(parts[1]).strip())
        else:
            parts_ws = line.split()
            if len(parts_ws) >= 2:
                names.append(str(parts_ws[1]).strip())
    if len(names) != 90:
        raise RuntimeError(f"Expected 90 ROI names in {path}, found {len(names)}.")
    return names


def build_roi_order_reference(data_root: Path) -> ROIOrderReference:
    """Build interleaved/symmetric ROI reference and verify mapping consistency."""
    data_root = Path(data_root)
    inter_path = data_root / "ROI_MNI_V4_90.txt"
    sym_path = data_root / "symmetric_lookuptable_clean.txt"
    if not inter_path.exists():
        raise FileNotFoundError(f"Missing ROI reference file: {inter_path}")
    if not sym_path.exists():
        raise FileNotFoundError(f"Missing ROI reference file: {sym_path}")

    inter = _parse_roi_mni_names(inter_path)
    sym = _parse_symmetric_lookup_names(sym_path)

    inter_idx = {name: i for i, name in enumerate(inter)}
    if len(inter_idx) != len(inter):
        raise RuntimeError("Duplicate ROI names found in ROI_MNI_V4_90.txt.")
    missing = [name for name in sym if name not in inter_idx]
    if missing:
        raise RuntimeError(f"Symmetric lookup contains ROI names missing from ROI_MNI file: {missing[:8]}")

    idx_from_names = np.asarray([inter_idx[name] for name in sym], dtype=int)
    idx_formula = _aal90_symmetry_reorder_index()
    if not np.array_equal(idx_from_names, idx_formula):
        mismatch = np.where(idx_from_names != idx_formula)[0]
        first = int(mismatch[0]) if mismatch.size else -1
        raise RuntimeError(
            "Name-derived AAL90 reorder index does not match hard-coded permutation. "
            f"First mismatch at symmetric position {first + 1}: "
            f"name-derived={int(idx_from_names[first])}, formula={int(idx_formula[first])}."
        )

    return ROIOrderReference(
        interleaved_names=tuple(inter),
        symmetric_names=tuple(sym),
        interleaved_to_symmetric_idx=idx_formula.astype(int),
        interleaved_source=str(inter_path),
        symmetric_source=str(sym_path),
    )


def resolve_roi_order_names(ref: ROIOrderReference, mode: str) -> tuple[list[str], list[str]]:
    """Resolve FC and SC ROI name order after applying a given reorder mode."""
    idx = np.asarray(ref.interleaved_to_symmetric_idx, dtype=int)
    fc0 = list(ref.interleaved_names)
    sc0 = list(ref.symmetric_names)

    if mode not in {"none", "aal90_fc", "aal90_sc", "aal90_both"}:
        raise ValueError(f"Unsupported reorder mode for name resolution: {mode}")

    fc = [fc0[i] for i in idx] if mode in {"aal90_fc", "aal90_both"} else fc0
    sc = [sc0[i] for i in idx] if mode in {"aal90_sc", "aal90_both"} else sc0
    return fc, sc


def validate_subject_alignment_qc(load_qc_df: pd.DataFrame) -> dict[str, Any]:
    """Validate subject-level FC/SC alignment metadata and fail loudly on inconsistencies."""
    if load_qc_df.empty:
        raise RuntimeError("subject_loading_qc is empty; no FC/SC pairs to validate.")

    required_cols = {
        "source_fc_file",
        "source_sc_file",
        "source_subject_index",
        "n_regions",
        "fc_subject_name",
        "sc_subject_name",
        "has_fc_subject_names",
        "has_sc_subject_names",
    }
    missing = sorted(required_cols - set(load_qc_df.columns))
    if missing:
        raise RuntimeError(f"subject_loading_qc missing required columns: {missing}")

    dup = load_qc_df.duplicated(subset=["source_fc_file", "source_sc_file", "source_subject_index"]).sum()
    if int(dup) > 0:
        raise RuntimeError(f"Detected {int(dup)} duplicate (fc_file, sc_file, subject_index) rows in loading QC.")

    bad_regions = load_qc_df[load_qc_df["n_regions"].astype(int) != 90]
    if not bad_regions.empty:
        raise RuntimeError(
            "Found subjects with n_regions != 90. "
            f"First offending subject: {bad_regions.iloc[0]['subject_id']}"
        )

    both_names = load_qc_df[
        load_qc_df["has_fc_subject_names"].astype(bool) & load_qc_df["has_sc_subject_names"].astype(bool)
    ].copy()
    if not both_names.empty:
        mism = both_names[both_names["fc_subject_name"].astype(str) != both_names["sc_subject_name"].astype(str)]
        if not mism.empty:
            first = mism.iloc[0]
            raise RuntimeError(
                "FC/SC subject-name mismatch at same source index. "
                f"subject_id={first['subject_id']}, fc_name={first['fc_subject_name']}, "
                f"sc_name={first['sc_subject_name']}."
            )

    pair_counts = (
        load_qc_df.groupby(["source_fc_file", "source_sc_file"], as_index=False)["source_subject_index"]
        .count()
        .rename(columns={"source_subject_index": "n_subjects"})
        .sort_values(["source_fc_file", "source_sc_file"])
    )
    return {
        "n_rows": int(load_qc_df.shape[0]),
        "n_unique_pairs": int(pair_counts.shape[0]),
        "pair_counts": pair_counts.to_dict(orient="records"),
        "n_with_both_subject_name_arrays": int(both_names.shape[0]),
    }


def validate_final_roi_order_or_raise(ref: ROIOrderReference, applied_mode: str) -> dict[str, Any]:
    """Validate final FC/SC ROI order equality after configured reordering."""
    fc_final, sc_final = resolve_roi_order_names(ref, applied_mode)
    aligned = fc_final == sc_final
    if not aligned:
        mismatch_rows = [
            {"position_1based": i + 1, "fc_roi": fc_final[i], "sc_roi": sc_final[i]}
            for i in range(len(fc_final))
            if fc_final[i] != sc_final[i]
        ]
        first = mismatch_rows[0]
        raise RuntimeError(
            "Final FC/SC ROI orders are not aligned. "
            f"First mismatch at position {first['position_1based']}: "
            f"fc={first['fc_roi']}, sc={first['sc_roi']}. "
            "Use reorder mode 'aal90_fc' for this dataset."
        )
    return {
        "aligned": True,
        "final_order_name": "aal90_symmetric_left_then_right_reverse",
        "n_regions": len(fc_final),
    }


def load_new_doc_subjects(data_root: Path, max_subjects_per_group: int | None = None) -> tuple[list[SubjectRecord], pd.DataFrame]:
    records: list[SubjectRecord] = []
    counter_by_cohort: dict[str, int] = defaultdict(int)
    qc_rows: list[dict[str, Any]] = []

    for spec in FILE_SPECS:
        cohort = str(spec["cohort"])
        if max_subjects_per_group is not None and counter_by_cohort[cohort] >= max_subjects_per_group:
            continue

        fc_path = data_root / str(spec["fc_path"])
        sc_path = data_root / str(spec["sc_path"])
        fc_mapping = _load_mat_mapping(fc_path)
        sc_mapping = _load_mat_mapping(sc_path)

        fc_var, fc_arr = _select_3d_numeric(fc_mapping, variable_hint=str(spec["fc_var"]))
        sc_var, sc_arr = _select_3d_numeric(sc_mapping, variable_hint=str(spec["sc_var"]))

        fc_srt = _to_subject_roi_time(fc_arr)  # (subjects, 90, T)
        sc_srr = _to_subject_roi_roi(sc_arr)  # (subjects, 90, 90)

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
            source_label = (
                fc_subject_names[i]
                if fc_subject_names
                else f"idx{i + 1:03d}"
            )
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


def _maybe_apply_roi_reordering(
    records: list[SubjectRecord],
    mode: str,
) -> tuple[list[SubjectRecord], pd.DataFrame, dict[str, Any]]:
    # Backward-compatible alias used by older helper scripts.
    if mode == "apply":
        mode = "aal90_fc"

    idx = _aal90_symmetry_reorder_index()

    qc_rows: list[dict[str, Any]] = []
    for rec in records:
        fc_id = _compute_static_fc(rec.timeseries)
        sc_id = rec.sc_matrix

        fc_vec_id = _upper_triangle_vector(fc_id)
        sc_vec_id = _upper_triangle_vector(sc_id)
        r_identity = _safe_pearson(fc_vec_id, sc_vec_id)

        fc_re = fc_id[np.ix_(idx, idx)]
        sc_re = sc_id[np.ix_(idx, idx)]

        r_reorder_fc = _safe_pearson(_upper_triangle_vector(fc_re), sc_vec_id)
        r_reorder_sc = _safe_pearson(fc_vec_id, _upper_triangle_vector(sc_re))
        r_reorder_both = _safe_pearson(_upper_triangle_vector(fc_re), _upper_triangle_vector(sc_re))

        qc_rows.append(
            {
                "subject_id": rec.subject_id,
                "cohort": rec.cohort,
                "stage": rec.stage,
                "sedation": rec.sedation,
                "coupling_identity": r_identity,
                "coupling_reorder_fc": r_reorder_fc,
                "coupling_reorder_sc": r_reorder_sc,
                "coupling_reorder_both": r_reorder_both,
                "delta_reorder_fc_minus_identity": r_reorder_fc - r_identity,
                "delta_reorder_sc_minus_identity": r_reorder_sc - r_identity,
            }
        )

    qc_df = pd.DataFrame(qc_rows)

    if mode == "auto":
        mean_delta_fc = float(qc_df["delta_reorder_fc_minus_identity"].mean())
        mean_delta_sc = float(qc_df["delta_reorder_sc_minus_identity"].mean())
        mode_use = "aal90_fc" if mean_delta_fc >= mean_delta_sc else "aal90_sc"
    else:
        mode_use = mode

    out: list[SubjectRecord] = []
    for rec in records:
        ts = rec.timeseries
        sc = rec.sc_matrix
        if mode_use in {"aal90_fc", "aal90_both"}:
            ts = ts[:, idx]
        if mode_use in {"aal90_sc", "aal90_both"}:
            sc = sc[np.ix_(idx, idx)]

        out.append(
            SubjectRecord(
                cohort=rec.cohort,
                subject_id=rec.subject_id,
                stage=rec.stage,
                sedation=rec.sedation,
                source_fc_file=rec.source_fc_file,
                source_sc_file=rec.source_sc_file,
                source_subject_index=rec.source_subject_index,
                source_subject_label=rec.source_subject_label,
                timeseries=np.asarray(ts, dtype=float),
                sc_matrix=np.asarray(sc, dtype=float),
            )
        )

    decision = {
        "requested_mode": mode,
        "applied_mode": mode_use,
        "aal90_index": idx.tolist(),
        "mean_identity": float(qc_df["coupling_identity"].mean()),
        "mean_reorder_fc": float(qc_df["coupling_reorder_fc"].mean()),
        "mean_reorder_sc": float(qc_df["coupling_reorder_sc"].mean()),
        "mean_reorder_both": float(qc_df["coupling_reorder_both"].mean()),
        "n_subjects": int(qc_df.shape[0]),
    }
    return out, qc_df, decision


def _compute_transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    tm = np.zeros((n_states, n_states), dtype=float)
    if labels.size < 2:
        return tm
    for i in range(labels.size - 1):
        a = int(labels[i])
        b = int(labels[i + 1])
        if 0 <= a < n_states and 0 <= b < n_states:
            tm[a, b] += 1.0
    rs = tm.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return tm / rs


def _compute_occupancy(labels: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_states)
    total = max(int(np.asarray(labels).size), 1)
    return counts.astype(float) / float(total)


def _collapse_runs(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=int)
    if arr.size == 0:
        return arr
    out = [int(arr[0])]
    for x in arr[1:]:
        xv = int(x)
        if xv != out[-1]:
            out.append(xv)
    return np.asarray(out, dtype=int)


def _markov_transition_no_self(labels: np.ndarray, n_states: int, collapse_runs: bool = True) -> np.ndarray:
    seq = _collapse_runs(labels) if collapse_runs else np.asarray(labels, dtype=int)
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
    k = P.shape[0]
    mu = np.ones(k, dtype=float) / float(max(k, 1))
    for _ in range(max_iter):
        nxt = mu @ P
        if np.allclose(nxt, mu, atol=tol, rtol=0.0):
            break
        mu = nxt
    return mu


def _markov_entropy_rate_bits(P: np.ndarray) -> float:
    mu = _stationary_distribution(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.where(P > 0, P * np.log2(P), 0.0)
    return float(-np.sum(mu[:, None] * inner))


def _markov_entropy_rate_norm(P: np.ndarray) -> float:
    k = int(P.shape[0])
    if k <= 1:
        return 0.0
    return float(_markov_entropy_rate_bits(P) / np.log2(k))


def _sanitize_timeseries(timeseries: np.ndarray) -> np.ndarray:
    """Validate finite input and return a numeric copy."""
    x = np.asarray(timeseries, dtype=float).copy()
    if x.ndim != 2:
        raise ValueError(f"Expected timeseries shape (T,R), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Timeseries contains non-finite values.")
    return x


def _extract_phase_patterns_clean(
    timeseries: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    x = _sanitize_timeseries(timeseries)
    patterns, global_sync, _, _ = phase_patterns(
        x,
        trim_edge_samples=args.trim_edge_samples,
        pipeline=args.pipeline,
        tr_seconds=args.tr_seconds,
        bandpass_hz=(args.bandpass_low_hz, args.bandpass_high_hz),
        filter_order=args.filter_order,
    )

    if patterns.size == 0:
        raise RuntimeError("No valid phase-pattern samples after preprocessing.")

    good = np.all(np.isfinite(patterns), axis=1)
    patterns = np.asarray(patterns[good], dtype=np.float32)
    global_sync = np.asarray(global_sync, dtype=float)[good]
    if patterns.shape[0] == 0:
        raise RuntimeError("All phase-pattern rows were non-finite.")

    return patterns, global_sync, int(good.size), int(np.sum(good))


def _summarize_brain_states_robust(
    timeseries: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, np.ndarray]:
    """Brain-state summary with explicit finite-row filtering before clustering."""
    patterns, global_sync, n_raw, n_kept = _extract_phase_patterns_clean(timeseries, args)

    labels, centers = cluster_brain_states(
        patterns,
        n_states=args.n_states,
        random_seed=seed,
        n_init=args.n_init_subject,
        max_iter=args.max_iter_subject,
        backend=args.clustering_backend,
    )

    centers = np.asarray(centers, dtype=float)
    labels = np.asarray(labels, dtype=int)
    k_eff = int(centers.shape[0]) if centers.ndim == 2 else 0
    occupancy = _compute_occupancy(labels, n_states=max(k_eff, 1))
    transitions = _compute_transition_matrix(labels, n_states=max(k_eff, 1))

    return {
        "labels": labels,
        "centers": centers,
        "occupancy": occupancy,
        "transition_matrix": transitions,
        "global_sync": np.asarray(global_sync, dtype=float),
        "n_pattern_rows_raw": np.asarray([int(n_raw)], dtype=int),
        "n_pattern_rows_kept": np.asarray([int(n_kept)], dtype=int),
    }


def _subject_local_analysis(rec: SubjectRecord, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    bs = _summarize_brain_states_robust(rec.timeseries, args=args, seed=seed)

    centers = np.asarray(bs["centers"], dtype=float)
    labels = np.asarray(bs["labels"], dtype=int)
    occupancy = np.asarray(bs["occupancy"], dtype=float)

    if centers.ndim != 2:
        raise RuntimeError(f"{rec.subject_id}: invalid centers shape {centers.shape}")
    if labels.ndim != 1:
        raise RuntimeError(f"{rec.subject_id}: invalid labels shape {labels.shape}")
    if occupancy.ndim != 1:
        raise RuntimeError(f"{rec.subject_id}: invalid occupancy shape {occupancy.shape}")

    k_eff = int(centers.shape[0])
    if k_eff != args.n_states:
        msg = f"{rec.subject_id}: expected k={args.n_states}, got k={k_eff}"
        if args.strict_k:
            raise RuntimeError(msg)
        print(f"WARNING {msg}; skipping subject.")
        return {}

    sc_vec = _upper_triangle_vector(rec.sc_matrix)
    sc_full_vec = _full_matrix_vector(rec.sc_matrix)
    sfc_local = np.asarray([_safe_pearson(c, sc_vec) for c in centers], dtype=float)
    center_mats = np.asarray([_vector_to_sym_matrix(c) for c in centers], dtype=float)
    sfc_local_full = np.asarray(
        [_safe_pearson(_full_matrix_vector(m), sc_full_vec) for m in center_mats],
        dtype=float,
    )

    # Rank local states by SC-coupling (ascending) for within-subject post-hoc alignment.
    sort_idx = np.argsort(np.nan_to_num(sfc_local, nan=np.inf))
    rank_of_local = np.empty(k_eff, dtype=int)
    rank_of_local[sort_idx] = np.arange(k_eff)

    labels_rank = rank_of_local[labels]
    transition_rank = _compute_transition_matrix(labels_rank, n_states=k_eff)
    transition_rank_no_self = _markov_transition_no_self(labels_rank, n_states=k_eff, collapse_runs=True)

    fc_matrix = _compute_static_fc(rec.timeseries)
    fc_vec = _upper_triangle_vector(fc_matrix)
    static_fc_sc = _safe_pearson(fc_vec, sc_vec)
    static_fc_sc_full = _safe_pearson(_full_matrix_vector(fc_matrix), sc_full_vec)

    return {
        "subject_id": rec.subject_id,
        "cohort": rec.cohort,
        "stage": rec.stage,
        "sedation": rec.sedation,
        "source_fc_file": rec.source_fc_file,
        "source_sc_file": rec.source_sc_file,
        "source_subject_index": rec.source_subject_index,
        "source_subject_label": rec.source_subject_label,
        "n_timepoints": int(rec.timeseries.shape[0]),
        "n_regions": int(rec.timeseries.shape[1]),
        "global_sync_mean": float(np.mean(np.asarray(bs["global_sync"], dtype=float))),
        "global_sync_std": float(np.std(np.asarray(bs["global_sync"], dtype=float))),
        "n_pattern_rows_raw": int(np.asarray(bs["n_pattern_rows_raw"]).reshape(-1)[0]),
        "n_pattern_rows_kept": int(np.asarray(bs["n_pattern_rows_kept"]).reshape(-1)[0]),
        "static_fc_sc_coupling": float(static_fc_sc),
        "static_fc_sc_coupling_full": float(static_fc_sc_full),
        "centers": centers,
        "center_matrices": center_mats,
        "labels_local": labels,
        "labels_rank": labels_rank,
        "occupancy_local": occupancy,
        "sfc_local": sfc_local,
        "sfc_local_full": sfc_local_full,
        "rank_sort_local_indices": sort_idx,
        "transition_local": np.asarray(bs["transition_matrix"], dtype=float),
        "transition_rank": transition_rank,
        "transition_rank_no_self": transition_rank_no_self,
        "entropy_rate_bits": float(_markov_entropy_rate_bits(transition_rank_no_self)),
        "entropy_rate_norm": float(_markov_entropy_rate_norm(transition_rank_no_self)),
    }


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


def _compute_rank_stats(rank_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    om_rows: list[dict[str, Any]] = []
    pw_rows: list[dict[str, Any]] = []

    for rank in sorted(rank_df["state_rank"].unique()):
        dt = rank_df[rank_df["state_rank"] == rank]
        samples = {
            c: dt.loc[dt["cohort"] == c, "occupancy"].to_numpy(dtype=float)
            for c in COHORTS
            if not dt.loc[dt["cohort"] == c].empty
        }
        if len(samples) < 2:
            continue

        try:
            h, p = kruskal(*samples.values())
        except ValueError:
            continue

        om_rows.append(
            {
                "state_rank": int(rank),
                "test": "Kruskal-Wallis",
                "H": float(h),
                "p": float(p),
            }
        )

        p_raw: list[float] = []
        idx_rows: list[int] = []
        keys = list(samples.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = keys[i]
                b = keys[j]
                xa = samples[a]
                xb = samples[b]
                u, pv = mannwhitneyu(xa, xb, alternative="two-sided")
                p_raw.append(float(pv))
                idx_rows.append(len(pw_rows))
                pw_rows.append(
                    {
                        "state_rank": int(rank),
                        "contrast": f"{a} vs {b}",
                        "n_a": int(xa.size),
                        "n_b": int(xb.size),
                        "U": float(u),
                        "p_raw": float(pv),
                        "p_holm": np.nan,
                        "median_a": float(np.median(xa)),
                        "median_b": float(np.median(xb)),
                    }
                )
        p_holm = _holm_correct(p_raw)
        for ii, val in zip(idx_rows, p_holm):
            pw_rows[ii]["p_holm"] = float(val)

    return pd.DataFrame(om_rows), pd.DataFrame(pw_rows)


def _plot_reorder_qc(qc_df: pd.DataFrame, decision: dict[str, Any], out_dir: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    for cohort in COHORTS:
        dc = qc_df[qc_df["cohort"] == cohort]
        if dc.empty:
            continue
        ax.scatter(
            dc["coupling_identity"],
            dc["coupling_reorder_fc"],
            s=26,
            alpha=0.72,
            color=PALETTE[cohort],
            label=cohort.upper(),
            edgecolors="black",
            linewidths=0.25,
        )

    mn = float(np.nanmin(np.r_[qc_df["coupling_identity"].to_numpy(), qc_df["coupling_reorder_fc"].to_numpy()]))
    mx = float(np.nanmax(np.r_[qc_df["coupling_identity"].to_numpy(), qc_df["coupling_reorder_fc"].to_numpy()]))
    pad = 0.02 * (mx - mn + 1e-6)
    lo, hi = mn - pad, mx + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Static FC-SC coupling (identity ROI order)")
    ax.set_ylabel("Static FC-SC coupling (FC reordered to AAL90 symmetric)")
    ax.set_title(f"ROI-order QC (applied mode: {decision['applied_mode']})")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "qc_roi_order_identity_vs_reordered_fc")


def _plot_occupancy_by_rank(rank_df: pd.DataFrame, n_states: int, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, n_states, figsize=(3.2 * n_states, 4.9), sharey=True)
    if n_states == 1:
        axes = [axes]

    for rank in range(1, n_states + 1):
        ax = axes[rank - 1]
        dt = rank_df[rank_df["state_rank"] == rank]
        for ci, cohort in enumerate(COHORTS):
            vals = dt.loc[dt["cohort"] == cohort, "occupancy"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            jitter = np.linspace(-0.1, 0.1, vals.size)
            ax.scatter(
                np.full(vals.size, ci, dtype=float) + jitter,
                vals,
                s=20,
                alpha=0.70,
                color=PALETTE[cohort],
                edgecolors="black",
                linewidths=0.2,
            )
            med = float(np.median(vals))
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            ax.plot([ci - 0.18, ci + 0.18], [med, med], color="black", lw=1.4)
            ax.vlines(ci, q1, q3, color="black", lw=1.0)

        ax.set_title(f"Rank {rank}")
        ax.set_xticks(np.arange(len(COHORTS)), [c.upper() for c in COHORTS], rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        if rank == 1:
            ax.set_ylabel("Occupancy")

    fig.suptitle("Subject-level Occupancy by Within-subject SC-coupling Rank", y=1.03)
    fig.tight_layout()
    _save_figure(fig, out_dir, "occupancy_by_sfc_rank")


def _plot_mean_occupancy_by_rank(rank_df: pd.DataFrame, n_states: int, out_dir: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.4, 5.2))

    x = np.arange(n_states)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(COHORTS))
    for off, cohort in zip(offsets, COHORTS):
        means = []
        sems = []
        for rank in range(1, n_states + 1):
            vals = rank_df.loc[
                (rank_df["cohort"] == cohort) & (rank_df["state_rank"] == rank),
                "occupancy",
            ].to_numpy(dtype=float)
            means.append(float(np.mean(vals)) if vals.size else np.nan)
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
        ax.bar(x + off, means, yerr=sems, width=width, capsize=3, color=PALETTE[cohort], alpha=0.9, label=cohort.upper())

    ax.set_xticks(x, [f"R{r}" for r in range(1, n_states + 1)])
    ax.set_xlabel("Within-subject state rank by SC coupling")
    ax.set_ylabel("Mean occupancy")
    ax.set_title("Cohort Mean Occupancy (Rank-aligned subject-local states)")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, out_dir, "cohort_mean_occupancy_by_sfc_rank")


def _plot_sfc_vs_occupancy(
    rank_df: pd.DataFrame,
    out_dir: Path,
    *,
    x_col: str = "sfc",
    stem: str = "subject_level_sfc_vs_occupancy",
    title: str = "Subject-level Occupancy vs SC-coupling",
    x_label: str = "State SC-coupling (Pearson r)",
) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.3))

    for cohort in COHORTS:
        dc = rank_df[rank_df["cohort"] == cohort]
        if dc.empty:
            continue
        x = dc[x_col].to_numpy(dtype=float)
        y = dc["occupancy"].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size == 0:
            continue

        ax.scatter(
            x,
            y,
            s=20,
            alpha=0.40,
            color=PALETTE[cohort],
            edgecolors="none",
            label=f"{cohort.upper()} states",
        )
        if x.size >= 3 and float(np.std(x)) > 0.0:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            yy = p[0] * xx + p[1]
            ax.plot(xx, yy, color=PALETTE[cohort], lw=2.0, alpha=0.92)

    ax.set_xlabel(x_label)
    ax.set_ylabel("State occupancy (subject-local)")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, out_dir, stem)


def _plot_sfc_method_comparison(rank_df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharey=True)
    specs = [
        ("sfc", "Upper-triangle coupling"),
        ("sfc_full", "Full-matrix flatten coupling (MATLAB-style)"),
    ]
    for ax, (x_col, ttl) in zip(axes, specs):
        for cohort in COHORTS:
            dc = rank_df[rank_df["cohort"] == cohort]
            if dc.empty:
                continue
            x = dc[x_col].to_numpy(dtype=float)
            y = dc["occupancy"].to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            x = x[good]
            y = y[good]
            if x.size == 0:
                continue
            ax.scatter(x, y, s=18, alpha=0.35, color=PALETTE[cohort], edgecolors="none", label=cohort.upper())
            if x.size >= 3 and float(np.std(x)) > 0.0:
                p = np.polyfit(x, y, 1)
                xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                yy = p[0] * xx + p[1]
                ax.plot(xx, yy, color=PALETTE[cohort], lw=2.0, alpha=0.9)
        ax.set_title(ttl)
        ax.set_xlabel("State SC-coupling (Pearson r)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("State occupancy (subject-local)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Occupancy vs SC-coupling: coupling-method comparison", y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_dir, "subject_level_sfc_vs_occupancy_method_comparison")


def _summarize_sfc_occupancy_methods(rank_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        dc = rank_df[rank_df["cohort"] == cohort]
        if dc.empty:
            continue
        y = dc["occupancy"].to_numpy(dtype=float)
        for col, label in [("sfc", "upper_triangle"), ("sfc_full", "full_flatten")]:
            x = dc[col].to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            xv = x[good]
            yv = y[good]
            if xv.size < 2:
                rows.append(
                    {
                        "cohort": cohort,
                        "method": label,
                        "n_points": int(xv.size),
                        "pearson_r": np.nan,
                        "slope": np.nan,
                        "intercept": np.nan,
                    }
                )
                continue
            r = _safe_pearson(xv, yv)
            if float(np.std(xv)) > 0.0:
                p = np.polyfit(xv, yv, deg=1)
                slope = float(p[0])
                intercept = float(p[1])
            else:
                slope = np.nan
                intercept = np.nan
            rows.append(
                {
                    "cohort": cohort,
                    "method": label,
                    "n_points": int(xv.size),
                    "pearson_r": float(r),
                    "slope": slope,
                    "intercept": intercept,
                }
            )
    return pd.DataFrame(rows).sort_values(["cohort", "method"]).reset_index(drop=True)


def _plot_static_fc_sc(subject_df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    vals = []
    labels = []
    for cohort in COHORTS:
        x = subject_df.loc[subject_df["cohort"] == cohort, "static_fc_sc_coupling"].to_numpy(dtype=float)
        if x.size == 0:
            continue
        vals.append(x)
        labels.append(cohort)

    if vals:
        vp = ax.violinplot(vals, showmeans=True, showextrema=False)
        for i, b in enumerate(vp["bodies"]):
            c = labels[i]
            b.set_facecolor(PALETTE[c])
            b.set_edgecolor("#111111")
            b.set_alpha(0.52)
        for i, x in enumerate(vals, start=1):
            jitter = np.linspace(-0.08, 0.08, x.size)
            ax.scatter(np.full(x.size, i) + jitter, x, s=18, alpha=0.65, color="#111111")

    ax.set_xticks(np.arange(1, len(labels) + 1), [x.upper() for x in labels])
    ax.set_ylabel("Static FC-SC coupling (Pearson r)")
    ax.set_title("Subject-level FC-SC Coupling by Cohort")
    fig.tight_layout()
    _save_figure(fig, out_dir, "subject_level_static_fc_sc_coupling")


def _plot_transition_heatmaps(cohort_tmean: dict[str, np.ndarray], n_states: int, out_dir: Path, stem: str, title: str) -> None:
    set_publication_style()

    vmax = 0.0
    for c in COHORTS:
        arr = cohort_tmean.get(c)
        if arr is not None and arr.size:
            vmax = max(vmax, float(np.max(arr)))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.flatten()
    im = None

    for ax, cohort in zip(axes_flat, COHORTS):
        arr = cohort_tmean.get(cohort, np.zeros((n_states, n_states), dtype=float))
        im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(cohort.upper())
        ax.set_xticks(np.arange(n_states), [f"R{i}" for i in range(1, n_states + 1)])
        ax.set_yticks(np.arange(n_states), [f"R{i}" for i in range(1, n_states + 1)])
        ax.set_xlabel("Next rank-state")
        ax.set_ylabel("Current rank-state")

    if im is not None:
        fig.colorbar(im, ax=axes_flat, fraction=0.025, pad=0.02, label="Transition probability")
    fig.suptitle(title, y=0.98)
    _save_figure(fig, out_dir, stem)


def _plot_markov_entropy(subject_df: pd.DataFrame, out_dir: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)

    metrics = [
        ("entropy_rate_bits", "Entropy rate (bits)"),
        ("entropy_rate_norm", "Normalized entropy rate"),
    ]

    for ax, (col, ylab) in zip(axes, metrics):
        vals = []
        labels = []
        for cohort in COHORTS:
            x = subject_df.loc[subject_df["cohort"] == cohort, col].to_numpy(dtype=float)
            if x.size == 0:
                continue
            vals.append(x)
            labels.append(cohort)

        if vals:
            vp = ax.violinplot(vals, showmeans=True, showextrema=False)
            for i, b in enumerate(vp["bodies"]):
                c = labels[i]
                b.set_facecolor(PALETTE[c])
                b.set_edgecolor("#111111")
                b.set_alpha(0.52)
            for i, x in enumerate(vals, start=1):
                jitter = np.linspace(-0.08, 0.08, x.size)
                ax.scatter(np.full(x.size, i) + jitter, x, s=16, alpha=0.60, color="#111111")

        ax.set_xticks(np.arange(1, len(labels) + 1), [x.upper() for x in labels])
        ax.set_ylabel(ylab)
        ax.set_title(ylab)

    fig.suptitle("Subject-level Markov entropy from rank-aligned local states", y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_dir, "subject_level_markov_entropy")


def _vector_to_sym_matrix(vec: np.ndarray) -> np.ndarray:
    m = squareform(np.asarray(vec, dtype=float))
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 0.0)
    return m


def _plot_mean_state_matrices_by_rank(
    subject_results: list[dict[str, Any]],
    n_states: int,
    out_dir: Path,
) -> None:
    set_publication_style()

    mats: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for res in subject_results:
        centers = np.asarray(res["centers"], dtype=float)
        order = np.asarray(res["rank_sort_local_indices"], dtype=int)
        for rank0, local_idx in enumerate(order):
            mats[(str(res["cohort"]), int(rank0) + 1)].append(_vector_to_sym_matrix(centers[int(local_idx)]))

    all_mean_vals = []
    mean_mats: dict[tuple[str, int], np.ndarray] = {}
    for cohort in COHORTS:
        for rank in range(1, n_states + 1):
            key = (cohort, rank)
            arrs = mats.get(key, [])
            if not arrs:
                mean_mats[key] = np.zeros((90, 90), dtype=float)
            else:
                mean_mats[key] = np.mean(np.stack(arrs, axis=0), axis=0)
            all_mean_vals.append(mean_mats[key])

    vmax = max(0.15, float(np.max(np.abs(np.stack(all_mean_vals, axis=0)))))

    fig, axes = plt.subplots(len(COHORTS), n_states, figsize=(3.0 * n_states, 2.7 * len(COHORTS)), sharex=True, sharey=True)
    if len(COHORTS) == 1:
        axes = np.asarray([axes])
    if n_states == 1:
        axes = np.asarray([[ax] for ax in axes.reshape(-1)])

    im = None
    for ri, cohort in enumerate(COHORTS):
        for ci in range(n_states):
            ax = axes[ri, ci]
            mat = mean_mats[(cohort, ci + 1)]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"Rank {ci + 1}")
            if ci == 0:
                ax.set_ylabel(cohort.upper())

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="Mean phase-coherence weight")
    fig.suptitle("Group mean local-state centroids (rank-aligned)", y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_dir, "group_mean_state_centroids_by_rank")


def _run_secondary_pooled_state_analysis(
    records_valid: list[SubjectRecord],
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, Any]:
    """Secondary pooled-state analysis across all subject timepoints."""
    sec_root = out_root / "secondary_pooled"
    sec_tables = sec_root / "tables"
    sec_figs = sec_root / "figures"
    sec_npz = sec_root / "npz"
    sec_logs = sec_root / "logs"
    for p in (sec_tables, sec_figs, sec_npz, sec_logs):
        p.mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict[str, Any]] = []
    pooled_patterns: list[np.ndarray] = []
    offset = 0
    for rec in records_valid:
        patt, _, n_raw, n_kept = _extract_phase_patterns_clean(rec.timeseries, args)
        start = offset
        end = offset + int(patt.shape[0])
        offset = end
        pooled_patterns.append(np.asarray(patt, dtype=np.float32))
        meta_rows.append(
            {
                "subject_id": rec.subject_id,
                "cohort": rec.cohort,
                "stage": rec.stage,
                "sedation": rec.sedation,
                "start": int(start),
                "end": int(end),
                "n_pattern_rows_raw": int(n_raw),
                "n_pattern_rows_kept": int(n_kept),
            }
        )

    if not pooled_patterns:
        raise RuntimeError("Secondary pooled analysis failed: no pooled patterns available.")

    X = np.vstack(pooled_patterns).astype(np.float32, copy=False)
    labels_all, centers = cluster_brain_states(
        X,
        n_states=args.n_states,
        random_seed=args.random_seed_pooled,
        n_init=args.n_init_pooled,
        max_iter=args.max_iter_pooled,
        backend=args.clustering_backend,
    )
    labels_all = np.asarray(labels_all, dtype=int)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] != args.n_states:
        raise RuntimeError(
            f"Secondary pooled analysis expected k={args.n_states}, got centers shape={centers.shape}."
        )

    # Build pooled SC reference for state ordering.
    sc_all = np.stack([np.asarray(r.sc_matrix, dtype=float) for r in records_valid], axis=0)
    sc_ref_all = np.mean(sc_all, axis=0)
    sc_ctrl = [np.asarray(r.sc_matrix, dtype=float) for r in records_valid if str(r.cohort) == "control"]
    sc_ref_control = np.mean(np.stack(sc_ctrl, axis=0), axis=0) if sc_ctrl else sc_ref_all
    if args.pooled_sc_reference == "control":
        sc_ref = sc_ref_control
    else:
        sc_ref = sc_ref_all

    sc_ref_vec = _upper_triangle_vector(sc_ref)
    sc_ref_full_vec = _full_matrix_vector(sc_ref)
    center_mats = np.asarray([_vector_to_sym_matrix(c) for c in centers], dtype=float)
    sfc_ref = np.asarray([_safe_pearson(c, sc_ref_vec) for c in centers], dtype=float)
    sfc_ref_full = np.asarray(
        [_safe_pearson(_full_matrix_vector(m), sc_ref_full_vec) for m in center_mats],
        dtype=float,
    )
    order = np.argsort(np.nan_to_num(sfc_ref, nan=np.inf))
    rank_of_state = np.empty(args.n_states, dtype=int)
    rank_of_state[order] = np.arange(args.n_states)

    state_ref_rows: list[dict[str, Any]] = []
    for s in range(args.n_states):
        state_ref_rows.append(
            {
                "pooled_state": int(s) + 1,
                "state_rank": int(rank_of_state[s]) + 1,
                "sfc_pooled_sc_ref": float(sfc_ref[s]),
                "sfc_pooled_sc_ref_full": float(sfc_ref_full[s]),
            }
        )

    subject_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    transition_no_self_rows: list[dict[str, Any]] = []

    meta_df = pd.DataFrame(meta_rows)
    for rec, m in zip(records_valid, meta_rows):
        a = int(m["start"])
        b = int(m["end"])
        lab = labels_all[a:b]
        if lab.size == 0:
            continue
        occ_state = _compute_occupancy(lab, n_states=args.n_states)
        lab_rank = rank_of_state[lab]
        occ_rank = _compute_occupancy(lab_rank, n_states=args.n_states)
        tm_rank = _compute_transition_matrix(lab_rank, n_states=args.n_states)
        tm_rank_ns = _markov_transition_no_self(lab_rank, n_states=args.n_states, collapse_runs=True)

        sc_vec_sub = _upper_triangle_vector(rec.sc_matrix)
        sc_full_sub = _full_matrix_vector(rec.sc_matrix)
        sfc_sub = np.asarray([_safe_pearson(c, sc_vec_sub) for c in centers], dtype=float)
        sfc_sub_full = np.asarray(
            [_safe_pearson(_full_matrix_vector(m), sc_full_sub) for m in center_mats],
            dtype=float,
        )
        static_fc_sc = _safe_pearson(_upper_triangle_vector(_compute_static_fc(rec.timeseries)), sc_vec_sub)
        static_fc_sc_full = _safe_pearson(_full_matrix_vector(_compute_static_fc(rec.timeseries)), sc_full_sub)

        subject_rows.append(
            {
                "subject_id": rec.subject_id,
                "cohort": rec.cohort,
                "stage": rec.stage,
                "sedation": rec.sedation,
                "n_pattern_rows_kept": int(lab.size),
                "static_fc_sc_coupling": float(static_fc_sc),
                "static_fc_sc_coupling_full": float(static_fc_sc_full),
                "entropy_rate_bits": float(_markov_entropy_rate_bits(tm_rank_ns)),
                "entropy_rate_norm": float(_markov_entropy_rate_norm(tm_rank_ns)),
            }
        )

        for s in range(args.n_states):
            state_rows.append(
                {
                    "subject_id": rec.subject_id,
                    "cohort": rec.cohort,
                    "stage": rec.stage,
                    "sedation": rec.sedation,
                    "pooled_state": int(s) + 1,
                    "state_rank": int(rank_of_state[s]) + 1,
                    "occupancy": float(occ_state[s]),
                    "sfc_subject": float(sfc_sub[s]),
                    "sfc_subject_full": float(sfc_sub_full[s]),
                    "sfc_pooled_sc_ref": float(sfc_ref[s]),
                    "sfc_pooled_sc_ref_full": float(sfc_ref_full[s]),
                }
            )

        for rank0, s_idx in enumerate(order):
            rank_rows.append(
                {
                    "subject_id": rec.subject_id,
                    "cohort": rec.cohort,
                    "stage": rec.stage,
                    "sedation": rec.sedation,
                    "state_rank": int(rank0) + 1,
                    "pooled_state": int(s_idx) + 1,
                    "occupancy": float(occ_state[s_idx]),
                    "sfc": float(sfc_sub[s_idx]),
                    "sfc_full": float(sfc_sub_full[s_idx]),
                }
            )

        for i in range(args.n_states):
            for j in range(args.n_states):
                transition_rows.append(
                    {
                        "subject_id": rec.subject_id,
                        "cohort": rec.cohort,
                        "from_state_rank": int(i) + 1,
                        "to_state_rank": int(j) + 1,
                        "probability": float(tm_rank[i, j]),
                    }
                )
                transition_no_self_rows.append(
                    {
                        "subject_id": rec.subject_id,
                        "cohort": rec.cohort,
                        "from_state_rank": int(i) + 1,
                        "to_state_rank": int(j) + 1,
                        "probability": float(tm_rank_ns[i, j]),
                    }
                )

    subject_df = pd.DataFrame(subject_rows).sort_values(["cohort", "subject_id"]).reset_index(drop=True)
    state_df = pd.DataFrame(state_rows)
    rank_df = pd.DataFrame(rank_rows)
    tr_df = pd.DataFrame(transition_rows)
    tr_ns_df = pd.DataFrame(transition_no_self_rows)
    state_ref_df = pd.DataFrame(state_ref_rows).sort_values("state_rank")
    sfc_method_summary = _summarize_sfc_occupancy_methods(rank_df)

    # Cohort-level occupancy denominator comparison: condition-specific vs global denominator.
    denom_rows: list[dict[str, Any]] = []
    total_rows = int(labels_all.size)
    for cohort in COHORTS:
        ds = state_df[state_df["cohort"] == cohort]
        if ds.empty:
            continue
        denom_condition = int(meta_df.loc[meta_df["cohort"] == cohort, "n_pattern_rows_kept"].sum())
        for s in range(1, args.n_states + 1):
            count_exact = 0
            for m in meta_rows:
                if m["cohort"] != cohort:
                    continue
                a = int(m["start"])
                b = int(m["end"])
                count_exact += int(np.sum(labels_all[a:b] == (s - 1)))
            occ_cond = float(count_exact / max(denom_condition, 1))
            occ_global = float(count_exact / max(total_rows, 1))
            denom_rows.append(
                {
                    "cohort": cohort,
                    "pooled_state": int(s),
                    "state_rank": int(state_ref_df.loc[state_ref_df["pooled_state"] == s, "state_rank"].iloc[0]),
                    "count_timepoints": int(count_exact),
                    "denom_condition": int(denom_condition),
                    "denom_global": int(total_rows),
                    "occupancy_condition_specific": occ_cond,
                    "occupancy_global_denominator": occ_global,
                }
            )

    denom_df = pd.DataFrame(denom_rows).sort_values(["cohort", "state_rank"])

    cohort_tmean: dict[str, np.ndarray] = {}
    cohort_tmean_ns: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        d = tr_df[tr_df["cohort"] == cohort]
        if d.empty:
            continue
        piv = (
            d.pivot_table(index="from_state_rank", columns="to_state_rank", values="probability", aggfunc="mean")
            .reindex(index=np.arange(1, args.n_states + 1), columns=np.arange(1, args.n_states + 1), fill_value=0.0)
        )
        cohort_tmean[cohort] = piv.to_numpy(dtype=float)

        d2 = tr_ns_df[tr_ns_df["cohort"] == cohort]
        piv2 = (
            d2.pivot_table(index="from_state_rank", columns="to_state_rank", values="probability", aggfunc="mean")
            .reindex(index=np.arange(1, args.n_states + 1), columns=np.arange(1, args.n_states + 1), fill_value=0.0)
        )
        cohort_tmean_ns[cohort] = piv2.to_numpy(dtype=float)

    # Save secondary outputs.
    meta_df.to_csv(sec_tables / "pooled_subject_pattern_spans.csv", index=False)
    subject_df.to_csv(sec_tables / "pooled_subject_summary.csv", index=False)
    state_df.to_csv(sec_tables / "pooled_subject_state_metrics_long.csv", index=False)
    rank_df.to_csv(sec_tables / "pooled_rank_aligned_state_metrics_long.csv", index=False)
    sfc_method_summary.to_csv(sec_tables / "pooled_sfc_occupancy_method_comparison.csv", index=False)
    tr_df.to_csv(sec_tables / "pooled_rank_transition_matrices_long.csv", index=False)
    tr_ns_df.to_csv(sec_tables / "pooled_rank_transition_no_self_long.csv", index=False)
    state_ref_df.to_csv(sec_tables / "pooled_state_reference_coupling.csv", index=False)
    denom_df.to_csv(sec_tables / "pooled_condition_occupancy_denominator_comparison.csv", index=False)

    np.savez_compressed(
        sec_npz / "pooled_states.npz",
        centers=np.asarray(centers, dtype=float),
        center_matrices=np.asarray(center_mats, dtype=float),
        labels_all=np.asarray(labels_all, dtype=int),
        order_state_by_ref=np.asarray(order, dtype=int),
        rank_of_state=np.asarray(rank_of_state, dtype=int),
        sfc_ref=np.asarray(sfc_ref, dtype=float),
        sfc_ref_full=np.asarray(sfc_ref_full, dtype=float),
        pooled_sc_reference=np.asarray([args.pooled_sc_reference]),
    )

    _plot_occupancy_by_rank(rank_df, args.n_states, sec_figs)
    _plot_mean_occupancy_by_rank(rank_df, args.n_states, sec_figs)
    _plot_sfc_vs_occupancy(rank_df, sec_figs, x_col="sfc", stem="subject_level_sfc_vs_occupancy")
    _plot_sfc_vs_occupancy(
        rank_df,
        sec_figs,
        x_col="sfc_full",
        stem="subject_level_sfc_vs_occupancy_full_flatten",
        title="Subject-level Occupancy vs SC-coupling (full flatten)",
        x_label="State SC-coupling (Pearson r, full flatten)",
    )
    _plot_sfc_method_comparison(rank_df, sec_figs)
    _plot_transition_heatmaps(
        cohort_tmean,
        args.n_states,
        sec_figs,
        stem="pooled_cohort_mean_transition_rank_aligned",
        title="Secondary pooled-state transitions (rank-aligned by pooled SC reference)",
    )
    _plot_transition_heatmaps(
        cohort_tmean_ns,
        args.n_states,
        sec_figs,
        stem="pooled_cohort_mean_transition_rank_aligned_no_self",
        title="Secondary pooled-state transitions (rank-aligned, no self-transitions)",
    )

    pooled_meta = {
        "n_subjects": int(subject_df.shape[0]),
        "n_total_pattern_rows": int(labels_all.size),
        "n_states": int(args.n_states),
        "pooled_sc_reference": args.pooled_sc_reference,
        "random_seed_pooled": int(args.random_seed_pooled),
    }
    (sec_logs / "pooled_run_metadata.json").write_text(json.dumps(pooled_meta, indent=2))

    return pooled_meta


def _run_posthoc_template_alignment(
    subject_results: list[dict[str, Any]],
    args: argparse.Namespace,
    out_root: Path,
) -> None:
    out_dir = out_root / "posthoc_template_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.template_source == "control_only":
        pool = [np.asarray(r["centers"], dtype=float) for r in subject_results if str(r["cohort"]) == "control"]
        if not pool:
            raise RuntimeError("template_source='control_only' but no control subjects available.")
    else:
        pool = [np.asarray(r["centers"], dtype=float) for r in subject_results]

    templates = fit_state_templates(
        np.vstack(pool),
        n_states=args.n_states,
        random_seed=args.random_seed_template,
        n_init=args.n_init_template,
        max_iter=args.max_iter_template,
    )

    rows = []
    for res in subject_results:
        aligned = align_states_to_templates(
            np.asarray(res["centers"], dtype=float),
            np.asarray(res["occupancy_local"], dtype=float),
            np.asarray(templates, dtype=float),
        )
        assignment = np.asarray(aligned.assignment_local_to_template, dtype=int)
        occ = np.asarray(aligned.occupancy_aligned, dtype=float)
        sim = np.asarray(aligned.matched_similarity, dtype=float)

        for ti in range(args.n_states):
            rows.append(
                {
                    "subject_id": res["subject_id"],
                    "cohort": res["cohort"],
                    "stage": res["stage"],
                    "sedation": res["sedation"],
                    "template_state": int(ti) + 1,
                    "occupancy": float(occ[ti]),
                    "matched_similarity_mean": float(np.mean(sim)),
                    "matched_similarity_min": float(np.min(sim)),
                }
            )

    posthoc_df = pd.DataFrame(rows)
    posthoc_df.to_csv(out_dir / "posthoc_template_occupancy_long.csv", index=False)
    np.savez_compressed(
        out_dir / "posthoc_templates.npz",
        templates=np.asarray(templates, dtype=float),
        n_states=np.asarray([args.n_states], dtype=int),
        source=np.asarray([args.template_source]),
    )


def _save_reference_dataset_summary(reference_root: Path, out_tables: Path) -> None:
    out_rows: list[dict[str, Any]] = []

    idx_json = reference_root / "converted" / "index.json"
    if idx_json.exists():
        data = json.loads(idx_json.read_text())
        cohorts = data.get("cohorts", {})
        for key, val in cohorts.items():
            out_rows.append(
                {
                    "dataset": "brain_act_reference",
                    "cohort": str(key),
                    "n_subjects": int(val.get("n_subjects", 0)),
                    "source_cohort": str(val.get("source_cohort", "")),
                }
            )

    if out_rows:
        pd.DataFrame(out_rows).sort_values(["cohort"]).to_csv(out_tables / "reference_dataset_counts.csv", index=False)


def run_analysis(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    reference_root = Path(args.reference_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()

    fig_root = out_root / "figures"
    tab_root = out_root / "tables"
    npz_root = out_root / "npz"
    log_root = out_root / "logs"
    for p in (fig_root, tab_root, npz_root, log_root):
        p.mkdir(parents=True, exist_ok=True)

    print(f"Loading new dataset from: {data_root}")
    records, load_qc_df = load_new_doc_subjects(data_root, max_subjects_per_group=args.max_subjects_per_group)
    if not records:
        raise RuntimeError("No subject records loaded.")

    load_qc_df.to_csv(tab_root / "subject_loading_qc.csv", index=False)
    subject_alignment_summary = validate_subject_alignment_qc(load_qc_df)
    (log_root / "subject_alignment_summary.json").write_text(json.dumps(subject_alignment_summary, indent=2))

    roi_ref = build_roi_order_reference(data_root)
    roi_map_df = pd.DataFrame(
        {
            "symmetric_position_1based": np.arange(1, 91, dtype=int),
            "interleaved_index_0based": np.asarray(roi_ref.interleaved_to_symmetric_idx, dtype=int),
            "interleaved_position_1based": np.asarray(roi_ref.interleaved_to_symmetric_idx, dtype=int) + 1,
            "interleaved_roi": [roi_ref.interleaved_names[i] for i in roi_ref.interleaved_to_symmetric_idx],
            "symmetric_roi": list(roi_ref.symmetric_names),
        }
    )
    roi_map_df.to_csv(tab_root / "roi_reorder_mapping_interleaved_to_symmetric.csv", index=False)
    pd.DataFrame(
        {"position_1based": np.arange(1, 91, dtype=int), "roi_name": list(roi_ref.interleaved_names)}
    ).to_csv(tab_root / "roi_order_interleaved_aal90.csv", index=False)
    pd.DataFrame(
        {"position_1based": np.arange(1, 91, dtype=int), "roi_name": list(roi_ref.symmetric_names)}
    ).to_csv(tab_root / "roi_order_symmetric_deco.csv", index=False)
    print(
        "ROI reference loaded:",
        {
            "interleaved_source": roi_ref.interleaved_source,
            "symmetric_source": roi_ref.symmetric_source,
            "n_regions": len(roi_ref.interleaved_names),
        },
    )

    counts_by_cohort = pd.Series([r.cohort for r in records]).value_counts().to_dict()
    counts_by_stage = pd.DataFrame([{"cohort": r.cohort, "stage": r.stage} for r in records]).value_counts().to_dict()
    counts_by_sed = pd.DataFrame([{"cohort": r.cohort, "sedation": r.sedation} for r in records]).value_counts().to_dict()
    print("Loaded subjects by cohort:", counts_by_cohort)
    print("Loaded subjects by (cohort, stage):", counts_by_stage)
    print("Loaded subjects by (cohort, sedation):", counts_by_sed)

    records_use, reorder_qc_df, reorder_decision = _maybe_apply_roi_reordering(records, mode=args.roi_reorder_mode)
    reorder_qc_df.to_csv(tab_root / "roi_order_qc_coupling_checks.csv", index=False)
    applied_mode = str(reorder_decision["applied_mode"])
    final_order_summary = validate_final_roi_order_or_raise(roi_ref, applied_mode=applied_mode)
    fc_final_names, sc_final_names = resolve_roi_order_names(roi_ref, mode=applied_mode)
    pd.DataFrame(
        {"position_1based": np.arange(1, 91, dtype=int), "roi_name": fc_final_names}
    ).to_csv(tab_root / "roi_order_final_fc.csv", index=False)
    pd.DataFrame(
        {"position_1based": np.arange(1, 91, dtype=int), "roi_name": sc_final_names}
    ).to_csv(tab_root / "roi_order_final_sc.csv", index=False)

    reorder_decision["functional_input_order"] = "aal90_interleaved_lr"
    reorder_decision["structural_input_order"] = "aal90_symmetric_left_then_right_reverse"
    reorder_decision["functional_input_source"] = roi_ref.interleaved_source
    reorder_decision["structural_input_source"] = roi_ref.symmetric_source
    reorder_decision["final_alignment_summary"] = final_order_summary
    (log_root / "roi_reorder_decision.json").write_text(json.dumps(reorder_decision, indent=2))
    print("ROI reorder decision:", reorder_decision)

    _save_reference_dataset_summary(reference_root, tab_root)

    _plot_reorder_qc(reorder_qc_df, reorder_decision, fig_root)

    excluded_rows: list[dict[str, Any]] = []
    records_valid: list[SubjectRecord] = []
    for rec in records_use:
        bad = ~np.isfinite(np.asarray(rec.timeseries, dtype=float))
        n_bad = int(np.sum(bad))
        if n_bad > 0:
            excluded_rows.append(
                {
                    "subject_id": rec.subject_id,
                    "cohort": rec.cohort,
                    "stage": rec.stage,
                    "sedation": rec.sedation,
                    "source_fc_file": rec.source_fc_file,
                    "source_sc_file": rec.source_sc_file,
                    "n_nonfinite_values": n_bad,
                    "n_bad_timepoints": int(np.sum(np.any(bad, axis=1))),
                    "n_bad_rois": int(np.sum(np.any(bad, axis=0))),
                    "reason": "excluded_nonfinite_bold",
                }
            )
        else:
            records_valid.append(rec)

    excluded_df = pd.DataFrame(
        excluded_rows,
        columns=[
            "subject_id",
            "cohort",
            "stage",
            "sedation",
            "source_fc_file",
            "source_sc_file",
            "n_nonfinite_values",
            "n_bad_timepoints",
            "n_bad_rois",
            "reason",
        ],
    )
    excluded_df.to_csv(tab_root / "excluded_subjects_nonfinite_bold.csv", index=False)
    print(f"Excluded subjects with non-finite BOLD: {int(excluded_df.shape[0])}")
    print(f"Subjects retained for analysis: {len(records_valid)}")

    subject_results: list[dict[str, Any]] = []
    total = len(records_valid)
    for i, rec in enumerate(records_valid, start=1):
        seed = int(args.random_seed_subject + i)
        out = _subject_local_analysis(rec, args, seed=seed)
        if out:
            subject_results.append(out)
        if args.progress_every > 0 and (i % args.progress_every == 0 or i == total):
            print(f"[{i:03d}/{total:03d}] processed {rec.subject_id} ({rec.cohort})")

    if not subject_results:
        raise RuntimeError("No valid subject outputs produced by local state analysis.")

    subject_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    transition_no_self_rows: list[dict[str, Any]] = []

    centers_stack = []
    occ_stack = []
    sfc_stack = []
    rank_sort_stack = []
    labels_local_stack = []
    labels_rank_stack = []
    subject_ids_stack = []

    for res in subject_results:
        subject_ids_stack.append(res["subject_id"])
        centers_stack.append(np.asarray(res["centers"], dtype=float))
        occ_stack.append(np.asarray(res["occupancy_local"], dtype=float))
        sfc_stack.append(np.asarray(res["sfc_local"], dtype=float))
        rank_sort_stack.append(np.asarray(res["rank_sort_local_indices"], dtype=int))
        labels_local_stack.append(np.asarray(res["labels_local"], dtype=int))
        labels_rank_stack.append(np.asarray(res["labels_rank"], dtype=int))

        subject_row = {
            "subject_id": res["subject_id"],
            "cohort": res["cohort"],
            "stage": res["stage"],
            "sedation": res["sedation"],
            "source_fc_file": res["source_fc_file"],
            "source_sc_file": res["source_sc_file"],
            "source_subject_index": int(res["source_subject_index"]),
            "source_subject_label": res["source_subject_label"],
            "n_timepoints": int(res["n_timepoints"]),
            "n_regions": int(res["n_regions"]),
            "global_sync_mean": float(res["global_sync_mean"]),
            "global_sync_std": float(res["global_sync_std"]),
            "static_fc_sc_coupling": float(res["static_fc_sc_coupling"]),
            "static_fc_sc_coupling_full": float(res["static_fc_sc_coupling_full"]),
            "entropy_rate_bits": float(res["entropy_rate_bits"]),
            "entropy_rate_norm": float(res["entropy_rate_norm"]),
        }

        occ = np.asarray(res["occupancy_local"], dtype=float)
        sfc = np.asarray(res["sfc_local"], dtype=float)
        sfc_full = np.asarray(res["sfc_local_full"], dtype=float)
        order = np.asarray(res["rank_sort_local_indices"], dtype=int)

        for li in range(args.n_states):
            subject_row[f"local_state_{li + 1}_occupancy"] = float(occ[li])
            subject_row[f"local_state_{li + 1}_sfc"] = float(sfc[li])
            subject_row[f"local_state_{li + 1}_sfc_full"] = float(sfc_full[li])
            local_rows.append(
                {
                    "subject_id": res["subject_id"],
                    "cohort": res["cohort"],
                    "stage": res["stage"],
                    "sedation": res["sedation"],
                    "local_state": int(li) + 1,
                    "occupancy": float(occ[li]),
                    "sfc": float(sfc[li]),
                    "sfc_full": float(sfc_full[li]),
                }
            )

        for rank0, local_idx in enumerate(order):
            rank_rows.append(
                {
                    "subject_id": res["subject_id"],
                    "cohort": res["cohort"],
                    "stage": res["stage"],
                    "sedation": res["sedation"],
                    "state_rank": int(rank0) + 1,
                    "local_state": int(local_idx) + 1,
                    "occupancy": float(occ[local_idx]),
                    "sfc": float(sfc[local_idx]),
                    "sfc_full": float(sfc_full[local_idx]),
                }
            )

        tm = np.asarray(res["transition_rank"], dtype=float)
        tm_ns = np.asarray(res["transition_rank_no_self"], dtype=float)
        for a in range(args.n_states):
            for b in range(args.n_states):
                transition_rows.append(
                    {
                        "subject_id": res["subject_id"],
                        "cohort": res["cohort"],
                        "from_state_rank": int(a) + 1,
                        "to_state_rank": int(b) + 1,
                        "probability": float(tm[a, b]),
                    }
                )
                transition_no_self_rows.append(
                    {
                        "subject_id": res["subject_id"],
                        "cohort": res["cohort"],
                        "from_state_rank": int(a) + 1,
                        "to_state_rank": int(b) + 1,
                        "probability": float(tm_ns[a, b]),
                    }
                )

        subject_rows.append(subject_row)

    subject_df = pd.DataFrame(subject_rows).sort_values(["cohort", "subject_id"]).reset_index(drop=True)
    local_df = pd.DataFrame(local_rows)
    rank_df = pd.DataFrame(rank_rows)
    transition_df = pd.DataFrame(transition_rows)
    transition_no_self_df = pd.DataFrame(transition_no_self_rows)

    # Group summaries
    cohort_rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        ds = subject_df[subject_df["cohort"] == cohort]
        if ds.empty:
            continue
        cohort_rows.append(
            {
                "cohort": cohort,
                "n_subjects": int(ds.shape[0]),
                "static_fc_sc_mean": float(ds["static_fc_sc_coupling"].mean()),
                "static_fc_sc_std": float(ds["static_fc_sc_coupling"].std(ddof=1) if ds.shape[0] > 1 else 0.0),
                "static_fc_sc_mean_full": float(ds["static_fc_sc_coupling_full"].mean()),
                "static_fc_sc_std_full": float(ds["static_fc_sc_coupling_full"].std(ddof=1) if ds.shape[0] > 1 else 0.0),
                "global_sync_mean": float(ds["global_sync_mean"].mean()),
                "global_sync_std": float(ds["global_sync_mean"].std(ddof=1) if ds.shape[0] > 1 else 0.0),
                "entropy_rate_norm_mean": float(ds["entropy_rate_norm"].mean()),
                "entropy_rate_norm_std": float(ds["entropy_rate_norm"].std(ddof=1) if ds.shape[0] > 1 else 0.0),
            }
        )

    rank_summary = (
        rank_df.groupby(["cohort", "state_rank"], as_index=False)
        .agg(
            n=("occupancy", "size"),
            occupancy_mean=("occupancy", "mean"),
            occupancy_std=("occupancy", "std"),
            sfc_mean=("sfc", "mean"),
            sfc_std=("sfc", "std"),
            sfc_full_mean=("sfc_full", "mean"),
            sfc_full_std=("sfc_full", "std"),
        )
        .sort_values(["cohort", "state_rank"])
    )
    rank_summary["occupancy_sem"] = rank_summary["occupancy_std"] / np.sqrt(np.maximum(rank_summary["n"], 1))
    sfc_method_summary = _summarize_sfc_occupancy_methods(rank_df)

    stats_omnibus_df, stats_pairwise_df = _compute_rank_stats(rank_df)

    # Mean transitions per cohort
    cohort_tmean: dict[str, np.ndarray] = {}
    cohort_tmean_ns: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        dt = transition_df[transition_df["cohort"] == cohort]
        if dt.empty:
            continue
        pivot = (
            dt.pivot_table(index="from_state_rank", columns="to_state_rank", values="probability", aggfunc="mean")
            .reindex(index=np.arange(1, args.n_states + 1), columns=np.arange(1, args.n_states + 1), fill_value=0.0)
        )
        cohort_tmean[cohort] = pivot.to_numpy(dtype=float)

        dt_ns = transition_no_self_df[transition_no_self_df["cohort"] == cohort]
        pivot_ns = (
            dt_ns.pivot_table(index="from_state_rank", columns="to_state_rank", values="probability", aggfunc="mean")
            .reindex(index=np.arange(1, args.n_states + 1), columns=np.arange(1, args.n_states + 1), fill_value=0.0)
        )
        cohort_tmean_ns[cohort] = pivot_ns.to_numpy(dtype=float)

    # Save tables
    subject_df.to_csv(tab_root / "subject_level_summary.csv", index=False)
    local_df.to_csv(tab_root / "local_state_metrics_long.csv", index=False)
    rank_df.to_csv(tab_root / "rank_aligned_state_metrics_long.csv", index=False)
    transition_df.to_csv(tab_root / "rank_aligned_transition_matrices_long.csv", index=False)
    transition_no_self_df.to_csv(tab_root / "rank_aligned_transition_no_self_long.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(tab_root / "cohort_summary.csv", index=False)
    rank_summary.to_csv(tab_root / "cohort_rank_occupancy_summary.csv", index=False)
    sfc_method_summary.to_csv(tab_root / "sfc_occupancy_method_comparison.csv", index=False)
    stats_omnibus_df.to_csv(tab_root / "stats_occupancy_rank_omnibus.csv", index=False)
    stats_pairwise_df.to_csv(tab_root / "stats_occupancy_rank_pairwise.csv", index=False)

    # Save arrays
    np.savez_compressed(
        npz_root / "subject_local_state_arrays.npz",
        subject_ids=np.asarray(subject_ids_stack, dtype=object),
        centers=np.asarray(centers_stack, dtype=float),
        occupancy=np.asarray(occ_stack, dtype=float),
        sfc=np.asarray(sfc_stack, dtype=float),
        rank_sort_local_indices=np.asarray(rank_sort_stack, dtype=int),
        labels_local=np.asarray(labels_local_stack, dtype=int),
        labels_rank=np.asarray(labels_rank_stack, dtype=int),
    )

    np.savez_compressed(
        npz_root / "cohort_rank_transition_means.npz",
        cohorts=np.asarray([c for c in COHORTS if c in cohort_tmean], dtype=object),
        transition_rank=np.asarray([cohort_tmean[c] for c in COHORTS if c in cohort_tmean], dtype=float),
        transition_rank_no_self=np.asarray([cohort_tmean_ns[c] for c in COHORTS if c in cohort_tmean_ns], dtype=float),
    )

    # Figures
    _plot_occupancy_by_rank(rank_df, args.n_states, fig_root)
    _plot_mean_occupancy_by_rank(rank_df, args.n_states, fig_root)
    _plot_sfc_vs_occupancy(rank_df, fig_root, x_col="sfc", stem="subject_level_sfc_vs_occupancy")
    _plot_sfc_vs_occupancy(
        rank_df,
        fig_root,
        x_col="sfc_full",
        stem="subject_level_sfc_vs_occupancy_full_flatten",
        title="Subject-level Occupancy vs SC-coupling (full flatten)",
        x_label="State SC-coupling (Pearson r, full flatten)",
    )
    _plot_sfc_method_comparison(rank_df, fig_root)
    _plot_static_fc_sc(subject_df, fig_root)
    _plot_transition_heatmaps(
        cohort_tmean,
        args.n_states,
        fig_root,
        stem="cohort_mean_transition_rank_aligned",
        title="Cohort mean transition matrices (rank-aligned local states)",
    )
    _plot_transition_heatmaps(
        cohort_tmean_ns,
        args.n_states,
        fig_root,
        stem="cohort_mean_transition_rank_aligned_no_self",
        title="Cohort mean transition matrices (rank-aligned, no self-transitions)",
    )
    _plot_markov_entropy(subject_df, fig_root)
    _plot_mean_state_matrices_by_rank(subject_results, args.n_states, fig_root)

    pooled_meta: dict[str, Any] = {"enabled": False}
    if not args.skip_secondary_pooled:
        pooled_meta = _run_secondary_pooled_state_analysis(records_valid, args, out_root)
        pooled_meta["enabled"] = True

    if args.run_posthoc_template:
        _run_posthoc_template_alignment(subject_results, args, out_root)

    # Save run metadata
    run_meta = {
        "data_root": str(data_root),
        "reference_root": str(reference_root),
        "output_root": str(out_root),
        "n_subjects": int(subject_df.shape[0]),
        "n_states": int(args.n_states),
        "pipeline": args.pipeline,
        "clustering_backend": args.clustering_backend,
        "trim_edge_samples": int(args.trim_edge_samples),
        "n_init_subject": int(args.n_init_subject),
        "max_iter_subject": int(args.max_iter_subject),
        "roi_reorder_requested": args.roi_reorder_mode,
        "roi_reorder_applied": reorder_decision["applied_mode"],
        "posthoc_template_enabled": bool(args.run_posthoc_template),
        "template_source": args.template_source,
        "secondary_pooled_enabled": bool(pooled_meta.get("enabled", False)),
        "pooled_sc_reference": args.pooled_sc_reference,
    }
    (log_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    print(f"Saved audited outputs to: {out_root}")
    print(f"Subjects analyzed: {subject_df.shape[0]}")
    print("Cohort counts:", subject_df["cohort"].value_counts().to_dict())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/doc_patients_new_data",
        help="Path to new DoC dataset root.",
    )
    parser.add_argument(
        "--reference-root",
        type=str,
        default="data/brain_act",
        help="Path to original/reference dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited",
        help="Directory where outputs will be saved.",
    )

    parser.add_argument("--n-states", type=int, default=5)
    parser.add_argument("--strict-k", action="store_true", help="Fail if effective k differs from requested n-states.")

    parser.add_argument(
        "--pipeline",
        type=str,
        default="brain_act_legacy",
        choices=["standard", "brain_act_legacy"],
        help="Brain-state preprocessing pipeline.",
    )
    parser.add_argument(
        "--clustering-backend",
        type=str,
        default="sklearn",
        choices=["scipy", "sklearn"],
        help="Clustering backend for local state extraction.",
    )

    parser.add_argument("--trim-edge-samples", type=int, default=9)
    parser.add_argument("--tr-seconds", type=float, default=2.4)
    parser.add_argument("--bandpass-low-hz", type=float, default=0.01)
    parser.add_argument("--bandpass-high-hz", type=float, default=0.20)
    parser.add_argument("--filter-order", type=int, default=3)

    parser.add_argument("--n-init-subject", type=int, default=20)
    parser.add_argument("--max-iter-subject", type=int, default=200)
    parser.add_argument("--random-seed-subject", type=int, default=0)

    parser.add_argument(
        "--roi-reorder-mode",
        type=str,
        default="aal90_fc",
        choices=["auto", "none", "aal90_fc", "aal90_sc", "aal90_both"],
        help=(
            "ROI reordering mode. For the new DoC raw files, use 'aal90_fc' "
            "(functional is interleaved AAL90, SC is already symmetric/deco order). "
            "'auto' is available for diagnostics but final FC/SC order equality is "
            "always validated and will fail loudly on mismatch."
        ),
    )

    parser.add_argument(
        "--max-subjects-per-group",
        type=int,
        default=None,
        help="Optional debug cap per cohort.",
    )
    parser.add_argument("--progress-every", type=int, default=1)

    parser.add_argument(
        "--skip-secondary-pooled",
        action="store_true",
        help="Skip secondary pooled-timepoint state analysis (enabled by default).",
    )
    parser.add_argument(
        "--pooled-sc-reference",
        type=str,
        default="control",
        choices=["control", "all"],
        help="Reference SC used to order pooled states by SC-coupling.",
    )
    parser.add_argument("--n-init-pooled", type=int, default=32, help="KMeans restarts for secondary pooled analysis.")
    parser.add_argument("--max-iter-pooled", type=int, default=240, help="KMeans max iterations for pooled analysis.")
    parser.add_argument("--random-seed-pooled", type=int, default=11, help="Random seed for pooled analysis.")

    parser.add_argument("--run-posthoc-template", action="store_true")
    parser.add_argument(
        "--template-source",
        type=str,
        default="control_only",
        choices=["control_only", "all_cohorts"],
    )
    parser.add_argument("--n-init-template", type=int, default=48)
    parser.add_argument("--max-iter-template", type=int, default=240)
    parser.add_argument("--random-seed-template", type=int, default=7)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
