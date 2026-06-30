"""PhiID helpers for empirical BOLD workflows.

This module supports the audited workflow traced from:

- ``/Users/borjan/code/python/TVBEmergence/test/matlab/emergence_measures.m``
- ``/Users/borjan/code/python/AnesthesiaProjectEmergence/phiid_plot.ipynb``

The intended use is:

1. Load subject-level empirical BOLD timeseries in Python.
2. Export each subject to a MATLAB ``.mat`` input file with shape
   ``(regions, time)``.
3. Run the MATLAB batch runner that calls ``PhiIDFull`` for every ROI pair.
4. Re-load per-atom output matrices into Python and average them by cohort.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import scipy.io

PHIID_ATOMS: tuple[str, ...] = (
    "sts",
    "rtr",
    "rtx",
    "rty",
    "rts",
    "xtr",
    "xtx",
    "xty",
    "xts",
    "ytr",
    "ytx",
    "yty",
    "yts",
    "str",
    "stx",
    "sty",
)
PRIMARY_ATOMS: tuple[str, ...] = ("sts", "rtr")
PUBLICATION_COHORT_ORDER: tuple[str, ...] = ("control", "emcs", "mcs", "uws", "coma")

LEGACY_REFERENCE_PATHS: dict[str, str] = {
    "legacy_generation_matlab": "/Users/borjan/code/python/TVBEmergence/test/matlab/emergence_measures.m",
    "legacy_plot_notebook": "/Users/borjan/code/python/AnesthesiaProjectEmergence/phiid_plot.ipynb",
    "legacy_results_dir": "/Users/borjan/code/python/TVBEmergence/results/phiid/Idep_xtb",
    "legacy_figures_dir": "/Users/borjan/code/python/AnesthesiaProjectEmergence/results/phiid/used-figures",
}

_NEW_STYLE_RE = re.compile(
    r"^(?P<subject>.+)__(?P<atom>sr_gradient|[a-z]{3})(?:_mat)?_(?P<redundancy>[A-Za-z0-9_]+)$"
)
_LEGACY_STYLE_RE = re.compile(
    r"^(?P<subject>[^_]+)_(?P<atom>sr_gradient|[a-z]{3})(?:_mat)?_(?P<redundancy>[A-Za-z0-9_]+)$"
)


def sanitize_subject_stub(subject_id: str) -> str:
    """Return a filesystem-safe subject stub."""
    text = str(subject_id).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _coerce_timeseries_time_by_region(timeseries: np.ndarray) -> np.ndarray:
    """Return subject timeseries as ``(time, regions)``."""
    x = np.asarray(timeseries, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {x.shape}.")
    if x.shape[1] == 90:
        return x
    if x.shape[0] == 90:
        return x.T
    if x.shape[0] > x.shape[1]:
        return x
    if x.shape[1] > x.shape[0]:
        return x.T
    raise ValueError(f"Could not infer time/region axes from square shape {x.shape}.")


def _apply_standardization(timeseries: np.ndarray, standardize: str | None) -> np.ndarray:
    """Optionally standardize each ROI over time."""
    x = np.asarray(timeseries, dtype=float).copy()
    if standardize is None or standardize == "none":
        return x
    if standardize == "demean":
        return x - np.mean(x, axis=0, keepdims=True)
    if standardize == "zscore":
        mu = np.mean(x, axis=0, keepdims=True)
        sigma = np.std(x, axis=0, keepdims=True)
        sigma[sigma <= 0.0] = 1.0
        return (x - mu) / sigma
    raise ValueError("standardize must be one of: None, 'none', 'demean', 'zscore'.")


def _drop_nonfinite_timepoints(timeseries: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove timepoints containing any non-finite ROI value."""
    x = np.asarray(timeseries, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {x.shape}.")
    keep = np.all(np.isfinite(x), axis=1)
    return x[keep, :], int((~keep).sum())


def subject_records_to_frame(records: Sequence[Any]) -> pd.DataFrame:
    """Convert subject-like records into a manifest DataFrame.

    Expected attributes per record:
    ``subject_id``, ``cohort``, ``stage``, ``sedation``, ``timeseries``,
    ``source_fc_file``, ``source_sc_file``, ``source_subject_index``,
    ``source_subject_label``.
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        ts = _coerce_timeseries_time_by_region(np.asarray(rec.timeseries, dtype=float))
        rows.append(
            {
                "subject_id": str(rec.subject_id),
                "subject_stub": sanitize_subject_stub(str(rec.subject_id)),
                "cohort": str(getattr(rec, "cohort", "")),
                "stage": str(getattr(rec, "stage", "")),
                "sedation": str(getattr(rec, "sedation", "")),
                "n_timepoints": int(ts.shape[0]),
                "n_regions": int(ts.shape[1]),
                "source_fc_file": str(getattr(rec, "source_fc_file", "")),
                "source_sc_file": str(getattr(rec, "source_sc_file", "")),
                "source_subject_index": int(getattr(rec, "source_subject_index", -1)),
                "source_subject_label": str(getattr(rec, "source_subject_label", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["cohort", "stage", "sedation", "subject_id"]).reset_index(drop=True)


def export_phiid_subject_inputs(
    records: Sequence[Any],
    output_dir: str | Path,
    *,
    manifest_name: str = "manifest.csv",
    roi_labels: Sequence[str] | None = None,
    max_timepoints: int | None = None,
    standardize: str | None = None,
    tr_seconds: float | None = 2.4,
) -> pd.DataFrame:
    """Export subject BOLD inputs for MATLAB PhiID.

    Saved ``.mat`` files use ``time_series`` with shape ``(regions, time)`` to
    match the legacy MATLAB loop, which indexes ``time_series(row, :)``.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels = list(roi_labels) if roi_labels is not None else []

    manifest = subject_records_to_frame(records)
    dropped_counts: dict[str, int] = {}
    final_lengths: dict[str, int] = {}
    for rec in records:
        ts_tr = _coerce_timeseries_time_by_region(np.asarray(rec.timeseries, dtype=float))
        if max_timepoints is not None:
            ts_tr = ts_tr[: int(max_timepoints), :]
        ts_tr, n_dropped = _drop_nonfinite_timepoints(ts_tr)
        dropped_counts[str(rec.subject_id)] = int(n_dropped)
        ts_tr = _apply_standardization(ts_tr, standardize=standardize)
        final_lengths[str(rec.subject_id)] = int(ts_tr.shape[0])
        if ts_tr.shape[0] <= 1:
            raise ValueError(
                f"Subject {rec.subject_id} has insufficient finite timepoints after cleaning: {ts_tr.shape[0]}."
            )
        if not np.all(np.isfinite(ts_tr)):
            raise ValueError(f"Non-finite values detected for subject {rec.subject_id}.")

        stub = sanitize_subject_stub(str(rec.subject_id))
        save_path = out / f"{stub}.mat"
        payload: dict[str, Any] = {
            "time_series": np.asarray(ts_tr.T, dtype=np.float64),
            "subject_id": np.asarray([str(rec.subject_id)], dtype=object),
            "subject_stub": np.asarray([stub], dtype=object),
            "cohort": np.asarray([str(getattr(rec, 'cohort', ''))], dtype=object),
            "stage": np.asarray([str(getattr(rec, 'stage', ''))], dtype=object),
            "sedation": np.asarray([str(getattr(rec, 'sedation', ''))], dtype=object),
            "source_fc_file": np.asarray([str(getattr(rec, 'source_fc_file', ''))], dtype=object),
            "source_sc_file": np.asarray([str(getattr(rec, 'source_sc_file', ''))], dtype=object),
            "source_subject_index": np.asarray([[int(getattr(rec, "source_subject_index", -1))]], dtype=np.int32),
            "source_subject_label": np.asarray([str(getattr(rec, 'source_subject_label', ''))], dtype=object),
            "n_regions": np.asarray([[int(ts_tr.shape[1])]], dtype=np.int32),
            "n_timepoints": np.asarray([[int(ts_tr.shape[0])]], dtype=np.int32),
        }
        if tr_seconds is not None:
            payload["tr_seconds"] = np.asarray([[float(tr_seconds)]], dtype=np.float64)
        if labels:
            payload["roi_labels"] = np.asarray(labels, dtype=object).reshape(1, -1)
        scipy.io.savemat(save_path, payload, do_compression=True)

    if not manifest.empty:
        manifest["dropped_nonfinite_timepoints"] = manifest["subject_id"].map(dropped_counts).fillna(0).astype(int)
        manifest["n_timepoints"] = manifest["subject_id"].map(final_lengths).fillna(manifest["n_timepoints"]).astype(int)

    manifest_path = out / manifest_name
    manifest.to_csv(manifest_path, index=False)
    return manifest


def build_matlab_batch_command(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    redundancy: str = "idep_xtb",
    matlab_bin: str = "/Applications/MATLAB_R2023b.app/bin/matlab",
    matlab_toolbox_root: str | Path = "/Users/borjan/code/matlab/elph",
    runner_path: str | Path | None = None,
    use_parallel: bool = True,
    n_workers: int = 0,
) -> str:
    """Build a one-line MATLAB batch command for the empirical PhiID runner."""
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    runner = Path(runner_path).expanduser().resolve() if runner_path is not None else None
    runner_dir = runner.parent if runner is not None else None
    toolbox_root = Path(matlab_toolbox_root).expanduser().resolve()

    statements = [f"addpath(genpath('{toolbox_root.as_posix()}'))"]
    if runner_dir is not None:
        statements.append(f"addpath('{runner_dir.as_posix()}')")
    statements.append(
        "phiid_empirical_bold_aal90("
        f"'{in_dir.as_posix()}', "
        f"'{out_dir.as_posix()}', "
        f"'{redundancy}', "
        f"{str(bool(use_parallel)).lower()}, "
        f"{int(n_workers)})"
    )
    batch = "; ".join(statements)
    return f'{matlab_bin} -batch "{batch}"'


def parse_phiid_output_name(path: str | Path) -> dict[str, str] | None:
    """Parse new or legacy PhiID output names."""
    stem = Path(path).stem
    match = _NEW_STYLE_RE.match(stem)
    if match is None:
        match = _LEGACY_STYLE_RE.match(stem)
    if match is None:
        return None
    meta = match.groupdict()
    meta["subject_stub"] = sanitize_subject_stub(meta["subject"])
    return meta


def load_phiid_matrix(path: str | Path, *, atom: str | None = None) -> np.ndarray:
    """Load a PhiID atom matrix or the synergy-redundancy gradient from MATLAB."""
    mat = scipy.io.loadmat(str(path))
    variables = {k: v for k, v in mat.items() if not k.startswith("__")}
    atom_name = atom
    if atom_name is None:
        parsed = parse_phiid_output_name(path)
        atom_name = parsed["atom"] if parsed is not None else None

    candidates: list[str] = []
    if atom_name == "sr_gradient":
        candidates.extend(["gradient", "sr_gradient"])
    elif atom_name:
        candidates.append(f"{atom_name}_mat")

    for name in candidates:
        if name in variables:
            return np.asarray(variables[name], dtype=float)

    for value in variables.values():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return np.asarray(arr, dtype=float)
    raise KeyError(f"Could not find numeric PhiID matrix in {path}.")


def load_phiid_index(
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Index per-atom PhiID output files and merge subject metadata when available."""
    out = Path(output_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(out.glob("*.mat")):
        parsed = parse_phiid_output_name(path)
        if parsed is None:
            continue
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "subject_stub": parsed["subject_stub"],
                "subject_from_filename": parsed["subject"],
                "atom": parsed["atom"],
                "redundancy": parsed["redundancy"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    manifest: pd.DataFrame | None = None
    if manifest_path is not None:
        manifest = pd.read_csv(Path(manifest_path).expanduser().resolve())
    else:
        candidate = out.parent / "inputs" / "manifest.csv"
        if candidate.exists():
            manifest = pd.read_csv(candidate)

    if manifest is not None and not manifest.empty and "subject_stub" in manifest.columns:
        df = df.merge(manifest, how="left", on="subject_stub")
    return df.sort_values(["atom", "cohort", "stage", "sedation", "subject_stub"]).reset_index(drop=True)


def average_atom_matrices_by_group(
    index_df: pd.DataFrame,
    *,
    atom: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Average a selected atom matrix across subject groups."""
    if index_df.empty:
        raise ValueError("index_df is empty; no PhiID outputs were indexed.")

    subset = index_df.loc[index_df["atom"] == atom].copy()
    if subset.empty:
        raise ValueError(f"No indexed outputs found for atom '{atom}'.")

    missing = [col for col in group_cols if col not in subset.columns]
    if missing:
        raise KeyError(f"group columns missing from index: {missing}")

    rows: list[dict[str, Any]] = []
    for group_key, group in subset.groupby(list(group_cols), dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        matrices = [load_phiid_matrix(path, atom=atom) for path in group["path"]]
        shapes = {tuple(np.asarray(m).shape) for m in matrices}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent matrix shapes for atom '{atom}' group {group_key}: {sorted(shapes)}")
        stacked = np.stack(matrices, axis=0)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "atom": atom,
                "n_subjects": int(stacked.shape[0]),
                "matrix_shape": tuple(int(x) for x in stacked.shape[1:]),
                "matrix": np.mean(stacked, axis=0),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_group_average_outputs(
    averages_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Persist group-averaged matrices as both ``.npy`` and MATLAB ``.mat``."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    saved_rows: list[dict[str, Any]] = []

    for row in averages_df.to_dict(orient="records"):
        matrix = np.asarray(row["matrix"], dtype=float)
        atom = str(row["atom"])
        group_parts = []
        for key, value in row.items():
            if key in {"matrix", "matrix_shape", "n_subjects", "atom"}:
                continue
            group_parts.append(f"{key}-{sanitize_subject_stub(value)}")
        stem = "__".join(group_parts + [f"{atom}_avg"])
        npy_path = out / f"{stem}.npy"
        mat_path = out / f"{stem}.mat"
        np.save(npy_path, matrix)
        scipy.io.savemat(mat_path, {f"{atom}_avg": matrix}, do_compression=True)
        saved_rows.append(
            {
                "stem": stem,
                "atom": atom,
                "n_subjects": int(row["n_subjects"]),
                "npy_path": str(npy_path),
                "mat_path": str(mat_path),
            }
        )
    return pd.DataFrame(saved_rows)


def default_atom_cmap(atom: str) -> str:
    """Return the legacy-style colormap for a PhiID atom."""
    if atom == "sts":
        return "YlOrRd"
    if atom == "rtr":
        return "bone_r"
    return "viridis"


def publication_atom_cmap(atom: str) -> mcolors.Colormap:
    """Return a publication-oriented colormap inspired by Wes Anderson palettes."""
    if atom == "rtr":
        colors = ["#F7F1E3", "#D9D4C7", "#B7C2C8", "#7B9AA6", "#3E5C67"]
    elif atom == "sts":
        colors = ["#FBF5E9", "#EBCB8B", "#D88C4A", "#B65D4A", "#7C2F39"]
    else:
        colors = ["#FBF5E9", "#D9D4C7", "#7B9AA6", "#3E5C67"]
    cmap = mcolors.LinearSegmentedColormap.from_list(f"phiid_pub_{atom}", colors)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def format_roi_label_compact(label: str) -> str:
    """Return a compact, human-readable AAL90 ROI label."""
    text = str(label).strip()
    if not text:
        return text

    side = ""
    if text.endswith("_L"):
        text = text[:-2]
        side = "L"
    elif text.endswith("_R"):
        text = text[:-2]
        side = "R"

    token_map = {
        "Frontal": "Front",
        "Parietal": "Par",
        "Temporal": "Temp",
        "Occipital": "Occ",
        "Cingulum": "Cing",
        "Hippocampus": "Hipp",
        "ParaHippocampal": "ParaHip",
        "Supp": "Supp",
        "Motor": "Mot",
        "Olfactory": "Olf",
        "Calcarine": "Calc",
        "Lingual": "Ling",
        "Fusiform": "Fusi",
        "Amygdala": "Amyg",
        "Thalamus": "Thal",
        "Caudate": "Caud",
        "Putamen": "Put",
        "Pallidum": "Pall",
        "Precuneus": "Precun",
        "Precentral": "Precent",
        "Postcentral": "Postcent",
        "Rolandic": "Rol",
        "Oper": "Oper",
        "Orb": "Orb",
        "Medial": "Med",
        "Rectus": "Rect",
        "Insula": "Ins",
        "Cerebelum": "Cb",
        "Crus": "Crus",
        "Vermis": "Verm",
    }
    tokens = [token_map.get(tok, tok) for tok in text.split("_") if tok]
    compact = " ".join(tokens)
    return f"{compact} {side}".strip()


def compact_roi_labels(labels: Sequence[str]) -> list[str]:
    """Return compact plotting labels for a sequence of ROI names."""
    return [format_roi_label_compact(label) for label in labels]


def format_roi_label_coarse(label: str) -> str:
    """Return a coarse canonical anatomical label for an AAL90 ROI."""
    text = str(label).strip()
    if not text:
        return text

    side = ""
    if text.endswith("_L"):
        text = text[:-2]
        side = "L"
    elif text.endswith("_R"):
        text = text[:-2]
        side = "R"

    if any(tok in text for tok in ("Caudate", "Putamen", "Pallidum", "Thalamus")):
        group = "Subcortical"
    elif any(tok in text for tok in ("Hippocampus", "ParaHippocampal", "Amygdala", "Cingulum")):
        group = "Limbic"
    elif any(tok in text for tok in ("Calcarine", "Cuneus", "Lingual", "Occipital", "Fusiform")):
        group = "Occipital"
    elif any(tok in text for tok in ("Temporal", "Heschl")):
        group = "Temporal"
    elif any(tok in text for tok in ("Postcentral", "Paracentral", "Precentral", "Rolandic", "Supp_Motor_Area")):
        group = "Sensorimotor"
    elif any(tok in text for tok in ("Parietal", "SupraMarginal", "Angular", "Precuneus")):
        group = "Parietal"
    elif "Insula" in text:
        group = "Insular"
    elif any(tok in text for tok in ("Frontal", "Rectus", "Olfactory")):
        group = "Frontal"
    else:
        group = format_roi_label_compact(label)
        return group

    return f"{group} {side}".strip()


def coarse_roi_labels(labels: Sequence[str]) -> list[str]:
    """Return coarse canonical anatomical labels for ROI names."""
    return [format_roi_label_coarse(label) for label in labels]


def plot_phiid_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    atom: str,
    roi_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
    cmap: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one PhiID matrix using the audited legacy color conventions."""
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=(cmap or default_atom_cmap(atom)), origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    if roi_labels is not None and len(roi_labels) == arr.shape[0]:
        tick_idx = np.arange(0, arr.shape[0], 10, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_yticks(tick_idx)
        ax.set_xticklabels([roi_labels[i] for i in tick_idx], rotation=90, fontsize=7)
        ax.set_yticklabels([roi_labels[i] for i in tick_idx], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_group_average_grid(
    averages_df: pd.DataFrame,
    *,
    atom: str,
    title_cols: Sequence[str],
    roi_labels: Sequence[str] | None = None,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.0, 4.2),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a grid of group-averaged matrices for one selected atom."""
    subset = averages_df.loc[averages_df["atom"] == atom].copy()
    if subset.empty:
        raise ValueError(f"No average matrices found for atom '{atom}'.")

    n = int(subset.shape[0])
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    cmap = default_atom_cmap(atom)
    for ax, row in zip(axes.reshape(-1), subset.to_dict(orient="records"), strict=False):
        mat = np.asarray(row["matrix"], dtype=float)
        label = " | ".join(f"{col}={row[col]}" for col in title_cols)
        im = ax.imshow(mat, cmap=cmap, origin="lower", aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
        if roi_labels is not None and len(roi_labels) == mat.shape[0]:
            tick_idx = np.arange(0, mat.shape[0], 10, dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_yticks(tick_idx)
            ax.set_xticklabels([roi_labels[i] for i in tick_idx], rotation=90, fontsize=6)
            ax.set_yticklabels([roi_labels[i] for i in tick_idx], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.reshape(-1)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_publication_cohort_grid(
    averages_df: pd.DataFrame,
    *,
    cohort_order: Sequence[str] = PUBLICATION_COHORT_ORDER,
    roi_labels: Sequence[str] | None = None,
    atoms_in_rows: Sequence[str] = ("rtr", "sts"),
    figsize: tuple[float, float] = (16.0, 6.6),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a 2x5 cohort grid for ``rtr`` and ``sts`` with publication styling."""
    subset = averages_df.copy()
    needed = {"cohort", "atom", "matrix"}
    missing = needed.difference(subset.columns)
    if missing:
        raise KeyError(f"averages_df is missing required columns: {sorted(missing)}")

    cohort_order = [str(x) for x in cohort_order]
    atoms_in_rows = [str(x) for x in atoms_in_rows]
    fig, axes = plt.subplots(
        len(atoms_in_rows),
        len(cohort_order),
        figsize=figsize,
        squeeze=False,
        facecolor="none",
    )

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
        }
    ):
        for row_idx, atom in enumerate(atoms_in_rows):
            row_df = subset.loc[subset["atom"] == atom].copy()
            matrices: list[np.ndarray] = []
            for cohort in cohort_order:
                hit = row_df.loc[row_df["cohort"] == cohort]
                if hit.empty:
                    raise ValueError(f"Missing cohort '{cohort}' for atom '{atom}'.")
                matrices.append(np.asarray(hit.iloc[0]["matrix"], dtype=float))

            vmax = max(float(np.nanmax(m)) for m in matrices)
            vmin = min(float(np.nanmin(m)) for m in matrices)
            if atom in {"rtr", "sts"}:
                vmin = max(0.0, vmin)

            im = None
            for col_idx, (cohort, matrix) in enumerate(zip(cohort_order, matrices, strict=True)):
                ax = axes[row_idx, col_idx]
                mat = np.asarray(matrix, dtype=float).copy()
                if mat.shape[0] == mat.shape[1]:
                    np.fill_diagonal(mat, np.nan)
                im = ax.imshow(
                    mat,
                    cmap=publication_atom_cmap(atom),
                    origin="lower",
                    aspect="equal",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_title(cohort.upper(), pad=6)
                if col_idx == 0:
                    ax.set_ylabel("ROI")
                    row_name = "Redundancy (RTR)" if atom == "rtr" else "Synergy (STS)"
                    ax.text(
                        -0.24,
                        0.5,
                        row_name,
                        rotation=90,
                        va="center",
                        ha="center",
                        transform=ax.transAxes,
                        fontsize=9.5,
                    )
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("ROI" if row_idx == len(atoms_in_rows) - 1 else "")
                tick_idx = np.arange(0, mat.shape[0], 15, dtype=int)
                ax.set_xticks(tick_idx)
                ax.set_yticks(tick_idx)
                if roi_labels is not None and len(roi_labels) == mat.shape[0]:
                    ax.set_xticklabels([roi_labels[i] for i in tick_idx], rotation=90)
                    ax.set_yticklabels([roi_labels[i] for i in tick_idx])
                else:
                    ax.set_xticklabels([str(i + 1) for i in tick_idx], rotation=90)
                    ax.set_yticklabels([str(i + 1) for i in tick_idx])
                ax.tick_params(length=1.5, width=0.5, color="#666666")
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if im is not None:
                cbar = fig.colorbar(
                    im,
                    ax=axes[row_idx, :],
                    fraction=0.015,
                    pad=0.012,
                    shrink=0.96,
                    location="right",
                )
                cbar.outline.set_linewidth(0.4)
                cbar.ax.tick_params(labelsize=7, width=0.4, length=2.0)
                cbar.set_label("PhiID value", fontsize=8)

        fig.suptitle(
            "Cohort-Averaged Persistent Redundancy and Synergy Across AAL90 Regions",
            y=0.995,
            fontsize=11,
        )
        fig.subplots_adjust(left=0.075, right=0.92, top=0.90, bottom=0.12, wspace=0.12, hspace=0.18)
    return fig, axes


def plot_publication_method_comparison_grid(
    mmi_averages: pd.DataFrame,
    ccs_averages: pd.DataFrame,
    *,
    cohort_order: Sequence[str] = PUBLICATION_COHORT_ORDER,
    roi_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (18.6, 13.8),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a publication-style MMI vs CCS comparison grid by cohort."""
    needed = {"cohort", "atom", "matrix"}
    for label, frame in (("mmi", mmi_averages), ("ccs", ccs_averages)):
        missing = needed.difference(frame.columns)
        if missing:
            raise KeyError(f"{label}_averages is missing required columns: {sorted(missing)}")

    row_specs = [
        ("mmi", "rtr", "Redundancy (RTR)"),
        ("mmi", "sts", "Synergy (STS)"),
        ("ccs", "rtr", "Redundancy (RTR)"),
        ("ccs", "sts", "Synergy (STS)"),
    ]
    frames = {"mmi": mmi_averages.copy(), "ccs": ccs_averages.copy()}
    cohort_order = [str(x) for x in cohort_order]

    matrix_lookup: dict[tuple[str, str, str], np.ndarray] = {}
    for method_name, frame in frames.items():
        for row in frame.to_dict(orient="records"):
            matrix_lookup[(method_name, str(row["atom"]), str(row["cohort"]))] = np.asarray(row["matrix"], dtype=float)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize, facecolor="none")
    gs = GridSpec(
        nrows=len(row_specs),
        ncols=len(cohort_order) + 1,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 0.11],
        wspace=0.14,
        hspace=0.16,
    )
    axes = np.empty((len(row_specs), len(cohort_order)), dtype=object)
    cbar_axes = []
    for row_idx in range(len(row_specs)):
        for col_idx in range(len(cohort_order)):
            axes[row_idx, col_idx] = fig.add_subplot(gs[row_idx, col_idx])
        cbar_axes.append(fig.add_subplot(gs[row_idx, len(cohort_order)]))

    label_ticks = None
    pretty_labels = coarse_roi_labels(roi_labels) if roi_labels is not None else None
    first_matrix = next(iter(matrix_lookup.values()))
    if pretty_labels is not None and first_matrix is not None:
        n_roi = int(np.asarray(first_matrix).shape[0])
        label_ticks = np.linspace(0, n_roi - 1, 4, dtype=int)

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
        }
    ):
        for row_idx, (method_name, atom, row_label) in enumerate(row_specs):
            row_df = frames[method_name].loc[frames[method_name]["atom"] == atom].copy()
            matrices: list[np.ndarray] = []
            for cohort in cohort_order:
                hit = row_df.loc[row_df["cohort"] == cohort]
                if hit.empty:
                    raise ValueError(f"Missing cohort '{cohort}' for atom '{atom}' in '{method_name}'.")
                matrices.append(np.asarray(hit.iloc[0]["matrix"], dtype=float))

            vmax = max(float(np.nanmax(m)) for m in matrices)
            vmin = min(float(np.nanmin(m)) for m in matrices)
            if atom in {"rtr", "sts"}:
                vmin = max(0.0, vmin)

            row_im = None
            for col_idx, cohort in enumerate(cohort_order):
                ax = axes[row_idx, col_idx]
                matrix = np.asarray(matrix_lookup[(method_name, atom, cohort)], dtype=float).copy()
                if matrix.shape[0] == matrix.shape[1]:
                    np.fill_diagonal(matrix, np.nan)

                if atom == "rtr":
                    im = ax.imshow(
                        matrix,
                        cmap=publication_atom_cmap("rtr"),
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                        aspect="equal",
                        interpolation="nearest",
                    )
                else:
                    im = ax.imshow(
                        matrix,
                        cmap=publication_atom_cmap("sts"),
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                        aspect="equal",
                        interpolation="nearest",
                    )
                row_im = im

                if row_idx == 0:
                    ax.set_title(cohort.upper(), pad=8)
                if col_idx == 0:
                    ax.text(
                        -0.18,
                        0.5,
                        row_label,
                        rotation=90,
                        va="center",
                        ha="center",
                        transform=ax.transAxes,
                        fontsize=10.0,
                        color="#1F2430",
                    )
                else:
                    pass

                show_bottom_labels = row_idx == len(row_specs) - 1
                if label_ticks is not None and pretty_labels is not None:
                    ax.set_xticks(label_ticks)
                    ax.set_yticks(label_ticks)
                    if col_idx == 0:
                        ax.set_yticklabels([pretty_labels[i] for i in label_ticks])
                    else:
                        ax.set_yticklabels([])
                    if show_bottom_labels:
                        ax.set_xticklabels([pretty_labels[i] for i in label_ticks], rotation=90)
                    else:
                        ax.set_xticklabels([])
                else:
                    tick_idx = np.arange(0, matrix.shape[0], 18, dtype=int)
                    ax.set_xticks(tick_idx)
                    ax.set_yticks(tick_idx)
                    if col_idx == 0:
                        ax.set_yticklabels([str(i + 1) for i in tick_idx])
                    else:
                        ax.set_yticklabels([])
                    if show_bottom_labels:
                        ax.set_xticklabels([str(i + 1) for i in tick_idx], rotation=90)
                    else:
                        ax.set_xticklabels([])
                ax.tick_params(length=1.8, width=0.55, color="#6E6A61")
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if row_im is not None:
                cax = cbar_axes[row_idx]
                cax.set_facecolor("none")
                cbar = fig.colorbar(row_im, cax=cax)
                cbar.outline.set_linewidth(0.45)
                cbar.ax.tick_params(labelsize=7.4, width=0.45, length=2.2)
                cbar.set_label("PhiID value", fontsize=8.5)

        fig.suptitle("Persistent PhiID Comparison Across Cohorts", x=0.5, y=0.985, ha="center", fontsize=14, color="#1B1B1B")
        fig.text(
            0.5,
            0.958,
            "Gaussian MMI and CCS decompositions on cohort-averaged AAL90 BOLD connectivity",
            ha="center",
            va="top",
            fontsize=9.2,
            color="#544F46",
        )
        fig.text(0.065, 0.765, "MMI", ha="left", va="center", fontsize=12.0, color="#1F2430")
        fig.text(0.065, 0.36, "CCS", ha="left", va="center", fontsize=12.0, color="#1F2430")
        fig.text(0.5, 0.062, "Coarse anatomical labels (AAL90 regions, FC-reordered)", ha="center", va="center", fontsize=8.4, color="#5A544B")
        fig.subplots_adjust(left=0.135, right=0.94, top=0.92, bottom=0.12)
    return fig, axes
