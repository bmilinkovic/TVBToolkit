"""Brain-Act AAL90 structural dataset conversion and loading utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

RAW_TO_CANONICAL_COHORT = {
    "CNT": "control",
    "COMA": "coma",
    "MCS": "mcs",
    "UWS": "uws",
}
CANONICAL_TO_RAW_COHORT = {v: k for k, v in RAW_TO_CANONICAL_COHORT.items()}
SUBJECT_RE = re.compile(r"(sub-\d+)")


@dataclass(frozen=True)
class AAL90Atlas:
    """Atlas metadata for AAL90-parcellated datasets."""

    labels: np.ndarray
    region_codes: np.ndarray | None = None
    region_indices: np.ndarray | None = None
    ordering: str = "interleaved_lr"
    source: str | None = None
    centres: np.ndarray | None = None
    name: str = "AAL90"

    @property
    def n_regions(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class TractLengthSanity:
    """Summary of tract-length unit plausibility checks."""

    min_length_mm: float
    max_length_mm: float
    percentile_99_mm: float
    plausible_range_mm: tuple[float, float]
    is_plausible: bool
    warning: str | None = None


@dataclass(frozen=True)
class StructuralMetadata:
    """Metadata returned with subject-level structural matrices."""

    subject_id: str
    cohort: str
    source_cohort: str
    dataset_index: int
    connectivity_shape: tuple[int, int]
    tract_lengths_shape: tuple[int, int]
    stage: str | None = None
    sedation: str | None = None
    validation_report: dict[str, Any] | None = None


def _canonicalize_cohort(cohort: str) -> str:
    key = str(cohort).strip()
    if key in RAW_TO_CANONICAL_COHORT:
        return RAW_TO_CANONICAL_COHORT[key]
    if key.upper() in {"EMCS", "COMA"}:
        return key.lower()
    if key.upper() == "EMCS":
        return "emcs"
    key_lower = key.lower()
    if key_lower in {"emcs", "coma"}:
        return key_lower
    if key_lower in CANONICAL_TO_RAW_COHORT:
        return key_lower
    raise ValueError(
        f"Unknown cohort '{cohort}'. Expected one of "
        f"{sorted(list(RAW_TO_CANONICAL_COHORT) + list(CANONICAL_TO_RAW_COHORT) + ['EMCS', 'emcs', 'COMA', 'coma'])}."
    )


def _resolve_data_root(source_root: str | Path) -> Path:
    root = Path(source_root).expanduser().resolve()
    candidates = [
        root,
        root / "data",
        root / "brain-act" / "data",
    ]
    for cand in candidates:
        if (cand / "organized").exists() and (cand / "atlases").exists():
            return cand
    raise FileNotFoundError(
        f"Could not find Brain-Act data root under {root}. Expected folders: "
        "'organized/' and 'atlases/'."
    )


def _sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    h = sha256()
    contiguous = np.ascontiguousarray(arr)
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    h.update(contiguous.tobytes(order="C"))
    return h.hexdigest()


def _parse_lookup_table(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    labels: list[str] = []
    codes: list[str] = []
    indices: list[int] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx == 0:
            continue
        code = parts[1]
        name = parts[2]
        indices.append(idx)
        codes.append(code)
        labels.append(name)

    if not indices:
        raise ValueError(f"No atlas entries found in lookup table: {path}")

    order = np.argsort(np.asarray(indices))
    sorted_indices = np.asarray(indices, dtype=np.int32)[order]
    sorted_codes = np.asarray(codes, dtype="U64")[order]
    sorted_labels = np.asarray(labels, dtype="U128")[order]
    return sorted_labels, sorted_codes, sorted_indices


def _find_subject_files(sc_dir: Path, tl_dir: Path) -> list[str]:
    sc_subjects = {
        m.group(1)
        for p in sc_dir.glob("sub-*_structural_connectome.mat")
        if (m := SUBJECT_RE.search(p.name))
    }
    tl_subjects = {
        m.group(1)
        for p in tl_dir.glob("sub-*_tract_lengths.txt")
        if (m := SUBJECT_RE.search(p.name))
    }
    return sorted(sc_subjects & tl_subjects)


def _load_subject_sc(path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(str(path))
    if "structural_connectome" in mat:
        arr = np.asarray(mat["structural_connectome"], dtype=float)
        if arr.ndim == 2:
            return arr
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            return arr.astype(float)
    raise ValueError(f"No 2D numeric structural connectome array found in {path}")


def normalize_connectivity(connectivity: np.ndarray, mode: str | None = "max") -> np.ndarray:
    """Normalize connectivity matrix values with a simple deterministic scheme."""
    c = np.asarray(connectivity, dtype=float).copy()
    if mode is None:
        return c
    mode = mode.lower()
    if mode == "max":
        scale = float(np.max(np.abs(c)))
        return c if scale <= 0 else c / scale
    if mode == "sum":
        scale = float(np.sum(c))
        return c if abs(scale) <= 1e-12 else c / scale
    if mode == "zscore":
        mu = float(np.mean(c))
        sigma = float(np.std(c))
        return c - mu if sigma <= 1e-12 else (c - mu) / sigma
    raise ValueError("normalize mode must be one of: None, 'max', 'sum', 'zscore'.")


def threshold_connectivity(
    connectivity: np.ndarray,
    threshold: float | None = None,
    percentile: float | None = None,
) -> np.ndarray:
    """Threshold small connectivity weights by absolute threshold or percentile."""
    c = np.asarray(connectivity, dtype=float).copy()
    if threshold is None and percentile is None:
        return c
    if threshold is not None and percentile is not None:
        raise ValueError("Set either threshold or percentile, not both.")
    if percentile is not None:
        if not (0 <= percentile <= 100):
            raise ValueError("percentile must be in [0, 100].")
        nonzero = c[c > 0]
        if nonzero.size == 0:
            return c
        threshold = float(np.percentile(nonzero, percentile))
    c[c < float(threshold)] = 0.0
    return c


def _handle_nonfinite(arr: np.ndarray, policy: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    mask = ~np.isfinite(out)
    if not np.any(mask):
        return out
    if policy == "raise":
        raise ValueError("Encountered NaN or Inf values in matrix.")
    if policy == "zero":
        out[mask] = 0.0
        return out
    if policy == "mean":
        finite = out[np.isfinite(out)]
        fill = 0.0 if finite.size == 0 else float(np.mean(finite))
        out[mask] = fill
        return out
    raise ValueError("nonfinite policy must be one of: 'raise', 'zero', 'mean'.")


def tract_length_unit_sanity(
    tract_lengths: np.ndarray,
    plausible_range_mm: tuple[float, float] = (1.0, 400.0),
) -> TractLengthSanity:
    """Check whether non-zero tract lengths look plausible in millimetres."""
    l = np.asarray(tract_lengths, dtype=float)
    values = l[l > 0]
    if values.size == 0:
        return TractLengthSanity(
            min_length_mm=0.0,
            max_length_mm=0.0,
            percentile_99_mm=0.0,
            plausible_range_mm=plausible_range_mm,
            is_plausible=False,
            warning="All tract lengths are zero; units cannot be assessed.",
        )
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    p99 = float(np.percentile(values, 99))
    low, high = plausible_range_mm
    is_plausible = (vmin >= 0.0) and (p99 <= high) and (vmax <= (high * 2.0))
    warning = None
    if not is_plausible:
        warning = (
            f"Tract lengths appear outside expected mm range {plausible_range_mm}: "
            f"min={vmin:.3f}, p99={p99:.3f}, max={vmax:.3f}."
        )
    return TractLengthSanity(
        min_length_mm=vmin,
        max_length_mm=vmax,
        percentile_99_mm=p99,
        plausible_range_mm=plausible_range_mm,
        is_plausible=is_plausible,
        warning=warning,
    )


def validate_structural_matrices(
    connectivity: np.ndarray,
    tract_lengths: np.ndarray,
    *,
    symmetry_tol: float = 1e-8,
    enforce_symmetry: bool = True,
    zero_diagonal: bool = True,
    nonfinite: str = "raise",
    normalize: str | None = None,
    threshold: float | None = None,
    percentile: float | None = None,
    tract_length_plausible_mm: tuple[float, float] = (1.0, 400.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate and optionally clean structural matrices for simulation."""
    c = _handle_nonfinite(connectivity, policy=nonfinite)
    l = _handle_nonfinite(tract_lengths, policy=nonfinite)

    if c.ndim != 2 or l.ndim != 2:
        raise ValueError("connectivity and tract_lengths must both be 2D matrices.")
    if c.shape[0] != c.shape[1] or l.shape[0] != l.shape[1]:
        raise ValueError("connectivity and tract_lengths must be square matrices.")
    if c.shape != l.shape:
        raise ValueError("connectivity and tract_lengths must have matching shapes.")

    asym_c = float(np.max(np.abs(c - c.T)))
    asym_l = float(np.max(np.abs(l - l.T)))
    if enforce_symmetry:
        if asym_c > symmetry_tol:
            c = 0.5 * (c + c.T)
        if asym_l > symmetry_tol:
            l = 0.5 * (l + l.T)

    if zero_diagonal:
        np.fill_diagonal(c, 0.0)
        np.fill_diagonal(l, 0.0)

    c = threshold_connectivity(c, threshold=threshold, percentile=percentile)
    c = normalize_connectivity(c, mode=normalize)
    tl_sanity = tract_length_unit_sanity(l, plausible_range_mm=tract_length_plausible_mm)

    report = {
        "asymmetry_connectivity_max_abs": asym_c,
        "asymmetry_tract_lengths_max_abs": asym_l,
        "symmetry_tol": symmetry_tol,
        "enforce_symmetry": enforce_symmetry,
        "zero_diagonal": zero_diagonal,
        "nonfinite_policy": nonfinite,
        "normalize": normalize,
        "threshold": threshold,
        "percentile": percentile,
        "tract_length_sanity": {
            "min_length_mm": tl_sanity.min_length_mm,
            "max_length_mm": tl_sanity.max_length_mm,
            "percentile_99_mm": tl_sanity.percentile_99_mm,
            "plausible_range_mm": list(tl_sanity.plausible_range_mm),
            "is_plausible": tl_sanity.is_plausible,
            "warning": tl_sanity.warning,
        },
    }
    return c, l, report


def convert_brain_act_dataset(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    atlas_lookup_name: str = "custom_lookuptable_AAL.txt",
    dtype: str = "float32",
    overwrite: bool = False,
) -> Path:
    """Convert Brain-Act organized structural data into fast cohort NPZ bundles."""
    data_root = _resolve_data_root(source_root)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    if index_path.exists() and not overwrite:
        raise FileExistsError(f"{index_path} already exists. Use overwrite=True to rebuild.")

    atlas_path = data_root / "atlases" / atlas_lookup_name
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas lookup table not found: {atlas_path}")
    labels, codes, indices = _parse_lookup_table(atlas_path)
    n_regions = int(labels.shape[0])

    np.savez_compressed(
        output_dir / "atlas.npz",
        labels=labels,
        region_codes=codes,
        region_indices=indices,
    )

    structured_subjects: list[dict[str, Any]] = []
    cohorts_index: dict[str, Any] = {}

    organized = data_root / "organized"
    for raw_cohort, canonical_cohort in RAW_TO_CANONICAL_COHORT.items():
        sc_dir = organized / "structural_connectomes" / raw_cohort
        tl_dir = organized / "tract_lengths" / raw_cohort
        subject_ids = _find_subject_files(sc_dir, tl_dir)
        if not subject_ids:
            raise FileNotFoundError(
                f"No matched structural_connectome/tract_length pairs found for cohort {raw_cohort}"
            )

        cohort_c: list[np.ndarray] = []
        cohort_l: list[np.ndarray] = []
        cohort_sc_checksum: list[str] = []
        cohort_tl_checksum: list[str] = []

        for local_idx, subject_id in enumerate(subject_ids):
            sc_path = sc_dir / f"{subject_id}_structural_connectome.mat"
            tl_path = tl_dir / f"{subject_id}_tract_lengths.txt"
            if not sc_path.exists() or not tl_path.exists():
                continue
            c = _load_subject_sc(sc_path)
            l = np.loadtxt(tl_path, dtype=float)
            if c.shape != (n_regions, n_regions):
                raise ValueError(
                    f"{sc_path} has shape {c.shape}; expected {(n_regions, n_regions)} "
                    f"from atlas {atlas_lookup_name}."
                )
            if l.shape != (n_regions, n_regions):
                raise ValueError(
                    f"{tl_path} has shape {l.shape}; expected {(n_regions, n_regions)} "
                    f"from atlas {atlas_lookup_name}."
                )

            c = np.asarray(c, dtype=dtype)
            l = np.asarray(l, dtype=dtype)
            cohort_c.append(c)
            cohort_l.append(l)
            sc_hash = _sha256_array(c)
            tl_hash = _sha256_array(l)
            cohort_sc_checksum.append(sc_hash)
            cohort_tl_checksum.append(tl_hash)
            structured_subjects.append(
                {
                    "subject_id": subject_id,
                    "cohort": canonical_cohort,
                    "source_cohort": raw_cohort,
                    "dataset_index": local_idx,
                    "connectivity_shape": list(c.shape),
                    "tract_lengths_shape": list(l.shape),
                    "connectivity_sha256": sc_hash,
                    "tract_lengths_sha256": tl_hash,
                }
            )

        out_npz = output_dir / f"subjects_{canonical_cohort}.npz"
        np.savez_compressed(
            out_npz,
            subject_ids=np.asarray(subject_ids, dtype="U16"),
            connectivity=np.stack(cohort_c, axis=0),
            tract_lengths=np.stack(cohort_l, axis=0),
            sc_checksums=np.asarray(cohort_sc_checksum, dtype="U128"),
            tl_checksums=np.asarray(cohort_tl_checksum, dtype="U128"),
        )
        cohorts_index[canonical_cohort] = {
            "source_cohort": raw_cohort,
            "n_subjects": len(subject_ids),
            "subjects_file": out_npz.name,
            "subjects_file_sha256": _sha256_file(out_npz),
            "subject_ids": subject_ids,
            "matrix_shape": [n_regions, n_regions],
        }

    index = {
        "format": "tvbtoolkit.brain_act.structural_npz",
        "format_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_root": str(data_root),
        "atlas": {
            "name": "AAL90",
            "lookup_file": atlas_lookup_name,
            "lookup_file_sha256": _sha256_file(atlas_path),
            "n_regions": n_regions,
            "ordering": "interleaved_lr",
            "labels_sha256": _sha256_array(labels),
            "codes_sha256": _sha256_array(codes),
        },
        "cohorts": cohorts_index,
        "subjects": structured_subjects,
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index_path


def _default_dataset_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "brain_act" / "converted"


def _resolve_dataset_root(dataset_root: str | Path | None) -> Path:
    if dataset_root is None:
        root = _default_dataset_root()
    else:
        root = Path(dataset_root).expanduser().resolve()
    return root


def _load_index(dataset_root: str | Path | None) -> dict[str, Any]:
    root = _resolve_dataset_root(dataset_root)
    index_path = root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_aal90_atlas(
    dataset_root: str | Path | None = None,
    *,
    lookup_name: str | None = None,
) -> AAL90Atlas:
    """Load atlas labels and ordering for Brain-Act AAL90 data.

    Parameters
    ----------
    dataset_root : str | Path | None
        Converted dataset root (folder containing ``index.json``).
    lookup_name : str | None, optional
        Optional raw-data lookup filename override used only when ``atlas.npz``
        is missing. When ``None``, the value declared in ``index.json`` is used.

    Notes
    -----
    The converted dataset stores matrices and atlas labels in a declared ordering
    under ``index.json`` -> ``atlas.ordering``. This function propagates that
    ordering instead of assuming interleaved AAL ordering.
    """
    root = _resolve_dataset_root(dataset_root)
    index = _load_index(root)
    atlas_meta = dict(index.get("atlas", {}))
    ordering = str(atlas_meta.get("ordering", "interleaved_lr"))

    atlas_npz = root / "atlas.npz"
    if atlas_npz.exists():
        with np.load(atlas_npz, allow_pickle=False) as data:
            labels = np.asarray(data["labels"], dtype="U128")
            codes = np.asarray(data["region_codes"], dtype="U64")
            indices = np.asarray(data["region_indices"], dtype=np.int32)
        return AAL90Atlas(
            labels=labels,
            region_codes=codes,
            region_indices=indices,
            ordering=ordering,
            source=str(atlas_npz),
        )

    data_root = _resolve_data_root(root)
    lookup_eff = lookup_name or str(atlas_meta.get("lookup_file", "custom_lookuptable_AAL.txt"))
    atlas_path = data_root / lookup_eff
    labels, codes, indices = _parse_lookup_table(atlas_path)
    return AAL90Atlas(
        labels=labels,
        region_codes=codes,
        region_indices=indices,
        ordering=ordering,
        source=str(atlas_path),
    )


def list_subjects(
    dataset_root: str | Path | None = None,
    cohort: str | None = None,
) -> dict[str, list[str]] | list[str]:
    """List subject IDs in the converted dataset, optionally filtered by cohort."""
    index = _load_index(dataset_root)
    cohorts = index["cohorts"]
    if cohort is None:
        return {name: list(meta["subject_ids"]) for name, meta in cohorts.items()}
    canonical = _canonicalize_cohort(cohort)
    if canonical not in cohorts:
        raise KeyError(f"Cohort '{canonical}' not found in dataset.")
    return list(cohorts[canonical]["subject_ids"])


def load_subject_structural(
    subject_id: str,
    dataset_root: str | Path | None = None,
    *,
    cohort: str | None = None,
    validate: bool = True,
    symmetry_tol: float = 1e-8,
    enforce_symmetry: bool = True,
    zero_diagonal: bool = True,
    nonfinite: str = "raise",
    normalize: str | None = None,
    threshold: float | None = None,
    percentile: float | None = None,
    tract_length_plausible_mm: tuple[float, float] = (1.0, 400.0),
) -> tuple[np.ndarray, np.ndarray, AAL90Atlas, StructuralMetadata]:
    """Load subject-level connectivity and tract lengths from converted Brain-Act data."""
    root = _resolve_dataset_root(dataset_root)
    index = _load_index(root)
    atlas = load_aal90_atlas(root)
    canonical_cohort = _canonicalize_cohort(cohort) if cohort is not None else None

    matches = [s for s in index["subjects"] if s["subject_id"] == subject_id]
    if canonical_cohort is not None:
        matches = [s for s in matches if s["cohort"] == canonical_cohort]
    if not matches:
        raise KeyError(f"Subject '{subject_id}' not found for cohort={cohort!r}.")
    if len(matches) > 1:
        raise ValueError(
            f"Subject '{subject_id}' exists in multiple cohorts; pass cohort explicitly."
        )

    entry = matches[0]
    canonical_cohort = entry["cohort"]
    cohort_meta = index["cohorts"][canonical_cohort]
    subjects_npz = root / cohort_meta["subjects_file"]
    with np.load(subjects_npz, allow_pickle=False) as data:
        ids = np.asarray(data["subject_ids"], dtype="U16")
        idx_candidates = np.where(ids == subject_id)[0]
        if idx_candidates.size == 0:
            raise KeyError(f"Subject '{subject_id}' not found in {subjects_npz}.")
        idx = int(idx_candidates[0])
        c = np.asarray(data["connectivity"][idx], dtype=float)
        l = np.asarray(data["tract_lengths"][idx], dtype=float)

    if c.shape[0] != atlas.n_regions or c.shape[1] != atlas.n_regions:
        raise ValueError(
            f"Connectivity shape {c.shape} does not match atlas size {atlas.n_regions}."
        )

    validation_report = None
    if validate:
        c, l, validation_report = validate_structural_matrices(
            c,
            l,
            symmetry_tol=symmetry_tol,
            enforce_symmetry=enforce_symmetry,
            zero_diagonal=zero_diagonal,
            nonfinite=nonfinite,
            normalize=normalize,
            threshold=threshold,
            percentile=percentile,
            tract_length_plausible_mm=tract_length_plausible_mm,
        )

    metadata = StructuralMetadata(
        subject_id=subject_id,
        cohort=canonical_cohort,
        source_cohort=cohort_meta["source_cohort"],
        dataset_index=idx,
        stage=str(entry.get("stage", "")) if entry.get("stage", None) is not None else None,
        sedation=str(entry.get("sedation", "")) if entry.get("sedation", None) is not None else None,
        connectivity_shape=(int(c.shape[0]), int(c.shape[1])),
        tract_lengths_shape=(int(l.shape[0]), int(l.shape[1])),
        validation_report=validation_report,
    )
    return c, l, atlas, metadata
