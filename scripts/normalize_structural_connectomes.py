#!/usr/bin/env python3
"""Create an auditable cohort-global-max-normalized structural dataset.

Every structural weight in every subject is divided by one shared scalar: the
largest off-diagonal weight in the complete input cohort. This maps weights to
[0, 1] while preserving zeros, symmetry, and all within- and between-subject
weight ratios. Tract lengths and subject ordering are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEME = "cohort_global_max"
FORMAT_VERSION = "1.1.0"


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _global_max(input_root: Path, index: dict[str, Any]) -> float:
    maximum = 0.0
    for cohort_meta in index["cohorts"].values():
        with np.load(input_root / cohort_meta["subjects_file"], allow_pickle=False) as data:
            weights = np.asarray(data["connectivity"], dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("Structural weights must be finite and non-negative.")
        maximum = max(maximum, float(np.max(weights)))
    if maximum <= 0.0:
        raise ValueError("Cannot normalize a dataset with no positive structural weights.")
    return maximum


def build_normalized_dataset(input_root: Path, output_root: Path) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if input_root == output_root:
        raise ValueError("Input and output roots must differ; normalization is non-destructive.")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    index_path = input_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    source_index = json.loads(index_path.read_text(encoding="utf-8"))
    divisor = _global_max(input_root, source_index)
    output_root.mkdir(parents=True, exist_ok=True)

    for name in ("atlas.npz", "source_subject_map.csv"):
        source = input_root / name
        if source.exists():
            shutil.copy2(source, output_root / name)

    audit_rows: list[dict[str, Any]] = []
    new_index = dict(source_index)
    new_index["format_version"] = FORMAT_VERSION
    new_index["created_utc"] = datetime.now(timezone.utc).isoformat()
    new_index["source_dataset_root"] = str(input_root)
    new_index["source_index_sha256"] = _sha256_file(index_path)
    new_index["connectivity_normalization"] = {
        "scheme": SCHEME,
        "scope": "all_subjects_all_cohorts",
        "divisor": divisor,
        "formula": "normalized_weight = source_weight / divisor",
        "range": [0.0, 1.0],
        "zero_preserving": True,
        "ratio_preserving": True,
        "simulator_normalization_required": "none",
    }

    new_cohorts: dict[str, Any] = {}
    new_subject_entries = {(
        str(row["cohort"]), str(row["subject_id"])
    ): dict(row) for row in source_index["subjects"]}

    for cohort, cohort_meta in source_index["cohorts"].items():
        source_file = input_root / cohort_meta["subjects_file"]
        with np.load(source_file, allow_pickle=False) as data:
            ids = np.asarray(data["subject_ids"])
            source_weights = np.asarray(data["connectivity"], dtype=np.float64)
            lengths = np.asarray(data["tract_lengths"])
            tl_checksums = np.asarray(data["tl_checksums"])
        weights = (source_weights / divisor).astype(np.float32)
        sc_checksums = np.asarray(
            [_sha256_array(subject) for subject in weights], dtype="U128"
        )
        output_file = output_root / cohort_meta["subjects_file"]
        np.savez_compressed(
            output_file,
            subject_ids=ids,
            connectivity=weights,
            tract_lengths=lengths,
            sc_checksums=sc_checksums,
            tl_checksums=tl_checksums,
        )
        updated_cohort = dict(cohort_meta)
        updated_cohort["subjects_file_sha256"] = _sha256_file(output_file)
        new_cohorts[cohort] = updated_cohort

        for subject_id, raw, normalized, checksum in zip(
            ids.astype(str), source_weights, weights, sc_checksums.astype(str)
        ):
            zero_raw = raw == 0.0
            zero_norm = normalized == 0.0
            if not np.array_equal(zero_raw, zero_norm):
                raise RuntimeError(f"Zero mask changed for {cohort}:{subject_id}.")
            if not np.allclose(normalized, normalized.T, rtol=1e-6, atol=1e-8):
                raise RuntimeError(f"Symmetry changed for {cohort}:{subject_id}.")
            entry = new_subject_entries[(cohort, subject_id)]
            entry["connectivity_sha256"] = checksum
            positive = raw > 0.0
            legacy = raw / (raw.sum(axis=0, keepdims=True) + 1e-12)
            legacy_delta = normalized.astype(np.float64) - legacy
            legacy_norm = float(np.linalg.norm(legacy))
            audit_rows.append(
                {
                    "cohort": cohort,
                    "subject_id": subject_id,
                    "n_zero_edges": int(np.count_nonzero(zero_raw)),
                    "zero_mask_identical": True,
                    "source_max": float(raw.max()),
                    "normalized_max": float(normalized.max()),
                    "source_total_strength": float(raw.sum()),
                    "normalized_total_strength": float(normalized.sum()),
                    "legacy_effective_total_strength": float(legacy.sum()),
                    "normalization_divisor": divisor,
                    "max_abs_ratio_error": float(
                        np.max(np.abs(normalized[positive] / raw[positive] - 1.0 / divisor))
                    ),
                    "legacy_column_sum_symmetry_error_max": float(
                        np.max(np.abs(legacy - legacy.T))
                    ),
                    "corrected_symmetry_error_max": float(
                        np.max(np.abs(normalized - normalized.T))
                    ),
                    "corrected_vs_legacy_pearson": float(
                        np.corrcoef(normalized.ravel(), legacy.ravel())[0, 1]
                    ),
                    "corrected_vs_legacy_relative_frobenius": float(
                        np.linalg.norm(legacy_delta) / legacy_norm
                    ),
                }
            )

    new_index["cohorts"] = new_cohorts
    new_index["subjects"] = list(new_subject_entries.values())
    (output_root / "index.json").write_text(
        json.dumps(new_index, indent=2) + "\n", encoding="utf-8"
    )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_root / "normalization_audit.csv", index=False)
    report = {
        "scheme": SCHEME,
        "n_subjects": int(len(audit)),
        "divisor": divisor,
        "normalized_global_max": float(audit["normalized_max"].max()),
        "all_zero_masks_identical": bool(audit["zero_mask_identical"].all()),
        "maximum_ratio_error": float(audit["max_abs_ratio_error"].max()),
        "maximum_corrected_symmetry_error": float(
            audit["corrected_symmetry_error_max"].max()
        ),
        "median_legacy_column_sum_symmetry_error": float(
            audit["legacy_column_sum_symmetry_error_max"].median()
        ),
        "corrected_total_strength_range": [
            float(audit["normalized_total_strength"].min()),
            float(audit["normalized_total_strength"].max()),
        ],
        "legacy_effective_total_strength_range": [
            float(audit["legacy_effective_total_strength"].min()),
            float(audit["legacy_effective_total_strength"].max()),
        ],
        "median_corrected_vs_legacy_pearson": float(
            audit["corrected_vs_legacy_pearson"].median()
        ),
        "median_corrected_vs_legacy_relative_frobenius": float(
            audit["corrected_vs_legacy_relative_frobenius"].median()
        ),
        "output_root": str(output_root),
    }
    (output_root / "normalization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_normalized_dataset(args.input_root, args.output_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
