#!/usr/bin/env python3
"""Convert new DoC structural data into tvbtoolkit converted dataset format.

Builds:
- atlas.npz
- subjects_<cohort>.npz
- index.json

from the external DOC raw tree:
- doc_data/SC_send/{norm,indevnol}/anon/*.mat (connectivity)
- doc_data/SC_send/lengths/anon/*.mat (tract lengths)
- doc_data/CNT_send/SC/CNT_SC*.mat (control)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from tvbtoolkit.core.paths import doc_liege_raw  # noqa: E402

from brain_states_new_doc_bold_audited import (
    FILE_SPECS,
    _load_mat_mapping,
    _select_3d_numeric,
    _to_subject_roi_roi,
)

COMA_FILE_SPECS = (
    {
        "cohort": "coma",
        "stage": "acute",
        "sedation": "non_sedated",
        "sc_path": "SC_send/norm/anon/DoC_acute_COMA_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "coma",
        "stage": "acute",
        "sedation": "sedated",
        "sc_path": "SC_send/norm/anon/DoC_acute_sedated_COMA_SC_matched.mat",
        "sc_var": "SC",
    },
)


def _sha256_array(arr: np.ndarray) -> str:
    h = sha256()
    c = np.ascontiguousarray(arr)
    h.update(str(c.dtype).encode("utf-8"))
    h.update(np.asarray(c.shape, dtype=np.int64).tobytes())
    h.update(c.tobytes(order="C"))
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_roi_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        code = parts[0]
        name = parts[1]
        rows.append((code, name))
    if len(rows) != 90:
        raise RuntimeError(f"Expected 90 ROI rows in {path}, found {len(rows)}")
    codes = np.asarray([r[0] for r in rows], dtype="U64")
    labels = np.asarray([r[1] for r in rows], dtype="U128")
    indices = np.arange(1, 91, dtype=np.int32)
    return labels, codes, indices


@dataclass
class CohortBuffers:
    subject_ids: list[str]
    connectivity: list[np.ndarray]
    tract_lengths: list[np.ndarray]
    source_files: list[str]
    source_indices: list[int]
    source_stage: list[str]
    source_sedation: list[str]


def _length_path_from_sc(sc_rel: str) -> str:
    if sc_rel in {
        "CNT_send/SC/CNT_SC.mat",
        "CNT_send/SC/CNT_SC_scale_invnodevol.mat",
    }:
        return "CNT_send/SC/CNT_SC_lengths.mat"
    return (
        sc_rel.replace("SC_send/norm/anon/", "SC_send/lengths/anon/")
        .replace("SC_send/indevnol/anon/", "SC_send/lengths/anon/")
    )


def _connectivity_path(sc_rel: str, sc_variant: str) -> str:
    """Resolve matched raw or inverse-node-volume connectivity input."""
    if sc_variant == "raw":
        return sc_rel
    if sc_variant != "invnodevol":
        raise ValueError("sc_variant must be 'raw' or 'invnodevol'.")
    if sc_rel == "CNT_send/SC/CNT_SC.mat":
        return "CNT_send/SC/CNT_SC_scale_invnodevol.mat"
    return sc_rel.replace("SC_send/norm/anon/", "SC_send/indevnol/anon/")


def build_dataset(
    data_root: Path,
    out_root: Path,
    *,
    sc_variant: str = "raw",
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    atlas_lookup_path = data_root / "symmetric_lookuptable_clean.txt"
    if not atlas_lookup_path.exists():
        atlas_lookup_path = data_root / "ROI_MNI_V4_90.txt"
    labels, codes, indices = _parse_roi_labels(atlas_lookup_path)
    np.savez_compressed(out_root / "atlas.npz", labels=labels, region_codes=codes, region_indices=indices)

    by_cohort: dict[str, CohortBuffers] = {}
    serial_by_cohort: dict[str, int] = {}
    source_rows: list[dict[str, Any]] = []
    zero_mask_qc_rows: list[dict[str, Any]] = []
    subjects_index_rows: list[dict[str, Any]] = []

    effective_specs = list(FILE_SPECS)
    declared_sc_paths = {str(spec["sc_path"]) for spec in effective_specs}
    for coma_spec in COMA_FILE_SPECS:
        coma_path = str(coma_spec["sc_path"])
        if coma_path not in declared_sc_paths and (data_root / coma_path).exists():
            effective_specs.append(coma_spec)

    for spec in effective_specs:
        cohort = str(spec["cohort"])
        stage = str(spec["stage"])
        sedation = str(spec["sedation"])
        sc_rel = _connectivity_path(str(spec["sc_path"]), sc_variant)
        raw_sc_rel = str(spec["sc_path"])
        tl_rel = _length_path_from_sc(sc_rel)
        sc_path = data_root / sc_rel
        tl_path = data_root / tl_rel

        sc_map = _load_mat_mapping(sc_path)
        tl_map = _load_mat_mapping(tl_path)
        _, sc_arr = _select_3d_numeric(sc_map, variable_hint=str(spec["sc_var"]))
        _, tl_arr = _select_3d_numeric(tl_map, variable_hint=str(spec["sc_var"]))
        sc_srr = _to_subject_roi_roi(sc_arr)
        tl_srr = _to_subject_roi_roi(tl_arr)
        if sc_srr.shape != tl_srr.shape:
            raise RuntimeError(f"SC/TL shape mismatch: {sc_path.name} {sc_srr.shape} vs {tl_path.name} {tl_srr.shape}")
        raw_sc_srr = None
        if sc_variant == "invnodevol":
            raw_map = _load_mat_mapping(data_root / raw_sc_rel)
            _, raw_arr = _select_3d_numeric(
                raw_map,
                variable_hint=str(spec["sc_var"]),
            )
            raw_sc_srr = _to_subject_roi_roi(raw_arr)
            if raw_sc_srr.shape != sc_srr.shape:
                raise RuntimeError(
                    f"Raw/invnodevol shape mismatch for {sc_path.name}: "
                    f"{raw_sc_srr.shape} vs {sc_srr.shape}"
                )

        buf = by_cohort.setdefault(
            cohort,
            CohortBuffers(
                subject_ids=[],
                connectivity=[],
                tract_lengths=[],
                source_files=[],
                source_indices=[],
                source_stage=[],
                source_sedation=[],
            ),
        )
        serial_by_cohort.setdefault(cohort, 0)

        for i in range(sc_srr.shape[0]):
            serial_by_cohort[cohort] += 1
            sid = f"{cohort[:1]}{serial_by_cohort[cohort]:04d}"
            c = np.asarray(sc_srr[i], dtype=np.float32)
            l = np.asarray(tl_srr[i], dtype=np.float32)
            if c.shape != (90, 90) or l.shape != (90, 90):
                raise RuntimeError(f"Unexpected matrix shape for {sid}: {c.shape}, {l.shape}")
            if raw_sc_srr is not None:
                raw_zero_mask = np.asarray(raw_sc_srr[i]) == 0.0
                inv_zero_mask = c == 0.0
                raw_only = int(np.count_nonzero(raw_zero_mask & ~inv_zero_mask))
                inv_only = int(np.count_nonzero(inv_zero_mask & ~raw_zero_mask))
                # Conservative damage mask: an edge absent in either matched
                # export cannot transmit activity. Differences are retained in
                # a QC table rather than silently discarded.
                c[raw_zero_mask] = 0.0
                zero_mask_qc_rows.append(
                    {
                        "subject_id": sid,
                        "cohort": cohort,
                        "source_sc_file": sc_rel,
                        "source_subject_index": int(i),
                        "raw_zero_inv_nonzero_entries": raw_only,
                        "inv_zero_raw_nonzero_entries": inv_only,
                        "union_zero_entries": int(np.count_nonzero(c == 0.0)),
                        "zero_masks_identical": bool(raw_only == 0 and inv_only == 0),
                    }
                )
            # A structural zero means no anatomical route. Its tract length is
            # therefore unusable and is masked here as a dataset invariant.
            np.fill_diagonal(c, 0.0)
            np.fill_diagonal(l, 0.0)
            l[c == 0.0] = 0.0

            buf.subject_ids.append(sid)
            buf.connectivity.append(c)
            buf.tract_lengths.append(l)
            buf.source_files.append(sc_rel)
            buf.source_indices.append(i)
            buf.source_stage.append(stage)
            buf.source_sedation.append(sedation)

            source_rows.append(
                {
                    "subject_id": sid,
                    "cohort": cohort,
                    "stage": stage,
                    "sedation": sedation,
                    "source_sc_file": sc_rel,
                    "source_tl_file": tl_rel,
                    "source_subject_index": int(i),
                }
            )

    cohort_meta: dict[str, Any] = {}
    for cohort in sorted(by_cohort.keys()):
        buf = by_cohort[cohort]
        ids = np.asarray(buf.subject_ids, dtype="U32")
        c_stack = np.stack(buf.connectivity, axis=0).astype(np.float32, copy=False)
        l_stack = np.stack(buf.tract_lengths, axis=0).astype(np.float32, copy=False)
        out_npz = out_root / f"subjects_{cohort}.npz"
        sc_checksums = np.asarray([_sha256_array(x) for x in c_stack], dtype="U128")
        tl_checksums = np.asarray([_sha256_array(x) for x in l_stack], dtype="U128")
        np.savez_compressed(
            out_npz,
            subject_ids=ids,
            connectivity=c_stack,
            tract_lengths=l_stack,
            sc_checksums=sc_checksums,
            tl_checksums=tl_checksums,
        )

        cohort_meta[cohort] = {
            "source_cohort": cohort.upper(),
            "n_subjects": int(ids.size),
            "subjects_file": out_npz.name,
            "subjects_file_sha256": _sha256_file(out_npz),
            "subject_ids": ids.tolist(),
            "matrix_shape": [90, 90],
        }

        for i, sid in enumerate(ids.tolist()):
            subjects_index_rows.append(
                {
                    "subject_id": sid,
                    "cohort": cohort,
                    "source_cohort": cohort.upper(),
                    "dataset_index": int(i),
                    "stage": str(buf.source_stage[i]),
                    "sedation": str(buf.source_sedation[i]),
                    "connectivity_shape": [90, 90],
                    "tract_lengths_shape": [90, 90],
                    "connectivity_sha256": str(sc_checksums[i]),
                    "tract_lengths_sha256": str(tl_checksums[i]),
                }
            )

    index = {
        "format": "tvbtoolkit.brain_act.structural_npz",
        "format_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_root": str(data_root),
        "connectivity_weights": {
            "variant": str(sc_variant),
            "source_metric": (
                "MRtrix SIFT2 streamline weights scaled by inverse endpoint-node volume"
                if sc_variant == "invnodevol"
                else "MRtrix SIFT2 streamline weights without inverse-node-volume scaling"
            ),
            "subject_rescaling": "none",
            "cohort_rescaling": "none",
            "damage_mask": "connectivity == 0; matching tract lengths forced to 0",
            "damage_mask_source": (
                "union of matched raw and invnodevol zero masks"
                if sc_variant == "invnodevol"
                else "raw connectivity zero mask"
            ),
        },
        "connectivity_normalization": {
            "scheme": "native_invnodevol" if sc_variant == "invnodevol" else "native_raw",
            "divisor": None,
            "simulator_normalization": "none",
        },
        "atlas": {
            "name": "AAL90",
            "lookup_file": atlas_lookup_path.name,
            "lookup_file_sha256": _sha256_file(atlas_lookup_path),
            "n_regions": 90,
            "ordering": "aal90_symmetric_left_then_right_reverse",
            "labels_sha256": _sha256_array(labels),
            "codes_sha256": _sha256_array(codes),
        },
        "cohorts": cohort_meta,
        "subjects": subjects_index_rows,
    }
    (out_root / "index.json").write_text(json.dumps(index, indent=2))
    pd.DataFrame(source_rows).to_csv(out_root / "source_subject_map.csv", index=False)
    if zero_mask_qc_rows:
        pd.DataFrame(zero_mask_qc_rows).to_csv(
            out_root / "zero_mask_qc.csv",
            index=False,
        )

    print("Converted dataset written to:", out_root)
    print("Counts:", {k: int(v["n_subjects"]) for k, v in cohort_meta.items()})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=str,
        default=str(doc_liege_raw("doc_data")),
    )
    p.add_argument(
        "--sc-variant",
        choices=["raw", "invnodevol"],
        default="raw",
        help="Connectivity weights to convert; no additional rescaling is applied.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=str(doc_liege_raw("doc_data", "converted_structural")),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    build_dataset(data_root, out_root, sc_variant=str(args.sc_variant))


if __name__ == "__main__":
    main()
