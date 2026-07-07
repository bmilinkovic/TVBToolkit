#!/usr/bin/env python3
"""Export the tDCS/TMS-EEG PCI data into a clean Python-facing layout.

The raw dataset is kept untouched. This script creates a derived directory with:

- CSV tables for record metadata, stored PCI values, PCI time courses, and D30
  reconstruction metadata.
- Compressed NumPy arrays for dense matrices that should not be written as CSV
  (``binJ`` and source-average ``J``).
- A small README describing the layout and provenance.

Large per-trial inverse maps (``AllTf``) and sensor trials (``Droutine_*.dat``)
are referenced by path and shape instead of duplicated by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.core.paths import stimulation_raw  # noqa: E402
from tvbtoolkit.datasets.stim_pci import (  # noqa: E402
    PciRecord,
    discover_pci_files,
    load_pci_mat,
    singletrials_path,
)


def _slug(record: PciRecord) -> str:
    sess = "NA" if record.session is None else str(record.session)
    variant = re.sub(r"[^A-Za-z0-9]+", "-", record.variant).strip("-")
    return f"{record.subject}_S{sess}_{record.condition}_{variant}"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _d30_structure_path(record: PciRecord) -> Path:
    return record.path.parent.parent / "D30_structure.mat"


def _atlas_brodmann_path(record: PciRecord) -> Path:
    return Path(re.sub(r"_PCI\.mat$", "_atlasmap_bootstraps_broadmann.mat", str(record.path), flags=re.I))


def _loadmat(path: Path) -> dict:
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)


def _scalar(value, default=np.nan):
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return arr.ravel()[0].item() if hasattr(arr.ravel()[0], "item") else arr.ravel()[0]


def _write_readme(out_dir: Path) -> None:
    text = """# Clean tDCS/TMS-EEG PCI Dataset

This directory is a derived, Python-facing index of the raw tDCS stimulation
data. The raw files are not modified.

## Layout

- `manifest/pci_records.csv`: one row per discovered `Droutine_*_PCI.mat` file.
- `tables/stored_pci_summary.csv`: scalar PCI, entropy, normalization and
  active-fraction summaries.
- `tables/pci_timecourses.csv`: long-format PCI time courses saved by the
  original pipeline.
- `tables/d30_reconstruction_manifest.csv`: D30 sensor/source reconstruction
  metadata, including `.dat`, `AllTf`, and source-average paths/shapes.
- `arrays/binj/*.npz`: compressed stored `binJ` matrices (`sources x post_time`).
- `arrays/source_average_j/*.npz`: compressed source-average `J` matrices
  (`sources x time`) plus their time vectors.
- `metadata/export_summary.json`: export counts and provenance.

## Notes

The original vertex-level PCI was computed from source-space trial data. Those
trials are not saved directly as a single ready-made matrix. They can be
reconstructed from:

1. D30 sensor trials in `Droutine_*.dat`.
2. Per-trial inverse maps in `*_resfile_singletrials.mat::AllTf`.
3. D30 timing/channel metadata in `D30_structure.mat`.

The script `scripts/stim_pci_reconstructed_vertex_compare.py` performs that
reconstruction without permanently duplicating the large trial tensor.
"""
    (out_dir / "README.md").write_text(text)


def _record_base_row(record: PciRecord, raw_root: Path) -> dict[str, object]:
    return {
        "record_id": _slug(record),
        "subject": record.subject,
        "condition": record.condition,
        "session": record.session,
        "variant": record.variant,
        "is_primary": record.is_primary,
        "pci_mat_path": _rel(record.path, raw_root),
        "singletrials_path": (
            _rel(singletrials_path(record), raw_root)
            if singletrials_path(record) is not None
            else ""
        ),
        "d30_structure_path": (
            _rel(_d30_structure_path(record), raw_root)
            if _d30_structure_path(record).exists()
            else ""
        ),
    }


def export_clean_dataset(eeg_root: Path, out_dir: Path, *, primary_only: bool = False) -> dict:
    raw_root = _REPO_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in [
        "manifest",
        "tables",
        "arrays/binj",
        "arrays/source_average_j",
        "arrays/brodmann_trial_signal",
        "metadata",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    records = discover_pci_files(eeg_root)
    if primary_only:
        records = [r for r in records if r.is_primary]

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pci_time_rows: list[dict[str, object]] = []
    d30_rows: list[dict[str, object]] = []
    brodmann_rows: list[dict[str, object]] = []
    brodmann_area_rows: list[dict[str, object]] = []

    for i, record in enumerate(records, 1):
        rid = _slug(record)
        print(f"  [{i:2d}/{len(records)}] {rid}", flush=True)
        base_row = _record_base_row(record, raw_root)
        manifest_rows.append(base_row)

        pci_data = load_pci_mat(record.path)
        binJ = np.asarray(pci_data["binJ"], dtype=np.uint8)
        binj_file = out_dir / "arrays/binj" / f"{rid}_binJ.npz"
        np.savez_compressed(
            binj_file,
            binJ=binJ,
            record_id=rid,
            subject=record.subject,
            condition=record.condition,
            session=-1 if record.session is None else int(record.session),
        )

        mat = _loadmat(record.path)
        params = mat["parameters"]
        times = np.asarray(params.times, dtype=float).ravel()
        pci_curve = np.asarray(mat["PCI"], dtype=float).ravel()
        for t, v in zip(times, pci_curve):
            pci_time_rows.append({**base_row, "time_ms": float(t), "pci": float(v)})

        summary_rows.append(
            {
                **base_row,
                "n_sources": int(binJ.shape[0]),
                "n_post_bins": int(binJ.shape[1]),
                "active_bins": int(binJ.sum()),
                "active_fraction": float(binJ.mean()),
                "stored_pci": float(pci_data["stored_pci"]),
                "stored_H": float(pci_data["stored_H"]),
                "stored_norm": float(pci_data["stored_norm"]),
                "pci_time_start_ms": float(times[0]) if times.size else np.nan,
                "pci_time_stop_ms": float(times[-1]) if times.size else np.nan,
                "binj_npz": _rel(binj_file, out_dir),
            }
        )

        d30_path = _d30_structure_path(record)
        sib = singletrials_path(record)
        d30_row = {**base_row}
        if d30_path.exists():
            d30 = _loadmat(d30_path)["D30_structure"]
            times_full = np.asarray(d30.Data.samples.times, dtype=float).ravel()
            bad = np.atleast_1d(np.asarray(d30.Data.channels.Bad, dtype=int)).ravel()
            bad = bad[bad > 0]
            dat_path = record.path.parent / str(d30.Data.fnamedat)
            d30_row.update(
                {
                    "n_sensor_samples": int(d30.Data.Nsamples),
                    "n_sensor_channels": int(d30.Data.Nchannels),
                    "n_trials": int(d30.Data.Nevents),
                    "sampling_rate_hz": float(d30.Data.Radc),
                    "sensor_time_start_ms": float(times_full[0]),
                    "sensor_time_stop_ms": float(times_full[-1]),
                    "inverse_window_start_ms": float(np.asarray(d30.Inverse.woi).ravel()[0]),
                    "inverse_window_stop_ms": float(np.asarray(d30.Inverse.woi).ravel()[-1]),
                    "baseline_start_ms": float(np.asarray(d30.Data.BaseCorr.times).ravel()[0]),
                    "baseline_stop_ms": float(np.asarray(d30.Data.BaseCorr.times).ravel()[-1]),
                    "bad_channels_1based": ";".join(map(str, bad.tolist())),
                    "sensor_dat_path": _rel(dat_path, raw_root),
                    "sensor_dat_dtype": str(d30.Data.datatype),
                }
            )
        if sib is not None:
            smat = _loadmat(sib)
            J = np.asarray(smat["J"], dtype=np.float32)
            alltf_shape = tuple(int(v) for v in np.asarray(smat["AllTf"]).shape) if "AllTf" in smat else ()
            j_file = out_dir / "arrays/source_average_j" / f"{rid}_source_average_J.npz"
            times_for_j = (
                np.linspace(
                    float(d30_row.get("inverse_window_start_ms", -400.0)),
                    float(d30_row.get("inverse_window_stop_ms", 400.0)),
                    J.shape[1],
                )
                if J.ndim == 2
                else np.array([], dtype=float)
            )
            np.savez_compressed(
                j_file,
                J=J,
                times_ms=times_for_j.astype(np.float32),
                record_id=rid,
            )
            d30_row.update(
                {
                    "source_average_shape": "x".join(map(str, J.shape)),
                    "alltf_shape": "x".join(map(str, alltf_shape)),
                    "source_average_j_npz": _rel(j_file, out_dir),
                    "full_vertex_source_trials": "reconstructable_from_sensor_dat_and_AllTf",
                }
            )
        d30_rows.append(d30_row)

        atlas_path = _atlas_brodmann_path(record)
        if atlas_path.exists():
            amat = _loadmat(atlas_path)
            signal = np.asarray(amat["signal"], dtype=np.float32)
            areas = np.asarray(amat["area"], dtype=object).ravel()
            vertexes = np.asarray(amat["vertexes"], dtype=object).ravel()
            area_labels = np.asarray([str(x) for x in areas], dtype=object)
            vertex_counts = np.asarray(
                [int(np.asarray(v).size) for v in vertexes],
                dtype=np.int32,
            )
            b_file = out_dir / "arrays/brodmann_trial_signal" / f"{rid}_brodmann_signal.npz"
            times_for_signal = (
                np.linspace(
                    float(d30_row.get("inverse_window_start_ms", -400.0)),
                    float(d30_row.get("inverse_window_stop_ms", 400.0)),
                    signal.shape[2],
                )
                if signal.ndim == 3
                else np.array([], dtype=float)
            )
            # Stored orientation is areas x trials x time. Also include a
            # trials-first view for immediate use by the Casali routines.
            np.savez_compressed(
                b_file,
                signal_area_trial_time=signal,
                signal_trial_area_time=np.transpose(signal, (1, 0, 2)),
                times_ms=times_for_signal.astype(np.float32),
                area_labels=area_labels,
                vertex_counts=vertex_counts,
                record_id=rid,
            )
            brodmann_rows.append(
                {
                    **base_row,
                    "brodmann_npz": _rel(b_file, out_dir),
                    "brodmann_shape_area_trial_time": "x".join(map(str, signal.shape)),
                    "n_areas": int(signal.shape[0]),
                    "n_trials": int(signal.shape[1]),
                    "n_time_bins": int(signal.shape[2]),
                    "time_start_ms": float(times_for_signal[0]),
                    "time_stop_ms": float(times_for_signal[-1]),
                    "raw_brodmann_mat_path": _rel(atlas_path, raw_root),
                }
            )
            for area, n_vertices in zip(area_labels, vertex_counts):
                brodmann_area_rows.append(
                    {
                        **base_row,
                        "area_label": str(area),
                        "vertex_count": int(n_vertices),
                        "brodmann_npz": _rel(b_file, out_dir),
                    }
                )
        else:
            brodmann_rows.append(
                {
                    **base_row,
                    "brodmann_npz": "",
                    "brodmann_shape_area_trial_time": "",
                    "n_areas": 0,
                    "n_trials": 0,
                    "n_time_bins": 0,
                    "time_start_ms": np.nan,
                    "time_stop_ms": np.nan,
                    "raw_brodmann_mat_path": "",
                }
            )

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest/pci_records.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "tables/stored_pci_summary.csv", index=False)
    pd.DataFrame(pci_time_rows).to_csv(out_dir / "tables/pci_timecourses.csv", index=False)
    pd.DataFrame(d30_rows).to_csv(out_dir / "tables/d30_reconstruction_manifest.csv", index=False)
    pd.DataFrame(brodmann_rows).to_csv(out_dir / "tables/brodmann_signal_manifest.csv", index=False)
    pd.DataFrame(brodmann_area_rows).to_csv(out_dir / "tables/brodmann_area_vertices.csv", index=False)

    summary = {
        "eeg_root": str(eeg_root),
        "out_dir": str(out_dir),
        "primary_only": bool(primary_only),
        "n_records": len(records),
        "n_subjects": len({r.subject for r in records}),
        "subjects": sorted({r.subject for r in records}),
        "array_format": "compressed NumPy .npz",
        "table_format": "CSV",
    }
    (out_dir / "metadata/export_summary.json").write_text(json.dumps(summary, indent=2))
    _write_readme(out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-root", type=Path, default=stimulation_raw("stim_data", "tdcs-eeg"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=stimulation_raw("stim_data", "python_clean"),
    )
    ap.add_argument("--primary-only", action="store_true")
    args = ap.parse_args()

    summary = export_clean_dataset(args.eeg_root, args.out_dir, primary_only=args.primary_only)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
