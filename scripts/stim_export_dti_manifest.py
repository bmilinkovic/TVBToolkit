#!/usr/bin/env python3
"""Create a clean manifest for the tDCS DTI / navigation imaging files.

This does not convert imaging payloads. It indexes what is available per
subject, including DICOM, Analyze/NIfTI, screenshots, and Nexstim navigation
files (``.nbs/.nbx/.nbe``), so downstream Python code can find the right raw
assets without walking the messy tree.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _subject_from_path(path: Path) -> str:
    m = re.search(r"C_tD(?:CS|SC)_([A-Za-z0-9]+)", str(path))
    return m.group(1) if m else path.name


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def export_dti_manifest(dti_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "metadata").mkdir(exist_ok=True)

    rows = []
    file_rows = []
    for subject_dir in sorted([p for p in dti_root.iterdir() if p.is_dir()]):
        subject = _subject_from_path(subject_dir)
        files = [p for p in subject_dir.rglob("*") if p.is_file()]
        counts: dict[str, int] = {}
        for p in files:
            suffix = p.suffix.lower() or "<none>"
            counts[suffix] = counts.get(suffix, 0) + 1
            file_rows.append(
                {
                    "subject": subject,
                    "subject_folder": subject_dir.name,
                    "file_path": _rel(p, dti_root.parent.parent),
                    "relative_to_subject": str(p.relative_to(subject_dir)),
                    "suffix": suffix,
                    "size_bytes": int(p.stat().st_size),
                }
            )

        def first(pattern: str) -> str:
            matches = sorted(subject_dir.rglob(pattern))
            return _rel(matches[0], dti_root.parent.parent) if matches else ""

        rows.append(
            {
                "subject": subject,
                "subject_folder": subject_dir.name,
                "n_files": len(files),
                "n_dicom": counts.get(".dcm", 0),
                "n_bmp": counts.get(".bmp", 0),
                "n_logs": counts.get(".log", 0),
                "has_hdr_img": bool(counts.get(".hdr", 0) and counts.get(".img", 0)),
                "has_nifti": bool(counts.get(".nii", 0) or counts.get(".gz", 0)),
                "n_nbs": counts.get(".nbs", 0),
                "n_nbx": counts.get(".nbx", 0),
                "n_nbe": counts.get(".nbe", 0),
                "dicom_headers_mat": first("dicom_headers.mat"),
                "nifti_file": first("*.nii"),
                "analyze_hdr": first("*.hdr"),
                "analyze_img": first("*.img"),
                "nexstim_session_nbs": first("*.nbs"),
                "nexstim_export_nbx": first("*.nbx"),
                "nexstim_export_nbe": first("*.nbe"),
                "first_log": first("*.log"),
                "dti_dir": _rel(subject_dir / "DTI", dti_root.parent.parent)
                if (subject_dir / "DTI").exists()
                else "",
                "t1_dir": _rel(subject_dir / "T1", dti_root.parent.parent)
                if (subject_dir / "T1").exists()
                else "",
                "dicom_export_dir": _rel(subject_dir / "DICOMExport", dti_root.parent.parent)
                if (subject_dir / "DICOMExport").exists()
                else first("DICOMExport"),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "tables/dti_tdcs_subject_manifest.csv", index=False)
    pd.DataFrame(file_rows).to_csv(out_dir / "tables/dti_tdcs_file_manifest.csv", index=False)
    summary = {
        "dti_root": str(dti_root),
        "out_dir": str(out_dir),
        "n_subject_folders": len(rows),
        "subjects": sorted({r["subject"] for r in rows}),
        "note": (
            "This is an index of raw imaging/navigation assets. It does not "
            "convert DICOM/Analyze/NIfTI payloads or infer tDCS electrode fields."
        ),
    }
    (out_dir / "metadata/dti_tdcs_export_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dti-root", type=Path, default=Path("data/stim_data/tdcs-dti"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/stim_data/python_clean_primary"))
    args = ap.parse_args()
    summary = export_dti_manifest(args.dti_root, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
