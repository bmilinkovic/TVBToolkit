#!/usr/bin/env python3
"""Search the stimulation dataset for explicit tDCS protocol strings.

The acquisition protocol is described in the paper, but the raw dataset has a
mixture of MATLAB, Nexstim, DICOM, EEG, and binary exports. This script scans
file *contents* (not just paths) for protocol-relevant text such as tDCS,
anode/cathode placement, duration, current, and device names, then writes a
CSV of snippets that can be audited manually.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.core.paths import stimulation_raw  # noqa: E402

DEFAULT_ROOT = stimulation_raw("stim_data")
DEFAULT_OUT = stimulation_raw("stim_data", "python_clean_primary")
DEFAULT_SUFFIXES = {
    ".csv",
    ".dcm",
    ".hdr",
    ".ini",
    ".json",
    ".log",
    ".mat",
    ".nbe",
    ".nbs",
    ".nbx",
    ".nii",
    ".nxi",
    ".txt",
    ".xml",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "python_clean_primary",
}

SKIP_SUFFIXES = {
    ".bmp",
    ".dat",
    ".jpg",
    ".jpeg",
    ".npz",
    ".png",
    ".pyc",
}

PATTERNS: dict[str, bytes] = {
    "tdcs": rb"\btdcs\b",
    "pre_post_tdcs_label": rb"\b(?:pre|post)\s+tdcs\b",
    "anode": rb"\banod(?:e|al)?\b",
    "cathode": rb"\bcathod(?:e|al)?\b",
    "dlpfc": rb"\bdlpfc\b",
    "supra_orbitofrontal": rb"supra[-\s]?orbito(?:frontal)?",
    "neuroconn": rb"\bneuroconn\b",
    "dc_stimulator": rb"\bdc\s+stimulator\b",
    "duration_20_min": rb"\b20\s*(?:min|mins|minute|minutes)\b",
    "duration_1200_sec": rb"\b1200\s*(?:s|sec|secs|second|seconds)\b",
    "current_2ma": rb"\b2\s*mA\b|\b2mA\b|\b2000\s*(?:uA|microA|microamp)",
    "stimulation": rb"\bstimulation\b",
}


@dataclass(frozen=True)
class Hit:
    file_path: str
    file_size_bytes: int
    suffix: str
    pattern: str
    byte_offset: int
    snippet: str


def _iter_files(root: Path, max_size_mb: float, suffixes: set[str]) -> list[Path]:
    max_size = int(max_size_mb * 1024 * 1024)
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        try:
            if path.stat().st_size > max_size:
                continue
        except OSError:
            continue
        out.append(path)
    return sorted(out)


def _snippet(raw: bytes, offset: int, width: int = 140) -> str:
    start = max(0, offset - width)
    stop = min(len(raw), offset + width)
    chunk = raw[start:stop]
    # Preserve readable ASCII/Latin-1 text while collapsing binary noise.
    text = chunk.decode("latin-1", errors="ignore")
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scan_dataset(
    root: Path,
    *,
    max_size_mb: float,
    max_hits_per_file_pattern: int,
    suffixes: set[str],
) -> tuple[list[Hit], dict]:
    files = _iter_files(root, max_size_mb=max_size_mb, suffixes=suffixes)
    compiled = {
        name: re.compile(pattern, flags=re.IGNORECASE)
        for name, pattern in PATTERNS.items()
    }

    hits: list[Hit] = []
    scanned_bytes = 0
    skipped_unreadable = 0
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            skipped_unreadable += 1
            continue
        scanned_bytes += len(raw)
        for name, regex in compiled.items():
            count = 0
            for match in regex.finditer(raw):
                hits.append(
                    Hit(
                        file_path=str(path),
                        file_size_bytes=len(raw),
                        suffix=path.suffix.lower(),
                        pattern=name,
                        byte_offset=int(match.start()),
                        snippet=_snippet(raw, int(match.start())),
                    )
                )
                count += 1
                if count >= max_hits_per_file_pattern:
                    break

    summary = {
        "root": str(root),
        "n_files_scanned": len(files),
        "n_hits": len(hits),
        "scanned_bytes": scanned_bytes,
        "skipped_unreadable": skipped_unreadable,
        "patterns": sorted(PATTERNS),
        "suffixes": sorted(suffixes),
        "hits_by_pattern": {
            pattern: sum(1 for hit in hits if hit.pattern == pattern)
            for pattern in sorted(PATTERNS)
        },
    }
    return hits, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-size-mb", type=float, default=25.0)
    parser.add_argument("--max-hits-per-file-pattern", type=int, default=25)
    parser.add_argument(
        "--suffixes",
        default=",".join(sorted(DEFAULT_SUFFIXES)),
        help="Comma-separated suffix allow-list; use an empty string to scan every non-skipped suffix.",
    )
    args = parser.parse_args()
    suffixes = {
        item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}"
        for item in args.suffixes.split(",")
        if item.strip()
    }

    hits, summary = scan_dataset(
        args.root,
        max_size_mb=args.max_size_mb,
        max_hits_per_file_pattern=args.max_hits_per_file_pattern,
        suffixes=suffixes,
    )

    tables_dir = args.out_dir / "tables"
    metadata_dir = args.out_dir / "metadata"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "tdcs_protocol_search_hits.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "file_size_bytes",
                "suffix",
                "pattern",
                "byte_offset",
                "snippet",
            ],
        )
        writer.writeheader()
        for hit in hits:
            writer.writerow(hit.__dict__)

    summary_path = metadata_dir / "tdcs_protocol_search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {len(hits)} hits to {csv_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary["hits_by_pattern"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
