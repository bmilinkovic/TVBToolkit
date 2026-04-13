#!/usr/bin/env python3
"""Mirror Brain-Act source data into TVBToolkit/data/brain_act/source."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_src = Path("/Users/borjan/code/Brain-Act/brain-act/data")
    default_dst = repo_root / "data" / "brain_act" / "source"

    parser = argparse.ArgumentParser(
        description=(
            "Copy Brain-Act data locally into this repository so conversion/loading "
            "does not depend on an external checkout."
        )
    )
    parser.add_argument(
        "--source",
        default=str(default_src),
        help=f"Brain-Act data source directory (default: {default_src})",
    )
    parser.add_argument(
        "--dest",
        default=str(default_dst),
        help=f"Destination directory under this repo (default: {default_dst})",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip copying the raw/ folder (organized + atlases + metadata only).",
    )
    parser.add_argument(
        "--clean-dest",
        action="store_true",
        help="Delete existing destination before copy.",
    )
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    dst = Path(args.dest).expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")
    if not (src / "organized").exists():
        raise FileNotFoundError(f"Expected organized/ in source: {src}")

    if args.clean_dest and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    folders = ["atlases", "organized", "processed", "README.md"]
    if not args.no_raw:
        folders.append("raw")

    copied = []
    for name in folders:
        s = src / name
        d = dst / name
        if not s.exists():
            continue
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        copied.append(name)

    print(f"Synced Brain-Act data from: {src}")
    print(f"Into: {dst}")
    print("Copied entries:", ", ".join(copied))


if __name__ == "__main__":
    main()

