#!/usr/bin/env python3
"""Convert Brain-Act structural data into TVBToolkit fast-loading NPZ bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tvbtoolkit.datasets.brain_act import convert_brain_act_dataset
    from tvbtoolkit.core.paths import doc_liege_raw
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tvbtoolkit.datasets.brain_act import convert_brain_act_dataset
    from tvbtoolkit.core.paths import doc_liege_raw


def main() -> None:
    default_source = doc_liege_raw("brain_act", "source")
    default_output = doc_liege_raw("brain_act", "converted")
    parser = argparse.ArgumentParser(
        description=(
            "Convert Brain-Act AAL90 subject structural connectomes/tract lengths into "
            "cohort NPZ bundles + index.json for fast loading."
        )
    )
    parser.add_argument(
        "--source-root",
        default=str(default_source),
        help=(
            "Path to Brain-Act root or data root. Accepted examples: "
            "/Users/.../Brain-Act/brain-act or /Users/.../Brain-Act/brain-act/data. "
            f"Default: {default_source}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        help=(
            "Output directory for converted dataset files "
            f"(default: {default_output})."
        ),
    )
    parser.add_argument(
        "--atlas-lookup-name",
        default="custom_lookuptable_AAL.txt",
        help="Atlas lookup filename under data/atlases/ (default: custom_lookuptable_AAL.txt).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Numeric dtype used for saved cohort bundles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing converted dataset in output-dir.",
    )
    args = parser.parse_args()

    index_path = convert_brain_act_dataset(
        source_root=Path(args.source_root),
        output_dir=Path(args.output_dir),
        atlas_lookup_name=args.atlas_lookup_name,
        dtype=args.dtype,
        overwrite=args.overwrite,
    )
    print(f"Conversion complete: {index_path}")


if __name__ == "__main__":
    main()
