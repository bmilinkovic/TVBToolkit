#!/usr/bin/env python3
"""Render a composite cohort graph-metrics figure for MMI vs CCS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.core.paths import doc_liege_results  # noqa: E402
from run_luppi2022_doc_downstream import _plot_graph_metrics_comparison_composite  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=str, default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022")))
    p.add_argument(
        "--output-root",
        type=str,
        default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022", "mmi_ccs_comparison", "figures", "graph_metrics")),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mmi_df = pd.read_csv(results_root / "mmi" / "tables" / "cohort_metrics.csv")
    ccs_df = pd.read_csv(results_root / "ccs" / "tables" / "cohort_metrics.csv")

    _plot_graph_metrics_comparison_composite(
        mmi_df,
        ccs_df,
        out_dir=output_root,
        stem="cohort_graph_metrics_bar_mmi_vs_ccs",
    )


if __name__ == "__main__":
    main()
