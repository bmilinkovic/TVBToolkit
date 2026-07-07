#!/usr/bin/env python3
"""Render composite MMI-vs-CCS FC/SC similarity figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import pandas as pd

from tvbtoolkit.core.paths import doc_liege_results  # noqa: E402
from run_luppi2022_doc_downstream import _plot_similarity_comparison_composite  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=str, default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022")))
    p.add_argument(
        "--output-root",
        type=str,
        default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022", "mmi_ccs_comparison", "figures")),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mmi_subject_df = pd.read_csv(results_root / "mmi" / "tables" / "subject_similarity_metrics.csv")
    ccs_subject_df = pd.read_csv(results_root / "ccs" / "tables" / "subject_similarity_metrics.csv")

    _plot_similarity_comparison_composite(
        mmi_subject_df,
        ccs_subject_df,
        value_cols=["fc_vs_rtr_rho", "fc_vs_sts_rho"],
        titles=["FC vs RTR", "FC vs STS"],
        out_dir=output_root / "fc_similarity",
        stem="subject_fc_similarity_mmi_vs_ccs_by_cohort",
        figure_title="Subject-Level FC Similarity: MMI vs CCS",
    )
    _plot_similarity_comparison_composite(
        mmi_subject_df,
        ccs_subject_df,
        value_cols=["sc_vs_rtr_rho", "sc_vs_sts_rho"],
        titles=["SC vs RTR", "SC vs STS"],
        out_dir=output_root / "sc_similarity",
        stem="subject_sc_similarity_mmi_vs_ccs_by_cohort",
        figure_title="Subject-Level SC Similarity: MMI vs CCS",
    )


if __name__ == "__main__":
    main()
