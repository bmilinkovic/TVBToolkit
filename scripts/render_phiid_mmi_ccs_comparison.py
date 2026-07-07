#!/usr/bin/env python3
"""Render a publication-style MMI vs CCS PhiID cohort comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.analysis import (  # noqa: E402
    PUBLICATION_COHORT_ORDER,
    plot_publication_method_comparison_grid,
)
from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402
from brain_states_new_doc_bold_audited import (  # noqa: E402
    build_roi_order_reference,
    resolve_roi_order_names,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(doc_liege_raw("doc_data")))
    p.add_argument("--results-root", type=str, default=str(doc_liege_results("phiid_empirical_bold")))
    p.add_argument("--roi-reorder-mode", type=str, default="aal90_fc")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(doc_liege_results("phiid_empirical_bold", "figures", "mmi_ccs_comparison", "by_cohort")),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mmi_avgs = pd.read_pickle(results_root / "averages" / "mmi" / "cohort_averages.pkl")
    ccs_avgs = pd.read_pickle(results_root / "averages" / "ccs" / "cohort_averages.pkl")

    roi_ref = build_roi_order_reference(data_root)
    roi_labels, _ = resolve_roi_order_names(roi_ref, mode=args.roi_reorder_mode)

    fig, _ = plot_publication_method_comparison_grid(
        mmi_avgs,
        ccs_avgs,
        cohort_order=PUBLICATION_COHORT_ORDER,
        roi_labels=list(roi_labels),
    )

    stem = out_dir / "cohort_publication_mmi_vs_ccs_4x5_rtr_sts"
    fig.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(stem.with_name(stem.name + "_transparent.png"), dpi=360, bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_name(stem.name + "_transparent.svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
