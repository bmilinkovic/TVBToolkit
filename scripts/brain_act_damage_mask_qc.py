"""QC report for Brain-Act lesion/damage zero-mask parity.

This verifies the masking convention used in the original Brain-Act repository:
- Structural damage is represented by exact zeros in subject SC matrices.
- Patient cohorts (MCS/UWS) are expected to have matching zero edges in TL.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402
from tvbtoolkit.datasets.brain_act import list_subjects, load_subject_structural


def _upper_tri_stats(c: np.ndarray, l: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices_from(c, k=1)
    c0 = c[iu] == 0.0
    l0 = l[iu] == 0.0
    n_edges = int(c0.size)
    sc_zero = int(np.sum(c0))
    tl_zero = int(np.sum(l0))
    mismatch = int(np.sum(c0 & ~l0))
    return {
        "n_edges_upper": n_edges,
        "sc_zero_edges_upper": sc_zero,
        "tl_zero_edges_upper": tl_zero,
        "sc_zero_tl_nonzero_upper": mismatch,
        "sc_zero_fraction_upper": float(sc_zero / max(n_edges, 1)),
        "tl_zero_fraction_upper": float(tl_zero / max(n_edges, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(doc_liege_raw("brain_act", "converted")))
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(doc_liege_results("brain_act_mask_qc")),
        help="Where to save QC reports and figure.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["control", "mcs", "uws"],
        help="Cohorts to evaluate.",
    )
    args = parser.parse_args()

    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    per_subject_rows = []
    for cohort in args.cohorts:
        subjects = list_subjects(dataset_root=args.dataset_root, cohort=cohort)
        for sid in subjects:
            c, l, _, meta = load_subject_structural(
                subject_id=sid,
                cohort=cohort,
                dataset_root=args.dataset_root,
                validate=False,
            )
            np.fill_diagonal(c, 0.0)
            np.fill_diagonal(l, 0.0)
            stats = _upper_tri_stats(c, l)
            per_subject_rows.append(
                {
                    "cohort": meta.cohort,
                    "subject_id": sid,
                    **stats,
                }
            )

    # Save per-subject table
    table_path = out / "brain_act_damage_mask_per_subject.tsv"
    cols = [
        "cohort",
        "subject_id",
        "n_edges_upper",
        "sc_zero_edges_upper",
        "tl_zero_edges_upper",
        "sc_zero_tl_nonzero_upper",
        "sc_zero_fraction_upper",
        "tl_zero_fraction_upper",
    ]
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for row in per_subject_rows:
            w.writerow(row)

    # Cohort summary
    cohort_summary = {}
    for cohort in sorted(set(r["cohort"] for r in per_subject_rows)):
        rows = [r for r in per_subject_rows if r["cohort"] == cohort]
        cohort_summary[cohort] = {
            "n_subjects": len(rows),
            "mean_sc_zero_fraction_upper": float(np.mean([r["sc_zero_fraction_upper"] for r in rows])),
            "mean_tl_zero_fraction_upper": float(np.mean([r["tl_zero_fraction_upper"] for r in rows])),
            "max_sc_zero_tl_nonzero_upper": float(np.max([r["sc_zero_tl_nonzero_upper"] for r in rows])),
            "n_subjects_with_sc0_tl_nonzero": int(np.sum([r["sc_zero_tl_nonzero_upper"] > 0 for r in rows])),
        }

    summary_path = out / "brain_act_damage_mask_summary.json"
    summary_path.write_text(json.dumps(cohort_summary, indent=2), encoding="utf-8")

    # Figure: SC-zero fractions by cohort
    cohorts = sorted(cohort_summary.keys())
    values = [[r["sc_zero_fraction_upper"] for r in per_subject_rows if r["cohort"] == c] for c in cohorts]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bp = ax.boxplot(values, tick_labels=cohorts, showfliers=False, patch_artist=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    for i, c in enumerate(cohorts, start=1):
        y = np.asarray([r["sc_zero_fraction_upper"] for r in per_subject_rows if r["cohort"] == c])
        x = np.full_like(y, fill_value=i, dtype=float)
        jitter = np.linspace(-0.08, 0.08, num=max(len(y), 1))
        ax.scatter(x + jitter[: len(y)], y, s=18, color="#222222", alpha=0.7)

    ax.set_title("Brain-Act SC Damage (Zero-Edge Fraction)")
    ax.set_xlabel("Cohort")
    ax.set_ylabel("SC zero-edge fraction (upper triangle)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_path = out / "brain_act_damage_mask_qc.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {table_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
