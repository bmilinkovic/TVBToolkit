#!/usr/bin/env python3
"""Run empirical AAL90 BOLD PhiID end-to-end.

Workflow:
1. Load subject-level empirical BOLD data from the audited DoC loader.
2. Apply the validated AAL90 FC reorder.
3. Export one MATLAB PhiID input file per subject.
4. Run the MATLAB batch runner.
5. Re-load atom outputs and compute grouped averages.
6. Save cohort-level and exact-condition heatmaps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.analysis import (  # noqa: E402
    PHIID_ATOMS,
    PRIMARY_ATOMS,
    PUBLICATION_COHORT_ORDER,
    average_atom_matrices_by_group,
    build_matlab_batch_command,
    export_phiid_subject_inputs,
    load_phiid_index,
    plot_publication_cohort_grid,
    plot_group_average_grid,
    save_group_average_outputs,
)
from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402
from brain_states_new_doc_bold_audited import (  # noqa: E402
    _maybe_apply_roi_reordering,
    build_roi_order_reference,
    load_new_doc_subjects,
    resolve_roi_order_names,
    validate_final_roi_order_or_raise,
)


def _save_plot_set(
    averages_df: pd.DataFrame,
    *,
    atom: str,
    title_cols: list[str],
    roi_labels: list[str],
    out_dir: Path,
    stem: str,
    ncols: int,
) -> None:
    fig, _ = plot_group_average_grid(
        averages_df,
        atom=atom,
        title_cols=title_cols,
        roi_labels=roi_labels,
        ncols=ncols,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}_{atom}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_{atom}.svg", bbox_inches="tight")
    plt.close(fig)


def _frame_summary(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.drop(columns=["matrix", "matrix_shape"], errors="ignore").to_dict(orient="records"):
        summary = {key: row[key] for key in group_cols if key in row}
        summary["atom"] = row["atom"]
        summary["n_subjects"] = int(row["n_subjects"])
        rows.append(summary)
    return rows


def _validate_complete_output_set(
    index_df: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    redundancy: str,
) -> None:
    expected_subjects = {
        str(x).strip()
        for x in manifest.get("subject_stub", pd.Series(dtype=str)).astype(str).tolist()
        if str(x).strip()
    }
    found_subjects = {
        str(x).strip()
        for x in index_df.get("subject_stub", pd.Series(dtype=str)).astype(str).tolist()
        if str(x).strip()
    }

    missing_subjects = sorted(expected_subjects.difference(found_subjects))
    unexpected_subjects = sorted(found_subjects.difference(expected_subjects))

    atom_frame = (
        index_df.loc[:, ["subject_stub", "atom"]]
        .drop_duplicates()
        .assign(present=True)
        .pivot(index="subject_stub", columns="atom", values="present")
        .fillna(False)
    )
    incomplete_subjects: list[dict[str, Any]] = []
    for subject_stub in sorted(expected_subjects.intersection(set(atom_frame.index.astype(str)))):
        row = atom_frame.loc[subject_stub]
        missing_atoms = [atom for atom in PHIID_ATOMS if atom not in row.index or not bool(row[atom])]
        if missing_atoms:
            incomplete_subjects.append(
                {
                    "subject_stub": subject_stub,
                    "missing_atoms": missing_atoms,
                }
            )

    if not missing_subjects and not unexpected_subjects and not incomplete_subjects:
        return

    parts: list[str] = [
        f"Incomplete PhiID output set for redundancy '{redundancy}'.",
        f"Expected {len(expected_subjects)} subjects and {len(PHIID_ATOMS)} atoms per subject.",
        f"Indexed {len(found_subjects)} unique subjects and {int(index_df.shape[0])} output files.",
    ]
    if missing_subjects:
        preview = ", ".join(missing_subjects[:10])
        suffix = " ..." if len(missing_subjects) > 10 else ""
        parts.append(f"Missing subjects ({len(missing_subjects)}): {preview}{suffix}")
    if unexpected_subjects:
        preview = ", ".join(unexpected_subjects[:10])
        suffix = " ..." if len(unexpected_subjects) > 10 else ""
        parts.append(f"Unexpected subjects ({len(unexpected_subjects)}): {preview}{suffix}")
    if incomplete_subjects:
        preview = "; ".join(
            f"{item['subject_stub']} -> {','.join(item['missing_atoms'][:6])}"
            for item in incomplete_subjects[:10]
        )
        suffix = " ..." if len(incomplete_subjects) > 10 else ""
        parts.append(f"Subjects with missing atom files ({len(incomplete_subjects)}): {preview}{suffix}")
    raise RuntimeError(" ".join(parts))


def _save_publication_cohort_figure(
    cohort_avgs: pd.DataFrame,
    *,
    roi_labels: list[str],
    out_dir: Path,
) -> None:
    fig, _ = plot_publication_cohort_grid(
        cohort_avgs,
        cohort_order=PUBLICATION_COHORT_ORDER,
        roi_labels=roi_labels,
        atoms_in_rows=("rtr", "sts"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "cohort_publication_2x5_rtr_sts"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_name(stem.name + "_transparent.png"), dpi=320, bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_name(stem.name + "_transparent.svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    input_dir = out_root / "inputs"
    phiid_dir = out_root / "phiid" / args.redundancy
    avg_root = out_root / "averages" / args.redundancy
    fig_root = out_root / "figures" / args.redundancy
    log_root = out_root / "logs"
    matlab_runner = Path(args.matlab_runner).expanduser().resolve()

    for path in (out_root, input_dir, phiid_dir, avg_root, fig_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    records, _ = load_new_doc_subjects(data_root, max_subjects_per_group=args.max_subjects_per_group)
    records_use, reorder_qc, reorder_decision = _maybe_apply_roi_reordering(records, mode=args.roi_reorder_mode)
    roi_ref = build_roi_order_reference(data_root)
    validate_final_roi_order_or_raise(roi_ref, applied_mode=str(reorder_decision["applied_mode"]))
    roi_labels, _ = resolve_roi_order_names(roi_ref, mode=str(reorder_decision["applied_mode"]))

    manifest = export_phiid_subject_inputs(
        records_use,
        input_dir,
        roi_labels=roi_labels,
        max_timepoints=args.max_timepoints,
        standardize=args.standardize,
        tr_seconds=args.tr_seconds,
    )
    manifest_path = input_dir / "manifest.csv"
    reorder_qc.to_csv(log_root / "roi_reorder_qc.csv", index=False)
    (log_root / "roi_reorder_decision.json").write_text(json.dumps(reorder_decision, indent=2))

    matlab_cmd = build_matlab_batch_command(
        input_dir=input_dir,
        output_dir=phiid_dir,
        redundancy=args.redundancy,
        matlab_bin=args.matlab_bin,
        matlab_toolbox_root=args.matlab_toolbox_root,
        runner_path=matlab_runner,
        use_parallel=args.matlab_parallel,
        n_workers=args.matlab_workers,
    )
    (log_root / "matlab_command.txt").write_text(matlab_cmd + "\n")

    if args.run_matlab:
        subprocess.run(matlab_cmd, shell=True, cwd=_REPO_ROOT, check=True)

    index_df = load_phiid_index(phiid_dir, manifest_path=manifest_path)
    if index_df.empty:
        if not args.run_matlab:
            summary = {
                "data_root": str(data_root),
                "output_root": str(out_root),
                "redundancy": args.redundancy,
                "roi_reorder_mode": str(reorder_decision["applied_mode"]),
                "n_subjects_exported": int(manifest.shape[0]),
                "n_output_files_indexed": 0,
                "matlab_command": matlab_cmd,
                "status": "exported_inputs_only",
            }
            (log_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
            return summary
        raise RuntimeError(
            f"No PhiID output matrices were found in {phiid_dir}. "
            "The MATLAB batch appears to have completed without producing outputs."
        )
    if args.require_complete:
        _validate_complete_output_set(index_df, manifest, redundancy=args.redundancy)
    index_df.to_csv(log_root / "phiid_output_index.csv", index=False)

    cohort_avgs = pd.concat(
        [
            average_atom_matrices_by_group(index_df, atom=atom, group_cols=["cohort"])
            for atom in PRIMARY_ATOMS
        ],
        ignore_index=True,
    )
    condition_avgs = pd.concat(
        [
            average_atom_matrices_by_group(index_df, atom=atom, group_cols=["cohort", "stage", "sedation"])
            for atom in PRIMARY_ATOMS
        ],
        ignore_index=True,
    )

    cohort_avgs.to_pickle(avg_root / "cohort_averages.pkl")
    condition_avgs.to_pickle(avg_root / "condition_averages.pkl")
    save_group_average_outputs(cohort_avgs, avg_root / "by_cohort")
    save_group_average_outputs(condition_avgs, avg_root / "by_condition")

    for atom in PRIMARY_ATOMS:
        _save_plot_set(
            cohort_avgs,
            atom=atom,
            title_cols=["cohort"],
            roi_labels=roi_labels,
            out_dir=fig_root / "by_cohort",
            stem="cohort_grid",
            ncols=args.cohort_plot_ncols,
        )
        _save_plot_set(
            condition_avgs,
            atom=atom,
            title_cols=["cohort", "stage", "sedation"],
            roi_labels=roi_labels,
            out_dir=fig_root / "by_condition",
            stem="condition_grid",
            ncols=args.condition_plot_ncols,
        )

    if set(PUBLICATION_COHORT_ORDER).issubset(set(cohort_avgs["cohort"].astype(str))):
        _save_publication_cohort_figure(
            cohort_avgs,
            roi_labels=roi_labels,
            out_dir=fig_root / "by_cohort",
        )

    summary = {
        "data_root": str(data_root),
        "output_root": str(out_root),
        "redundancy": args.redundancy,
        "roi_reorder_mode": str(reorder_decision["applied_mode"]),
        "n_subjects_exported": int(manifest.shape[0]),
        "n_output_files_indexed": int(index_df.shape[0]),
        "cohort_groups": _frame_summary(cohort_avgs, ["cohort"]),
        "condition_groups": _frame_summary(condition_avgs, ["cohort", "stage", "sedation"]),
    }
    (log_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=str(doc_liege_raw("doc_data")))
    parser.add_argument("--output-root", type=str, default=str(doc_liege_results("phiid_empirical_bold")))
    parser.add_argument("--redundancy", type=str, default="mmi")
    parser.add_argument("--roi-reorder-mode", type=str, default="aal90_fc")
    parser.add_argument("--standardize", type=str, default=None)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--tr-seconds", type=float, default=2.4)
    parser.add_argument("--max-subjects-per-group", type=int, default=None)
    parser.add_argument("--cohort-plot-ncols", type=int, default=3)
    parser.add_argument("--condition-plot-ncols", type=int, default=3)
    parser.add_argument("--run-matlab", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--matlab-parallel", action="store_true", default=False)
    parser.add_argument("--matlab-workers", type=int, default=0)
    parser.add_argument("--matlab-bin", type=str, default="/Applications/MATLAB_R2023b.app/bin/matlab")
    parser.add_argument("--matlab-toolbox-root", type=str, default="/Users/borjan/code/matlab/elph")
    parser.add_argument(
        "--matlab-runner",
        type=str,
        default=str(_REPO_ROOT / "scripts" / "phiid_empirical_bold_aal90.m"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
