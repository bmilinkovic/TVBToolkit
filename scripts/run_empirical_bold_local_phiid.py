#!/usr/bin/env python3
"""Prepare local dynamic STS/RTR PhiID extraction for empirical AAL90 BOLD."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.analysis import export_phiid_subject_inputs  # noqa: E402
from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402
from brain_states_new_doc_bold_audited import (  # noqa: E402
    _maybe_apply_roi_reordering,
    build_roi_order_reference,
    load_new_doc_subjects,
    resolve_roi_order_names,
    validate_final_roi_order_or_raise,
)


def build_command(
    *,
    input_dir: Path,
    output_dir: Path,
    redundancy: str,
    matlab_bin: str,
    matlab_toolbox_root: str,
    runner_path: Path,
    use_parallel: bool,
    n_workers: int,
) -> str:
    statements = [f"addpath(genpath('{Path(matlab_toolbox_root).expanduser().resolve().as_posix()}'))"]
    statements.append(f"addpath('{runner_path.parent.as_posix()}')")
    statements.append(
        "phiid_empirical_bold_local_sts_rtr_aal90("
        f"'{input_dir.as_posix()}', "
        f"'{output_dir.as_posix()}', "
        f"'{redundancy}', "
        f"{str(bool(use_parallel)).lower()}, "
        f"{int(n_workers)})"
    )
    return f'{matlab_bin} -batch "' + "; ".join(statements) + '"'


def run(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    input_dir = output_root / "inputs"
    local_dir = output_root / "local_phiid" / args.redundancy
    log_dir = output_root / "logs"
    for p in (output_root, input_dir, local_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

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
    reorder_qc.to_csv(log_dir / "roi_reorder_qc_local.csv", index=False)
    (log_dir / "roi_reorder_decision_local.json").write_text(json.dumps(reorder_decision, indent=2))

    runner_path = Path(args.matlab_runner).expanduser().resolve()
    matlab_cmd = build_command(
        input_dir=input_dir,
        output_dir=local_dir,
        redundancy=args.redundancy,
        matlab_bin=args.matlab_bin,
        matlab_toolbox_root=args.matlab_toolbox_root,
        runner_path=runner_path,
        use_parallel=args.matlab_parallel,
        n_workers=args.matlab_workers,
    )
    (log_dir / "matlab_local_phiid_command.txt").write_text(matlab_cmd + "\n")
    if args.run_matlab:
        subprocess.run(matlab_cmd, shell=True, cwd=_REPO_ROOT, check=True)

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "local_output_dir": str(local_dir),
        "n_subjects_exported": int(manifest.shape[0]),
        "redundancy": args.redundancy,
        "roi_reorder_mode": str(reorder_decision["applied_mode"]),
        "matlab_command": matlab_cmd,
        "status": "prepared_local_phiid_inputs",
    }
    (log_dir / "run_local_phiid_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(doc_liege_raw("doc_data")))
    p.add_argument("--output-root", type=str, default=str(doc_liege_results("phiid_empirical_bold_dynamic")))
    p.add_argument("--redundancy", type=str, default="mmi")
    p.add_argument("--roi-reorder-mode", type=str, default="aal90_fc")
    p.add_argument("--standardize", type=str, default=None)
    p.add_argument("--max-timepoints", type=int, default=None)
    p.add_argument("--tr-seconds", type=float, default=2.4)
    p.add_argument("--max-subjects-per-group", type=int, default=None)
    p.add_argument("--run-matlab", action="store_true")
    p.add_argument("--matlab-parallel", action="store_true", default=False)
    p.add_argument("--matlab-workers", type=int, default=0)
    p.add_argument("--matlab-bin", type=str, default="/Applications/MATLAB_R2023b.app/bin/matlab")
    p.add_argument("--matlab-toolbox-root", type=str, default="/Users/borjan/code/matlab/elph")
    p.add_argument(
        "--matlab-runner",
        type=str,
        default=str(_REPO_ROOT / "scripts" / "phiid_empirical_bold_local_sts_rtr_aal90.m"),
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))
