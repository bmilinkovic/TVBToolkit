"""Scale the *unchanged* local Dalla Porta AAL90 protocol to all non-COMA subjects.

The local study's run_trial is deliberately reused, so waveform, source,
stochastic drive, seed construction, monitor and diagnostic rules stay exact.
One Slurm array element runs the 50 trials of one subject × leak × occupancy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiments/della_porta_second_order"
sys.path[:0] = [str(PILOT), str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import numpy as np

import study as local

TRIALS = local.TRIALS
PROTOCOL = dict(local.PROTOCOL, version="della_porta_aal90_full_noncoma_50_v1",
                scope="all_noncoma")


def manifest_for(dataset: Path) -> dict:
    index = json.loads((dataset / "index.json").read_text())
    norm = index["connectivity_normalization"]
    if (norm["scheme"] != PROTOCOL["structural_normalization"]
            or not np.isclose(float(norm["divisor"]), PROTOCOL["structural_divisor"],
                              rtol=0, atol=1e-9)):
        raise ValueError("Expected shared-control-maximum AAL90 connectomes")
    subjects = {cohort: sorted(index["cohorts"][cohort]["subject_ids"])
                for cohort in local.GROUPS}
    if sum(map(len, subjects.values())) != 179:
        raise ValueError(f"Expected 179 non-COMA subjects, found {subjects}")
    cases = [dict(cohort=cohort, group=local.GROUPS[cohort][0], subject=subject,
                  b_pa=local.GROUPS[cohort][1], leak_mv=leak, occupancy=occ)
             for leak in local.LEAKS for cohort in local.GROUPS
             for subject in subjects[cohort] for occ in local.OCCUPANCIES]
    sources = (Path(__file__), PILOT / "study.py",
               ROOT / "scripts/sweep_split_leak_tep_drive.py",
               ROOT / "scripts/b_sweep_publication_run.py",
               ROOT / "scripts/run_adex_impulse_response.py",
               ROOT / "scripts/first_order_amplitude_trials.py",
               ROOT / "experiments/gnak_order_comparison/model_adapter.py",
               ROOT / "src/tvbtoolkit/workflows/pharmacology.py",
               ROOT / "src/tvbtoolkit/whole_brain/simulation.py",
               ROOT / "src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_gK_gNa.py")
    return dict(protocol=PROTOCOL, subjects=subjects, cases=cases,
                dataset_index_sha256=local.sha256(dataset / "index.json"),
                receptor_sha256=local.sha256(ROOT / "data/receptors/hansen_receptors_aal90.csv"),
                source_sha256={str(path.relative_to(ROOT)): local.sha256(path)
                               for path in sources})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "case", "status"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    manifest_path = args.bank / "manifest.json"
    if args.command == "prepare":
        planned = manifest_for(args.dataset)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != planned:
                raise ValueError("Existing Dalla Porta bank has different inputs/code")
        else:
            local.write_json(manifest_path, planned)
        print(f"{sum(map(len, planned['subjects'].values()))} subjects; "
              f"{len(planned['cases'])} cases; {len(planned['cases']) * TRIALS} trials; "
              f"SLURM array=0-{len(planned['cases'])//2-1} (two cases per task)")
        return
    saved = json.loads(manifest_path.read_text())
    if saved != manifest_for(args.dataset):
        raise ValueError("Dataset, receptor map or model code changed since prepare")
    if args.command == "status":
        done = sum((args.bank / "case_complete" / f"case_{i:04d}.json").exists()
                   for i in range(len(saved["cases"])))
        print(f"{done}/{len(saved['cases'])} subject×leak×occupancy cases complete")
        return
    if args.case_index is None or not 0 <= args.case_index < len(saved["cases"]):
        parser.error("case requires a valid --case-index")
    if args.workers < 1:
        parser.error("workers must be positive")
    case = saved["cases"][args.case_index]
    jobs = [(str(args.bank), str(args.dataset), args.case_index, case, trial)
            for trial in range(TRIALS)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(local.run_trial, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == TRIALS:
                print(f"case {args.case_index}: {done}/{TRIALS} trials", flush=True)
    marker = args.bank / "case_complete" / f"case_{args.case_index:04d}.json"
    local.write_json(marker, dict(case_index=args.case_index, case=case,
                                  trials=TRIALS))


if __name__ == "__main__":
    main()
