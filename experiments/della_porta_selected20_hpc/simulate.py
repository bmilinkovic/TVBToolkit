"""Run the exact paused 20-subject Dalla Porta study as a Slurm case array.

One array task runs 50 trials of one subject × leak × occupancy case. The
scientific simulation is the local study.run_trial, unchanged. Trial seeds,
randomized onsets, source, waveform, noise and recordings therefore match the
local study; only the parallel scheduling and output root differ.
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
sys.path[:0] = [str(ROOT / "experiments/della_porta_second_order"),
                str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import study as local


def manifest_for(dataset: Path) -> dict:
    manifest = local.manifest_for(dataset)
    if sum(map(len, manifest["subjects"].values())) != 20 or len(manifest["cases"]) != 160:
        raise ValueError("Expected the exact 5 subjects/cohort × 2 leaks × 4 occupancies study")
    # Keep the original protocol and case order; only record the HPC wrapper
    # and additional executed sources to protect against mixed-code restarts.
    sources = manifest["source_sha256"]
    for path in (Path(__file__),
                 ROOT / "scripts/run_adex_impulse_response.py",
                 ROOT / "scripts/first_order_amplitude_trials.py",
                 ROOT / "src/tvbtoolkit/workflows/pharmacology.py"):
        sources[str(path.relative_to(ROOT))] = local.sha256(path)
    return manifest


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
                raise ValueError("Existing selected20 bank has different code or data")
        else:
            local.write_json(manifest_path, planned)
        print(f"20 subjects; {len(planned['cases'])} cases; "
              f"{len(planned['cases']) * local.TRIALS} trials; "
              f"SLURM array=0-{len(planned['cases'])-1}")
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest != manifest_for(args.dataset):
        raise ValueError("Code, dataset or receptor map changed since prepare")
    if args.command == "status":
        complete = sum((args.bank / "case_complete" / f"case_{i:03d}.json").exists()
                       for i in range(len(manifest["cases"])))
        print(f"{complete}/{len(manifest['cases'])} cases complete")
        return
    if args.case_index is None or not 0 <= args.case_index < len(manifest["cases"]):
        parser.error("case requires a valid --case-index")
    if args.workers < 1:
        parser.error("workers must be positive")
    case = manifest["cases"][args.case_index]
    payloads = [(str(args.bank), str(args.dataset), args.case_index, case, trial)
                for trial in range(local.TRIALS)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(local.run_trial, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == local.TRIALS:
                print(f"case {args.case_index}: {done}/{local.TRIALS} trials", flush=True)
    marker = args.bank / "case_complete" / f"case_{args.case_index:03d}.json"
    local.write_json(marker, dict(case_index=args.case_index, case=case,
                                  trials=local.TRIALS))


if __name__ == "__main__":
    main()
