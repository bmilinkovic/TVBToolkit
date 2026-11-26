"""Prepare, run and check only the 75 MCS subjects at b=32 pA.

One SLURM array task handles one subject and one occupancy (100 stochastic
trials). All other model settings are the frozen first-order full100 settings.
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
sys.path[:0] = [str(ROOT / "experiments/sma8s_full100"), str(ROOT / "scripts"),
                str(ROOT / "src"), str(ROOT)]

from experiments.sma8s_mcs_b32_full100.contract import (
    TRIALS, atomic_json, manifest_for, protocol, sha256_file,
    validate_dataset, verify_reference,
)


def run_trial(payload: tuple) -> dict:
    from experiments.sma8s_full100 import simulate as engine

    bank, dataset, manifest, case, trial = payload
    engine.PROTOCOL = manifest["protocol"]
    return engine.run_trial((bank, dataset, manifest, case, trial))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "case", "status"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-bank", type=Path, required=True,
                        help="Completed original b=35/40 first-order full100 bank")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=TRIALS)
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.max_trials <= TRIALS:
        parser.error("workers must be positive and max-trials must be 1–100")
    validate_dataset(args.dataset_root)
    manifest_path = args.bank / "manifest.json"
    if args.command == "prepare":
        planned = manifest_for(args.dataset_root, args.reference_bank)
        planned["workflow_sha256"] = sha256_file(Path(__file__))
        planned["contract_sha256"] = sha256_file(Path(__file__).with_name("contract.py"))
        args.bank.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != planned:
                raise ValueError("Existing bank has different inputs or code")
        else:
            atomic_json(manifest_path, planned)
        print(f"MCS b=32 pA: 75 subjects, {len(planned['cases'])} cases, "
              f"{len(planned['cases']) * TRIALS} trials; array=0-{len(planned['cases'])-1}")
        return
    manifest = json.loads(manifest_path.read_text())
    if (manifest["protocol"] != protocol() or manifest["scope"] != "mcs_only"
            or manifest["reference_manifest_sha256"] !=
            verify_reference(args.reference_bank, manifest)
            or manifest["dataset_index_sha256"] !=
            sha256_file(args.dataset_root / "index.json")
            or manifest["receptor_csv_sha256"] !=
            sha256_file(ROOT / "data/receptors/hansen_receptors_aal90.csv")
            or manifest["workflow_sha256"] != sha256_file(Path(__file__))
            or manifest["contract_sha256"] != sha256_file(Path(__file__).with_name("contract.py"))
            or any(sha256_file(ROOT / relative) != digest for relative, digest in
                   manifest["model_source_sha256"].items())):
        raise ValueError("Bank inputs or code differ from prepare; do not mix runs")
    cases = manifest["cases"]
    if args.command == "status":
        completed = sum((args.bank / "case_complete" /
                         f"case_{i:04d}.json").exists() for i in range(len(cases)))
        print(f"{completed}/{len(cases)} MCS subject×occupancy cases complete")
        return
    if args.case_index is None or not 0 <= args.case_index < len(cases):
        parser.error("case requires a valid --case-index")
    case = cases[args.case_index]
    payloads = [(str(args.bank), str(args.dataset_root), manifest, case, trial)
                for trial in range(args.max_trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trial, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == len(payloads):
                print(f"case {args.case_index}: {done}/{len(payloads)} trials", flush=True)
    if args.max_trials == TRIALS:
        marker = args.bank / "case_complete" / f"case_{args.case_index:04d}.json"
        atomic_json(marker, dict(case_index=args.case_index, case=case, n_trials=TRIALS))


if __name__ == "__main__":
    main()
