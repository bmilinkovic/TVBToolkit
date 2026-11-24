"""Run the full-cohort MCS/UWS b=27/42-pA bank without changing engines.

One array task runs 100 trials for one subject and one 5-HT2A occupancy.
The first- and second-order engines are the frozen SMA full100 implementations.
Only MCS/UWS b change from 35/40 to 27/42 pA; CNT/EMCS remain 5/15 pA.
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.sma8s_b27_full100.common import (  # noqa: E402
    TRIALS, atomic_json, manifest_for, protocol_for, sha256_file,
    validate_dataset,
)


def run_trial(payload: tuple) -> dict:
    """Use the unchanged order-specific engine with this bank's frozen protocol."""
    order, bank, dataset, manifest, case, trial = payload
    if order == 1:
        from experiments.sma8s_full100 import simulate as engine
    elif order == 2:
        from experiments.sma8s_full100_second_order import simulate as engine
    else:
        raise ValueError(order)
    engine.PROTOCOL = manifest["protocol"]
    return engine.run_trial((bank, dataset, manifest, case, trial))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "case", "status"))
    parser.add_argument("--order", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--scope", choices=("all_noncoma", "selected40"),
                        default="all_noncoma")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=TRIALS,
                        help="Smoke-test prefix; never marks a case complete")
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.max_trials <= TRIALS:
        parser.error("workers and max-trials must be positive and within 100 trials")
    validate_dataset(args.dataset_root)
    manifest_path = args.bank / "manifest.json"
    if args.command == "prepare":
        planned = manifest_for(args.dataset_root, args.scope, args.order)
        planned["workflow_sha256"] = sha256_file(Path(__file__))
        planned["common_sha256"] = sha256_file(Path(__file__).with_name("common.py"))
        args.bank.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != planned:
                raise ValueError("Existing bank has different inputs/code; choose a new bank")
        else:
            atomic_json(manifest_path, planned)
        subjects = sum(len(ids) for ids in planned["subjects"].values())
        cases = len(planned["cases"])
        print(f"order={args.order}; {subjects} subjects; {cases} "
              f"subject×occupancy cases; {cases * TRIALS} simulations; "
              f"array=0-{cases - 1}")
        return

    manifest = json.loads(manifest_path.read_text())
    if (manifest["protocol"] != protocol_for(args.order)
            or manifest["scope"] != args.scope
            or manifest["dataset_index_sha256"] !=
            sha256_file(args.dataset_root / "index.json")
            or manifest["receptor_csv_sha256"] !=
            sha256_file(ROOT / "data/receptors/hansen_receptors_aal90.csv")
            or manifest["workflow_sha256"] != sha256_file(Path(__file__))
            or manifest["common_sha256"] !=
            sha256_file(Path(__file__).with_name("common.py"))
            or any(sha256_file(ROOT / path) != expected for path, expected in
                   manifest["model_source_sha256"].items())):
        raise ValueError("Bank inputs/code differ from prepare; do not mix runs")
    cases = manifest["cases"]
    if args.command == "status":
        complete = sum((args.bank / "case_complete" /
                        f"case_{i:04d}.json").exists() for i in range(len(cases)))
        print(f"{complete}/{len(cases)} subject×occupancy cases complete")
        return
    if args.case_index is None or not 0 <= args.case_index < len(cases):
        parser.error("case requires a valid --case-index")
    case = cases[args.case_index]
    payloads = [(args.order, str(args.bank), str(args.dataset_root),
                 manifest, case, trial) for trial in range(args.max_trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trial, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done % 10 == 0 or done == len(payloads):
                print(f"case {args.case_index}: {done}/{len(payloads)} "
                      "trials complete", flush=True)
    if args.max_trials == TRIALS:
        marker = args.bank / "case_complete" / f"case_{args.case_index:04d}.json"
        atomic_json(marker, dict(case_index=args.case_index, case=case,
                                 n_trials=TRIALS, model_order=args.order))


if __name__ == "__main__":
    main()
