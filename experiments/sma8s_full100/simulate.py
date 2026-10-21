"""Build one reusable, PCI-free 100-trial SMA recording bank on SLURM.

`prepare` freezes the subject/case manifest. One SLURM array element then
runs all 100 stochastic trials for a subject × occupancy case. All four
occupancies use the same seed and randomized onset for a given subject/trial.
Only the 1-s peri-onset 1-ms TemporalAverage E trace is saved; the 8-s model
is run continuously and peak/high-rate diagnostics cover the full 8 s.

The mean-field order is deliberately fixed to first-order. A separately
reviewed second-order implementation can be added as a new bank/version.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TVB_USER_HOME", str(ROOT / ".tvb-temp"))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import numpy as np

from common import (PROTOCOL, TRIALS, atomic_json, manifest_for, recording_path,
                    sha256_file, trial_onset_ms, trial_seed, validate_dataset)
from sweep_split_leak_tep_drive import make_sim
from tvb.simulator.monitors import TemporalAverage


def _signature(manifest: dict, case: dict, trial: int, seed: int, onset: int) -> str:
    frozen = dict(protocol=manifest["protocol"], case=case, trial=trial, seed=seed,
                  onset_ms=onset, dataset_index_sha256=manifest["dataset_index_sha256"],
                  receptor_csv_sha256=manifest["receptor_csv_sha256"],
                  model_source_sha256=manifest["model_source_sha256"])
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def run_trial(payload: tuple) -> dict:
    bank_str, dataset_str, manifest, case, trial = payload
    bank, dataset = Path(bank_str), Path(dataset_str)
    seed = trial_seed(case["cohort"], case["subject"], trial, manifest["master_seed"])
    onset = trial_onset_ms(seed)
    target = recording_path(bank, case, trial)
    meta_path = target.with_suffix(".json")
    signature = _signature(manifest, case, trial, seed, onset)
    if target.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("signature") != signature:
            raise ValueError(f"Existing trial has different parameters: {target}")
        with np.load(target) as saved:
            if saved["E_hz"].shape != (1000, 90) or int(saved["seed"]) != seed:
                raise ValueError(f"Incomplete existing trial: {target}")
        return old

    sim, labels, source, delivered = make_sim(
        1, PROTOCOL["G"], PROTOCOL["external_drive_hz"],
        PROTOCOL["pulse_hz_per_ms"], dataset,
        duration=PROTOCOL["simulation_ms"], seed=seed, onset_ms=onset,
        source_label=PROTOCOL["source_label"],
        pulse_ms=PROTOCOL["pulse_duration_ms"], column_normalize=False,
        b_map_pa=np.full(90, case["b_pa"]),
        weight_noise=PROTOCOL["weight_noise"],
        tau_ou_ms=PROTOCOL["regional_OU_tau_ms"],
        leak_e_mv=PROTOCOL["E_L_e_mv"],
        cohort=case["cohort"], subject=case["subject"],
        receptor_occupancy=case["occupancy"],
        drug_leak_e_mv=PROTOCOL["drug_E_L_e_mv"],
    )
    weights_sha256 = hashlib.sha256(
        np.ascontiguousarray(sim.connectivity.weights).tobytes()).hexdigest()
    tracts_sha256 = hashlib.sha256(
        np.ascontiguousarray(sim.connectivity.tract_lengths).tobytes()).hexdigest()
    temporal = TemporalAverage(period=PROTOCOL["monitor_period_ms"],
                               variables_of_interest=np.arange(5))
    temporal.config_for_sim(sim)
    sim.monitors = tuple(sim.monitors) + (temporal,)
    times, e_rates = [], []
    peak_hz, high_steps, longest_high_steps, current_high_steps = 0.0, 0, 0, 0
    peak_post_transient_hz, high_post_steps = 0.0, 0
    longest_high_post_steps, current_high_post_steps = 0, 0
    first_high_ms = None
    for samples in sim(simulation_length=PROTOCOL["simulation_ms"]):
        t_raw, state_raw = samples[0]
        rates = state_raw[:2, :, 0] * 1000.0
        if not np.isfinite(rates).all():
            raise FloatingPointError(f"Nonfinite E/I rate: {case}, trial {trial}")
        peak_hz = max(peak_hz, float(rates.max()))
        is_high = bool(np.any(rates > 150.0))
        high_steps += int(is_high)
        current_high_steps = current_high_steps + 1 if is_high else 0
        longest_high_steps = max(longest_high_steps, current_high_steps)
        if is_high and first_high_ms is None:
            first_high_ms = float(t_raw)
        if float(t_raw) >= PROTOCOL["transient_discard_ms"]:
            peak_post_transient_hz = max(peak_post_transient_hz, float(rates.max()))
            high_post_steps += int(is_high)
            current_high_post_steps = current_high_post_steps + 1 if is_high else 0
            longest_high_post_steps = max(longest_high_post_steps, current_high_post_steps)
        if samples[-1] is not None:
            t_avg, averaged = samples[-1]
            if onset - 500 <= t_avg < onset + 500:
                times.append(float(t_avg) - onset)
                e_rates.append((averaged[0, :, 0] * 1000.0).astype(np.float32))
    time_ms = np.asarray(times, dtype=np.float64)
    e_hz = np.asarray(e_rates, dtype=np.float32)
    if e_hz.shape != (1000, 90) or np.sum((-300 <= time_ms) & (time_ms < 0)) != 300 \
            or np.sum((0 <= time_ms) & (time_ms < 300)) != 300:
        raise ValueError(f"Unexpected 1-ms peri-onset recording: {case}, trial {trial}, {e_hz.shape}")
    if abs(delivered["actual_onset_ms"] - onset) > .100001 \
            or abs(delivered["delivered_duration_ms"] - 10.0) > .100001:
        raise ValueError(f"Incorrect delivered pulse: {delivered}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, time_ms=time_ms, E_hz=e_hz, labels=labels,
                        source=source, seed=seed, onset_ms=onset,
                        actual_pulse_onset_ms=delivered["actual_onset_ms"],
                        delivered_pulse_duration_ms=delivered["delivered_duration_ms"],
                        peak_EI_hz=peak_hz)
    temp.replace(target)
    row = dict(case=case, trial=trial, seed=seed, onset_ms=onset,
               signature=signature, source_index=source,
               source_label=str(labels[source]), peak_EI_hz=peak_hz,
               weights_sha256=weights_sha256, tracts_sha256=tracts_sha256,
               high_rate_above_150_hz=peak_hz > 150.0,
               high_rate_raw_steps=high_steps,
               longest_high_rate_run_ms=longest_high_steps * .1,
               peak_EI_post_transient_hz=peak_post_transient_hz,
               high_rate_post_transient_steps=high_post_steps,
               longest_high_rate_post_transient_run_ms=longest_high_post_steps * .1,
               first_high_rate_ms=first_high_ms,
               actual_pulse_onset_ms=delivered["actual_onset_ms"],
               delivered_pulse_duration_ms=delivered["delivered_duration_ms"],
               recording=str(target.relative_to(bank)))
    atomic_json(meta_path, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "case", "status"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--scope", choices=("all_noncoma", "selected40"), default="all_noncoma")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=TRIALS,
                        help="Smoke-test prefix; never marks a case complete")
    args = parser.parse_args()
    validate_dataset(args.dataset_root)
    manifest_path = args.bank / "manifest.json"
    if args.command == "prepare":
        planned = manifest_for(args.dataset_root, args.scope)
        planned["workflow_sha256"] = sha256_file(Path(__file__))
        planned["common_sha256"] = sha256_file(Path(__file__).with_name("common.py"))
        args.bank.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != planned:
                raise ValueError("Existing bank has different inputs/code; choose a new bank")
        else:
            atomic_json(manifest_path, planned)
        print(f"{len(planned['subjects']['control']) + len(planned['subjects']['emcs']) + len(planned['subjects']['mcs']) + len(planned['subjects']['uws'])} subjects; "
              f"{len(planned['cases'])} subject×occupancy cases; "
              f"{len(planned['cases']) * TRIALS} simulations; array=0-{len(planned['cases'])-1}")
        return
    manifest = json.loads(manifest_path.read_text())
    model_source_changed = any(
        sha256_file(ROOT / relative) != expected
        for relative, expected in manifest["model_source_sha256"].items())
    if manifest["protocol"] != PROTOCOL or manifest["dataset_index_sha256"] != sha256_file(args.dataset_root / "index.json") \
            or manifest["receptor_csv_sha256"] != sha256_file(ROOT / "data/receptors/hansen_receptors_aal90.csv") \
            or model_source_changed \
            or manifest["workflow_sha256"] != sha256_file(Path(__file__)) \
            or manifest["common_sha256"] != sha256_file(Path(__file__).with_name("common.py")):
        raise ValueError("Inputs/code changed since prepare; do not mix simulations")
    if args.command == "status":
        complete = sum((args.bank / "case_complete" / f"case_{i:04d}.json").exists()
                       for i in range(len(manifest["cases"])))
        print(f"{complete}/{len(manifest['cases'])} subject×occupancy cases complete")
        return
    if args.case_index is None or not 0 <= args.case_index < len(manifest["cases"]):
        parser.error("case requires a valid --case-index")
    if not 1 <= args.max_trials <= TRIALS or args.workers < 1:
        parser.error("invalid --max-trials or --workers")
    case = manifest["cases"][args.case_index]
    payloads = [(str(args.bank), str(args.dataset_root), manifest, case, trial)
                for trial in range(args.max_trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trial, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done % 10 == 0 or done == len(payloads):
                print(f"case {args.case_index}: {done}/{len(payloads)} trials complete", flush=True)
    if args.max_trials == TRIALS:
        marker = args.bank / "case_complete" / f"case_{args.case_index:04d}.json"
        atomic_json(marker, dict(case_index=args.case_index, case=case, n_trials=TRIALS))


if __name__ == "__main__":
    main()
