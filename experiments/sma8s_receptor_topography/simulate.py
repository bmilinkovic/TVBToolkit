"""Independent, restartable first-order MCS/UWS receptor-topography bank.

One array element runs 100 matched trials of one subject/map case. The zero-
occupancy case is rerun once per subject and shared across all high-dose map
comparisons. Original high-dose uses make_sim's production map path unchanged;
null high-dose cases replace *only* the regional gK arrays after TVB setup.
No PCI is calculated here.
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np

from experiments.sma8s_receptor_topography.common import (
    HIGHEST_OCCUPANCY, PROTOCOL, TRIALS, atomic_json, make_manifest,
    recording_path, sha256_file, trial_onset_ms, trial_seed, validate_frozen,
)
from sweep_split_leak_tep_drive import make_sim
from tvb.simulator.monitors import TemporalAverage
from tvbtoolkit.workflows.pharmacology import leak_to_conductances


def apply_receptor_map(sim, rho: np.ndarray, occupancy: float) -> None:
    """Apply the production gNa-fixed/gK-only occupancy rule to one map.

    This exactly mirrors make_sim's leak_to_conductances calculations, without
    editing the production model or re-min-max-scaling a scrambled map. In
    particular, a uniform map must remain at mean effective density rather
    than collapsing to zero after a second min-max transformation.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (90,) or not np.isfinite(rho).all() or np.any((rho < 0) | (rho > 1)):
        raise ValueError("Expected 90 finite effective receptor densities in [0,1]")
    for pop in ("e", "i"):
        baseline = PROTOCOL[f"E_L_{pop}_mv"]
        target = PROTOCOL[f"drug_E_L_{pop}_mv"]
        gk, gna = leak_to_conductances(50.0, -90.0, baseline, g_L=10.0)
        drug_gk, _ = leak_to_conductances(50.0, -90.0, target, g_Na=gna)
        expected = np.asarray(gk - occupancy * rho * (gk - drug_gk), dtype=float)
        np.testing.assert_allclose(np.asarray(getattr(sim.model, f"g_Na_{pop}")), gna)
        # TVB configures a regional trait as (region, mode), not a flat vector.
        original_shape = np.asarray(getattr(sim.model, f"g_K_{pop}")).shape
        if original_shape != (90, 1):
            raise ValueError(f"Unexpected configured g_K_{pop} shape: {original_shape}")
        expected = expected.reshape(original_shape)
        setattr(sim.model, f"g_K_{pop}", expected)
        np.testing.assert_allclose(np.asarray(getattr(sim.model, f"g_K_{pop}")), expected)


def _signature(manifest: dict, case: dict, trial: int, seed: int, onset: int) -> str:
    payload = dict(version=PROTOCOL["version"], case=case, trial=trial,
                   seed=seed, onset_ms=onset,
                   dataset_index_sha256=manifest["dataset_index_sha256"],
                   map_matrix_sha256=manifest["map_matrix_sha256"],
                   model_source_sha256=manifest["model_source_sha256"],
                   workflow_sha256=manifest["workflow_sha256"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_trial(payload: tuple) -> dict:
    bank_string, dataset_string, manifest, case, trial = payload
    bank, dataset = Path(bank_string), Path(dataset_string)
    seed = trial_seed(case["cohort"], case["subject"], trial,
                      manifest["master_seed"])
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
    with np.load(bank / "maps.npz") as saved:
        names = saved["map_ids"].astype(str).tolist()
        map_labels = saved["labels"].astype(str)
        rho = None if case["map_id"] == "baseline" else saved["rho"][names.index(case["map_id"])].copy()
    occupancy = float(case["occupancy"])
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
        receptor_occupancy=occupancy,
        drug_leak_e_mv=PROTOCOL["drug_E_L_e_mv"],
    )
    np.testing.assert_array_equal(labels.astype(str), map_labels)
    if case["map_id"] not in ("baseline", "original"):
        apply_receptor_map(sim, rho, occupancy)
    elif case["map_id"] == "original":
        # Validate that the existing production route and independent rule agree.
        expected = [np.asarray(getattr(sim.model, f"g_K_{pop}")).copy()
                    for pop in ("e", "i")]
        apply_receptor_map(sim, rho, occupancy)
        for pop, value in zip(("e", "i"), expected, strict=True):
            np.testing.assert_allclose(np.asarray(getattr(sim.model, f"g_K_{pop}")), value,
                                       rtol=0, atol=1e-10)
    gk_hashes = {pop: hashlib.sha256(np.ascontiguousarray(
        getattr(sim.model, f"g_K_{pop}")).tobytes()).hexdigest()
                 for pop in ("e", "i")}
    weight_hash = hashlib.sha256(np.ascontiguousarray(sim.connectivity.weights).tobytes()).hexdigest()
    tract_hash = hashlib.sha256(np.ascontiguousarray(sim.connectivity.tract_lengths).tobytes()).hexdigest()
    temporal = TemporalAverage(period=PROTOCOL["monitor_period_ms"],
                               variables_of_interest=np.arange(5))
    temporal.config_for_sim(sim)
    sim.monitors = tuple(sim.monitors) + (temporal,)
    times, excitatory = [], []
    peak, high_steps, longest, current = 0.0, 0, 0, 0
    post_peak, post_high, post_longest, post_current = 0.0, 0, 0, 0
    first_high_ms = None
    for samples in sim(simulation_length=PROTOCOL["simulation_ms"]):
        t_raw, raw = samples[0]
        rates = raw[:2, :, 0] * 1000.0
        if not np.isfinite(rates).all():
            raise FloatingPointError(f"Nonfinite E/I rate: {case}, trial {trial}")
        peak = max(peak, float(rates.max()))
        is_high = bool(np.any(rates > 150.0))
        high_steps += int(is_high)
        current = current + 1 if is_high else 0
        longest = max(longest, current)
        if is_high and first_high_ms is None:
            first_high_ms = float(t_raw)
        if float(t_raw) >= PROTOCOL["transient_discard_ms"]:
            post_peak = max(post_peak, float(rates.max()))
            post_high += int(is_high)
            post_current = post_current + 1 if is_high else 0
            post_longest = max(post_longest, post_current)
        if samples[-1] is not None:
            t_avg, averaged = samples[-1]
            if onset - 500 <= t_avg < onset + 500:
                times.append(float(t_avg) - onset)
                excitatory.append((averaged[0, :, 0] * 1000.0).astype(np.float32))
    time_ms = np.asarray(times, dtype=np.float64)
    e_hz = np.asarray(excitatory, dtype=np.float32)
    if e_hz.shape != (1000, 90) or np.sum((-300 <= time_ms) & (time_ms < 0)) != 300 \
            or np.sum((0 <= time_ms) & (time_ms < 300)) != 300:
        raise ValueError(f"Unexpected peri-onset recording: {case}, trial {trial}, {e_hz.shape}")
    if abs(delivered["actual_onset_ms"] - onset) > .100001 \
            or abs(delivered["delivered_duration_ms"] - PROTOCOL["pulse_duration_ms"]) > .100001:
        raise ValueError(f"Incorrect delivered pulse: {delivered}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, time_ms=time_ms, E_hz=e_hz, labels=labels,
                        source=source, seed=seed, onset_ms=onset,
                        actual_pulse_onset_ms=delivered["actual_onset_ms"],
                        delivered_pulse_duration_ms=delivered["delivered_duration_ms"],
                        peak_EI_hz=peak)
    temp.replace(target)
    row = dict(case=case, trial=trial, seed=seed, onset_ms=onset,
               signature=signature, source_index=source, source_label=str(labels[source]),
               peak_EI_hz=peak, weights_sha256=weight_hash, tracts_sha256=tract_hash,
               gK_sha256=gk_hashes, high_rate_above_150_hz=peak > 150.0,
               high_rate_raw_steps=high_steps,
               longest_high_rate_run_ms=longest * .1,
               peak_EI_post_transient_hz=post_peak,
               high_rate_post_transient_steps=post_high,
               longest_high_rate_post_transient_run_ms=post_longest * .1,
               first_high_rate_ms=first_high_ms,
               actual_pulse_onset_ms=delivered["actual_onset_ms"],
               delivered_pulse_duration_ms=delivered["delivered_duration_ms"],
               recording=str(target.relative_to(bank)))
    atomic_json(meta_path, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "case", "status"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--n-spin", type=int, default=99)
    parser.add_argument("--n-permutation", type=int, default=99)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=TRIALS,
                        help="Smoke-test prefix; never marks a case complete")
    args = parser.parse_args()
    manifest_path = args.bank / "manifest.json"
    if args.command == "prepare":
        manifest, labels, maps = make_manifest(args.dataset_root, args.n_spin,
                                               args.n_permutation)
        manifest["workflow_sha256"] = sha256_file(Path(__file__))
        manifest["common_sha256"] = sha256_file(Path(__file__).with_name("common.py"))
        args.bank.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing bank has different inputs/code; choose a new bank")
            validate_frozen(args.bank, args.dataset_root, manifest)
        else:
            temp = args.bank / "maps.tmp.npz"
            np.savez_compressed(temp, map_ids=np.asarray(manifest["map_ids"]),
                                labels=labels, rho=maps)
            temp.replace(args.bank / "maps.npz")
            atomic_json(manifest_path, manifest)
        print(f"{sum(map(len, manifest['subjects'].values()))} subjects; "
              f"{len(manifest['map_ids'])} high-dose maps; "
              f"{len(manifest['cases'])} cases; "
              f"{len(manifest['cases']) * TRIALS} simulations; "
              f"array=0-{len(manifest['cases']) - 1}")
        return
    manifest = json.loads(manifest_path.read_text())
    validate_frozen(args.bank, args.dataset_root, manifest)
    if sha256_file(Path(__file__)) != manifest["workflow_sha256"] \
            or sha256_file(Path(__file__).with_name("common.py")) != manifest["common_sha256"]:
        raise ValueError("Workflow code changed since prepare")
    if args.command == "status":
        complete = sum((args.bank / "case_complete" / f"case_{i:05d}.json").exists()
                       for i in range(len(manifest["cases"])))
        print(f"{complete}/{len(manifest['cases'])} cases complete")
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
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == len(payloads):
                print(f"case {args.case_index}: {done}/{len(payloads)} trials", flush=True)
    if args.max_trials == TRIALS:
        atomic_json(args.bank / "case_complete" / f"case_{args.case_index:05d}.json",
                    dict(case_index=args.case_index, case=case, n_trials=TRIALS))


if __name__ == "__main__":
    main()
