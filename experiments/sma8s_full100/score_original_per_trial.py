"""Score saved 100-trial SMA banks with the original per-trial PCI-LZ rule.

This reproduces the *per-trial Gaglioti-style* calculation used by
``scripts/gaglioti_sma8s_el_occupancy.py``. It is not the author's literal
multi-trial PCI, which binarises the trial-averaged evoked response.

For each trial independently, calculate each region's baseline mean and
sample SD over [-300, 0) ms. Build the two-sided alpha=.01 cutoff using the
author's 500-bootstrap, spatial-maximum, time-pooled procedure with one
prestimulus trial. Apply it to that SAME trial's [0, 300) ms response, then
compute one PCI-LZ. A subject's PCI is the mean of its 100 trial PCI values.

No model is simulated, no trial is excluded, and no entropy floor is used.
The numerical masks and per-trial thresholds are saved separately from the
existing hybrid, TVBSim, and Dalla Porta analyses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import aggregate, epochs_for
from common import TRIALS, atomic_json
from gaglioti_adex_analysis import complexity

METHOD = "gaglioti_per_trial"
NBOOT = 500
NBASE = 300
ALPHA = 0.01


def original_trial_mask(epoch: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Exact one-trial specialization of supplied_binJproc_on_windows.

    With one trial, each bootstrap simply resamples prestimulus time points.
    Cache the spatial maximum at every original baseline time to avoid
    repeatedly materialising the same 90-region values. Keep the original
    RandomState call order, including its one-trial-ID draw.
    """
    if epoch.shape != (90, 2 * NBASE):
        raise ValueError(f"Expected 90 x 600 trial window, got {epoch.shape}")
    pre = epoch[:, :NBASE]
    mean = pre.mean(axis=1)
    sd = pre.std(axis=1, ddof=1)
    if np.any(sd <= 0) or not np.all(np.isfinite(sd)):
        raise ValueError("Invalid per-trial prestimulus SD")
    maxima_by_time = np.max(np.abs((pre - mean[:, None]) / sd[:, None]), axis=0)
    rng = np.random.RandomState(seed)
    surrogate_maxima = np.empty((NBOOT, NBASE), dtype=np.float64)
    for boot in range(NBOOT):
        rng.randint(0, 1, 1)  # Supplied routine draws the sole trial ID.
        times = rng.randint(0, NBASE, (1, NBASE))
        surrogate_maxima[boot] = maxima_by_time[times[0]]
    ordered = np.sort(surrogate_maxima.ravel())
    index = int(np.round((1 - ALPHA) * NBOOT * NBASE)) - 1
    threshold_z = float(ordered[index])
    mask = np.abs(epoch[:, NBASE:] - mean[:, None]) > sd[:, None] * threshold_z
    return mask, threshold_z


def score_case(payload: tuple[str, str, int, dict]) -> dict:
    bank_string, output_string, case_index, case = payload
    bank, output = Path(bank_string), Path(output_string)
    case_path = output / "cases" / f"case_{case_index:04d}.json"
    mask_path = output / "masks" / f"case_{case_index:04d}.npz"
    if case_path.exists() and mask_path.exists():
        saved = json.loads(case_path.read_text())
        if saved.get("case") != case or saved.get("method") != METHOD:
            raise ValueError(f"Output collision: {case_path}")
        with np.load(mask_path) as data:
            if data["binary"].shape != (TRIALS, 90, NBASE):
                raise ValueError(f"Incomplete saved masks: {mask_path}")
        return saved

    epochs, meta, labels, source = epochs_for(bank, case)
    masks = np.empty((TRIALS, 90, NBASE), dtype=bool)
    thresholds = np.empty(TRIALS, dtype=np.float64)
    trial_rows = []
    for trial, (epoch, info) in enumerate(zip(epochs, meta)):
        seed = int(np.random.SeedSequence([int(info["seed"]), 55008]).generate_state(1)[0])
        mask, cutoff = original_trial_mask(epoch, seed)
        masks[trial], thresholds[trial] = mask, cutoff
        score = complexity(mask)
        trial_rows.append(dict(
            trial=trial, seed=info["seed"], bootstrap_seed=seed,
            onset_ms=info["onset_ms"], model_order=info.get("model_order", 1),
            threshold_z=cutoff, peak_EI_hz=info["peak_EI_hz"],
            high_rate_above_150_hz=info["high_rate_above_150_hz"],
            longest_high_rate_run_ms=info["longest_high_rate_run_ms"],
            peak_EI_post_transient_hz=info["peak_EI_post_transient_hz"],
            high_rate_post_transient_steps=info["high_rate_post_transient_steps"],
            longest_high_rate_post_transient_run_ms=info["longest_high_rate_post_transient_run_ms"],
            covariance_invalid_steps_post_transient=info.get("covariance_invalid_steps_post_transient", 0),
            covariance_invalid_steps_peri_pulse=info.get("covariance_invalid_steps_peri_pulse", 0),
            min_covariance_eigenvalue_post_transient_khz2=info.get(
                "min_covariance_eigenvalue_post_transient_khz2"),
            significant_bins=int(mask.sum()), **score))

    pci = np.asarray([row["pci_lz"] for row in trial_rows])
    result = dict(
        method=METHOD, case_index=case_index, case=case,
        source_index=source, source_label=str(labels[source]), n_trials=TRIALS,
        threshold_z=None,  # There is no one cutoff shared across trials.
        mean_threshold_z=float(thresholds.mean()),
        sd_threshold_z=float(thresholds.std(ddof=1)),
        mean_pci_lz=float(pci.mean()), sd_trial_pci_lz=float(pci.std(ddof=1)),
        median_pci_lz=float(np.median(pci)),
        mean_significant_bins=float(np.mean([row["significant_bins"] for row in trial_rows])),
        mean_responsive_regions=float(np.mean([row["responsive_regions"] for row in trial_rows])),
        high_rate_trials=sum(row["high_rate_above_150_hz"] for row in trial_rows),
        high_rate_post_transient_trials=sum(row["high_rate_post_transient_steps"] > 0
                                            for row in trial_rows),
        covariance_invalid_post_transient_trials=sum(
            row["covariance_invalid_steps_post_transient"] > 0 for row in trial_rows),
        trials=trial_rows)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, binary=masks, labels=labels,
                        source_index=source, time_ms=np.arange(NBASE) + .5,
                        threshold_z_by_trial=thresholds)
    temporary.replace(mask_path)
    atomic_json(case_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest_path = args.bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol"]["trials_per_subject_occupancy"] != TRIALS:
        raise ValueError("Expected the frozen 100-trial SMA simulation bank")
    cases = manifest["cases"]
    missing = [index for index in range(len(cases))
               if not (args.bank / "case_complete" / f"case_{index:04d}.json").exists()]
    if missing:
        raise ValueError(f"Simulation bank incomplete: {len(missing)} cases missing; first {missing[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    methods_path = args.output / "methods.json"
    methods = dict(
        method=METHOD, simulation_bank=str(args.bank),
        bank_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trials_per_subject=TRIALS, baseline_and_cutoff="independent per trial",
        bootstrap_draws=NBOOT, alpha_nominal=ALPHA, two_sided=True,
        spatial_maximum=True, prestim_ms=[-300, 0], poststim_ms=[0, 300],
        pci_aggregation="one mask and PCI-LZ per trial; arithmetic mean of 100 PCI values",
        no_entropy_floor=True, no_high_rate_exclusion=True, no_covariance_exclusion=True,
        warning="Simulation-specific per-trial Gaglioti-style adaptation, not the author's literal trial-averaged empirical PCI")
    if methods_path.exists() and json.loads(methods_path.read_text()) != methods:
        raise ValueError("Analysis output belongs to different inputs/code")
    if not methods_path.exists():
        atomic_json(methods_path, methods)
    payloads = [(str(args.bank), str(args.output), index, case)
                for index, case in enumerate(cases)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_case, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 20 == 0 or done == len(cases):
                print(f"{METHOD}: {done}/{len(cases)} cases", flush=True)
    results.sort(key=lambda row: row["case_index"])
    aggregate(results, args.output, METHOD)
    atomic_json(args.output / "completion.json", dict(
        method=METHOD, cases=len(cases), trials=len(cases) * TRIALS,
        bank_manifest_sha256=methods["bank_manifest_sha256"]))


if __name__ == "__main__":
    main()
