"""Score one frozen SMA trial bank with either PCI-LZ rule, without simulation.

Methods:
  hybrid: pooled region-wise prestimulus baseline; 500 Gaglioti bootstraps
          of a 100-trial mean; apply that cutoff two-sided to each trial.
  tvbsim_two_sided_min: TVBSim's 10 within-region/time shuffles, minimum
          surrogate maximum and per-trial normalisation, but two-sided.

Each trial yields one Toolkit LZ/entropy PCI. Subject scores are the arithmetic
mean of 100 values. Full trial masks, cutoffs, high-rate flags and scores are
saved. Neither method excludes high-rate trials or uses an entropy floor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tvbtoolkit-mpl-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "scripts"), str(ROOT / "src")]
from common import OCCUPANCIES, TRIALS, atomic_json, recording_path
from compare_gaglioti_supplied_binJproc import supplied_binJproc_on_windows
from gaglioti_adex_analysis import complexity

COLORS = {"CNT": "#3275a9", "EMCS": "#49a78c", "MCS": "#c58921", "UWS": "#bd5b71"}
METHODS = ("hybrid", "tvbsim_two_sided_min")


def epochs_for(bank: Path, case: dict):
    epochs, meta, labels, source = [], [], None, None
    for trial in range(TRIALS):
        path = recording_path(bank, case, trial)
        with np.load(path) as d:
            t = d["time_ms"]
            pre = (t >= -300) & (t < 0)
            post = (t >= 0) & (t < 300)
            if d["E_hz"].shape != (1000, 90) or int(pre.sum()) != 300 or int(post.sum()) != 300:
                raise ValueError(f"Incomplete peri-onset recording: {path}")
            if labels is None:
                labels, source = d["labels"].astype(str), int(d["source"])
            elif not np.array_equal(labels, d["labels"].astype(str)) or source != int(d["source"]):
                raise ValueError(f"Atlas or SMA source changed: {path}")
            x = d["E_hz"].astype(np.float64)
            epochs.append(np.concatenate((x[pre].T, x[post].T), axis=1))
        row = json.loads(path.with_suffix(".json").read_text())
        if row["case"] != case or row["trial"] != trial:
            raise ValueError(f"Trial metadata mismatch: {path}")
        meta.append(row)
    return np.stack(epochs), meta, labels, source


def hybrid_masks(epochs, seed):
    _, cuts, sd, _, _ = supplied_binJproc_on_windows(
        epochs, n0=300, nboot=500, seed=seed)
    mean = epochs[:, :, :300].mean(axis=(0, 2))
    if np.any(sd <= 0) or not np.isfinite(sd).all():
        raise ValueError("Invalid pooled baseline SD")
    z = (epochs[:, :, 300:] - mean[None, :, None]) / sd[None, :, None]
    return np.abs(z) > cuts[.01], float(cuts[.01]), np.array([])


def tvbsim_two_sided_masks(epochs):
    """Literal 10-shuffle minimum threshold, with only the final tail changed."""
    means = epochs[:, :, :300].mean(axis=2)
    if np.any(means == 0):
        raise ValueError("TVBSim prestimulus mean is zero")
    centered = epochs / means[:, :, None] - 1.0
    sd = centered[:, :, :300].std(axis=2)
    if np.any(sd <= 0) or not np.isfinite(sd).all():
        raise ValueError("Invalid per-trial TVBSim baseline SD")
    z = centered / sd[:, :, None]
    shuffled = z[:, :, :300].copy()
    rng = np.random.RandomState(0)
    maxima = np.empty(10)
    for iteration in range(10):
        for region in range(shuffled.shape[1]):
            for trial in range(shuffled.shape[0]):
                rng.shuffle(shuffled[trial, region])
        maxima[iteration] = np.max(np.abs(np.mean(shuffled, axis=0)))
    cutoff = float(np.sort(maxima)[-int(10 / 100)])  # Historical index 0.
    return np.abs(z[:, :, 300:]) > cutoff, cutoff, maxima


def score_case(payload):
    bank_str, output_str, case_index, case, method = payload
    bank, output = Path(bank_str), Path(output_str)
    case_path = output / "cases" / f"case_{case_index:04d}.json"
    mask_path = output / "masks" / f"case_{case_index:04d}.npz"
    if case_path.exists() and mask_path.exists():
        saved = json.loads(case_path.read_text())
        if saved.get("case") != case or saved.get("method") != method:
            raise ValueError(f"Output collision: {case_path}")
        return saved
    epochs, meta, labels, source = epochs_for(bank, case)
    if method == "hybrid":
        seed = int(np.random.SeedSequence([int(meta[0]["seed"]), 63001]).generate_state(1)[0])
        masks, cutoff, surrogates = hybrid_masks(epochs, seed)
    elif method == "tvbsim_two_sided_min":
        masks, cutoff, surrogates = tvbsim_two_sided_masks(epochs)
    else:
        raise ValueError(method)
    if masks.shape != (TRIALS, 90, 300):
        raise ValueError(f"Wrong mask shape: {masks.shape}")
    trials = []
    for trial, mask in enumerate(masks):
        comp = complexity(mask)
        trials.append(dict(trial=trial, seed=meta[trial]["seed"],
                           onset_ms=meta[trial]["onset_ms"],
                           model_order=meta[trial].get("model_order", 1),
                           peak_EI_hz=meta[trial]["peak_EI_hz"],
                           high_rate_above_150_hz=meta[trial]["high_rate_above_150_hz"],
                           longest_high_rate_run_ms=meta[trial]["longest_high_rate_run_ms"],
                           peak_EI_post_transient_hz=meta[trial]["peak_EI_post_transient_hz"],
                           high_rate_post_transient_steps=meta[trial]["high_rate_post_transient_steps"],
                           longest_high_rate_post_transient_run_ms=meta[trial]["longest_high_rate_post_transient_run_ms"],
                           covariance_invalid_steps_post_transient=meta[trial].get("covariance_invalid_steps_post_transient", 0),
                           covariance_invalid_steps_peri_pulse=meta[trial].get("covariance_invalid_steps_peri_pulse", 0),
                           min_covariance_eigenvalue_post_transient_khz2=meta[trial].get("min_covariance_eigenvalue_post_transient_khz2"),
                           significant_bins=int(mask.sum()), **comp))
    subject_pci = np.asarray([r["pci_lz"] for r in trials])
    row = dict(method=method, case_index=case_index, case=case,
               source_index=source, source_label=str(labels[source]),
               n_trials=TRIALS, threshold_z=cutoff,
               mean_pci_lz=float(subject_pci.mean()),
               sd_trial_pci_lz=float(subject_pci.std(ddof=1)),
               median_pci_lz=float(np.median(subject_pci)),
               mean_significant_bins=float(np.mean([r["significant_bins"] for r in trials])),
               mean_responsive_regions=float(np.mean([r["responsive_regions"] for r in trials])),
               high_rate_trials=sum(r["high_rate_above_150_hz"] for r in trials),
               high_rate_post_transient_trials=sum(r["high_rate_post_transient_steps"] > 0
                                                  for r in trials),
               covariance_invalid_post_transient_trials=sum(
                   r["covariance_invalid_steps_post_transient"] > 0 for r in trials),
               trials=trials)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temp = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, binary=masks, labels=labels,
                        source_index=source, time_ms=np.arange(300) + .5,
                        threshold_z=cutoff, surrogate_maxima=surrogates)
    temp.replace(mask_path)
    atomic_json(case_path, row)
    return row


def save_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, output: Path, method: str):
    subjects, trials = [], []
    for row in rows:
        flat = dict(method=method, **row["case"], case_index=row["case_index"],
                    source_index=row["source_index"], source_label=row["source_label"],
                    n_trials=row["n_trials"], threshold_z=row["threshold_z"],
                    mean_pci_lz=row["mean_pci_lz"], sd_trial_pci_lz=row["sd_trial_pci_lz"],
                    median_pci_lz=row["median_pci_lz"],
                    mean_significant_bins=row["mean_significant_bins"],
                    mean_responsive_regions=row["mean_responsive_regions"],
                    high_rate_trials=row["high_rate_trials"],
                    high_rate_post_transient_trials=row["high_rate_post_transient_trials"],
                    covariance_invalid_post_transient_trials=row.get("covariance_invalid_post_transient_trials", 0))
        subjects.append(flat)
        for item in row["trials"]:
            trials.append(dict(method=method, **row["case"], case_index=row["case_index"], **item))
    save_csv(output / "subject_summary.csv", subjects)
    save_csv(output / "trial_scores.csv", trials)
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    x = np.asarray(OCCUPANCIES)
    for cohort, group in (("control", "CNT"), ("emcs", "EMCS"),
                          ("mcs", "MCS"), ("uws", "UWS")):
        by_subject = {}
        for row in subjects:
            if row["cohort"] == cohort:
                by_subject.setdefault(row["subject"], {})[float(row["occupancy"])] = row["mean_pci_lz"]
        members = np.stack([np.array([v[occ] for occ in OCCUPANCIES])
                            for _, v in sorted(by_subject.items())])
        for values in members:
            ax.plot(x, values, color=COLORS[group], alpha=.13, lw=.75)
        ax.plot(x, members.mean(axis=0), color=COLORS[group], marker="o", lw=2.8,
                label=f"{group} (n={len(members)})")
    ax.set(xlabel="5-HT2A occupancy", ylabel="PCI", xlim=(-.02, .79), xticks=x)
    ax.set_xticklabels(("0", "0.25", "0.50", "0.766"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out = output / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pci_by_occupancy.pdf", bbox_inches="tight")
    fig.savefig(out / "pci_by_occupancy.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    bank_manifest = args.bank / "manifest.json"
    manifest = json.loads(bank_manifest.read_text())
    if manifest["protocol"]["trials_per_subject_occupancy"] != TRIALS:
        raise ValueError("Analysis requires the frozen 100-trial protocol")
    cases = manifest["cases"]
    missing = [i for i in range(len(cases))
               if not (args.bank / "case_complete" / f"case_{i:04d}.json").exists()]
    if missing:
        raise ValueError(f"Simulation bank incomplete: {len(missing)} cases missing; first {missing[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output / "methods.json"
    methods = dict(method=args.method,
                   simulation_bank=str(args.bank),
                   bank_manifest_sha256=hashlib.sha256(bank_manifest.read_bytes()).hexdigest(),
                   algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   trials_per_subject=TRIALS, bootstrap_draws=500 if args.method == "hybrid" else None,
                   surrogate_shuffles=10 if args.method == "tvbsim_two_sided_min" else None,
                   alpha_nominal=.01 if args.method == "hybrid" else None,
                   no_entropy_floor=True, no_high_rate_exclusion=True,
                   no_covariance_exclusion=True,
                   prestim_ms=[-300, 0], poststim_ms=[0, 300],
                   note="Hybrid applies a trial-mean null to individual trials; its nominal alpha is not a calibrated per-trial error rate."
                   if args.method == "hybrid" else
                   "TVBSim's minimum of ten surrogate maxima, changed only to a two-sided final test.")
    if meta_path.exists() and json.loads(meta_path.read_text()) != methods:
        raise ValueError("Analysis output belongs to different inputs/code")
    if not meta_path.exists():
        atomic_json(meta_path, methods)
    payloads = [(str(args.bank), str(args.output), i, case, args.method)
                for i, case in enumerate(cases)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_case, p) for p in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 20 == 0 or done == len(cases):
                print(f"{args.method}: {done}/{len(cases)} cases", flush=True)
    results.sort(key=lambda row: row["case_index"])
    aggregate(results, args.output, args.method)
    atomic_json(args.output / "completion.json", dict(method=args.method,
                cases=len(cases), trials=len(cases) * TRIALS,
                bank_manifest_sha256=methods["bank_manifest_sha256"]))


if __name__ == "__main__":
    main()
