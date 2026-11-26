"""Calculate the unchanged 100-trial hybrid PCI for the MCS b=32 bank only.

This calls the frozen full100 hybrid mask and LZ implementation directly.
It never re-simulates and never computes the historical alternative methods.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tvbtoolkit-mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments/sma8s_full100"), str(ROOT / "scripts"),
                str(ROOT / "src"), str(ROOT)]
from experiments.sma8s_full100 import analyze as original
from experiments.sma8s_mcs_b32_full100.contract import OCCUPANCIES, TRIALS, atomic_json

COLOR = "#A96643"


def aggregate(rows: list[dict], output: Path) -> None:
    subjects, trials = [], []
    for row in rows:
        case = row["case"]
        assert case["cohort"] == "mcs" and case["b_pa"] == 32.0
        subjects.append(dict(method="hybrid", **case, case_index=row["case_index"],
                             source_index=row["source_index"], source_label=row["source_label"],
                             n_trials=row["n_trials"], threshold_z=row["threshold_z"],
                             mean_pci_lz=row["mean_pci_lz"],
                             sd_trial_pci_lz=row["sd_trial_pci_lz"],
                             median_pci_lz=row["median_pci_lz"],
                             mean_significant_bins=row["mean_significant_bins"],
                             mean_responsive_regions=row["mean_responsive_regions"],
                             high_rate_trials=row["high_rate_trials"],
                             high_rate_post_transient_trials=row["high_rate_post_transient_trials"],
                             covariance_invalid_post_transient_trials=
                             row.get("covariance_invalid_post_transient_trials", 0)))
        for trial in row["trials"]:
            trials.append(dict(method="hybrid", **case, case_index=row["case_index"], **trial))
    original.save_csv(output / "subject_summary.csv", subjects)
    original.save_csv(output / "trial_scores.csv", trials)
    wide = pd.DataFrame(subjects).pivot(index="subject", columns="occupancy", values="mean_pci_lz")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for values in wide.reindex(columns=OCCUPANCIES).to_numpy(float):
        ax.plot(OCCUPANCIES, values, color=COLOR, alpha=.14, lw=.8)
    ax.plot(OCCUPANCIES, wide.reindex(columns=OCCUPANCIES).mean(axis=0),
            color=COLOR, lw=3.0, label="MCS b=32 (n=75)")
    ax.set(xlabel="5-HT2A occupancy", ylabel="PCI", xticks=OCCUPANCIES)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "mcs_pci_by_occupancy.pdf")
    fig.savefig(figure_dir / "mcs_pci_by_occupancy.png", dpi=230)
    plt.close(fig)


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
    cases = manifest["cases"]
    if (manifest["scope"] != "mcs_only" or len(cases) != 300 or
            manifest["protocol"]["b_pa_by_cohort"]["mcs"] != 32.0 or
            any(case["cohort"] != "mcs" for case in cases)):
        raise ValueError("Expected the complete MCS-only b=32 simulation bank")
    missing = [i for i in range(len(cases)) if not
               (args.bank / "case_complete" / f"case_{i:04d}.json").exists()]
    if missing:
        raise ValueError(f"Simulation bank incomplete: {len(missing)} cases; first {missing[:10]}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    methods = dict(method="hybrid", source="experiments/sma8s_full100/analyze.py",
                   simulation_bank=str(args.bank),
                   bank_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   algorithm_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                   trials_per_subject=TRIALS, bootstrap_draws=500, alpha_nominal=.01,
                   no_entropy_floor=True, no_high_rate_exclusion=True,
                   prestim_ms=[-300, 0], poststim_ms=[0, 300])
    methods_path = output / "methods.json"
    if methods_path.exists() and json.loads(methods_path.read_text()) != methods:
        raise ValueError("Analysis output belongs to different inputs or code")
    if not methods_path.exists():
        atomic_json(methods_path, methods)
    payloads = [(str(args.bank), str(output), i, case, "hybrid")
                for i, case in enumerate(cases)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(original.score_case, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 20 == 0 or done == len(cases):
                print(f"hybrid: {done}/{len(cases)} cases", flush=True)
    rows.sort(key=lambda row: row["case_index"])
    aggregate(rows, output)
    atomic_json(output / "completion.json", dict(method="hybrid", cases=len(cases),
                trials=len(cases) * TRIALS,
                bank_manifest_sha256=methods["bank_manifest_sha256"]))


if __name__ == "__main__":
    main()
