"""Apply the *same* Dalla Porta trial-mean PCI rule to either SMA full100 bank.

This adds a third downstream PCI rule to the first-/second-order SMA banks;
no neural simulation is rerun. It uses all 100 trials per subject/occupancy,
the same 300-ms pre/post windows, 1,000 bootstraps and LZ76 normalization as
the full-cohort Dalla Porta analysis. This permits within-bank method checks.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tvb-della-on-sma-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiments/della_porta_second_order"
sys.path[:0] = [str(PILOT), str(ROOT / "src"), str(ROOT)]

from experiments.della_porta_second_order import analyze as dalla
from experiments.della_porta_second_order import study as local
from experiments.sma8s_full100.common import recording_path

TRIALS = 100
COLORS = {"CNT": "#3275a9", "EMCS": "#49a78c", "MCS": "#c58921", "UWS": "#bd5b71"}


def score_case(payload: tuple) -> dict:
    bank_str, out_str, index, case = payload
    bank, out = Path(bank_str), Path(out_str)
    result_path = out / "cases" / f"case_{index:04d}.json"
    mask_path = out / "masks" / f"case_{index:04d}.npz"
    if result_path.exists() and mask_path.exists():
        row = json.loads(result_path.read_text())
        if row["case"] != case:
            raise ValueError(f"Output collision: {result_path}")
        return row
    epochs, metadata, labels, source = [], [], None, None
    for trial in range(TRIALS):
        path = recording_path(bank, case, trial)
        with np.load(path) as d:
            time = d["time_ms"]
            pre = (time >= -300) & (time < 0)
            post = (time >= 0) & (time < 300)
            if d["E_hz"].shape != (1000, 90) or pre.sum() != 300 or post.sum() != 300:
                raise ValueError(f"Incomplete or misaligned trial: {path}")
            current_labels, current_source = d["labels"].astype(str), int(d["source"])
            if labels is None:
                labels, source = current_labels, current_source
            elif not np.array_equal(labels, current_labels) or source != current_source:
                raise ValueError(f"Atlas or source changed: {path}")
            rates = d["E_hz"].astype(np.float64)
            epochs.append(np.concatenate((rates[pre].T, rates[post].T), axis=1))
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["case"] != case or meta["trial"] != trial:
            raise ValueError(f"Trial metadata mismatch: {path}")
        metadata.append(meta)
    array = np.stack(epochs)
    seed = int(np.random.SeedSequence([metadata[0]["seed"], 7611]).generate_state(1)[0])
    mask, cutoff, response, baseline_sd = dalla.notebook_binJproc(
        array, seed=seed, nboot=1000, alpha=.01)
    score = dalla.pci_from_mask(mask)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temp = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, binary=mask, response_hz=response.astype(np.float32),
                        baseline_sd_hz=baseline_sd.astype(np.float32),
                        labels=labels, source=source, time_ms=np.arange(300),
                        threshold_z=cutoff)
    temp.replace(mask_path)
    row = dict(case_index=index, case=case, trials=TRIALS,
               source_index=source, source_label=str(labels[source]),
               pci=float(score["pci"]) if np.isfinite(score["pci"]) else None,
               status=score["status"],
               significant_bins=int(mask.sum()),
               responsive_regions=score["responsive_regions"],
               threshold_z=cutoff,
               high_rate_trials=sum(m["high_rate_post_transient_steps"] > 0
                                    for m in metadata),
               covariance_invalid_trials=sum(
                   m.get("covariance_invalid_steps_post_transient", 0) > 0
                   for m in metadata))
    local.write_json(result_path, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest_path = args.bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["protocol"].get("model_order") not in (1, 2)
            or manifest["protocol"].get("trials_per_subject_occupancy") != TRIALS
            or len(manifest["cases"]) != 716):
        raise ValueError("Expected an SMA first-/second-order full100 bank")
    missing = [i for i in range(len(manifest["cases"]))
               if not (args.bank / "case_complete" / f"case_{i:04d}.json").exists()]
    if missing:
        raise ValueError(f"Simulation bank incomplete: {len(missing)} cases; first {missing[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    methods = dict(method="Dalla Porta trial-mean binJproc + LZ76 on SMA bank",
                   simulation_bank=str(args.bank),
                   bank_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_sha256={str(path.relative_to(ROOT)): local.sha256(path)
                                  for path in (PILOT / "analyze.py",
                                               ROOT / "src/tvbtoolkit/complexity/pci_casali.py")},
                   trials=TRIALS, bootstrap_draws=1000, alpha=.01,
                   baseline_ms=[-300, 0], response_ms=[0, 300],
                   no_high_rate_or_covariance_exclusion=True)
    methods_path = args.output / "methods.json"
    if methods_path.exists() and json.loads(methods_path.read_text()) != methods:
        raise ValueError("Analysis output belongs to different inputs/code")
    if not methods_path.exists():
        local.write_json(methods_path, methods)
    payloads = [(str(args.bank), str(args.output), i, case)
                for i, case in enumerate(manifest["cases"])]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_case, job) for job in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 20 == 0 or done == len(payloads):
                print(f"{done}/{len(payloads)} cases scored", flush=True)
    rows.sort(key=lambda row: row["case_index"])
    flat = [dict(case_index=r["case_index"], **r["case"],
                 trials=r["trials"], source_index=r["source_index"],
                 source_label=r["source_label"], pci=r["pci"],
                 status=r["status"], significant_bins=r["significant_bins"],
                 responsive_regions=r["responsive_regions"],
                 threshold_z=r["threshold_z"],
                 high_rate_trials=r["high_rate_trials"],
                 covariance_invalid_trials=r["covariance_invalid_trials"])
            for r in rows]
    with (args.output / "subject_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    for group, color in COLORS.items():
        by_subject = {}
        for row in flat:
            if row["group"] == group:
                by_subject.setdefault(row["subject"], {})[row["occupancy"]] = row["pci"]
        lines = np.asarray([[value[occ] if value[occ] is not None else np.nan
                             for occ in local.OCCUPANCIES]
                            for _, value in sorted(by_subject.items())], dtype=float)
        for values in lines:
            ax.plot(local.OCCUPANCIES, values, color=color, alpha=.13, lw=.7)
        ax.plot(local.OCCUPANCIES, np.nanmean(lines, axis=0), "o-",
                color=color, lw=2.5, label=f"{group} (n={len(lines)})")
    ax.set(xlabel="5-HT2A occupancy", ylabel="PCI", xticks=local.OCCUPANCIES,
           xlim=(-.02, .79))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "pci_by_occupancy.pdf")
    fig.savefig(figures / "pci_by_occupancy.png", dpi=220)
    plt.close(fig)
    local.write_json(args.output / "completion.json", dict(cases=len(rows),
                     simulations=len(rows)*TRIALS,
                     bank_manifest_sha256=methods["bank_manifest_sha256"]))


if __name__ == "__main__":
    main()
