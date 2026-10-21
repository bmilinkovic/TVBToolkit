"""Score the full Dalla Porta recording bank with three separate PCI-LZ rules.

All methods use the same 50 time-locked trials, [-300,0) ms baseline and
[0,300) ms response. Dalla Porta first averages poststimulus trials; hybrid
and TVBSim-minimum score each poststimulus trial and average the 50 PCI values.
No high-rate or covariance-invalid trial is removed.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tvb-della-full-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiments/della_porta_second_order"
FIRST = ROOT / "experiments/sma8s_full100"
sys.path[:0] = [str(PILOT), str(FIRST), str(ROOT / "scripts"),
                str(ROOT / "src"), str(ROOT)]

from experiments.della_porta_second_order import analyze as notebook
from experiments.della_porta_second_order import study as local
from experiments.della_porta_full_cohort.simulate import PROTOCOL as FULL_PROTOCOL
from experiments.sma8s_full100 import analyze as sma

METHODS = ("dalla_porta_trial_mean", "hybrid_per_trial", "tvbsim_two_sided_min")
COLORS = {"CNT": "#3275a9", "EMCS": "#49a78c", "MCS": "#c58921", "UWS": "#bd5b71"}


def _safe(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def score_case(payload: tuple) -> dict:
    bank_str, output_str, case_index, case = payload
    bank, output = Path(bank_str), Path(output_str)
    result_path = output / "cases" / f"case_{case_index:04d}.json"
    masks_path = output / "masks" / f"case_{case_index:04d}.npz"
    if result_path.exists() and masks_path.exists():
        row = json.loads(result_path.read_text())
        if row["case"] != case or row["case_index"] != case_index:
            raise ValueError(f"Output collision: {result_path}")
        return row
    epochs, metadata, labels, source = notebook.load_case(bank, case_index)
    if epochs.shape != (local.TRIALS, 90, 600):
        raise ValueError(f"Incorrect trial bank shape: {epochs.shape}")
    if any(m["case"] != case or m["trial"] != trial
           for trial, m in enumerate(metadata)):
        raise ValueError(f"Trial metadata does not match case {case_index}")
    seed = int(np.random.SeedSequence([metadata[0]["seed"], 7611]).generate_state(1)[0])

    # Literal supplied-notebook binarisation with the pilot's explicit N0=300
    # correction and common 300-ms poststimulus window.
    dalla_mask, dalla_cut, response_hz, baseline_sd = notebook.notebook_binJproc(
        epochs, seed=seed, nboot=1000, alpha=.01)
    dalla_score = notebook.pci_from_mask(dalla_mask)

    hybrid_seed = int(np.random.SeedSequence([metadata[0]["seed"], 63001]).generate_state(1)[0])
    hybrid_masks, hybrid_cut, _ = sma.hybrid_masks(epochs, hybrid_seed)
    tvbsim_masks, tvbsim_cut, surrogate_maxima = sma.tvbsim_two_sided_masks(epochs)
    trial_results = []
    for method, masks in (("hybrid_per_trial", hybrid_masks),
                          ("tvbsim_two_sided_min", tvbsim_masks)):
        for trial, mask in enumerate(masks):
            score = sma.complexity(mask)
            trial_results.append(dict(method=method, trial=trial,
                                      seed=metadata[trial]["seed"],
                                      onset_ms=metadata[trial]["onset_ms"],
                                      high_rate=metadata[trial]["high_rate_post_transient_steps"] > 0,
                                      covariance_invalid=metadata[trial]["covariance_invalid_1ms_samples"] > 0,
                                      significant_bins=int(mask.sum()), **score))
    by_method = {method: [r["pci_lz"] for r in trial_results if r["method"] == method]
                 for method in METHODS[1:]}
    masks_path.parent.mkdir(parents=True, exist_ok=True)
    temp = masks_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, dalla_porta=dalla_mask,
                        hybrid=hybrid_masks, tvbsim=tvbsim_masks,
                        response_hz=response_hz.astype(np.float32),
                        baseline_sd_hz=baseline_sd.astype(np.float32),
                        labels=labels, source=source,
                        time_ms=np.arange(300),
                        dalla_cutoff_z=dalla_cut,
                        hybrid_cutoff_z=hybrid_cut,
                        tvbsim_cutoff_z=tvbsim_cut,
                        tvbsim_surrogate_maxima=surrogate_maxima)
    temp.replace(masks_path)
    row = dict(case_index=case_index, case=case, trials=local.TRIALS,
               source_index=source, source_label=str(labels[source]),
               high_rate_trials=sum(m["high_rate_post_transient_steps"] > 0 for m in metadata),
               covariance_invalid_trials=sum(m["covariance_invalid_1ms_samples"] > 0 for m in metadata),
               dalla_porta_pci=_safe(dalla_score["pci"]),
               dalla_porta_status=dalla_score["status"],
               dalla_porta_significant_bins=int(dalla_mask.sum()),
               dalla_porta_responsive_regions=dalla_score["responsive_regions"],
               dalla_porta_cutoff_z=dalla_cut,
               hybrid_pci_mean=float(np.mean(by_method["hybrid_per_trial"])),
               hybrid_pci_sd=float(np.std(by_method["hybrid_per_trial"], ddof=1)),
               hybrid_cutoff_z=hybrid_cut,
               tvbsim_pci_mean=float(np.mean(by_method["tvbsim_two_sided_min"])),
               tvbsim_pci_sd=float(np.std(by_method["tvbsim_two_sided_min"], ddof=1)),
               tvbsim_cutoff_z=tvbsim_cut,
               per_trial=trial_results)
    local.write_json(result_path, row)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    measures = (("dalla_porta_pci", "Dalla Porta: trial-mean PCI-LZ76"),
                ("hybrid_pci_mean", "Hybrid: mean single-trial PCI-LZ"),
                ("tvbsim_pci_mean", "TVBSim two-sided minimum: mean PCI-LZ"))
    for leak in local.LEAKS:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), sharex=True,
                                 layout="constrained")
        for ax, (field, title) in zip(axes, measures):
            for group, color in COLORS.items():
                members = [r for r in rows if r["case"]["group"] == group
                           and r["case"]["leak_mv"] == leak]
                by_subject = {}
                for r in members:
                    by_subject.setdefault(r["case"]["subject"], {})[r["case"]["occupancy"]] = r[field]
                lines = np.asarray([[values[occ] if values[occ] is not None else np.nan
                                     for occ in local.OCCUPANCIES]
                                    for _, values in sorted(by_subject.items())], dtype=float)
                for values in lines:
                    ax.plot(local.OCCUPANCIES, values, color=color, lw=.7, alpha=.13)
                ax.plot(local.OCCUPANCIES, np.nanmean(lines, axis=0), "o-",
                        color=color, lw=2.5, label=f"{group} (n={len(lines)})")
            ax.set(title=title, xlabel="5-HT2A occupancy", ylabel="PCI",
                   xticks=local.OCCUPANCIES, xlim=(-.02, .79))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
        axes[-1].legend(frameon=False, fontsize=9, loc="best")
        fig.savefig(figures / f"pci_three_methods_el{int(-leak)}.pdf", bbox_inches="tight")
        fig.savefig(figures / f"pci_three_methods_el{int(-leak)}.png", dpi=220,
                    bbox_inches="tight")
        plt.close(fig)

    # Full binary maps are saved for every case; these eight figures are a
    # readable five-subject overview at zero occupancy, not a subset analysis.
    for leak in local.LEAKS:
        for group in COLORS:
            candidates = sorted((r for r in rows if r["case"]["leak_mv"] == leak
                                 and r["case"]["group"] == group
                                 and r["case"]["occupancy"] == 0),
                                key=lambda r: r["case"]["subject"])
            chosen = [candidates[i] for i in np.rint(np.linspace(0, len(candidates)-1, 5)).astype(int)]
            fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True,
                                     layout="constrained")
            for ax, row in zip(axes, chosen):
                with np.load(output / "masks" / f"case_{row['case_index']:04d}.npz") as d:
                    mask = d["dalla_porta"].astype(bool)
                    source = int(d["source"])
                rank = mask.sum(axis=1).argsort()[::-1]
                source_row = int(np.flatnonzero(rank == source)[0])
                ax.imshow(mask[rank], aspect="auto", interpolation="nearest",
                          origin="upper", cmap="Greys", vmin=0, vmax=1,
                          extent=[0, 300, 89.5, -.5])
                ax.axhline(source_row, color="#df3d39", lw=.9, ls="--")
                ax.set(ylabel=row["case"]["subject"], xlim=(0, 300),
                       ylim=(89.5, -.5))
            axes[-1].set_xlabel("Time after pulse onset (ms)")
            fig.savefig(figures / f"zero_occ_dalla_masks_{group.lower()}_el{int(-leak)}.pdf")
            fig.savefig(figures / f"zero_occ_dalla_masks_{group.lower()}_el{int(-leak)}.png",
                        dpi=220)
            plt.close(fig)


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
    if manifest["protocol"] != FULL_PROTOCOL:
        raise ValueError("Not the frozen full-cohort Dalla Porta bank")
    missing = [i for i in range(len(manifest["cases"]))
               if not (args.bank / "case_complete" / f"case_{i:04d}.json").exists()]
    if missing:
        raise ValueError(f"Simulation bank incomplete: {len(missing)} cases; first {missing[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    methods_path = args.output / "methods.json"
    method_info = dict(bank_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       source_sha256={str(path.relative_to(ROOT)): local.sha256(path)
                                      for path in (PILOT / "analyze.py",
                                                   FIRST / "analyze.py",
                                                   ROOT / "scripts/compare_gaglioti_supplied_binJproc.py",
                                                   ROOT / "scripts/gaglioti_adex_analysis.py",
                                                   ROOT / "src/tvbtoolkit/complexity/pci_casali.py")},
                       methods=list(METHODS), trials=local.TRIALS,
                       baseline_ms=[-300, 0], response_ms=[0, 300],
                       dalla_bootstraps=1000, hybrid_bootstraps=500,
                       alpha=.01, tvbsim_shuffles=10,
                       all_finite_trials_retained=True,
                       note="Dalla Porta averages trials before threshold and PCI; hybrid/TVBSim score each poststimulus trial. Hybrid alpha is not calibrated per-trial.")
    if methods_path.exists() and json.loads(methods_path.read_text()) != method_info:
        raise ValueError("Analysis output belongs to different code or input bank")
    if not methods_path.exists():
        local.write_json(methods_path, method_info)
    payloads = [(str(args.bank), str(args.output), i, case)
                for i, case in enumerate(manifest["cases"])]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_case, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 20 == 0 or done == len(payloads):
                print(f"{done}/{len(payloads)} cases analysed", flush=True)
    rows.sort(key=lambda row: row["case_index"])
    summary = [dict(case_index=r["case_index"], **r["case"],
                    trials=r["trials"], source_index=r["source_index"],
                    source_label=r["source_label"],
                    high_rate_trials=r["high_rate_trials"],
                    covariance_invalid_trials=r["covariance_invalid_trials"],
                    dalla_porta_pci=r["dalla_porta_pci"],
                    dalla_porta_status=r["dalla_porta_status"],
                    dalla_porta_significant_bins=r["dalla_porta_significant_bins"],
                    dalla_porta_responsive_regions=r["dalla_porta_responsive_regions"],
                    dalla_porta_cutoff_z=r["dalla_porta_cutoff_z"],
                    hybrid_pci_mean=r["hybrid_pci_mean"],
                    hybrid_pci_sd=r["hybrid_pci_sd"],
                    hybrid_cutoff_z=r["hybrid_cutoff_z"],
                    tvbsim_pci_mean=r["tvbsim_pci_mean"],
                    tvbsim_pci_sd=r["tvbsim_pci_sd"],
                    tvbsim_cutoff_z=r["tvbsim_cutoff_z"])
               for r in rows]
    trials = [dict(case_index=r["case_index"], **r["case"], **score)
              for r in rows for score in r["per_trial"]]
    write_csv(args.output / "subject_summary.csv", summary)
    write_csv(args.output / "trial_scores.csv", trials)
    plot_results(rows, args.output)
    local.write_json(args.output / "completion.json", dict(cases=len(rows),
                     simulations=len(rows)*local.TRIALS,
                     bank_manifest_sha256=method_info["bank_manifest_sha256"]))


if __name__ == "__main__":
    main()
