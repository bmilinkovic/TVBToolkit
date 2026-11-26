"""Hybrid PCI-LZ analysis of the frozen receptor-topography recording bank.

Each case uses the established 500-bootstrap pooled-prestimulus hybrid mask,
two-sided at nominal alpha=.01, followed by per-trial PCI-LZ and averaging of
100 trial scores. No recordings are resimulated or excluded for high rates.
Map-null tests compare the observed *group mean delta* (and MCS-UWS gradient)
with a common map-null distribution across the same participants.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tvbtoolkit-mpl-cache")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.sma8s_full100.analyze import hybrid_masks
from experiments.sma8s_receptor_topography.common import (
    TRIALS, atomic_json, recording_path, sha256_file,
)
from gaglioti_adex_analysis import complexity


def _epochs(bank: Path, case: dict):
    epochs, meta, labels, source = [], [], None, None
    for trial in range(TRIALS):
        path = recording_path(bank, case, trial)
        with np.load(path) as saved:
            t = saved["time_ms"]
            pre, post = (t >= -300) & (t < 0), (t >= 0) & (t < 300)
            if saved["E_hz"].shape != (1000, 90) or pre.sum() != 300 or post.sum() != 300:
                raise ValueError(f"Incomplete recording: {path}")
            if labels is None:
                labels, source = saved["labels"].astype(str), int(saved["source"])
            elif not np.array_equal(labels, saved["labels"].astype(str)) or source != int(saved["source"]):
                raise ValueError(f"Atlas or source changed: {path}")
            x = saved["E_hz"].astype(np.float64)
            epochs.append(np.concatenate((x[pre].T, x[post].T), axis=1))
        row = json.loads(path.with_suffix(".json").read_text())
        if row["case"] != case or row["trial"] != trial:
            raise ValueError(f"Trial metadata mismatch: {path}")
        meta.append(row)
    return np.stack(epochs), meta, labels, source


def _score_case(payload: tuple) -> dict:
    bank_string, output_string, index, case = payload
    bank, output = Path(bank_string), Path(output_string)
    case_path = output / "cases" / f"case_{index:05d}.json"
    mask_path = output / "masks" / f"case_{index:05d}.npz"
    if case_path.exists() and mask_path.exists():
        saved = json.loads(case_path.read_text())
        if saved["case"] != case or saved["case_index"] != index:
            raise ValueError(f"Existing PCI case belongs to different inputs: {case_path}")
        return dict(case_index=index, mean_pci_lz=saved["mean_pci_lz"])
    epochs, meta, labels, source = _epochs(bank, case)
    # Match the original full100 hybrid bootstrap seed for the same subject.
    seed = int(np.random.SeedSequence([int(meta[0]["seed"]), 63001]).generate_state(1)[0])
    masks, cutoff, _ = hybrid_masks(epochs, seed)
    if masks.shape != (TRIALS, 90, 300):
        raise ValueError(f"Wrong hybrid mask shape: {masks.shape}")
    trials = []
    for trial, mask in enumerate(masks):
        result = complexity(mask)
        trials.append(dict(trial=trial, seed=meta[trial]["seed"],
                           onset_ms=meta[trial]["onset_ms"],
                           peak_EI_hz=meta[trial]["peak_EI_hz"],
                           high_rate_post_transient_steps=meta[trial]["high_rate_post_transient_steps"],
                           significant_bins=int(mask.sum()), **result))
    values = np.asarray([row["pci_lz"] for row in trials])
    row = dict(case_index=index, case=case, method="hybrid", n_trials=TRIALS,
               source_index=source, source_label=str(labels[source]),
               threshold_z=float(cutoff), mean_pci_lz=float(values.mean()),
               sd_trial_pci_lz=float(values.std(ddof=1)),
               mean_significant_bins=float(np.mean([r["significant_bins"] for r in trials])),
               mean_responsive_regions=float(np.mean([r["responsive_regions"] for r in trials])),
               high_rate_trials=sum(row["peak_EI_hz"] > 150 for row in meta),
               high_rate_post_transient_trials=sum(
                   row["high_rate_post_transient_steps"] > 0 for row in meta),
               trials=trials)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, binary=masks, labels=labels,
                        source_index=source, time_ms=np.arange(300) + .5,
                        threshold_z=cutoff)
    temporary.replace(mask_path)
    atomic_json(case_path, row)
    return dict(case_index=index, mean_pci_lz=row["mean_pci_lz"])


def _stream_tables(output: Path, cases: list[dict]) -> pd.DataFrame:
    subjects = []
    trial_path = output / "trial_scores.csv"
    with trial_path.open("w", newline="") as stream:
        writer = None
        for index, case in enumerate(cases):
            row = json.loads((output / "cases" / f"case_{index:05d}.json").read_text())
            flat = dict(case_index=index, **case, method="hybrid",
                        n_trials=TRIALS, source_index=row["source_index"],
                        source_label=row["source_label"], threshold_z=row["threshold_z"],
                        mean_pci_lz=row["mean_pci_lz"],
                        sd_trial_pci_lz=row["sd_trial_pci_lz"],
                        mean_significant_bins=row["mean_significant_bins"],
                        mean_responsive_regions=row["mean_responsive_regions"],
                        high_rate_trials=row["high_rate_trials"],
                        high_rate_post_transient_trials=row["high_rate_post_transient_trials"])
            subjects.append(flat)
            for trial in row["trials"]:
                record = dict(case_index=index, **case, **trial)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
    summary = pd.DataFrame(subjects)
    summary.to_csv(output / "subject_summary.csv", index=False)
    return summary


def _null_results(summary: pd.DataFrame, manifest: dict, output: Path) -> None:
    baseline = summary[summary.map_id.eq("baseline")][["group", "subject", "mean_pci_lz"]]
    if len(baseline) != 126 or baseline.duplicated(["group", "subject"]).any():
        raise ValueError("Expected one zero-dose baseline per MCS/UWS subject")
    high = summary[~summary.map_id.eq("baseline")].merge(
        baseline.rename(columns={"mean_pci_lz": "baseline_pci_lz"}),
        on=["group", "subject"], validate="many_to_one")
    high["delta_pci_lz"] = high.mean_pci_lz - high.baseline_pci_lz
    kinds = {row["map_id"]: row["kind"] for row in manifest["map_rows"]}
    high["map_kind"] = high.map_id.map(kinds)
    high.to_csv(output / "delta_by_subject_and_map.csv", index=False)
    grouped = high.groupby(["map_id", "map_kind", "group"], as_index=False).agg(
        mean_delta_pci_lz=("delta_pci_lz", "mean"),
        sd_delta_pci_lz=("delta_pci_lz", "std"),
        n_subjects=("subject", "nunique"),
        high_rate_post_transient_trials=("high_rate_post_transient_trials", "sum"))
    grouped.to_csv(output / "null_map_group_summary.csv", index=False)
    pivot = grouped.pivot(index=["map_id", "map_kind"], columns="group",
                          values="mean_delta_pci_lz").reset_index()
    pivot["mcs_minus_uws"] = pivot.MCS - pivot.UWS
    pivot.to_csv(output / "null_map_gradient_summary.csv", index=False)
    observed = pivot.loc[pivot.map_id.eq("original")].iloc[0]
    tests = []
    for metric in ("MCS", "UWS", "mcs_minus_uws"):
        value = float(observed[metric])
        uniform = float(pivot.loc[pivot.map_id.eq("uniform"), metric].iloc[0])
        for kind, n in (("spin", manifest["n_spin"]),
                        ("permutation", manifest["n_permutation"])):
            if n == 0:
                continue
            null = pivot.loc[pivot.map_kind.eq(kind), metric].to_numpy(float)
            if len(null) != n:
                raise ValueError(f"Missing {kind} null maps for {metric}")
            tests.append(dict(metric=metric, kind=kind, observed=value,
                              uniform=uniform, n_null=len(null),
                              null_mean=float(null.mean()),
                              null_q025=float(np.quantile(null, .025)),
                              null_q975=float(np.quantile(null, .975)),
                              p_observed_greater=(1 + int(np.sum(null >= value))) / (len(null) + 1),
                              p_two_sided=(1 + int(np.sum(np.abs(null-null.mean()) >=
                                                          abs(value-null.mean())))) / (len(null) + 1)))
    pd.DataFrame(tests).to_csv(output / "null_tests.csv", index=False)
    colors = {"MCS": "#8d4bad", "UWS": "#4a226c", "mcs_minus_uws": "#726477"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, metric in zip(axes, ("MCS", "UWS", "mcs_minus_uws"), strict=True):
        for kind, color in (("spin", "#8baaa4"), ("permutation", "#c99d7a")):
            values = pivot.loc[pivot.map_kind.eq(kind), metric].to_numpy(float)
            if len(values) == 0:
                continue
            ax.hist(values, bins=min(18, max(5, len(values)//5)),
                    color=color, alpha=.55, label=kind)
        ax.axvline(float(observed[metric]), color=colors[metric], lw=2.5,
                   label="original")
        ax.axvline(float(pivot.loc[pivot.map_id.eq("uniform"), metric].iloc[0]),
                   color="#3f424b", lw=2, ls="--", label="uniform")
        ax.set(xlabel=r"$\Delta$PCI" if metric != "mcs_minus_uws"
               else r"MCS $-$ UWS $\Delta$PCI", ylabel="Null maps",
               title=metric.replace("mcs_minus_uws", "MCS - UWS"))
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=9)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "original_vs_spatial_nulls.pdf")
    fig.savefig(figures / "original_vs_spatial_nulls.png", dpi=220)
    plt.close(fig)


def _single_permutation_figure(summary: pd.DataFrame, output: Path) -> None:
    """Show each patient's paired baseline, real-map and shuffled-map PCI."""
    selected = summary[summary.map_id.isin(("baseline", "original", "permutation_000"))]
    paired = selected.pivot(index=["group", "subject"], columns="map_id",
                            values="mean_pci_lz")
    if len(paired) != 126 or paired.isna().any().any():
        raise ValueError("The one-permutation comparison lacks paired patient scores")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"MCS": "#8d4bad", "UWS": "#4a226c"}
    for row, group in enumerate(("MCS", "UWS")):
        values = paired.loc[group]
        baseline = values["baseline"].to_numpy(float)
        original = values["original"].to_numpy(float)
        shuffled = values["permutation_000"].to_numpy(float)
        ax, delta_ax = axes[row]
        for a, b, c in zip(baseline, original, shuffled, strict=True):
            ax.plot((0, 1), (a, b), color=colors[group], alpha=.16, lw=.8)
            ax.plot((0, 2), (a, c), color="#9a969e", alpha=.16, lw=.8)
        ax.scatter((0, 1, 2), (baseline.mean(), original.mean(), shuffled.mean()),
                   color=("#35323a", colors[group], "#88848e"), s=85, zorder=4)
        ax.set(xticks=(0, 1, 2), xticklabels=("Zero dose", "Real map", "Permuted map"),
               ylabel="PCI", title=f"{group}: paired patient scores")
        delta_real, delta_permuted = original - baseline, shuffled - baseline
        delta_ax.scatter(delta_real, delta_permuted, color=colors[group],
                         s=24, alpha=.55, edgecolors="none")
        limits = np.array([delta_real.min(), delta_real.max(),
                           delta_permuted.min(), delta_permuted.max()])
        low, high = float(limits.min()), float(limits.max())
        pad = max(.005, .05 * (high-low))
        delta_ax.plot((low-pad, high+pad), (low-pad, high+pad),
                      color="#8d8993", ls="--", lw=1)
        delta_ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad),
                     xlabel=r"Real-map $\Delta$PCI", ylabel=r"Permuted-map $\Delta$PCI",
                     title=f"{group}: response to high dose")
        for panel in (ax, delta_ax):
            panel.spines[["top", "right"]].set_visible(False)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "one_permutation_paired_comparison.pdf")
    fig.savefig(figures / "one_permutation_paired_comparison.png", dpi=220)
    plt.close(fig)


def _verify_and_delete_recordings(bank: Path, output: Path, manifest: dict,
                                  bank_hash: str) -> None:
    """Delete only this bank's raw recordings after all PCI products validate."""
    expected_cases = len(manifest["cases"])
    baseline_cases = sum(case["map_id"] == "baseline" for case in manifest["cases"])
    summary = pd.read_csv(output / "subject_summary.csv")
    deltas = pd.read_csv(output / "delta_by_subject_and_map.csv")
    if len(summary) != expected_cases or len(deltas) != expected_cases - baseline_cases:
        raise ValueError("Summary tables incomplete; raw recordings retained")
    if not np.isfinite(summary["mean_pci_lz"]).all():
        raise ValueError("Nonfinite PCI scores; raw recordings retained")
    with (output / "trial_scores.csv").open() as stream:
        trial_rows = sum(1 for _ in stream) - 1
    if trial_rows != expected_cases * TRIALS:
        raise ValueError("Trial scores incomplete; raw recordings retained")
    for index in range(expected_cases):
        case_path = output / "cases" / f"case_{index:05d}.json"
        mask_path = output / "masks" / f"case_{index:05d}.npz"
        if not case_path.is_file() or not mask_path.is_file():
            raise ValueError(f"Missing case or mask {index}; raw recordings retained")
        case = json.loads(case_path.read_text())
        if case.get("case_index") != index or len(case.get("trials", [])) != TRIALS:
            raise ValueError(f"Incomplete PCI case {index}; raw recordings retained")
        with np.load(mask_path) as saved:
            if saved["binary"].shape != (TRIALS, 90, 300):
                raise ValueError(f"Incomplete PCI mask {index}; raw recordings retained")
    required = ("null_tests.csv", "null_map_group_summary.csv",
                "null_map_gradient_summary.csv", "completion.json",
                "figures/original_vs_spatial_nulls.pdf")
    if manifest["n_spin"] == 0 and manifest["n_permutation"] == 1:
        required += ("figures/one_permutation_paired_comparison.pdf",)
    if any(not (output / name).is_file() for name in required):
        raise ValueError("Missing downstream results; raw recordings retained")
    completion = json.loads((output / "completion.json").read_text())
    if completion["bank_manifest_sha256"] != bank_hash:
        raise ValueError("Wrong bank hash in PCI completion; raw recordings retained")
    recordings = bank / "recordings"
    if recordings.is_symlink() or recordings.resolve().parent != bank.resolve() \
            or output.resolve().is_relative_to(recordings.resolve()):
        raise ValueError("Unsafe recordings target; raw recordings retained")
    if not recordings.is_dir():
        raise ValueError("Recordings already absent; nothing deleted")
    shutil.rmtree(recordings)
    atomic_json(bank / "recordings_deleted_after_pci.json",
                dict(bank_manifest_sha256=bank_hash, pci_output=str(output.resolve()),
                     cases=expected_cases, trials=trial_rows,
                     deleted_path=str(recordings.resolve())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--delete-recordings-after-analysis", action="store_true",
                        help="After verifying all PCI outputs, remove this bank's raw recordings")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest_path = args.bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    missing = [i for i in range(len(cases)) if not
               (args.bank / "case_complete" / f"case_{i:05d}.json").exists()]
    if missing:
        raise ValueError(f"Simulation bank incomplete: {len(missing)} cases; first {missing[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    methods = dict(method="hybrid", simulation_bank=str(args.bank),
                   bank_manifest_sha256=sha256_file(manifest_path),
                   algorithm_sha256=sha256_file(Path(__file__)),
                   original_hybrid_sha256=sha256_file(
                       ROOT / "experiments/sma8s_full100/analyze.py"),
                   trials_per_case=TRIALS, bootstrap_draws=500,
                   nominal_alpha=.01, no_entropy_floor=True,
                   no_high_rate_exclusion=True,
                   prestim_ms=[-300, 0], poststim_ms=[0, 300])
    meta = args.output / "methods.json"
    if meta.exists() and json.loads(meta.read_text()) != methods:
        raise ValueError("PCI output belongs to different inputs/code")
    if not meta.exists():
        atomic_json(meta, methods)
    payloads = [(str(args.bank), str(args.output), i, case)
                for i, case in enumerate(cases)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score_case, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 100 == 0 or done == len(cases):
                print(f"Hybrid PCI: {done}/{len(cases)} cases", flush=True)
    summary = _stream_tables(args.output, cases)
    _null_results(summary, manifest, args.output)
    if manifest["n_spin"] == 0 and manifest["n_permutation"] == 1:
        _single_permutation_figure(summary, args.output)
    atomic_json(args.output / "completion.json",
                dict(method="hybrid", cases=len(cases), trials=len(cases)*TRIALS,
                     bank_manifest_sha256=methods["bank_manifest_sha256"],
                     n_spin=manifest["n_spin"],
                     n_permutation=manifest["n_permutation"]))
    if args.delete_recordings_after_analysis:
        _verify_and_delete_recordings(args.bank, args.output, manifest,
                                      methods["bank_manifest_sha256"])
        print("Verified PCI outputs; deleted this bank's raw recordings", flush=True)


if __name__ == "__main__":
    main()
