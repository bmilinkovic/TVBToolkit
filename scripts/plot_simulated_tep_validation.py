#!/usr/bin/env python3
"""Generate empirical-style TEP diagnostics from two AdEx connectomes.

This is a mechanistic validation, not a cohort-level statistical comparison.
It simulates matched trials for one control and one UWS subject, aligns every
trial to the left-SMA pulse, averages trials, and displays all 90 regional
responses together with AAL-centroid spatial snapshots and three PCI routes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(_ROOT / ".tvb-temp"))
for _path in (_ROOT / "src", _ROOT / "notebooks", _ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_serotonergic_pci_pilot as pilot
import seaborn as sns
from scipy.signal import find_peaks

import calibrate_pci_stimulus_g_homeostasis as calibration
from brain_act_hybrid_common import COND_COLORS, DATASET_ROOT
from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial
from tvbtoolkit.complexity.pci_st import pci_st_from_trials
from tvbtoolkit.datasets.brain_act import load_aal90_atlas
from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "results" / "simulated_tep_validation",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--g", type=float, default=0.25)
    parser.add_argument("--pulse-shape", default="square")
    parser.add_argument("--pulse-duration-ms", type=float, default=10.0)
    parser.add_argument("--pulse-amplitude-khz", type=float, default=0.0003)
    parser.add_argument("--monitor-period-ms", type=float, default=3.0)
    parser.add_argument("--analysis-ms", type=float, default=300.0)
    parser.add_argument("--response-start-ms", type=float, default=8.0)
    parser.add_argument("--pci-permutations", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _simulation_args(args: argparse.Namespace, *, shared_b_e: float | None):
    dataset_index = json.loads(
        (Path(args.dataset_root) / "index.json").read_text(encoding="utf-8")
    )
    normalization = dict(dataset_index.get("connectivity_normalization", {}))
    scheme = str(normalization.get("scheme", "legacy_column_sum"))
    return argparse.Namespace(
        dataset_root=args.dataset_root,
        transient_ms=4000.0,
        analysis_ms=args.analysis_ms,
        monitor_period_ms=args.monitor_period_ms,
        response_start_ms=args.response_start_ms,
        saturation_hz=100.0,
        stim_region_index=args.stim_region_index,
        shared_b_e=shared_b_e,
        e_l_e_drug=-61.2,
        e_l_i_drug=-64.4,
        homeostatic_post_ms=1500.0,
        structural_connectivity_normalization=scheme,
        structural_connectivity_normalization_divisor=normalization.get("divisor"),
        simulator_connectivity_normalization=(
            "none" if scheme == "cohort_global_max" else "legacy_column_sum"
        ),
    )


def _simulate_all(tasks: list[dict], workers: int) -> list[dict]:
    if workers == 1:
        return [calibration._evoked_task(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=pilot.worker_initializer
    ) as pool:
        return list(pool.map(calibration._evoked_task, tasks, chunksize=1))


def _align_and_average(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(rows[0]["time_relative_ms"], dtype=float)
    trials = np.stack([np.asarray(row["rate_hz"], dtype=float) for row in rows])
    for row in rows[1:]:
        np.testing.assert_allclose(row["time_relative_ms"], times, atol=1e-9)
    baseline = times < 0.0
    centered_trials = trials - trials[:, baseline, :].mean(axis=1, keepdims=True)
    return times, trials, centered_trials.mean(axis=0)


def _pci_metrics(times: np.ndarray, trials: np.ndarray, args: argparse.Namespace) -> dict:
    onset = int(np.argmin(np.abs(times)))
    dt_ms = float(np.median(np.diff(times)))
    t_analysis_ms = min(onset, trials.shape[1] - onset) * dt_ms
    cut = int(round(t_analysis_ms / dt_ms))
    start, stop = onset - cut, onset + cut
    aligned = [trial[start:stop] for trial in trials]
    onset_cut = cut

    tvbsim, _ = pci_casali_like_multi_trial(
        aligned,
        stimulation_index=onset_cut,
        t_analysis_ms=t_analysis_ms,
        dt_ms=dt_ms,
        binarise_method="tvbsim",
        response_start_ms=args.response_start_ms,
        min_source_entropy=None,
    )
    casali = pci_casali_like_multi_trial(
        aligned,
        stimulation_index=onset_cut,
        t_analysis_ms=t_analysis_ms,
        dt_ms=dt_ms,
        binarise_method="casali",
        binarise_kwargs={
            "n_bootstrap": args.pci_permutations,
            "alpha": 0.05,
            "seed": 0,
            "significance_method": "pre_post_swap",
        },
        response_start_ms=args.response_start_ms,
        min_source_entropy=0.08,
        return_debug=True,
    )
    stack = np.stack([trial.T for trial in aligned], axis=0)
    cut_times = (np.arange(2 * cut) - cut) * dt_ms
    pci_st = pci_st_from_trials(
        stack,
        cut_times,
        baseline_center_trials=True,
        baseline_window_ms=(-t_analysis_ms, -50.0),
        response_window_ms=(args.response_start_ms, t_analysis_ms),
        k=1.2,
        min_snr=1.1,
        max_var_percent=99.0,
        n_steps=100,
        return_details=True,
    )
    return {
        "pci_lz_tvbsim": float(tvbsim),
        "pci_lz_significance": float(casali["pci"]),
        "pci_lz_active_fraction": float(casali["active_fraction"]),
        "pci_lz_entropy": float(casali["entropy"]),
        "pci_st": float(pci_st.pci_st),
        "pci_st_components": int(pci_st.n_components),
    }


def _aal_centres(labels: np.ndarray) -> np.ndarray:
    path = _ROOT / "data" / "connectivity" / "average_aal90" / "centres.txt"
    lookup = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, x, y, z = line.split()
        lookup[name] = (float(x), float(y), float(z))
    return np.asarray([lookup[str(label)] for label in labels], dtype=float)


def _snapshot_latencies(times: np.ndarray, evoked: np.ndarray, n: int = 5) -> np.ndarray:
    post = times >= 8.0
    gfp = np.sqrt(np.mean(evoked**2, axis=1))
    indices = np.flatnonzero(post)
    distance = max(1, int(round(15.0 / np.median(np.diff(times)))))
    peaks, _ = find_peaks(gfp[post], distance=distance)
    candidates = indices[peaks]
    if candidates.size < n:
        candidates = indices
    selected = candidates[np.argsort(gfp[candidates])[-n:]]
    return np.sort(selected)


def _plot_aal_snapshot(ax, centres, values, limit, title):
    ax.scatter(
        centres[:, 0], centres[:, 1], c=values, cmap="RdBu_r",
        vmin=-limit, vmax=limit, s=14, edgecolor="0.25", linewidth=0.2,
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(52 * np.cos(theta), -30 + 82 * np.sin(theta), color="0.15", lw=0.55)
    ax.axvline(0, color="0.75", lw=0.35)
    ax.set(xlim=(-62, 62), ylim=(-118, 68), title=title)
    ax.set_aspect("equal")
    ax.axis("off")


def _plot_butterflies(records, labels, centres, output_root):
    colors = plt.cm.hsv(np.linspace(0, 1, 90, endpoint=False))
    for regime in ("shared_b", "diagnosis_b"):
        selected = [record for record in records if record["regime"] == regime]
        fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, hspace=0.26)
        for row_index, record in enumerate(selected):
            sub = outer[row_index].subgridspec(2, 5, height_ratios=(1.0, 1.25), hspace=0.01)
            time = record["times"]
            evoked = record["evoked"]
            peak_indices = _snapshot_latencies(time, evoked)
            limit = max(0.1, float(np.percentile(np.abs(evoked[time >= 8]), 99)))
            for column, index in enumerate(peak_indices):
                _plot_aal_snapshot(
                    fig.add_subplot(sub[0, column]), centres, evoked[index], limit,
                    f"{time[index]:.0f} ms",
                )
            ax = fig.add_subplot(sub[1, :])
            for region in range(90):
                ax.plot(time, evoked[:, region], color=colors[region], lw=0.45, alpha=0.42)
            ax.axvspan(0, 8, color="0.86", alpha=0.8, lw=0)
            ax.axvline(0, color="0.1", lw=0.75)
            ax.axhline(0, color="0.25", lw=0.45)
            ax.set_xlim(-50, 300)
            ax.set_ylabel("Δ firing rate (Hz)")
            ax.set_title(
                f"{record['condition']}  {record['subject_id']}  "
                f"({record['n_trials']} aligned trials)", loc="left", weight="bold"
            )
            if row_index == 1:
                ax.set_xlabel("Time from left-SMA pulse (ms)")
        title = (
            "Shared adaptation (b = 5 pA): anatomy-only comparison"
            if regime == "shared_b"
            else "Diagnosis-configured adaptation: model-state comparison"
        )
        fig.suptitle(title, fontsize=10, weight="bold")
        sns.despine(fig=fig)
        for ext in ("pdf", "png"):
            fig.savefig(output_root / f"simulated_tep_{regime}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)


def _plot_summary(records, metrics, output_root):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    palette = COND_COLORS
    for record in records:
        times, evoked = record["times"], record["evoked"]
        gfp = np.sqrt(np.mean(evoked**2, axis=1))
        style = "-" if record["regime"] == "diagnosis_b" else "--"
        axes[0, 0].plot(
            times, gfp, color=palette[record["condition"]], ls=style, lw=1.25,
            label=f"{record['condition']}; {record['regime'].replace('_', ' ')}",
        )
        pre = times < 0
        threshold = 5.0 * np.std(evoked[pre], axis=0, ddof=1)
        active = np.sum(np.abs(evoked) > threshold, axis=1)
        axes[0, 1].plot(times, active, color=palette[record["condition"]], ls=style, lw=1.25)
    for ax in axes[0]:
        ax.axvspan(0, 8, color="0.86", lw=0)
        ax.axvline(0, color="0.15", lw=0.65)
        ax.set_xlim(-50, 300)
        ax.set_xlabel("Time from pulse (ms)")
    axes[0, 0].set(title="A  Global response magnitude", ylabel="Regional RMS (Hz)")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[0, 1].set(title="B  Spatial recruitment", ylabel="Regions above baseline + 5 SD")

    table = pd.DataFrame(metrics)
    lz_ax = axes[1, 0]
    regime_names = ["shared_b", "diagnosis_b"]
    method_names = ["pci_lz_tvbsim", "pci_lz_significance"]
    condition_names = ["CNT", "UWS"]
    width = 0.105
    for regime_index, regime in enumerate(regime_names):
        for method_index, method in enumerate(method_names):
            for condition_index, condition in enumerate(condition_names):
                value = float(
                    table.loc[
                        (table.regime == regime) & (table.condition == condition), method
                    ].iloc[0]
                )
                offset = (method_index - 0.5) * 0.30 + (condition_index - 0.5) * 0.12
                lz_ax.bar(
                    regime_index + offset,
                    value,
                    width=width,
                    color=palette[condition],
                    edgecolor="0.25",
                    linewidth=0.35,
                    hatch=("///" if method == "pci_lz_significance" else None),
                )
    lz_ax.set_xticks(range(len(regime_names)), [name.replace("_", " ") for name in regime_names])
    from matplotlib.patches import Patch

    lz_ax.legend(
        handles=[
            Patch(facecolor=palette[condition], label=condition)
            for condition in condition_names
        ]
        + [
            Patch(facecolor="white", edgecolor="0.25", label="TVBSim threshold"),
            Patch(facecolor="white", edgecolor="0.25", hatch="///", label="Significance threshold"),
        ],
        frameon=False,
        ncol=2,
    )
    sns.barplot(data=table, x="regime", y="pci_st", hue="condition", palette=palette, ax=axes[1, 1])
    axes[1, 0].set_title("C  PCI-LZ", loc="left")
    axes[1, 0].set_ylabel("PCI")
    axes[1, 0].set_xlabel("")
    axes[1, 1].set_title("D  PCI-ST", loc="left")
    axes[1, 1].set_xlabel("")
    if axes[1, 1].get_legend() is not None:
        axes[1, 1].get_legend().remove()
    sns.despine(fig=fig)
    for ext in ("pdf", "png"):
        fig.savefig(output_root / f"simulated_tep_pci_summary.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_root}; use --overwrite.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    atlas = load_aal90_atlas(args.dataset_root)
    labels = np.asarray(atlas.labels).astype(str)
    args.stim_region_index = labels.tolist().index(pilot.DEFAULT_STIM_REGION_LABEL)
    receptor = get_5ht2a_aal90(
        tracer="cimbi", csv_path=pilot.DEFAULT_RECEPTOR_CSV, target_labels=labels
    )
    jobs = pilot._select_subjects(
        args.dataset_root,
        ["control", "uws"],
        1,
        explicit_subjects=args.subject,
    )
    if sorted(job.cohort for job in jobs) != ["control", "uws"]:
        raise ValueError("Supply exactly one control and one UWS subject.")

    tasks = []
    for regime, shared_b in (("shared_b", 5.0), ("diagnosis_b", None)):
        sim_args = _simulation_args(args, shared_b_e=shared_b)
        for job in jobs:
            for seed in range(args.trials):
                tasks.append(
                    {
                        "job": job,
                        "occupancy": 0.0,
                        "g_value": args.g,
                        "seed": seed,
                        "shape": args.pulse_shape,
                        "duration": args.pulse_duration_ms,
                        "amplitude": args.pulse_amplitude_khz,
                        "receptor": receptor,
                        "args": copy.deepcopy(sim_args),
                    }
                )
    print(f"Running {len(tasks)} simulations on {args.workers} workers", flush=True)
    rows = _simulate_all(tasks, args.workers)

    records, metrics = [], []
    cursor = 0
    for regime in ("shared_b", "diagnosis_b"):
        for job in jobs:
            subject_rows = rows[cursor : cursor + args.trials]
            cursor += args.trials
            times, trials, evoked = _align_and_average(subject_rows)
            metric = _pci_metrics(times, trials, args)
            b_e = 5.0 if regime == "shared_b" else pilot.CONDITION_B_GRADIENT[job.condition]
            metric.update(
                regime=regime, condition=job.condition, cohort=job.cohort,
                subject_id=job.subject_id, b_e_pA=b_e, n_trials=args.trials,
            )
            metrics.append(metric)
            records.append(
                {
                    "regime": regime,
                    "condition": job.condition,
                    "subject_id": job.subject_id,
                    "n_trials": args.trials,
                    "times": times,
                    "trials": trials,
                    "evoked": evoked,
                }
            )
            np.savez_compressed(
                args.output_root / f"{regime}_{job.condition}_{job.subject_id}.npz",
                times_ms=times,
                trials_hz=trials,
                trial_average_delta_hz=evoked,
                region_labels=labels,
            )
    pd.DataFrame(metrics).to_csv(args.output_root / "simulated_tep_pci_metrics.csv", index=False)
    _style()
    centres = _aal_centres(labels)
    _plot_butterflies(records, labels, centres, args.output_root)
    _plot_summary(records, metrics, args.output_root)
    manifest = {
        "purpose": "two-subject mechanistic validation; not cohort inference",
        "subjects": [f"{job.cohort}:{job.subject_id}" for job in jobs],
        "n_trials": args.trials,
        "sampling_hz": 1000.0 / args.monitor_period_ms,
        "stimulus": {
            "region": pilot.DEFAULT_STIM_REGION_LABEL,
            "zero_based_index": args.stim_region_index,
            "shape": args.pulse_shape,
            "duration_ms": args.pulse_duration_ms,
            "amplitude_khz": args.pulse_amplitude_khz,
        },
        "G": args.g,
        "regimes": {"shared_b": 5.0, "diagnosis_b": pilot.CONDITION_B_GRADIENT},
        "pci": {
            "trial_average": True,
            "response_start_ms": args.response_start_ms,
            "significance_alpha": 0.05,
            "significance_method": "pre_post_swap",
            "permutations": args.pci_permutations,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(f"Figures written to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
