#!/usr/bin/env python3
"""Combine native/Laplacian three-subject PCI results into readable plots."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Ellipse

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from tvbtoolkit.complexity.pci_casali import binarise_signals_casali


COLORS = {"control": "#2E86AB", "uws": "#C0392B"}
SUBJECTS = [("control", "c0015"), ("uws", "u0001"), ("uws", "u0020")]


def _aal90_centres(labels: np.ndarray) -> np.ndarray:
    path = _REPO_ROOT / "data" / "connectivity" / "average_aal90" / "centres.txt"
    by_label: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4:
            by_label[parts[0]] = np.asarray(parts[1:], dtype=float)
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise KeyError(f"AAL90 centres missing labels: {missing[:5]}")
    return np.stack([by_label[label] for label in labels])


def _peak_latencies(time: np.ndarray, delta: np.ndarray, n_peaks: int = 5) -> np.ndarray:
    post = (time >= 0.0) & (time <= 300.0)
    t_post = time[post]
    gfp = np.std(delta[post], axis=1)
    selected: list[int] = []
    for index in np.argsort(gfp)[::-1]:
        if all(abs(t_post[index] - t_post[previous]) >= 24.0 for previous in selected):
            selected.append(int(index))
        if len(selected) == n_peaks:
            break
    return np.sort(t_post[selected])


def _plot_region_brain(ax: plt.Axes, centres: np.ndarray, values: np.ndarray, vlim: float) -> None:
    """Compact superior-view AAL90 activation map for a butterfly peak."""
    xy = centres[:, :2].copy()
    xy[:, 0] /= max(np.max(np.abs(xy[:, 0])), 1.0)
    xy[:, 1] /= max(np.max(np.abs(xy[:, 1])), 1.0)
    ax.add_patch(Ellipse((-0.47, 0), 0.88, 1.92, facecolor="#F2F2F2", edgecolor="#555555", lw=0.7))
    ax.add_patch(Ellipse((0.47, 0), 0.88, 1.92, facecolor="#F2F2F2", edgecolor="#555555", lw=0.7))
    ax.plot([0, 0], [-0.94, 0.94], color="white", lw=2.0, zorder=2)
    ax.scatter(
        xy[:, 0], xy[:, 1], c=values, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        s=18 + 38 * np.abs(values) / max(vlim, np.finfo(float).eps),
        edgecolors="#303030", linewidths=0.25, zorder=3,
    )
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal"); ax.axis("off")


def _tdcs_style_subject_figure(root: Path, cohort: str, subject_id: str) -> None:
    modes = ["native", "laplacian"]
    region_colors = plt.cm.hsv(np.linspace(0, 1, 90, endpoint=False))
    fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, hspace=0.42, top=0.91, bottom=0.08)
    fig.suptitle(
        f"Regional TMS-like response — {cohort.upper()} {subject_id}",
        fontsize=16, weight="bold", y=0.98,
    )
    for row, mode in enumerate(modes):
        path = root / mode / "aligned_time_courses" / f"{cohort}_{subject_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            time = np.asarray(data["time_relative_ms"], dtype=float)
            rates = np.asarray(data["mean_rate_hz"], dtype=float)
            labels = np.asarray(data["region_labels"]).astype(str)
            stim_index = int(np.asarray(data["stim_region_index"]).reshape(-1)[0])
        delta = rates - np.mean(rates[time < 0.0], axis=0)
        centres = _aal90_centres(labels)
        peaks = _peak_latencies(time, delta)
        plot_mask = (time >= -50.0) & (time <= 300.0)
        trace_limit = float(np.percentile(np.abs(delta[plot_mask]), 99.7))
        brain_limit = float(np.percentile(np.abs(delta[time >= 0.0]), 98.5))
        grid = outer[row, 0].subgridspec(2, 1, height_ratios=[0.9, 1.0], hspace=0.02)
        ax_maps = fig.add_subplot(grid[0, 0]); ax_maps.set_xlim(-50, 300); ax_maps.set_ylim(0, 1); ax_maps.axis("off")
        ax = fig.add_subplot(grid[1, 0])
        for region in range(delta.shape[1]):
            ax.plot(time[plot_mask], delta[plot_mask, region], color=region_colors[region], alpha=0.42, lw=0.75)
        ax.plot(time[plot_mask], delta[plot_mask, stim_index], color="#111111", lw=1.6, label=labels[stim_index], zorder=5)
        ax.axvspan(-50, 0, color="0.95", zorder=-10); ax.axvspan(0, 10, color="#F4C95D", alpha=0.25, zorder=-9)
        ax.axvline(0, color="black", lw=1.0); ax.axhline(0, color="0.3", lw=0.65)
        ax.set_xlim(-50, 300); ax.set_ylim(-1.08 * trace_limit, 1.08 * trace_limit)
        ax.grid(color="0.9", lw=0.5); ax.set_ylabel("Δ firing rate (Hz)"); ax.set_xlabel("Time from stimulation (ms)")
        ax.text(0.01, 0.91, mode.capitalize(), transform=ax.transAxes, weight="bold", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75})
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fractions = np.linspace(0.12, 0.88, len(peaks))
        for peak, fraction in zip(peaks, fractions, strict=True):
            inset = ax_maps.inset_axes([fraction - 0.047, 0.03, 0.094, 0.88])
            sample = int(np.argmin(np.abs(time - peak)))
            _plot_region_brain(inset, centres, delta[sample], brain_limit)
            inset.set_title(f"{peak:.0f} ms", fontsize=8)
            ax.axvline(peak, color="0.2", lw=0.8, alpha=0.55)
            fig.add_artist(ConnectionPatch(
                xyA=(0.5, 0.0), coordsA=inset.transAxes,
                xyB=(peak, trace_limit), coordsB=ax.transData,
                color="0.25", lw=0.8, alpha=0.65, clip_on=False,
            ))
    output = root / "tdcs_style_butterflies"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{cohort}_{subject_id}_regional_butterfly_brainmaps.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/momi_pci_operator_comparison"))
    args = parser.parse_args()
    root = args.root.resolve()
    modes = ["native", "laplacian"]
    rows: list[dict[str, str]] = []
    for mode in modes:
        rows.extend(_read_rows(root / mode / "subject_metrics.csv"))

    fields = list(rows[0])
    with (root / "combined_subject_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["connectivity_mode"], row["cohort"], row["subject_id"]): row for row in rows}
    labels = [f"{cohort.upper()}\n{subject}" for cohort, subject in SUBJECTS]
    colors = [COLORS[cohort] for cohort, _ in SUBJECTS]
    x = np.arange(len(SUBJECTS))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, mode in enumerate(modes):
        current = [float(lookup[(mode, *subject)]["pci_lz"]) for subject in SUBJECTS]
        legacy = [float(lookup[(mode, *subject)]["tvbsim_pci_lz"]) for subject in SUBJECTS]
        axes[0, column].bar(x - 0.18, current, width=0.36, color=colors, alpha=1.0, label="Current")
        axes[0, column].bar(x + 0.18, legacy, width=0.36, color=colors, alpha=0.45, hatch="//", label="Legacy TVBSim")
        axes[0, column].set_xticks(x, labels)
        axes[0, column].set_ylabel("PCI-LZ")
        axes[0, column].set_title(f"{mode.capitalize()} connectivity")
        axes[0, column].legend(frameon=False)
        entropy = [float(lookup[(mode, *subject)]["pci_lz_source_entropy"]) for subject in SUBJECTS]
        active = [100.0 * float(lookup[(mode, *subject)]["pci_lz_active_fraction"]) for subject in SUBJECTS]
        axes[1, column].bar(x - 0.18, entropy, width=0.36, color=colors, label="Entropy")
        twin = axes[1, column].twinx()
        twin.bar(x + 0.18, active, width=0.36, color=colors, alpha=0.35, hatch="..", label="Active cells")
        axes[1, column].set_xticks(x, labels)
        axes[1, column].set_ylabel("Response entropy (diagnostic)")
        twin.set_ylabel("Significant cells (%)")
    fig.suptitle("PCI-LZ depends on both connectivity operator and estimator\n(no entropy cutoff)")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"operator_and_pci_method_comparison.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True, sharex=True)
    for row_index, mode in enumerate(modes):
        for column, (cohort, subject_id) in enumerate(SUBJECTS):
            path = root / mode / "aligned_time_courses" / f"{cohort}_{subject_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                time = np.asarray(data["time_relative_ms"], dtype=float)
                rates = np.asarray(data["mean_rate_hz"], dtype=float)
            delta = rates - np.mean(rates[time < 0.0], axis=0)
            mean_delta = np.mean(delta, axis=1)
            axes[row_index, column].plot(time, mean_delta, color=COLORS[cohort], lw=1.3)
            axes[row_index, column].axvline(0.0, color="black", ls="--", lw=0.8)
            axes[row_index, column].axvspan(0.0, 10.0, color="#F4C95D", alpha=0.3)
            if row_index == 0:
                axes[row_index, column].set_title(f"{cohort.upper()} {subject_id}")
            if column == 0:
                axes[row_index, column].set_ylabel(f"{mode.capitalize()}\nmean Δ rate (Hz)")
            if row_index == 1:
                axes[row_index, column].set_xlabel("Time from stimulation (ms)")
    fig.suptitle("Whole-brain response under the two connectivity operators")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"operator_time_course_comparison.{suffix}", dpi=300)
    plt.close(fig)

    # True butterfly plots: one line per AAL90 region, with the stimulated
    # left SMA shown explicitly instead of hiding propagation in an average.
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True, sharex=True)
    for row_index, mode in enumerate(modes):
        for column, (cohort, subject_id) in enumerate(SUBJECTS):
            path = root / mode / "aligned_time_courses" / f"{cohort}_{subject_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                time = np.asarray(data["time_relative_ms"], dtype=float)
                rates = np.asarray(data["mean_rate_hz"], dtype=float)
                labels_array = np.asarray(data["region_labels"]).astype(str)
                stim_index = int(np.asarray(data["stim_region_index"]).reshape(-1)[0])
            delta = rates - np.mean(rates[time < 0.0], axis=0)
            axis = axes[row_index, column]
            for region in range(delta.shape[1]):
                if region == stim_index:
                    continue
                axis.plot(time, delta[:, region], color="#7F8C8D", alpha=0.28, lw=0.65)
            axis.plot(
                time,
                delta[:, stim_index],
                color="#D62728",
                lw=1.8,
                label=str(labels_array[stim_index]),
                zorder=5,
            )
            axis.axvline(0.0, color="black", ls="--", lw=0.8)
            axis.axvspan(0.0, 10.0, color="#F4C95D", alpha=0.3)
            if row_index == 0:
                axis.set_title(f"{cohort.upper()} {subject_id}")
            if column == 0:
                axis.set_ylabel(f"{mode.capitalize()}\nΔ firing rate (Hz)")
            if row_index == 1:
                axis.set_xlabel("Time from stimulation (ms)")
            axis.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Regional propagation: every AAL90 time course (butterfly plot)")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"operator_regional_butterfly.{suffix}", dpi=300)
    plt.close(fig)

    # Show exactly how the PCI threshold is generated and what survives it.
    for mode in modes:
        fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
        for column, (cohort, subject_id) in enumerate(SUBJECTS):
            path = root / mode / "aligned_time_courses" / f"{cohort}_{subject_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                time = np.asarray(data["time_relative_ms"], dtype=float)
                trials = np.asarray(data["trial_rates_hz"], dtype=float).transpose(0, 2, 1) / 1000.0
                labels_array = np.asarray(data["region_labels"]).astype(str)
            onset = int(np.argmin(np.abs(time)))
            significance = binarise_signals_casali(
                trials,
                t_stim=onset,
                n_bootstrap=500,
                alpha=0.01,
                seed=0,
                significance_method="trial_bootstrap",
                return_details=True,
            )
            response_start = onset + int(np.ceil(8.0 / np.median(np.diff(time))))
            response_time = time[response_start:]
            observed = np.asarray(significance.observed_statistic[:, response_start:], dtype=float)
            binary = np.asarray(significance.binary[:, response_start:], dtype=np.uint8)
            threshold = float(significance.threshold)

            axis = axes[0, column]
            for region in range(observed.shape[0]):
                axis.plot(response_time, observed[region], color="#7F8C8D", alpha=0.25, lw=0.55)
            axis.axhline(threshold, color="#C0392B", ls="--", lw=1.2, label=f"threshold = {threshold:.2f}")
            axis.axhline(-threshold, color="#C0392B", ls="--", lw=1.2)
            axis.set_title(f"{cohort.upper()} {subject_id}")
            axis.set_ylabel("Normalized response")
            axis.legend(frameon=False, fontsize=8)

            axis = axes[1, column]
            null = np.asarray(significance.null_maxima, dtype=float).reshape(-1)
            axis.hist(null, bins=28, color="#95A5A6", edgecolor="white")
            axis.axvline(threshold, color="#C0392B", ls="--", lw=1.5, label="99th percentile")
            axis.set_xlabel("Maximum response in a baseline resample")
            axis.set_ylabel("Resamples")
            axis.legend(frameon=False, fontsize=8)

            axis = axes[2, column]
            axis.imshow(
                binary,
                aspect="auto",
                origin="lower",
                extent=[response_time[0], response_time[-1], 0, binary.shape[0]],
                cmap="Greys",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            active_sources = np.flatnonzero(binary.any(axis=1))
            names = ", ".join(labels_array[index] for index in active_sources) or "none"
            axis.set_title(f"Survives threshold: {names}", fontsize=9)
            axis.set_xlabel("Time after stimulation (ms)")
            axis.set_ylabel("AAL90 region")
        fig.suptitle(
            f"{mode.capitalize()} PCI threshold audit: response, bootstrap null, and surviving map\n"
            "500 baseline-trial resamples, family-wise α=0.01"
        )
        for suffix in ("png", "pdf"):
            fig.savefig(root / f"{mode}_pci_threshold_audit.{suffix}", dpi=300)
        plt.close(fig)

    for cohort, subject_id in SUBJECTS:
        _tdcs_style_subject_figure(root, cohort, subject_id)


if __name__ == "__main__":
    main()
