#!/usr/bin/env python3
"""Focused PCI b-state stimulation tests.

This script runs two diagnostics:

1. average_aal90: b-state sweep on the public average AAL90 connectome.
2. subject_smoke: one real Brain-Act subject per condition using subject-specific
   structural connectivity and tract lengths.

The default full average-AAL test is 4 b-states x 25 trials = 100 simulations,
using the calibrated PCI stimulation setting: 0.0003 kHz for 10 ms.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
NOTEBOOKS = PROJECT_ROOT / "notebooks"
for path in (SRC, NOTEBOOKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(PROJECT_ROOT / ".tvb-temp"))

from brain_act_hybrid_common import (  # noqa: E402
    BASE_PARAMETER_MODEL_NEW,
    DATASET_ROOT,
    RATE_MONITOR_PERIOD_MS_OLD,
    SCENARIOS,
    get_subject_jobs,
)
from tvbtoolkit.core.config import WholeBrainConfig  # noqa: E402
from tvbtoolkit.core.paths import doc_liege_results  # noqa: E402
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: E402
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import run_pci_trial_job  # noqa: E402
from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial  # noqa: E402


B_STATES: dict[str, dict[str, Any]] = {
    "b005_control_wake": {"b_e": 5.0, "label": "Control / wake-like"},
    "b035_sleep_emcs": {"b_e": 35.0, "label": "Sleep-like / EMCS"},
    "b055_mcs": {"b_e": 55.0, "label": "MCS"},
    "b075_uws": {"b_e": 75.0, "label": "UWS"},
}

CONDITION_B = {
    "control": ("b005_control_wake", 5.0),
    "emcs": ("b035_sleep_emcs", 35.0),
    "mcs": ("b055_mcs", 55.0),
    "uws": ("b075_uws", 75.0),
}

STIM_PROTOCOLS: dict[str, dict[str, float | str]] = {
    "amp00015_dur10": {"amplitude": 0.00015, "duration_ms": 10.0, "label": "0.15 Hz, 10 ms"},
    "amp00030_dur10": {"amplitude": 0.00030, "duration_ms": 10.0, "label": "0.30 Hz, 10 ms"},
    "amp00050_dur10": {"amplitude": 0.00050, "duration_ms": 10.0, "label": "0.50 Hz, 10 ms"},
    "amp00015_dur50": {"amplitude": 0.00015, "duration_ms": 50.0, "label": "0.15 Hz, 50 ms"},
    "amp00030_dur50": {"amplitude": 0.00030, "duration_ms": 50.0, "label": "0.30 Hz, 50 ms"},
    "amp00050_dur50": {"amplitude": 0.00050, "duration_ms": 50.0, "label": "0.50 Hz, 50 ms"},
}

STIM_CALIBRATION_B_STATES: dict[str, dict[str, float | str]] = {
    "b005": {"b_e": 5.0, "label": "b=5"},
    "b035": {"b_e": 35.0, "label": "b=35"},
    "b040": {"b_e": 40.0, "label": "b=40"},
}

STIM_CALIBRATION_HIGH_B_STATES: dict[str, dict[str, float | str]] = {
    "b055": {"b_e": 55.0, "label": "b=55"},
    "b075": {"b_e": 75.0, "label": "b=75"},
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_average_aal90(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.loadtxt(root / "weights.txt")
    lengths = np.loadtxt(root / "tract_lengths.txt")
    labels: list[str] = []
    for line in (root / "centres.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(line.split()[0])
    return weights, lengths, np.asarray(labels, dtype="U128")


def make_parameter_stimulus(stim_onset_ms: float, stim_duration_ms: float, total_sim_ms: float, stim_amplitude: float, stim_region: list[int]) -> dict[str, Any]:
    return {
        "stimtime": float(stim_onset_ms),
        "stimdur": float(stim_duration_ms),
        "stimperiod": float(total_sim_ms) * 10.0,
        "stimval": float(stim_amplitude),
        "stimregion": list(stim_region),
        "stimvariables": [0],
    }


def run_average_aal_trial(
    *,
    output_dir: Path,
    b_key: str,
    b_e: float,
    trial_seed: int,
    weights: np.ndarray,
    tract_lengths: np.ndarray,
    region_labels: np.ndarray,
    scenario_key: str,
    noise_alpha: float,
    shared_noise_mode: str,
    transient_ms: float,
    t_analysis_ms: float,
    total_sim_ms: float,
    stim_onset_ms: float,
    stim_amplitude: float,
    stim_duration_ms: float,
    stim_region: list[int],
    rate_monitor_period_ms: float,
    overwrite: bool,
) -> dict[str, Any]:
    save_path = output_dir / b_key / f"trial_{trial_seed:03d}.npz"
    if save_path.exists() and not overwrite:
        return {"save_path": str(save_path), "skipped_existing": True, "runtime_s": 0.0}

    base_model = deepcopy(BASE_PARAMETER_MODEL_NEW)
    base_model.update({
        "b_e": float(b_e),
        "noise_alpha": float(noise_alpha),
        "shared_noise_mode": str(shared_noise_mode),
    })
    parameter_stimulus = make_parameter_stimulus(
        stim_onset_ms=stim_onset_ms,
        stim_duration_ms=stim_duration_ms,
        total_sim_ms=total_sim_ms,
        stim_amplitude=stim_amplitude,
        stim_region=stim_region,
    )
    cfg = WholeBrainConfig(
        simulation_length_ms=float(total_sim_ms),
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=0.25,
        zerlaut_order=2,
        zerlaut_matteo=False,
        zerlaut_gk_gna=False,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(rate_monitor_period_ms),
        monitor_variables=(0, 1),
        weights=np.asarray(weights, dtype=float),
        tract_lengths=np.asarray(tract_lengths, dtype=float),
        parameter_overrides={
            "parameter_model": base_model,
            "parameter_stimulus": parameter_stimulus,
        },
    )
    t0 = perf_counter()
    sim = run_whole_brain_simulation(cfg, seed=int(trial_seed))
    runtime_s = float(perf_counter() - t0)

    t_ms = np.asarray(sim.time_ms, dtype=float)
    x = np.asarray(sim.raw, dtype=float)
    keep = t_ms >= float(transient_ms)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        time_ms=t_ms[keep],
        rate=x[keep],
        region_labels=region_labels,
        b_key=np.asarray([b_key]),
        b_e=np.asarray([float(b_e)]),
        trial_seed=np.asarray([int(trial_seed)]),
        scenario_key=np.asarray([scenario_key]),
        noise_alpha=np.asarray([float(noise_alpha)]),
        shared_noise_mode=np.asarray([shared_noise_mode]),
        stim_onset_ms=np.asarray([float(stim_onset_ms)]),
        t_analysis_ms=np.asarray([float(t_analysis_ms)]),
        rate_monitor_period_ms=np.asarray([float(rate_monitor_period_ms)]),
        stim_amplitude=np.asarray([float(stim_amplitude)]),
        stim_duration_ms=np.asarray([float(stim_duration_ms)]),
        stim_region=np.asarray(stim_region, dtype=int),
        connectivity_source=np.asarray(["average_aal90"]),
    )
    return {"save_path": str(save_path), "skipped_existing": False, "runtime_s": runtime_s}


def load_trial_window(path: Path, t_analysis_ms: float) -> tuple[np.ndarray, float]:
    d = np.load(path, allow_pickle=True)
    x = np.asarray(d["rate"], dtype=float)
    t = np.asarray(d["time_ms"], dtype=float)
    dt = float(np.median(np.diff(t)))
    stim = float(d["stim_onset_ms"][0])
    nbins = int(round(t_analysis_ms / dt))
    stim_idx = int(round((stim - float(t[0])) / dt))
    return x[stim_idx - nbins: stim_idx + nbins, :].T, dt


def summarize_trials(trial_paths: list[Path], t_analysis_ms: float, stim_region: int) -> dict[str, Any]:
    trials = []
    dt_ref = None
    for p in trial_paths:
        win, dt = load_trial_window(p, t_analysis_ms)
        trials.append(win)
        dt_ref = dt if dt_ref is None else dt_ref
    if not trials:
        return {}
    stack = np.stack(trials, axis=0)  # trials, regions, time
    nbins = stack.shape[2] // 2
    pre = stack[:, :, :nbins] * 1e3
    post = stack[:, :, nbins:] * 1e3
    baseline = pre.mean(axis=2, keepdims=True)
    delta = post - baseline
    stim_delta = delta[:, stim_region, :]
    late = stim_delta[:, -max(1, int(round(100.0 / dt_ref))):]
    out = {
        "n_trials": len(trials),
        "dt_ms": float(dt_ref),
        "pre_whole_mean_hz": float(pre.mean()),
        "pre_whole_std_hz": float(pre.mean(axis=1).std()),
        "pre_stim_region_mean_hz": float(pre[:, stim_region, :].mean()),
        "pre_stim_region_min_hz": float(pre[:, stim_region, :].min()),
        "pre_stim_region_max_hz": float(pre[:, stim_region, :].max()),
        "stim_peak_delta_hz": float(stim_delta.max()),
        "whole_peak_delta_hz": float(delta.max()),
        "late_residual_stim_delta_hz": float(late.mean()),
        "stim_abs_peak_hz": float(post[:, stim_region, :].max()),
        "explosive_abs_flag": bool(post[:, stim_region, :].max() > 100.0),
        "poor_recovery_flag": bool(late.mean() > max(5.0, 0.25 * max(float(stim_delta.max()), 1e-9))),
    }
    try:
        np.random.seed(0)
        pci_mean, pci_trials = pci_casali_like_multi_trial(
            trials,
            stimulation_index=nbins,
            t_analysis_ms=t_analysis_ms,
            dt_ms=float(dt_ref),
            nshuffles=10,
            percentile=100.0,
        )
        out["pci_mean"] = float(pci_mean)
        out["pci_trials_mean"] = float(np.mean(pci_trials))
    except Exception as exc:  # keep summaries robust during smoke tests
        out["pci_error"] = str(exc)
    return out


def load_aligned_delta(
    trial_paths: list[Path],
    *,
    t_analysis_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return trial stack in Hz and baseline-corrected mean response.

    Returns
    -------
    t_rel_ms : np.ndarray, shape (2*nbins,)
        Time relative to stimulus onset.
    mean_rate_hz : np.ndarray, shape (time, regions)
        Trial-averaged absolute excitatory firing rate in Hz.
    delta_hz : np.ndarray, shape (time, regions)
        Trial-averaged response after subtracting each region's pre-stimulus
        mean.
    """
    aligned: list[np.ndarray] = []
    t_rel_ref: np.ndarray | None = None
    for path in trial_paths:
        d = np.load(path, allow_pickle=True)
        x_hz = np.asarray(d["rate"], dtype=float) * 1e3
        t = np.asarray(d["time_ms"], dtype=float)
        dt = float(np.median(np.diff(t)))
        stim = float(d["stim_onset_ms"][0])
        nbins = int(round(t_analysis_ms / dt))
        stim_idx = int(round((stim - float(t[0])) / dt))
        i0 = stim_idx - nbins
        i1 = stim_idx + nbins
        if i0 < 0 or i1 > x_hz.shape[0]:
            raise ValueError(f"Window out of bounds for {path}")
        aligned.append(x_hz[i0:i1, :])
        t_rel = t[i0:i1] - stim
        if t_rel_ref is None:
            t_rel_ref = t_rel
    if not aligned or t_rel_ref is None:
        raise ValueError("No trial paths provided.")
    stack = np.stack(aligned, axis=0)  # trials, time, regions
    mean_rate = stack.mean(axis=0)
    baseline = mean_rate[t_rel_ref < 0].mean(axis=0)
    delta = mean_rate - baseline[np.newaxis, :]
    return t_rel_ref, mean_rate, delta


def _nn_axes(ax) -> None:
    """Small Nature Neuroscience-style axis cleanup."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7, width=0.8, length=3)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def plot_average_response_figures(
    output_dir: Path,
    *,
    t_analysis_ms: float,
    stim_duration_ms: float,
    stim_region: int,
    n_trials: int,
) -> None:
    """Create publication-ready average-AAL response and PCI diagnostic figures."""
    b_items = list(B_STATES.items())
    trial_paths_by_b = {
        b_key: sorted((output_dir / "trials" / b_key).glob("trial_*.npz"))[:n_trials]
        for b_key, _cfg in b_items
    }
    missing = [b for b, paths in trial_paths_by_b.items() if len(paths) < n_trials]
    if missing:
        print(f"[plot] skipping full figures; missing trials for {missing}")
        return

    aligned_by_b = {
        b_key: load_aligned_delta(paths, t_analysis_ms=t_analysis_ms)
        for b_key, paths in trial_paths_by_b.items()
    }

    # Use a robust shared colour scale so panels are comparable without one
    # outlier flattening all other structure.
    all_abs = np.concatenate([np.abs(v[2]).ravel() for v in aligned_by_b.values()])
    vmax = float(np.nanpercentile(all_abs, 99.0))
    vmax = max(vmax, 1.0)

    colors = {
        "b005_control_wake": "#315F72",
        "b035_sleep_emcs": "#B4823A",
        "b055_mcs": "#A64E35",
        "b075_uws": "#5F4B66",
    }

    # Figure 1: regions sorted by peak-time, one row per b-state.
    fig, axes = plt.subplots(
        len(b_items),
        1,
        figsize=(4.8, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    if len(b_items) == 1:
        axes = [axes]
    im = None
    for ax, (b_key, cfg) in zip(axes, b_items):
        t_rel, _mean_rate, delta = aligned_by_b[b_key]
        post = delta[t_rel >= 0]
        t_post = t_rel[t_rel >= 0]
        peak_times = t_post[np.argmax(post, axis=0)]
        order = np.argsort(peak_times)
        if stim_region in order:
            order = np.r_[stim_region, order[order != stim_region]]
        im = ax.imshow(
            delta[:, order].T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[float(t_rel[0]), float(t_rel[-1]), 0, delta.shape[1]],
        )
        # The stimulated region is forced to row 0 (bottom row with
        # origin="lower").  Draw a red window so the reader can see it even
        # though the remaining regions are sorted by response timing.
        ax.add_patch(
            plt.Rectangle(
                (float(t_rel[0]), 0.0),
                float(t_rel[-1] - t_rel[0]),
                1.0,
                fill=False,
                edgecolor="#C41E3A",
                linewidth=1.2,
                zorder=5,
            )
        )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axvspan(0, stim_duration_ms, color="#E7C65A", alpha=0.22, lw=0)
        ax.set_ylabel(f"{cfg['label']}\nregions", fontsize=7)
        _nn_axes(ax)
    axes[-1].set_xlabel("Time from stimulation (ms)", fontsize=8)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Firing-rate change (Hz)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(output_dir / "fig02_b_state_propagation_peak_sorted.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig02_b_state_propagation_peak_sorted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: regional traces, one row per b-state.
    fig, axes = plt.subplots(
        len(b_items),
        1,
        figsize=(4.8, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(b_items) == 1:
        axes = [axes]
    for ax, (b_key, cfg) in zip(axes, b_items):
        t_rel, _mean_rate, delta = aligned_by_b[b_key]
        for reg in range(delta.shape[1]):
            if reg == stim_region:
                continue
            ax.plot(t_rel, delta[:, reg], color="#6D7776", lw=0.35, alpha=0.18)
        ax.plot(t_rel, delta[:, stim_region], color=colors[b_key], lw=1.7, label="stimulated region")
        ax.plot(t_rel, delta.mean(axis=1), color="black", lw=1.1, label="whole-brain mean")
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axvspan(0, stim_duration_ms, color="#E7C65A", alpha=0.22, lw=0)
        ax.set_ylabel(f"{cfg['label']}\nΔHz", fontsize=7)
        _nn_axes(ax)
    axes[-1].set_xlabel("Time from stimulation (ms)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7)
    fig.savefig(output_dir / "fig03_b_state_regional_traces.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig03_b_state_regional_traces.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: PCI and safety metrics.
    summary_path = output_dir / "average_summary.csv"
    if not summary_path.exists():
        return
    import pandas as pd  # local import keeps script startup lighter

    df = pd.read_csv(summary_path)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4), constrained_layout=True)
    metric_specs = [
        ("pci_mean", "PCI"),
        ("stim_peak_delta_hz", "Peak response (Hz)"),
        ("late_residual_stim_delta_hz", "Late residual (Hz)"),
    ]
    x = np.arange(len(b_items))
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        vals = []
        bar_colors = []
        for b_key, _cfg in b_items:
            row = df[df["b_key"] == b_key]
            vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            bar_colors.append(colors[b_key])
        ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(B_STATES[b]["b_e"])) for b, _ in b_items], fontsize=7)
        ax.set_xlabel(r"$b_e$ (pA)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.18, lw=0.5)
        _nn_axes(ax)
    fig.savefig(output_dir / "fig04_b_state_pci_and_safety.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig04_b_state_pci_and_safety.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_average_summary(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    labels = [r["b_label"] for r in summary_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    metrics = [
        ("pre_whole_mean_hz", "Pre-stim whole-brain mean (Hz)"),
        ("stim_peak_delta_hz", "Stim-region peak ΔHz"),
        ("late_residual_stim_delta_hz", "Late residual ΔHz"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        vals = [float(r.get(key, np.nan)) for r in summary_rows]
        ax.bar(x, vals, color=["#5B8A72", "#E8B56D", "#C5622F", "#8B6B8B"][: len(x)])
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.grid(alpha=0.2, axis="y")
    fig.savefig(output_dir / "fig_average_b_state_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stim_protocol_summary(
    summary_csv: Path,
    out_dir: Path,
    *,
    protocol_keys: list[str] | None = None,
    b_states: dict[str, dict[str, float | str]] | None = None,
) -> None:
    """Plot stimulation-calibration metrics as protocol x b-state heatmaps."""
    import pandas as pd

    df = pd.read_csv(summary_csv)
    b_states = STIM_CALIBRATION_B_STATES if b_states is None else b_states
    b_order = list(b_states)
    protocol_order = list(STIM_PROTOCOLS) if protocol_keys is None else protocol_keys
    protocol_labels = [str(STIM_PROTOCOLS[p]["label"]) for p in protocol_order]
    b_labels = [str(b_states[b]["label"]) for b in b_order]
    metrics = [
        ("stim_abs_peak_hz", "Peak rate (Hz)", "magma", 100.0),
        ("late_residual_stim_delta_hz", "Late residual (Hz)", "RdBu_r", None),
        ("pci_mean", "PCI", "viridis", None),
        ("explosive_abs_flag", "Explosive flag", "Reds", 1.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.0, 3.2), constrained_layout=True)
    for ax, (metric, title, cmap, vmax_fixed) in zip(axes, metrics):
        arr = np.full((len(protocol_order), len(b_order)), np.nan)
        for i, protocol in enumerate(protocol_order):
            for j, b_key in enumerate(b_order):
                row = df[(df["protocol_key"] == protocol) & (df["b_key"] == b_key)]
                if len(row):
                    arr[i, j] = float(row[metric].iloc[0])
        if metric == "late_residual_stim_delta_hz":
            vmax = max(1.0, float(np.nanpercentile(np.abs(arr), 95.0)))
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = vmax_fixed if vmax_fixed is not None else float(np.nanmax(arr))
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(len(b_labels)))
        ax.set_xticklabels(b_labels, fontsize=7)
        ax.set_yticks(np.arange(len(protocol_labels)))
        ax.set_yticklabels(protocol_labels if ax is axes[0] else [], fontsize=7)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if metric == "stim_abs_peak_hz" else "black")
        cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        _nn_axes(ax)
    fig.savefig(out_dir / "fig_stim_protocol_summary_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_stim_protocol_summary_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def run_stim_protocol_sweep(args: argparse.Namespace) -> None:
    """Calibrate stimulation amplitude/duration on average AAL90 across b=5/35/40."""
    run_stim_protocol_sweep_impl(
        args,
        out_name="stim_protocol_sweep",
        b_states=STIM_CALIBRATION_B_STATES,
        protocol_keys=list(STIM_PROTOCOLS),
    )


def run_stim_protocol_high_b_sweep(args: argparse.Namespace) -> None:
    """Calibrate non-explosive protocols on average AAL90 across b=55/75."""
    protocol_keys = [p for p in STIM_PROTOCOLS if p != "amp00050_dur50"]
    run_stim_protocol_sweep_impl(
        args,
        out_name="stim_protocol_sweep_high_b",
        b_states=STIM_CALIBRATION_HIGH_B_STATES,
        protocol_keys=protocol_keys,
    )


def run_stim_protocol_sweep_impl(
    args: argparse.Namespace,
    *,
    out_name: str,
    b_states: dict[str, dict[str, float | str]],
    protocol_keys: list[str],
) -> None:
    """Shared implementation for stimulation-calibration sweeps."""
    avg_root = PROJECT_ROOT / "data" / "connectivity" / "average_aal90"
    weights, lengths, labels = load_average_aal90(avg_root)
    scenario = SCENARIOS[args.scenario]
    out_dir = args.output_root / out_name
    trial_root = out_dir / "trials"
    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []

    for protocol_key in protocol_keys:
        protocol = STIM_PROTOCOLS[protocol_key]
        for b_key, b_cfg in b_states.items():
            combo_key = f"{protocol_key}_{b_key}"
            for trial_seed in range(int(args.calibration_trials)):
                tasks.append({
                    "output_dir": trial_root,
                    "b_key": combo_key,
                    "b_e": float(b_cfg["b_e"]),
                    "trial_seed": trial_seed,
                    "weights": weights,
                    "tract_lengths": lengths,
                    "region_labels": labels,
                    "scenario_key": args.scenario,
                    "noise_alpha": float(scenario["noise_alpha"]),
                    "shared_noise_mode": str(scenario["shared_noise_mode"]),
                    "transient_ms": float(args.transient_ms),
                    "t_analysis_ms": float(args.t_analysis_ms),
                    "total_sim_ms": float(args.total_sim_ms),
                    "stim_onset_ms": float(args.stim_onset_ms),
                    "stim_amplitude": float(protocol["amplitude"]),
                    "stim_duration_ms": float(protocol["duration_ms"]),
                    "stim_region": list(args.stim_region),
                    "rate_monitor_period_ms": float(RATE_MONITOR_PERIOD_MS_OLD),
                    "overwrite": bool(args.overwrite),
                    "protocol_key": protocol_key,
                    "protocol_label": str(protocol["label"]),
                    "plain_b_key": b_key,
                    "b_label": str(b_cfg["label"]),
                })

    workers = max(1, int(args.calibration_workers))
    print(f"[stim-sweep] queued {len(tasks)} trials on {workers} workers -> {out_dir}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(
                run_average_aal_trial,
                output_dir=task["output_dir"],
                b_key=task["b_key"],
                b_e=task["b_e"],
                trial_seed=task["trial_seed"],
                weights=task["weights"],
                tract_lengths=task["tract_lengths"],
                region_labels=task["region_labels"],
                scenario_key=task["scenario_key"],
                noise_alpha=task["noise_alpha"],
                shared_noise_mode=task["shared_noise_mode"],
                transient_ms=task["transient_ms"],
                t_analysis_ms=task["t_analysis_ms"],
                total_sim_ms=task["total_sim_ms"],
                stim_onset_ms=task["stim_onset_ms"],
                stim_amplitude=task["stim_amplitude"],
                stim_duration_ms=task["stim_duration_ms"],
                stim_region=task["stim_region"],
                rate_monitor_period_ms=task["rate_monitor_period_ms"],
                overwrite=task["overwrite"],
            ): task
            for task in tasks
        }
        for fut in as_completed(future_to_task):
            task = future_to_task[fut]
            res = fut.result()
            row = {
                "mode": "stim_protocol_sweep",
                "protocol_key": task["protocol_key"],
                "protocol_label": task["protocol_label"],
                "stim_amplitude": task["stim_amplitude"],
                "stim_duration_ms": task["stim_duration_ms"],
                "b_key": task["plain_b_key"],
                "b_label": task["b_label"],
                "b_e": task["b_e"],
                "trial_seed": task["trial_seed"],
                "combo_key": task["b_key"],
                "save_path": res["save_path"],
                "runtime_s": res["runtime_s"],
                "skipped_existing": res["skipped_existing"],
            }
            rows.append(row)
            print(
                f"[stim-sweep] {task['protocol_key']} {task['plain_b_key']} "
                f"trial {task['trial_seed']:03d} runtime={res['runtime_s']:.1f}s skip={res['skipped_existing']}"
            )
    rows = sorted(rows, key=lambda r: (str(r["protocol_key"]), str(r["b_key"]), int(r["trial_seed"])))
    write_csv(out_dir / "stim_protocol_trial_index.csv", rows)

    summary: list[dict[str, Any]] = []
    for protocol_key in protocol_keys:
        protocol = STIM_PROTOCOLS[protocol_key]
        for b_key, b_cfg in b_states.items():
            combo_key = f"{protocol_key}_{b_key}"
            paths = sorted((trial_root / combo_key).glob("trial_*.npz"))[: int(args.calibration_trials)]
            s = summarize_trials(paths, float(args.t_analysis_ms), int(args.stim_region[0]))
            s.update({
                "mode": "stim_protocol_sweep",
                "protocol_key": protocol_key,
                "protocol_label": str(protocol["label"]),
                "stim_amplitude": float(protocol["amplitude"]),
                "stim_duration_ms": float(protocol["duration_ms"]),
                "b_key": b_key,
                "b_label": str(b_cfg["label"]),
                "b_e": float(b_cfg["b_e"]),
                "combo_key": combo_key,
            })
            summary.append(s)
    summary_csv = out_dir / "stim_protocol_summary.csv"
    write_csv(summary_csv, summary)
    plot_stim_protocol_summary(summary_csv, out_dir, protocol_keys=protocol_keys, b_states=b_states)
    print(f"[stim-sweep] wrote {out_dir}")


def run_average_aal90(args: argparse.Namespace) -> None:
    avg_root = PROJECT_ROOT / "data" / "connectivity" / "average_aal90"
    weights, lengths, labels = load_average_aal90(avg_root)
    scenario = SCENARIOS[args.scenario]
    out_dir = args.output_root / "average_aal90"
    rows: list[dict[str, Any]] = []
    for b_key, cfg in B_STATES.items():
        for trial_seed in range(int(args.n_trials)):
            res = run_average_aal_trial(
                output_dir=out_dir / "trials",
                b_key=b_key,
                b_e=float(cfg["b_e"]),
                trial_seed=trial_seed,
                weights=weights,
                tract_lengths=lengths,
                region_labels=labels,
                scenario_key=args.scenario,
                noise_alpha=float(scenario["noise_alpha"]),
                shared_noise_mode=str(scenario["shared_noise_mode"]),
                transient_ms=float(args.transient_ms),
                t_analysis_ms=float(args.t_analysis_ms),
                total_sim_ms=float(args.total_sim_ms),
                stim_onset_ms=float(args.stim_onset_ms),
                stim_amplitude=float(args.stim_amplitude),
                stim_duration_ms=float(args.stim_duration_ms),
                stim_region=list(args.stim_region),
                rate_monitor_period_ms=float(RATE_MONITOR_PERIOD_MS_OLD),
                overwrite=bool(args.overwrite),
            )
            rows.append({
                "mode": "average_aal90",
                "b_key": b_key,
                "b_label": cfg["label"],
                "b_e": float(cfg["b_e"]),
                "trial_seed": trial_seed,
                "save_path": res["save_path"],
                "runtime_s": res["runtime_s"],
                "skipped_existing": res["skipped_existing"],
            })
            print(f"[average] {b_key} trial {trial_seed:03d} runtime={res['runtime_s']:.1f}s skip={res['skipped_existing']}")
    write_csv(out_dir / "average_trial_index.csv", rows)

    summary: list[dict[str, Any]] = []
    for b_key, cfg in B_STATES.items():
        paths = sorted((out_dir / "trials" / b_key).glob("trial_*.npz"))[: int(args.n_trials)]
        s = summarize_trials(paths, float(args.t_analysis_ms), int(args.stim_region[0]))
        s.update({"mode": "average_aal90", "b_key": b_key, "b_label": cfg["label"], "b_e": float(cfg["b_e"])})
        summary.append(s)
    write_csv(out_dir / "average_summary.csv", summary)
    plot_average_summary(out_dir, summary)
    plot_average_response_figures(
        out_dir,
        t_analysis_ms=float(args.t_analysis_ms),
        stim_duration_ms=float(args.stim_duration_ms),
        stim_region=int(args.stim_region[0]),
        n_trials=int(args.n_trials),
    )
    print(f"[average] wrote {out_dir}")


def run_subject_smoke(args: argparse.Namespace) -> None:
    scenario = SCENARIOS[args.scenario]
    jobs = get_subject_jobs(DATASET_ROOT)
    chosen = []
    for cohort in ["control", "emcs", "mcs", "uws"]:
        cohort_jobs = [j for j in jobs if j.cohort == cohort]
        if cohort_jobs:
            chosen.append(cohort_jobs[0])
    out_dir = args.output_root / "subject_smoke"
    rows: list[dict[str, Any]] = []
    for sj in chosen:
        b_key, b_e = CONDITION_B[sj.cohort]
        b_model = deepcopy(BASE_PARAMETER_MODEL_NEW)
        b_model["b_e"] = float(b_e)
        subj_out = out_dir / "trials" / b_key / sj.cohort / sj.subject_id
        for trial_seed in range(int(args.smoke_trials)):
            expected = subj_out / f"trial_{trial_seed:03d}.npz"
            if expected.exists() and not args.overwrite:
                res = {"save_path": str(expected), "runtime_s": 0.0, "skipped_existing": True}
            else:
                t0 = perf_counter()
                r = run_pci_trial_job(
                    scenario_key=args.scenario,
                    noise_alpha=float(scenario["noise_alpha"]),
                    shared_noise_mode=str(scenario["shared_noise_mode"]),
                    cohort=sj.cohort,
                    subject_id=sj.subject_id,
                    trial_seed=int(trial_seed),
                    dataset_root=str(DATASET_ROOT),
                    output_dir=str(subj_out),
                    transient_ms=float(args.transient_ms),
                    t_analysis_ms=float(args.t_analysis_ms),
                    rate_monitor_period_ms=float(RATE_MONITOR_PERIOD_MS_OLD),
                    base_parameter_model=deepcopy(b_model),
                    stim_amplitude=float(args.stim_amplitude),
                    stim_duration_ms=float(args.stim_duration_ms),
                    stim_region=list(args.stim_region),
                    stim_onset_ms=float(args.stim_onset_ms),
                    total_sim_ms=float(args.total_sim_ms),
                )
                res = {"save_path": r["save_path"], "runtime_s": float(perf_counter() - t0), "skipped_existing": False}
            rows.append({
                "mode": "subject_smoke",
                "cohort": sj.cohort,
                "subject_id": sj.subject_id,
                "condition": sj.condition,
                "b_key": b_key,
                "b_e": float(b_e),
                "trial_seed": trial_seed,
                **res,
            })
            print(f"[subject] {sj.cohort}/{sj.subject_id} {b_key} trial {trial_seed:03d} runtime={res['runtime_s']:.1f}s skip={res['skipped_existing']}")
    write_csv(out_dir / "subject_smoke_trial_index.csv", rows)

    summary: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["save_path"])
        s = summarize_trials([path], float(args.t_analysis_ms), int(args.stim_region[0]))
        s.update({k: row[k] for k in ["cohort", "subject_id", "condition", "b_key", "b_e"]})
        summary.append(s)
    write_csv(out_dir / "subject_smoke_summary.csv", summary)
    print(f"[subject] wrote {out_dir}")


def aggregate_subject_summary(summary_csv: Path, out_csv: Path) -> None:
    """Aggregate per-trial subject summaries into one row per subject."""
    import pandas as pd

    df = pd.read_csv(summary_csv)
    group_cols = ["cohort", "subject_id", "condition", "b_key", "b_e"]
    agg = df.groupby(group_cols, sort=False).agg(
        n_trials=("pci_mean", "size"),
        pci_mean=("pci_mean", "mean"),
        pci_sd=("pci_mean", "std"),
        pre_whole_mean_hz=("pre_whole_mean_hz", "mean"),
        pre_whole_std_hz=("pre_whole_std_hz", "mean"),
        stim_peak_delta_hz_mean=("stim_peak_delta_hz", "mean"),
        stim_peak_delta_hz_max=("stim_peak_delta_hz", "max"),
        late_residual_mean=("late_residual_stim_delta_hz", "mean"),
        stim_abs_peak_hz_mean=("stim_abs_peak_hz", "mean"),
        stim_abs_peak_hz_max=("stim_abs_peak_hz", "max"),
        explosive_trials=("explosive_abs_flag", "sum"),
        poor_recovery_trials=("poor_recovery_flag", "sum"),
    ).reset_index()
    agg.to_csv(out_csv, index=False)


def plot_subject_aggregate(summary_csv: Path, out_dir: Path, *, prefix: str) -> None:
    """Create compact subject-level PCI/safety figure."""
    import pandas as pd

    df = pd.read_csv(summary_csv)
    order = ["control", "emcs", "mcs", "uws"]
    labels = {
        "control": "CNT",
        "emcs": "EMCS",
        "mcs": "MCS",
        "uws": "UWS",
    }
    colors = {
        "control": "#315F72",
        "emcs": "#B4823A",
        "mcs": "#A64E35",
        "uws": "#5F4B66",
    }
    df["cohort"] = np.asarray(df["cohort"], dtype=str)
    df = df.set_index("cohort").loc[[c for c in order if c in set(df["cohort"] if "cohort" in df else df.index)]].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    metric_specs = [
        ("pci_mean", "PCI"),
        ("stim_abs_peak_hz_mean", "Peak response (Hz)"),
        ("poor_recovery_trials", "Poor-recovery trials"),
    ]
    x = np.arange(len(df))
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        vals = np.asarray(df[metric], dtype=float)
        ax.bar(x, vals, color=[colors[c] for c in df["cohort"]], edgecolor="black", linewidth=0.5)
        if metric == "pci_mean" and "pci_sd" in df:
            ax.errorbar(x, vals, yerr=np.asarray(df["pci_sd"], dtype=float), fmt="none", ecolor="black", lw=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in df["cohort"]], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.18, lw=0.5)
        _nn_axes(ax)
    fig.savefig(out_dir / f"{prefix}_pci_and_safety.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{prefix}_pci_and_safety.pdf", bbox_inches="tight")
    plt.close(fig)


def _subject_plot_rows(summary_csv: Path) -> list[dict[str, Any]]:
    """Return subject rows in a stable clinical-condition order."""
    import pandas as pd

    order = ["control", "emcs", "mcs", "uws"]
    df = pd.read_csv(summary_csv)
    df["cohort"] = np.asarray(df["cohort"], dtype=str)
    present = set(df["cohort"])
    df = df.set_index("cohort").loc[[c for c in order if c in present]].reset_index()
    return df.to_dict("records")


def plot_subject_response_figures(
    output_dir: Path,
    *,
    summary_csv: Path,
    t_analysis_ms: float,
    stim_duration_ms: float,
    n_trials: int,
    prefix: str,
) -> None:
    """Create subject-specific propagation heatmaps and response traces."""
    rows = _subject_plot_rows(summary_csv)
    labels = {
        "control": "CNT",
        "emcs": "EMCS",
        "mcs": "MCS",
        "uws": "UWS",
    }
    colors = {
        "control": "#315F72",
        "emcs": "#B4823A",
        "mcs": "#A64E35",
        "uws": "#5F4B66",
    }
    aligned: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, int]] = []
    for row in rows:
        trial_dir = output_dir / "trials" / str(row["b_key"]) / str(row["cohort"]) / str(row["subject_id"])
        trial_paths = sorted(trial_dir.glob("trial_*.npz"))[:n_trials]
        if not trial_paths:
            print(f"[plot-subject] no trials found in {trial_dir}")
            continue
        t_rel, mean_rate, delta = load_aligned_delta(trial_paths, t_analysis_ms=t_analysis_ms)
        d0 = np.load(trial_paths[0], allow_pickle=True)
        stim_region = int(np.asarray(d0["stim_region"]).ravel()[0])
        aligned.append((row, t_rel, mean_rate, delta, stim_region))

    if not aligned:
        print("[plot-subject] no subject figures written; no aligned trials.")
        return

    all_abs = np.concatenate([np.abs(delta).ravel() for _row, _t, _mean, delta, _stim in aligned])
    vmax = float(np.nanpercentile(all_abs, 99.0))
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(
        len(aligned),
        1,
        figsize=(4.8, max(2.0, 1.7 * len(aligned))),
        sharex=True,
        constrained_layout=True,
    )
    if len(aligned) == 1:
        axes = [axes]
    im = None
    for ax, (row, t_rel, _mean_rate, delta, stim_region) in zip(axes, aligned):
        post = delta[t_rel >= 0]
        t_post = t_rel[t_rel >= 0]
        peak_times = t_post[np.argmax(post, axis=0)]
        order = np.argsort(peak_times)
        if stim_region in order:
            order = np.r_[stim_region, order[order != stim_region]]
        im = ax.imshow(
            delta[:, order].T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[float(t_rel[0]), float(t_rel[-1]), 0, delta.shape[1]],
        )
        ax.add_patch(
            plt.Rectangle(
                (float(t_rel[0]), 0.0),
                float(t_rel[-1] - t_rel[0]),
                1.0,
                fill=False,
                edgecolor="#C41E3A",
                linewidth=1.2,
                zorder=5,
            )
        )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axvspan(0, stim_duration_ms, color="#E7C65A", alpha=0.22, lw=0)
        cohort = str(row["cohort"])
        subject_id = str(row["subject_id"])
        ax.set_ylabel(f"{labels.get(cohort, cohort)}\n{subject_id}", fontsize=7)
        _nn_axes(ax)
    axes[-1].set_xlabel("Time from stimulation (ms)", fontsize=8)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Firing-rate change (Hz)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(output_dir / f"{prefix}_propagation_peak_sorted.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{prefix}_propagation_peak_sorted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        len(aligned),
        1,
        figsize=(4.8, max(2.0, 1.55 * len(aligned))),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(aligned) == 1:
        axes = [axes]
    for ax, (row, t_rel, _mean_rate, delta, stim_region) in zip(axes, aligned):
        cohort = str(row["cohort"])
        for reg in range(delta.shape[1]):
            if reg == stim_region:
                continue
            ax.plot(t_rel, delta[:, reg], color="#6D7776", lw=0.35, alpha=0.18)
        ax.plot(t_rel, delta[:, stim_region], color=colors.get(cohort, "#315F72"), lw=1.7, label="stimulated region")
        ax.plot(t_rel, delta.mean(axis=1), color="black", lw=1.1, label="whole-brain mean")
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axvspan(0, stim_duration_ms, color="#E7C65A", alpha=0.22, lw=0)
        ax.set_ylabel(f"{labels.get(cohort, cohort)}\nΔHz", fontsize=7)
        _nn_axes(ax)
    axes[-1].set_xlabel("Time from stimulation (ms)", fontsize=8)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, fontsize=7)
    fig.savefig(output_dir / f"{prefix}_regional_traces.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{prefix}_regional_traces.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _first_condition_subjects() -> list[Any]:
    jobs = get_subject_jobs(DATASET_ROOT)
    chosen = []
    for cohort in ["control", "emcs", "mcs", "uws"]:
        cohort_jobs = [j for j in jobs if j.cohort == cohort]
        if cohort_jobs:
            chosen.append(cohort_jobs[0])
    return chosen


def run_subject_shared_b(args: argparse.Namespace) -> None:
    """Run one real subject per condition with identical b_e for all subjects."""
    scenario = SCENARIOS[args.scenario]
    chosen = _first_condition_subjects()
    b_val = float(args.shared_b)
    b_key = f"shared_b{int(round(b_val)):03d}" if abs(b_val - round(b_val)) < 1e-12 else f"shared_b{str(b_val).replace('.', 'p')}"
    out_dir = args.output_root / f"subject_shared_b_{b_key.replace('shared_', '')}"
    rows: list[dict[str, Any]] = []

    for sj in chosen:
        b_model = deepcopy(BASE_PARAMETER_MODEL_NEW)
        b_model["b_e"] = b_val
        subj_out = out_dir / "trials" / b_key / sj.cohort / sj.subject_id
        for trial_seed in range(int(args.shared_b_trials)):
            expected = subj_out / f"trial_{trial_seed:03d}.npz"
            if expected.exists() and not args.overwrite:
                res = {"save_path": str(expected), "runtime_s": 0.0, "skipped_existing": True}
            else:
                t0 = perf_counter()
                r = run_pci_trial_job(
                    scenario_key=args.scenario,
                    noise_alpha=float(scenario["noise_alpha"]),
                    shared_noise_mode=str(scenario["shared_noise_mode"]),
                    cohort=sj.cohort,
                    subject_id=sj.subject_id,
                    trial_seed=int(trial_seed),
                    dataset_root=str(DATASET_ROOT),
                    output_dir=str(subj_out),
                    transient_ms=float(args.transient_ms),
                    t_analysis_ms=float(args.t_analysis_ms),
                    rate_monitor_period_ms=float(RATE_MONITOR_PERIOD_MS_OLD),
                    base_parameter_model=deepcopy(b_model),
                    stim_amplitude=float(args.stim_amplitude),
                    stim_duration_ms=float(args.stim_duration_ms),
                    stim_region=list(args.stim_region),
                    stim_onset_ms=float(args.stim_onset_ms),
                    total_sim_ms=float(args.total_sim_ms),
                )
                res = {"save_path": r["save_path"], "runtime_s": float(perf_counter() - t0), "skipped_existing": False}
            rows.append({
                "mode": "subject_shared_b",
                "cohort": sj.cohort,
                "subject_id": sj.subject_id,
                "condition": sj.condition,
                "b_key": b_key,
                "b_e": b_val,
                "trial_seed": trial_seed,
                **res,
            })
            print(
                f"[shared-b] {sj.cohort}/{sj.subject_id} b={b_val:g} "
                f"trial {trial_seed:03d} runtime={res['runtime_s']:.1f}s skip={res['skipped_existing']}"
            )
    write_csv(out_dir / "subject_shared_b_trial_index.csv", rows)

    summary: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["save_path"])
        s = summarize_trials([path], float(args.t_analysis_ms), int(args.stim_region[0]))
        s.update({k: row[k] for k in ["cohort", "subject_id", "condition", "b_key", "b_e"]})
        summary.append(s)
    summary_csv = out_dir / "subject_shared_b_summary.csv"
    aggregate_csv = out_dir / "subject_shared_b_summary_aggregate.csv"
    write_csv(summary_csv, summary)
    aggregate_subject_summary(summary_csv, aggregate_csv)
    plot_subject_aggregate(aggregate_csv, out_dir, prefix="fig_subject_shared_b")
    plot_subject_response_figures(
        out_dir,
        summary_csv=aggregate_csv,
        t_analysis_ms=float(args.t_analysis_ms),
        stim_duration_ms=float(args.stim_duration_ms),
        n_trials=int(args.shared_b_trials),
        prefix="fig_subject_shared_b",
    )
    print(f"[shared-b] wrote {out_dir}")


def plot_existing_subject_shared_b(args: argparse.Namespace) -> None:
    """Plot already-completed shared-b subject trials without rerunning them."""
    b_val = float(args.shared_b)
    b_key = f"shared_b{int(round(b_val)):03d}" if abs(b_val - round(b_val)) < 1e-12 else f"shared_b{str(b_val).replace('.', 'p')}"
    out_dir = args.output_root / f"subject_shared_b_{b_key.replace('shared_', '')}"
    aggregate_csv = out_dir / "subject_shared_b_summary_aggregate.csv"
    if not aggregate_csv.exists():
        raise FileNotFoundError(f"Missing aggregate summary: {aggregate_csv}")
    plot_subject_response_figures(
        out_dir,
        summary_csv=aggregate_csv,
        t_analysis_ms=float(args.t_analysis_ms),
        stim_duration_ms=float(args.stim_duration_ms),
        n_trials=int(args.shared_b_trials),
        prefix="fig_subject_shared_b",
    )
    print(f"[plot-subject-shared-b] wrote propagation figures in {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=(
            "average_aal90",
            "subject_smoke",
            "subject_shared_b",
            "plot_subject_shared_b",
            "stim_protocol_sweep",
            "stim_protocol_high_b_sweep",
            "both",
        ),
        default="both",
    )
    p.add_argument("--output-root", type=Path, default=doc_liege_results("notebooks_outputs", "08_pci_b_state_test"))
    p.add_argument("--scenario", default="private_alpha0")
    p.add_argument("--n-trials", type=int, default=25, help="Trials per b-state for average_aal90 mode.")
    p.add_argument("--smoke-trials", type=int, default=1, help="Trials per subject for subject_smoke mode.")
    p.add_argument("--shared-b", type=float, default=7.0, help="Shared b_e for subject_shared_b mode.")
    p.add_argument("--shared-b-trials", type=int, default=10, help="Trials per subject for subject_shared_b mode.")
    p.add_argument("--calibration-trials", type=int, default=3, help="Trials per protocol x b-state in stim_protocol_sweep mode.")
    p.add_argument("--calibration-workers", type=int, default=3, help="Parallel workers for stim_protocol_sweep mode.")
    p.add_argument("--transient-ms", type=float, default=4000.0)
    p.add_argument("--t-analysis-ms", type=float, default=300.0)
    p.add_argument("--stim-onset-ms", type=float, default=4300.0)
    p.add_argument("--total-sim-ms", type=float, default=4700.0)
    p.add_argument("--stim-amplitude", type=float, default=0.0003)
    p.add_argument("--stim-duration-ms", type=float, default=10.0)
    p.add_argument("--stim-region", type=int, nargs="+", default=[18])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode in ("average_aal90", "both"):
        run_average_aal90(args)
    if args.mode in ("subject_smoke", "both"):
        run_subject_smoke(args)
    if args.mode == "subject_shared_b":
        run_subject_shared_b(args)
    if args.mode == "plot_subject_shared_b":
        plot_existing_subject_shared_b(args)
    if args.mode == "stim_protocol_sweep":
        run_stim_protocol_sweep(args)
    if args.mode == "stim_protocol_high_b_sweep":
        run_stim_protocol_high_b_sweep(args)


if __name__ == "__main__":
    main()
