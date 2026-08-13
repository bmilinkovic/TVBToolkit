#!/usr/bin/env python3
"""Calibrate left-SMA PCI stimulation, global coupling, and slow homeostasis.

This is deliberately a small-subject *calibration*, not a cohort analysis. It
uses an outcome-blind structural-damage panel, the anatomically resolved left
supplementary motor area, matched seeds, and configurable 5-HT2A
occupancy.  It first compares pulse waveform/duration/amplitude at a reference
G, then sweeps G using the automatically selected safe pulse.  An optional
final stage learns regional inhibitory conductance during unstimulated epochs,
freezes it, and repeats the selected perturbation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from brain_act_hybrid_common import (
    COND_COLORS,
    DATASET_ROOT,
    SCENARIOS,
)

from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial
from tvbtoolkit.complexity.pci_st import PCIStResult, pci_st_from_trials
from tvbtoolkit.datasets.brain_act import load_aal90_atlas, load_subject_structural
from tvbtoolkit.whole_brain.homeostasis import (
    InhibitoryHomeostasisConfig,
    baseline_relative_activation_threshold,
    update_inhibitory_conductance,
)
from tvbtoolkit.whole_brain.simulation import (
    GaussianPulse,
    RaisedCosinePulse,
    run_whole_brain_simulation,
)
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import _apply_damage_parity
from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "results" / "pci_stimulus_g_calibration",
    )
    p.add_argument(
        "--subject",
        action="append",
        default=None,
        help="cohort:subject_id; supply one control, one MCS, and one UWS",
    )
    p.add_argument("--occupancies", type=float, nargs="+", default=[0.0, 0.766])
    p.add_argument("--trial-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--pulse-shapes",
        nargs="+",
        choices=["square", "raised_cosine", "gaussian"],
        default=["square", "raised_cosine", "gaussian"],
    )
    p.add_argument("--durations-ms", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    p.add_argument(
        "--amplitudes-khz",
        type=float,
        nargs="+",
        default=[0.00010, 0.00020, 0.00030, 0.00050],
    )
    p.add_argument(
        "--g-values",
        type=float,
        nargs="+",
        default=[0.30, 0.40, 0.50, 0.60, 0.75, 1.00],
    )
    p.add_argument(
        "--reference-g",
        type=float,
        default=0.55,
        help=(
            "Reference used during pulse calibration. For cohort-global-max "
            "SC, 0.55 is only a first-order starting estimate; select final G "
            "from the calibration results."
        ),
    )
    p.add_argument(
        "--shared-b-e",
        type=float,
        default=None,
        help=(
            "Optional common b_e value. Omit to retain the prespecified "
            "diagnosis gradient while G is swept."
        ),
    )
    p.add_argument("--transient-ms", type=float, default=4000.0)
    p.add_argument("--analysis-ms", type=float, default=300.0)
    p.add_argument(
        "--monitor-period-ms",
        type=float,
        default=3.0,
        help="3 ms gives exact 333.33 Hz with dt=0.1 ms",
    )
    p.add_argument("--response-start-ms", type=float, default=8.0)
    p.add_argument("--pci-permutation-replicates", type=int, default=1000)
    p.add_argument("--pci-alpha", type=float, default=0.05)
    p.add_argument("--pci-min-source-entropy", type=float, default=0.08)
    p.add_argument("--pci-st-k", type=float, default=1.2)
    p.add_argument("--pci-st-min-snr", type=float, default=1.1)
    p.add_argument("--pci-st-max-var-percent", type=float, default=99.0)
    p.add_argument("--pci-st-n-steps", type=int, default=100)
    p.add_argument("--saturation-hz", type=float, default=100.0)
    p.add_argument("--stim-region-label", default=pilot.DEFAULT_STIM_REGION_LABEL)
    p.add_argument(
        "--receptor-tracer", choices=["cimbi", "savli", "talbot"], default="cimbi"
    )
    p.add_argument("--receptor-csv", type=Path, default=pilot.DEFAULT_RECEPTOR_CSV)
    p.add_argument("--e-l-e-drug", type=float, default=-61.2)
    p.add_argument("--e-l-i-drug", type=float, default=-64.4)
    p.add_argument("--homeostasis", choices=["off", "compare"], default="compare")
    p.add_argument(
        "--homeostatic-target", choices=["baseline", "fixed"], default="baseline"
    )
    p.add_argument("--homeostatic-target-hz", type=float, default=2.5)
    p.add_argument("--homeostatic-epochs", type=int, default=6)
    p.add_argument("--homeostatic-epoch-ms", type=float, default=1000.0)
    p.add_argument("--homeostatic-tau-s", type=float, default=2.0)
    p.add_argument("--homeostatic-detector-tau-ms", type=float, default=50.0)
    p.add_argument(
        "--homeostatic-activation-sd",
        type=float,
        default=5.0,
        help=(
            "Activate online inhibition per region above its unstimulated "
            "baseline mean plus this many baseline SDs."
        ),
    )
    p.add_argument("--homeostatic-post-ms", type=float, default=1500.0)
    p.add_argument("--homeostatic-base-qi-ns", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument(
        "--quick-local",
        action="store_true",
        help="Use a reduced one-seed grid suitable for a workstation smoke analysis.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.quick_local:
        args.trial_seeds = [0]
        args.durations_ms = [1.0, 5.0, 10.0]
        args.amplitudes_khz = [0.00020, 0.00030]
        args.g_values = [0.175, 0.25, 0.325]
        args.homeostatic_epochs = 3
        args.homeostatic_post_ms = 1000.0
    return args


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


def _validate(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.receptor_csv = args.receptor_csv.expanduser().resolve()
    if not (args.dataset_root / "index.json").is_file():
        raise FileNotFoundError(
            f"Missing converted dataset index: {args.dataset_root / 'index.json'}"
        )
    dataset_index = json.loads(
        (args.dataset_root / "index.json").read_text(encoding="utf-8")
    )
    normalization = dict(dataset_index.get("connectivity_normalization", {}))
    args.structural_connectivity_normalization = str(
        normalization.get("scheme", "legacy_column_sum")
    )
    args.structural_connectivity_normalization_divisor = normalization.get(
        "divisor"
    )
    args.simulator_connectivity_normalization = (
        "none"
        if args.structural_connectivity_normalization
        in {"cohort_global_max", "native_invnodevol", "native_raw"}
        else "legacy_column_sum"
    )
    if any(x <= 0 for x in args.durations_ms + args.amplitudes_khz + args.g_values):
        raise ValueError("Durations, amplitudes, and G values must be positive.")
    if len(args.trial_seeds) != 10:
        raise ValueError("The seven-subject calibration requires exactly 10 trials.")
    if not 0.0 < float(args.pci_alpha) < 1.0:
        raise ValueError("--pci-alpha must lie between zero and one.")
    if args.monitor_period_ms / 0.1 != round(args.monitor_period_ms / 0.1):
        raise ValueError(
            "Monitor period must be an exact multiple of the 0.1-ms integrator step."
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least one.")
    if args.homeostatic_activation_sd <= 0.0:
        raise ValueError("--homeostatic-activation-sd must be positive.")


def _subjects(args: argparse.Namespace):
    if args.subject:
        selected = pilot._select_subjects(
            args.dataset_root,
            ["control", "emcs", "mcs", "uws"],
            10**9,
            explicit_subjects=args.subject,
        )
        if len(selected) != 7:
            raise ValueError("Explicit calibration selection must contain 7 subjects.")
        return selected
    selected = pilot._select_subjects(
        args.dataset_root,
        ["control", "mcs", "uws"],
        1,
        explicit_subjects=args.subject,
    )
    cohorts = [job.cohort for job in selected]
    if sorted(cohorts) != ["control", "mcs", "uws"]:
        raise ValueError(
            "Calibration requires exactly one control, one MCS, and one UWS subject."
        )
    return selected


def _subject_selection_metadata(args: argparse.Namespace, jobs) -> list[dict[str, Any]]:
    """Describe the outcome-blind calibration panel from dataset provenance."""
    index = json.loads((args.dataset_root / "index.json").read_text(encoding="utf-8"))
    metadata = {
        (str(item["cohort"]), str(item["subject_id"])): item
        for item in index["subjects"]
    }
    records = []
    for job in jobs:
        weights, _, _, _ = load_subject_structural(
            subject_id=job.subject_id,
            cohort=job.cohort,
            dataset_root=args.dataset_root,
            validate=True,
            enforce_symmetry=True,
            zero_diagonal=True,
        )
        upper = np.triu_indices(weights.shape[0], k=1)
        zero_percent = 100.0 * float(np.mean(np.asarray(weights)[upper] <= 0.0))
        item = metadata[(job.cohort, job.subject_id)]
        records.append(
            {
                "cohort": job.cohort,
                "condition": job.condition,
                "subject_id": job.subject_id,
                "stage": str(item.get("stage", "")),
                "sedation": str(item.get("sedation", "")),
                "zero_masked_connections_percent": zero_percent,
                "selection_basis": (
                    "outcome-blind structural calibration; sedation pair "
                    "matched approximately within diagnosis"
                ),
            }
        )
    return records


def _pulse_values(
    shape: str, duration_ms: float, dt_ms: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(-0.2 * duration_ms, 1.2 * duration_ms + dt_ms / 2, dt_ms)
    if shape == "square":
        value = ((time > 0.0) & (time < duration_ms)).astype(float)
    else:
        pulse = RaisedCosinePulse() if shape == "raised_cosine" else GaussianPulse()
        pulse.parameters.update({"onset": 0.0, "tau": duration_ms, "amp": 1.0})
        value = np.asarray(pulse.evaluate(time), dtype=float)
    return time, value


def _base_model(
    job,
    occupancy: float,
    receptor: np.ndarray,
    args: argparse.Namespace,
    q_i: np.ndarray | None = None,
    homeostasis_target_khz: np.ndarray | None = None,
    homeostasis_activation_khz: np.ndarray | None = None,
) -> dict[str, Any]:
    shared_b_e = getattr(args, "shared_b_e", None)
    proxy = argparse.Namespace(
        b_e_override=(None if shared_b_e is None else float(shared_b_e)),
        split_model_all_occupancies=True,
        e_l_e_drug=args.e_l_e_drug,
        e_l_i_drug=args.e_l_i_drug,
    )
    model = pilot._build_parameter_model(job.condition, occupancy, receptor, proxy)
    scenario = SCENARIOS["private_alpha0"]
    model.update(
        noise_alpha=scenario["noise_alpha"],
        shared_noise_mode=scenario["shared_noise_mode"],
    )
    if q_i is not None:
        model["Q_i_e"] = np.asarray(q_i, dtype=float).tolist()
    if homeostasis_target_khz is not None:
        if homeostasis_activation_khz is None:
            raise ValueError(
                "Online homeostasis requires a regional activation threshold."
            )
        model.update(
            homeostasis_target_rate=np.asarray(
                homeostasis_target_khz, dtype=float
            ).tolist(),
            homeostasis_detector_tau_ms=float(args.homeostatic_detector_tau_ms),
            homeostasis_tau_ms=float(args.homeostatic_tau_s) * 1000.0,
            homeostasis_activation_rate=np.asarray(
                homeostasis_activation_khz, dtype=float
            ).tolist(),
            homeostasis_beta=1.0,
            homeostasis_min_scale=0.25,
            homeostasis_max_scale=4.0,
        )
    return model


def _simulate(
    job,
    *,
    occupancy: float,
    receptor: np.ndarray,
    g_value: float,
    seed: int,
    args: argparse.Namespace,
    shape: str | None,
    duration_ms: float = 0.0,
    amplitude_khz: float = 0.0,
    q_i: np.ndarray | None = None,
    online_homeostasis: bool = False,
    homeostasis_target_khz: np.ndarray | None = None,
    homeostasis_activation_khz: np.ndarray | None = None,
    post_window_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    weights, lengths, _atlas, _meta = load_subject_structural(
        subject_id=job.subject_id,
        cohort=job.cohort,
        dataset_root=args.dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
    )
    weights, lengths, _ = _apply_damage_parity(
        weights,
        lengths,
        job.cohort,
        normalize_subject_max=(
            str(args.structural_connectivity_normalization)
            not in {"cohort_global_max", "native_invnodevol", "native_raw"}
        ),
    )
    onset = float(args.transient_ms + args.analysis_ms)
    total = float(
        onset + (args.analysis_ms if post_window_ms is None else post_window_ms)
    )
    stimulus = {
        "stimtime": onset,
        "stimdur": float(duration_ms),
        "stimperiod": total * 10.0,
        "stimval": float(amplitude_khz if shape is not None else 0.0),
        "stimregion": [int(args.stim_region_index)],
        "stimvariables": [0],
        "stimshape": shape or "square",
    }
    cfg = WholeBrainConfig(
        simulation_length_ms=total,
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=float(g_value),
        zerlaut_order=2,
        zerlaut_gk_gna=True,
        stochastic_integrator=True,
        online_inhibitory_homeostasis=bool(online_homeostasis),
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(args.monitor_period_ms),
        monitor_variables=(0, 1, 8, 9, 10) if online_homeostasis else (0, 1),
        weights=np.asarray(weights, dtype=float),
        tract_lengths=np.asarray(lengths, dtype=float),
        connectivity_normalization=str(
            args.simulator_connectivity_normalization
        ),
        parameter_overrides={
            "parameter_model": _base_model(
                job,
                occupancy,
                receptor,
                args,
                q_i=q_i,
                homeostasis_target_khz=homeostasis_target_khz,
                homeostasis_activation_khz=homeostasis_activation_khz,
            ),
            "parameter_stimulus": stimulus,
        },
    )
    result = run_whole_brain_simulation(cfg, seed=int(seed))
    extra = None
    if online_homeostasis:
        monitored = np.asarray(result.full_monitor_output[0][1], dtype=float)
        extra = {
            "inhibitory_scale": monitored[:, 2, :, 0],
            "filtered_e_khz": monitored[:, 3, :, 0],
            "filtered_i_khz": monitored[:, 4, :, 0],
        }
    return (
        np.asarray(result.time_ms),
        np.asarray(result.raw),
        np.asarray(result.raw_inh),
        extra,
    )


def _metrics(
    time: np.ndarray, rate: np.ndarray, *, args: argparse.Namespace
) -> dict[str, float]:
    onset = float(args.transient_ms + args.analysis_ms)
    pre = (time >= onset - args.analysis_ms) & (time < onset)
    full_post = (time >= onset) & (time < onset + args.analysis_ms)
    post = (time >= onset + args.response_start_ms) & (time < onset + args.analysis_ms)
    late = (time >= onset + args.analysis_ms - 50.0) & (time < onset + args.analysis_ms)
    baseline = np.mean(rate[pre], axis=0)
    baseline_sd = np.std(rate[pre], axis=0, ddof=1)
    delta_hz = (rate - baseline) * 1000.0
    post_delta = delta_hz[post]
    threshold_hz = np.maximum(3.0 * baseline_sd * 1000.0, 0.5)
    return {
        "baseline_mean_hz": float(np.mean(baseline) * 1000.0),
        "instantaneous_peak_delta_hz": float(np.max(delta_hz[full_post])),
        "peak_delta_hz": float(np.max(post_delta)),
        "peak_abs_hz": float(np.max(rate[full_post]) * 1000.0),
        "stim_peak_delta_hz": float(np.max(post_delta[:, args.stim_region_index])),
        "propagated_regions": int(np.sum(np.max(post_delta, axis=0) > threshold_hz)),
        "late_residual_hz": float(np.mean(np.abs(delta_hz[late]))),
        "saturated_fraction": float(
            np.mean(rate[full_post] * 1000.0 >= args.saturation_hz)
        ),
    }


def _run_evoked(
    job,
    occupancy,
    g_value,
    seed,
    shape,
    duration,
    amplitude,
    receptor,
    args,
    q_i=None,
    online_homeostasis=False,
    homeostasis_target_khz=None,
    homeostasis_activation_khz=None,
    post_window_ms=None,
):
    time, rate, inh, homeostasis = _simulate(
        job,
        occupancy=occupancy,
        receptor=receptor,
        g_value=g_value,
        seed=seed,
        args=args,
        shape=shape,
        duration_ms=duration,
        amplitude_khz=amplitude,
        q_i=q_i,
        online_homeostasis=online_homeostasis,
        homeostasis_target_khz=homeostasis_target_khz,
        homeostasis_activation_khz=homeostasis_activation_khz,
        post_window_ms=post_window_ms,
    )
    row = {
        "cohort": job.cohort,
        "condition": job.condition,
        "subject_id": job.subject_id,
        "occupancy": occupancy,
        "G": g_value,
        "seed": seed,
        "shape": shape,
        "duration_ms": duration,
        "amplitude_khz": amplitude,
    }
    row.update(_metrics(time, rate, args=args))
    onset = args.transient_ms + args.analysis_ms
    peri = (time >= onset - args.analysis_ms) & (time < onset + args.analysis_ms)
    row["time_relative_ms"] = time[peri] - onset
    row["rate_hz"] = rate[peri] * 1000.0
    row["inh_hz"] = inh[peri] * 1000.0
    if post_window_ms is not None and post_window_ms > args.analysis_ms:
        extended = time >= onset - args.analysis_ms
        row["extended_time_relative_ms"] = time[extended] - onset
        row["extended_rate_hz"] = rate[extended] * 1000.0
    if homeostasis is not None:
        extended = time >= onset - args.analysis_ms
        row["homeostasis_time_relative_ms"] = time[extended] - onset
        row["inhibitory_scale"] = homeostasis["inhibitory_scale"][extended]
        row["filtered_e_hz"] = homeostasis["filtered_e_khz"][extended] * 1000.0
        row["filtered_i_hz"] = homeostasis["filtered_i_khz"][extended] * 1000.0
        late_online = time >= onset + max(0.0, args.homeostatic_post_ms - 200.0)
        row["online_late_mean_hz"] = float(np.mean(rate[late_online]) * 1000.0)
        row["online_max_inhibitory_scale"] = float(
            np.max(homeostasis["inhibitory_scale"][time >= onset])
        )
    return row


def _evoked_task(payload):
    return _run_evoked(**payload)


def _map_tasks(tasks, workers: int):
    if workers == 1:
        return [_evoked_task(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=pilot.worker_initializer
    ) as pool:
        return list(pool.map(_evoked_task, tasks, chunksize=1))


def _select_pulse(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    keys = ["shape", "duration_ms", "amplitude_khz"]
    agg = frame.groupby(keys, as_index=False).agg(
        peak_abs_hz=("peak_abs_hz", "max"),
        peak_delta_hz=("peak_delta_hz", "median"),
        propagated_regions=("propagated_regions", "median"),
        late_residual_hz=("late_residual_hz", "median"),
        saturated_fraction=("saturated_fraction", "max"),
    )
    safe = agg[
        (agg.peak_abs_hz < args.saturation_hz) & (agg.saturated_fraction == 0.0)
    ].copy()
    if safe.empty:
        raise RuntimeError("No pulse avoided the configured saturation threshold.")
    # Prefer visible but non-explosive local/global responses and good recovery.
    safe["score"] = (
        np.minimum(safe.peak_delta_hz, 40.0) / 40.0
        + safe.propagated_regions / 90.0
        - safe.late_residual_hz / 10.0
        - np.maximum(safe.peak_delta_hz - 40.0, 0.0) / 20.0
    )
    winner = safe.sort_values(["score", "amplitude_khz"], ascending=[False, True]).iloc[
        0
    ]
    return {
        key: winner[key].item() if hasattr(winner[key], "item") else winner[key]
        for key in keys + ["score"]
    }


def _select_g(frame: pd.DataFrame, args: argparse.Namespace) -> float:
    """Select a non-saturating G with propagation and recovery across subjects."""
    summary = frame.groupby("G", as_index=False).agg(
        peak_abs_hz=("peak_abs_hz", "max"),
        saturated_fraction=("saturated_fraction", "max"),
        propagated_regions=("propagated_regions", "median"),
        late_residual_hz=("late_residual_hz", "median"),
        pci_st=("pci_st", "median"),
    )
    safe = summary.loc[
        summary["peak_abs_hz"].lt(float(args.saturation_hz))
        & summary["saturated_fraction"].eq(0.0)
    ].copy()
    if safe.empty:
        raise RuntimeError("No global-coupling value avoided saturation.")
    safe["score"] = (
        safe["propagated_regions"] / 90.0
        + np.minimum(safe["pci_st"], 100.0) / 100.0
        - safe["late_residual_hz"] / 10.0
    )
    return float(safe.sort_values(["score", "G"], ascending=[False, True]).iloc[0]["G"])


def _aggregate_trials(rows, group_keys, args):
    """Time-lock trials and compute evoked metrics, PCI-LZ, and PCI-ST."""
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)
    out = []
    onset = float(args.transient_ms + args.analysis_ms)
    for key, trials in grouped.items():
        reference_time = np.asarray(trials[0]["time_relative_ms"], dtype=float)
        for trial in trials[1:]:
            np.testing.assert_allclose(
                trial["time_relative_ms"], reference_time, atol=1e-9, rtol=0.0
            )
        averaged_rate_khz = (
            np.mean(np.stack([trial["rate_hz"] for trial in trials]), axis=0) / 1000.0
        )
        row = dict(zip(group_keys, key, strict=True))
        row["n_time_locked_trials"] = len(trials)
        row.update(_metrics(reference_time + onset, averaged_rate_khz, args=args))
        row["time_relative_ms"] = reference_time
        row["rate_hz"] = averaged_rate_khz * 1000.0
        aligned_trials = [
            np.asarray(trial["rate_hz"], dtype=float) / 1000.0 for trial in trials
        ]
        onset_index = int(np.argmin(np.abs(reference_time)))
        dt_ms = float(np.median(np.diff(reference_time)))
        lz = pci_casali_like_multi_trial(
            aligned_trials,
            stimulation_index=onset_index,
            t_analysis_ms=float(args.analysis_ms),
            dt_ms=dt_ms,
            binarise_method="casali",
            binarise_kwargs={
                "n_bootstrap": int(args.pci_permutation_replicates),
                "alpha": float(args.pci_alpha),
                "seed": 0,
                "significance_method": "pre_post_swap",
            },
            response_start_ms=float(args.response_start_ms),
            min_source_entropy=float(args.pci_min_source_entropy),
        )
        row["pci_lz"] = float(lz[0])
        trial_stack = np.stack(
            [np.asarray(trial, dtype=float).T for trial in aligned_trials], axis=0
        )
        st = pci_st_from_trials(
            trial_stack,
            reference_time,
            baseline_center_trials=True,
            baseline_window_ms=(-float(args.analysis_ms), -50.0),
            response_window_ms=(float(args.response_start_ms), float(args.analysis_ms)),
            k=float(args.pci_st_k),
            min_snr=float(args.pci_st_min_snr),
            max_var_percent=float(args.pci_st_max_var_percent),
            n_steps=int(args.pci_st_n_steps),
            return_details=True,
        )
        if not isinstance(st, PCIStResult):
            raise AssertionError("Detailed PCI-ST result was not returned.")
        row["pci_st"] = float(st.pci_st)
        row["pci_st_n_components"] = int(st.n_components)
        out.append(row)
    return out


def _save_aligned_time_courses(rows, output_dir: Path, family: str) -> None:
    """Save each ten-trial average without expanding arrays into CSV cells."""
    root = output_dir / "aligned_time_courses" / family
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        parts = [
            str(row.get("condition", "condition")),
            str(row.get("subject_id", "subject")),
            f"g_{float(row.get('G', 0.0)):.6g}",
            str(row.get("shape", "pulse")),
            f"dur_{float(row.get('duration_ms', 0.0)):.6g}",
            f"amp_{float(row.get('amplitude_khz', 0.0)):.6g}",
        ]
        if "homeostasis" in row:
            parts.append(str(row["homeostasis"]))
        filename = "__".join(parts).replace("/", "-") + ".npz"
        np.savez_compressed(
            root / filename,
            time_relative_ms=np.asarray(row["time_relative_ms"], dtype=float),
            rate_hz=np.asarray(row["rate_hz"], dtype=float),
            n_trials=np.asarray([int(row["n_time_locked_trials"])]),
            pci_lz=np.asarray([float(row["pci_lz"])]),
            pci_st=np.asarray([float(row["pci_st"])]),
        )


def _learn_q_i(job, occupancy, g_value, receptor, target_khz, args):
    config = InhibitoryHomeostasisConfig(tau_s=args.homeostatic_tau_s)
    base_q_i = float(args.homeostatic_base_qi_ns)
    q_i = np.full(90, base_q_i, dtype=float)
    history = []
    for epoch in range(args.homeostatic_epochs):
        time, rate, inh, _ = _simulate(
            job,
            occupancy=occupancy,
            receptor=receptor,
            g_value=g_value,
            seed=10000 + epoch,
            args=args,
            shape=None,
            q_i=q_i,
        )
        tail = time >= time[-1] - args.homeostatic_epoch_ms
        mean_e = np.mean(rate[tail], axis=0)
        mean_i = np.mean(inh[tail], axis=0)
        q_i = update_inhibitory_conductance(
            q_i,
            mean_e,
            mean_i,
            target_khz,
            base_q_i_ns=base_q_i,
            epoch_ms=args.homeostatic_epoch_ms,
            config=config,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "mean_e_hz": float(np.mean(mean_e) * 1000.0),
                "target_hz": float(np.mean(target_khz) * 1000.0),
                "mean_q_i_ns": float(np.mean(q_i)),
                "min_q_i_ns": float(np.min(q_i)),
                "max_q_i_ns": float(np.max(q_i)),
            }
        )
    return q_i, history


def _homeostasis_task(payload):
    job = payload["job"]
    occupancy = payload["occupancy"]
    g_value = payload["g_value"]
    receptor = payload["receptor"]
    target = payload["target"]
    activation_threshold = payload["activation_threshold"]
    args = payload["args"]
    winner = payload["winner"]
    q_i, history = _learn_q_i(job, occupancy, g_value, receptor, target, args)
    rows = []
    for mode, learned_q, online in [
        ("off", None, False),
        ("prefit_frozen", q_i, False),
        ("online", None, True),
    ]:
        for seed in args.trial_seeds:
            row = _run_evoked(
                job,
                occupancy,
                g_value,
                seed,
                winner["shape"],
                winner["duration_ms"],
                winner["amplitude_khz"],
                receptor,
                args,
                q_i=learned_q,
                online_homeostasis=online,
                homeostasis_target_khz=(target if online else None),
                homeostasis_activation_khz=(
                    activation_threshold if online else None
                ),
                post_window_ms=args.homeostatic_post_ms,
            )
            row["homeostasis"] = mode
            rows.append(row)
    for item in history:
        item.update(
            cohort=job.cohort,
            condition=job.condition,
            subject_id=job.subject_id,
            occupancy=occupancy,
            G=g_value,
        )
    return rows, history


def _save_figures(
    pulse_df, g_df, g_average_rows, homeo_df, homeo_rows, histories,
    winner, selected_g, args
):
    _style()
    fig_dir = args.output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    for shape in args.pulse_shapes:
        t, y = _pulse_values(shape, float(winner["duration_ms"]))
        ax.plot(t, y, lw=1.4, label=shape.replace("_", " "))
    ax.set(
        xlabel="Time from pulse onset (ms)",
        ylabel="Normalized input",
        title="A  Candidate pulse waveforms",
    )
    ax.legend(frameon=False)
    summary = pulse_df.groupby(
        ["shape", "duration_ms", "amplitude_khz"], as_index=False
    ).median(numeric_only=True)
    for ax, metric, title in [
        (axes[0, 1], "peak_abs_hz", "B  Maximum firing rate"),
        (axes[1, 0], "propagated_regions", "C  Spatial propagation"),
        (axes[1, 1], "late_residual_hz", "D  Late residual"),
    ]:
        sns.lineplot(
            data=summary,
            x="amplitude_khz",
            y=metric,
            hue="shape",
            style="duration_ms",
            marker="o",
            ax=ax,
            legend=(metric == "peak_abs_hz"),
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Peak input (kHz)")
        if metric == "peak_abs_hz":
            ax.axhline(args.saturation_hz, color="0.25", ls="--", lw=0.8)
        if metric != "peak_abs_hz" and ax.get_legend() is not None:
            ax.get_legend().remove()
    sns.despine(fig=fig)
    for ext in ("pdf", "png"):
        fig.savefig(
            fig_dir / f"stimulus_calibration.{ext}", dpi=600 if ext == "png" else None
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "pci_lz", "A  PCI-LZ"),
        (axes[1], "pci_st", "B  PCI-ST"),
    ):
        sns.lineplot(
            data=g_df,
            x="G",
            y=metric,
            hue="condition",
            palette=COND_COLORS,
            marker="o",
            estimator=None,
            ax=ax,
        )
        ax.set_title(title, loc="left")
        ax.set_ylabel(metric.upper().replace("_", "-"))
        if ax is axes[1] and ax.get_legend() is not None:
            ax.get_legend().remove()
    sns.despine(fig=fig)
    for ext in ("pdf", "png"):
        fig.savefig(
            fig_dir / f"dual_pci_calibration.{ext}",
            dpi=600 if ext == "png" else None,
        )
    plt.close(fig)

    for row in g_average_rows:
        if not np.isclose(float(row["G"]), float(selected_g)):
            continue
        times = np.asarray(row["time_relative_ms"], dtype=float)
        rates = np.asarray(row["rate_hz"], dtype=float)
        baseline = rates[times < 0.0].mean(axis=0, keepdims=True)
        delta = rates - baseline
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.7), sharex=True,
                                 constrained_layout=True)
        axes[0].plot(times, delta, color=COND_COLORS[row["condition"]],
                     lw=0.45, alpha=0.22)
        axes[0].plot(times, delta.mean(axis=1), color="#111111", lw=1.7)
        axes[0].axvline(0.0, color="#222222", lw=0.8, ls="--")
        axes[0].set_ylabel("Δ firing rate (Hz)")
        axes[0].set_title(
            f"{row['condition']} {row['subject_id']} — regional evoked responses",
            loc="left",
        )
        axes[1].imshow(
            delta.T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], 1, delta.shape[1]],
            cmap="RdBu_r",
            vmin=-np.nanpercentile(np.abs(delta), 99),
            vmax=np.nanpercentile(np.abs(delta), 99),
            interpolation="nearest",
        )
        axes[1].axvline(0.0, color="#222222", lw=0.8, ls="--")
        axes[1].set(xlabel="Time from stimulation (ms)", ylabel="AAL90 region")
        sns.despine(fig=fig)
        stem = fig_dir / f"evoked_timecourse_{row['condition']}_{row['subject_id']}"
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    for ax, metric, ylabel in zip(
        axes,
        ["peak_delta_hz", "propagated_regions", "late_residual_hz"],
        ["Peak response (Hz)", "Regions exceeding baseline", "Late residual (Hz)"],
    ):
        sns.lineplot(
            data=g_df,
            x="G",
            y=metric,
            hue="condition",
            style="occupancy",
            palette=COND_COLORS,
            marker="o",
            estimator=None,
            ax=ax,
        )
        ax.set_ylabel(ylabel)
        if ax is not axes[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[0].set_title("A  Response amplitude", loc="left")
    axes[1].set_title("B  Propagation", loc="left")
    axes[2].set_title("C  Recovery", loc="left")
    sns.despine(fig=fig)
    for ext in ("pdf", "png"):
        fig.savefig(
            fig_dir / f"global_coupling_calibration.{ext}",
            dpi=600 if ext == "png" else None,
        )
    plt.close(fig)

    if homeo_df is not None and not homeo_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
        hist = pd.DataFrame(histories)
        sns.lineplot(
            data=hist,
            x="epoch",
            y="mean_e_hz",
            hue="condition",
            palette=COND_COLORS,
            marker="o",
            ax=axes[0],
        )
        sns.lineplot(
            data=hist,
            x="epoch",
            y="mean_q_i_ns",
            hue="condition",
            palette=COND_COLORS,
            marker="o",
            ax=axes[1],
            legend=False,
        )
        sns.lineplot(
            data=homeo_df[np.isclose(homeo_df["occupancy"], max(args.occupancies))],
            x="G",
            y="peak_delta_hz",
            hue="condition",
            style="homeostasis",
            palette=COND_COLORS,
            marker="o",
            estimator=None,
            ax=axes[2],
        )
        axes[0].set_title("A  Rate convergence", loc="left")
        axes[1].set_title("B  Learned inhibition", loc="left")
        axes[2].set_title("C  Frozen-rule response", loc="left")
        axes[0].set_ylabel("Mean excitatory rate (Hz)")
        axes[1].set_ylabel(r"Mean learned $Q_{i\to e}$ (nS)")
        axes[2].set_ylabel("Peak response (Hz)")
        sns.despine(fig=fig)
        for ext in ("pdf", "png"):
            fig.savefig(
                fig_dir / f"homeostasis_calibration.{ext}",
                dpi=600 if ext == "png" else None,
            )
        plt.close(fig)

        selected_rows = [
            row
            for row in homeo_rows
            if row["homeostasis"] in {"off", "online"}
            and np.isclose(row["occupancy"], max(args.occupancies))
            and np.isclose(row["G"], selected_g)
        ]
        trajectory_records = []
        scale_records = []
        late_records = []
        for row in selected_rows:
            relative_time = np.asarray(row["extended_time_relative_ms"])
            global_rate = np.mean(np.asarray(row["extended_rate_hz"]), axis=1)
            for time_value, rate_value in zip(relative_time, global_rate, strict=True):
                trajectory_records.append(
                    {
                        "time_ms": time_value,
                        "rate_hz": rate_value,
                        "condition": row["condition"],
                        "homeostasis": row["homeostasis"],
                        "seed": row["seed"],
                    }
                )
            late_records.append(
                {
                    "condition": row["condition"],
                    "homeostasis": row["homeostasis"],
                    "late_rate_hz": float(
                        np.mean(
                            global_rate[
                                relative_time >= args.homeostatic_post_ms - 200.0
                            ]
                        )
                    ),
                }
            )
            if row["homeostasis"] == "online":
                scale_time = np.asarray(row["homeostasis_time_relative_ms"])
                global_scale = np.mean(np.asarray(row["inhibitory_scale"]), axis=1)
                for time_value, scale_value in zip(
                    scale_time, global_scale, strict=True
                ):
                    scale_records.append(
                        {
                            "time_ms": time_value,
                            "scale": scale_value,
                            "condition": row["condition"],
                            "seed": row["seed"],
                        }
                    )
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
        sns.lineplot(
            data=pd.DataFrame(trajectory_records),
            x="time_ms",
            y="rate_hz",
            hue="condition",
            style="homeostasis",
            palette=COND_COLORS,
            errorbar="se",
            ax=axes[0],
        )
        sns.lineplot(
            data=pd.DataFrame(scale_records),
            x="time_ms",
            y="scale",
            hue="condition",
            palette=COND_COLORS,
            errorbar="se",
            ax=axes[1],
        )
        sns.pointplot(
            data=pd.DataFrame(late_records),
            x="condition",
            y="late_rate_hz",
            hue="homeostasis",
            errorbar="se",
            dodge=0.3,
            ax=axes[2],
        )
        for ax in axes[:2]:
            ax.axvline(0.0, color="0.2", lw=0.7, ls="--")
            ax.set_xlabel("Time from stimulation (ms)")
        axes[0].set_title("A  Post-stimulation firing", loc="left")
        axes[0].set_ylabel("Whole-brain mean rate (Hz)")
        axes[1].set_title("B  Adaptive inhibitory efficacy", loc="left")
        axes[1].set_ylabel(r"Mean $Q_{i\to e}$ scale")
        axes[2].set_title("C  Late firing-rate control", loc="left")
        axes[2].set_ylabel("Mean rate in final 200 ms (Hz)")
        axes[2].set_xlabel("")
        sns.despine(fig=fig)
        for ext in ("pdf", "png"):
            fig.savefig(
                fig_dir / f"online_homeostasis_dynamics.{ext}",
                dpi=600 if ext == "png" else None,
            )
        plt.close(fig)


def _strip_arrays(rows):
    return [
        {k: v for k, v in row.items() if not isinstance(v, np.ndarray)} for row in rows
    ]


def main() -> None:
    args = parse_args()
    _validate(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = _subjects(args)
    selection_metadata = _subject_selection_metadata(args, jobs)
    pd.DataFrame(selection_metadata).to_csv(
        args.output_root / "calibration_subjects.csv", index=False
    )
    atlas = load_aal90_atlas(args.dataset_root)
    labels = list(np.asarray(atlas.labels).astype(str))
    if args.stim_region_label not in labels:
        raise KeyError(f"{args.stim_region_label!r} is absent from {atlas.ordering}.")
    args.stim_region_index = labels.index(args.stim_region_label)
    receptor = get_5ht2a_aal90(
        tracer=args.receptor_tracer, csv_path=args.receptor_csv, target_labels=labels
    )
    manifest_path = args.output_root / "run_manifest.json"

    planned_pulse = (
        len(jobs)
        * len(args.occupancies)
        * len(args.trial_seeds)
        * len(args.pulse_shapes)
        * len(args.durations_ms)
        * len(args.amplitudes_khz)
    )
    planned_g = (
        len(jobs) * len(args.occupancies) * len(args.trial_seeds) * len(args.g_values)
    )
    # Homeostasis is a focused method comparison at the reference G, after the
    # pulse grid; it is not multiplied across the complete coupling sweep.
    homeostasis_g_values = [float(args.reference_g)]
    planned_homeostasis = 0
    if args.homeostasis == "compare":
        target_runs = (
            0
            if args.homeostatic_target == "fixed"
            else len(jobs) * len(args.occupancies) * len(homeostasis_g_values)
        )
        learning_runs = (
            len(jobs)
            * len(args.occupancies)
            * len(homeostasis_g_values)
            * args.homeostatic_epochs
        )
        comparison_runs = (
            len(jobs)
            * len(args.occupancies)
            * len(homeostasis_g_values)
            * len(args.trial_seeds)
            * 3
        )
        planned_homeostasis = target_runs + learning_runs + comparison_runs
    print(
        json.dumps(
            {
                "subjects": [f"{j.cohort}:{j.subject_id}" for j in jobs],
                "stim_region_label": args.stim_region_label,
                "stim_region_index_zero_based": args.stim_region_index,
                "monitor_rate_hz": 1000.0 / args.monitor_period_ms,
                "pulse_runs": planned_pulse,
                "g_runs": planned_g,
                "homeostasis_runs": planned_homeostasis,
                "total_runs": planned_pulse + planned_g + planned_homeostasis,
                "homeostasis": args.homeostasis,
                "workers": args.workers,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Completed calibration already exists at {manifest_path}. "
            "Use a new output root or pass --overwrite."
        )

    pulse_tasks = [
        {
            "job": job,
            "occupancy": occupancy,
            "g_value": args.reference_g,
            "seed": seed,
            "shape": shape,
            "duration": duration,
            "amplitude": amplitude,
            "receptor": receptor,
            "args": args,
        }
        for job in jobs
        for occupancy in args.occupancies
        for shape in args.pulse_shapes
        for duration in args.durations_ms
        for amplitude in args.amplitudes_khz
        for seed in args.trial_seeds
    ]
    pulse_rows = _map_tasks(pulse_tasks, args.workers)
    pd.DataFrame(_strip_arrays(pulse_rows)).to_csv(
        args.output_root / "stimulus_calibration_trial_metrics.csv", index=False
    )
    pulse_average_rows = _aggregate_trials(
        pulse_rows,
        [
            "cohort",
            "condition",
            "subject_id",
            "occupancy",
            "G",
            "shape",
            "duration_ms",
            "amplitude_khz",
        ],
        args,
    )
    pulse_df = pd.DataFrame(_strip_arrays(pulse_average_rows))
    pulse_df.to_csv(args.output_root / "stimulus_calibration_metrics.csv", index=False)
    winner = _select_pulse(pulse_df, args)

    g_tasks = [
        {
            "job": job,
            "occupancy": occupancy,
            "g_value": g_value,
            "seed": seed,
            "shape": winner["shape"],
            "duration": winner["duration_ms"],
            "amplitude": winner["amplitude_khz"],
            "receptor": receptor,
            "args": args,
        }
        for job in jobs
        for occupancy in args.occupancies
        for g_value in args.g_values
        for seed in args.trial_seeds
    ]
    g_rows = _map_tasks(g_tasks, args.workers)
    pd.DataFrame(_strip_arrays(g_rows)).to_csv(
        args.output_root / "global_coupling_trial_metrics.csv", index=False
    )
    g_average_rows = _aggregate_trials(
        g_rows,
        [
            "cohort",
            "condition",
            "subject_id",
            "occupancy",
            "G",
            "shape",
            "duration_ms",
            "amplitude_khz",
        ],
        args,
    )
    g_df = pd.DataFrame(_strip_arrays(g_average_rows))
    g_df.to_csv(args.output_root / "global_coupling_metrics.csv", index=False)
    selected_g = _select_g(g_df, args)
    homeostasis_g_values = [selected_g]

    homeo_rows, histories = [], []
    if args.homeostasis == "compare":
        targets = {}
        activation_thresholds = {}
        for job in jobs:
            for occupancy in args.occupancies:
                for g_value in homeostasis_g_values:
                    key = (job.cohort, float(occupancy), float(g_value))
                    time, rate, _, _ = _simulate(
                        job,
                        occupancy=occupancy,
                        receptor=receptor,
                        g_value=g_value,
                        seed=9999,
                        args=args,
                        shape=None,
                    )
                    tail = time >= time[-1] - args.homeostatic_epoch_ms
                    baseline = rate[tail]
                    activation_thresholds[key] = (
                        baseline_relative_activation_threshold(
                            baseline, n_sd=args.homeostatic_activation_sd
                        )
                    )
                    if args.homeostatic_target == "fixed":
                        targets[key] = np.full(90, args.homeostatic_target_hz / 1000.0)
                    else:
                        targets[key] = np.maximum(np.mean(rate[tail], axis=0), 0.0005)
        homeo_tasks = [
            {
                "job": job,
                "occupancy": occupancy,
                "g_value": g_value,
                "receptor": receptor,
                "target": targets[(job.cohort, float(occupancy), float(g_value))],
                "activation_threshold": activation_thresholds[
                    (job.cohort, float(occupancy), float(g_value))
                ],
                "args": args,
                "winner": winner,
            }
            for job in jobs
            for occupancy in args.occupancies
            for g_value in homeostasis_g_values
        ]
        if args.workers == 1:
            homeo_results = [_homeostasis_task(task) for task in homeo_tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=pilot.worker_initializer
            ) as pool:
                homeo_results = list(
                    pool.map(_homeostasis_task, homeo_tasks, chunksize=1)
                )
        for rows, hist in homeo_results:
            homeo_rows.extend(rows)
            histories.extend(hist)
        pd.DataFrame(_strip_arrays(homeo_rows)).to_csv(
            args.output_root / "homeostasis_evoked_trial_metrics.csv", index=False
        )
        homeo_average_rows = _aggregate_trials(
            homeo_rows,
            [
                "cohort",
                "condition",
                "subject_id",
                "occupancy",
                "G",
                "shape",
                "duration_ms",
                "amplitude_khz",
                "homeostasis",
            ],
            args,
        )
        homeo_df = pd.DataFrame(_strip_arrays(homeo_average_rows))
        homeo_df.to_csv(
            args.output_root / "homeostasis_evoked_metrics.csv", index=False
        )
        pd.DataFrame(histories).to_csv(
            args.output_root / "homeostasis_learning.csv", index=False
        )
    else:
        homeo_df = pd.DataFrame()

    _save_aligned_time_courses(pulse_average_rows, args.output_root, "pulse")
    _save_aligned_time_courses(g_average_rows, args.output_root, "global_coupling")
    if homeo_df is not None and not homeo_df.empty:
        _save_aligned_time_courses(homeo_average_rows, args.output_root, "homeostasis")
    _save_figures(
        pulse_df,
        g_df,
        g_average_rows,
        homeo_df,
        homeo_rows,
        histories,
        winner,
        selected_g,
        args,
    )
    manifest = {
        "purpose": "seven-subject calibration only; not inferential cohort analysis",
        "selected_pulse": winner,
        "selected_global_coupling": selected_g,
        "subjects": [asdict(job) for job in jobs],
        "subject_selection": selection_metadata,
        "stimulus_target": {
            "label": args.stim_region_label,
            "zero_based_index": args.stim_region_index,
        },
        "sampling": {
            "dt_ms": 0.1,
            "monitor_period_ms": args.monitor_period_ms,
            "rate_hz": 1000.0 / args.monitor_period_ms,
        },
        "response_estimator": (
            "metrics from stimulation-onset-aligned, matched-trial averaged rates"
        ),
        "homeostasis": {
            "mode": args.homeostasis,
            "target": args.homeostatic_target,
            "fixed_target_hz": args.homeostatic_target_hz,
            "epochs": args.homeostatic_epochs,
            "epoch_ms": args.homeostatic_epoch_ms,
            "tau_s": args.homeostatic_tau_s,
            "detector_tau_ms": args.homeostatic_detector_tau_ms,
            "activation_threshold": (
                "per-region unstimulated baseline mean + "
                f"{args.homeostatic_activation_sd:g} baseline SD"
            ),
            "activation_sd": args.homeostatic_activation_sd,
            "online_post_window_ms": args.homeostatic_post_ms,
            "comparison_modes": ["off", "prefit_frozen", "online"],
        },
        "shared_b_e_pA": args.shared_b_e,
        "diagnosis_b_gradient_disabled_for_calibration": args.shared_b_e is not None,
        "pci": {
            "n_trials": len(args.trial_seeds),
            "estimators": ["PCI-LZ", "PCI-ST"],
            "response_start_ms": args.response_start_ms,
            "lz_significance": "pre_post_swap",
            "lz_permutations": args.pci_permutation_replicates,
            "lz_alpha": args.pci_alpha,
            "st_k": args.pci_st_k,
            "st_min_snr": args.pci_st_min_snr,
            "st_max_var_percent": args.pci_st_max_var_percent,
            "st_n_steps": args.pci_st_n_steps,
        },
        "G_values": args.g_values,
        "structural_connectivity_normalization": str(
            args.structural_connectivity_normalization
        ),
        "structural_connectivity_normalization_divisor": (
            None
            if args.structural_connectivity_normalization_divisor is None
            else float(args.structural_connectivity_normalization_divisor)
        ),
        "simulator_connectivity_normalization": str(
            args.simulator_connectivity_normalization
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Selected pulse: {winner}")
    print(f"Figures: {args.output_root / 'figures'}")


if __name__ == "__main__":
    main()
