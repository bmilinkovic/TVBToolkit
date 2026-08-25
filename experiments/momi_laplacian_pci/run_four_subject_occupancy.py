#!/usr/bin/env python3
"""Run a four-condition 5-HT2A occupancy sweep with Momi Laplacian coupling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(_REPO_ROOT / ".tvb-temp"))
for _path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "notebooks"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

from brain_act_hybrid_common import BASE_PARAMETER_MODEL_NEW  # noqa: E402
from experiments.momi_laplacian_pci.graph_laplacian import momi_laplacian  # noqa: E402
from experiments.momi_laplacian_pci.plot_operator_pci_comparison import (  # noqa: E402
    _aal90_centres,
    _peak_latencies,
    _plot_region_brain,
)
from experiments.momi_laplacian_pci.run_two_subject_robustness import (  # noqa: E402
    _subject_metrics,
)
from scripts.run_serotonergic_pci_pilot import _gk_profile_from_occupancy  # noqa: E402
from tvbtoolkit.core.config import WholeBrainConfig  # noqa: E402
from tvbtoolkit.datasets.brain_act import (  # noqa: E402
    list_subjects,
    load_aal90_atlas,
    load_subject_structural,
)
from tvbtoolkit.datasets.structural_provenance import validate_native_invnodevol_dataset  # noqa: E402
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: E402
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import worker_initializer  # noqa: E402
from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90  # noqa: E402


PROTOCOL_VERSION = "momi-laplacian-5ht2a-occupancy-1.1-zerlaut-so-khz-v2"
DEFAULT_DATASET = _REPO_ROOT / "data" / "doc_data" / "converted_structural_invnodevol_native"
DEFAULT_OUTPUT = _REPO_ROOT / "results" / "momi_laplacian_5ht2a_four_subject"
DEFAULT_SUBJECTS = (
    "control:c0015",
    "emcs:e0003",
    "mcs:m0075",
    "uws:u0020",
)
CONDITIONS = {"control": "CNT", "emcs": "EMCS", "mcs": "MCS", "uws": "UWS"}
COLORS = {"CNT": "#2E86AB", "EMCS": "#2CA25F", "MCS": "#E67E22", "UWS": "#C0392B"}


@dataclass(frozen=True)
class Subject:
    cohort: str
    subject_id: str
    condition: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subject", action="append")
    parser.add_argument("--occupancies", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.766])
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--coupling-strength", type=float, default=3.5)
    parser.add_argument("--conduction-speed-m-s", type=float, default=4.0)
    parser.add_argument("--b-e-pa", type=float, default=10.0)
    parser.add_argument("--e-l-e-drug", type=float, default=-61.2)
    parser.add_argument("--e-l-i-drug", type=float, default=-64.4)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--transient-ms", type=float, default=4000.0)
    parser.add_argument("--analysis-ms", type=float, default=300.0)
    parser.add_argument("--monitor-period-ms", type=float, default=3.0)
    parser.add_argument("--response-start-ms", type=float, default=8.0)
    parser.add_argument("--stim-region-label", default="Supp_Motor_Area_L")
    parser.add_argument("--stim-duration-ms", type=float, default=10.0)
    parser.add_argument("--stim-amplitude-khz", type=float, default=0.0003)
    parser.add_argument("--pci-permutation-replicates", type=int, default=500)
    parser.add_argument("--pci-alpha", type=float, default=0.01)
    parser.add_argument("--pci-random-seed", type=int, default=0)
    parser.add_argument("--pci-st-k", type=float, default=1.2)
    parser.add_argument("--pci-st-min-snr", type=float, default=1.1)
    parser.add_argument("--pci-st-max-var-percent", type=float, default=99.0)
    parser.add_argument("--pci-st-n-steps", type=int, default=100)
    parser.set_defaults(
        pci_min_source_entropy=None,
        pci_significance_method="trial_bootstrap",
        pulse_shape="square",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _occ_tag(value: float) -> str:
    return f"occ_{int(round(value * 1000)):03d}"


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate(args: argparse.Namespace) -> list[Subject]:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_native_invnodevol_dataset(args.dataset_root)
    raw_subjects = args.subject or list(DEFAULT_SUBJECTS)
    if len(raw_subjects) != 4:
        raise ValueError("exactly four subjects are required")
    available = list_subjects(args.dataset_root, cohort=None)
    subjects = []
    for raw in raw_subjects:
        cohort, subject_id = raw.split(":", 1)
        cohort = cohort.lower().strip()
        if cohort not in CONDITIONS or subject_id not in available[cohort]:
            raise KeyError(f"unknown or unsupported subject {raw}")
        subjects.append(Subject(cohort, subject_id, CONDITIONS[cohort]))
    if {subject.cohort for subject in subjects} != set(CONDITIONS):
        raise ValueError("selection must contain one CNT, EMCS, MCS, and UWS subject")
    if sorted(args.occupancies) != sorted(set(args.occupancies)) or any(
        value < 0.0 or value > 1.0 for value in args.occupancies
    ):
        raise ValueError("occupancies must be unique values in [0, 1]")
    if not any(np.isclose(value, 0.0) for value in args.occupancies):
        raise ValueError("occupancies must include zero")
    if len(set(args.trial_seeds)) != len(args.trial_seeds) or not args.trial_seeds:
        raise ValueError("trial seeds must be a non-empty unique list")
    return subjects


def _parameter_model(occupancy: float, receptor_map: np.ndarray, args: dict[str, Any]) -> dict[str, Any]:
    model = deepcopy(BASE_PARAMETER_MODEL_NEW)
    model["b_e"] = float(args["b_e_pa_shared"])
    g_ke, g_na_e, _, e_eff_e = _gk_profile_from_occupancy(
        occupancy=occupancy,
        receptor_map=receptor_map,
        e_l_start=float(model["E_L_e"]),
        e_l_drug=float(args["e_l_e_drug"]),
    )
    g_ki, g_na_i, _, e_eff_i = _gk_profile_from_occupancy(
        occupancy=occupancy,
        receptor_map=receptor_map,
        e_l_start=float(model["E_L_i"]),
        e_l_drug=float(args["e_l_i_drug"]),
    )
    model.update(
        {
            "g_K_e": g_ke.tolist(),
            "g_Na_e": float(g_na_e),
            "g_K_i": g_ki.tolist(),
            "g_Na_i": float(g_na_i),
            "noise_alpha": 0.0,
            "shared_noise_mode": "none",
            "serotonergic_occupancy": float(occupancy),
            "serotonergic_e_eff_e_highest_receptor": float(e_eff_e),
            "serotonergic_e_eff_i_highest_receptor": float(e_eff_i),
        }
    )
    return model


def _trial_path(root: Path, subject: Subject, occupancy: float, seed: int) -> Path:
    return root / "trials" / subject.cohort / subject.subject_id / _occ_tag(occupancy) / f"seed_{seed:03d}.npz"


def _run_trial(task: dict[str, Any]) -> dict[str, Any]:
    subject = Subject(**task["subject"])
    occupancy = float(task["occupancy"])
    seed = int(task["seed"])
    output = Path(task["output"])
    if output.is_file() and not task["overwrite"]:
        with np.load(output, allow_pickle=False) as cached:
            cached_version = str(
                np.asarray(cached["protocol_version"]).reshape(-1)[0]
            )
        if cached_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"stale pre-correction trial at {output}; pass --overwrite "
                "or select a new output root"
            )
        return {"path": str(output), "cached": True}
    args = task["simulation"]
    weights, lengths, atlas, _ = load_subject_structural(
        subject_id=subject.subject_id,
        cohort=subject.cohort,
        dataset_root=Path(task["dataset_root"]),
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    laplacian, _ = momi_laplacian(weights)
    labels = np.asarray(atlas.labels).astype(str)
    stim_index = list(labels).index(args["stim_region_label"])
    onset_ms = float(args["transient_ms"] + args["analysis_ms"])
    total_ms = float(onset_ms + args["analysis_ms"])
    model = _parameter_model(occupancy, np.asarray(task["receptor_map"]), args)
    stimulus = {
        "stimtime": onset_ms,
        "stimdur": args["stim_duration_ms"],
        "stimperiod": total_ms * 10.0,
        "stimval": args["stim_amplitude_khz"],
        "stimregion": [stim_index],
        "stimvariables": [0],
        "stimshape": "square",
    }
    config = WholeBrainConfig(
        simulation_length_ms=total_ms,
        dt_ms=args["dt_ms"],
        conduction_speed=args["conduction_speed_m_s"],
        coupling_strength=args["coupling_strength"],
        zerlaut_order=2,
        zerlaut_gk_gna=True,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=args["monitor_period_ms"],
        monitor_variables=(0, 1),
        weights=-laplacian,
        tract_lengths=np.asarray(lengths),
        connectivity_normalization="none",
        parameter_overrides={
            "parameter_model": model,
            "parameter_stimulus": stimulus,
            "nullify_diagonals": False,
        },
    )
    result = run_whole_brain_simulation(config, seed=seed)
    time = np.asarray(result.time_ms, dtype=float)
    rates = np.asarray(result.raw, dtype=float) * 1000.0
    peri = (time >= onset_ms - args["analysis_ms"]) & (time < onset_ms + args["analysis_ms"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        protocol_version=np.asarray([PROTOCOL_VERSION]),
        cohort=np.asarray([subject.cohort]),
        condition=np.asarray([subject.condition]),
        subject_id=np.asarray([subject.subject_id]),
        occupancy=np.asarray([occupancy]),
        seed=np.asarray([seed]),
        time_relative_ms=time[peri] - onset_ms,
        rate_hz=rates[peri],
        region_labels=labels.astype("U128"),
        stim_region_index=np.asarray([stim_index]),
        receptor_map_sha256=np.asarray([task["receptor_map_sha256"]]),
    )
    return {"path": str(output), "cached": False}


def _load_trials(root: Path, subject: Subject, occupancy: float, seeds: list[int]):
    trials = []
    time = labels = None
    stim_index = -1
    for seed in seeds:
        with np.load(_trial_path(root, subject, occupancy, seed), allow_pickle=False) as data:
            if time is None:
                time = np.asarray(data["time_relative_ms"], dtype=float)
                labels = np.asarray(data["region_labels"]).astype(str)
                stim_index = int(np.asarray(data["stim_region_index"]).reshape(-1)[0])
            trials.append(np.asarray(data["rate_hz"], dtype=float))
    return np.stack(trials), np.asarray(time), np.asarray(labels), stim_index


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(root: Path, aggregates: dict[tuple[str, float], dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    subjects = list(dict.fromkeys(row["subject_id"] for row in rows))
    occupancies = sorted(set(float(row["occupancy"]) for row in rows))
    lookup = {(row["subject_id"], float(row["occupancy"])): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for subject_id in subjects:
        condition = next(row["condition"] for row in rows if row["subject_id"] == subject_id)
        color = COLORS[condition]
        axes[0].plot(occupancies, [lookup[(subject_id, o)]["pci_lz"] for o in occupancies], "o-", color=color, label=f"{condition} {subject_id}")
        axes[1].plot(occupancies, [lookup[(subject_id, o)]["tvbsim_pci_lz"] for o in occupancies], "o-", color=color)
        axes[2].plot(occupancies, [lookup[(subject_id, o)]["pci_st"] for o in occupancies], "o-", color=color)
    for axis, title, ylabel in zip(
        axes,
        ("Current trial-average PCI-LZ", "Legacy TVBSim PCI-LZ", "PCI-ST"),
        ("PCI-LZ", "PCI-LZ", "PCI-ST"),
        strict=True,
    ):
        axis.set_title(title); axis.set_xlabel("5-HT2A occupancy"); axis.set_ylabel(ylabel); axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Laplacian AdEx response across 5-HT2A occupancy (shared bₑ=10 pA)")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"pci_by_5ht2a_occupancy.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(subjects), len(occupancies), figsize=(15, 10), constrained_layout=True,
        sharex=True, squeeze=False,
    )
    for row_index, subject_id in enumerate(subjects):
        condition = next(row["condition"] for row in rows if row["subject_id"] == subject_id)
        for column, occupancy in enumerate(occupancies):
            data = aggregates[(subject_id, occupancy)]
            time = data["time"]
            delta = data["mean"] - np.mean(data["mean"][time < 0.0], axis=0)
            axis = axes[row_index, column]
            for region in range(delta.shape[1]):
                if region != data["stim_index"]:
                    axis.plot(time, delta[:, region], color="#7F8C8D", alpha=0.22, lw=0.5)
            axis.plot(time, delta[:, data["stim_index"]], color=COLORS[condition], lw=1.5)
            axis.axvline(0, color="black", ls="--", lw=0.7); axis.axvspan(0, 10, color="#F4C95D", alpha=0.3)
            if row_index == 0: axis.set_title(f"Occupancy {occupancy:g}")
            if column == 0: axis.set_ylabel(f"{condition} {subject_id}\nΔ rate (Hz)")
            if row_index == len(subjects)-1: axis.set_xlabel("Time from stimulation (ms)")
    fig.suptitle("All 90 regional time courses across occupancy")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"occupancy_regional_butterflies.{suffix}", dpi=300)
    plt.close(fig)

    # Publication-style propagation views: every region has its own colour, and
    # five response peaks are projected onto the AAL90 centres.  No PCI cutoff
    # is drawn here; statistical binarisation is a separate analysis product.
    brain_root = root / "tdcs_style_butterflies"
    brain_root.mkdir(parents=True, exist_ok=True)
    exemplar_labels = np.asarray(next(iter(aggregates.values()))["labels"]).astype(str)
    centres = _aal90_centres(exemplar_labels)
    region_colours = plt.cm.hsv(np.linspace(0.0, 1.0, 91)[:-1])
    for subject_id in subjects:
        condition = next(row["condition"] for row in rows if row["subject_id"] == subject_id)
        figure = plt.figure(figsize=(20, 15), constrained_layout=True)
        grid = figure.add_gridspec(
            2 * len(occupancies), 5, height_ratios=[1.0, 1.5] * len(occupancies)
        )
        for occ_index, occupancy in enumerate(occupancies):
            data = aggregates[(subject_id, occupancy)]
            time = data["time"]
            response = data["mean"] - np.mean(data["mean"][time < 0.0], axis=0)
            trace_axis = figure.add_subplot(grid[2 * occ_index + 1, :])
            for region in range(response.shape[1]):
                trace_axis.plot(time, response[:, region], color=region_colours[region], alpha=0.65, lw=0.75)
            trace_axis.plot(
                time,
                response[:, data["stim_index"]],
                color="black",
                lw=2.3,
                label=str(data["labels"][data["stim_index"]]),
                zorder=10,
            )
            trace_axis.axvline(0.0, color="black", lw=1.0)
            trace_axis.axvspan(0.0, 10.0, color="#F4C95D", alpha=0.22)
            trace_axis.set_xlim(-50.0, 300.0)
            trace_axis.set_ylabel(f"Occupancy {occupancy:g}\nΔ rate (Hz)")
            trace_axis.grid(alpha=0.18)
            trace_axis.legend(frameon=False, loc="upper right", fontsize=8)
            if occ_index == len(occupancies) - 1:
                trace_axis.set_xlabel("Time from stimulation (ms)")
            peaks = _peak_latencies(time, response, n_peaks=5)
            brain_limit = float(np.percentile(np.abs(response[time >= 0.0]), 98.5))
            for map_column, peak in enumerate(peaks):
                peak_index = int(np.argmin(np.abs(time - peak)))
                brain_axis = figure.add_subplot(grid[2 * occ_index, map_column])
                _plot_region_brain(brain_axis, centres, response[peak_index], brain_limit)
                brain_axis.set_title(f"{peak:.0f} ms", fontsize=10)
                trace_axis.axvline(peak, color="#777777", lw=0.65, alpha=0.65)
                figure.add_artist(
                    ConnectionPatch(
                        xyA=(0.5, 0.0), coordsA=brain_axis.transAxes,
                        xyB=(peak, trace_axis.get_ylim()[1]), coordsB=trace_axis.transData,
                        color="#777777", lw=0.65, alpha=0.65,
                    )
                )
        figure.suptitle(
            f"Regional TMS-like propagation across 5-HT2A occupancy — {condition} {subject_id}",
            fontsize=22,
            fontweight="bold",
        )
        for suffix in ("png", "pdf"):
            figure.savefig(
                brain_root / f"{condition.lower()}_{subject_id}_occupancy_butterfly_brainmaps.{suffix}",
                dpi=300,
            )
        plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    subjects = _validate(args)
    atlas = load_aal90_atlas(args.dataset_root)
    labels = np.asarray(atlas.labels).astype(str)
    receptor_map = get_5ht2a_aal90(tracer="cimbi", target_labels=labels)
    receptor_hash = _hash_array(np.asarray(receptor_map, dtype=np.float64))
    simulation = {
        "protocol_version": PROTOCOL_VERSION,
        "subjects": [asdict(subject) for subject in subjects],
        "occupancies": [float(value) for value in args.occupancies],
        "trial_seeds": [int(seed) for seed in args.trial_seeds],
        "coupling_strength": float(args.coupling_strength),
        "conduction_speed_m_s": float(args.conduction_speed_m_s),
        "b_e_pa_shared": float(args.b_e_pa),
        "e_l_e_drug": float(args.e_l_e_drug),
        "e_l_i_drug": float(args.e_l_i_drug),
        "dt_ms": float(args.dt_ms),
        "transient_ms": float(args.transient_ms),
        "analysis_ms": float(args.analysis_ms),
        "monitor_period_ms": float(args.monitor_period_ms),
        "response_start_ms": float(args.response_start_ms),
        "stim_region_label": str(args.stim_region_label),
        "stim_duration_ms": float(args.stim_duration_ms),
        "stim_amplitude_khz": float(args.stim_amplitude_khz),
        "operator": "-[(D-A)/||D-A||F]",
        "receptor_tracer": "cimbi",
        "receptor_map_sha256": receptor_hash,
        "source_entropy_cutoff": None,
    }
    print(json.dumps({**simulation, "total_simulations": len(subjects)*len(args.occupancies)*len(args.trial_seeds)}, indent=2))
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "subject": asdict(subject), "occupancy": occupancy, "seed": seed,
            "output": str(_trial_path(args.output_root, subject, occupancy, seed)),
            "dataset_root": str(args.dataset_root), "simulation": simulation,
            "receptor_map": receptor_map, "receptor_map_sha256": receptor_hash,
            "overwrite": bool(args.overwrite),
        }
        for subject in subjects for occupancy in args.occupancies for seed in args.trial_seeds
    ]
    if args.workers == 1:
        records = [_run_trial(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_initializer) as executor:
            records = list(executor.map(_run_trial, tasks, chunksize=1))

    rows = []
    aggregates = {}
    for subject in subjects:
        weights, _, _, _ = load_subject_structural(subject_id=subject.subject_id, cohort=subject.cohort, dataset_root=args.dataset_root, validate=True, enforce_symmetry=True, zero_diagonal=True)
        missing = 1.0 - np.count_nonzero(np.triu(weights, 1)) / (weights.shape[0]*(weights.shape[0]-1)/2)
        for occupancy in args.occupancies:
            trials, time, region_labels, stim_index = _load_trials(args.output_root, subject, occupancy, args.trial_seeds)
            metrics = _subject_metrics(trials, time, stim_index, args)
            rows.append({**asdict(subject), "missing_edge_fraction": missing, "occupancy": occupancy, **metrics})
            aggregates[(subject.subject_id, occupancy)] = {"time": time, "mean": np.mean(trials, axis=0), "labels": region_labels, "stim_index": stim_index}
    _write_csv(args.output_root / "subject_occupancy_metrics.csv", rows)
    _plot(args.output_root, aggregates, rows)
    manifest = {"simulation": simulation, "pci": {"method": "baseline trial bootstrap", "replicates": args.pci_permutation_replicates, "alpha": args.pci_alpha, "source_entropy_cutoff": None}, "trial_records": records, "metrics": rows}
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed occupancy sweep: {args.output_root}")


if __name__ == "__main__":
    main()
