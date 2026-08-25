#!/usr/bin/env python3
"""Run a matched control/UWS AdEx stimulation through Momi graph Laplacians."""

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

from brain_act_hybrid_common import BASE_PARAMETER_MODEL_NEW  # noqa: E402
from experiments.momi_laplacian_pci.graph_laplacian import (  # noqa: E402
    momi_laplacian,
    validate_tract_lengths,
)
from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial  # noqa: E402
from tvbtoolkit.complexity.pci_st import PCIStResult, pci_st_from_trials  # noqa: E402
from tvbtoolkit.core.config import WholeBrainConfig  # noqa: E402
from tvbtoolkit.datasets.brain_act import (  # noqa: E402
    list_subjects,
    load_aal90_atlas,
    load_subject_structural,
)
from tvbtoolkit.datasets.structural_provenance import (  # noqa: E402
    validate_native_invnodevol_dataset,
)
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: E402
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import (  # noqa: E402
    worker_initializer,
)
from tvbtoolkit.workflows.pharmacology import leak_to_conductances  # noqa: E402


PROTOCOL_VERSION = "momi-operator-adex-pci-1.2-zerlaut-so-khz-v2"
DEFAULT_DATASET = (
    _REPO_ROOT / "data" / "doc_data" / "converted_structural_invnodevol_native"
)
DEFAULT_SUBJECTS = ("control:c0015", "uws:u0001", "uws:u0020")
COHORT_TO_CONDITION = {"control": "CNT", "uws": "UWS"}
COLORS = {"control": "#2E86AB", "uws": "#C0392B"}


@dataclass(frozen=True)
class SubjectSpec:
    cohort: str
    subject_id: str
    condition: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "results" / "momi_laplacian_pci_two_subject",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="One control:id and one or more uws:id selections.",
    )
    parser.add_argument(
        "--connectivity-mode",
        choices=["laplacian", "native"],
        default="laplacian",
        help="Apply Momi's normalized graph Laplacian or native invnodevol weights.",
    )
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--coupling-strength", type=float, default=3.5)
    parser.add_argument("--conduction-speed-m-s", type=float, default=4.0)
    parser.add_argument("--b-e-pa", type=float, default=10.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--transient-ms", type=float, default=4000.0)
    parser.add_argument("--analysis-ms", type=float, default=300.0)
    parser.add_argument("--monitor-period-ms", type=float, default=3.0)
    parser.add_argument("--response-start-ms", type=float, default=8.0)
    parser.add_argument("--stim-region-label", default="Supp_Motor_Area_L")
    parser.add_argument(
        "--pulse-shape",
        choices=["square", "raised_cosine", "gaussian"],
        default="square",
    )
    parser.add_argument("--stim-duration-ms", type=float, default=10.0)
    parser.add_argument("--stim-amplitude-khz", type=float, default=0.0003)
    parser.add_argument(
        "--pci-significance-method",
        choices=["trial_bootstrap", "pre_post_swap", "temporal_shuffle"],
        default="trial_bootstrap",
        help=(
            "PCI-LZ null. The simulation-validated baseline trial bootstrap is "
            "the primary route; the other methods are sensitivity analyses."
        ),
    )
    parser.add_argument("--pci-permutation-replicates", type=int, default=500)
    parser.add_argument("--pci-alpha", type=float, default=0.01)
    parser.add_argument("--pci-random-seed", type=int, default=0)
    parser.set_defaults(pci_min_source_entropy=None)
    parser.add_argument("--pci-st-k", type=float, default=1.2)
    parser.add_argument("--pci-st-min-snr", type=float, default=1.1)
    parser.add_argument("--pci-st-max-var-percent", type=float, default=99.0)
    parser.add_argument("--pci-st-n-steps", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Recalculate metrics/figures from complete saved trials without simulation.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use two seeds and a 100-ms transient; analysis windows remain unchanged.",
    )
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.trial_seeds = [0, 1]
        args.transient_ms = 100.0
        args.pci_permutation_replicates = min(args.pci_permutation_replicates, 100)
    return args


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> list[SubjectSpec]:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_native_invnodevol_dataset(args.dataset_root)
    specs = args.subject or list(DEFAULT_SUBJECTS)
    if len(specs) < 2:
        raise ValueError("at least two subjects are required")
    available = list_subjects(args.dataset_root, cohort=None)
    subjects: list[SubjectSpec] = []
    for raw in specs:
        try:
            cohort, subject_id = raw.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"subject must be cohort:id, got {raw!r}") from exc
        cohort = cohort.strip().lower()
        subject_id = subject_id.strip()
        if cohort not in COHORT_TO_CONDITION:
            raise ValueError("subjects must contain only control and uws cohorts")
        if subject_id not in available.get(cohort, []):
            raise KeyError(f"unknown subject {cohort}:{subject_id}")
        subjects.append(SubjectSpec(cohort, subject_id, COHORT_TO_CONDITION[cohort]))
    if sum(subject.cohort == "control" for subject in subjects) != 1 or not any(
        subject.cohort == "uws" for subject in subjects
    ):
        raise ValueError("selection must contain one control and at least one UWS subject")
    if not args.trial_seeds or len(set(args.trial_seeds)) != len(args.trial_seeds):
        raise ValueError("trial seeds must be a non-empty unique list")
    positive = [
        args.workers,
        args.coupling_strength,
        args.conduction_speed_m_s,
        args.dt_ms,
        args.analysis_ms,
        args.monitor_period_ms,
        args.stim_duration_ms,
        args.pci_permutation_replicates,
    ]
    if any(value <= 0 for value in positive) or args.transient_ms < 0:
        raise ValueError("simulation durations, scales, and counts must be positive")
    if not np.isclose(
        args.monitor_period_ms / args.dt_ms,
        round(args.monitor_period_ms / args.dt_ms),
        atol=1e-12,
    ):
        raise ValueError("monitor period must be an integer multiple of dt")
    if not 0 <= args.response_start_ms < args.analysis_ms:
        raise ValueError("response start must lie inside the post-stimulation window")
    if args.pci_min_source_entropy is not None:
        raise ValueError("simulation PCI must run without a source-entropy cutoff")
    return subjects


def _baseline_split_model(n_regions: int, b_e_pa: float) -> dict[str, Any]:
    """Occupancy-zero v6 model with identical physiology in both cohorts."""

    model = deepcopy(BASE_PARAMETER_MODEL_NEW)
    model["b_e"] = float(b_e_pa)
    g_k_e, g_na_e = leak_to_conductances(
        50.0, -90.0, float(model["E_L_e"]), g_L=10.0
    )
    g_k_i, g_na_i = leak_to_conductances(
        50.0, -90.0, float(model["E_L_i"]), g_L=10.0
    )
    model.update(
        {
            "g_K_e": np.full(n_regions, g_k_e).tolist(),
            "g_Na_e": float(g_na_e),
            "g_K_i": np.full(n_regions, g_k_i).tolist(),
            "g_Na_i": float(g_na_i),
            "noise_alpha": 0.0,
            "shared_noise_mode": "none",
        }
    )
    return model


def _load_transformed_subject(
    subject: SubjectSpec, dataset_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    weights, lengths, atlas, metadata = load_subject_structural(
        subject_id=subject.subject_id,
        cohort=subject.cohort,
        dataset_root=dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    length_diagnostics = validate_tract_lengths(lengths, weights)
    transformed, laplacian_diagnostics = momi_laplacian(weights)
    diagnostics = {
        "subject": asdict(subject),
        "dataset_index": int(metadata.dataset_index),
        "adjacency_sha256": _sha256_array(np.asarray(weights, dtype=np.float64)),
        "tract_lengths_sha256": _sha256_array(np.asarray(lengths, dtype=np.float64)),
        "laplacian_sha256": _sha256_array(transformed),
        "tvb_coupling_operator_sha256": _sha256_array(-transformed),
        "laplacian": laplacian_diagnostics.to_dict(),
        "tract_lengths": length_diagnostics,
    }
    return transformed, np.asarray(lengths), np.asarray(atlas.labels), diagnostics


def _simulation_payload(args: argparse.Namespace, subjects: list[SubjectSpec]) -> dict[str, Any]:
    is_laplacian = args.connectivity_mode == "laplacian"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "subjects": [asdict(subject) for subject in subjects],
        "trial_seeds": [int(seed) for seed in args.trial_seeds],
        "dt_ms": float(args.dt_ms),
        "transient_ms": float(args.transient_ms),
        "analysis_ms": float(args.analysis_ms),
        "monitor_period_ms": float(args.monitor_period_ms),
        "response_start_ms": float(args.response_start_ms),
        "coupling_strength": float(args.coupling_strength),
        "connectivity_mode": str(args.connectivity_mode),
        "conduction_speed_m_s": float(args.conduction_speed_m_s),
        "b_e_pa": float(args.b_e_pa),
        "stim_region_label": str(args.stim_region_label),
        "pulse_shape": str(args.pulse_shape),
        "stim_duration_ms": float(args.stim_duration_ms),
        "stim_amplitude_khz": float(args.stim_amplitude_khz),
        "graph_laplacian": "L = (diag(sum(A, axis=1)) - A) / FrobeniusNorm",
        "tvb_coupling_operator": (
            "-L, reproducing A*x_delayed - D*x_current"
            if is_laplacian
            else "native volume-normalized adjacency A"
        ),
        "tract_length_role": "delays_only; delay_ms = length_mm / speed_m_per_s",
        "connectivity_diagonal_policy": (
            "preserve_negative_laplacian_diagonal" if is_laplacian else "zero"
        ),
    }


def _trial_path(output_root: Path, subject: SubjectSpec, seed: int) -> Path:
    return output_root / "trials" / subject.cohort / subject.subject_id / f"seed_{seed:03d}.npz"


def _run_trial(task: dict[str, Any]) -> dict[str, Any]:
    subject = SubjectSpec(**task["subject"])
    dataset_root = Path(task["dataset_root"])
    output_path = Path(task["output_path"])
    args = task["simulation"]
    expected_fingerprint = task["fingerprint"]
    if output_path.is_file() and not task["overwrite"]:
        with np.load(output_path, allow_pickle=False) as cached:
            fingerprint = str(np.asarray(cached["fingerprint"]).reshape(-1)[0])
            if fingerprint != expected_fingerprint:
                raise RuntimeError(
                    f"stale trial fingerprint at {output_path}; use --overwrite"
                )
            return {
                "subject": asdict(subject),
                "seed": int(task["seed"]),
                "path": str(output_path),
                "cached": True,
            }

    laplacian, lengths, labels, _ = _load_transformed_subject(subject, dataset_root)
    if args["connectivity_mode"] == "laplacian":
        coupling_operator = -laplacian
        preserve_diagonal = True
    else:
        coupling_operator, _, _, _ = load_subject_structural(
            subject_id=subject.subject_id,
            cohort=subject.cohort,
            dataset_root=dataset_root,
            validate=True,
            enforce_symmetry=True,
            zero_diagonal=True,
        )
        coupling_operator = np.asarray(coupling_operator, dtype=float)
        preserve_diagonal = False
    stim_index = list(labels.astype(str)).index(args["stim_region_label"])
    onset_ms = float(args["transient_ms"] + args["analysis_ms"])
    total_ms = float(onset_ms + args["analysis_ms"])
    model = _baseline_split_model(laplacian.shape[0], args["b_e_pa"])
    stimulus = {
        "stimtime": onset_ms,
        "stimdur": args["stim_duration_ms"],
        "stimperiod": total_ms * 10.0,
        "stimval": args["stim_amplitude_khz"],
        "stimregion": [stim_index],
        "stimvariables": [0],
        "stimshape": args["pulse_shape"],
    }
    cfg = WholeBrainConfig(
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
        weights=coupling_operator,
        tract_lengths=lengths,
        connectivity_normalization="none",
        parameter_overrides={
            "parameter_model": model,
            "parameter_stimulus": stimulus,
            # The legacy route normally zeros the diagonal. Momi's -D
            # self-term is part of the applied -L operator and must stay.
            "nullify_diagonals": not preserve_diagonal,
        },
    )
    result = run_whole_brain_simulation(cfg, seed=int(task["seed"]))
    time_ms = np.asarray(result.time_ms, dtype=float)
    rate_hz = np.asarray(result.raw, dtype=float) * 1000.0
    inh_hz = np.asarray(result.raw_inh, dtype=float) * 1000.0
    if rate_hz.shape != (time_ms.size, labels.size) or not np.isfinite(rate_hz).all():
        raise RuntimeError("AdEx simulation produced an invalid rate array")
    peri = (time_ms >= onset_ms - args["analysis_ms"]) & (
        time_ms < onset_ms + args["analysis_ms"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fingerprint=np.asarray([expected_fingerprint], dtype="U128"),
        protocol_version=np.asarray([PROTOCOL_VERSION], dtype="U64"),
        cohort=np.asarray([subject.cohort], dtype="U16"),
        subject_id=np.asarray([subject.subject_id], dtype="U32"),
        seed=np.asarray([int(task["seed"])], dtype=np.int64),
        stim_region_index=np.asarray([stim_index], dtype=np.int64),
        stim_region_label=np.asarray([args["stim_region_label"]], dtype="U128"),
        onset_ms=np.asarray([onset_ms]),
        time_relative_ms=time_ms[peri] - onset_ms,
        rate_hz=rate_hz[peri],
        inhibitory_rate_hz=inh_hz[peri],
        region_labels=labels.astype("U128"),
    )
    return {
        "subject": asdict(subject),
        "seed": int(task["seed"]),
        "path": str(output_path),
        "cached": False,
    }


def _load_trials(
    output_root: Path, subject: SubjectSpec, seeds: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    arrays = []
    time = labels = None
    stim_index = -1
    for seed in seeds:
        with np.load(_trial_path(output_root, subject, seed), allow_pickle=False) as data:
            this_time = np.asarray(data["time_relative_ms"], dtype=float)
            this_labels = np.asarray(data["region_labels"]).astype(str)
            if time is None:
                time, labels = this_time, this_labels
                stim_index = int(np.asarray(data["stim_region_index"]).reshape(-1)[0])
            else:
                np.testing.assert_allclose(this_time, time, rtol=0.0, atol=1e-9)
                np.testing.assert_array_equal(this_labels, labels)
            arrays.append(np.asarray(data["rate_hz"], dtype=float))
    return np.stack(arrays), np.asarray(time), np.asarray(labels), stim_index


def _subject_metrics(
    trials_hz: np.ndarray,
    time_ms: np.ndarray,
    stim_index: int,
    args: argparse.Namespace,
) -> dict[str, float | int | bool | str]:
    mean_hz = np.mean(trials_hz, axis=0)
    baseline_mask = time_ms < 0.0
    response_mask = (time_ms >= args.response_start_ms) & (time_ms < args.analysis_ms)
    late_mask = (time_ms >= args.analysis_ms - 50.0) & (time_ms < args.analysis_ms)
    baseline = np.mean(mean_hz[baseline_mask], axis=0)
    baseline_sd = np.std(mean_hz[baseline_mask], axis=0, ddof=1)
    delta = mean_hz - baseline
    thresholds = np.maximum(3.0 * baseline_sd, 0.5)
    dt_ms = float(np.median(np.diff(time_ms)))
    onset_index = int(np.argmin(np.abs(time_ms)))
    lz_debug = pci_casali_like_multi_trial(
        [trial / 1000.0 for trial in trials_hz],
        stimulation_index=onset_index,
        t_analysis_ms=float(args.analysis_ms),
        dt_ms=dt_ms,
        binarise_method="casali",
        binarise_kwargs={
            "n_bootstrap": int(args.pci_permutation_replicates),
            "alpha": float(args.pci_alpha),
            "seed": int(args.pci_random_seed),
            "significance_method": str(args.pci_significance_method),
        },
        response_start_ms=float(args.response_start_ms),
        min_source_entropy=args.pci_min_source_entropy,
        return_debug=True,
    )
    if not isinstance(lz_debug, dict):
        raise AssertionError("PCI-LZ debug diagnostics were not returned")
    np.random.seed(int(args.pci_random_seed))
    tvbsim_debug = pci_casali_like_multi_trial(
        [trial / 1000.0 for trial in trials_hz],
        stimulation_index=onset_index,
        t_analysis_ms=float(args.analysis_ms),
        dt_ms=dt_ms,
        nshuffles=10,
        percentile=100.0,
        binarise_method="tvbsim",
        response_start_ms=0.0,
        min_source_entropy=None,
        return_debug=True,
    )
    if not isinstance(tvbsim_debug, dict):
        raise AssertionError("TVBSim PCI-LZ debug diagnostics were not returned")
    st = pci_st_from_trials(
        trials_hz.transpose(0, 2, 1) / 1000.0,
        time_ms,
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
        raise AssertionError("PCI-ST detailed result was not returned")
    return {
        "n_trials": int(trials_hz.shape[0]),
        "baseline_mean_hz": float(np.mean(baseline)),
        "peak_delta_hz": float(np.max(delta[response_mask])),
        "stimulated_region_peak_delta_hz": float(
            np.max(delta[response_mask, stim_index])
        ),
        "propagated_regions": int(
            np.count_nonzero(np.max(delta[response_mask], axis=0) > thresholds)
        ),
        "late_residual_hz": float(np.mean(np.abs(delta[late_mask]))),
        "peak_absolute_hz": float(np.max(mean_hz[time_ms >= 0.0])),
        "pci_lz_significance_method": str(lz_debug["significance_method"]),
        "pci_lz_null_replicates": int(lz_debug["n_surrogates"]),
        "pci_lz_alpha": float(lz_debug["alpha"]),
        "pci_lz_threshold": float(lz_debug["threshold"]),
        "pci_lz": float(lz_debug["pci"]),
        "pci_lz_calculated_complexity": float(
            lz_debug["pci_calculated_complexity"]
        ),
        "pci_lz_active_fraction": float(lz_debug["active_fraction"]),
        "pci_lz_source_entropy": float(lz_debug["entropy"]),
        "pci_lz_reliability_check_failed": bool(
            lz_debug["low_activation_forced_zero"]
        ),
        "tvbsim_pci_lz": float(tvbsim_debug["pci"]),
        "tvbsim_pci_lz_trial_sd": float(np.std(tvbsim_debug["pci_values"], ddof=1)),
        "tvbsim_pci_lz_trial_min": float(np.min(tvbsim_debug["pci_values"])),
        "tvbsim_pci_lz_trial_max": float(np.max(tvbsim_debug["pci_values"])),
        "pci_st": float(st.pci_st),
        "pci_st_n_components": int(st.n_components),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figures(
    aggregates: dict[str, dict[str, Any]], metrics: list[dict[str, Any]], output: Path
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    n_subjects = len(aggregates)
    fig, axes = plt.subplots(
        n_subjects, 3, figsize=(12, 3.1 * n_subjects), constrained_layout=True,
        squeeze=False,
    )
    for row, (key, data) in enumerate(aggregates.items()):
        cohort = data["subject"].cohort
        color = COLORS[cohort]
        time = data["time_ms"]
        mean = data["mean_hz"]
        baseline = np.mean(mean[time < 0.0], axis=0)
        delta = mean - baseline
        axes[row, 0].plot(time, np.mean(delta, axis=1), color=color, lw=1.2)
        axes[row, 0].axvline(0, color="black", lw=0.7, ls="--")
        axes[row, 0].axvspan(0, data["stim_duration_ms"], color="#F4C95D", alpha=0.3)
        axes[row, 0].set_ylabel(f"{cohort.upper()}\nmean Δ rate (Hz)")
        axes[row, 1].plot(time, delta[:, data["stim_index"]], color=color, lw=1.2)
        axes[row, 1].axvline(0, color="black", lw=0.7, ls="--")
        image = axes[row, 2].imshow(
            delta.T,
            aspect="auto",
            origin="lower",
            extent=[time[0], time[-1], 0, delta.shape[1]],
            cmap="RdBu_r",
            vmin=-max(1.0, float(np.percentile(np.abs(delta), 99))),
            vmax=max(1.0, float(np.percentile(np.abs(delta), 99))),
        )
        axes[row, 2].axvline(0, color="black", lw=0.7, ls="--")
        fig.colorbar(image, ax=axes[row, 2], label="Δ rate (Hz)", shrink=0.8)
    axes[0, 0].set_title("Whole-brain response")
    axes[0, 1].set_title("Stimulated left SMA")
    axes[0, 2].set_title("Regional response")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time from stimulation (ms)")
    axes[-1, 2].set_ylabel("AAL90 region index")
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"subject_evoked_time_courses.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), constrained_layout=True)
    labels = [f'{row["cohort"].upper()}\n{row["subject_id"]}' for row in metrics]
    colors = [COLORS[row["cohort"]] for row in metrics]
    calculated = [row["pci_lz"] for row in metrics]
    tvbsim = [row["tvbsim_pci_lz"] for row in metrics]
    entropy = [row["pci_lz_source_entropy"] for row in metrics]
    pci_st = [row["pci_st"] for row in metrics]

    axes[0, 0].bar(labels, calculated, color=colors)
    axes[0, 0].set_ylabel("PCI-LZ")
    axes[0, 0].set_title("Current: trial-average response")
    axes[0, 1].bar(labels, entropy, color=colors)
    axes[0, 1].set_ylabel("Information in response pattern")
    axes[0, 1].set_title("Response entropy (diagnostic only)")
    axes[1, 0].bar(labels, tvbsim, color=colors)
    axes[1, 0].set_ylabel("PCI-LZ")
    axes[1, 0].set_title("Original TVBSim: mean of trial PCIs")
    axes[1, 1].bar(labels, pci_st, color=colors)
    axes[1, 1].set_ylabel("PCI-ST")
    axes[1, 1].set_title("Alternative complexity measure (PCI-ST)")
    for axis, values in zip(
        axes.ravel(), [calculated, entropy, tvbsim, pci_st], strict=True
    ):
        for index, value in enumerate(values):
            axis.text(
                index,
                value,
                f"{value:.3f}" if value < 1.0 else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("PCI estimators; no source-entropy cutoff", fontsize=14)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"subject_pci_comparison.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(
        n_subjects, 2, figsize=(7, 3 * n_subjects), constrained_layout=True,
        squeeze=False,
    )
    for row, data in enumerate(aggregates.values()):
        adjacency = data["adjacency"]
        laplacian = data["laplacian"]
        axes[row, 0].imshow(np.log1p(adjacency), cmap="magma", aspect="equal")
        lim = float(np.max(np.abs(laplacian)))
        axes[row, 1].imshow(laplacian, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        axes[row, 0].set_ylabel(data["subject"].cohort.upper())
    axes[0, 0].set_title("log(1 + native invnodevol A)")
    axes[0, 1].set_title("Graph Laplacian (D − A) / ||D − A||F")
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"subject_connectomes_and_laplacians.{extension}", dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    subjects = _validate_args(args)
    atlas = load_aal90_atlas(args.dataset_root)
    labels = list(np.asarray(atlas.labels).astype(str))
    if args.stim_region_label not in labels:
        raise KeyError(f"{args.stim_region_label!r} is absent from the dataset atlas")
    stim_index = labels.index(args.stim_region_label)
    provenance = validate_native_invnodevol_dataset(args.dataset_root)
    transform_diagnostics: list[dict[str, Any]] = []
    for subject in subjects:
        _, _, _, diagnostics = _load_transformed_subject(subject, args.dataset_root)
        transform_diagnostics.append(diagnostics)
    simulation = _simulation_payload(args, subjects)
    fingerprint = hashlib.sha256(_canonical_json(simulation).encode()).hexdigest()
    plan = {
        "subjects": [f"{subject.cohort}:{subject.subject_id}" for subject in subjects],
        "stim_region_label": args.stim_region_label,
        "stim_region_index_zero_based": stim_index,
        "trials_per_subject": len(args.trial_seeds),
        "total_simulations": len(subjects) * len(args.trial_seeds),
        "workers": args.workers,
        "simulation_fingerprint": fingerprint,
        "transform_diagnostics": transform_diagnostics,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "run_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("simulation_fingerprint") != fingerprint:
            raise RuntimeError("output root contains a different run; use --overwrite")

    tasks = [
        {
            "subject": asdict(subject),
            "dataset_root": str(args.dataset_root),
            "output_path": str(_trial_path(args.output_root, subject, seed)),
            "seed": int(seed),
            "simulation": simulation,
            "fingerprint": fingerprint,
            "overwrite": bool(args.overwrite),
        }
        for subject in subjects
        for seed in args.trial_seeds
    ]
    if args.analysis_only:
        missing = [task["output_path"] for task in tasks if not Path(task["output_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"analysis-only requested but {len(missing)} trials are missing")
        trial_records = [
            {
                "subject": task["subject"],
                "seed": task["seed"],
                "path": task["output_path"],
                "cached": True,
            }
            for task in tasks
        ]
    elif args.workers == 1:
        trial_records = [_run_trial(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=worker_initializer
            ) as executor:
                trial_records = list(executor.map(_run_trial, tasks, chunksize=1))
        except PermissionError as exc:
            # Some restricted workstation sandboxes prohibit POSIX semaphores.
            # The scientific result is unchanged by a serial fallback.
            print(f"Parallel workers unavailable ({exc}); falling back to one worker.")
            trial_records = [_run_trial(task) for task in tasks]

    metric_rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        trials, time_ms, region_labels, this_stim_index = _load_trials(
            args.output_root, subject, args.trial_seeds
        )
        metrics = _subject_metrics(trials, time_ms, this_stim_index, args)
        row = {
            **asdict(subject),
            "connectivity_mode": args.connectivity_mode,
            "coupling_strength": args.coupling_strength,
            **metrics,
        }
        metric_rows.append(row)
        weights, _, _, _ = load_subject_structural(
            subject_id=subject.subject_id,
            cohort=subject.cohort,
            dataset_root=args.dataset_root,
            validate=True,
            enforce_symmetry=True,
            zero_diagonal=True,
        )
        laplacian, _ = momi_laplacian(weights)
        key = f"{subject.cohort}:{subject.subject_id}"
        aggregates[key] = {
            "subject": subject,
            "time_ms": time_ms,
            "mean_hz": np.mean(trials, axis=0),
            "trials_hz": trials,
            "labels": region_labels,
            "stim_index": this_stim_index,
            "stim_duration_ms": args.stim_duration_ms,
            "adjacency": weights,
            "laplacian": laplacian,
        }
        aligned_dir = args.output_root / "aligned_time_courses"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            aligned_dir / f"{subject.cohort}_{subject.subject_id}.npz",
            time_relative_ms=time_ms,
            mean_rate_hz=np.mean(trials, axis=0),
            trial_rates_hz=trials,
            region_labels=region_labels.astype("U128"),
            stim_region_index=np.asarray([this_stim_index]),
            pci_lz=np.asarray([metrics["pci_lz"]]),
            tvbsim_pci_lz=np.asarray([metrics["tvbsim_pci_lz"]]),
            pci_st=np.asarray([metrics["pci_st"]]),
            pci_lz_significance_method=np.asarray(
                [args.pci_significance_method], dtype="U32"
            ),
        )
    _write_csv(args.output_root / "subject_metrics.csv", metric_rows)
    _save_figures(aggregates, metric_rows, args.output_root)
    manifest = {
        "purpose": "three-subject operator/PCI-method robustness check; not an inferential cohort result",
        "protocol_version": PROTOCOL_VERSION,
        "simulation_fingerprint": fingerprint,
        "dataset": provenance,
        "simulation": simulation,
        "transform_diagnostics": transform_diagnostics,
        "pci": {
            "estimators": ["current trial-average PCI-LZ", "TVBSim PCI-LZ", "PCI-ST"],
            "matched_trial_average": True,
            "response_start_ms": args.response_start_ms,
            "lz_significance_method": args.pci_significance_method,
            "lz_null_role": (
                "primary"
                if args.pci_significance_method == "trial_bootstrap"
                else "sensitivity_analysis"
            ),
            "lz_bootstrap_or_permutation_replicates": (
                args.pci_permutation_replicates
            ),
            "lz_alpha": args.pci_alpha,
            "lz_random_seed": args.pci_random_seed,
            "lz_min_source_entropy": None,
            "source_entropy_policy": "diagnostic_only; never forces PCI to zero",
            "tvbsim_nshuffles": 10,
            "tvbsim_percentile": 100,
            "tvbsim_aggregation": "calculate one PCI per trial, then average",
            "st_k": args.pci_st_k,
            "st_min_snr": args.pci_st_min_snr,
            "st_max_var_percent": args.pci_st_max_var_percent,
            "st_n_steps": args.pci_st_n_steps,
        },
        "trial_records": trial_records,
        "subject_metrics": metric_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed robustness run: {args.output_root}")


if __name__ == "__main__":
    main()
