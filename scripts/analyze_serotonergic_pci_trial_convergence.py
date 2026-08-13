#!/usr/bin/env python3
"""Read-only nested trial-count convergence analysis for serotonergic PCI.

The corrected 100-trial simulation cache is treated as an immutable input.
For an outcome-blind, cohort-stratified subject sample, the same deterministic
nested trial subsets are applied at every occupancy.  Each subset yields:

* PCI-LZ: Casali-style Lempel-Ziv PCI using the canonical within-trial
  pre/post-block swap null; and
* PCI-ST: state-transition PCI from the continuous trial-averaged response.

The 100-trial estimate is evaluated once per subject and occupancy.  Smaller
20/40/60/80-trial estimates are repeated to describe convergence; overlapping
nested subsets are descriptive resamples, not independent observations.
Nothing is ever written beneath ``--cache-root``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable
import uuid

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for _path in (_SCRIPT_DIR, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_serotonergic_pci_pilot as pilot
from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial
from tvbtoolkit.complexity.pci_st import PCIStResult, pci_st_from_trials


PROTOCOL_VERSION = "1.0-cache-only-nested-pci-lz-pci-st"
CANONICAL_TRIAL_SEEDS = tuple(range(100))
CANONICAL_OCCUPANCIES = (0.0, 0.25, 0.5, 0.766)
DEFAULT_COHORTS = ("coma", "uws", "mcs", "emcs", "control")
DEFAULT_SUBSET_SIZES = (20, 40, 60, 80)


@dataclass(frozen=True)
class SelectedSubject:
    """Manifest-backed subject selected without consulting PCI outcomes."""

    cohort: str
    condition: str
    subject_id: str
    selection_rank: int
    selection_score: str
    selection_method: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=(
            _REPO_ROOT
            / "notebooks"
            / "outputs"
            / "serotonergic_pci_full_100trials_corrected"
        ),
        help="Immutable root produced by run_serotonergic_pci_full.py.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Defaults to CACHE_ROOT/logs/run_manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            _REPO_ROOT
            / "notebooks"
            / "outputs"
            / "serotonergic_pci_trial_convergence"
        ),
    )
    parser.add_argument("--cohorts", nargs="+", default=list(DEFAULT_COHORTS))
    parser.add_argument("--subjects-per-cohort", type=int, default=1)
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Explicit cohort:subject_id; may be repeated.",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUBSET_SIZES),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--pci-significance-method",
        choices=["pre_post_swap", "temporal_shuffle", "trial_bootstrap"],
        default="pre_post_swap",
    )
    parser.add_argument(
        "--pci-permutation-replicates",
        "--pci-bootstrap-replicates",
        dest="pci_permutation_replicates",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--pci-alpha",
        type=float,
        default=pilot.DEFAULT_PCI_ALPHA,
    )
    parser.add_argument("--pci-seed", type=int, default=0)
    parser.add_argument("--pci-response-start-ms", type=float, default=8.0)
    parser.add_argument(
        "--pci-min-source-entropy",
        type=float,
        default=0.08,
        help="Canonical low-activation entropy floor for PCI-LZ.",
    )
    parser.add_argument(
        "--pci-st-baseline-window-ms",
        type=float,
        nargs=2,
        default=[-300.0, -50.0],
        metavar=("START", "STOP"),
        help="Model-cache baseline window; upper bound is exclusive.",
    )
    parser.add_argument(
        "--pci-st-response-window-ms",
        type=float,
        nargs=2,
        default=[8.0, 300.0],
        metavar=("START", "STOP"),
        help=(
            "PCI-ST response window. This simulation protocol begins at 8 ms "
            "to exclude the imposed stimulation pulse/artefact."
        ),
    )
    parser.add_argument("--pci-st-k", type=float, default=1.2)
    parser.add_argument("--pci-st-min-snr", type=float, default=1.1)
    parser.add_argument("--pci-st-max-var-percent", type=float, default=99.0)
    parser.add_argument("--pci-st-n-steps", type=int, default=100)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source manifest and write the deterministic plan only.",
    )
    return parser.parse_args(argv)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_uint64(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_cli(args: argparse.Namespace) -> None:
    if int(args.subjects_per_cohort) < 1:
        raise ValueError("--subjects-per-cohort must be >= 1.")
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be >= 1.")
    if int(args.workers) < 1:
        raise ValueError("--workers must be >= 1.")
    sizes = [int(value) for value in args.subset_sizes]
    if sizes != sorted(set(sizes)):
        raise ValueError("--subset-sizes must be unique and strictly increasing.")
    if not sizes or sizes[0] < 2 or sizes[-1] >= 100:
        raise ValueError("--subset-sizes must lie between 2 and 99.")
    if int(args.pci_permutation_replicates) < 1:
        raise ValueError("--pci-permutation-replicates must be >= 1.")
    if not 0.0 < float(args.pci_alpha) < 1.0:
        raise ValueError("--pci-alpha must be between 0 and 1.")
    baseline = tuple(float(value) for value in args.pci_st_baseline_window_ms)
    response = tuple(float(value) for value in args.pci_st_response_window_ms)
    if baseline[0] >= baseline[1] or response[0] >= response[1]:
        raise ValueError("PCI-ST windows must have START < STOP.")
    if baseline[1] > response[0]:
        raise ValueError("PCI-ST baseline and response windows must not overlap.")
    canonical_lz = (
        str(args.pci_significance_method) == "pre_post_swap"
        and int(args.pci_permutation_replicates) == 1000
        and np.isclose(
            float(args.pci_alpha),
            pilot.DEFAULT_PCI_ALPHA,
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            float(args.pci_response_start_ms), 8.0, rtol=0.0, atol=1e-12
        )
        and np.isclose(
            float(args.pci_min_source_entropy), 0.08, rtol=0.0, atol=1e-12
        )
    )
    if not canonical_lz:
        raise ValueError(
            "This production convergence workflow locks PCI-LZ to "
            "pre_post_swap, 1000 permutations, "
            f"alpha={pilot.DEFAULT_PCI_ALPHA:g}, response start 8 ms, and "
            "minimum source entropy=.08."
        )


def _assert_separate_output(cache_root: Path, output_root: Path) -> None:
    cache = cache_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if output == cache or cache in output.parents or output in cache.parents:
        raise ValueError(
            "--output-root must be a separate sibling location: neither path "
            f"may contain the other (cache root: {cache})."
        )


def _load_source_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Source run manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read source run manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Source run manifest must contain a JSON object.")
    return manifest, _sha256_file(path)


def _validate_source_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "script",
        "protocol_version",
        "protocol_fingerprint",
        "subjects",
        "trial_seeds",
        "n_trials",
        "stim_onsets_ms_by_trial_seed",
        "occupancies",
        "simulate_baseline",
        "scenario",
        "t_analysis_ms",
        "atlas_labels_sha256",
        "receptor_tracer",
        "receptor_csv_sha256",
        "receptor_map_sha256",
        "stim_region_labels",
        "model_form",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Source manifest lacks required fields: {missing}")
    if Path(str(manifest["script"])).name != "run_serotonergic_pci_full.py":
        raise ValueError("Source cache must come from run_serotonergic_pci_full.py.")
    seeds = [int(value) for value in manifest["trial_seeds"]]
    if seeds != list(CANONICAL_TRIAL_SEEDS) or int(manifest["n_trials"]) != 100:
        raise ValueError("Source cache must contain ordered trial seeds 0..99.")
    occupancies = np.asarray(manifest["occupancies"], dtype=float)
    if occupancies.shape != (4,) or not np.allclose(
        occupancies,
        np.asarray(CANONICAL_OCCUPANCIES),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Source cache must contain occupancies 0, .25, .5, .766.")
    if not bool(manifest["simulate_baseline"]):
        raise ValueError("Source occupancy zero must have been simulated fresh.")
    if str(manifest["model_form"]) != "split_gK_gNa_all_occupancies":
        raise ValueError("Source cache must use one split-gK/gNa model form.")
    if not np.isclose(float(manifest["t_analysis_ms"]), 300.0):
        raise ValueError("Source cache must contain the corrected ±300-ms epochs.")

    subjects = manifest["subjects"]
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("Source manifest contains no subjects.")
    keys: set[tuple[str, str]] = set()
    for record in subjects:
        if not isinstance(record, dict):
            raise ValueError("Every source subject record must be a JSON object.")
        for field in ("cohort", "condition", "subject_id"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"Source subject record lacks {field!r}: {record}")
        key = (str(record["cohort"]).lower(), str(record["subject_id"]))
        if key in keys:
            raise ValueError(f"Duplicate source subject: {key[0]}:{key[1]}")
        keys.add(key)
    if "n_subjects" in manifest and int(manifest["n_subjects"]) != len(subjects):
        raise ValueError("Source manifest n_subjects disagrees with subjects.")
    onset_map = manifest["stim_onsets_ms_by_trial_seed"]
    if not isinstance(onset_map, dict) or any(str(seed) not in onset_map for seed in seeds):
        raise ValueError("Source manifest lacks stimulation onsets for all seeds.")


def select_subjects(
    manifest: dict[str, Any],
    *,
    cohorts: list[str],
    subjects_per_cohort: int,
    selection_seed: int,
    explicit_subjects: list[str] | None,
) -> list[SelectedSubject]:
    """Select manifest subjects deterministically without using outcomes."""

    records = {
        (str(item["cohort"]).lower(), str(item["subject_id"])): item
        for item in manifest["subjects"]
    }
    cohort_order = [str(cohort).lower() for cohort in cohorts]
    if len(cohort_order) != len(set(cohort_order)):
        raise ValueError("--cohorts contains duplicates.")

    selected: list[SelectedSubject] = []
    if explicit_subjects:
        seen: set[tuple[str, str]] = set()
        for raw in explicit_subjects:
            if str(raw).count(":") != 1:
                raise ValueError(f"Expected cohort:subject_id, got {raw!r}.")
            cohort, subject_id = str(raw).split(":", 1)
            key = (cohort.lower(), subject_id)
            if key in seen:
                raise ValueError(f"Duplicate explicit subject: {raw}")
            if key not in records:
                raise KeyError(f"Subject is absent from source manifest: {raw}")
            if key[0] not in cohort_order:
                raise ValueError(f"Explicit subject cohort not in --cohorts: {raw}")
            seen.add(key)
            record = records[key]
            selected.append(
                SelectedSubject(
                    cohort=key[0],
                    condition=str(record["condition"]),
                    subject_id=key[1],
                    selection_rank=1,
                    selection_score="explicit",
                    selection_method="explicit",
                )
            )
        return selected

    for cohort in cohort_order:
        candidates = [
            item
            for (item_cohort, _), item in records.items()
            if item_cohort == cohort
        ]
        if len(candidates) < int(subjects_per_cohort):
            raise ValueError(
                f"Cohort {cohort!r} has {len(candidates)} subjects; "
                f"{subjects_per_cohort} requested."
            )
        ranked = sorted(
            candidates,
            key=lambda item: (
                _stable_uint64(
                    "subject-selection",
                    int(selection_seed),
                    cohort,
                    str(item["subject_id"]),
                ),
                str(item["subject_id"]),
            ),
        )
        for rank, record in enumerate(ranked[: int(subjects_per_cohort)], start=1):
            score = _stable_uint64(
                "subject-selection",
                int(selection_seed),
                cohort,
                str(record["subject_id"]),
            )
            selected.append(
                SelectedSubject(
                    cohort=cohort,
                    condition=str(record["condition"]),
                    subject_id=str(record["subject_id"]),
                    selection_rank=rank,
                    selection_score=f"{score:016x}",
                    selection_method="sha256_rank_within_cohort",
                )
            )
    return selected


def build_nested_subsets(
    trial_seeds: list[int],
    *,
    subset_sizes: list[int],
    repeats: int,
    subset_seed: int,
    cohort: str,
    subject_id: str,
) -> list[dict[str, Any]]:
    """Return deterministic nested subsets, plus one unrepeated full subset."""

    seeds = [int(value) for value in trial_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("trial_seeds must be unique.")
    sizes = [int(value) for value in subset_sizes]
    if sizes != sorted(set(sizes)) or any(size >= len(seeds) for size in sizes):
        raise ValueError("subset_sizes must be increasing and smaller than total.")

    subsets: list[dict[str, Any]] = []
    for repeat in range(1, int(repeats) + 1):
        derived_seed = _stable_uint64(
            "nested-trial-subsets",
            int(subset_seed),
            str(cohort),
            str(subject_id),
            repeat,
        )
        permutation = np.random.default_rng(derived_seed).permutation(seeds)
        for size in sizes:
            chosen_in_draw_order = [int(value) for value in permutation[:size]]
            subsets.append(
                {
                    "repeat": repeat,
                    "n_trials": size,
                    "subset_kind": "nested_resample",
                    "permutation_seed_uint64": int(derived_seed),
                    "trial_seeds": sorted(chosen_in_draw_order),
                    "draw_order": chosen_in_draw_order,
                }
            )
    subsets.append(
        {
            "repeat": 0,
            "n_trials": len(seeds),
            "subset_kind": "full_once",
            "permutation_seed_uint64": None,
            "trial_seeds": list(seeds),
            "draw_order": list(seeds),
        }
    )
    return subsets


def _trial_path(
    cache_root: Path,
    *,
    occupancy: float,
    scenario: str,
    cohort: str,
    subject_id: str,
    trial_seed: int,
) -> Path:
    return (
        cache_root
        / "sims_pci"
        / pilot._occ_tag(float(occupancy))
        / str(scenario)
        / str(cohort)
        / str(subject_id)
        / f"trial_{int(trial_seed):03d}.npz"
    )


def _validate_trial(
    path: Path,
    *,
    manifest: dict[str, Any],
    subject: dict[str, Any],
    occupancy: float,
    trial_seed: int,
) -> None:
    pilot._validate_existing_trial(
        path,
        protocol_fingerprint=str(manifest["protocol_fingerprint"]),
        trial_seed=int(trial_seed),
        occupancy=float(occupancy),
        stim_region_labels=[
            str(value) for value in manifest["stim_region_labels"]
        ],
        receptor_map_sha256=str(manifest["receptor_map_sha256"]),
        cohort=str(subject["cohort"]),
        condition=str(subject["condition"]),
        subject_id=str(subject["subject_id"]),
        scenario=str(manifest["scenario"]),
        expected_stim_onset_ms=float(
            manifest["stim_onsets_ms_by_trial_seed"][str(int(trial_seed))]
        ),
        atlas_labels_sha256=str(manifest["atlas_labels_sha256"]),
        receptor_tracer=str(manifest["receptor_tracer"]),
        receptor_csv_sha256=str(manifest["receptor_csv_sha256"]),
        expected_t_analysis_ms=float(manifest["t_analysis_ms"]),
        expected_protocol_version=str(manifest["protocol_version"]),
    )


def compute_pci_metrics(
    trials_time_region: list[np.ndarray],
    *,
    onset: int,
    dt_ms: float,
    t_analysis_ms: float,
    metric_config: dict[str, Any],
) -> dict[str, Any]:
    """Compute PCI-LZ and PCI-ST once from the same aligned trial subset."""

    lz_result = pci_casali_like_multi_trial(
        trials_time_region,
        stimulation_index=int(onset),
        t_analysis_ms=float(t_analysis_ms),
        dt_ms=float(dt_ms),
        binarise_method="casali",
        binarise_kwargs={
            "n_bootstrap": int(metric_config["pci_permutation_replicates"]),
            "alpha": float(metric_config["pci_alpha"]),
            "seed": int(metric_config["pci_seed"]),
            "significance_method": str(
                metric_config["pci_significance_method"]
            ),
        },
        response_start_ms=float(metric_config["pci_response_start_ms"]),
        min_source_entropy=float(metric_config["pci_min_source_entropy"]),
        return_debug=True,
    )
    if not isinstance(lz_result, dict):
        raise AssertionError("Detailed PCI-LZ call did not return diagnostics.")

    trial_stack = np.stack(
        [np.asarray(trial, dtype=float).T for trial in trials_time_region],
        axis=0,
    )
    times_ms = (
        np.arange(trial_stack.shape[2], dtype=float) - int(onset)
    ) * float(dt_ms)
    available_start_ms = float(times_ms[0])
    available_stop_ms = float(times_ms[-1] + float(dt_ms))
    target_baseline = tuple(metric_config["pci_st_baseline_window_ms"])
    target_response = tuple(metric_config["pci_st_response_window_ms"])
    effective_baseline = (
        max(float(target_baseline[0]), available_start_ms),
        min(float(target_baseline[1]), available_stop_ms),
    )
    effective_response = (
        max(float(target_response[0]), available_start_ms),
        min(float(target_response[1]), available_stop_ms),
    )
    if (
        effective_baseline[0] >= effective_baseline[1]
        or effective_response[0] >= effective_response[1]
    ):
        raise ValueError(
            "Cached epoch does not cover the requested PCI-ST baseline/response "
            "windows."
        )
    st_result = pci_st_from_trials(
        trial_stack,
        times_ms,
        baseline_center_trials=True,
        baseline_window_ms=effective_baseline,
        response_window_ms=effective_response,
        k=float(metric_config["pci_st_k"]),
        min_snr=float(metric_config["pci_st_min_snr"]),
        max_var_percent=float(metric_config["pci_st_max_var_percent"]),
        n_steps=int(metric_config["pci_st_n_steps"]),
        return_details=True,
    )
    if not isinstance(st_result, PCIStResult):
        raise AssertionError("Detailed PCI-ST call did not return diagnostics.")

    binary = np.asarray(lz_result["binary_response"], dtype=np.uint8)
    return {
        "pci_lz": float(lz_result["pci"]),
        "pci_lz_threshold": float(lz_result["threshold"]),
        "pci_lz_source_entropy": float(lz_result["entropy"]),
        "pci_lz_min_source_entropy": float(
            lz_result["min_source_entropy"]
        ),
        "pci_lz_low_activation_forced_zero": bool(
            lz_result["low_activation_forced_zero"]
        ),
        "pci_lz_response_start_ms_requested": float(
            lz_result["response_start_ms_requested"]
        ),
        "pci_lz_response_start_ms_effective": float(
            lz_result["response_start_ms_effective"]
        ),
        "pci_lz_active_fraction": float(lz_result["active_fraction"]),
        "pci_lz_active_entries": int(binary.sum()),
        "pci_lz_lz_complexity": float(lz_result["lz"]),
        "pci_lz_normalization": float(lz_result["norm"]),
        "pci_st": float(st_result.pci_st),
        "pci_st_n_components": int(st_result.n_components),
        "pci_st_n_variance_components": int(
            st_result.n_variance_components
        ),
        "pci_st_available_start_ms": available_start_ms,
        "pci_st_available_stop_ms": available_stop_ms,
        "pci_st_target_baseline_start_ms": float(target_baseline[0]),
        "pci_st_target_baseline_stop_ms": float(target_baseline[1]),
        "pci_st_effective_baseline_start_ms": float(effective_baseline[0]),
        "pci_st_effective_baseline_stop_ms": float(effective_baseline[1]),
        "pci_st_target_response_start_ms": float(target_response[0]),
        "pci_st_target_response_stop_ms": float(target_response[1]),
        "pci_st_effective_response_start_ms": float(effective_response[0]),
        "pci_st_effective_response_stop_ms": float(effective_response[1]),
        "pci_st_n_baseline_samples": int(st_result.n_baseline_samples),
        "pci_st_n_response_samples": int(st_result.n_response_samples),
        "pci_st_first_baseline_sample_ms": float(
            st_result.baseline_sample_times_ms[0]
        ),
        "pci_st_last_baseline_sample_ms": float(
            st_result.baseline_sample_times_ms[-1]
        ),
        "pci_st_first_response_sample_ms": float(
            st_result.response_sample_times_ms[0]
        ),
        "pci_st_last_response_sample_ms": float(
            st_result.response_sample_times_ms[-1]
        ),
    }


def estimate_nested_subsets(
    aligned_trials: list[np.ndarray],
    *,
    trial_seeds: list[int],
    subsets: list[dict[str, Any]],
    onset: int,
    dt_ms: float,
    t_analysis_ms: float,
    metric_config: dict[str, Any],
    metric_function: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every requested subset; the full set occurs exactly once."""

    if len(aligned_trials) != len(trial_seeds):
        raise ValueError("aligned_trials and trial_seeds must have equal length.")
    index_by_seed = {
        int(seed): index for index, seed in enumerate(trial_seeds)
    }
    if len(index_by_seed) != len(trial_seeds):
        raise ValueError("trial_seeds must be unique.")
    compute = compute_pci_metrics if metric_function is None else metric_function

    rows: list[dict[str, Any]] = []
    for subset in subsets:
        seeds = [int(value) for value in subset["trial_seeds"]]
        selected_trials = [aligned_trials[index_by_seed[seed]] for seed in seeds]
        started = perf_counter()
        metrics = compute(
            selected_trials,
            onset=int(onset),
            dt_ms=float(dt_ms),
            t_analysis_ms=float(t_analysis_ms),
            metric_config=metric_config,
        )
        rows.append(
            {
                "repeat": int(subset["repeat"]),
                "n_trials": int(subset["n_trials"]),
                "subset_kind": str(subset["subset_kind"]),
                "trial_seeds_json": json.dumps(seeds),
                "runtime_s": float(perf_counter() - started),
                **metrics,
            }
        )
    full_rows = [row for row in rows if row["subset_kind"] == "full_once"]
    if len(full_rows) != 1:
        raise AssertionError("The 100-trial subset must be evaluated exactly once.")
    return rows


def _checkpoint_path(
    output_root: Path,
    *,
    cohort: str,
    subject_id: str,
    occupancy: float,
) -> Path:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in f"{cohort}__{subject_id}"
    )
    return (
        output_root
        / "checkpoints"
        / f"{safe}__{pilot._occ_tag(float(occupancy))}.json"
    )


def _analyse_subject_occupancy(job: dict[str, Any]) -> dict[str, Any]:
    cache_root = Path(job["cache_root"])
    output_root = Path(job["output_root"])
    manifest = job["source_manifest"]
    subject = job["subject"]
    occupancy = float(job["occupancy"])
    trial_seeds = [int(value) for value in manifest["trial_seeds"]]
    paths = [
        _trial_path(
            cache_root,
            occupancy=occupancy,
            scenario=str(manifest["scenario"]),
            cohort=str(subject["cohort"]),
            subject_id=str(subject["subject_id"]),
            trial_seed=seed,
        )
        for seed in trial_seeds
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} cached trials for "
            f"{subject['cohort']}:{subject['subject_id']} at occupancy "
            f"{occupancy:g}; first missing: {missing[0]}"
        )

    inventory: list[dict[str, Any]] = []
    signature_rows: list[tuple[str, int, int]] = []
    for path, seed in zip(paths, trial_seeds, strict=True):
        _validate_trial(
            path,
            manifest=manifest,
            subject=subject,
            occupancy=occupancy,
            trial_seed=seed,
        )
        stat = path.stat()
        signature_rows.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
        inventory.append(
            {
                "cohort": str(subject["cohort"]),
                "condition": str(subject["condition"]),
                "subject_id": str(subject["subject_id"]),
                "occupancy": occupancy,
                "trial_seed": seed,
                "stim_onset_ms": float(
                    manifest["stim_onsets_ms_by_trial_seed"][str(seed)]
                ),
                "file_size_bytes": int(stat.st_size),
                "file_mtime_ns": int(stat.st_mtime_ns),
                "trial_path": str(path),
                "validation": "passed_corrected_protocol",
            }
        )
    input_signature = _fingerprint(signature_rows)

    checkpoint = _checkpoint_path(
        output_root,
        cohort=str(subject["cohort"]),
        subject_id=str(subject["subject_id"]),
        occupancy=occupancy,
    )
    if checkpoint.is_file() and not bool(job["overwrite"]):
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            previous.get("analysis_fingerprint") == job["analysis_fingerprint"]
            and previous.get("input_signature") == input_signature
        ):
            return {
                "estimates": previous["estimates"],
                "inventory": inventory,
                "resumed": True,
            }

    aligned_trials, onset, dt_ms, t_analysis_ms = pilot._load_trials(paths)
    rows = estimate_nested_subsets(
        aligned_trials,
        trial_seeds=trial_seeds,
        subsets=job["subsets"],
        onset=onset,
        dt_ms=dt_ms,
        t_analysis_ms=t_analysis_ms,
        metric_config=job["metric_config"],
    )
    for row in rows:
        row.update(
            {
                "cohort": str(subject["cohort"]),
                "condition": str(subject["condition"]),
                "subject_id": str(subject["subject_id"]),
                "scenario": str(manifest["scenario"]),
                "occupancy": occupancy,
                "dt_ms": float(dt_ms),
                "t_analysis_ms": float(t_analysis_ms),
            }
        )
    payload = {
        "analysis_fingerprint": job["analysis_fingerprint"],
        "input_signature": input_signature,
        "estimates": rows,
    }
    _atomic_write_json(checkpoint, payload)
    return {"estimates": rows, "inventory": inventory, "resumed": False}


def _subset_membership_frame(
    selected: list[SelectedSubject],
    subsets_by_subject: dict[tuple[str, str], list[dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject in selected:
        subsets = subsets_by_subject[(subject.cohort, subject.subject_id)]
        for subset in subsets:
            draw_rank = {
                int(seed): rank
                for rank, seed in enumerate(subset["draw_order"], start=1)
            }
            for seed in subset["trial_seeds"]:
                rows.append(
                    {
                        "cohort": subject.cohort,
                        "condition": subject.condition,
                        "subject_id": subject.subject_id,
                        "repeat": int(subset["repeat"]),
                        "n_trials": int(subset["n_trials"]),
                        "subset_kind": str(subset["subset_kind"]),
                        "trial_seed": int(seed),
                        "draw_rank": int(draw_rank[int(seed)]),
                        "permutation_seed_uint64": subset[
                            "permutation_seed_uint64"
                        ],
                        "matched_across_occupancies": True,
                    }
                )
    return pd.DataFrame(rows)


def _attach_full_reference(estimates: pd.DataFrame) -> pd.DataFrame:
    key = ["cohort", "condition", "subject_id", "occupancy"]
    full = estimates.loc[
        estimates["subset_kind"].eq("full_once"),
        key + ["pci_lz", "pci_st"],
    ].rename(
        columns={
            "pci_lz": "pci_lz_full_100",
            "pci_st": "pci_st_full_100",
        }
    )
    if full.duplicated(key).any():
        raise RuntimeError("More than one full estimate exists for an input cell.")
    output = estimates.merge(full, on=key, how="left", validate="many_to_one")
    for metric in ("pci_lz", "pci_st"):
        reference = f"{metric}_full_100"
        output[f"{metric}_error_vs_100"] = output[metric] - output[reference]
        output[f"{metric}_absolute_error_vs_100"] = np.abs(
            output[f"{metric}_error_vs_100"]
        )
    return output


def _summarize(estimates: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "cohort",
        "condition",
        "occupancy",
        "n_trials",
        "subset_kind",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in estimates.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["n_estimates"] = int(len(group))
        row["n_subjects"] = int(group["subject_id"].nunique())
        for metric in ("pci_lz", "pci_st"):
            values = group[metric].to_numpy(dtype=float)
            errors = group[f"{metric}_absolute_error_vs_100"].to_numpy(
                dtype=float
            )
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
            )
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
            row[f"{metric}_mean_absolute_error_vs_100"] = float(
                np.mean(errors)
            )
            row[f"{metric}_q95_absolute_error_vs_100"] = float(
                np.quantile(errors, 0.95)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["occupancy", "condition", "n_trials"]
    )


def _dose_effects(estimates: pd.DataFrame) -> pd.DataFrame:
    """Compute matched max-dose-minus-baseline effects for every subset."""

    group_columns = [
        "cohort",
        "condition",
        "subject_id",
        "repeat",
        "n_trials",
        "subset_kind",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in estimates.groupby(group_columns, sort=True, dropna=False):
        occupancies = group["occupancy"].to_numpy(dtype=float)
        baseline_mask = np.isclose(occupancies, CANONICAL_OCCUPANCIES[0])
        maximum_mask = np.isclose(occupancies, CANONICAL_OCCUPANCIES[-1])
        if int(baseline_mask.sum()) != 1 or int(maximum_mask.sum()) != 1:
            raise RuntimeError(
                "Each matched subset must contain exactly one baseline and "
                "one maximum-dose estimate."
            )
        row = dict(zip(group_columns, keys, strict=True))
        row["baseline_occupancy"] = float(CANONICAL_OCCUPANCIES[0])
        row["maximum_occupancy"] = float(CANONICAL_OCCUPANCIES[-1])
        for metric in ("pci_lz", "pci_st"):
            baseline = float(group.loc[baseline_mask, metric].iloc[0])
            maximum = float(group.loc[maximum_mask, metric].iloc[0])
            row[f"{metric}_baseline"] = baseline
            row[f"{metric}_maximum_dose"] = maximum
            row[f"delta_{metric}_maximum_minus_baseline"] = maximum - baseline
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cohort", "subject_id", "n_trials", "repeat"]
    )


def _summarize_dose_effects(dose_effects: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["cohort", "condition", "n_trials", "subset_kind"]
    rows: list[dict[str, Any]] = []
    for keys, group in dose_effects.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        row = dict(zip(group_columns, keys, strict=True))
        row["n_estimates"] = int(len(group))
        row["n_subjects"] = int(group["subject_id"].nunique())
        for metric in ("pci_lz", "pci_st"):
            column = f"delta_{metric}_maximum_minus_baseline"
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sd"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
            )
            row[f"{column}_q05"] = float(np.quantile(values, 0.05))
            row[f"{column}_q95"] = float(np.quantile(values, 0.95))
            row[f"{column}_positive_fraction"] = float(np.mean(values > 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "n_trials"])


def _plot_convergence(estimates: pd.DataFrame, output_root: Path) -> None:
    occupancies = sorted(estimates["occupancy"].unique())
    metrics = (
        ("pci_lz", "PCI-LZ"),
        ("pci_st", "PCI-ST (unbounded)"),
    )
    fig, axes = plt.subplots(
        len(metrics),
        len(occupancies),
        figsize=(4.0 * len(occupancies), 7.2),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    condition_order = [
        condition
        for condition in pilot.CONDITION_ORDER
        if condition in set(estimates["condition"])
    ]
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column_index, occupancy in enumerate(occupancies):
            axis = axes[row_index, column_index]
            occupancy_data = estimates[
                np.isclose(estimates["occupancy"], occupancy)
            ]
            for condition in condition_order:
                condition_data = occupancy_data[
                    occupancy_data["condition"].eq(condition)
                ]
                summary_rows = []
                for n_trials, group in condition_data.groupby("n_trials"):
                    values = group[metric].to_numpy(dtype=float)
                    summary_rows.append(
                        (
                            int(n_trials),
                            float(np.mean(values)),
                            float(np.quantile(values, 0.05)),
                            float(np.quantile(values, 0.95)),
                        )
                    )
                if not summary_rows:
                    continue
                summary_rows.sort()
                array = np.asarray(summary_rows, dtype=float)
                color = pilot.COND_COLORS.get(condition, "#555555")
                axis.plot(
                    array[:, 0],
                    array[:, 1],
                    marker="o",
                    linewidth=1.8,
                    color=color,
                    label=condition,
                )
                axis.fill_between(
                    array[:, 0],
                    array[:, 2],
                    array[:, 3],
                    color=color,
                    alpha=0.13,
                    linewidth=0.0,
                )
            axis.set_title(f"5-HT$_{{2A}}$ occupancy = {occupancy:g}")
            axis.grid(alpha=0.25)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Trials in nested subset")
            if column_index == 0:
                axis.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Nested trial-count convergence (bands: descriptive 5th–95th percentile)",
        fontsize=14,
    )
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_root / f"serotonergic_pci_trial_convergence.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def _analysis_manifest(
    args: argparse.Namespace,
    *,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    source_manifest_sha256: str,
    selected: list[SelectedSubject],
) -> dict[str, Any]:
    scientific = {
        "protocol_version": PROTOCOL_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "source_protocol_fingerprint": str(
            source_manifest["protocol_fingerprint"]
        ),
        "source_protocol_version": str(source_manifest["protocol_version"]),
        "selected_subjects": [
            {
                "cohort": subject.cohort,
                "condition": subject.condition,
                "subject_id": subject.subject_id,
            }
            for subject in selected
        ],
        "selection_seed": int(args.selection_seed),
        "subset_seed": int(args.subset_seed),
        "subset_sizes": [int(value) for value in args.subset_sizes],
        "full_trial_count": 100,
        "full_estimate_repeats": 1,
        "nested_subset_repeats": int(args.repeats),
        "occupancies": list(CANONICAL_OCCUPANCIES),
        "pci_lz": {
            "binarise_method": "casali",
            "significance_method": str(args.pci_significance_method),
            "permutation_replicates": int(
                args.pci_permutation_replicates
            ),
            "alpha": float(args.pci_alpha),
            "seed": int(args.pci_seed),
            "response_start_ms": float(args.pci_response_start_ms),
            "min_source_entropy": float(args.pci_min_source_entropy),
        },
        "pci_st": {
            "input": "continuous_trial_average",
            "baseline_center_trials": True,
            "baseline_window_ms": [
                float(value) for value in args.pci_st_baseline_window_ms
            ],
            "response_window_ms": [
                float(value) for value in args.pci_st_response_window_ms
            ],
            "effective_window_policy": (
                "half-open target clipped to cached [first_sample,last_sample+dt)"
            ),
            "k": float(args.pci_st_k),
            "min_snr": float(args.pci_st_min_snr),
            "max_var_percent": float(args.pci_st_max_var_percent),
            "n_steps": int(args.pci_st_n_steps),
            "bounded_zero_one": False,
        },
    }
    return {
        "script": "scripts/analyze_serotonergic_pci_trial_convergence.py",
        "protocol_version": PROTOCOL_VERSION,
        "source_cache_access": "read_only",
        "source_cache_root": str(args.cache_root.resolve()),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "output_root": str(args.output_root.resolve()),
        "workers": int(args.workers),
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "scientific_protocol": scientific,
        "analysis_fingerprint": _fingerprint(scientific),
        "interpretation_note": (
            "Nested-subset repetitions overlap and quantify numerical trial-count "
            "stability; they are not independent biological replicates."
        ),
    }


def _write_or_validate_analysis_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.is_file() and not overwrite:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("analysis_fingerprint") != manifest["analysis_fingerprint"]:
            raise RuntimeError(
                "Existing convergence output uses a different protocol. "
                "Choose another --output-root or pass --overwrite."
            )
        _atomic_write_json(path, manifest)
        return
    _atomic_write_json(path, manifest)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_cli(args)
    args.cache_root = args.cache_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    _assert_separate_output(args.cache_root, args.output_root)
    source_manifest_path = (
        args.source_manifest.expanduser().resolve()
        if args.source_manifest is not None
        else args.cache_root / "logs" / "run_manifest.json"
    )
    source_manifest, source_manifest_sha256 = _load_source_manifest(
        source_manifest_path
    )
    _validate_source_manifest(source_manifest)
    selected = select_subjects(
        source_manifest,
        cohorts=[str(value) for value in args.cohorts],
        subjects_per_cohort=int(args.subjects_per_cohort),
        selection_seed=int(args.selection_seed),
        explicit_subjects=args.subject,
    )
    subsets_by_subject = {
        (subject.cohort, subject.subject_id): build_nested_subsets(
            [int(value) for value in source_manifest["trial_seeds"]],
            subset_sizes=[int(value) for value in args.subset_sizes],
            repeats=int(args.repeats),
            subset_seed=int(args.subset_seed),
            cohort=subject.cohort,
            subject_id=subject.subject_id,
        )
        for subject in selected
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables_root = args.output_root / "tables"
    logs_root = args.output_root / "logs"
    tables_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    manifest = _analysis_manifest(
        args,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        selected=selected,
    )
    _write_or_validate_analysis_manifest(
        logs_root / "run_manifest.json",
        manifest,
        overwrite=bool(args.overwrite),
    )
    selected_frame = pd.DataFrame(
        [
            {
                **subject.__dict__,
                "selection_seed": int(args.selection_seed),
            }
            for subject in selected
        ]
    )
    membership = _subset_membership_frame(selected, subsets_by_subject)
    _atomic_write_csv(tables_root / "selected_subjects.csv", selected_frame)
    _atomic_write_csv(tables_root / "trial_subsets.csv", membership)

    print(
        "[pci-convergence] "
        f"selected={len(selected)} subjects, repeats={args.repeats}, "
        f"subsets={args.subset_sizes} + [100 once]"
    )
    if args.dry_run:
        print(
            "[pci-convergence] dry run complete; no cached trial file was opened."
        )
        return

    metric_config = {
        "pci_significance_method": str(args.pci_significance_method),
        "pci_permutation_replicates": int(args.pci_permutation_replicates),
        "pci_alpha": float(args.pci_alpha),
        "pci_seed": int(args.pci_seed),
        "pci_response_start_ms": float(args.pci_response_start_ms),
        "pci_min_source_entropy": float(args.pci_min_source_entropy),
        "pci_st_baseline_window_ms": [
            float(value) for value in args.pci_st_baseline_window_ms
        ],
        "pci_st_response_window_ms": [
            float(value) for value in args.pci_st_response_window_ms
        ],
        "pci_st_k": float(args.pci_st_k),
        "pci_st_min_snr": float(args.pci_st_min_snr),
        "pci_st_max_var_percent": float(args.pci_st_max_var_percent),
        "pci_st_n_steps": int(args.pci_st_n_steps),
    }
    jobs = []
    for subject in selected:
        subject_dict = {
            "cohort": subject.cohort,
            "condition": subject.condition,
            "subject_id": subject.subject_id,
        }
        for occupancy in CANONICAL_OCCUPANCIES:
            jobs.append(
                {
                    "cache_root": str(args.cache_root),
                    "output_root": str(args.output_root),
                    "source_manifest": source_manifest,
                    "subject": subject_dict,
                    "occupancy": float(occupancy),
                    "subsets": subsets_by_subject[
                        (subject.cohort, subject.subject_id)
                    ],
                    "metric_config": metric_config,
                    "analysis_fingerprint": manifest[
                        "analysis_fingerprint"
                    ],
                    "overwrite": bool(args.overwrite),
                }
            )

    results: list[dict[str, Any]] = []
    if int(args.workers) == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_analyse_subject_occupancy(job))
            print(f"[pci-convergence] completed {index}/{len(jobs)} cells")
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(_analyse_subject_occupancy, job): job
                for job in jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(f"[pci-convergence] completed {index}/{len(jobs)} cells")

    estimates = pd.DataFrame(
        [row for result in results for row in result["estimates"]]
    ).sort_values(
        ["cohort", "subject_id", "occupancy", "n_trials", "repeat"]
    )
    estimates = _attach_full_reference(estimates)
    inventory = pd.DataFrame(
        [row for result in results for row in result["inventory"]]
    ).sort_values(["cohort", "subject_id", "occupancy", "trial_seed"])
    summary = _summarize(estimates)
    dose_effects = _dose_effects(estimates)
    dose_effect_summary = _summarize_dose_effects(dose_effects)

    _atomic_write_csv(
        tables_root / "trial_convergence_estimates.csv",
        estimates,
    )
    _atomic_write_csv(
        tables_root / "trial_convergence_summary.csv",
        summary,
    )
    _atomic_write_csv(
        tables_root / "validated_trial_inputs.csv",
        inventory,
    )
    _atomic_write_csv(
        tables_root / "max_dose_minus_baseline_by_subset.csv",
        dose_effects,
    )
    _atomic_write_csv(
        tables_root / "max_dose_minus_baseline_summary.csv",
        dose_effect_summary,
    )
    _plot_convergence(estimates, args.output_root)

    resumed = sum(bool(result["resumed"]) for result in results)
    completion = {
        "analysis_fingerprint": manifest["analysis_fingerprint"],
        "n_selected_subjects": int(len(selected)),
        "n_subject_occupancy_cells": int(len(jobs)),
        "n_estimates": int(len(estimates)),
        "n_resumed_cells": int(resumed),
        "outputs": {
            "estimates_csv": str(
                tables_root / "trial_convergence_estimates.csv"
            ),
            "summary_csv": str(
                tables_root / "trial_convergence_summary.csv"
            ),
            "dose_effect_csv": str(
                tables_root / "max_dose_minus_baseline_by_subset.csv"
            ),
            "figure_png": str(
                args.output_root
                / "figures"
                / "serotonergic_pci_trial_convergence.png"
            ),
        },
    }
    _atomic_write_json(logs_root / "completion.json", completion)
    print(
        "[pci-convergence] complete: "
        f"{len(estimates)} estimates; {resumed}/{len(jobs)} cells resumed"
    )


if __name__ == "__main__":
    main()
