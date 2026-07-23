#!/usr/bin/env python3
"""Small subject-level PCI pilot with 5-HT2A-weighted split-leak modulation.

This script runs a small subject/trial subset of the corrected serotonergic PCI
protocol. By default, occupancy zero and all positive occupancies use the same
split gK/gNa Zerlaut implementation. ``--legacy-model-switch`` is retained only
to reproduce the earlier condition-b baseline-cache comparison.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(_REPO_ROOT / ".tvb-temp"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = _REPO_ROOT / "src"
_NOTEBOOKS = _REPO_ROOT / "notebooks"
for _p in (_SRC, _NOTEBOOKS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from brain_act_hybrid_common import (  # noqa: E402
    BASE_PARAMETER_MODEL_NEW,
    CONDITION_ORDER,
    COND_COLORS,
    DATASET_ROOT,
    RATE_MONITOR_PERIOD_MS_OLD,
    SCENARIOS,
    get_subject_jobs,
)
from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial  # noqa: E402
from tvbtoolkit.core.config import WholeBrainConfig  # noqa: E402
from tvbtoolkit.core.paths import doc_liege_results  # noqa: E402
from tvbtoolkit.datasets.brain_act import load_aal90_atlas, load_subject_structural  # noqa: E402
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: E402
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import _apply_damage_parity, worker_initializer  # noqa: E402
from tvbtoolkit.workflows.pharmacology import (  # noqa: E402
    el_eff_from_gK_gNa,
    get_5ht2a_aal90,
    leak_to_conductances,
)


CONDITION_B_GRADIENT = {
    "CNT": 10.0,
    "EMCS": 30.0,
    "MCS": 55.0,
    "UWS": 75.0,
    "COMA": 75.0,
}

PROTOCOL_VERSION = "3.1-time-locked-trial-average-atlas-aligned-provenance"
DEFAULT_STIM_REGION_LABEL = "Supp_Motor_Area_L"
DEFAULT_RECEPTOR_CSV = (
    _REPO_ROOT / "data" / "receptors" / "hansen_receptors_aal90.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=doc_liege_results("doc_simulation_run", "ba_sim_hybrid", "condition_b", "sims_pci"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "results" / "serotonergic_pci_pilot_corrected",
    )
    p.add_argument("--subjects-per-cohort", type=int, default=3)
    p.add_argument("--cohorts", nargs="+", default=["coma", "uws", "mcs", "emcs", "control"])
    p.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Explicit subject as cohort:subject_id. Can be passed multiple times.",
    )
    p.add_argument("--scenario", default="private_alpha0")
    p.add_argument("--trial-seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--occupancies", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.766])
    p.add_argument("--transient-ms", type=float, default=4000.0)
    p.add_argument("--t-analysis-ms", type=float, default=300.0)
    p.add_argument("--trial-sim-ms", type=float, default=8000.0)
    p.add_argument("--stim-amplitude", type=float, default=0.00030)
    p.add_argument("--stim-duration-ms", type=float, default=10.0)
    target_group = p.add_mutually_exclusive_group()
    target_group.add_argument(
        "--stim-region",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit zero-based indices in the converted dataset ordering. "
            "This legacy/exploratory option cannot be combined with "
            "--stim-region-label."
        ),
    )
    target_group.add_argument(
        "--stim-region-label",
        nargs="+",
        default=None,
        help=(
            "AAL90 target labels resolved against the converted dataset. "
            f"Default: {DEFAULT_STIM_REGION_LABEL}."
        ),
    )
    p.add_argument("--stim-onset-seed", type=int, default=0)
    p.add_argument(
        "--receptor-tracer",
        choices=["cimbi", "savli", "talbot"],
        default="cimbi",
    )
    p.add_argument(
        "--receptor-csv",
        type=Path,
        default=DEFAULT_RECEPTOR_CSV,
    )
    p.add_argument(
        "--pci-binarise-method",
        choices=["casali", "tvbsim"],
        default="casali",
        help=(
            "PCI significance route. 'casali' computes one PCI from the "
            "trial-averaged, bootstrap-thresholded response; 'tvbsim' retains "
            "the legacy mean of per-trial PCI values."
        ),
    )
    p.add_argument("--pci-bootstrap-replicates", type=int, default=500)
    p.add_argument("--pci-alpha", type=float, default=0.01)
    p.add_argument("--pci-bootstrap-seed", type=int, default=0)
    p.add_argument("--e-l-e-drug", type=float, default=-61.2)
    p.add_argument("--e-l-i-drug", type=float, default=-64.4)
    p.add_argument(
        "--b-e-override",
        type=float,
        default=None,
        help=(
            "Use one excitatory-adaptation value (pA) for every diagnosis. "
            "Omit for the diagnosis-configured gradient."
        ),
    )
    p.add_argument(
        "--legacy-model-switch",
        action="store_false",
        dest="split_model_all_occupancies",
        help=(
            "Reproduce the legacy model-form switch (standard Zerlaut at occupancy 0, "
            "split gK/gNa model at positive occupancy). By default the split model is "
            "used at every occupancy so dose is the only model change."
        ),
    )
    p.set_defaults(split_model_all_occupancies=True)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.stim_region is None and args.stim_region_label is None:
        args.stim_region_label = [DEFAULT_STIM_REGION_LABEL]
    return args


def _occ_tag(occupancy: float) -> str:
    return f"occ_{int(round(float(occupancy) * 1000)):03d}"


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash scientific protocol fields while excluding execution-only state."""
    excluded = {
        "output_root",
        "workers",
        "overwrite",
        "protocol_fingerprint",
    }
    protocol = {
        key: value
        for key, value in manifest.items()
        if key not in excluded
    }
    return hashlib.sha256(_canonical_json(protocol).encode("utf-8")).hexdigest()


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("protocol_fingerprint") != manifest["protocol_fingerprint"]:
            raise RuntimeError(
                "Existing output uses a different serotonergic PCI protocol. "
                f"Choose another --output-root or pass --overwrite: {path}"
            )
        return
    _write_json(path, manifest)


def _validate_existing_trial(
    path: Path,
    *,
    protocol_fingerprint: str,
    trial_seed: int,
    occupancy: float,
    stim_region_labels: list[str],
    receptor_map_sha256: str,
    cohort: str,
    condition: str,
    subject_id: str,
    scenario: str,
    expected_stim_onset_ms: float,
    atlas_labels_sha256: str,
    receptor_tracer: str,
    receptor_csv_sha256: str,
    expected_t_analysis_ms: float,
    expected_protocol_version: str = PROTOCOL_VERSION,
) -> None:
    """Reject stale, corrupt, misplaced, or differently parameterized trials."""
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "protocol_version",
                "protocol_fingerprint",
                "cohort",
                "condition",
                "subject_id",
                "scenario",
                "trial_seed",
                "occupancy",
                "time_ms",
                "rate",
                "region_labels",
                "atlas_labels_sha256",
                "receptor_map",
                "receptor_map_alignment",
                "receptor_tracer",
                "receptor_csv_sha256",
                "stim_region_labels",
                "stim_region",
                "receptor_map_sha256",
                "stim_onset_ms",
                "stim_onset_sample_index",
                "stim_onset_sample_ms",
                "stim_onset_alignment_residual_ms",
                "t_analysis_ms",
                "rate_monitor_period_ms",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise KeyError(f"missing metadata keys {missing}")
            actual_protocol_version = str(
                np.asarray(data["protocol_version"]).reshape(-1)[0]
            )
            actual_fingerprint = str(
                np.asarray(data["protocol_fingerprint"]).reshape(-1)[0]
            )
            actual_cohort = str(np.asarray(data["cohort"]).reshape(-1)[0])
            actual_condition = str(np.asarray(data["condition"]).reshape(-1)[0])
            actual_subject_id = str(
                np.asarray(data["subject_id"]).reshape(-1)[0]
            )
            actual_scenario = str(np.asarray(data["scenario"]).reshape(-1)[0])
            actual_seed = int(np.asarray(data["trial_seed"]).reshape(-1)[0])
            actual_occupancy = float(
                np.asarray(data["occupancy"]).reshape(-1)[0]
            )
            time_ms = np.asarray(data["time_ms"], dtype=float)
            rate = np.asarray(data["rate"], dtype=float)
            region_labels = np.asarray(
                data["region_labels"],
                dtype="U128",
            ).reshape(-1)
            actual_atlas_hash = str(
                np.asarray(data["atlas_labels_sha256"]).reshape(-1)[0]
            )
            receptor_map = np.asarray(
                data["receptor_map"],
                dtype=np.float64,
            ).reshape(-1)
            actual_receptor_alignment = str(
                np.asarray(data["receptor_map_alignment"]).reshape(-1)[0]
            )
            actual_receptor_tracer = str(
                np.asarray(data["receptor_tracer"]).reshape(-1)[0]
            )
            actual_receptor_csv_hash = str(
                np.asarray(data["receptor_csv_sha256"]).reshape(-1)[0]
            )
            actual_labels = [
                str(value)
                for value in np.asarray(data["stim_region_labels"]).reshape(-1)
            ]
            stim_region = np.asarray(
                data["stim_region"],
                dtype=int,
            ).reshape(-1)
            actual_receptor_hash = str(
                np.asarray(data["receptor_map_sha256"]).reshape(-1)[0]
            )
            actual_stim_onset_ms = float(
                np.asarray(data["stim_onset_ms"]).reshape(-1)[0]
            )
            actual_onset_index = int(
                np.asarray(data["stim_onset_sample_index"]).reshape(-1)[0]
            )
            actual_onset_sample_ms = float(
                np.asarray(data["stim_onset_sample_ms"]).reshape(-1)[0]
            )
            actual_onset_residual_ms = float(
                np.asarray(
                    data["stim_onset_alignment_residual_ms"]
                ).reshape(-1)[0]
            )
            actual_t_analysis_ms = float(
                np.asarray(data["t_analysis_ms"]).reshape(-1)[0]
            )
            actual_monitor_period_ms = float(
                np.asarray(data["rate_monitor_period_ms"]).reshape(-1)[0]
            )
    except Exception as exc:
        raise RuntimeError(
            f"Existing trial is unreadable or lacks corrected-protocol provenance: {path}"
        ) from exc

    mismatches: list[str] = []
    expected_filename = f"trial_{int(trial_seed):03d}.npz"
    if path.name != expected_filename:
        mismatches.append("filename")
    if actual_protocol_version != str(expected_protocol_version):
        mismatches.append("protocol_version")
    if actual_fingerprint != str(protocol_fingerprint):
        mismatches.append("protocol_fingerprint")
    if actual_cohort != str(cohort):
        mismatches.append("cohort")
    if actual_condition != str(condition):
        mismatches.append("condition")
    if actual_subject_id != str(subject_id):
        mismatches.append("subject_id")
    if actual_scenario != str(scenario):
        mismatches.append("scenario")
    if actual_seed != int(trial_seed):
        mismatches.append("trial_seed")
    if not np.isclose(actual_occupancy, float(occupancy), rtol=0.0, atol=1e-12):
        mismatches.append("occupancy")
    if time_ms.ndim != 1 or time_ms.size < 2 or not np.isfinite(time_ms).all():
        mismatches.append("time_ms")
        sampling_interval_ms = float("nan")
    else:
        time_differences = np.diff(time_ms)
        sampling_interval_ms = float(np.median(time_differences))
        if (
            np.any(time_differences <= 0.0)
            or not np.allclose(
                time_differences,
                sampling_interval_ms,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            mismatches.append("time_ms")
    if (
        rate.ndim != 2
        or rate.shape != (time_ms.size, 90)
        or not np.isfinite(rate).all()
    ):
        mismatches.append("rate")
    if (
        region_labels.shape != (90,)
        or len(set(region_labels.tolist())) != 90
        or _sha256_array(region_labels) != actual_atlas_hash
    ):
        mismatches.append("region_labels")
    if actual_atlas_hash != str(atlas_labels_sha256):
        mismatches.append("atlas_labels_sha256")
    if (
        receptor_map.shape != (90,)
        or not np.isfinite(receptor_map).all()
        or _sha256_array(receptor_map) != actual_receptor_hash
    ):
        mismatches.append("receptor_map")
    if actual_receptor_alignment != "AAL region-label join":
        mismatches.append("receptor_map_alignment")
    if actual_receptor_tracer != str(receptor_tracer):
        mismatches.append("receptor_tracer")
    if actual_receptor_csv_hash != str(receptor_csv_sha256):
        mismatches.append("receptor_csv_sha256")
    if actual_labels != [str(label) for label in stim_region_labels]:
        mismatches.append("stim_region_labels")
    if (
        stim_region.size != len(actual_labels)
        or np.any(stim_region < 0)
        or np.any(stim_region >= region_labels.size)
        or (
            stim_region.size == len(actual_labels)
            and [str(region_labels[index]) for index in stim_region]
            != actual_labels
        )
    ):
        mismatches.append("stim_region")
    if actual_receptor_hash != str(receptor_map_sha256):
        mismatches.append("receptor_map_sha256")
    if not np.isclose(
        actual_stim_onset_ms,
        float(expected_stim_onset_ms),
        rtol=0.0,
        atol=1e-9,
    ):
        mismatches.append("stim_onset_ms")
    if not np.isclose(
        actual_t_analysis_ms,
        float(expected_t_analysis_ms),
        rtol=0.0,
        atol=1e-9,
    ):
        mismatches.append("t_analysis_ms")
    if np.isfinite(sampling_interval_ms):
        nearest_onset_index = int(
            np.argmin(np.abs(time_ms - actual_stim_onset_ms))
        )
        expected_onset_sample_ms = float(time_ms[nearest_onset_index])
        expected_residual_ms = (
            expected_onset_sample_ms - actual_stim_onset_ms
        )
        analysis_bins = int(
            round(actual_t_analysis_ms / sampling_interval_ms)
        )
        if actual_onset_index != nearest_onset_index:
            mismatches.append("stim_onset_sample_index")
        if not np.isclose(
            actual_onset_sample_ms,
            expected_onset_sample_ms,
            rtol=0.0,
            atol=1e-9,
        ):
            mismatches.append("stim_onset_sample_ms")
        if (
            not np.isclose(
                actual_onset_residual_ms,
                expected_residual_ms,
                rtol=0.0,
                atol=1e-9,
            )
            or abs(expected_residual_ms) > 0.5 * sampling_interval_ms + 1e-6
        ):
            mismatches.append("stim_onset_alignment_residual_ms")
        if (
            analysis_bins < 1
            or nearest_onset_index - analysis_bins < 0
            or nearest_onset_index + analysis_bins > time_ms.size
        ):
            mismatches.append("peri_stimulus_window")
    if not np.isfinite(actual_monitor_period_ms):
        mismatches.append("rate_monitor_period_ms")
    if mismatches:
        raise RuntimeError(
            "Existing trial provenance/content mismatch "
            f"({', '.join(dict.fromkeys(mismatches))}): {path}. "
            "Use another output root or pass --overwrite."
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _select_subjects(dataset_root: Path, cohorts: list[str], subjects_per_cohort: int, explicit_subjects: list[str] | None = None):
    jobs = get_subject_jobs(dataset_root)
    if explicit_subjects:
        by_key = {(j.cohort, j.subject_id): j for j in jobs}
        selected = []
        for spec in explicit_subjects:
            try:
                cohort, subject_id = spec.split(":", 1)
            except ValueError as exc:
                raise ValueError(f"Explicit subject must be cohort:subject_id, got {spec!r}") from exc
            key = (cohort.strip().lower(), subject_id.strip())
            if key not in by_key:
                raise KeyError(f"Unknown subject spec {spec!r}.")
            selected.append(by_key[key])
        return selected

    out = []
    for cohort in cohorts:
        cohort_jobs = [j for j in jobs if j.cohort == cohort]
        out.extend(cohort_jobs[: max(0, int(subjects_per_cohort))])
    return out


def _validate_protocol_args(args: argparse.Namespace) -> None:
    args.dataset_root = Path(args.dataset_root).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.receptor_csv = Path(args.receptor_csv).expanduser().resolve()
    args.trial_seeds = [int(seed) for seed in args.trial_seeds]
    args.occupancies = [float(occupancy) for occupancy in args.occupancies]

    if not (args.dataset_root / "index.json").exists():
        raise FileNotFoundError(
            f"Converted dataset index not found: {args.dataset_root / 'index.json'}"
        )
    if not args.receptor_csv.is_file():
        raise FileNotFoundError(f"Receptor table not found: {args.receptor_csv}")
    if not args.trial_seeds:
        raise ValueError("--trial-seeds must contain at least one seed.")
    if len(set(args.trial_seeds)) != len(args.trial_seeds):
        raise ValueError("--trial-seeds must be unique.")
    if any(seed < 0 for seed in args.trial_seeds):
        raise ValueError("--trial-seeds must be non-negative.")
    if not args.occupancies:
        raise ValueError("--occupancies must contain at least one dose.")
    if len(set(args.occupancies)) != len(args.occupancies):
        raise ValueError("--occupancies must be unique.")
    if any(occupancy < 0.0 or occupancy > 1.0 for occupancy in args.occupancies):
        raise ValueError("--occupancies must lie within [0, 1].")
    if not any(np.isclose(occupancy, 0.0) for occupancy in args.occupancies):
        raise ValueError("--occupancies must include baseline occupancy 0.0.")
    if not any(occupancy > 0.0 for occupancy in args.occupancies):
        raise ValueError("--occupancies must include at least one positive dose.")
    if float(args.transient_ms) < 0.0:
        raise ValueError("--transient-ms must be non-negative.")
    if float(args.t_analysis_ms) <= 0.0:
        raise ValueError("--t-analysis-ms must be positive.")
    if float(args.trial_sim_ms) <= (
        float(args.transient_ms) + 2.0 * float(args.t_analysis_ms)
    ):
        raise ValueError(
            "--trial-sim-ms must leave complete pre/post analysis windows "
            "after the transient."
        )
    if float(args.stim_duration_ms) <= 0.0:
        raise ValueError("--stim-duration-ms must be positive.")
    if not np.isfinite(float(args.stim_amplitude)):
        raise ValueError("--stim-amplitude must be finite.")
    if int(args.workers) < 1:
        raise ValueError("--workers must be at least 1.")
    if int(args.pci_bootstrap_replicates) < 1:
        raise ValueError("--pci-bootstrap-replicates must be at least 1.")
    if not 0.0 < float(args.pci_alpha) < 1.0:
        raise ValueError("--pci-alpha must lie strictly between 0 and 1.")


def _resolve_stim_regions(args: argparse.Namespace):
    """Resolve stimulation targets against the dataset's declared atlas order."""
    atlas = load_aal90_atlas(args.dataset_root)
    atlas_labels = [str(label) for label in np.asarray(atlas.labels).reshape(-1)]
    if len(atlas_labels) != 90:
        raise ValueError(
            f"Expected a 90-region AAL atlas, found {len(atlas_labels)} regions."
        )
    if len(set(atlas_labels)) != len(atlas_labels):
        raise ValueError("Dataset atlas labels must be unique.")

    if args.stim_region is not None:
        indices = [int(index) for index in args.stim_region]
        invalid = [index for index in indices if index < 0 or index >= len(atlas_labels)]
        if invalid:
            raise ValueError(f"Stimulation indices out of range for AAL90: {invalid}")
        if len(set(indices)) != len(indices):
            raise ValueError("Stimulation indices must be unique.")
    else:
        requested = [str(label) for label in args.stim_region_label]
        if len(set(requested)) != len(requested):
            raise ValueError("Stimulation labels must be unique.")
        missing = [label for label in requested if label not in atlas_labels]
        if missing:
            raise ValueError(
                f"Stimulation labels not present in dataset atlas ({atlas.ordering}): {missing}"
            )
        indices = [atlas_labels.index(label) for label in requested]

    args.stim_region = indices
    args.stim_region_label = [atlas_labels[index] for index in indices]
    return atlas


def _stim_onsets(trial_seeds: list[int], *, transient_ms: float, t_analysis_ms: float, trial_sim_ms: float, seed: int) -> dict[int, float]:
    rng = np.random.default_rng(int(seed))
    max_seed = max(trial_seeds) if trial_seeds else -1
    lower = int(np.ceil(float(transient_ms) + float(t_analysis_ms)))
    upper = int(np.floor(float(trial_sim_ms) - float(t_analysis_ms)))
    candidates = np.arange(lower, upper, dtype=int)
    n_needed = max_seed + 1
    if n_needed > candidates.size:
        raise ValueError(
            f"Cannot draw {n_needed} unique stimulation onsets from "
            f"[{lower}, {upper}) ms."
        )
    draws = rng.choice(candidates, size=n_needed, replace=False)
    all_onsets = {
        trial_seed: float(draws[trial_seed])
        for trial_seed in range(n_needed)
    }
    return {int(s): all_onsets[int(s)] for s in trial_seeds}


def _gk_profile_from_occupancy(
    *,
    occupancy: float,
    receptor_map: np.ndarray,
    e_l_start: float,
    e_l_drug: float,
    e_na: float = 50.0,
    e_k: float = -90.0,
) -> tuple[np.ndarray, float, float, float]:
    g_k_ctrl, g_na = leak_to_conductances(e_na, e_k, e_l_start, g_L=10.0)
    g_k_drug, _ = leak_to_conductances(e_na, e_k, e_l_drug, g_Na=g_na)
    rec = np.asarray(receptor_map, dtype=float).reshape(-1)
    rec_norm = (rec - float(np.min(rec))) / (float(np.max(rec)) - float(np.min(rec)) + 1e-12)
    g_k = float(g_k_ctrl) - float(occupancy) * rec_norm * (float(g_k_ctrl) - float(g_k_drug))
    eff_max_region = float(el_eff_from_gK_gNa(float(np.min(g_k)), g_na, E_K=e_k, E_Na=e_na))
    return np.asarray(g_k, dtype=float), float(g_na), float(g_k_ctrl), eff_max_region


def _build_parameter_model(condition: str, occupancy: float, receptor_map: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    base = deepcopy(BASE_PARAMETER_MODEL_NEW)
    b_e_override = getattr(args, "b_e_override", None)
    base["b_e"] = float(
        CONDITION_B_GRADIENT[condition]
        if b_e_override is None
        else b_e_override
    )

    if float(occupancy) <= 0.0 and not bool(getattr(args, "split_model_all_occupancies", False)):
        return base

    g_ke, g_na_e, _gke_ctrl, e_eff_e_max = _gk_profile_from_occupancy(
        occupancy=occupancy,
        receptor_map=receptor_map,
        e_l_start=float(base["E_L_e"]),
        e_l_drug=float(args.e_l_e_drug),
    )
    g_ki, g_na_i, _gki_ctrl, e_eff_i_max = _gk_profile_from_occupancy(
        occupancy=occupancy,
        receptor_map=receptor_map,
        e_l_start=float(base["E_L_i"]),
        e_l_drug=float(args.e_l_i_drug),
    )

    base.update(
        {
            "g_K_e": g_ke.tolist(),
            "g_Na_e": float(g_na_e),
            "g_K_i": g_ki.tolist(),
            "g_Na_i": float(g_na_i),
            # Stored for output provenance only. Model selection is controlled
            # by WholeBrainConfig.zerlaut_gk_gna.
            "serotonergic_occupancy": float(occupancy),
            "serotonergic_e_eff_e_highest_receptor": float(e_eff_e_max),
            "serotonergic_e_eff_i_highest_receptor": float(e_eff_i_max),
        }
    )
    return base


def _run_trial(
    *,
    scenario_key: str,
    scenario_cfg: dict[str, Any],
    cohort: str,
    condition: str,
    subject_id: str,
    trial_seed: int,
    occupancy: float,
    receptor_map: np.ndarray,
    output_dir: Path,
    stim_onset_ms: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    t0 = perf_counter()
    c, l, atlas, meta = load_subject_structural(
        subject_id=subject_id,
        cohort=cohort,
        dataset_root=args.dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    atlas_labels = np.asarray(atlas.labels, dtype="U128").reshape(-1)
    if atlas_labels.shape != (90,) or len(set(atlas_labels.tolist())) != 90:
        raise ValueError("Subject structural atlas must contain 90 unique labels.")
    atlas_labels_sha256 = _sha256_array(atlas_labels)
    if atlas_labels_sha256 != str(args.atlas_labels_sha256):
        raise RuntimeError(
            "Subject atlas labels differ from the atlas used to align the "
            "stimulation target and receptor map."
        )
    receptor_map_array = np.asarray(receptor_map, dtype=np.float64).reshape(-1)
    receptor_map_sha256 = _sha256_array(receptor_map_array)
    if receptor_map_array.shape != (90,) or not np.isfinite(receptor_map_array).all():
        raise ValueError("Aligned receptor map must contain 90 finite values.")
    if receptor_map_sha256 != str(args.receptor_map_sha256):
        raise RuntimeError("Receptor-map hash changed between setup and simulation.")
    c, l, sc_zero_frac = _apply_damage_parity(c, l, cohort)

    parameter_model = _build_parameter_model(condition, occupancy, receptor_map, args)
    parameter_model.update(
        {
            "noise_alpha": float(scenario_cfg["noise_alpha"]),
            "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
        }
    )

    parameter_stimulus = {
        "stimtime": float(stim_onset_ms),
        "stimdur": float(args.stim_duration_ms),
        "stimperiod": float(args.trial_sim_ms) * 10.0,
        "stimval": float(args.stim_amplitude),
        "stimregion": [int(x) for x in args.stim_region],
        "stimvariables": [0],
    }

    wb_cfg = WholeBrainConfig(
        simulation_length_ms=float(args.trial_sim_ms),
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=0.25,
        model_family="adex_zerlaut",
        zerlaut_matteo=False,
        zerlaut_gk_gna=bool(
            float(occupancy) > 0.0
            or bool(getattr(args, "split_model_all_occupancies", False))
        ),
        zerlaut_order=2,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(RATE_MONITOR_PERIOD_MS_OLD),
        monitor_variables=(0, 1),
        weights=np.asarray(c, dtype=float),
        tract_lengths=np.asarray(l, dtype=float),
        parameter_overrides={
            "parameter_model": parameter_model,
            "parameter_stimulus": parameter_stimulus,
        },
    )

    sim = run_whole_brain_simulation(wb_cfg, seed=int(trial_seed))
    t_ms = np.asarray(sim.time_ms, dtype=float)
    x = np.asarray(sim.raw, dtype=float)
    keep = t_ms >= float(args.transient_ms)
    t_post = t_ms[keep]
    x_post = x[keep]
    if t_post.ndim != 1 or t_post.size < 2:
        raise RuntimeError("Simulation produced too few post-transient time samples.")
    if x_post.ndim != 2 or x_post.shape != (t_post.size, 90):
        raise RuntimeError(
            "Simulation rate output must have shape (post-transient time, 90); "
            f"got {x_post.shape}."
        )
    if not np.isfinite(t_post).all() or not np.isfinite(x_post).all():
        raise RuntimeError("Simulation produced non-finite time or rate values.")
    monitor_dt_ms = float(np.median(np.diff(t_post)))
    onset_sample_index = int(np.argmin(np.abs(t_post - float(stim_onset_ms))))
    onset_sample_ms = float(t_post[onset_sample_index])
    onset_residual_ms = float(onset_sample_ms - float(stim_onset_ms))
    if abs(onset_residual_ms) > 0.5 * monitor_dt_ms + 1e-6:
        raise RuntimeError(
            "Saved stimulation onset is farther than half a monitor sample "
            "from the post-transient time grid."
        )
    analysis_bins = int(round(float(args.t_analysis_ms) / monitor_dt_ms))
    if (
        onset_sample_index - analysis_bins < 0
        or onset_sample_index + analysis_bins > t_post.size
    ):
        raise RuntimeError(
            "Simulation does not contain the complete requested peri-stimulus window."
        )

    save_path = output_dir / f"trial_{int(trial_seed):03d}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        save_path,
        time_ms=t_post,
        rate=x_post,
        region_labels=atlas_labels,
        simulation_region_labels=np.asarray(sim.region_labels),
        atlas_ordering=np.asarray([str(atlas.ordering)], dtype="U128"),
        atlas_labels_sha256=np.asarray([atlas_labels_sha256], dtype="U128"),
        receptor_tracer=np.asarray([str(args.receptor_tracer)], dtype="U32"),
        receptor_csv_sha256=np.asarray(
            [str(args.receptor_csv_sha256)],
            dtype="U128",
        ),
        receptor_map_alignment=np.asarray(
            ["AAL region-label join"],
            dtype="U128",
        ),
        receptor_map_sha256=np.asarray([receptor_map_sha256], dtype="U128"),
        receptor_map=receptor_map_array,
        protocol_version=np.asarray([PROTOCOL_VERSION], dtype="U128"),
        protocol_fingerprint=np.asarray(
            [str(args.protocol_fingerprint)],
            dtype="U128",
        ),
        cohort=np.asarray([str(cohort)], dtype="U32"),
        condition=np.asarray([str(condition)], dtype="U32"),
        subject_id=np.asarray([str(subject_id)], dtype="U128"),
        scenario=np.asarray([str(scenario_key)], dtype="U128"),
        stim_onset_ms=np.array([float(stim_onset_ms)]),
        stim_onset_sample_index=np.asarray([onset_sample_index], dtype=np.int64),
        stim_onset_sample_ms=np.asarray([onset_sample_ms], dtype=float),
        stim_onset_alignment_residual_ms=np.asarray(
            [onset_residual_ms],
            dtype=float,
        ),
        stim_onset_alignment=np.asarray(
            ["nearest temporal-average sample after per-trial epoching"],
            dtype="U128",
        ),
        t_analysis_ms=np.array([float(args.t_analysis_ms)]),
        rate_monitor_period_ms=np.array([float(RATE_MONITOR_PERIOD_MS_OLD)]),
        trial_seed=np.array([int(trial_seed)]),
        noise_alpha=np.array([float(scenario_cfg["noise_alpha"])]),
        stim_amplitude=np.array([float(args.stim_amplitude)]),
        stim_duration_ms=np.array([float(args.stim_duration_ms)]),
        stim_region=np.array(args.stim_region, dtype=int),
        stim_region_labels=np.asarray(
            [str(atlas_labels[int(index)]) for index in args.stim_region],
            dtype="U128",
        ),
        occupancy=np.array([float(occupancy)]),
        sc_zero_fraction_upper=np.array([float(sc_zero_frac)]),
    )

    return {
        "cohort": cohort,
        "condition": condition,
        "subject_id": subject_id,
        "stage": str(getattr(meta, "stage", "") or ""),
        "sedation": str(getattr(meta, "sedation", "") or ""),
        "scenario": scenario_key,
        "occupancy": float(occupancy),
        "trial_seed": int(trial_seed),
        "runtime_s": float(perf_counter() - t0),
        "save_path": str(save_path),
    }


def _run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    return _run_trial(**job)


def _load_trials(paths: list[Path]) -> tuple[list[np.ndarray], int, float, float]:
    """Cut every trial around its own onset and align on the common midpoint."""
    records: list[tuple[Path, np.ndarray, np.ndarray, float]] = []
    dt_vals: list[float] = []
    t_analysis_vals: list[float] = []
    n_regions_ref: int | None = None
    region_labels_ref: list[str] | None = None
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            t = np.asarray(d["time_ms"], dtype=float)
            x = np.asarray(d["rate"], dtype=float)
            onset_abs = float(np.ravel(d["stim_onset_ms"])[0])
            t_analysis = float(np.ravel(d["t_analysis_ms"])[0])
            region_labels = (
                [
                    str(value)
                    for value in np.asarray(d["region_labels"]).reshape(-1)
                ]
                if "region_labels" in d.files
                else None
            )

        if t.ndim != 1 or t.size < 2:
            raise ValueError(f"{p}: time_ms must be one-dimensional with >=2 samples.")
        if x.ndim != 2 or x.shape[0] != t.size:
            raise ValueError(
                f"{p}: rate must have shape (time, regions); got {x.shape} "
                f"for {t.size} time samples."
            )
        if not np.isfinite(t).all() or not np.isfinite(x).all():
            raise ValueError(f"{p}: time_ms and rate must contain only finite values.")
        differences = np.diff(t)
        if np.any(differences <= 0.0):
            raise ValueError(f"{p}: time_ms must be strictly increasing.")
        dt = float(np.median(differences))
        if not np.allclose(differences, dt, rtol=0.0, atol=1e-6):
            raise ValueError(f"{p}: time_ms must use a uniform sampling interval.")
        if not np.isfinite(onset_abs) or not np.isfinite(t_analysis):
            raise ValueError(f"{p}: stimulation onset/window metadata must be finite.")
        if t_analysis <= 0.0:
            raise ValueError(f"{p}: t_analysis_ms must be positive.")

        n_regions = int(x.shape[1])
        if n_regions_ref is None:
            n_regions_ref = n_regions
        elif n_regions != n_regions_ref:
            raise ValueError(
                f"{p}: region count {n_regions} differs from {n_regions_ref}."
            )
        if region_labels is not None:
            if len(region_labels) != n_regions:
                raise ValueError(
                    f"{p}: region_labels has {len(region_labels)} entries for "
                    f"{n_regions} signal regions."
                )
            if region_labels_ref is None:
                region_labels_ref = region_labels
            elif region_labels != region_labels_ref:
                raise ValueError(f"{p}: region label order differs across trials.")

        records.append((p, t, x, onset_abs))
        dt_vals.append(dt)
        t_analysis_vals.append(t_analysis)

    if not records:
        raise ValueError("No trial files provided.")
    dt_ref = float(np.median(dt_vals))
    t_analysis_ref = float(np.median(t_analysis_vals))
    if not np.allclose(dt_vals, dt_ref, rtol=0.0, atol=1e-6):
        raise ValueError(f"Trial sampling intervals differ: {dt_vals}")
    if not np.allclose(t_analysis_vals, t_analysis_ref, rtol=0.0, atol=1e-6):
        raise ValueError(f"Trial analysis windows differ: {t_analysis_vals}")

    nbins_analysis = int(round(t_analysis_ref / dt_ref))
    if nbins_analysis < 1:
        raise ValueError("Analysis window is shorter than one sampled time bin.")
    aligned_trials: list[np.ndarray] = []
    aligned_shape: tuple[int, int] | None = None
    for trial_index, (path, t, x, onset_abs) in enumerate(records):
        onset = int(np.argmin(np.abs(t - onset_abs)))
        alignment_residual_ms = float(t[onset] - onset_abs)
        if abs(alignment_residual_ms) > (0.5 * dt_ref + 1e-6):
            raise ValueError(
                f"{path}: nearest-sample stimulation alignment residual "
                f"{alignment_residual_ms:.6g} ms exceeds half a sample."
            )
        start = onset - nbins_analysis
        stop = onset + nbins_analysis
        if start < 0 or stop > x.shape[0]:
            raise ValueError(
                f"Trial {trial_index} ({path}): stimulation window "
                f"[{start}, {stop}) is out of bounds for {x.shape[0]} samples"
            )
        aligned = x[start:stop]
        if aligned_shape is None:
            aligned_shape = aligned.shape
        elif aligned.shape != aligned_shape:
            raise ValueError(
                f"{path}: aligned shape {aligned.shape} differs from "
                f"{aligned_shape}."
            )
        aligned_trials.append(aligned)

    # Every returned (time, region) epoch has its onset at nbins_analysis.
    return aligned_trials, nbins_analysis, dt_ref, t_analysis_ref


def _compute_pci_for_condition(
    paths: list[Path],
    *,
    binarise_method: str = "casali",
    n_bootstrap: int = 500,
    alpha: float = 0.01,
    bootstrap_seed: int = 0,
) -> tuple[float, np.ndarray]:
    trials, onset, dt_ms, t_analysis_ms = _load_trials(paths)
    binarise_kwargs = None
    if str(binarise_method).lower() == "casali":
        binarise_kwargs = {
            "n_bootstrap": int(n_bootstrap),
            "alpha": float(alpha),
            "seed": int(bootstrap_seed),
        }
    pci_mean, pci_values = pci_casali_like_multi_trial(
        trials,
        stimulation_index=onset,
        t_analysis_ms=t_analysis_ms,
        dt_ms=dt_ms,
        binarise_method=binarise_method,
        binarise_kwargs=binarise_kwargs,
    )
    if str(binarise_method).lower() == "casali" and np.asarray(pci_values).shape != (1,):
        raise AssertionError(
            "Canonical Casali PCI must return one value for the trial-averaged response."
        )
    return pci_mean, pci_values


def _condition_paths(
    root: Path,
    occ: float,
    scenario: str,
    cohort: str,
    subject_id: str,
    trial_seeds: list[int],
    *,
    simulated_baseline: bool = False,
) -> list[Path]:
    if float(occ) <= 0.0 and not simulated_baseline:
        base = root / "condb_doc_gradient" / scenario / cohort / subject_id
    else:
        base = root / "sims_pci" / _occ_tag(occ) / scenario / cohort / subject_id
    return [base / f"trial_{int(seed):03d}.npz" for seed in trial_seeds]


def _plot(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"

    subj = metrics.copy()
    subj["condition"] = pd.Categorical(subj["condition"], categories=CONDITION_ORDER, ordered=True)
    subj = subj.sort_values(["condition", "subject_id", "occupancy"])

    baseline = subj.loc[subj["occupancy"] == 0.0, ["cohort", "subject_id", "pci_mean"]].rename(
        columns={"pci_mean": "pci_baseline"}
    )
    subj = subj.merge(baseline, on=["cohort", "subject_id"], how="left")
    subj["pci_rescue"] = subj["pci_mean"] - subj["pci_baseline"]
    subj.to_csv(tables_dir / "serotonergic_pci_subject_metrics_with_rescue.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax = axes[0]
    for (condition, subject_id), g in subj.groupby(["condition", "subject_id"], observed=True):
        color = COND_COLORS.get(str(condition), "#555555")
        ax.plot(g["occupancy"], g["pci_mean"], marker="o", linewidth=1.8, color=color, alpha=0.9)
        ax.text(float(g["occupancy"].max()) + 0.01, float(g["pci_mean"].iloc[-1]), f"{condition}:{subject_id}", fontsize=7, color=color)
    ax.set_xlabel("5-HT2A occupancy")
    ax.set_ylabel("PCI")
    ax.set_title("Subject PCI")
    ax.grid(alpha=0.25)

    ax = axes[1]
    nz = subj[subj["occupancy"] > 0.0]
    for condition, g in nz.groupby("condition", observed=True):
        color = COND_COLORS.get(str(condition), "#555555")
        ax.scatter(g["occupancy"], g["pci_rescue"], label=str(condition), s=42, color=color, alpha=0.9)
        mean = g.groupby("occupancy", as_index=False)["pci_rescue"].mean()
        ax.plot(mean["occupancy"], mean["pci_rescue"], color=color, linewidth=2.0)
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xlabel("5-HT2A occupancy")
    ax.set_ylabel("PCI rescue vs baseline")
    ax.set_title("PCI Rescue")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"serotonergic_pci_pilot.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _validate_protocol_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    if args.scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {args.scenario!r}.")
    scenario_cfg = SCENARIOS[args.scenario]
    subjects = _select_subjects(args.dataset_root, args.cohorts, args.subjects_per_cohort, args.subject)
    atlas = _resolve_stim_regions(args)
    receptor_map = get_5ht2a_aal90(
        tracer=str(args.receptor_tracer),
        csv_path=args.receptor_csv,
        target_labels=atlas.labels,
    )
    args.atlas_labels_sha256 = _sha256_array(
        np.asarray(atlas.labels, dtype="U128")
    )
    args.receptor_csv_sha256 = _sha256_file(args.receptor_csv)
    args.receptor_map_sha256 = _sha256_array(
        np.asarray(receptor_map, dtype=np.float64)
    )
    stim_onsets = _stim_onsets(
        [int(s) for s in args.trial_seeds],
        transient_ms=float(args.transient_ms),
        t_analysis_ms=float(args.t_analysis_ms),
        trial_sim_ms=float(args.trial_sim_ms),
        seed=int(args.stim_onset_seed),
    )

    manifest = {
        "script": "scripts/run_serotonergic_pci_pilot.py",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_root": str(args.dataset_root),
        "baseline_root": str(args.baseline_root),
        "output_root": str(args.output_root),
        "scenario": args.scenario,
        "scenario_cfg": scenario_cfg,
        "subjects": [s.__dict__ for s in subjects],
        "trial_seeds": [int(s) for s in args.trial_seeds],
        "n_trials": int(len(args.trial_seeds)),
        "stim_onsets_ms_by_trial_seed": {str(k): float(v) for k, v in stim_onsets.items()},
        "stim_onset_schedule": "unique integer-millisecond onsets sampled without replacement",
        "occupancies": [float(o) for o in args.occupancies],
        "transient_ms": float(args.transient_ms),
        "t_analysis_ms": float(args.t_analysis_ms),
        "trial_sim_ms": float(args.trial_sim_ms),
        "integration_dt_ms": 0.1,
        "rate_monitor_period_ms": float(RATE_MONITOR_PERIOD_MS_OLD),
        "conduction_speed": 4.0,
        "coupling_strength": 0.25,
        "stim_amplitude": float(args.stim_amplitude),
        "stim_duration_ms": float(args.stim_duration_ms),
        "stim_variables": [0],
        "stim_onset_alignment": (
            "per-trial nearest temporal-average sample, then [-window,+window) "
            "epoching to a common midpoint"
        ),
        "pci_binarise_method": str(args.pci_binarise_method),
        "pci_estimator": (
            "one Casali PCI from the baseline-normalized, time-locked "
            "trial-averaged response"
            if str(args.pci_binarise_method) == "casali"
            else "legacy mean of per-trial TVBSim PCI values"
        ),
        "pci_bootstrap_replicates": int(args.pci_bootstrap_replicates),
        "pci_alpha": float(args.pci_alpha),
        "pci_bootstrap_seed": int(args.pci_bootstrap_seed),
        "atlas_ordering": str(atlas.ordering),
        "atlas_source": str(atlas.source),
        "atlas_labels_sha256": str(args.atlas_labels_sha256),
        "receptor_map_alignment": "AAL region-label join",
        "receptor_tracer": str(args.receptor_tracer),
        "receptor_csv": str(args.receptor_csv),
        "receptor_csv_sha256": str(args.receptor_csv_sha256),
        "receptor_map_sha256": str(args.receptor_map_sha256),
        "stim_region_indices_zero_based": [int(index) for index in args.stim_region],
        "stim_region_labels": [str(label) for label in args.stim_region_label],
        "stim_target_provenance": (
            "Resolved by AAL label; default follows the original notebook's "
            "documented Supp_Motor_Area_L intent, not the legacy positional index."
        ),
        "model_form": (
            "split_gK_gNa_all_occupancies"
            if bool(args.split_model_all_occupancies)
            else "legacy_switch_at_positive_occupancy"
        ),
        "simulate_baseline": bool(args.split_model_all_occupancies),
        "e_l_e_drug": float(args.e_l_e_drug),
        "e_l_i_drug": float(args.e_l_i_drug),
        "b_e_override": (
            None if args.b_e_override is None else float(args.b_e_override)
        ),
        "diagnosis_configured_b_e_pA": {
            str(condition): float(value)
            for condition, value in CONDITION_B_GRADIENT.items()
        },
        "workers": int(args.workers),
        "overwrite": bool(args.overwrite),
    }
    manifest["protocol_fingerprint"] = _protocol_fingerprint(manifest)
    args.protocol_fingerprint = str(manifest["protocol_fingerprint"])
    _write_or_validate_manifest(
        args.output_root / "logs" / "run_manifest.json",
        manifest,
        overwrite=bool(args.overwrite),
    )

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    trial_jobs: list[dict[str, Any]] = []
    for occ in [
        float(o)
        for o in args.occupancies
        if float(o) > 0.0 or bool(args.split_model_all_occupancies)
    ]:
        for sj in subjects:
            out_dir = args.output_root / "sims_pci" / _occ_tag(occ) / args.scenario / sj.cohort / sj.subject_id
            for trial_seed in [int(s) for s in args.trial_seeds]:
                save_path = out_dir / f"trial_{trial_seed:03d}.npz"
                if save_path.exists() and not args.overwrite:
                    _validate_existing_trial(
                        save_path,
                        protocol_fingerprint=args.protocol_fingerprint,
                        trial_seed=trial_seed,
                        occupancy=occ,
                        stim_region_labels=args.stim_region_label,
                        receptor_map_sha256=args.receptor_map_sha256,
                        cohort=sj.cohort,
                        condition=sj.condition,
                        subject_id=sj.subject_id,
                        scenario=args.scenario,
                        expected_stim_onset_ms=stim_onsets[trial_seed],
                        atlas_labels_sha256=args.atlas_labels_sha256,
                        receptor_tracer=args.receptor_tracer,
                        receptor_csv_sha256=args.receptor_csv_sha256,
                        expected_t_analysis_ms=args.t_analysis_ms,
                    )
                    continue
                trial_jobs.append(
                    {
                        "scenario_key": args.scenario,
                        "scenario_cfg": scenario_cfg,
                        "cohort": sj.cohort,
                        "condition": sj.condition,
                        "subject_id": sj.subject_id,
                        "trial_seed": trial_seed,
                        "occupancy": occ,
                        "receptor_map": receptor_map,
                        "output_dir": out_dir,
                        "stim_onset_ms": float(stim_onsets[trial_seed]),
                        "args": args,
                    }
                )

    completed_rows: list[dict[str, Any]] = []
    print(f"[sero-pci] queued {len(trial_jobs)} serotonergic trial simulations on {int(args.workers)} workers", flush=True)
    if trial_jobs:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=worker_initializer) as ex:
            futures = [ex.submit(_run_trial_job, job) for job in trial_jobs]
            total = len(futures)
            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                completed_rows.append(row)
                print(
                    "[sero-pci] "
                    f"{i}/{total} done occ={row['occupancy']:.3f} "
                    f"{row['condition']}/{row['subject_id']} trial={row['trial_seed']} "
                    f"runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    _write_csv(args.output_root / "logs" / "completed_trials.csv", completed_rows)

    metric_rows: list[dict[str, Any]] = []
    for sj in subjects:
        for occ in [float(o) for o in args.occupancies]:
            simulated_baseline = bool(
                occ <= 0.0 and args.split_model_all_occupancies
            )
            root = args.output_root if simulated_baseline or occ > 0.0 else args.baseline_root
            paths = _condition_paths(
                root,
                occ,
                args.scenario,
                sj.cohort,
                sj.subject_id,
                [int(s) for s in args.trial_seeds],
                simulated_baseline=simulated_baseline,
            )
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing trial files for {sj.subject_id} occ={occ}: {missing[:3]}")
            pci_mean, pci_per_trial = _compute_pci_for_condition(
                paths,
                binarise_method=args.pci_binarise_method,
                n_bootstrap=args.pci_bootstrap_replicates,
                alpha=args.pci_alpha,
                bootstrap_seed=args.pci_bootstrap_seed,
            )
            metric_rows.append(
                {
                    "cohort": sj.cohort,
                    "condition": sj.condition,
                    "subject_id": sj.subject_id,
                    "scenario": args.scenario,
                    "occupancy": float(occ),
                    "n_trials": int(len(paths)),
                    "pci_estimator": (
                        "casali_trial_average"
                        if str(args.pci_binarise_method) == "casali"
                        else "tvbsim_mean_per_trial"
                    ),
                    "pci_mean": float(pci_mean),
                    "pci_std": (
                        float("nan")
                        if str(args.pci_binarise_method) == "casali"
                        else float(np.std(pci_per_trial))
                    ),
                    "n_returned_pci_values": int(len(pci_per_trial)),
                    "pci_values": json.dumps([float(x) for x in pci_per_trial]),
                    "trial_paths": json.dumps([str(p) for p in paths]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_root / "tables" / "serotonergic_pci_subject_metrics.csv", index=False)
    _plot(metrics, args.output_root)
    print(f"[sero-pci] wrote {args.output_root}")


if __name__ == "__main__":
    main()
