#!/usr/bin/env python3
"""Single-subject AdEx sensitivity analysis for the EMCS PCI rescue result.

The production grid uses one pre-specified, neutral EMCS representative
(``e0001``), 100 matched random seeds, four 5-HT2A occupancies, and one-at-a-time
plus/minus 20% changes to six AdEx parameter families.  The nominal condition
plus the twelve perturbations gives 13 configurations (5,200 simulations).

This runner deliberately follows the corrected serotonergic PCI protocol:

* every trial is time-locked to its own stored stimulation onset;
* the 5-HT2A map is joined to the converted AAL90 atlas by region label;
* stimulation defaults to the explicit ``Supp_Motor_Area_L`` label; and
* the split gK/gNa Zerlaut model is used at occupancy zero and positive doses.

Outputs are resumable.  Existing trial files are reused only when their
protocol fingerprint and model-override provenance match the requested run.
Use ``--dry-run --max-configs 2 --trial-seeds 0 1`` for a small planning check;
the dry run does not launch simulations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
import uuid

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(_REPO_ROOT / ".tvb-temp"))

import sys

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_serotonergic_pci_pilot as pilot
from run_serotonergic_pci_pilot import worker_initializer
from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.whole_brain.legacy_engine.parameter.parameter_M_Berlin_new import (
    Parameters,
)


PROTOCOL_VERSION = (
    "emcs-adex-robustness-v2.1-100trials-time-locked-atlas-aligned-provenance"
)
SUBJECT_SELECTION_RATIONALE = (
    "Pre-specified EMCS exemplar fixed before the corrected full-cohort rerun, "
    "so the robustness subject is not selected on the corrected PCI outcome."
)

# Each family is changed one at a time.  The excitatory decay pair has an
# additional backend alias (tau_e) because the split-gK/gNa implementation
# consumes one shared excitatory synaptic time constant internally.
PARAMETER_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "b_e",
        "slug": "b_e",
        "display": r"$b_e$",
        "keys": ("b_e",),
    },
    {
        "family": "tau_w_e",
        "slug": "tau_w_e",
        "display": r"$\tau_{w,e}$",
        "keys": ("tau_w_e",),
    },
    {
        "family": "tau_e_e_tau_e_i",
        "slug": "tau_e",
        "display": r"$\tau_{e,e}$ / $\tau_{e,i}$",
        "keys": ("tau_e_e", "tau_e_i"),
    },
    {
        "family": "tau_i",
        "slug": "tau_i",
        "display": r"$\tau_i$",
        "keys": ("tau_i",),
    },
    {
        "family": "external_input_ex_ex_external_input_ex_in",
        "slug": "external_input_e",
        "display": "External E drive",
        "keys": ("external_input_ex_ex", "external_input_ex_in"),
    },
    {
        "family": "weight_noise",
        "slug": "weight_noise",
        "display": "Noise weight",
        "keys": ("weight_noise",),
    },
)

FAMILY_COLORS = {
    "nominal": "#111111",
    "b_e": "#0072B2",
    "tau_w_e": "#D55E00",
    "tau_e_e_tau_e_i": "#009E73",
    "tau_i": "#CC79A7",
    "external_input_ex_ex_external_input_ex_in": "#E69F00",
    "weight_noise": "#56B4E9",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=pilot.DATASET_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "results" / "serotonergic_pci_emcs_robustness_100trials",
    )
    parser.add_argument("--subject-id", default="e0001")
    parser.add_argument("--scenario", default="private_alpha0")
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=list(range(100)))
    parser.add_argument(
        "--occupancies",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.766],
    )
    parser.add_argument("--transient-ms", type=float, default=4000.0)
    parser.add_argument("--t-analysis-ms", type=float, default=300.0)
    parser.add_argument("--trial-sim-ms", type=float, default=8000.0)
    parser.add_argument("--stim-amplitude", type=float, default=0.00030)
    parser.add_argument("--stim-duration-ms", type=float, default=10.0)
    parser.add_argument(
        "--stim-region-label",
        nargs="+",
        default=["Supp_Motor_Area_L"],
        help="AAL90 labels, resolved against the converted dataset ordering.",
    )
    parser.add_argument("--stim-onset-seed", type=int, default=0)
    parser.add_argument(
        "--receptor-tracer",
        choices=["cimbi", "savli", "talbot"],
        default="cimbi",
    )
    parser.add_argument(
        "--receptor-csv",
        type=Path,
        default=pilot.DEFAULT_RECEPTOR_CSV,
    )
    parser.add_argument(
        "--pci-binarise-method",
        choices=["casali", "tvbsim"],
        default="casali",
        help=(
            "Use canonical trial-averaged Casali PCI by default; 'tvbsim' is "
            "available only for legacy comparison."
        ),
    )
    parser.add_argument("--pci-bootstrap-replicates", type=int, default=500)
    parser.add_argument("--pci-alpha", type=float, default=0.01)
    parser.add_argument("--pci-bootstrap-seed", type=int, default=0)
    parser.add_argument("--e-l-e-drug", type=float, default=-61.2)
    parser.add_argument("--e-l-i-drug", type=float, default=-64.4)
    parser.add_argument(
        "--variation-fraction",
        type=float,
        default=0.20,
        help="Fractional one-at-a-time change around each nominal value.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Run only the first N of 13 configurations (nominal is first).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip simulations and rebuild tables/figure from existing trial files.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run simulations without computing final tables and figure.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print one progress message every N completed simulations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and write a dry-run manifest without simulating.",
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    return value


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _npz_scalar(data: Any, key: str) -> Any:
    return np.asarray(data[key]).reshape(-1)[0].item()


def _validate_inputs(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.receptor_csv = args.receptor_csv.expanduser().resolve()
    args.trial_seeds = [int(seed) for seed in args.trial_seeds]
    args.occupancies = [float(occupancy) for occupancy in args.occupancies]
    args.stim_region = None
    # This workflow never permits the legacy model-form switch.
    args.split_model_all_occupancies = True

    if not (args.dataset_root / "index.json").exists():
        raise FileNotFoundError(
            f"Converted dataset index not found: {args.dataset_root / 'index.json'}"
        )
    if not args.receptor_csv.is_file():
        raise FileNotFoundError(f"Receptor table not found: {args.receptor_csv}")
    if args.scenario not in pilot.SCENARIOS:
        raise KeyError(f"Unknown scenario {args.scenario!r}.")
    if not args.trial_seeds:
        raise ValueError("--trial-seeds must contain at least one seed.")
    if len(set(args.trial_seeds)) != len(args.trial_seeds):
        raise ValueError("--trial-seeds must be unique.")
    if any(seed < 0 for seed in args.trial_seeds):
        raise ValueError("--trial-seeds must be non-negative.")
    if not args.dry_run and args.trial_seeds != list(range(100)):
        raise ValueError(
            "The production robustness run requires exactly trial seeds "
            "0..99 (100 matched trials). Use --dry-run for smaller planning checks."
        )
    if len(set(args.occupancies)) != len(args.occupancies):
        raise ValueError("--occupancies must be unique.")
    if not any(np.isclose(occupancy, 0.0) for occupancy in args.occupancies):
        raise ValueError("--occupancies must include baseline occupancy 0.0.")
    if not any(occupancy > 0.0 for occupancy in args.occupancies):
        raise ValueError("--occupancies must include at least one positive dose.")
    if any(occupancy < 0.0 or occupancy > 1.0 for occupancy in args.occupancies):
        raise ValueError("--occupancies must lie within [0, 1].")
    if not 0.0 < float(args.variation_fraction) < 1.0:
        raise ValueError("--variation-fraction must lie strictly between 0 and 1.")
    if args.max_configs is not None and int(args.max_configs) < 1:
        raise ValueError("--max-configs must be at least 1.")
    if int(args.workers) < 1:
        raise ValueError("--workers must be at least 1.")
    if int(args.progress_every) < 1:
        raise ValueError("--progress-every must be at least 1.")
    if int(args.pci_bootstrap_replicates) < 1:
        raise ValueError("--pci-bootstrap-replicates must be at least 1.")
    if not 0.0 < float(args.pci_alpha) < 1.0:
        raise ValueError("--pci-alpha must lie strictly between 0 and 1.")
    if args.aggregate_only and args.overwrite:
        raise ValueError(
            "--aggregate-only cannot be combined with --overwrite because "
            "aggregation must validate the existing manifest and trial files."
        )
    if not args.dry_run:
        expected_occupancies = [0.0, 0.25, 0.5, 0.766]
        if len(args.occupancies) != len(expected_occupancies) or not np.allclose(
            np.asarray(args.occupancies, dtype=float),
            np.asarray(expected_occupancies, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "The production robustness dose schedule is exactly "
                "0, 0.25, 0.5, and 0.766."
            )
        if str(args.subject_id) != "e0001":
            raise ValueError(
                "The production robustness subject is the pre-specified EMCS "
                "subject 'e0001'."
            )
        if str(args.scenario) != "private_alpha0":
            raise ValueError(
                "The approved robustness scenario is 'private_alpha0'."
            )
        if [str(label) for label in args.stim_region_label] != [
            pilot.DEFAULT_STIM_REGION_LABEL
        ]:
            raise ValueError(
                "The approved robustness target is exactly "
                f"{pilot.DEFAULT_STIM_REGION_LABEL!r}."
            )
        if str(args.receptor_tracer) != "cimbi":
            raise ValueError(
                "The approved production receptor tracer is 'cimbi'."
            )
        if str(args.pci_binarise_method) != "casali":
            raise ValueError(
                "The production robustness analysis requires Casali PCI."
            )
        if not np.isclose(
            float(args.e_l_e_drug),
            -61.2,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "The production excitatory drug endpoint is -61.2 mV."
            )
        if not np.isclose(
            float(args.e_l_i_drug),
            -64.4,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "The production inhibitory drug endpoint is -64.4 mV."
            )
        if not np.isclose(
            float(args.variation_fraction),
            0.20,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "The production robustness grid requires exactly ±20% "
                "one-at-a-time parameter changes."
            )
        if args.max_configs is not None:
            raise ValueError(
                "The production robustness analysis requires all 13 "
                "configurations; --max-configs is dry-run only."
            )


def _select_subject(dataset_root: Path, subject_id: str) -> Any:
    matches = [
        subject
        for subject in pilot.get_subject_jobs(dataset_root)
        if subject.cohort == "emcs" and subject.subject_id == str(subject_id)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one EMCS subject {subject_id!r}, found {len(matches)}."
        )
    subject = matches[0]
    if subject.condition != "EMCS":
        raise ValueError(
            f"Subject {subject_id!r} resolved to condition {subject.condition!r}, not EMCS."
        )
    return subject


def _nominal_model_values() -> dict[str, float]:
    """Return the scalar values around which the sensitivity grid is built."""
    defaults = Parameters().parameter_model
    base = pilot.BASE_PARAMETER_MODEL_NEW
    values = {
        # Condition-b uses the diagnosis gradient, so the EMCS nominal is 30 pA.
        "b_e": float(pilot.CONDITION_B_GRADIENT["EMCS"]),
        "tau_w_e": float(defaults["tau_w_e"]),
        "tau_e_e": float(base.get("tau_e_e", defaults["tau_e_e"])),
        "tau_e_i": float(base.get("tau_e_i", defaults["tau_e_i"])),
        "tau_i": float(defaults["tau_i"]),
        "external_input_ex_ex": float(defaults["external_input_ex_ex"]),
        "external_input_ex_in": float(defaults["external_input_ex_in"]),
        "weight_noise": float(defaults["weight_noise"]),
    }
    return values


def _effective_model_overrides(
    nominal_values: dict[str, float],
    *,
    family_keys: tuple[str, ...],
    factor: float,
) -> dict[str, float]:
    overrides = {key: float(value) for key, value in nominal_values.items()}
    for key in family_keys:
        overrides[key] = float(nominal_values[key]) * float(factor)

    # Zerlaut_gK_gNa uses a shared tau_e internally.  Keeping this alias equal
    # to the requested tau_e_e/tau_e_i pair makes that sensitivity operational,
    # while retaining the explicit pair for cross-model provenance.
    overrides["tau_e"] = float(overrides["tau_e_e"])
    return overrides


def build_robustness_configs(
    nominal_values: dict[str, float],
    variation_fraction: float = 0.20,
) -> list[dict[str, Any]]:
    nominal_overrides = _effective_model_overrides(
        nominal_values,
        family_keys=(),
        factor=1.0,
    )
    configs: list[dict[str, Any]] = [
        {
            "config_index": 0,
            "config_id": "nominal",
            "config_label": "Nominal",
            "family": "nominal",
            "family_display": "Nominal",
            "direction": "nominal",
            "factor": 1.0,
            "varied_parameters": [],
            "parameter_model_overrides": nominal_overrides,
        }
    ]

    factors = (
        ("minus20", "minus", 1.0 - float(variation_fraction)),
        ("plus20", "plus", 1.0 + float(variation_fraction)),
    )
    for family in PARAMETER_FAMILIES:
        keys = tuple(str(key) for key in family["keys"])
        for tag, direction, factor in factors:
            label_sign = "\N{MINUS SIGN}" if direction == "minus" else "+"
            configs.append(
                {
                    "config_index": len(configs),
                    "config_id": f"{family['slug']}_{tag}",
                    "config_label": (
                        f"{family['display']} {label_sign}"
                        f"{100.0 * float(variation_fraction):.0f}%"
                    ),
                    "family": str(family["family"]),
                    "family_display": str(family["display"]),
                    "direction": direction,
                    "factor": float(factor),
                    "varied_parameters": list(keys),
                    "parameter_model_overrides": _effective_model_overrides(
                        nominal_values,
                        family_keys=keys,
                        factor=factor,
                    ),
                }
            )
    if len(configs) != 13:
        raise AssertionError(f"Expected 13 robustness configurations, got {len(configs)}.")
    return configs


def _trial_path(
    output_root: Path,
    config_id: str,
    occupancy: float,
    scenario: str,
    subject_id: str,
    trial_seed: int,
) -> Path:
    return (
        output_root
        / "sims_pci"
        / config_id
        / pilot._occ_tag(occupancy)
        / scenario
        / "emcs"
        / subject_id
        / f"trial_{int(trial_seed):03d}.npz"
    )


def _build_protocol_manifest(
    args: argparse.Namespace,
    subject: Any,
    scenario_cfg: dict[str, Any],
    atlas: Any,
    receptor_map: np.ndarray,
    stim_onsets: dict[int, float],
    configs: list[dict[str, Any]],
    nominal_values: dict[str, float],
) -> dict[str, Any]:
    protocol = {
        "script": "scripts/run_serotonergic_pci_emcs_robustness.py",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_root": str(args.dataset_root),
        "scenario": str(args.scenario),
        "scenario_cfg": deepcopy(scenario_cfg),
        "subject": subject.__dict__,
        "subject_selection_rationale": SUBJECT_SELECTION_RATIONALE,
        "cohort_scope": "single pre-specified EMCS representative",
        "trial_seeds": [int(seed) for seed in args.trial_seeds],
        "n_trials": int(len(args.trial_seeds)),
        "occupancies": [float(occupancy) for occupancy in args.occupancies],
        "stim_onsets_ms_by_trial_seed": {
            str(seed): float(onset) for seed, onset in stim_onsets.items()
        },
        "transient_ms": float(args.transient_ms),
        "t_analysis_ms": float(args.t_analysis_ms),
        "trial_sim_ms": float(args.trial_sim_ms),
        "integration_dt_ms": 0.1,
        "rate_monitor_period_ms": float(pilot.RATE_MONITOR_PERIOD_MS_OLD),
        "conduction_speed": 4.0,
        "coupling_strength": 0.25,
        "stim_amplitude": float(args.stim_amplitude),
        "stim_duration_ms": float(args.stim_duration_ms),
        "stim_variables": [0],
        "stim_onset_alignment": (
            "per-trial nearest temporal-average sample, then [-window,+window) "
            "epoching to a common midpoint"
        ),
        "stim_region_indices_zero_based": [
            int(index) for index in args.stim_region
        ],
        "stim_region_labels": [str(label) for label in args.stim_region_label],
        "stim_target_provenance": (
            "Resolved by explicit AAL label. The default is Supp_Motor_Area_L."
        ),
        "atlas_ordering": str(atlas.ordering),
        "atlas_source": str(atlas.source),
        "atlas_labels_sha256": pilot._sha256_array(
            np.asarray(atlas.labels, dtype="U128")
        ),
        "receptor_map_alignment": "AAL region-label join",
        "receptor_tracer": str(args.receptor_tracer),
        "receptor_csv": str(args.receptor_csv),
        "receptor_csv_sha256": str(args.receptor_csv_sha256),
        "receptor_map_sha256": pilot._sha256_array(
            np.asarray(receptor_map, dtype=np.float64)
        ),
        "pci_binarise_method": str(args.pci_binarise_method),
        "pci_estimator": (
            "single canonical PCI from the time-locked trial-averaged response"
            if str(args.pci_binarise_method) == "casali"
            else "legacy mean of per-trial TVBSim PCI values"
        ),
        "pci_bootstrap_replicates": int(args.pci_bootstrap_replicates),
        "pci_alpha": float(args.pci_alpha),
        "pci_bootstrap_seed": int(args.pci_bootstrap_seed),
        "model_family": "adex_zerlaut",
        "model_form": "split_gK_gNa_all_occupancies",
        "split_gK_gNa_at_occupancy_zero": True,
        "simulate_baseline": True,
        "e_l_e_drug": float(args.e_l_e_drug),
        "e_l_i_drug": float(args.e_l_i_drug),
        "sensitivity_design": "one-at-a-time",
        "variation_fraction": float(args.variation_fraction),
        "nominal_model_values": deepcopy(nominal_values),
        "tau_e_backend_note": (
            "tau_e_e and tau_e_i are varied jointly; the split-gK/gNa model's "
            "shared tau_e backend alias is set to the same value."
        ),
        "whole_brain_config_override_path": (
            'WholeBrainConfig.parameter_overrides["parameter_model"]'
        ),
        "configurations": deepcopy(configs),
        "n_configurations": int(len(configs)),
        "n_expected_trial_files": int(
            len(configs) * len(args.occupancies) * len(args.trial_seeds)
        ),
    }
    protocol_fingerprint = _fingerprint(protocol)
    return {
        **protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "output_root": str(args.output_root),
    }


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
                "Existing output has a different protocol fingerprint. "
                f"Choose another --output-root or pass --overwrite: {path}"
            )
        return
    _atomic_write_json(path, manifest)


def _validate_existing_trial(
    path: Path,
    *,
    protocol_fingerprint: str,
    config: dict[str, Any],
    occupancy: float,
    trial_seed: int,
    cohort: str,
    condition: str,
    subject_id: str,
    scenario: str,
    expected_stim_onset_ms: float,
    stim_region_labels: list[str],
    atlas_labels_sha256: str,
    receptor_tracer: str,
    receptor_csv_sha256: str,
    receptor_map_sha256: str,
    expected_t_analysis_ms: float,
) -> None:
    pilot._validate_existing_trial(
        path,
        protocol_fingerprint=protocol_fingerprint,
        trial_seed=trial_seed,
        occupancy=occupancy,
        stim_region_labels=stim_region_labels,
        receptor_map_sha256=receptor_map_sha256,
        cohort=cohort,
        condition=condition,
        subject_id=subject_id,
        scenario=scenario,
        expected_stim_onset_ms=expected_stim_onset_ms,
        atlas_labels_sha256=atlas_labels_sha256,
        receptor_tracer=receptor_tracer,
        receptor_csv_sha256=receptor_csv_sha256,
        expected_t_analysis_ms=expected_t_analysis_ms,
        expected_protocol_version=PROTOCOL_VERSION,
    )
    try:
        with np.load(path, allow_pickle=False) as data:
            checks = {
                "protocol_fingerprint": str(_npz_scalar(data, "protocol_fingerprint")),
                "robustness_config_id": str(_npz_scalar(data, "robustness_config_id")),
                "robustness_family": str(_npz_scalar(data, "robustness_family")),
                "robustness_direction": str(
                    _npz_scalar(data, "robustness_direction")
                ),
                "robustness_factor": float(
                    _npz_scalar(data, "robustness_factor")
                ),
                "occupancy": float(_npz_scalar(data, "occupancy")),
                "trial_seed": int(_npz_scalar(data, "trial_seed")),
                "parameter_model_overrides_json": str(
                    _npz_scalar(data, "parameter_model_overrides_json")
                ),
                "parameter_model_override_keys": [
                    str(value)
                    for value in np.asarray(
                        data["parameter_model_override_keys"]
                    ).reshape(-1)
                ],
                "parameter_model_override_values": np.asarray(
                    data["parameter_model_override_values"],
                    dtype=float,
                ).reshape(-1),
                "whole_brain_parameter_model_verified": bool(
                    _npz_scalar(data, "whole_brain_parameter_model_verified")
                ),
                "zerlaut_gk_gna": bool(
                    _npz_scalar(data, "zerlaut_gk_gna")
                ),
            }
    except Exception as exc:
        raise RuntimeError(f"Existing trial is unreadable or lacks provenance: {path}") from exc

    expected_json = _canonical_json(config["parameter_model_overrides"])
    expected_override_keys = sorted(config["parameter_model_overrides"])
    expected_override_values = np.asarray(
        [
            config["parameter_model_overrides"][key]
            for key in expected_override_keys
        ],
        dtype=float,
    )
    mismatches: list[str] = []
    if checks["protocol_fingerprint"] != protocol_fingerprint:
        mismatches.append("protocol_fingerprint")
    if checks["robustness_config_id"] != config["config_id"]:
        mismatches.append("robustness_config_id")
    if checks["robustness_family"] != config["family"]:
        mismatches.append("robustness_family")
    if checks["robustness_direction"] != config["direction"]:
        mismatches.append("robustness_direction")
    if not np.isclose(
        checks["robustness_factor"],
        float(config["factor"]),
        rtol=0.0,
        atol=1e-12,
    ):
        mismatches.append("robustness_factor")
    if not np.isclose(checks["occupancy"], float(occupancy), rtol=0.0, atol=1e-12):
        mismatches.append("occupancy")
    if checks["trial_seed"] != int(trial_seed):
        mismatches.append("trial_seed")
    if checks["parameter_model_overrides_json"] != expected_json:
        mismatches.append("parameter_model_overrides_json")
    if checks["parameter_model_override_keys"] != expected_override_keys:
        mismatches.append("parameter_model_override_keys")
    if (
        checks["parameter_model_override_values"].shape
        != expected_override_values.shape
        or not np.allclose(
            checks["parameter_model_override_values"],
            expected_override_values,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        mismatches.append("parameter_model_override_values")
    if not checks["whole_brain_parameter_model_verified"]:
        mismatches.append("whole_brain_parameter_model_verified")
    if not checks["zerlaut_gk_gna"]:
        mismatches.append("zerlaut_gk_gna")
    if mismatches:
        raise RuntimeError(
            f"Existing trial provenance mismatch ({', '.join(mismatches)}): {path}. "
            "Use another output root or pass --overwrite."
        )


def _build_trial_jobs(
    args: argparse.Namespace,
    subject: Any,
    scenario_cfg: dict[str, Any],
    receptor_map: np.ndarray,
    receptor_map_sha256: str,
    stim_onsets: dict[int, float],
    configs: list[dict[str, Any]],
    protocol_fingerprint: str,
) -> tuple[list[dict[str, Any]], int]:
    jobs: list[dict[str, Any]] = []
    reused = 0
    for config in configs:
        for occupancy in args.occupancies:
            for trial_seed in args.trial_seeds:
                path = _trial_path(
                    args.output_root,
                    config["config_id"],
                    occupancy,
                    args.scenario,
                    subject.subject_id,
                    trial_seed,
                )
                if path.exists() and not args.overwrite:
                    _validate_existing_trial(
                        path,
                        protocol_fingerprint=protocol_fingerprint,
                        config=config,
                        occupancy=occupancy,
                        trial_seed=trial_seed,
                        cohort=subject.cohort,
                        condition=subject.condition,
                        subject_id=subject.subject_id,
                        scenario=args.scenario,
                        expected_stim_onset_ms=stim_onsets[int(trial_seed)],
                        stim_region_labels=args.stim_region_label,
                        atlas_labels_sha256=args.atlas_labels_sha256,
                        receptor_tracer=args.receptor_tracer,
                        receptor_csv_sha256=args.receptor_csv_sha256,
                        receptor_map_sha256=receptor_map_sha256,
                        expected_t_analysis_ms=args.t_analysis_ms,
                    )
                    reused += 1
                    continue
                jobs.append(
                    {
                        "scenario_key": args.scenario,
                        "scenario_cfg": scenario_cfg,
                        "cohort": subject.cohort,
                        "condition": subject.condition,
                        "subject_id": subject.subject_id,
                        "trial_seed": int(trial_seed),
                        "occupancy": float(occupancy),
                        "receptor_map": receptor_map,
                        "receptor_map_sha256": receptor_map_sha256,
                        "save_path": path,
                        "stim_onset_ms": float(stim_onsets[int(trial_seed)]),
                        "config": config,
                        "protocol_fingerprint": protocol_fingerprint,
                        "args": args,
                    }
                )
    return jobs, reused


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
    receptor_map_sha256: str,
    save_path: Path,
    stim_onset_ms: float,
    config: dict[str, Any],
    protocol_fingerprint: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = perf_counter()
    connectivity, lengths, atlas, metadata = pilot.load_subject_structural(
        subject_id=subject_id,
        cohort=cohort,
        dataset_root=args.dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    atlas_labels = np.asarray(atlas.labels, dtype="U128")
    if pilot._sha256_array(atlas_labels) != str(args.atlas_labels_sha256):
        raise RuntimeError(
            "Subject atlas labels differ from the atlas used to align the "
            "stimulation target and receptor map."
        )
    connectivity, lengths, sc_zero_fraction = pilot._apply_damage_parity(
        connectivity,
        lengths,
        cohort,
    )

    parameter_model = pilot._build_parameter_model(
        condition,
        occupancy,
        receptor_map,
        args,
    )
    model_overrides = {
        key: float(value)
        for key, value in config["parameter_model_overrides"].items()
    }
    # This is the critical sensitivity injection: the complete explicit scalar
    # configuration is merged into the exact dict passed to WholeBrainConfig.
    parameter_model.update(model_overrides)
    parameter_model.update(
        {
            "noise_alpha": float(scenario_cfg["noise_alpha"]),
            "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
        }
    )
    for key, expected in model_overrides.items():
        if not np.isclose(float(parameter_model[key]), expected, rtol=0.0, atol=1e-15):
            raise AssertionError(f"Model override did not reach parameter_model: {key}")

    parameter_stimulus = {
        "stimtime": float(stim_onset_ms),
        "stimdur": float(args.stim_duration_ms),
        "stimperiod": float(args.trial_sim_ms) * 10.0,
        "stimval": float(args.stim_amplitude),
        "stimregion": [int(index) for index in args.stim_region],
        "stimvariables": [0],
    }
    wb_config = WholeBrainConfig(
        simulation_length_ms=float(args.trial_sim_ms),
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=0.25,
        model_family="adex_zerlaut",
        zerlaut_matteo=False,
        zerlaut_gk_gna=True,
        zerlaut_order=2,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(pilot.RATE_MONITOR_PERIOD_MS_OLD),
        monitor_variables=(0, 1),
        weights=np.asarray(connectivity, dtype=float),
        tract_lengths=np.asarray(lengths, dtype=float),
        parameter_overrides={
            "parameter_model": parameter_model,
            "parameter_stimulus": parameter_stimulus,
        },
    )
    config_parameter_model = wb_config.parameter_overrides["parameter_model"]
    for key, expected in model_overrides.items():
        if not np.isclose(
            float(config_parameter_model[key]),
            expected,
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError(
                f"Model override did not reach WholeBrainConfig.parameter_model: {key}"
            )

    simulation = pilot.run_whole_brain_simulation(wb_config, seed=int(trial_seed))
    time_ms = np.asarray(simulation.time_ms, dtype=float)
    rate = np.asarray(simulation.raw, dtype=float)
    keep = time_ms >= float(args.transient_ms)
    time_post = time_ms[keep]
    rate_post = rate[keep]
    if time_post.ndim != 1 or time_post.size < 2:
        raise RuntimeError("Simulation produced too few post-transient time samples.")
    if rate_post.ndim != 2 or rate_post.shape != (time_post.size, 90):
        raise RuntimeError(
            "Simulation rate output must have shape (post-transient time, 90); "
            f"got {rate_post.shape}."
        )
    if not np.isfinite(time_post).all() or not np.isfinite(rate_post).all():
        raise RuntimeError("Simulation produced non-finite time or rate values.")
    monitor_dt_ms = float(np.median(np.diff(time_post)))
    onset_sample_index = int(
        np.argmin(np.abs(time_post - float(stim_onset_ms)))
    )
    onset_sample_ms = float(time_post[onset_sample_index])
    onset_residual_ms = float(onset_sample_ms - float(stim_onset_ms))
    if abs(onset_residual_ms) > 0.5 * monitor_dt_ms + 1e-6:
        raise RuntimeError(
            "Saved stimulation onset is farther than half a monitor sample "
            "from the post-transient time grid."
        )
    analysis_bins = int(round(float(args.t_analysis_ms) / monitor_dt_ms))
    if (
        onset_sample_index - analysis_bins < 0
        or onset_sample_index + analysis_bins > time_post.size
    ):
        raise RuntimeError(
            "Simulation does not contain the complete requested peri-stimulus window."
        )

    override_keys = sorted(model_overrides)
    _atomic_savez(
        save_path,
        time_ms=time_post,
        rate=rate_post,
        region_labels=atlas_labels,
        simulation_region_labels=np.asarray(simulation.region_labels),
        atlas_ordering=np.asarray([str(atlas.ordering)], dtype="U128"),
        atlas_labels_sha256=np.asarray(
            [
                pilot._sha256_array(
                    atlas_labels
                )
            ],
            dtype="U128",
        ),
        protocol_version=np.asarray([PROTOCOL_VERSION], dtype="U128"),
        protocol_fingerprint=np.asarray([protocol_fingerprint], dtype="U128"),
        cohort=np.asarray([str(cohort)], dtype="U32"),
        condition=np.asarray([str(condition)], dtype="U32"),
        subject_id=np.asarray([str(subject_id)], dtype="U128"),
        scenario=np.asarray([str(scenario_key)], dtype="U128"),
        robustness_config_id=np.asarray([config["config_id"]], dtype="U128"),
        robustness_family=np.asarray([config["family"]], dtype="U128"),
        robustness_direction=np.asarray([config["direction"]], dtype="U32"),
        robustness_factor=np.asarray([float(config["factor"])]),
        parameter_model_override_keys=np.asarray(override_keys, dtype="U128"),
        parameter_model_override_values=np.asarray(
            [model_overrides[key] for key in override_keys],
            dtype=float,
        ),
        parameter_model_overrides_json=np.asarray(
            [_canonical_json(model_overrides)],
            dtype="U4096",
        ),
        whole_brain_parameter_model_verified=np.asarray([True]),
        zerlaut_gk_gna=np.asarray([True]),
        receptor_map_alignment=np.asarray(["AAL region-label join"], dtype="U128"),
        receptor_tracer=np.asarray([str(args.receptor_tracer)], dtype="U32"),
        receptor_csv_sha256=np.asarray(
            [str(args.receptor_csv_sha256)],
            dtype="U128",
        ),
        receptor_map_sha256=np.asarray([receptor_map_sha256], dtype="U128"),
        receptor_map=np.asarray(receptor_map, dtype=np.float64),
        stim_onset_ms=np.asarray([float(stim_onset_ms)]),
        stim_onset_sample_index=np.asarray([onset_sample_index], dtype=np.int64),
        stim_onset_sample_ms=np.asarray([onset_sample_ms], dtype=float),
        stim_onset_alignment_residual_ms=np.asarray(
            [onset_residual_ms],
            dtype=float,
        ),
        t_analysis_ms=np.asarray([float(args.t_analysis_ms)]),
        rate_monitor_period_ms=np.asarray(
            [float(pilot.RATE_MONITOR_PERIOD_MS_OLD)]
        ),
        trial_seed=np.asarray([int(trial_seed)]),
        noise_alpha=np.asarray([float(scenario_cfg["noise_alpha"])]),
        stim_amplitude=np.asarray([float(args.stim_amplitude)]),
        stim_duration_ms=np.asarray([float(args.stim_duration_ms)]),
        stim_region=np.asarray(args.stim_region, dtype=int),
        stim_region_labels=np.asarray(
            [str(atlas.labels[int(index)]) for index in args.stim_region],
            dtype="U128",
        ),
        occupancy=np.asarray([float(occupancy)]),
        sc_zero_fraction_upper=np.asarray([float(sc_zero_fraction)]),
    )
    return {
        "cohort": cohort,
        "condition": condition,
        "subject_id": subject_id,
        "stage": str(getattr(metadata, "stage", "") or ""),
        "sedation": str(getattr(metadata, "sedation", "") or ""),
        "scenario": scenario_key,
        "config_id": config["config_id"],
        "family": config["family"],
        "direction": config["direction"],
        "occupancy": float(occupancy),
        "trial_seed": int(trial_seed),
        "runtime_s": float(perf_counter() - started),
        "save_path": str(save_path),
    }


def _run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    return _run_trial(**job)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(np.unique(x)) < 2:
        return float("nan")
    return float(np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)[0])


def _value_sign(value: float, atol: float = 1e-12) -> int:
    if not np.isfinite(value) or abs(float(value)) <= atol:
        return 0
    return 1 if value > 0.0 else -1


def _configuration_rows(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for config in configs:
        overrides = config["parameter_model_overrides"]
        rows.append(
            {
                "config_index": int(config["config_index"]),
                "config_id": config["config_id"],
                "config_label": config["config_label"],
                "family": config["family"],
                "direction": config["direction"],
                "factor": float(config["factor"]),
                "varied_parameters": ";".join(config["varied_parameters"]),
                "b_e": float(overrides["b_e"]),
                "tau_w_e": float(overrides["tau_w_e"]),
                "tau_e_e": float(overrides["tau_e_e"]),
                "tau_e_i": float(overrides["tau_e_i"]),
                "tau_e_backend_alias": float(overrides["tau_e"]),
                "tau_i": float(overrides["tau_i"]),
                "external_input_ex_ex": float(overrides["external_input_ex_ex"]),
                "external_input_ex_in": float(overrides["external_input_ex_in"]),
                "weight_noise": float(overrides["weight_noise"]),
                "parameter_model_overrides_json": _canonical_json(overrides),
            }
        )
    return rows


def _aggregate(
    args: argparse.Namespace,
    subject: Any,
    configs: list[dict[str, Any]],
    protocol_fingerprint: str,
    receptor_map_sha256: str,
    stim_onsets: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occupancy_values = np.asarray(sorted(args.occupancies), dtype=float)
    positive_mask = occupancy_values > 0.0
    dose_rows: list[dict[str, Any]] = []
    trial_input_rows: list[dict[str, Any]] = []

    for config in configs:
        config_id = str(config["config_id"])
        for occupancy in occupancy_values:
            paths = [
                _trial_path(
                    args.output_root,
                    config_id,
                    float(occupancy),
                    args.scenario,
                    subject.subject_id,
                    seed,
                )
                for seed in args.trial_seeds
            ]
            missing = [path for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing {len(missing)} trial files for config={config_id}, "
                    f"occupancy={occupancy:g}; first missing: {missing[0]}"
                )
            for path, seed in zip(paths, args.trial_seeds):
                _validate_existing_trial(
                    path,
                    protocol_fingerprint=protocol_fingerprint,
                    config=config,
                    occupancy=float(occupancy),
                    trial_seed=seed,
                    cohort=subject.cohort,
                    condition=subject.condition,
                    subject_id=subject.subject_id,
                    scenario=args.scenario,
                    expected_stim_onset_ms=stim_onsets[int(seed)],
                    stim_region_labels=args.stim_region_label,
                    atlas_labels_sha256=args.atlas_labels_sha256,
                    receptor_tracer=args.receptor_tracer,
                    receptor_csv_sha256=args.receptor_csv_sha256,
                    receptor_map_sha256=receptor_map_sha256,
                    expected_t_analysis_ms=args.t_analysis_ms,
                )
                with np.load(path, allow_pickle=False) as data:
                    stim_onset_ms = float(_npz_scalar(data, "stim_onset_ms"))
                trial_input_rows.append(
                    {
                        "cohort": subject.cohort,
                        "condition": subject.condition,
                        "subject_id": subject.subject_id,
                        "scenario": args.scenario,
                        "config_index": int(config["config_index"]),
                        "config_id": config_id,
                        "occupancy": float(occupancy),
                        "trial_seed": int(seed),
                        "stim_onset_ms": stim_onset_ms,
                        "trial_path": str(path),
                    }
                )
            pci_mean, pci_values = pilot._compute_pci_for_condition(
                paths,
                binarise_method=str(args.pci_binarise_method),
                n_bootstrap=int(args.pci_bootstrap_replicates),
                alpha=float(args.pci_alpha),
                bootstrap_seed=int(args.pci_bootstrap_seed),
            )
            pci_values = np.asarray(pci_values, dtype=float)
            dose_rows.append(
                {
                    "cohort": subject.cohort,
                    "condition": subject.condition,
                    "subject_id": subject.subject_id,
                    "scenario": args.scenario,
                    "config_index": int(config["config_index"]),
                    "config_id": config_id,
                    "config_label": config["config_label"],
                    "family": config["family"],
                    "direction": config["direction"],
                    "factor": float(config["factor"]),
                    "occupancy": float(occupancy),
                    "n_trials": int(len(paths)),
                    "pci_binarise_method": str(args.pci_binarise_method),
                    "pci_estimator": (
                        "trial-averaged canonical PCI"
                        if str(args.pci_binarise_method) == "casali"
                        else "mean per-trial TVBSim PCI"
                    ),
                    "pci_mean": float(pci_mean),
                    "n_returned_pci_values": int(pci_values.size),
                    "pci_values": json.dumps(
                        [float(value) for value in pci_values]
                    ),
                    "trial_seeds": json.dumps([int(seed) for seed in args.trial_seeds]),
                    "trial_paths": json.dumps([str(path) for path in paths]),
                }
            )

    dose_metrics = pd.DataFrame(dose_rows).sort_values(
        ["config_index", "occupancy"]
    )
    baseline = dose_metrics.loc[
        np.isclose(dose_metrics["occupancy"], 0.0),
        ["config_id", "pci_mean"],
    ].rename(columns={"pci_mean": "pci_baseline"})
    dose_metrics = dose_metrics.merge(baseline, on="config_id", how="left")
    dose_metrics["delta_pci_from_baseline"] = (
        dose_metrics["pci_mean"] - dose_metrics["pci_baseline"]
    )

    config_rows: list[dict[str, Any]] = []
    max_occupancy = float(np.max(occupancy_values))
    for config in configs:
        config_id = str(config["config_id"])
        group = dose_metrics[dose_metrics["config_id"].eq(config_id)].sort_values(
            "occupancy"
        )
        pci_values = group["pci_mean"].to_numpy(dtype=float)
        delta_values = group["delta_pci_from_baseline"].to_numpy(dtype=float)
        adjacent = np.diff(pci_values)
        slope_all = _linear_slope(occupancy_values, pci_values)
        slope_positive = _linear_slope(
            occupancy_values[positive_mask],
            delta_values[positive_mask],
        )
        max_delta = float(
            group.loc[
                np.isclose(group["occupancy"], max_occupancy),
                "delta_pci_from_baseline",
            ].iloc[0]
        )

        config_rows.append(
            {
                "cohort": subject.cohort,
                "condition": subject.condition,
                "subject_id": subject.subject_id,
                "scenario": args.scenario,
                "config_index": int(config["config_index"]),
                "config_id": config_id,
                "config_label": config["config_label"],
                "family": config["family"],
                "direction": config["direction"],
                "factor": float(config["factor"]),
                "n_trials": int(len(args.trial_seeds)),
                "n_doses": int(len(occupancy_values)),
                "slope_pci_per_occupancy": float(slope_all),
                "slope_positive_dose_delta_pci_per_occupancy": float(
                    slope_positive
                ),
                "max_dose_occupancy": max_occupancy,
                "max_dose_delta_pci": max_delta,
                "n_positive_adjacent_steps": int(np.sum(adjacent > 0.0)),
                "fraction_non_decreasing_adjacent_steps": float(
                    np.mean(adjacent >= 0.0)
                ),
                "monotonic_non_decreasing": bool(np.all(adjacent >= 0.0)),
                "max_delta_sign": _value_sign(max_delta),
                "slope_sign": _value_sign(slope_all),
            }
        )

    trial_inputs = pd.DataFrame(trial_input_rows).sort_values(
        ["config_index", "occupancy", "trial_seed"]
    )
    config_summary = pd.DataFrame(config_rows).sort_values("config_index")
    nominal = config_summary.loc[config_summary["config_id"].eq("nominal")].iloc[0]
    nominal_max_sign = int(nominal["max_delta_sign"])
    nominal_slope_sign = int(nominal["slope_sign"])
    config_summary["same_max_delta_sign_as_nominal"] = (
        config_summary["max_delta_sign"].astype(int) == nominal_max_sign
    )
    config_summary["same_slope_sign_as_nominal"] = (
        config_summary["slope_sign"].astype(int) == nominal_slope_sign
    )
    config_summary["result_sign_stable"] = (
        config_summary["same_max_delta_sign_as_nominal"]
        & config_summary["same_slope_sign_as_nominal"]
    )
    config_summary["max_dose_delta_difference_from_nominal"] = (
        config_summary["max_dose_delta_pci"]
        - float(nominal["max_dose_delta_pci"])
    )
    config_summary["slope_difference_from_nominal"] = (
        config_summary["slope_pci_per_occupancy"]
        - float(nominal["slope_pci_per_occupancy"])
    )

    stability_summary = pd.DataFrame(
        [
            {
                "cohort": subject.cohort,
                "condition": subject.condition,
                "subject_id": subject.subject_id,
                "pci_binarise_method": str(args.pci_binarise_method),
                "n_matched_trial_seeds_per_condition": int(len(args.trial_seeds)),
                "n_configurations": int(len(config_summary)),
                "nominal_max_dose_delta_pci": float(nominal["max_dose_delta_pci"]),
                "nominal_slope_pci_per_occupancy": float(
                    nominal["slope_pci_per_occupancy"]
                ),
                "n_positive_max_dose_deltas": int(
                    np.sum(config_summary["max_dose_delta_pci"] > 0.0)
                ),
                "fraction_positive_max_dose_deltas": float(
                    np.mean(config_summary["max_dose_delta_pci"] > 0.0)
                ),
                "n_positive_slopes": int(
                    np.sum(config_summary["slope_pci_per_occupancy"] > 0.0)
                ),
                "fraction_positive_slopes": float(
                    np.mean(config_summary["slope_pci_per_occupancy"] > 0.0)
                ),
                "n_monotonic_configurations": int(
                    np.sum(config_summary["monotonic_non_decreasing"])
                ),
                "fraction_monotonic_configurations": float(
                    np.mean(config_summary["monotonic_non_decreasing"])
                ),
                "n_same_result_sign_as_nominal": int(
                    np.sum(config_summary["result_sign_stable"])
                ),
                "fraction_same_result_sign_as_nominal": float(
                    np.mean(config_summary["result_sign_stable"])
                ),
                "all_configurations_preserve_result_sign": bool(
                    np.all(config_summary["result_sign_stable"])
                ),
                "minimum_max_dose_delta_pci": float(
                    config_summary["max_dose_delta_pci"].min()
                ),
                "maximum_max_dose_delta_pci": float(
                    config_summary["max_dose_delta_pci"].max()
                ),
                "minimum_slope_pci_per_occupancy": float(
                    config_summary["slope_pci_per_occupancy"].min()
                ),
                "maximum_slope_pci_per_occupancy": float(
                    config_summary["slope_pci_per_occupancy"].max()
                ),
            }
        ]
    )

    tables_dir = args.output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_configuration_rows(configs)).to_csv(
        tables_dir / "serotonergic_pci_emcs_robustness_configurations.csv",
        index=False,
    )
    dose_metrics.to_csv(
        tables_dir / "serotonergic_pci_emcs_robustness_dose_metrics.csv",
        index=False,
    )
    trial_inputs.to_csv(
        tables_dir / "serotonergic_pci_emcs_robustness_matched_trial_inputs.csv",
        index=False,
    )
    config_summary.to_csv(
        tables_dir / "serotonergic_pci_emcs_robustness_config_summary.csv",
        index=False,
    )
    stability_summary.to_csv(
        tables_dir / "serotonergic_pci_emcs_robustness_stability_summary.csv",
        index=False,
    )
    _plot_figure4(dose_metrics, config_summary, stability_summary, args.output_root)
    return dose_metrics, trial_inputs, config_summary, stability_summary


def _plot_figure4(
    dose_metrics: pd.DataFrame,
    config_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    output_root: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.0, 8.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.35))
    axis_dose = figure.add_subplot(grid[0, :])
    axis_delta = figure.add_subplot(grid[1, 0])
    axis_slope = figure.add_subplot(grid[1, 1])

    ordered = config_summary.sort_values("config_index")
    for _, summary_row in ordered.iterrows():
        config_id = str(summary_row["config_id"])
        group = dose_metrics[dose_metrics["config_id"].eq(config_id)].sort_values(
            "occupancy"
        )
        family = str(summary_row["family"])
        direction = str(summary_row["direction"])
        nominal = config_id == "nominal"
        axis_dose.plot(
            group["occupancy"],
            group["delta_pci_from_baseline"],
            color=FAMILY_COLORS.get(family, "#777777"),
            linestyle="-" if nominal or direction == "plus" else "--",
            linewidth=2.8 if nominal else 1.35,
            marker="o",
            markersize=5.0 if nominal else 3.4,
            alpha=1.0 if nominal else 0.78,
            label=str(summary_row["config_label"]),
            zorder=5 if nominal else 2,
        )
    axis_dose.axhline(0.0, color="#777777", linewidth=0.8)
    axis_dose.set_xlabel(r"5-HT$_{2A}$ occupancy")
    axis_dose.set_ylabel(r"$\Delta$PCI from matched baseline")
    axis_dose.set_title("A   EMCS PCI dose-response under AdEx parameter perturbations", loc="left")
    axis_dose.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis_dose.legend(
        frameon=False,
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.23),
        columnspacing=1.2,
        handlelength=2.5,
    )

    forest = ordered.sort_values("config_index", ascending=False).reset_index(drop=True)
    positions = np.arange(len(forest), dtype=float)
    labels = [str(label) for label in forest["config_label"]]
    nominal_delta = float(
        ordered.loc[ordered["config_id"].eq("nominal"), "max_dose_delta_pci"].iloc[0]
    )
    nominal_slope = float(
        ordered.loc[
            ordered["config_id"].eq("nominal"), "slope_pci_per_occupancy"
        ].iloc[0]
    )

    for position, (_, row) in zip(positions, forest.iterrows()):
        color = FAMILY_COLORS.get(str(row["family"]), "#777777")
        delta_center = float(row["max_dose_delta_pci"])
        slope_center = float(row["slope_pci_per_occupancy"])
        axis_delta.plot(
            delta_center,
            position,
            marker="o",
            linestyle="none",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6.4 if row["config_id"] == "nominal" else 5.2,
        )
        axis_slope.plot(
            slope_center,
            position,
            marker="o",
            linestyle="none",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6.4 if row["config_id"] == "nominal" else 5.2,
        )

    axis_delta.axvline(0.0, color="#555555", linewidth=0.9)
    axis_delta.axvline(
        nominal_delta,
        color=FAMILY_COLORS["nominal"],
        linewidth=0.8,
        linestyle=":",
    )
    axis_delta.set_yticks(positions, labels)
    axis_delta.set_xlabel(r"Maximum-dose $\Delta$PCI")
    axis_delta.set_title("B   Maximum-dose effect", loc="left")
    axis_delta.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)

    axis_slope.axvline(0.0, color="#555555", linewidth=0.9)
    axis_slope.axvline(
        nominal_slope,
        color=FAMILY_COLORS["nominal"],
        linewidth=0.8,
        linestyle=":",
    )
    axis_slope.set_yticks(positions, [])
    axis_slope.set_xlabel(r"PCI slope per occupancy")
    axis_slope.set_title("C   Linear dose-response slope", loc="left")
    axis_slope.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)

    stability = stability_summary.iloc[0]
    annotation = (
        f"{int(stability['n_positive_max_dose_deltas'])}/"
        f"{int(stability['n_configurations'])} configurations retain a positive "
        r"maximum-dose $\Delta$PCI"
        "\n"
        f"{int(stability['n_positive_slopes'])}/"
        f"{int(stability['n_configurations'])} retain a positive dose-response slope; "
        f"{int(stability['n_monotonic_configurations'])}/"
        f"{int(stability['n_configurations'])} are monotonic"
        "\n"
        f"Canonical condition-level PCI uses the same "
        f"{int(stability['n_matched_trial_seeds_per_condition'])} time-locked seeds "
        "for every dose and parameter configuration"
    )
    figure.suptitle(
        "Robustness of simulated serotonergic PCI rescue in a representative EMCS connectome",
        fontsize=13,
        y=1.025,
    )
    figure.text(
        0.5,
        -0.025,
        annotation,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#222222",
    )
    for axis in (axis_dose, axis_delta, axis_slope):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = figure_dir / "serotonergic_pci_emcs_robustness_figure4"
    figure.savefig(prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    _validate_inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_root / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_root / "figures").mkdir(parents=True, exist_ok=True)

    subject = _select_subject(args.dataset_root, args.subject_id)
    atlas = pilot._resolve_stim_regions(args)
    args.atlas_labels_sha256 = pilot._sha256_array(
        np.asarray(atlas.labels, dtype="U128")
    )
    receptor_map = pilot.get_5ht2a_aal90(
        tracer=str(args.receptor_tracer),
        csv_path=args.receptor_csv,
        target_labels=atlas.labels,
    )
    args.receptor_csv_sha256 = pilot._sha256_file(args.receptor_csv)
    receptor_map_sha256 = pilot._sha256_array(
        np.asarray(receptor_map, dtype=np.float64)
    )
    scenario_cfg = pilot.SCENARIOS[args.scenario]
    stim_onsets = pilot._stim_onsets(
        args.trial_seeds,
        transient_ms=float(args.transient_ms),
        t_analysis_ms=float(args.t_analysis_ms),
        trial_sim_ms=float(args.trial_sim_ms),
        seed=int(args.stim_onset_seed),
    )
    nominal_values = _nominal_model_values()
    all_configs = build_robustness_configs(
        nominal_values,
        variation_fraction=float(args.variation_fraction),
    )
    configs = all_configs
    if args.max_configs is not None:
        configs = all_configs[: min(int(args.max_configs), len(all_configs))]
    manifest = _build_protocol_manifest(
        args,
        subject,
        scenario_cfg,
        atlas,
        receptor_map,
        stim_onsets,
        configs,
        nominal_values,
    )

    if args.dry_run:
        dry_path = args.output_root / "logs" / "dry_run_manifest.json"
        _atomic_write_json(dry_path, manifest)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "subject": subject.subject_id,
                    "n_configurations": len(configs),
                    "n_occupancies": len(args.occupancies),
                    "n_trial_seeds": len(args.trial_seeds),
                    "n_expected_trial_files": manifest["n_expected_trial_files"],
                    "stim_region_labels": args.stim_region_label,
                    "protocol_fingerprint": manifest["protocol_fingerprint"],
                    "manifest": str(dry_path),
                },
                indent=2,
            )
        )
        return

    manifest_path = args.output_root / "logs" / "run_manifest.json"
    _write_or_validate_manifest(
        manifest_path,
        manifest,
        overwrite=bool(args.overwrite),
    )
    _atomic_write_csv(
        args.output_root
        / "tables"
        / "serotonergic_pci_emcs_robustness_configurations.csv",
        _configuration_rows(configs),
    )

    completed_rows: list[dict[str, Any]] = []
    reused = 0
    if not args.aggregate_only:
        jobs, reused = _build_trial_jobs(
            args,
            subject,
            scenario_cfg,
            receptor_map,
            receptor_map_sha256,
            stim_onsets,
            configs,
            manifest["protocol_fingerprint"],
        )
        print(
            "[sero-pci-emcs-robustness] "
            f"expected={manifest['n_expected_trial_files']} reused={reused} "
            f"queued={len(jobs)} workers={int(args.workers)}",
            flush=True,
        )
        if jobs:
            with ProcessPoolExecutor(
                max_workers=int(args.workers),
                initializer=worker_initializer,
            ) as executor:
                futures = [executor.submit(_run_trial_job, job) for job in jobs]
                total = len(futures)
                for completed, future in enumerate(
                    as_completed(futures),
                    start=1,
                ):
                    row = future.result()
                    completed_rows.append(row)
                    if (
                        completed == 1
                        or completed == total
                        or completed % int(args.progress_every) == 0
                    ):
                        print(
                            "[sero-pci-emcs-robustness] "
                            f"{completed}/{total} config={row['config_id']} "
                            f"occ={row['occupancy']:.3f} seed={row['trial_seed']} "
                            f"runtime={row['runtime_s']:.1f}s",
                            flush=True,
                        )

    stamp = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime(
        "%Y%m%dT%H%M%S"
    )
    _atomic_write_csv(
        args.output_root / "logs" / f"completed_trials_{stamp}.csv",
        completed_rows,
    )
    print(
        "[sero-pci-emcs-robustness] "
        f"completed_now={len(completed_rows)} reused={reused}",
        flush=True,
    )

    if args.skip_aggregate:
        print(
            "[sero-pci-emcs-robustness] simulations complete; aggregation skipped",
            flush=True,
        )
        return
    dose, trial_inputs, summary, stability = _aggregate(
        args,
        subject,
        configs,
        manifest["protocol_fingerprint"],
        receptor_map_sha256,
        stim_onsets,
    )
    print(
        "[sero-pci-emcs-robustness] "
        f"wrote {len(dose)} dose rows, {len(trial_inputs)} matched-trial input rows, "
        f"{len(summary)} configuration summaries; "
        f"all_sign_stable={bool(stability.iloc[0]['all_configurations_preserve_result_sign'])}",
        flush=True,
    )
    print(f"[sero-pci-emcs-robustness] output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
