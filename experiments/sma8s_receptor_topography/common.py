"""Frozen contract and spatially constrained receptor-map controls.

The cortical spin is a *parcel-centroid approximation* to a surface spin:
hemispheres are rotated on separate unit spheres, and Hungarian reassignment
gives a one-to-one cortical value permutation. Non-cortical AAL90 parcels
cannot be rotated on a cortical sphere; they are permuted within hemisphere.
This preserves the entire 90-value distribution but is not a vertex-wise spin
test. Interpret map-null p-values with this limitation explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.sma8s_full100.common import (
    COHORTS as ORIGINAL_COHORTS, EXPECTED_DIVISOR, MASTER_SEED, TRIALS,
    atomic_json, sha256_file, trial_onset_ms, trial_seed, validate_dataset,
)
from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90

ROOT = Path(__file__).resolve().parents[2]
COHORTS = {key: ORIGINAL_COHORTS[key] for key in ("mcs", "uws")}
HIGHEST_OCCUPANCY = 0.766
MAP_SEED = 20261126
VERSION = "sma8s_el62_b35_u40_receptor_topography_v1"
NONCORTICAL = frozenset((
    "Hippocampus", "ParaHippocampal", "Amygdala", "Caudate", "Putamen",
    "Pallidum", "Thalamus",
))
PROTOCOL = {
    "version": VERSION, "model_order": 1, "split_leak": True,
    "G": 0.3, "external_drive_hz": 0.315, "weight_noise": 0.0002,
    "regional_OU_tau_ms": 2.0, "tau_w_ms": 500.0,
    "E_L_e_mv": -62.0, "drug_E_L_e_mv": -60.2,
    "E_L_i_mv": -65.0, "drug_E_L_i_mv": -64.4,
    "source_label": "Supp_Motor_Area_L", "pulse_hz_per_ms": 0.55,
    "pulse_duration_ms": 10.0, "simulation_ms": 8000.0,
    "transient_discard_ms": 2000.0,
    "onset_range_ms_inclusive": [2500, 7500],
    "monitor": "TVB TemporalAverage", "monitor_period_ms": 1.0,
    "saved_peri_onset_ms": [-500, 500], "prestim_ms": [-300, 0],
    "poststim_ms": [0, 300], "SC_normalization_scheme": "shared_control_maximum_edge",
    "SC_divisor": EXPECTED_DIVISOR, "trials_per_case": TRIALS,
    "b_pa_by_cohort": {name: value[1] for name, value in COHORTS.items()},
    "highest_occupancy": HIGHEST_OCCUPANCY,
    "baseline_reuse": "one independently rerun zero-occupancy case per subject; identical seeds/onsets",
    "receptor_rule": "min-max Cimbi AAL90; gNa fixed; gK interpolated toward drug E_L",
    "spin_rule": "AAL90 cortical-centroid spherical rotation/Hungarian assignment per hemisphere; noncortical within-hemisphere permutation",
}


def _centres(labels: np.ndarray) -> np.ndarray:
    path = ROOT / "data/connectivity/average_aal90/centres.txt"
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    by_label = {row[0]: np.asarray([float(value) for value in row[1:4]]) for row in rows}
    if len(by_label) != 90 or set(by_label) != set(labels):
        raise ValueError("AAL90 centre labels do not match the structural atlas")
    return np.stack([by_label[label] for label in labels])


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _spin_index(labels: np.ndarray, centres: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """Return a bijective target->source parcel index for one spin draw."""
    labels = np.asarray(labels, dtype=str)
    index = np.arange(len(labels))
    result = index.copy()
    rotation = _random_rotation(rng)
    reflection = np.diag([-1.0, 1.0, 1.0])
    for hemi in ("L", "R"):
        ids = np.asarray([i for i, label in enumerate(labels)
                          if label.endswith("_" + hemi)
                          and label.rsplit("_", 1)[0] not in NONCORTICAL])
        sub = np.asarray([i for i, label in enumerate(labels)
                          if label.endswith("_" + hemi)
                          and label.rsplit("_", 1)[0] in NONCORTICAL])
        if len(ids) != 38 or len(sub) != 7:
            raise ValueError(f"Unexpected cortical/noncortical AAL90 split: {hemi}")
        xyz = centres[ids].copy()
        if hemi == "R":
            xyz = xyz @ reflection
        xyz -= xyz.mean(axis=0)
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        rotated = xyz @ rotation
        source, target = linear_sum_assignment(1.0 - rotated @ xyz.T)
        result[ids[target]] = ids[source]
        result[sub] = rng.permutation(sub)
    if sorted(result.tolist()) != list(range(90)):
        raise ValueError("Spin reassignment was not bijective")
    return result


def _cortical_moran_knn5(labels: np.ndarray, centres: np.ndarray,
                         values: np.ndarray) -> float:
    """Diagnostic Moran I on a fixed within-hemisphere 5-neighbour graph."""
    cortex = np.asarray([i for i, label in enumerate(labels)
                         if label.rsplit("_", 1)[0] not in NONCORTICAL])
    positions = centres[cortex]
    hemis = np.asarray([labels[i].rsplit("_", 1)[1] for i in cortex])
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    distances[hemis[:, None] != hemis[None, :]] = np.inf
    np.fill_diagonal(distances, np.inf)
    weights = np.zeros_like(distances)
    nearest = np.argsort(distances, axis=1)[:, :5]
    for i, neighbours in enumerate(nearest):
        weights[i, neighbours] = 1.0 / distances[i, neighbours]
    weights = (weights + weights.T) / 2.0
    z = values[cortex] - np.mean(values[cortex])
    denominator = float(z @ z)
    if denominator == 0:
        return float("nan")
    return float(len(cortex) / weights.sum() * (z @ weights @ z) / denominator)


def build_maps(labels: np.ndarray, n_spin: int, n_permutation: int) -> tuple[list[str], np.ndarray, list[dict]]:
    if n_spin < 0 or n_permutation < 1:
        raise ValueError("Spins must be nonnegative and at least one permutation is required")
    labels = np.asarray(labels, dtype=str)
    if labels.shape != (90,) or len(set(labels)) != 90:
        raise ValueError("Expected 90 unique AAL90 labels")
    receptor = get_5ht2a_aal90(target_labels=labels)
    rho = (receptor - receptor.min()) / (receptor.max() - receptor.min() + 1e-12)
    centres = _centres(labels)
    rng = np.random.default_rng(MAP_SEED)
    names = ["original", "uniform"]
    maps = [rho, np.full(90, float(rho.mean()))]
    rows = [dict(map_id="original", kind="original", permutation=list(range(90))),
            dict(map_id="uniform", kind="uniform", permutation=None)]
    seen = {tuple(range(90))}
    for kind, count in (("spin", n_spin), ("permutation", n_permutation)):
        for draw in range(count):
            for _attempt in range(10000):
                perm = (_spin_index(labels, centres, rng) if kind == "spin"
                        else rng.permutation(90))
                key = tuple(int(value) for value in perm)
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Could not generate a distinct receptor surrogate")
            name = f"{kind}_{draw:03d}"
            names.append(name)
            maps.append(rho[perm])
            rows.append(dict(map_id=name, kind=kind, permutation=perm.tolist()))
    matrix = np.stack(maps)
    for row, values in zip(rows, matrix, strict=True):
        row.update(mean=float(values.mean()), minimum=float(values.min()),
                   maximum=float(values.max()),
                   cortical_moran_knn5=_cortical_moran_knn5(labels, centres, values)
                   if np.ptp(values) > 1e-12 else None,
                   correlation_with_original=float(np.corrcoef(rho, values)[0, 1])
                   if np.ptp(values) > 1e-12 else None,
                   sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())
        if row["kind"] not in ("original", "uniform"):
            np.testing.assert_array_equal(np.sort(values), np.sort(rho))
    return names, matrix, rows


def make_manifest(dataset: Path, n_spin: int, n_permutation: int) -> tuple[dict, np.ndarray, np.ndarray]:
    index = validate_dataset(dataset)
    with np.load(dataset / "atlas.npz") as atlas:
        labels = atlas["labels"].astype(str)
    names, maps, map_rows = build_maps(labels, n_spin, n_permutation)
    subjects = {cohort: sorted(index["cohorts"][cohort]["subject_ids"])
                for cohort in COHORTS}
    cases = []
    for cohort, (group, b_pa) in COHORTS.items():
        for subject in subjects[cohort]:
            cases.append(dict(cohort=cohort, group=group, subject=subject,
                              b_pa=b_pa, occupancy=0.0, map_id="baseline"))
            for name in names:
                cases.append(dict(cohort=cohort, group=group, subject=subject,
                                  b_pa=b_pa, occupancy=HIGHEST_OCCUPANCY,
                                  map_id=name))
    model_sources = [
        "scripts/sweep_split_leak_tep_drive.py",
        "scripts/b_sweep_publication_run.py",
        "scripts/run_adex_impulse_response.py",
        "scripts/first_order_amplitude_trials.py",
        "src/tvbtoolkit/workflows/pharmacology.py",
        "src/tvbtoolkit/whole_brain/simulation.py",
        "src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_gK_gNa.py",
    ]
    manifest = dict(protocol=PROTOCOL, subjects=subjects, cases=cases,
                    map_rows=map_rows, map_ids=names,
                    n_spin=n_spin, n_permutation=n_permutation,
                    dataset_index_sha256=sha256_file(dataset / "index.json"),
                    receptor_csv_sha256=sha256_file(ROOT / "data/receptors/hansen_receptors_aal90.csv"),
                    centres_sha256=sha256_file(ROOT / "data/connectivity/average_aal90/centres.txt"),
                    model_source_sha256={name: sha256_file(ROOT / name)
                                         for name in model_sources},
                    master_seed=MASTER_SEED, map_seed=MAP_SEED,
                    map_matrix_sha256=hashlib.sha256(np.ascontiguousarray(maps).tobytes()).hexdigest())
    return manifest, labels, maps


def validate_frozen(bank: Path, dataset: Path, manifest: dict) -> np.ndarray:
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Frozen protocol changed")
    validate_dataset(dataset)
    checks = {
        "dataset_index_sha256": sha256_file(dataset / "index.json"),
        "receptor_csv_sha256": sha256_file(ROOT / "data/receptors/hansen_receptors_aal90.csv"),
        "centres_sha256": sha256_file(ROOT / "data/connectivity/average_aal90/centres.txt"),
    }
    for key, value in checks.items():
        if value != manifest[key]:
            raise ValueError(f"Frozen input changed: {key}")
    for source, expected in manifest["model_source_sha256"].items():
        if sha256_file(ROOT / source) != expected:
            raise ValueError(f"Frozen model source changed: {source}")
    with np.load(bank / "maps.npz") as saved:
        labels = saved["labels"].astype(str)
        matrix = saved["rho"]
        np.testing.assert_array_equal(saved["map_ids"].astype(str), manifest["map_ids"])
    if matrix.shape != (len(manifest["map_ids"]), 90):
        raise ValueError("Frozen map matrix has wrong dimensions")
    if hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest() != manifest["map_matrix_sha256"]:
        raise ValueError("Frozen map matrix changed")
    with np.load(dataset / "atlas.npz") as atlas:
        np.testing.assert_array_equal(labels, atlas["labels"].astype(str))
    return matrix


def recording_path(bank: Path, case: dict, trial: int) -> Path:
    return (bank / "recordings" / f"{case['cohort']}_{case['subject']}"
            / case["map_id"] / f"trial_{trial:03d}.npz")
