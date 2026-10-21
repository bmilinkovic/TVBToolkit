"""Frozen first-order SMA/5-HT2A trial-bank contract; PCI lives elsewhere."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COHORTS = {"control": ("CNT", 5.0), "emcs": ("EMCS", 15.0),
           "mcs": ("MCS", 35.0), "uws": ("UWS", 40.0)}
OCCUPANCIES = (0.0, 0.25, 0.5, 0.766)
TRIALS = 100
MASTER_SEED = 20261020
EXPECTED_DIVISOR = 122.63607788085938
PROTOCOL = {
    "version": "sma8s_el62_full100_first_order_v1",
    "model_order": 1,
    "split_leak": True,
    "G": 0.3,
    "external_drive_hz": 0.315,
    "weight_noise": 0.0002,
    "regional_OU_tau_ms": 2.0,
    "tau_w_ms": 500.0,
    "E_L_e_mv": -62.0,
    "drug_E_L_e_mv": -60.2,
    "E_L_i_mv": -65.0,
    "drug_E_L_i_mv": -64.4,
    "source_label": "Supp_Motor_Area_L",
    "pulse_hz_per_ms": 0.55,
    "pulse_duration_ms": 10.0,
    "simulation_ms": 8000.0,
    "transient_discard_ms": 2000.0,
    "onset_range_ms_inclusive": [2500, 7500],
    "monitor": "TVB TemporalAverage",
    "monitor_period_ms": 1.0,
    "saved_peri_onset_ms": [-500, 500],
    "prestim_ms": [-300, 0],
    "poststim_ms": [0, 300],
    "SC_normalization_scheme": "shared_control_maximum_edge",
    "SC_divisor": EXPECTED_DIVISOR,
    "occupancies": list(OCCUPANCIES),
    "trials_per_subject_occupancy": TRIALS,
    "b_pa_by_cohort": {name: b for name, (_, b) in COHORTS.items()},
    "second_order": "reserved; not enabled until pending validation is reviewed",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_dataset(root: Path) -> dict:
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    norm = index.get("connectivity_normalization", {})
    if norm.get("scheme") != PROTOCOL["SC_normalization_scheme"]:
        raise ValueError(f"Wrong SC normalization scheme: {norm.get('scheme')!r}")
    if not np.isclose(float(norm.get("divisor", np.nan)), EXPECTED_DIVISOR, rtol=0, atol=1e-9):
        raise ValueError(f"Wrong shared control maximum divisor: {norm.get('divisor')!r}")
    for cohort in COHORTS:
        if not index["cohorts"][cohort]["subject_ids"]:
            raise ValueError(f"Empty cohort: {cohort}")
    return index


def _selected40(root: Path) -> dict[str, list[str]]:
    """Reproduce the damage-stratified ten-per-cohort subject selection."""
    metadata = json.loads((root / "normalization_metadata.json").read_text())
    anchors = {"control": "c0015", "emcs": "e0003", "mcs": "m0075", "uws": "u0020"}
    out = {}
    for cohort in COHORTS:
        candidates = sorted((r for r in metadata["subjects"] if r["cohort"] == cohort),
                            key=lambda r: (r["zero_entries"], r["subject_id"]))
        ids = list(np.rint(np.linspace(0, len(candidates) - 1, 10)).astype(int))
        anchor = next(i for i, r in enumerate(candidates) if r["subject_id"] == anchors[cohort])
        if anchor not in ids:
            nearest = min(range(len(ids)), key=lambda i: abs(ids[i] - anchor))
            ids[nearest] = anchor
        if len(set(ids)) != 10:
            raise ValueError(f"Nonunique selection for {cohort}")
        out[cohort] = sorted(candidates[i]["subject_id"] for i in ids)
    return out


def manifest_for(dataset: Path, scope: str) -> dict:
    index = validate_dataset(dataset)
    if scope not in ("all_noncoma", "selected40"):
        raise ValueError(scope)
    subjects = (_selected40(dataset) if scope == "selected40" else
                {cohort: sorted(index["cohorts"][cohort]["subject_ids"])
                 for cohort in COHORTS})
    cases = []
    for cohort, (group, b_pa) in COHORTS.items():
        for subject in subjects[cohort]:
            for occupancy in OCCUPANCIES:
                cases.append(dict(cohort=cohort, group=group, subject=subject,
                                  b_pa=b_pa, occupancy=occupancy))
    receptor = ROOT / "data/receptors/hansen_receptors_aal90.csv"
    model_sources = (ROOT / "scripts/sweep_split_leak_tep_drive.py",
                     ROOT / "scripts/b_sweep_publication_run.py",
                     ROOT / "scripts/run_adex_impulse_response.py",
                     ROOT / "scripts/first_order_amplitude_trials.py",
                     ROOT / "src/tvbtoolkit/workflows/pharmacology.py",
                     ROOT / "src/tvbtoolkit/whole_brain/simulation.py",
                     ROOT / "src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_gK_gNa.py")
    return dict(protocol=PROTOCOL, scope=scope, subjects=subjects, cases=cases,
                dataset_index_sha256=sha256_file(dataset / "index.json"),
                receptor_csv_sha256=sha256_file(receptor),
                model_source_sha256={str(path.relative_to(ROOT)): sha256_file(path)
                                     for path in model_sources},
                master_seed=MASTER_SEED)


def trial_seed(cohort: str, subject: str, trial: int, master: int = MASTER_SEED) -> int:
    key = f"{master}:{cohort}:{subject}:{trial}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def trial_onset_ms(seed: int) -> int:
    return int(np.random.default_rng(seed ^ 123456).integers(2500, 7501))


def recording_path(bank: Path, case: dict, trial: int) -> Path:
    return (bank / "recordings" / f"{case['cohort']}_{case['subject']}"
            / f"occ_{case['occupancy']:.3f}" / f"trial_{trial:03d}.npz")
