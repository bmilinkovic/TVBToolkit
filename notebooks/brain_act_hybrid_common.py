from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def resolve_project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "src").exists() and (root.parent / "src").exists():
        root = root.parent
    return root


PROJECT_ROOT = resolve_project_root()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(PROJECT_ROOT / ".tvb-temp"))

DATASET_ROOT = PROJECT_ROOT / "data" / "doc_data" / "converted_structural"

COHORT_TO_CONDITION = {
    "control": "CNT",
    "emcs": "EMCS",
    "mcs": "MCS",
    "uws": "UWS",
    "coma": "COMA",
}
CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]

COND_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
    "EMCS": "#E8B56D",
    "CNT": "#5B8A72",
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "private_alpha0": {
        "label": "Private only (a=0.00)",
        "noise_alpha": 0.00,
        "shared_noise_mode": "none",
    },
    "global_alpha_low": {
        "label": "Global shared, low a (0.15)",
        "noise_alpha": 0.15,
        "shared_noise_mode": "global",
    },
    "global_alpha_med": {
        "label": "Global shared, medium a (0.40)",
        "noise_alpha": 0.40,
        "shared_noise_mode": "global",
    },
    "global_alpha_high": {
        "label": "Global shared, high a (0.70)",
        "noise_alpha": 0.70,
        "shared_noise_mode": "global",
    },
    "sc_alpha_med": {
        "label": "SC-shaped shared, medium a (0.40)",
        "noise_alpha": 0.40,
        "shared_noise_mode": "connectivity",
    },
}

for _alpha_i in range(5, 51, 5):
    _alpha = _alpha_i / 100.0
    _tag = f"{_alpha_i:03d}"
    SCENARIOS[f"global_alpha_{_tag}"] = {
        "label": f"Global shared (a={_alpha:.2f})",
        "noise_alpha": _alpha,
        "shared_noise_mode": "global",
    }
    SCENARIOS[f"sc_alpha_{_tag}"] = {
        "label": f"SC-shaped shared (a={_alpha:.2f})",
        "noise_alpha": _alpha,
        "shared_noise_mode": "connectivity",
    }

# Updated v2 mean-field/whole-brain model coefficients and settings.
BASE_PARAMETER_MODEL_NEW: dict[str, Any] = {
    "T": 20.0,
    "P_e": [
        -0.04983106,
        0.00506355,
        -0.02347012,
        0.00229515,
        -0.00041053,
        0.01054705,
        -0.03659253,
        0.00743749,
        0.00126506,
        -0.04072161,
    ],
    "P_i": [
        -0.05149122,
        0.00400369,
        -0.00835201,
        0.00024142,
        -0.00050706,
        0.00143454,
        -0.01468669,
        0.00450271,
        0.00284722,
        -0.01535780,
    ],
    "E_L_e": -63.0,
    "E_L_i": -65.0,
    "b_e": 5.0,
    "tau_e_e": 5.0,
    "tau_e_i": 5.0,
    "initial_condition": {
        "E": [0.004, 0.004],
        "I": [0.010, 0.010],
        "C_ee": [0.0, 0.0],
        "C_ei": [0.0, 0.0],
        "C_ii": [0.0, 0.0],
        "W_e": [50.0, 50.0],
        "W_i": [0.0, 0.0],
        "noise": [0.0, 0.0],
    },
}

# Keep old monitor sampling.
RATE_MONITOR_HZ_OLD = 128.0
RATE_MONITOR_PERIOD_MS_OLD = 1000.0 / RATE_MONITOR_HZ_OLD


@dataclass(frozen=True)
class SubjectJob:
    cohort: str
    subject_id: str
    condition: str



def get_subject_jobs(dataset_root: Path = DATASET_ROOT) -> list[SubjectJob]:
    from tvbtoolkit.datasets.brain_act import list_subjects

    per_cohort = list_subjects(dataset_root=dataset_root, cohort=None)
    jobs: list[SubjectJob] = []
    for cohort in ["control", "emcs", "mcs", "uws", "coma"]:
        ids = per_cohort.get(cohort, [])
        for sid in ids:
            jobs.append(SubjectJob(cohort=cohort, subject_id=sid, condition=COHORT_TO_CONDITION[cohort]))
    return jobs



def parse_sim_npz_path(npz_path: Path, sim_root: Path) -> tuple[str, str, str, int]:
    rel = npz_path.relative_to(sim_root)
    # Expected layouts:
    #   <sim_root>/<scenario>/<cohort>/<subject_id>/seed_000.npz
    # or legacy:
    #   <sim_root>/sims/<scenario>/<cohort>/<subject_id>/seed_000.npz
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected simulation path layout: {npz_path}")
    offset = 1 if (parts[0] == "sims" and len(parts) >= 5) else 0
    scenario = parts[offset + 0]
    cohort = parts[offset + 1]
    subject_id = parts[offset + 2]
    seed_str = rel.stem.split("_")[-1]
    seed = int(seed_str)
    return scenario, cohort, subject_id, seed



def iter_spontaneous_npz(sim_root: Path, scenario_filter: Iterable[str] | None = None) -> list[Path]:
    paths = sorted(sim_root.glob("*/*/*/seed_*.npz"))
    if scenario_filter is None:
        return paths
    allowed = set(scenario_filter)
    out: list[Path] = []
    for p in paths:
        if p.parts[-4] in allowed:
            out.append(p)
    return out



def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
