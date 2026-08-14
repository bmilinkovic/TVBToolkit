from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tvbtoolkit.datasets.structural_provenance import (
    validate_native_invnodevol_dataset,
    validate_spontaneous_cache,
)


def _write_index(root: Path, *, scheme: str = "native_invnodevol") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.json"
    path.write_text(
        json.dumps(
            {
                "connectivity_normalization": {
                    "scheme": scheme,
                    "divisor": None,
                    "simulator_normalization": "none",
                },
                "connectivity_weights": {
                    "variant": "invnodevol",
                    "subject_rescaling": "none",
                    "cohort_rescaling": "none",
                    "damage_mask": "connectivity == 0",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_native_invnodevol_dataset_provenance(tmp_path: Path) -> None:
    _write_index(tmp_path)
    provenance = validate_native_invnodevol_dataset(tmp_path)
    assert provenance["structural_connectivity_normalization"] == "native_invnodevol"
    assert provenance["simulator_connectivity_normalization"] == "none"
    assert len(provenance["dataset_index_sha256"]) == 64


def test_legacy_dataset_is_rejected(tmp_path: Path) -> None:
    _write_index(tmp_path, scheme="legacy_column_sum")
    with pytest.raises(ValueError, match="not the native inverse-node-volume"):
        validate_native_invnodevol_dataset(tmp_path)


def test_spontaneous_cache_fingerprint_and_shapes(tmp_path: Path) -> None:
    _write_index(tmp_path)
    provenance = validate_native_invnodevol_dataset(tmp_path)
    cache = tmp_path / "seed_000.npz"
    np.savez_compressed(
        cache,
        time_rate_ms=np.arange(12.0),
        rate=np.zeros((12, 90)),
        time_bold_ms=np.arange(10.0),
        bold=np.zeros((10, 90)),
        region_labels=np.asarray([f"R{i}" for i in range(90)]),
        dataset_index_sha256=np.array([provenance["dataset_index_sha256"]]),
        structural_connectivity_normalization=np.array(["native_invnodevol"]),
        simulator_connectivity_normalization=np.array(["none"]),
        subject_rescaling=np.array(["none"]),
        shared_noise_mode=np.array(["global"]),
        noise_alpha=np.array([0.25]),
        b_e_pa=np.array([10.0]),
        rate_monitor_period_ms=np.array([3.9]),
        bold_monitor_period_ms=np.array([2400.0]),
    )
    validate_spontaneous_cache(
        cache,
        expected_dataset_index_sha256=provenance["dataset_index_sha256"],
        expected_rate_monitor_period_ms=3.9,
        expected_bold_monitor_period_ms=2400.0,
    )
    with pytest.raises(ValueError, match="fingerprint differs"):
        validate_spontaneous_cache(
            cache,
            expected_dataset_index_sha256="0" * 64,
            validate_arrays=False,
        )
