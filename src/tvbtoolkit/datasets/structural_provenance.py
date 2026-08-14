"""Validation and provenance helpers for production structural datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PRODUCTION_STRUCTURAL_SCHEME = "native_invnodevol"
PRODUCTION_SIMULATOR_NORMALIZATION = "none"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_native_invnodevol_dataset(dataset_root: str | Path) -> dict[str, Any]:
    """Validate the structural dataset used by current production analyses.

    The inverse-node-volume weights are already in their analysis scale.  They
    must therefore enter TVB without any subject-wise or simulator-side
    normalization.
    """
    root = Path(dataset_root).expanduser().resolve()
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing structural dataset index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    normalization = dict(index.get("connectivity_normalization", {}))
    weights = dict(index.get("connectivity_weights", {}))

    checks = {
        "connectivity_normalization.scheme": (
            normalization.get("scheme"), PRODUCTION_STRUCTURAL_SCHEME
        ),
        "connectivity_normalization.simulator_normalization": (
            normalization.get("simulator_normalization"),
            PRODUCTION_SIMULATOR_NORMALIZATION,
        ),
        "connectivity_weights.variant": (weights.get("variant"), "invnodevol"),
        "connectivity_weights.subject_rescaling": (
            weights.get("subject_rescaling"), "none"
        ),
        "connectivity_weights.cohort_rescaling": (
            weights.get("cohort_rescaling"), "none"
        ),
    }
    failures = [
        f"{key}={actual!r} (expected {expected!r})"
        for key, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise ValueError(
            "Structural dataset is not the native inverse-node-volume production "
            f"dataset ({index_path}): " + "; ".join(failures)
        )

    return {
        "dataset_root": str(root),
        "dataset_index_path": str(index_path),
        "dataset_index_sha256": _sha256_file(index_path),
        "structural_connectivity_normalization": PRODUCTION_STRUCTURAL_SCHEME,
        "simulator_connectivity_normalization": PRODUCTION_SIMULATOR_NORMALIZATION,
        "subject_rescaling": "none",
        "cohort_rescaling": "none",
        "connectivity_variant": "invnodevol",
        "damage_mask": weights.get("damage_mask"),
        "damage_mask_source": weights.get("damage_mask_source"),
    }


def _npz_scalar(data: Any, key: str) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"Cache field {key!r} must contain exactly one value.")
    return value.reshape(-1)[0].item()


def validate_spontaneous_cache(
    npz_path: str | Path,
    *,
    expected_dataset_index_sha256: str,
    require_bold: bool = True,
    validate_arrays: bool = True,
    expected_rate_monitor_period_ms: float | None = None,
    expected_bold_monitor_period_ms: float | None = None,
) -> None:
    """Reject stale, partial, or structurally incompatible simulation caches."""
    path = Path(npz_path)
    required = {
        "time_rate_ms",
        "rate",
        "region_labels",
        "dataset_index_sha256",
        "structural_connectivity_normalization",
        "simulator_connectivity_normalization",
        "subject_rescaling",
        "shared_noise_mode",
        "noise_alpha",
        "b_e_pa",
    }
    if require_bold:
        required.update({"time_bold_ms", "bold", "bold_monitor_period_ms"})

    try:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"missing fields: {missing}")
            if str(_npz_scalar(data, "dataset_index_sha256")) != str(
                expected_dataset_index_sha256
            ):
                raise ValueError("dataset index fingerprint differs from this analysis")
            if str(_npz_scalar(data, "structural_connectivity_normalization")) != (
                PRODUCTION_STRUCTURAL_SCHEME
            ):
                raise ValueError("cache was not generated from native_invnodevol weights")
            if str(_npz_scalar(data, "simulator_connectivity_normalization")) != "none":
                raise ValueError("cache used simulator-side connectivity normalization")
            if str(_npz_scalar(data, "subject_rescaling")) != "none":
                raise ValueError("cache used subject-wise connectivity rescaling")
            if expected_rate_monitor_period_ms is not None:
                actual = float(_npz_scalar(data, "rate_monitor_period_ms"))
                if not np.isclose(actual, expected_rate_monitor_period_ms, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"rate monitor period is {actual} ms; expected "
                        f"{expected_rate_monitor_period_ms} ms"
                    )
            if require_bold and expected_bold_monitor_period_ms is not None:
                actual = float(_npz_scalar(data, "bold_monitor_period_ms"))
                if not np.isclose(actual, expected_bold_monitor_period_ms, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"BOLD monitor period is {actual} ms; expected "
                        f"{expected_bold_monitor_period_ms} ms"
                    )

            if validate_arrays:
                rate = np.asarray(data["rate"])
                time_rate = np.asarray(data["time_rate_ms"])
                if rate.ndim != 2 or rate.shape[0] != time_rate.size or rate.shape[1] != 90:
                    raise ValueError(
                        f"invalid rate shape/time axis: rate={rate.shape}, time={time_rate.shape}"
                    )
                if require_bold:
                    bold = np.asarray(data["bold"])
                    time_bold = np.asarray(data["time_bold_ms"])
                    if bold.ndim != 2 or bold.shape[0] != time_bold.size or bold.shape[1] != 90:
                        raise ValueError(
                            f"invalid BOLD shape/time axis: bold={bold.shape}, time={time_bold.shape}"
                        )
    except Exception as exc:
        raise ValueError(f"Invalid spontaneous simulation cache {path}: {exc}") from exc
