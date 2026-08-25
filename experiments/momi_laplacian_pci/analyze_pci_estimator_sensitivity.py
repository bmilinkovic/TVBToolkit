#!/usr/bin/env python3
"""Audit why current and legacy PCI-LZ rank cached subject responses differently."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial


ROOT = Path("results/momi_laplacian_5ht2a_four_subject")
OUTPUT = ROOT / "pci_estimator_sensitivity.csv"
SUBJECTS = (
    ("CNT", "control", "c0015"),
    ("EMCS", "emcs", "e0003"),
    ("MCS", "mcs", "m0075"),
    ("UWS", "uws", "u0020"),
)


def load_trials(cohort: str, subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    time = None
    for seed in range(10):
        path = ROOT / "trials" / cohort / subject_id / "occ_000" / f"seed_{seed:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            this_time = np.asarray(data["time_relative_ms"], dtype=float)
            if time is None:
                time = this_time
            else:
                np.testing.assert_allclose(this_time, time, rtol=0.0, atol=1e-9)
            arrays.append(np.asarray(data["rate_hz"], dtype=float) / 1000.0)
    return np.stack(arrays), np.asarray(time)


def rank_correlation(values: list[float]) -> float:
    """Pearson correlation of stable ranks with severity CNT..UWS."""
    order = np.argsort(np.argsort(np.asarray(values), kind="stable"), kind="stable").astype(float)
    severity = np.arange(len(values), dtype=float)
    return float(np.corrcoef(order, severity)[0, 1])


def calculate(
    trials: np.ndarray,
    time: np.ndarray,
    *,
    estimator: str,
    alpha: float = 0.01,
    response_start_ms: float = 8.0,
    two_sided: bool = True,
    n_trials: int = 10,
    nshuffles: int = 10,
) -> dict[str, float]:
    onset = int(np.argmin(np.abs(time)))
    dt = float(np.median(np.diff(time)))
    if estimator == "current":
        result = pci_casali_like_multi_trial(
            list(trials[:n_trials]), onset, 300.0, dt_ms=dt,
            binarise_method="casali",
            binarise_kwargs={
                "n_bootstrap": 1000,
                "alpha": alpha,
                "seed": 0,
                "significance_method": "trial_bootstrap",
                "two_sided": two_sided,
            },
            response_start_ms=response_start_ms,
            min_source_entropy=None,
            return_debug=True,
        )
        return {
            "pci": float(result["pci"]),
            "active_fraction": float(result["active_fraction"]),
            "threshold": float(result["threshold"]),
        }
    np.random.seed(0)
    result = pci_casali_like_multi_trial(
        list(trials[:n_trials]), onset, 300.0, dt_ms=dt,
        binarise_method="tvbsim",
        nshuffles=nshuffles,
        percentile=100.0,
        response_start_ms=response_start_ms,
        min_source_entropy=None,
        return_debug=True,
    )
    return {
        "pci": float(result["pci"]),
        "active_fraction": float("nan"),
        "threshold": float("nan"),
    }


def main() -> None:
    data = {sid: load_trials(cohort, sid) for _, cohort, sid in SUBJECTS}
    specifications: list[tuple[str, str, dict[str, float | int | bool]]] = []
    for alpha in (0.20, 0.10, 0.05, 0.02, 0.01, 0.005):
        specifications.append((f"current_alpha_{alpha:g}", "current", {"alpha": alpha}))
    for start in (0.0, 8.0, 15.0, 30.0, 50.0):
        specifications.append((f"current_start_{start:g}", "current", {"response_start_ms": start}))
    specifications.append(("current_one_sided", "current", {"two_sided": False}))
    for count in (3, 5, 10):
        specifications.append((f"current_trials_{count}", "current", {"n_trials": count}))
    for start in (0.0, 8.0, 15.0, 30.0, 50.0):
        specifications.append((f"legacy_start_{start:g}", "legacy", {"response_start_ms": start}))
    for shuffles in (10, 20, 50, 100, 500):
        specifications.append(
            (f"legacy_shuffles_{shuffles}", "legacy", {"nshuffles": shuffles, "response_start_ms": 0.0})
        )

    rows: list[dict[str, object]] = []
    for test_name, estimator, kwargs in specifications:
        values: list[float] = []
        results = []
        for condition, _, sid in SUBJECTS:
            trials, time = data[sid]
            result = calculate(trials, time, estimator=estimator, **kwargs)
            values.append(result["pci"])
            results.append((condition, sid, result))
        correlation = rank_correlation(values)
        for condition, sid, result in results:
            rows.append(
                {
                    "test": test_name,
                    "estimator": estimator,
                    "condition": condition,
                    "subject_id": sid,
                    "pci_lz": result["pci"],
                    "active_fraction": result["active_fraction"],
                    "threshold": result["threshold"],
                    "rank_correlation_with_severity": correlation,
                    **kwargs,
                }
            )
    fields = sorted({key for row in rows for key in row})
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for test_name, _, _ in specifications:
        selected = [row for row in rows if row["test"] == test_name]
        values = " ".join(f"{row['condition']}={float(row['pci_lz']):.4f}" for row in selected)
        print(f"{test_name:24s} rho={float(selected[0]['rank_correlation_with_severity']):+.2f} {values}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
