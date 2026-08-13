from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_serotonergic_pci_trial_convergence as convergence


def _source_manifest(subjects_per_cohort: int = 2) -> dict:
    subjects = []
    condition_by_cohort = {
        "coma": "COMA",
        "uws": "UWS",
        "mcs": "MCS",
        "emcs": "EMCS",
        "control": "CNT",
    }
    for cohort, condition in condition_by_cohort.items():
        for index in range(subjects_per_cohort):
            subjects.append(
                {
                    "cohort": cohort,
                    "condition": condition,
                    "subject_id": f"{cohort[:2]}{index:04d}",
                }
            )
    return {
        "script": "scripts/run_serotonergic_pci_full.py",
        "protocol_version": "test-corrected-protocol",
        "protocol_fingerprint": "f" * 64,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "trial_seeds": list(range(100)),
        "n_trials": 100,
        "stim_onsets_ms_by_trial_seed": {
            str(seed): 5000.0 + seed for seed in range(100)
        },
        "occupancies": [0.0, 0.25, 0.5, 0.766],
        "simulate_baseline": True,
        "scenario": "private_alpha0",
        "t_analysis_ms": 300.0,
        "atlas_labels_sha256": "a" * 64,
        "receptor_tracer": "cimbi",
        "receptor_csv_sha256": "c" * 64,
        "receptor_map_sha256": "r" * 64,
        "stim_region_labels": ["Supp_Motor_Area_L"],
        "model_form": "split_gK_gNa_all_occupancies",
    }


def test_defaults_lock_practical_canonical_workflow() -> None:
    args = convergence.parse_args([])

    assert args.subjects_per_cohort == 1
    assert args.repeats == 5
    assert args.subset_sizes == [20, 40, 60, 80]
    assert args.pci_significance_method == "pre_post_swap"
    assert args.pci_permutation_replicates == 1000
    assert args.pci_alpha == 0.05
    assert args.pci_response_start_ms == 8.0
    assert args.pci_min_source_entropy == 0.08
    assert args.pci_st_baseline_window_ms == [-300.0, -50.0]
    assert args.pci_st_response_window_ms == [8.0, 300.0]


def test_subject_selection_is_deterministic_and_cohort_stratified() -> None:
    manifest = _source_manifest(subjects_per_cohort=3)
    kwargs = {
        "cohorts": list(convergence.DEFAULT_COHORTS),
        "subjects_per_cohort": 1,
        "selection_seed": 17,
        "explicit_subjects": None,
    }

    first = convergence.select_subjects(manifest, **kwargs)
    second = convergence.select_subjects(manifest, **kwargs)

    assert first == second
    assert [subject.cohort for subject in first] == list(
        convergence.DEFAULT_COHORTS
    )
    assert len(
        {(subject.cohort, subject.subject_id) for subject in first}
    ) == 5
    assert all(
        subject.selection_method == "sha256_rank_within_cohort"
        for subject in first
    )


def test_nested_subsets_are_reproducible_nested_and_full_occurs_once() -> None:
    kwargs = {
        "trial_seeds": list(range(100)),
        "subset_sizes": [20, 40, 60, 80],
        "repeats": 3,
        "subset_seed": 23,
        "cohort": "emcs",
        "subject_id": "e0001",
    }

    first = convergence.build_nested_subsets(**kwargs)
    second = convergence.build_nested_subsets(**kwargs)

    assert first == second
    assert sum(item["subset_kind"] == "full_once" for item in first) == 1
    assert sum(item["n_trials"] == 100 for item in first) == 1
    for repeat in range(1, 4):
        repeated = [
            item for item in first if int(item["repeat"]) == repeat
        ]
        previous: set[int] = set()
        for item in repeated:
            current = set(item["trial_seeds"])
            assert previous.issubset(current)
            assert len(current) == item["n_trials"]
            previous = current


def test_estimator_computes_full_trial_set_exactly_once() -> None:
    trials = [
        np.full((8, 3), float(seed), dtype=float) for seed in range(100)
    ]
    subsets = convergence.build_nested_subsets(
        list(range(100)),
        subset_sizes=[20, 40, 60, 80],
        repeats=2,
        subset_seed=5,
        cohort="mcs",
        subject_id="m0001",
    )
    call_sizes: list[int] = []

    def fake_metric(trial_subset, **_kwargs):
        call_sizes.append(len(trial_subset))
        return {
            "pci_lz": float(len(trial_subset)),
            "pci_st": float(2 * len(trial_subset)),
        }

    rows = convergence.estimate_nested_subsets(
        trials,
        trial_seeds=list(range(100)),
        subsets=subsets,
        onset=4,
        dt_ms=10.0,
        t_analysis_ms=40.0,
        metric_config={},
        metric_function=fake_metric,
    )

    assert call_sizes.count(100) == 1
    assert call_sizes.count(20) == 2
    assert len(rows) == 2 * 4 + 1
    assert all("pci_lz" in row and "pci_st" in row for row in rows)


def test_both_estimators_run_on_synthetic_aligned_trials() -> None:
    rng = np.random.default_rng(41)
    dt_ms = 1000.0 / 128.0
    onset = int(round(300.0 / dt_ms))
    n_trials, n_times, n_regions = 8, 2 * onset, 6
    trials = []
    response_time = np.arange(n_times - onset, dtype=float)
    for trial_index in range(n_trials):
        signal = 2.0 + rng.normal(0.0, 0.08, (n_times, n_regions))
        for region in range(n_regions):
            signal[onset:, region] += (
                0.25
                + 0.04 * region
                + 0.12 * np.sin(
                    response_time / (2.5 + 0.2 * region) + 0.1 * trial_index
                )
            )
        trials.append(signal)

    metrics = convergence.compute_pci_metrics(
        trials,
        onset=onset,
        dt_ms=dt_ms,
        t_analysis_ms=300.0,
        metric_config={
            "pci_significance_method": "pre_post_swap",
            "pci_permutation_replicates": 20,
            "pci_alpha": 0.1,
            "pci_seed": 7,
            "pci_response_start_ms": 8.0,
            "pci_min_source_entropy": 0.08,
            "pci_st_baseline_window_ms": [-300.0, -50.0],
            "pci_st_response_window_ms": [0.0, 300.0],
            "pci_st_k": 1.2,
            "pci_st_min_snr": 1.1,
            "pci_st_max_var_percent": 99.0,
            "pci_st_n_steps": 20,
        },
    )

    assert np.isfinite(metrics["pci_lz"])
    assert np.isfinite(metrics["pci_st"])
    assert metrics["pci_lz"] >= 0.0
    assert metrics["pci_st"] >= 0.0
    assert metrics["pci_st_n_components"] >= 0
    assert metrics["pci_st_target_response_stop_ms"] == 300.0
    assert metrics["pci_st_effective_response_stop_ms"] == pytest.approx(
        296.875
    )


def test_output_must_not_be_nested_in_source_cache(tmp_path: Path) -> None:
    cache = tmp_path / "immutable-cache"

    with pytest.raises(ValueError, match="separate"):
        convergence._assert_separate_output(cache, cache / "analysis")


def test_max_dose_minus_baseline_is_matched_within_subset() -> None:
    rows = []
    for occupancy in convergence.CANONICAL_OCCUPANCIES:
        rows.append(
            {
                "cohort": "emcs",
                "condition": "EMCS",
                "subject_id": "e0001",
                "repeat": 2,
                "n_trials": 40,
                "subset_kind": "nested_resample",
                "occupancy": occupancy,
                "pci_lz": 0.5 + occupancy,
                "pci_st": 4.0 + 2.0 * occupancy,
            }
        )

    effects = convergence._dose_effects(pd.DataFrame(rows))
    summary = convergence._summarize_dose_effects(effects)

    assert len(effects) == 1
    assert effects.iloc[0][
        "delta_pci_lz_maximum_minus_baseline"
    ] == pytest.approx(0.766)
    assert effects.iloc[0][
        "delta_pci_st_maximum_minus_baseline"
    ] == pytest.approx(1.532)
    assert len(summary) == 1


def test_dry_run_writes_plan_without_opening_or_mutating_trials(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "immutable-cache"
    manifest_path = cache / "logs" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(_source_manifest(subjects_per_cohort=1)),
        encoding="utf-8",
    )
    before = {
        path.relative_to(cache): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in cache.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "separate-output"

    convergence.main(
        [
            "--cache-root",
            str(cache),
            "--output-root",
            str(output),
            "--dry-run",
        ]
    )

    after = {
        path.relative_to(cache): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in cache.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert (output / "logs" / "run_manifest.json").is_file()
    assert (output / "tables" / "selected_subjects.csv").is_file()
    subsets = pd.read_csv(output / "tables" / "trial_subsets.csv")
    assert set(subsets["n_trials"]) == {20, 40, 60, 80, 100}


def test_convergence_plot_is_created_from_synthetic_estimates(
    tmp_path: Path,
) -> None:
    rows = []
    for n_trials in (20, 40, 60, 80, 100):
        rows.append(
            {
                "condition": "EMCS",
                "occupancy": 0.766,
                "n_trials": n_trials,
                "pci_lz": 0.5 + n_trials / 1000.0,
                "pci_st": 4.0 + n_trials / 100.0,
            }
        )
    convergence._plot_convergence(pd.DataFrame(rows), tmp_path)

    assert (
        tmp_path
        / "figures"
        / "serotonergic_pci_trial_convergence.png"
    ).is_file()
    assert (
        tmp_path
        / "figures"
        / "serotonergic_pci_trial_convergence.pdf"
    ).is_file()
