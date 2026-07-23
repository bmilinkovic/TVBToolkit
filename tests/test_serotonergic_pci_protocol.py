from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_serotonergic_pci_full as sero_full
from scripts import run_serotonergic_pci_emcs_robustness as sero_robust
from scripts import run_serotonergic_pci_pilot as sero


def _write_trial(path, *, onset_ms: float, impulse_index: int) -> None:
    time_ms = np.arange(0.0, 1000.0, 10.0)
    rate = np.zeros((time_ms.size, 3), dtype=float)
    rate[impulse_index, 0] = 1.0
    np.savez_compressed(
        path,
        time_ms=time_ms,
        rate=rate,
        stim_onset_ms=np.asarray([onset_ms]),
        t_analysis_ms=np.asarray([100.0]),
    )


def _write_provenance_trial(
    path,
    *,
    protocol_version: str = sero.PROTOCOL_VERSION,
    robustness_config=None,
    payload_overrides=None,
):
    """Write a compact but complete corrected-protocol 90-region trial."""
    labels = np.asarray([f"ROI_{index:02d}" for index in range(90)], dtype="U128")
    labels[9] = "Supp_Motor_Area_L"
    receptor_map = np.linspace(0.0, 1.0, 90, dtype=np.float64)
    time_ms = np.arange(4000.0, 6000.0, 10.0)
    onset_ms = 5000.0
    onset_index = int(np.argmin(np.abs(time_ms - onset_ms)))
    fingerprint = "f" * 64
    atlas_hash = sero._sha256_array(labels)
    receptor_hash = sero._sha256_array(receptor_map)
    receptor_csv_hash = "c" * 64
    payload = {
        "time_ms": time_ms,
        "rate": np.zeros((time_ms.size, 90), dtype=float),
        "region_labels": labels,
        "simulation_region_labels": labels,
        "atlas_ordering": np.asarray(["test-order"], dtype="U128"),
        "atlas_labels_sha256": np.asarray([atlas_hash], dtype="U128"),
        "receptor_tracer": np.asarray(["cimbi"], dtype="U32"),
        "receptor_csv_sha256": np.asarray([receptor_csv_hash], dtype="U128"),
        "receptor_map_alignment": np.asarray(
            ["AAL region-label join"],
            dtype="U128",
        ),
        "receptor_map_sha256": np.asarray([receptor_hash], dtype="U128"),
        "receptor_map": receptor_map,
        "protocol_version": np.asarray([protocol_version], dtype="U128"),
        "protocol_fingerprint": np.asarray([fingerprint], dtype="U128"),
        "cohort": np.asarray(["emcs"], dtype="U32"),
        "condition": np.asarray(["EMCS"], dtype="U32"),
        "subject_id": np.asarray(["e0001"], dtype="U128"),
        "scenario": np.asarray(["private_alpha0"], dtype="U128"),
        "stim_onset_ms": np.asarray([onset_ms]),
        "stim_onset_sample_index": np.asarray([onset_index], dtype=np.int64),
        "stim_onset_sample_ms": np.asarray([time_ms[onset_index]]),
        "stim_onset_alignment_residual_ms": np.asarray([0.0]),
        "stim_onset_alignment": np.asarray(
            ["nearest temporal-average sample after per-trial epoching"],
            dtype="U128",
        ),
        "t_analysis_ms": np.asarray([300.0]),
        "rate_monitor_period_ms": np.asarray([10.0]),
        "trial_seed": np.asarray([0]),
        "noise_alpha": np.asarray([0.0]),
        "stim_amplitude": np.asarray([0.00030]),
        "stim_duration_ms": np.asarray([10.0]),
        "stim_region": np.asarray([9], dtype=int),
        "stim_region_labels": np.asarray(
            ["Supp_Motor_Area_L"],
            dtype="U128",
        ),
        "occupancy": np.asarray([0.25]),
        "sc_zero_fraction_upper": np.asarray([0.0]),
    }
    if robustness_config is not None:
        overrides = robustness_config["parameter_model_overrides"]
        override_keys = sorted(overrides)
        payload.update(
            {
                "robustness_config_id": np.asarray(
                    [robustness_config["config_id"]],
                    dtype="U128",
                ),
                "robustness_family": np.asarray(
                    [robustness_config["family"]],
                    dtype="U128",
                ),
                "robustness_direction": np.asarray(
                    [robustness_config["direction"]],
                    dtype="U32",
                ),
                "robustness_factor": np.asarray(
                    [float(robustness_config["factor"])]
                ),
                "parameter_model_override_keys": np.asarray(
                    override_keys,
                    dtype="U128",
                ),
                "parameter_model_override_values": np.asarray(
                    [overrides[key] for key in override_keys],
                    dtype=float,
                ),
                "parameter_model_overrides_json": np.asarray(
                    [sero_robust._canonical_json(overrides)],
                    dtype="U4096",
                ),
                "whole_brain_parameter_model_verified": np.asarray([True]),
                "zerlaut_gk_gna": np.asarray([True]),
            }
        )
    payload.update(payload_overrides or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "protocol_fingerprint": fingerprint,
        "trial_seed": 0,
        "occupancy": 0.25,
        "stim_region_labels": ["Supp_Motor_Area_L"],
        "receptor_map_sha256": receptor_hash,
        "cohort": "emcs",
        "condition": "EMCS",
        "subject_id": "e0001",
        "scenario": "private_alpha0",
        "expected_stim_onset_ms": onset_ms,
        "atlas_labels_sha256": atlas_hash,
        "receptor_tracer": "cimbi",
        "receptor_csv_sha256": receptor_csv_hash,
        "expected_t_analysis_ms": 300.0,
    }


def test_common_trial_validator_accepts_full_valid_provenance(tmp_path) -> None:
    path = tmp_path / "trial_000.npz"
    expected = _write_provenance_trial(path)

    sero._validate_existing_trial(path, **expected)


def test_common_trial_validator_rejects_file_copied_to_wrong_subject(tmp_path) -> None:
    path = tmp_path / "copied" / "trial_000.npz"
    expected = _write_provenance_trial(path)
    expected["subject_id"] = "e0002"

    with pytest.raises(RuntimeError, match="subject_id"):
        sero._validate_existing_trial(path, **expected)


def test_common_trial_validator_rejects_wrong_expected_onset(tmp_path) -> None:
    path = tmp_path / "trial_000.npz"
    expected = _write_provenance_trial(path)
    expected["expected_stim_onset_ms"] = 5010.0

    with pytest.raises(RuntimeError, match="stim_onset_ms"):
        sero._validate_existing_trial(path, **expected)


@pytest.mark.parametrize("failure_mode", ["wrong_shape", "nonfinite"])
def test_common_trial_validator_rejects_corrupt_rate(
    tmp_path,
    failure_mode,
) -> None:
    path = tmp_path / "trial_000.npz"
    time_ms = np.arange(4000.0, 6000.0, 10.0)
    if failure_mode == "wrong_shape":
        rate = np.zeros((time_ms.size, 89), dtype=float)
    else:
        rate = np.zeros((time_ms.size, 90), dtype=float)
        rate[100, 20] = np.nan
    expected = _write_provenance_trial(
        path,
        payload_overrides={"rate": rate},
    )

    with pytest.raises(RuntimeError, match="rate"):
        sero._validate_existing_trial(path, **expected)


@pytest.mark.parametrize("failure_mode", ["reordered_map", "altered_hash"])
def test_common_trial_validator_rejects_altered_receptor_provenance(
    tmp_path,
    failure_mode,
) -> None:
    path = tmp_path / "trial_000.npz"
    receptor_map = np.linspace(0.0, 1.0, 90, dtype=np.float64)
    if failure_mode == "reordered_map":
        altered = receptor_map[::-1].copy()
        payload_overrides = {
            "receptor_map": altered,
            "receptor_map_sha256": np.asarray(
                [sero._sha256_array(altered)],
                dtype="U128",
            ),
        }
    else:
        payload_overrides = {
            "receptor_map_sha256": np.asarray(["0" * 64], dtype="U128")
        }
    expected = _write_provenance_trial(
        path,
        payload_overrides=payload_overrides,
    )

    with pytest.raises(RuntimeError, match="receptor_map"):
        sero._validate_existing_trial(path, **expected)


def test_robustness_trial_validator_accepts_valid_layered_provenance(
    tmp_path,
) -> None:
    config = sero_robust.build_robustness_configs(
        sero_robust._nominal_model_values()
    )[0]
    path = tmp_path / "trial_000.npz"
    expected = _write_provenance_trial(
        path,
        protocol_version=sero_robust.PROTOCOL_VERSION,
        robustness_config=config,
    )

    sero_robust._validate_existing_trial(path, config=config, **expected)


def test_robustness_trial_validator_rejects_config_mismatch(tmp_path) -> None:
    configs = sero_robust.build_robustness_configs(
        sero_robust._nominal_model_values()
    )
    path = tmp_path / "trial_000.npz"
    expected = _write_provenance_trial(
        path,
        protocol_version=sero_robust.PROTOCOL_VERSION,
        robustness_config=configs[0],
    )

    with pytest.raises(RuntimeError, match="robustness_config_id"):
        sero_robust._validate_existing_trial(
            path,
            config=configs[1],
            **expected,
        )


def test_full_aggregate_rejects_wrong_subject_before_pci(
    tmp_path,
    monkeypatch,
) -> None:
    path = (
        tmp_path
        / "sims_pci"
        / "occ_250"
        / "private_alpha0"
        / "emcs"
        / "e0001"
        / "trial_000.npz"
    )
    expected = _write_provenance_trial(
        path,
        payload_overrides={
            "subject_id": np.asarray(["e0002"], dtype="U128")
        },
    )
    args = SimpleNamespace(
        output_root=tmp_path,
        baseline_root=tmp_path,
        scenario="private_alpha0",
        trial_seeds=[0],
        occupancies=[0.25],
        simulate_baseline=True,
        protocol_fingerprint=expected["protocol_fingerprint"],
        stim_region_label=expected["stim_region_labels"],
        receptor_map_sha256=expected["receptor_map_sha256"],
        atlas_labels_sha256=expected["atlas_labels_sha256"],
        receptor_tracer=expected["receptor_tracer"],
        receptor_csv_sha256=expected["receptor_csv_sha256"],
        t_analysis_ms=expected["expected_t_analysis_ms"],
        pci_binarise_method="casali",
        pci_bootstrap_replicates=10,
        pci_alpha=0.01,
        pci_bootstrap_seed=0,
    )
    subject = SimpleNamespace(
        cohort="emcs",
        condition="EMCS",
        subject_id="e0001",
    )

    def _pci_must_not_run(*_args, **_kwargs):
        raise AssertionError("PCI ran before aggregate provenance validation")

    monkeypatch.setattr(sero, "_compute_pci_for_condition", _pci_must_not_run)
    with pytest.raises(RuntimeError, match="subject_id"):
        sero_full._aggregate(args, [subject], {0: 5000.0})


def test_load_trials_aligns_each_trial_to_its_own_stimulation_onset(tmp_path) -> None:
    trial_a = tmp_path / "trial_a.npz"
    trial_b = tmp_path / "trial_b.npz"
    _write_trial(trial_a, onset_ms=300.0, impulse_index=30)
    _write_trial(trial_b, onset_ms=700.0, impulse_index=70)

    trials, onset, dt_ms, t_analysis_ms = sero._load_trials([trial_a, trial_b])

    assert onset == 10
    assert dt_ms == 10.0
    assert t_analysis_ms == 100.0
    assert [trial.shape for trial in trials] == [(20, 3), (20, 3)]
    assert [int(np.argmax(trial[:, 0])) for trial in trials] == [onset, onset]


def test_load_trials_places_last_baseline_sample_before_onset(tmp_path) -> None:
    path = tmp_path / "trial.npz"
    time_ms = np.arange(0.0, 1000.0, 10.0)
    rate = np.zeros((time_ms.size, 2), dtype=float)
    rate[49, 0] = -1.0
    rate[50, 0] = 1.0
    np.savez_compressed(
        path,
        time_ms=time_ms,
        rate=rate,
        stim_onset_ms=np.asarray([500.0]),
        t_analysis_ms=np.asarray([100.0]),
    )

    trials, onset, _, _ = sero._load_trials([path])

    assert onset == 10
    assert trials[0][onset - 1, 0] == -1.0
    assert trials[0][onset, 0] == 1.0


def test_corrected_pci_returns_one_trial_averaged_estimate(tmp_path) -> None:
    trial_a = tmp_path / "trial_a.npz"
    trial_b = tmp_path / "trial_b.npz"
    _write_trial(trial_a, onset_ms=300.0, impulse_index=30)
    _write_trial(trial_b, onset_ms=700.0, impulse_index=70)

    pci_mean, pci_values = sero._compute_pci_for_condition(
        [trial_a, trial_b],
        binarise_method="casali",
        n_bootstrap=10,
        alpha=0.05,
        bootstrap_seed=7,
    )

    assert np.isfinite(pci_mean)
    assert pci_values.shape == (1,)
    assert pci_values[0] == pci_mean


def test_production_defaults_lock_one_hundred_casali_trials() -> None:
    args = sero_full.parse_args([])

    assert args.trial_seeds == list(range(100))
    assert args.stim_region is None
    assert args.stim_region_label == ["Supp_Motor_Area_L"]
    assert args.receptor_tracer == "cimbi"
    assert args.pci_binarise_method == "casali"
    assert args.simulate_baseline is True
    assert args.split_model_all_occupancies is True


def test_production_parser_rejects_label_and_numeric_target_together() -> None:
    with pytest.raises(SystemExit):
        sero_full.parse_args(
            [
                "--stim-region",
                "18",
                "--stim-region-label",
                "Supp_Motor_Area_L",
            ]
        )


def test_stimulation_onsets_are_reproducible_and_unique_for_100_trials() -> None:
    kwargs = {
        "transient_ms": 4000.0,
        "t_analysis_ms": 300.0,
        "trial_sim_ms": 8000.0,
        "seed": 0,
    }
    first = sero._stim_onsets(list(range(100)), **kwargs)
    second = sero._stim_onsets(list(range(100)), **kwargs)

    assert first == second
    assert len(first) == 100
    assert len(set(first.values())) == 100
    assert min(first.values()) >= 4300.0
    assert max(first.values()) < 7700.0


def test_protocol_fingerprint_ignores_execution_only_fields() -> None:
    first = {
        "protocol_version": sero.PROTOCOL_VERSION,
        "stim_region_labels": ["Supp_Motor_Area_L"],
        "n_trials": 100,
        "workers": 48,
        "output_root": "/first",
        "overwrite": False,
    }
    second = {
        **first,
        "workers": 12,
        "output_root": "/second",
        "overwrite": True,
    }

    assert sero._protocol_fingerprint(first) == sero._protocol_fingerprint(second)
    second["stim_region_labels"] = ["Hippocampus_L"]
    assert sero._protocol_fingerprint(first) != sero._protocol_fingerprint(second)


def test_resolve_stim_regions_uses_dataset_label_order(tmp_path) -> None:
    labels = np.asarray([f"ROI_{i:02d}" for i in range(90)], dtype="U128")
    labels[9] = "Supp_Motor_Area_L"
    np.savez_compressed(
        tmp_path / "atlas.npz",
        labels=labels,
        region_codes=np.asarray([str(i + 1) for i in range(90)], dtype="U64"),
        region_indices=np.arange(1, 91, dtype=np.int32),
    )
    (tmp_path / "index.json").write_text(
        json.dumps({"atlas": {"ordering": "test_order"}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset_root=tmp_path,
        stim_region=None,
        stim_region_label=["Supp_Motor_Area_L"],
    )

    atlas = sero._resolve_stim_regions(args)

    assert atlas.ordering == "test_order"
    assert args.stim_region == [9]
    assert args.stim_region_label == ["Supp_Motor_Area_L"]


def test_split_model_is_parameterized_at_zero_occupancy() -> None:
    receptor_map = np.linspace(0.0, 1.0, 90)
    args = SimpleNamespace(
        e_l_e_drug=-61.2,
        e_l_i_drug=-64.4,
        split_model_all_occupancies=True,
    )

    parameter_model = sero._build_parameter_model("EMCS", 0.0, receptor_map, args)

    assert len(parameter_model["g_K_e"]) == 90
    assert len(parameter_model["g_K_i"]) == 90
    assert parameter_model["serotonergic_occupancy"] == 0.0


def test_legacy_zero_occupancy_model_remains_explicitly_available() -> None:
    receptor_map = np.linspace(0.0, 1.0, 90)
    args = SimpleNamespace(
        e_l_e_drug=-61.2,
        e_l_i_drug=-64.4,
        split_model_all_occupancies=False,
    )

    parameter_model = sero._build_parameter_model("EMCS", 0.0, receptor_map, args)

    assert "g_K_e" not in parameter_model
    assert "g_K_i" not in parameter_model


def test_common_b_e_override_removes_diagnosis_gradient() -> None:
    receptor_map = np.linspace(0.0, 1.0, 90)
    args = SimpleNamespace(
        e_l_e_drug=-61.2,
        e_l_i_drug=-64.4,
        split_model_all_occupancies=True,
        b_e_override=42.0,
    )

    control = sero._build_parameter_model("CNT", 0.0, receptor_map, args)
    coma = sero._build_parameter_model("COMA", 0.0, receptor_map, args)

    assert control["b_e"] == 42.0
    assert coma["b_e"] == 42.0
