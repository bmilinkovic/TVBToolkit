"""Regression checks for the saved-bank original per-trial PCI scorer."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "sma8s_full100"),
                str(ROOT / "scripts"), str(ROOT / "src")]
from compare_gaglioti_supplied_binJproc import supplied_binJproc_on_windows
import score_original_per_trial as scorer
from score_original_per_trial import original_trial_mask


def test_one_trial_cutoff_matches_supplied_gaglioti_bootstrap():
    rng = np.random.RandomState(149)
    baseline = rng.normal(0, 1, (90, 300))
    response = rng.normal(0, 1, (90, 300))
    response[9, 15:40] += 4.0
    epoch = np.concatenate((baseline, response), axis=1)
    seed = 44321
    actual, cutoff = original_trial_mask(epoch, seed)
    reference, reference_cutoffs, _, _, _ = supplied_binJproc_on_windows(
        epoch[None], n0=300, nboot=500, seed=seed)
    np.testing.assert_array_equal(actual, reference[.01])
    np.testing.assert_equal(cutoff, reference_cutoffs[.01])


def test_trial_specific_cutoff_uses_own_baseline():
    rng = np.random.RandomState(272)
    epoch = rng.normal(0, 1, (90, 600))
    first, first_cutoff = original_trial_mask(epoch, 123)
    changed = epoch.copy()
    changed[0, :300] += 100.0
    second, second_cutoff = original_trial_mask(changed, 123)
    assert not np.array_equal(first[0], second[0])
    assert np.all(second[0])


def test_case_saves_all_trial_masks_and_individual_thresholds(tmp_path):
    rng = np.random.RandomState(81)
    epochs = rng.normal(size=(100, 90, 600))
    metadata = [dict(seed=1000 + trial, onset_ms=4000.0,
                     peak_EI_hz=20.0, high_rate_above_150_hz=False,
                     longest_high_rate_run_ms=0.0,
                     peak_EI_post_transient_hz=20.0,
                     high_rate_post_transient_steps=0,
                     longest_high_rate_post_transient_run_ms=0.0)
                for trial in range(100)]
    labels = np.array([f"AAL_{index}" for index in range(90)])
    case = {"subject": "synthetic", "occupancy": 0.0}
    output = tmp_path / "original"
    with patch.object(scorer, "epochs_for", return_value=(epochs, metadata, labels, 9)):
        result = scorer.score_case((str(tmp_path / "bank"), str(output), 0, case))
    assert result["n_trials"] == 100
    assert result["threshold_z"] is None  # Each trial has its own cutoff.
    assert len(result["trials"]) == 100
    with np.load(output / "masks" / "case_0000.npz") as saved:
        assert saved["binary"].shape == (100, 90, 300)
        assert saved["threshold_z_by_trial"].shape == (100,)
        np.testing.assert_array_equal(saved["threshold_z_by_trial"],
                                      [row["threshold_z"] for row in result["trials"]])
