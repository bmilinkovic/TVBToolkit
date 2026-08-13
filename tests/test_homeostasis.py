from __future__ import annotations

import numpy as np

from tvbtoolkit.whole_brain.homeostasis import (
    baseline_relative_activation_threshold,
    update_inhibitory_conductance,
)
from tvbtoolkit.whole_brain.legacy_engine.src.Zerlaut_gK_gNa import (
    Zerlaut_adaptation_first_order,
)


def test_inhibition_strengthens_above_target_and_weakens_below() -> None:
    updated = update_inhibitory_conductance(
        np.array([5.0, 5.0, 5.0]),
        np.array([0.010, 0.005, 0.002]),
        np.array([0.005, 0.005, 0.005]),
        np.array([0.005, 0.005, 0.005]),
        base_q_i_ns=5.0,
        epoch_ms=100.0,
    )
    assert updated[0] > 5.0
    assert updated[1] == 5.0
    assert updated[2] < 5.0


def test_homeostasis_is_gated_by_inhibitory_activity() -> None:
    updated = update_inhibitory_conductance(
        np.array([5.0]),
        np.array([0.020]),
        np.array([0.0]),
        np.array([0.005]),
        base_q_i_ns=5.0,
        epoch_ms=1000.0,
    )
    np.testing.assert_allclose(updated, [5.0])


def test_q_i_e_changes_excitatory_but_not_inhibitory_transfer() -> None:
    model = Zerlaut_adaptation_first_order()
    inputs = tuple(np.array([value]) for value in (0.005, 0.010, 0.002, 0.0, 0.0))
    excitatory_before = model.TF_excitatory(*inputs)
    inhibitory_before = model.TF_inhibitory(*inputs)
    model.Q_i_e = np.array([7.5])
    excitatory_after = model.TF_excitatory(*inputs)
    inhibitory_after = model.TF_inhibitory(*inputs)

    assert not np.allclose(excitatory_before, excitatory_after)
    np.testing.assert_allclose(inhibitory_before, inhibitory_after)


def test_relative_activation_threshold_is_regional_mean_plus_five_sd() -> None:
    baseline = np.array(
        [[0.004, 0.010], [0.005, 0.012], [0.006, 0.014]]
    )
    threshold = baseline_relative_activation_threshold(baseline, n_sd=5.0)
    expected = baseline.mean(axis=0) + 5.0 * baseline.std(axis=0, ddof=1)
    np.testing.assert_allclose(threshold, expected)


def test_relative_activation_threshold_rejects_invalid_input() -> None:
    with np.testing.assert_raises(ValueError):
        baseline_relative_activation_threshold(np.array([0.004, 0.005]))
