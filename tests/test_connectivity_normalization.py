from __future__ import annotations

import numpy as np

from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.whole_brain.simulation import _build_connectivity
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import _apply_damage_parity
from tvbtoolkit.whole_brain.legacy_engine.parameter.parameter_M_Berlin_new import (
    Parameters,
)


def _config(mode: str) -> WholeBrainConfig:
    weights = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    return WholeBrainConfig(
        weights=weights,
        tract_lengths=np.ones((3, 3)),
        connectivity_normalization=mode,
    )


def test_none_preserves_supplied_weights_and_zeros() -> None:
    cfg = _config("none")
    connection = _build_connectivity(Parameters(), cfg)
    np.testing.assert_array_equal(connection.weights, cfg.weights)


def test_legacy_column_sum_remains_available_explicitly() -> None:
    cfg = _config("legacy_column_sum")
    connection = _build_connectivity(Parameters(), cfg)
    expected = cfg.weights / (cfg.weights.sum(axis=0) + 1e-12)
    np.testing.assert_allclose(connection.weights, expected)


def test_damage_parity_can_preserve_cohort_global_scale() -> None:
    weights = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.3], [0.0, 0.3, 0.0]])
    lengths = np.ones((3, 3))
    corrected, corrected_lengths, _ = _apply_damage_parity(
        weights, lengths, "uws", normalize_subject_max=False
    )
    np.testing.assert_array_equal(corrected, weights)
    assert corrected_lengths[0, 2] == 0.0
    assert corrected_lengths[2, 0] == 0.0
