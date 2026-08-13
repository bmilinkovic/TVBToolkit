from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.inference.adex import AdExBOLDSimulator, extract_bold_monitor
from tvbtoolkit.inference.features import BOLDFeatureConfig, BOLDFeatureExtractor
from tvbtoolkit.inference.parameters import AdExPrior
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation


def test_extract_bold_monitor_selects_expected_period() -> None:
    rate_t = np.arange(10, dtype=float)
    rate = np.zeros((10, 1, 3, 1))
    bold_t = np.arange(4, dtype=float) * 2400.0
    bold = np.ones((4, 1, 3, 1))

    time, signal = extract_bold_monitor(
        [(rate_t, rate), (bold_t, bold)], expected_period_ms=2400.0
    )

    np.testing.assert_array_equal(time, bold_t)
    assert signal.shape == (4, 3)


def test_adex_simulator_requires_tract_lengths_when_speed_is_in_prior() -> None:
    rng = np.random.default_rng(0)
    observed = rng.normal(size=(80, 3))
    extractor = BOLDFeatureExtractor(
        BOLDFeatureConfig(include_fc=True, include_fcd=False, include_states=False)
    )
    extractor.fit(observed)
    config = WholeBrainConfig(
        simulation_length_ms=1000.0,
        weights=np.ones((3, 3)) - np.eye(3),
        tract_lengths=None,
    )

    with pytest.raises(ValueError, match="Speed would be non-identifiable"):
        AdExBOLDSimulator(config, AdExPrior.default(), extractor, transient_ms=0.0)


def test_adex_whole_brain_run_emits_configured_bold_monitor() -> None:
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    lengths = np.array([[0.0, 10.0], [10.0, 0.0]])
    config = WholeBrainConfig(
        simulation_length_ms=500.0,
        dt_ms=0.2,
        zerlaut_order=1,
        stochastic_integrator=False,
        monitor_mode="temporal_average",
        temporal_average_period_ms=1.0,
        include_bold_monitor=True,
        bold_monitor_period_ms=100.0,
        weights=weights,
        tract_lengths=lengths,
        parameter_overrides={
            "parameter_connection_between_region": {"normalised": False}
        },
    )

    result = run_whole_brain_simulation(config, seed=1)
    time, bold = extract_bold_monitor(
        result.full_monitor_output, expected_period_ms=100.0
    )

    assert time.shape == (5,)
    assert bold.shape == (5, 2)
    assert np.isfinite(bold).all()
