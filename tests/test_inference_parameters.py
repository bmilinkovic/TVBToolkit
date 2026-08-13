from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.inference.parameters import AdExPrior


def test_default_adex_prior_maps_requested_parameters_without_mutating_base() -> None:
    prior = AdExPrior.default()
    base = WholeBrainConfig(
        coupling_strength=0.3,
        conduction_speed=4.0,
        parameter_overrides={"parameter_model": {"T": 20.0}},
    )
    theta = np.array([60.0, 0.25, 8.0, 1.2e-4])

    mapped = prior.apply(base, theta)

    assert prior.names == (
        "adaptation_b_e",
        "global_coupling",
        "conduction_speed",
        "noise_amplitude",
    )
    assert mapped.coupling_strength == pytest.approx(0.25)
    assert mapped.conduction_speed == pytest.approx(8.0)
    assert mapped.parameter_overrides["parameter_model"]["b_e"] == pytest.approx(60.0)
    assert mapped.parameter_overrides["parameter_model"]["weight_noise"] == pytest.approx(1.2e-4)
    assert mapped.parameter_overrides["parameter_model"]["T"] == pytest.approx(20.0)
    assert base.coupling_strength == pytest.approx(0.3)
    assert "b_e" not in base.parameter_overrides["parameter_model"]


def test_adex_prior_sampling_is_reproducible_and_bounded() -> None:
    prior = AdExPrior.default(include_external_drive=True)
    first = prior.sample(20, seed=12)
    second = prior.sample(20, seed=12)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 5)
    assert np.all(first >= prior.low)
    assert np.all(first <= prior.high)


def test_adex_prior_rejects_wrong_or_out_of_range_theta() -> None:
    prior = AdExPrior.default()
    with pytest.raises(ValueError, match="expects 4"):
        prior.as_dict(np.ones(3))
    with pytest.raises(ValueError, match="outside"):
        prior.as_dict(np.array([200.0, 0.2, 4.0, 1e-4]))
