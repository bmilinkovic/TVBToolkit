from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.whole_brain.simulation import GaussianPulse, RaisedCosinePulse


@pytest.mark.parametrize("pulse_type", [RaisedCosinePulse, GaussianPulse])
def test_smooth_pulses_are_finite_supported_and_unit_peak(pulse_type) -> None:
    pulse = pulse_type()
    pulse.parameters.update({"onset": 10.0, "tau": 6.0, "amp": 1.0})
    time_ms = np.linspace(0.0, 25.0, 2501)
    waveform = np.asarray(pulse.evaluate(time_ms), dtype=float)

    assert np.all(waveform[time_ms < 10.0] == 0.0)
    assert np.all(waveform[time_ms > 16.0] == 0.0)
    assert 0.99 <= float(np.max(waveform)) <= 1.0 + 1e-12
    assert np.all(waveform >= 0.0)


def test_raised_cosine_has_smooth_endpoints() -> None:
    pulse = RaisedCosinePulse()
    pulse.parameters.update({"onset": 10.0, "tau": 6.0, "amp": 1.0})
    waveform = np.asarray(pulse.evaluate(np.array([10.0, 13.0, 16.0])))
    np.testing.assert_allclose(waveform, [0.0, 1.0, 0.0], atol=1e-12)
