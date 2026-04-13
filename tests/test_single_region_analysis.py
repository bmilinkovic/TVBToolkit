from __future__ import annotations

import numpy as np

from tvbtoolkit.single_region.analysis import (
    bin_array,
    calculate_psd_fmax,
    heaviside,
    input_rate,
    prepare_population_rates,
)


def test_bin_array_reduces_length() -> None:
    t = np.arange(0.0, 100.0, 1.0)
    x = np.sin(2 * np.pi * 0.05 * t)
    b = bin_array(x, 5.0, t)
    assert b.ndim == 1
    assert b.size > 0
    assert b.size < x.size


def test_heaviside_shape() -> None:
    x = np.array([-1.0, 0.0, 2.0])
    y = heaviside(x)
    assert y.shape == x.shape
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_input_rate_output_is_finite() -> None:
    t = np.linspace(0.0, 2000.0, 2001)
    y = input_rate(t, t1_exc=500.0, tau1_exc=40.0, tau2_exc=80.0, ampl_exc=5.0, plateau=200.0)
    assert y.shape == t.shape
    assert np.all(np.isfinite(y))


def test_prepare_population_rates_shapes() -> None:
    total_time = 1000.0
    dt = 1.0
    t = np.arange(0.0, total_time, dt)
    exc = np.sin(2 * np.pi * 0.01 * t)
    inh = np.cos(2 * np.pi * 0.01 * t)
    w = np.ones_like(t)
    tb, eb, ib, wb = prepare_population_rates(total_time, dt, exc, inh, w, bin_width=5.0)
    assert tb.size == eb.size == ib.size == wb.size


def test_calculate_psd_fmax_returns_positive_frequency() -> None:
    t = np.arange(0.0, 4000.0, 5.0)
    exc = np.sin(2 * np.pi * 10.0 * (t / 1000.0))
    inh = np.sin(2 * np.pi * 10.0 * (t / 1000.0) + 0.3)
    fmax, frq, pexc, pinh = calculate_psd_fmax(exc, inh, t)
    assert fmax > 0.0
    assert frq.ndim == pexc.ndim == pinh.ndim == 1
    assert frq.size == pexc.size == pinh.size
