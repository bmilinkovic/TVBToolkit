from __future__ import annotations

from pathlib import Path

import numpy as np

from tvbtoolkit.bold import (
    BOLDParams,
    bold_from_firing_rates,
    butter_filtering,
    corr_fc_sc,
    preprocess_bold_signal,
)


def _load_tvbsim_bold_namespace() -> dict:
    """Load TVBSim BOLD.py in a lightweight namespace for parity checks."""
    ref_path = Path("/Users/borjan/CNRS/projects/TVBSim/tvbsim/BOLD.py")
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    src = ref_path.read_text(encoding="utf-8")
    # Not needed for these tests and may pull extra dependencies.
    src = src.replace("from common import create_dicts\n", "")
    ns: dict = {}
    exec(compile(src, str(ref_path), "exec"), ns)  # noqa: S102 - controlled local reference file.
    return ns


def test_butter_filtering_matches_tvbsim_reference() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1800, 16))
    bp = BOLDParams(TR=2.0, n_order=2, low_f_num=0.01, high_f_num=0.1)

    ref = _load_tvbsim_bold_namespace()
    bp_ref = ref["BOLDParams"](TR=bp.TR, n_order=bp.n_order, low_f_num=bp.low_f_num, high_f_num=bp.high_f_num)
    y_ref = ref["butter_filtering"](x.copy(), bp_ref)
    y_new = butter_filtering(x.copy(), bp)

    assert y_ref.shape == y_new.shape
    assert np.allclose(y_ref, y_new, atol=1e-12, rtol=1e-12)


def test_corr_fc_sc_matches_tvbsim_reference() -> None:
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(12, 800))  # (regions, time)
    a = rng.uniform(0.0, 1.0, size=(12, 12))
    sc = (a + a.T) / 2.0
    np.fill_diagonal(sc, 0.0)

    ref = _load_tvbsim_bold_namespace()
    fc_ref, coef_ref = ref["corr_FC_SC"](signal.copy(), sc.copy())
    fc_new, coef_new = corr_fc_sc(signal.copy(), sc.copy())

    assert np.allclose(fc_ref, fc_new, atol=1e-12, rtol=1e-12)
    assert np.isclose(coef_ref, coef_new, atol=1e-12, rtol=1e-12)


def test_preprocess_bold_signal_preserves_shape() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1200, 10))
    y = preprocess_bold_signal(x, params=BOLDParams(), apply_zscore=True, apply_bandpass=True)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_preprocess_bold_signal_short_input_uses_stable_filtering() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=(12, 6))
    y = preprocess_bold_signal(x, params=BOLDParams(TR=1.0), apply_zscore=True, apply_bandpass=True)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_bold_from_firing_rates_multiregion_and_orientation() -> None:
    rng = np.random.default_rng(13)
    rates = rng.normal(size=(5000, 8))

    bold_a = bold_from_firing_rates(rates, dt_ms=1.0, tr_ms=2000.0)
    bold_b = bold_from_firing_rates(rates.T, dt_ms=1.0, tr_ms=2000.0)

    assert bold_a.ndim == 2
    assert bold_a.shape[1] == 8
    assert np.all(np.isfinite(bold_a))
    assert np.allclose(bold_a, bold_b, atol=1e-10, rtol=1e-10)


def test_bold_from_firing_rates_constant_input_and_determinism() -> None:
    rates = np.ones((6000, 4), dtype=float)

    out1 = bold_from_firing_rates(rates, dt_ms=1.0, tr_ms=1000.0)
    out2 = bold_from_firing_rates(rates, dt_ms=1.0, tr_ms=1000.0)

    assert out1.shape == out2.shape
    assert np.all(np.isfinite(out1))
    assert np.allclose(out1, out2, atol=0.0, rtol=0.0)


def test_bold_from_firing_rates_raises_for_too_short_signal() -> None:
    short = np.ones((4, 3), dtype=float)
    try:
        bold_from_firing_rates(short, dt_ms=1.0, tr_ms=2000.0, interim_period_ms=4.0)
    except ValueError as exc:
        assert "too short" in str(exc).lower() or "need" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for short signal.")
