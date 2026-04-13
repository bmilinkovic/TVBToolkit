from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tvbtoolkit.brian_mf.mean_field.mf import calculate_mf_difference, run_mean_field_simulation
from tvbtoolkit.brian_mf.analysis import (
    compute_b_critical_grid,
    parse_np_arange_csv,
    run_dynamic_sweep,
)
from tvbtoolkit.brian_mf.adex import run_adex_network_simulation, run_single_cell_adex
from tvbtoolkit.brian_mf.mean_field.tf_calc import (
    TransferFunctionFitConfig,
    eff_thresh,
    fit_adex_transfer_function,
    get_neuron_params_double_cell,
    list_param_sets,
    load_param_set,
    mu_sig_tau_func,
    output_rate,
    save_param_set,
)
from tvbtoolkit.brian_mf.parity.compare import compare_mf_with_legacy, compare_subthreshold_with_legacy
from tvbtoolkit.brian_mf.parity.fixtures import fixed_fit_coefficients, fixed_rate_grid


def test_mu_sig_tau_shapes() -> None:
    params = get_neuron_params_double_cell("FS-RS", si_units=False)
    ve, vi, ff, adapt = fixed_rate_grid()
    vve, vvi = np.meshgrid(ve, vi)

    mu_v, sig_v, tau_v, tau_n_v = mu_sig_tau_func(vve, vvi, ff, adapt, params, "RS", w_prec=False)

    assert mu_v.shape == vve.shape
    assert sig_v.shape == vve.shape
    assert tau_v.shape == vve.shape
    assert tau_n_v.shape == vve.shape


def test_eff_thresh_and_output_rate_finite() -> None:
    params = np.array([-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407])
    mu_v = np.linspace(-0.07, -0.05, 20)
    sig_v = np.linspace(0.002, 0.006, 20)
    tau_n = np.linspace(0.2, 0.8, 20)
    tau_v = np.linspace(0.01, 0.03, 20)

    thr = eff_thresh(mu_v, sig_v, tau_n, params)
    rate = output_rate(params, mu_v, sig_v, tau_v, tau_n)

    assert np.all(np.isfinite(thr))
    assert np.all(np.isfinite(rate))
    assert np.all(rate >= 0)


def test_fit_adex_transfer_function_smoke() -> None:
    params = get_neuron_params_double_cell("FS-RS", si_units=False)
    ve, vi, ff, adapt = fixed_rate_grid()
    cfg = TransferFunctionFitConfig(loop_n=2, tf_maxiter=200, vthr_maxiter=200)

    res = fit_adex_transfer_function(
        ff,
        {
            "ve": ve,
            "vi": vi,
            "adapt": adapt,
            "params": params,
            "cell_type": "RS",
            "w_prec": False,
        },
        cfg,
    )

    assert res.fitted_params.shape == (10,)
    assert res.fit_rate.shape == ff.shape
    assert np.isfinite(res.rmse_hz)


def test_run_mean_field_seed_reproducible() -> None:
    prs, pfs = fixed_fit_coefficients()

    out1 = run_mean_field_simulation("FS-RS", 0.0, prs, pfs, iext_hz=0.3, total_time_s=0.5, seed=10)
    out2 = run_mean_field_simulation("FS-RS", 0.0, prs, pfs, iext_hz=0.3, total_time_s=0.5, seed=10)

    assert np.allclose(out1.exc_hz, out2.exc_hz)
    assert np.allclose(out1.inh_hz, out2.inh_hz)


def test_calculate_mf_difference_scalar() -> None:
    prs, pfs = fixed_fit_coefficients()
    fr_both = np.array(
        [
            [8.0, 6.0, 0.0],
            [9.0, 7.5, 0.5],
            [10.0, 8.5, 1.0],
        ]
    )
    inputs = fr_both[:, 2]

    diff = calculate_mf_difference("FS-RS", fr_both, inputs, prs, pfs)
    assert np.isfinite(diff)
    assert diff >= 0.0


@pytest.mark.skipif(
    not Path("/Users/borjan/CNRS/projects/TVBSim/brian_MF/Tf_calc/theoretical_tools.py").exists(),
    reason="Legacy brian_MF checkout not available",
)
def test_legacy_run_mf_parity_smoke() -> None:
    prs, pfs = fixed_fit_coefficients()
    report = compare_mf_with_legacy(prs, pfs, total_time_s=0.5, seed=77)

    # Loose tolerance for very short runtime smoke parity.
    assert report.abs_err_exc < 2.0
    assert report.abs_err_inh < 2.0


@pytest.mark.skipif(
    not Path("/Users/borjan/CNRS/projects/TVBSim/brian_MF/Tf_calc/theoretical_tools.py").exists(),
    reason="Legacy brian_MF checkout not available",
)
def test_legacy_subthreshold_parity() -> None:
    report = compare_subthreshold_with_legacy()
    assert report.max_abs_err_mu_v < 1e-12
    assert report.max_abs_err_sig_v < 1e-12
    assert report.max_abs_err_tau_v < 1e-12
    assert report.max_abs_err_tau_n_v < 1e-12


def test_param_db_roundtrip(tmp_path: Path) -> None:
    out = save_param_set(
        "unit_test_fit",
        {"P0": 0.1, "P1": -0.2},
        {
            "cell_type": "RS",
            "species": "test",
            "temperature": "test",
            "recording_condition": "test",
            "source": "pytest",
            "toolbox_version": "0.1.0",
            "date": "2026-02-20",
        },
        path=tmp_path,
    )
    assert out.exists()

    loaded = load_param_set("unit_test_fit", path=tmp_path)
    assert loaded.params["P0"] == 0.1

    listed = list_param_sets(path=tmp_path)
    assert any(r.name == "unit_test_fit" for r in listed)


def test_parse_np_arange_csv() -> None:
    assert np.allclose(parse_np_arange_csv("5"), np.array([5.0]))
    assert np.allclose(parse_np_arange_csv("0,1,0.5"), np.array([0.0, 0.5]))


def test_compute_b_critical_grid_smoke() -> None:
    prs, pfs = fixed_fit_coefficients()
    out = compute_b_critical_grid(
        prs,
        pfs,
        b_values_pa=np.array([0.0, 1.0]),
        tau_e_values_ms=np.array([5.0]),
        tau_i_values_ms=np.array([5.0]),
    )
    assert out.table.shape == (1, 3)


def test_single_cell_smoke() -> None:
    out = run_single_cell_adex(cell="RS", iext_na=0.0, time_ms=5.0)
    assert out.time_ms.ndim == 1
    assert out.voltage_mv.ndim == 1
    assert out.time_ms.size == out.voltage_mv.size


def test_single_cell_split_leak_smoke() -> None:
    out = run_single_cell_adex(cell="RS", iext_na=0.0, time_ms=5.0, split_leak=True, psych=False)
    assert out.split_leak is True
    assert out.gk_ns is not None
    assert out.gna_ns is not None


def test_network_smoke_small() -> None:
    out = run_adex_network_simulation(
        cells="FS-RS",
        seed_value=1,
        time_ms=20.0,
        iext_hz=2.0,
        input_hz=0.0,
        parameter_overrides={"use": True, "Ntot": 120, "p_con": 0.02},
    )
    assert out.time_ms.ndim == 1
    assert out.rate_exc_hz.shape == out.time_ms.shape
    assert out.rate_inh_hz.shape == out.time_ms.shape


def test_network_split_leak_smoke_small() -> None:
    out = run_adex_network_simulation(
        cells="FS-RS",
        seed_value=1,
        time_ms=20.0,
        iext_hz=2.0,
        input_hz=0.0,
        split_leak=True,
        psych=False,
        parameter_overrides={"use": True, "Ntot": 120, "p_con": 0.02},
    )
    assert out.split_leak is True
    assert out.rate_exc_hz.shape == out.time_ms.shape


def test_run_dynamic_sweep_smoke(tmp_path: Path) -> None:
    out = run_dynamic_sweep(
        b_e_range=np.array([0.0]),
        tau_e_range=np.array([5.0]),
        tau_i_range=np.array([5.0]),
        n_seeds=np.array([0]),
        time_ms=20.0,
        save_path=tmp_path / "dyn",
        overwrite=True,
        compute_survival=False,
        n_inh=20,
        n_exc=80,
    )
    assert out.saved_time_file.exists()
