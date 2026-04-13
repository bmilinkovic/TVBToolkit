"""Transfer-function fitting and theoretical tools for brian_MF parity.

Ported/adapted from legacy TVBSim modules:
- `brian_MF/Tf_calc/cell_library.py`
- `brian_MF/Tf_calc/syn_and_connec_library.py`
- `brian_MF/Tf_calc/theoretical_tools.py`

The objective is parity-first equations/defaults with a cleaner, reusable API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.special import erfc, erfcinv

from tvbtoolkit.brian_mf.io.storage import list_param_sets, load_param_set, save_param_set


def get_neuron_params_double_cell(name: str, si_units: bool = False) -> dict[str, float]:
    """Return legacy two-population AdEx parameter presets.

    Parameters
    ----------
    name
        Legacy preset name. Supported: ``FS-RS``, ``FS-RS_10``, ``FS-RS_faycal``,
        ``FS-RS_0``, ``FS``, ``RS``.
    si_units
        If ``True``, convert mV/ms/nS/pA/pF values to SI units.
    """

    if name == "FS-RS":
        params = {
            "V_m": -60,
            "V_r": -65,
            "Cm": 200,
            "Gl": 10,
            "tau_w": 500,
            "V_th": -50,
            "V_cut": -30,
            "a_e": 0,
            "b_e": 30,
            "delta_e": 2,
            "EL_e": -64,
            "a_i": 0,
            "b_i": 0,
            "delta_i": 0.5,
            "EL_i": -65,
            "tau_e": 5,
            "tau_i": 5,
            "E_e": 0,
            "E_i": -80,
            "Q_i": 5.0,
            "Q_e": 1.5,
            "p_con": 0.05,
            "gei": 0.2,
            "Ntot": 10000,
        }
    elif name == "FS-RS_10":
        params = {
            "V_m": -60,
            "V_r": -65,
            "Cm": 200,
            "Gl": 10,
            "tau_w": 500,
            "V_th": -50,
            "V_cut": -30,
            "a_e": 0,
            "b_e": 10,
            "delta_e": 2,
            "EL_e": -64,
            "a_i": 0,
            "b_i": 0,
            "delta_i": 0.5,
            "EL_i": -65,
            "tau_e": 5,
            "tau_i": 5,
            "E_e": 0,
            "E_i": -80,
            "Q_i": 5.0,
            "Q_e": 1.5,
            "p_con": 0.05,
            "gei": 0.2,
            "Ntot": 10000,
        }
    elif name == "FS-RS_faycal":
        params = {
            "V_m": -60,
            "V_r": -65,
            "Cm": 200,
            "Gl": 10,
            "tau_w": 500,
            "V_th": -50,
            "V_cut": -30,
            "a_e": 0,
            "b_e": 0,
            "delta_e": 2,
            "EL_e": -63,
            "a_i": 0,
            "b_i": 0,
            "delta_i": 0.5,
            "EL_i": -65,
            "tau_e": 5,
            "tau_i": 5,
            "E_e": 0,
            "E_i": -80,
            "Q_i": 5.0,
            "Q_e": 1.5,
            "p_con": 0.05,
            "gei": 0.2,
            "Ntot": 10000,
        }
    elif name == "FS-RS_0":
        params = {
            "V_m": -60,
            "V_r": -65,
            "Cm": 200,
            "Gl": 10,
            "tau_w": 500,
            "V_th": -50,
            "V_cut": -30,
            "a_e": 0,
            "b_e": 0,
            "delta_e": 2,
            "EL_e": -64,
            "a_i": 0,
            "b_i": 0,
            "delta_i": 0.5,
            "EL_i": -65,
            "tau_e": 5,
            "tau_i": 5,
            "E_e": 0,
            "E_i": -80,
            "Q_i": 5.0,
            "Q_e": 1.5,
            "p_con": 0.05,
            "gei": 0.2,
            "Ntot": 10000,
        }
    elif name == "FS":
        params = {
            "V_m": -60,
            "V_r": -65,
            "Cm": 200,
            "Gl": 10,
            "tau_w": 500,
            "V_th": -50,
            "V_cut": -30,
            "a": 0,
            "b": 0,
            "delta": 0.5,
            "EL": -65,
            "tau_e": 5,
            "tau_i": 5,
            "E_e": 0,
            "E_i": -80,
            "Q_i": 5.0,
            "Q_e": 1.5,
            "p_con": 0.05,
            "gei": 0.2,
            "Ntot": 10000,
        }
    elif name == "RS":
        params = {
            "V_m": -60,
            "V_r": -65,
            "Cm": 200,
            "Gl": 10,
            "tau_w": 500,
            "V_th": -50,
            "V_cut": -30,
            "a": 0,
            "b": 10,
            "delta": 2,
            "EL": -64,
            "tau_e": 5,
            "tau_i": 5,
            "E_e": 0,
            "E_i": -80,
            "Q_i": 5.0,
            "Q_e": 1.5,
            "p_con": 0.05,
            "gei": 0.2,
            "Ntot": 10000,
        }
    else:
        raise ValueError(f"Cell preset not recognised: {name}")

    if si_units:
        params = convert_params(params)
    return params.copy()


def get_neuron_params(name: str, *, sample_name: str = "", number: int = 1, si_units: bool = False) -> dict[str, float | str | int]:
    """Return legacy single-cell presets (`FS-cell` / `RS-cell`)."""

    if name == "FS-cell":
        params: dict[str, float | str | int] = {
            "name": sample_name,
            "N": number,
            "Gl": 10.0,
            "Cm": 200.0,
            "Trefrac": 5.0,
            "EL": -65.0,
            "V_th": -50.0,
            "V_r": -65.0,
            "delta": 0.5,
            "a": 0.0,
            "b": 0.0,
            "tau_w": 500.0,
        }
    elif name == "RS-cell":
        params = {
            "name": sample_name,
            "N": number,
            "Gl": 10.0,
            "Cm": 200.0,
            "Trefrac": 5.0,
            "EL": -64.0,
            "V_th": -50.0,
            "V_r": -65.0,
            "delta": 2.0,
            "a": 0.0,
            "b": 20.0,
            "tau_w": 500.0,
        }
    else:
        raise ValueError(f"Cell preset not recognised: {name}")

    if si_units:
        params = params.copy()
        params["EL"] *= 1e-3
        params["V_th"] *= 1e-3
        params["V_r"] *= 1e-3
        params["delta"] *= 1e-3
        params["Trefrac"] *= 1e-3
        params["tau_w"] *= 1e-3
        params["a"] *= 1e-9
        params["Gl"] *= 1e-9
        params["Cm"] *= 1e-12
        params["b"] *= 1e-12
    return params


def get_connectivity_and_synapses_matrix(name: str, number: int = 2, si_units: bool = False) -> np.ndarray:
    """Return legacy synaptic/connectivity matrix configuration."""

    mat = np.empty((number, number), dtype=object)
    if name != "CONFIG1":
        raise ValueError(f"Network preset not recognised: {name}")

    exc_pop = {"p_conn": 0.05, "Q": 1.5, "Tsyn": 5.0, "Erev": 0.0}
    inh_pop = {"p_conn": 0.05, "Q": 5.0, "Tsyn": 5.0, "Erev": -80.0}

    mat[:, 0] = [exc_pop.copy(), inh_pop.copy()]
    mat[:, 1] = [exc_pop.copy(), inh_pop.copy()]
    mat[0, 0]["name"], mat[1, 0]["name"] = "ee", "ie"
    mat[0, 1]["name"], mat[1, 1]["name"] = "ei", "ii"

    mat[0, 0]["Ntot"], mat[0, 0]["gei"] = 10000, 0.2
    mat[0, 0]["ext_drive"] = 4.0
    mat[0, 0]["afferent_exc_fraction"] = 1.0

    if si_units:
        for item in mat.flatten():
            item["Q"] *= 1e-9
            item["Erev"] *= 1e-3
            item["Tsyn"] *= 1e-3
    return mat


def convert_params(params: dict[str, float]) -> dict[str, float]:
    """Convert legacy parameter dict to SI units (parity function)."""

    out = params.copy()

    for key in ("EL_e", "EL_i", "E_e", "E_i"):
        if key in out:
            out[key] *= 1e-3
    for key in ("V_th", "V_r", "V_m", "V_cut"):
        if key in out:
            out[key] *= 1e-3
    for key in ("delta_e", "delta_i"):
        if key in out:
            out[key] *= 1e-3

    for key in ("tau_w", "tau_e", "tau_i"):
        if key in out:
            out[key] *= 1e-3

    for key in ("a_e", "a_i", "Q_e", "Q_i", "Gl"):
        if key in out:
            out[key] *= 1e-9

    for key in ("Cm", "b_e", "b_i"):
        if key in out:
            out[key] *= 1e-12

    return out


def eff_thresh(mu_v: np.ndarray, sig_v: np.ndarray, tau_n_v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Effective threshold polynomial model (legacy Casali/Di Volo form)."""

    p0, p_mu, p_sig, p_tau, p_mu2, p_sig2, p_tau2, p_mu_sig, p_mu_tau, p_sig_tau = params
    v0 = p0
    mu_0 = -60.0e-3
    mu_d = 0.01
    sig_0 = 0.004
    sig_d = 0.006
    tau_0 = 0.5
    tau_d = 1.0

    v1 = (
        p_mu * (mu_v - mu_0) / mu_d
        + p_sig * (sig_v - sig_0) / sig_d
        + p_tau * (tau_n_v - tau_0) / tau_d
    )
    v2 = (
        p_mu2 * ((mu_v - mu_0) / mu_d) ** 2
        + p_sig2 * ((sig_v - sig_0) / sig_d) ** 2
        + p_tau2 * ((tau_n_v - tau_0) / tau_d) ** 2
        + p_mu_sig * ((mu_v - mu_0) / mu_d) * ((sig_v - sig_0) / sig_d)
        + p_mu_tau * ((mu_v - mu_0) / mu_d) * ((tau_n_v - tau_0) / tau_d)
        + p_sig_tau * ((sig_v - sig_0) / sig_d) * ((tau_n_v - tau_0) / tau_d)
    )
    return v0 + v1 + v2


def mu_sig_tau_func(
    fexc: np.ndarray,
    finh: np.ndarray,
    fout: np.ndarray,
    w_ad: np.ndarray,
    params: dict[str, float],
    cell_type: str,
    w_prec: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute subthreshold moments for TF fitting (legacy equations).

    Parameters
    ----------
    fexc, finh
        Excitatory/inhibitory input rates in Hz.
    fout
        Output firing rate in Hz.
    w_ad
        Adaptation term in ampere (if ``w_prec=True``).
    params
        Non-SI legacy parameter dict (same structure as `cell_library.py`).
    cell_type
        ``"RS"`` or ``"FS"``.
    w_prec
        If ``True``, use ``w_ad`` directly; else use ``tau_w * b * fout``.
    """

    p = params
    q_e, q_i = p["Q_e"] * 1e-9, p["Q_i"] * 1e-9
    tau_e, tau_i = p["tau_e"] * 1e-3, p["tau_i"] * 1e-3
    e_e, e_i = p["E_e"] * 1e-3, p["E_i"] * 1e-3
    c_m, tw, g_l = p["Cm"] * 1e-12, p["tau_w"] * 1e-3, p["Gl"] * 1e-9
    gei, ntot, pconnec = p["gei"], p["Ntot"], p["p_con"]

    if cell_type == "RS":
        try:
            a, b, e_l = p["a_e"] * 1e-9, p["b_e"] * 1e-12, p["EL_e"] * 1e-3
        except KeyError:
            a, b, e_l = p["a"] * 1e-9, p["b"] * 1e-12, p["EL"] * 1e-3
    elif cell_type == "FS":
        try:
            a, b, e_l = p["a_i"] * 1e-9, p["b_i"] * 1e-12, p["EL_i"] * 1e-3
        except KeyError:
            a, b, e_l = p["a"] * 1e-9, p["b"] * 1e-12, p["EL"] * 1e-3
    else:
        raise ValueError("cell_type must be 'RS' or 'FS'")

    f_e = fexc * (1.0 - gei) * pconnec * ntot
    f_i = finh * gei * pconnec * ntot

    mu_g_e = f_e * tau_e * q_e
    mu_g_i = f_i * tau_i * q_i
    mu_g = mu_g_e + mu_g_i + g_l
    tau_eff = c_m / mu_g

    if w_prec:
        mu_v = (mu_g_e * e_e + mu_g_i * e_i + g_l * e_l - w_ad) / mu_g
    else:
        mu_v = (mu_g_e * e_e + mu_g_i * e_i + g_l * e_l - fout * tw * b + a * e_l) / mu_g

    u_e = q_e / mu_g * (e_e - mu_v)
    u_i = q_i / mu_g * (e_i - mu_v)

    sig_v = np.sqrt(
        f_e * (u_e * tau_e) * (u_e * tau_e) / (2.0 * (tau_eff + tau_e))
        + f_i * (u_i * tau_i) * (u_i * tau_i) / (2.0 * (tau_eff + tau_i))
    )

    tau_v = (
        f_e * (u_e * tau_e) * (u_e * tau_e)
        + f_i * (u_i * tau_i) * (u_i * tau_i)
    ) / (
        f_e * (u_e * tau_e) * (u_e * tau_e) / (tau_eff + tau_e)
        + f_i * (u_i * tau_i) * (u_i * tau_i) / (tau_eff + tau_i)
    )

    tau_n_v = tau_v * g_l / c_m
    return mu_v, sig_v, tau_v, tau_n_v


def output_rate(params: np.ndarray, mu_v: np.ndarray, sig_v: np.ndarray, tau_v: np.ndarray, tau_n_v: np.ndarray) -> np.ndarray:
    """Predict firing rate from fitted threshold parameters."""

    return erfc((eff_thresh(mu_v, sig_v, tau_n_v, params) - mu_v) / (np.sqrt(2) * sig_v)) / (2 * tau_v)


def eff_thresh_estimate(ydata: np.ndarray, mu_v: np.ndarray, sig_v: np.ndarray, tau_v: np.ndarray) -> np.ndarray:
    """Estimate effective threshold from empirical output rates."""

    return mu_v + np.sqrt(2) * sig_v * erfcinv(ydata * 2 * tau_v)


def get_rid_of_nans(
    vve: np.ndarray,
    vvi: np.ndarray,
    adapt: np.ndarray,
    ff: np.ndarray,
    params: dict[str, float],
    cell_type: str,
    return_index: bool = False,
    w_prec: bool = False,
):
    """Flatten and remove NaN/Inf points exactly as in legacy script."""

    ve2 = vve.flatten()
    vi2 = vvi.flatten()
    ff2 = ff.flatten()
    adapt2 = adapt.flatten()

    mu_v2, s_v2, t_v2, _ = mu_sig_tau_func(ve2, vi2, ff2, adapt2, params, cell_type, w_prec=w_prec)
    veff = eff_thresh_estimate(ff2, mu_v2, s_v2, t_v2)

    nanindex = np.where(np.isnan(veff))
    infindex = np.where(np.isinf(veff))
    bad = np.concatenate([nanindex, infindex], axis=1)

    ve2 = np.delete(ve2, bad)
    vi2 = np.delete(vi2, bad)
    ff2 = np.delete(ff2, bad)
    adapt2 = np.delete(adapt2, bad)

    if return_index:
        return ve2, vi2, ff2, adapt2, bad
    return ve2, vi2, ff2, adapt2


def find_max_error(out_rate: np.ndarray, fit_rate: np.ndarray, ve: np.ndarray, vi: np.ndarray, window: int = 12, thresh_pc: float = 0.9) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find a local high-error rectangle and return corresponding ve/vi ranges."""

    error = np.sqrt((out_rate - fit_rate) ** 2).T

    if window > len(ve) / 3:
        window = int(len(ve) / 3)
    rect_size = window

    local_max = maximum_filter(error, size=rect_size)
    max_indices = np.argwhere(local_max == error)

    all_errors = []
    for i, j in max_indices:
        mean_error = np.nanmean(
            error[
                max(0, i - rect_size) : min(error.shape[0], i + rect_size + 1),
                max(0, j - rect_size) : min(error.shape[1], j + rect_size + 1),
            ]
        )
        rect = (
            max(0, i - rect_size),
            max(0, j - rect_size),
            min(error.shape[0], i + rect_size + 1),
            min(error.shape[1], j + rect_size + 1),
        )
        all_errors.append([mean_error, sum(rect), rect])

    all_errors = np.array(all_errors, dtype=object)
    all_errors = all_errors[np.argsort(all_errors[:, 1], kind="mergesort")]

    thresh = np.max(all_errors[:, 0]) * thresh_pc
    max_rect = all_errors[-1, 2]
    for i in range(all_errors.shape[0]):
        if all_errors[i, 0] > thresh:
            max_rect = all_errors[i, 2]
            break

    x_start, y_start, x_end, y_end = max_rect
    range_exc = (ve[y_start], ve[y_end - 1])
    range_inh = (vi[x_start], vi[x_end - 1])
    return range_exc, range_inh


def adjust_ranges(
    ve: np.ndarray,
    vi: np.ndarray,
    ff: np.ndarray,
    adapt: np.ndarray,
    params: dict[str, float],
    cell_type: str,
    range_inh: tuple[float, float] | None,
    range_exc: tuple[float, float] | None,
    w_prec: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restrict fitting grid to selected ve/vi windows (legacy logic)."""

    vve, vvi = np.meshgrid(ve, vi)

    rid: list[int] | None = None
    red: list[int] | None = None
    add_vals = [0, 1, 3, 5, -3, -5, -1, -2]

    if range_inh:
        st, en = range_inh
        start, end = np.argmin(np.abs(vi - st)), np.argmin(np.abs(vi - en))
        rid = list(add_vals) + list(range(start, end))

    if range_exc:
        st, en = range_exc
        start, end = np.argmin(np.abs(ve - st)), np.argmin(np.abs(ve - en))
        red = list(add_vals) + list(range(start, end))

    if rid is not None and red is not None:
        ff_local = ff[red][:, rid]
        ve2 = vve[red][:, rid].flatten()
        vi2 = vvi[red][:, rid].flatten()
        ff2 = ff_local.flatten()
        adapt2 = adapt[red][:, rid].flatten()
    elif rid is not None:
        ff_local = ff[:, rid]
        ve2 = vve[:, rid].flatten()
        vi2 = vvi[:, rid].flatten()
        ff2 = ff_local.flatten()
        adapt2 = adapt[:, rid].flatten()
    elif red is not None:
        ff_local = ff[red]
        ve2 = vve[red].flatten()
        vi2 = vvi[red].flatten()
        ff2 = ff_local.flatten()
        adapt2 = adapt[red].flatten()
    else:
        ve2 = vve.flatten()
        vi2 = vvi.flatten()
        ff2 = ff.flatten()
        adapt2 = adapt.flatten()

    mu_v, sig_v, tau_v, tau_n_v = mu_sig_tau_func(ve2, vi2, ff2, adapt2, params, cell_type, w_prec=w_prec)
    return mu_v, sig_v, tau_v, tau_n_v, ff2


@dataclass(frozen=True)
class TransferFunctionFitConfig:
    """Configuration for TF fitting loops.

    Parameters mirror the legacy fitting options in `theoretical_tools.py`.
    """

    loop_n: int = 10
    window: int = 12
    thresh_pc: float = 0.9
    vthr_tol: float = 1e-17
    vthr_maxiter: int = 30000
    vthr_method: str = "SLSQP"
    tf_tol: float = 1e-17
    tf_maxiter: int = 30000
    tf_method: str = "nelder-mead"
    seed: int = 10


@dataclass(frozen=True)
class TransferFunctionFitResult:
    """Result object for fitted AdEx transfer function."""

    fitted_params: np.ndarray
    fit_rate: np.ndarray
    rmse_hz: float
    ranges_exc_hz: tuple[float, float] | None
    ranges_inh_hz: tuple[float, float] | None
    diagnostics: dict[str, Any]


def fit_adex_transfer_function(
    empirical_data: np.ndarray,
    model_cfg: dict[str, Any],
    fit_cfg: TransferFunctionFitConfig,
) -> TransferFunctionFitResult:
    """Fit TF coefficients on in-memory empirical transfer-function data.

    Parameters
    ----------
    empirical_data
        Empirical output rates with shape ``(n_vi, n_ve)``.
    model_cfg
        Dict containing:
        - ``ve``: excitatory input grid (Hz)
        - ``vi``: inhibitory input grid (Hz)
        - ``adapt``: adaptation array, same shape as empirical data
        - ``params``: legacy parameter dictionary (non-SI)
        - ``cell_type``: ``"RS"`` or ``"FS"``
        Optional: ``w_prec`` and initial fit ranges.
    fit_cfg
        Optimization configuration.
    """

    ff = np.asarray(empirical_data, dtype=float)
    ve = np.asarray(model_cfg["ve"], dtype=float)
    vi = np.asarray(model_cfg["vi"], dtype=float)
    adapt = np.asarray(model_cfg["adapt"], dtype=float)
    params = dict(model_cfg["params"])
    cell_type = str(model_cfg["cell_type"])
    w_prec = bool(model_cfg.get("w_prec", False))

    vve, vvi = np.meshgrid(ve, vi)
    ve2, vi2, ff2, adapt2 = get_rid_of_nans(vve, vvi, adapt, ff, params, cell_type, w_prec=w_prec)

    mu_v, sig_v, tau_v, tau_n_v = mu_sig_tau_func(ve2, vi2, ff2, adapt2, params, cell_type, w_prec=w_prec)
    v_eff = eff_thresh_estimate(ff2, mu_v, sig_v, tau_v)

    p_init = np.ones(10) * 1e-3

    def res_vthr(p: np.ndarray) -> float:
        return float(np.mean((v_eff - eff_thresh(mu_v, sig_v, tau_n_v, p)) ** 2))

    fit_v = minimize(
        res_vthr,
        p_init,
        method=fit_cfg.vthr_method,
        tol=fit_cfg.vthr_tol,
        options={"disp": False, "maxiter": fit_cfg.vthr_maxiter},
    )

    p_current = fit_v.x
    range_exc = model_cfg.get("range_exc")
    range_inh = model_cfg.get("range_inh")

    history: list[dict[str, Any]] = []
    for _ in range(fit_cfg.loop_n):
        mu_l, sig_l, tau_l, tau_n_l, ff_l = adjust_ranges(
            ve,
            vi,
            ff,
            adapt,
            params,
            cell_type,
            range_inh=range_inh,
            range_exc=range_exc,
            w_prec=w_prec,
        )

        def res_tf(p: np.ndarray) -> float:
            pred = output_rate(p, mu_l, sig_l, tau_l, tau_n_l)
            return float(np.mean((pred - ff_l) ** 2))

        fit_tf = minimize(
            res_tf,
            p_current,
            method=fit_cfg.tf_method,
            tol=fit_cfg.tf_tol,
            options={"disp": False, "maxiter": fit_cfg.tf_maxiter},
        )
        p_current = fit_tf.x

        mu_full, sig_full, tau_full, tau_n_full = mu_sig_tau_func(vve, vvi, ff, adapt, params, cell_type, w_prec=w_prec)
        fit_rate = output_rate(p_current, mu_full, sig_full, tau_full, tau_n_full)
        rmse = float(np.sqrt(np.mean((fit_rate - ff) ** 2)))

        range_exc, range_inh = find_max_error(ff, fit_rate, ve, vi, window=fit_cfg.window, thresh_pc=fit_cfg.thresh_pc)
        history.append(
            {
                "rmse": rmse,
                "range_exc": range_exc,
                "range_inh": range_inh,
                "success": bool(fit_tf.success),
                "message": str(fit_tf.message),
            }
        )

    mu_full, sig_full, tau_full, tau_n_full = mu_sig_tau_func(vve, vvi, ff, adapt, params, cell_type, w_prec=w_prec)
    fit_rate_final = output_rate(p_current, mu_full, sig_full, tau_full, tau_n_full)
    rmse_final = float(np.sqrt(np.mean((fit_rate_final - ff) ** 2)))

    return TransferFunctionFitResult(
        fitted_params=p_current,
        fit_rate=fit_rate_final,
        rmse_hz=rmse_final,
        ranges_exc_hz=range_exc,
        ranges_inh_hz=range_inh,
        diagnostics={
            "history": history,
            "vthreshold_success": bool(fit_v.success),
            "vthreshold_message": str(fit_v.message),
        },
    )


def make_fit_from_data(
    data_file: str | Path,
    cell_type: str,
    params_file: str | Path,
    adapt_file: str | Path,
    range_exc: tuple[float, float] | None = None,
    range_inh: tuple[float, float] | None = None,
    w_prec: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Legacy-compatible file-based TF fitting wrapper.

    This reproduces the behavior of `theoretical_tools.make_fit_from_data` while
    exposing it as a callable function in TVBToolkit.
    """

    cfg = TransferFunctionFitConfig(
        loop_n=int(kwargs.get("loop_n", 10)),
        window=int(kwargs.get("window", 12)),
        thresh_pc=float(kwargs.get("thresh_pc", 0.9)),
        vthr_tol=float(kwargs.get("vthr_tol", 1e-17)),
        vthr_maxiter=int(kwargs.get("vtrh_maxiter", 30000)),
        vthr_method=str(kwargs.get("vthr_method", "SLSQP")),
        tf_tol=float(kwargs.get("tf_tol", 1e-17)),
        tf_maxiter=int(kwargs.get("tf_maxiter", 30000)),
        tf_method=str(kwargs.get("tf_method", "nelder-mead")),
        seed=int(kwargs.get("seed", 10)),
    )

    ff = np.load(data_file).T
    adapt = np.load(adapt_file).T
    ve, vi, params = np.load(params_file, allow_pickle=True)

    result = fit_adex_transfer_function(
        ff,
        {
            "ve": ve,
            "vi": vi,
            "adapt": adapt,
            "params": params,
            "cell_type": cell_type,
            "w_prec": w_prec,
            "range_exc": range_exc,
            "range_inh": range_inh,
        },
        cfg,
    )

    out = Path(str(data_file).replace("ExpTF_", "").replace(".npy", "_fit.npy"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(result.fitted_params))
    return result.fitted_params


__all__ = [
    "TransferFunctionFitConfig",
    "TransferFunctionFitResult",
    "fit_adex_transfer_function",
    "make_fit_from_data",
    "get_neuron_params_double_cell",
    "get_neuron_params",
    "get_connectivity_and_synapses_matrix",
    "convert_params",
    "eff_thresh",
    "mu_sig_tau_func",
    "output_rate",
    "eff_thresh_estimate",
    "get_rid_of_nans",
    "find_max_error",
    "adjust_ranges",
    "save_param_set",
    "load_param_set",
    "list_param_sets",
]
