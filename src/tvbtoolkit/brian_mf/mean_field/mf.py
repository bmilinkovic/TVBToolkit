"""Mean-field AdEx routines ported from legacy `brian_MF`.

Ported/adapted from:
- `brian_MF/MF.py`
- `brian_MF/Tf_calc/theoretical_tools.py` (`run_MF`, `OU`, transfer function core)

The implementation below keeps the same equations/defaults while exposing a
cleaner API and deterministic seeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from typing import Any

import numpy as np

from tvbtoolkit.brian_mf.adex.brian_utils import input_rate
from tvbtoolkit.brian_mf.mean_field.tf_calc import get_neuron_params_double_cell


@dataclass(frozen=True)
class MeanFieldResult:
    """Result object for mean-field simulations.

    Parameters
    ----------
    time_s
        Simulation time axis in seconds, shape ``(T,)``.
    exc_hz
        Excitatory mean-field firing-rate trajectory in Hz, shape ``(T,)``.
    inh_hz
        Inhibitory mean-field firing-rate trajectory in Hz, shape ``(T,)``.
    adaptation_a
        Adaptation trajectory in ampere, shape ``(T,)``.
    mean_exc_hz
        Mean excitatory firing rate over the second half of the simulation.
    mean_inh_hz
        Mean inhibitory firing rate over the second half of the simulation.
    """

    time_s: np.ndarray
    exc_hz: np.ndarray
    inh_hz: np.ndarray
    adaptation_a: np.ndarray
    mean_exc_hz: float
    mean_inh_hz: float


def _legacy_transfer_function(
    p: np.ndarray,
    fexc_hz: float,
    finh_hz: float,
    adapt_a: float,
    e_l_v: float,
    *,
    gei: float,
    pconnec: float,
    ntot: float,
    qi_s: float,
    qe_s: float,
    ti_s: float,
    te_s: float,
    ee_v: float,
    ei_v: float,
    gl_s: float,
    cm_f: float,
) -> float:
    """Legacy Di Volo transfer-function formula used in brian_MF scripts."""

    fe = fexc_hz * (1.0 - gei) * pconnec * ntot
    fi = finh_hz * gei * pconnec * ntot

    mu_gi = qi_s * ti_s * fi
    mu_ge = qe_s * te_s * fe
    mu_g = gl_s + mu_ge + mu_gi
    mu_v = (mu_ge * ee_v + mu_gi * ei_v + gl_s * e_l_v - adapt_a) / mu_g

    tm = cm_f / mu_g

    ue = qe_s / mu_g * (ee_v - mu_v)
    ui = qi_s / mu_g * (ei_v - mu_v)
    s_v = np.sqrt(
        fe * (ue * te_s) * (ue * te_s) / (2.0 * (te_s + tm))
        + fi * (ui * ti_s) * (ui * ti_s) / (2.0 * (ti_s + tm))
    )

    fe = fe + 1e-9
    fi = fi + 1e-9
    tv = (
        (fe * (ue * te_s) * (ue * te_s) + fi * (qi_s * ui) * (qi_s * ui))
        / (
            fe * (ue * te_s) * (ue * te_s) / (te_s + tm)
            + fi * (qi_s * ui) * (qi_s * ui) / (ti_s + tm)
        )
    )
    tvn = tv * gl_s / cm_f

    mu_v0 = -60e-3
    dmu_v0 = 10e-3
    s_v0 = 4e-3
    ds_v0 = 6e-3
    tvn0 = 0.5
    dtvn0 = 1.0

    vthr = (
        p[0]
        + p[1] * (mu_v - mu_v0) / dmu_v0
        + p[2] * (s_v - s_v0) / ds_v0
        + p[3] * (tvn - tvn0) / dtvn0
        + p[4] * ((mu_v - mu_v0) / dmu_v0) * ((mu_v - mu_v0) / dmu_v0)
        + p[5] * ((s_v - s_v0) / ds_v0) * ((s_v - s_v0) / ds_v0)
        + p[6] * ((tvn - tvn0) / dtvn0) * ((tvn - tvn0) / dtvn0)
        + p[7] * (mu_v - mu_v0) / dmu_v0 * (s_v - s_v0) / ds_v0
        + p[8] * (mu_v - mu_v0) / dmu_v0 * (tvn - tvn0) / dtvn0
        + p[9] * (s_v - s_v0) / ds_v0 * (tvn - tvn0) / dtvn0
    )

    fr_out = 0.5 / tvn * gl_s / cm_f * (1.0 - erf((vthr - mu_v) / np.sqrt(2.0) / s_v))
    return float(fr_out)


def ornstein_uhlenbeck_process(
    tfin_s: float,
    *,
    theta_s_inv: float = 1.0 / (5e-3),
    mu: float = 0.0,
    sigma: float = 1.0,
    dt_s: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate the legacy OU process used for MF drive noise."""

    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(0.0, tfin_s, dt_s)
    x = np.zeros(len(t), dtype=float)
    for i in range(1, len(t)):
        dx = theta_s_inv * (mu - x[i - 1]) * dt_s + sigma * np.sqrt(dt_s) * rng.normal(0.0, 1.0)
        x[i] = x[i - 1] + dx
    return x


def run_mean_field_simulation(
    cells: str,
    amp_stim_hz: float,
    prs: np.ndarray,
    pfs: np.ndarray,
    *,
    iext_hz: float = 0.0,
    total_time_s: float = 2.0,
    seed: int | None = None,
) -> MeanFieldResult:
    """Run the legacy brian_MF mean-field simulation.

    Parameters
    ----------
    cells
        Cell preset name (legacy values include ``"FS-RS"``, ``"FS-RS_10"``).
    amp_stim_hz
        Amplitude of the transient stimulation profile in Hz.
    prs
        RS transfer-function fit coefficients, shape ``(10,)``.
    pfs
        FS transfer-function fit coefficients, shape ``(10,)``.
    iext_hz
        Baseline external drive in Hz (added to OU noise when no stimulus).
    total_time_s
        Simulation duration in seconds.
    seed
        Optional RNG seed for deterministic OU noise.

    Returns
    -------
    MeanFieldResult
        Full trajectories and mean rates over the second half of the run.
    """

    params = get_neuron_params_double_cell(cells, si_units=True)

    gl_s = params["Gl"]
    cm_f = params["Cm"]
    qe_s = params["Q_e"]
    qi_s = params["Q_i"]
    ee_v = params["E_e"]
    ei_v = params["E_i"]
    tw_rs_s = params["tau_w"]
    pconnec = params["p_con"]
    gei = params["gei"]
    ntot = params["Ntot"]

    dt_s = 1e-4
    t = np.linspace(0.0, total_time_s, int(total_time_s / dt_s))

    rng = np.random.default_rng(seed)
    os_noise = 3.5 * ornstein_uhlenbeck_process(total_time_s, rng=rng) + iext_hz

    time_peak_ms = 200.0
    tau_p_ms = 20.0
    plateau_ms = 900.0  # legacy run_MF overrides computed plateau with 900 ms
    t2_ms = np.arange(0.0, total_time_s * 1e3, 0.1)
    stim_profile = np.array([input_rate(tt, time_peak_ms, tau_p_ms, 1.0, amp_stim_hz, plateau_ms) for tt in t2_ms])

    b_rs_a = params["b_e"]
    te_s = params["tau_e"]
    ti_s = params["tau_i"]
    el_e_v = params["EL_e"]
    el_i_v = params["EL_i"]
    t_const_s = 20e-3

    fe = 6.0
    fi = 13.0
    w = fe * b_rs_a * tw_rs_s

    ls_w: list[float] = []
    ls_fe: list[float] = []
    ls_fi: list[float] = []

    for i in range(len(t)):
        external_input = stim_profile[i] if amp_stim_hz > 0 else os_noise[i]
        fe_old = fe

        fex = fe + external_input
        finh = fe_old + external_input

        if fex < 0:
            fex = 0.0
        if finh < 0:
            finh = 0.0

        fe += dt_s / t_const_s * (
            _legacy_transfer_function(
                prs,
                fex,
                fi,
                w,
                el_e_v,
                gei=gei,
                pconnec=pconnec,
                ntot=ntot,
                qi_s=qi_s,
                qe_s=qe_s,
                ti_s=ti_s,
                te_s=te_s,
                ee_v=ee_v,
                ei_v=ei_v,
                gl_s=gl_s,
                cm_f=cm_f,
            )
            - fe
        )

        w += dt_s * (-w / tw_rs_s + (b_rs_a) * fe_old)

        fi += dt_s / t_const_s * (
            _legacy_transfer_function(
                pfs,
                finh,
                fi,
                0.0,
                el_i_v,
                gei=gei,
                pconnec=pconnec,
                ntot=ntot,
                qi_s=qi_s,
                qe_s=qe_s,
                ti_s=ti_s,
                te_s=te_s,
                ee_v=ee_v,
                ei_v=ei_v,
                gl_s=gl_s,
                cm_f=cm_f,
            )
            - fi
        )

        ls_fe.append(float(fe))
        ls_fi.append(float(fi))
        ls_w.append(float(w))

    exc = np.asarray(ls_fe)
    inh = np.asarray(ls_fi)
    w_arr = np.asarray(ls_w)

    mid = int(0.5 * len(exc))
    return MeanFieldResult(
        time_s=t,
        exc_hz=exc,
        inh_hz=inh,
        adaptation_a=w_arr,
        mean_exc_hz=float(np.mean(exc[mid:])),
        mean_inh_hz=float(np.mean(inh[mid:])),
    )


def calculate_mf_difference(cells: str, fr_both: np.ndarray, inputs: np.ndarray, prs: np.ndarray, pfs: np.ndarray) -> float:
    """Compute mean absolute mismatch between SNN and MF rates.

    Port of legacy `brian_MF/MF.py::calculate_mf_difference`.

    Parameters
    ----------
    cells
        Cell preset name.
    fr_both
        Array with columns ``[inh_rate, exc_rate, input]``.
    inputs
        Input amplitudes used to run MF simulations.
    prs, pfs
        TF coefficients for RS and FS populations.
    """

    mean_both = []
    for amp_stim in inputs:
        mf = run_mean_field_simulation(cells, float(amp_stim), prs, pfs, iext_hz=0.0, total_time_s=2.0)
        mean_both.append([mf.mean_inh_hz, mf.mean_exc_hz, float(amp_stim)])

    dif_arr = np.abs(np.asarray(fr_both) - np.asarray(mean_both))
    if np.any(dif_arr[:, -1] != 0):
        raise ValueError("difference of inputs should be 0 but it is not")

    return float(np.mean(dif_arr[:, :2]))
