"""Network-level AdEx simulations ported from legacy `brian_MF` scripts.

Ported/adapted from:
- `brian_MF/adex_simulation_network.py`
- `brian_MF/adex_gK_gNa.py`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from brian2 import (
    Hz,
    NeuronGroup,
    PoissonGroup,
    PopulationRateMonitor,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    TimedArray,
    defaultclock,
    mV,
    ms,
    nA,
    nS,
    pA,
    pF,
    run,
    seed,
    start_scope,
)

from tvbtoolkit.brian_mf.adex.brian_utils import (
    PopulationRates,
    bin_array,
    input_rate,
    prepare_population_rates,
)
from tvbtoolkit.brian_mf.mean_field.tf_calc import get_neuron_params_double_cell
from tvbtoolkit.brian_mf.receptors import conversion


@dataclass
class NetworkSimulationResult:
    """Legacy-style two-population SNN outputs.

    Attributes
    ----------
    time_ms
        Binned time axis (ms).
    rate_exc_hz
        Binned excitatory firing rate (Hz).
    rate_inh_hz
        Binned inhibitory firing rate (Hz).
    adaptation
        Binned adaptation signal from legacy `P2mon` summed variable.
    raster_exc
        Raster tuple-like array ``[times_ms, indices]`` for excitatory neurons.
    raster_inh
        Raster tuple-like array ``[times_ms, indices_shifted]`` for inhibitory neurons.
    input_binned
        Binned stimulation input trace when `input_hz > 0`, else NaN array.
    parameters
        Effective simulation parameter dictionary.
    split_leak
        Whether split `gK/gNa` leak equations were used.
    sim_name
        Legacy-style simulation label.
    """

    time_ms: np.ndarray
    rate_exc_hz: np.ndarray
    rate_inh_hz: np.ndarray
    adaptation: np.ndarray | None
    raster_exc: np.ndarray
    raster_inh: np.ndarray
    input_binned: np.ndarray
    parameters: dict[str, Any]
    split_leak: bool
    sim_name: str


def _coerce_override_values(overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, str):
            low = value.lower()
            if low == "true":
                out[key] = True
            elif low == "false":
                out[key] = False
            else:
                try:
                    out[key] = int(value)
                except ValueError:
                    try:
                        out[key] = float(value)
                    except ValueError:
                        out[key] = value
        else:
            out[key] = value
    return out


def _base_legacy_params(cells: str, *, split_leak: bool, e_l_e_start_mv: float, e_l_i_start_mv: float) -> dict[str, Any]:
    # Legacy scripts effectively used hard-coded defaults instead of the full cell library.
    if split_leak:
        return {
            "V_m": -60,
            "V_r": -65,
            "Cm": 200,
            "Gl": 10,
            "tau_w": 500,
            "V_th": -50,
            "V_cut": -30,
            "a_e": 0,
            "b_e": 5,
            "delta_e": 2,
            "EL_e": e_l_e_start_mv,
            "a_i": 0,
            "b_i": 0,
            "delta_i": 0.5,
            "EL_i": e_l_i_start_mv,
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

    # `adex_simulation_network.py` defaults.
    return {
        "V_m": -60,
        "V_r": -65,
        "Cm": 200,
        "Gl": 10,
        "tau_w": 500,
        "V_th": -50,
        "V_cut": -30,
        "a_e": 0,
        "b_e": 5,
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


def _apply_parameter_overrides(params: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return params

    clean = _coerce_override_values(overrides)
    assert clean is not None
    if clean.get("use", True) is False:
        return params

    out = params.copy()
    for key, value in clean.items():
        if key == "use":
            continue
        if key not in out:
            raise KeyError(f"Unknown override key '{key}'. Valid keys: {sorted(out.keys())}")
        out[key] = value
    return out


def run_adex_network_simulation(
    *,
    cells: str = "FS-RS",
    seed_value: int = 0,
    time_ms: float = 1000.0,
    iext_hz: float = 0.5,
    input_hz: float = 0.0,
    plat_dur_ms: float = 0.0,
    external_rate_hz_trace: np.ndarray | None = None,
    external_rate_dt_ms: float | None = None,
    parameter_overrides: dict[str, Any] | None = None,
    split_leak: bool = False,
    psych: bool = False,
    e_l_e_start_mv: float = -65.0,
    e_l_i_start_mv: float = -65.0,
    e_l_e_end_mv: float = -59.0,
    e_l_i_end_mv: float = -63.0,
    dt_ms: float = 0.1,
    bin_width_ms: float = 5.0,
    save_path: str | Path | None = None,
    save_mean: bool = False,
    save_all: bool = False,
) -> NetworkSimulationResult:
    """Run legacy-style two-population AdEx network simulations.

    Parameters mirror the original scripts while exposing one unified function.

    `split_leak=False` reproduces `adex_simulation_network.py` style equations.
    `split_leak=True` reproduces `adex_gK_gNa.py` style equations and receptor-like
    leak conversion (`gK/gNa`), with optional `psych` leak shift.
    """

    start_scope()
    defaultclock.dt = dt_ms * ms
    seed(seed_value)

    params = _base_legacy_params(
        cells,
        split_leak=split_leak,
        e_l_e_start_mv=e_l_e_start_mv,
        e_l_i_start_mv=e_l_i_start_mv,
    )
    params = _apply_parameter_overrides(params, parameter_overrides)

    n1 = int(params["gei"] * params["Ntot"])
    n2 = int((1.0 - params["gei"]) * params["Ntot"])

    amp_stim = float(input_hz)
    time_peak = 200.0
    tau_p = 20.0
    if not plat_dur_ms:
        plat = float(time_ms) - time_peak - tau_p
    else:
        plat = float(plat_dur_ms)

    t2 = np.arange(0.0, float(time_ms), float(dt_ms))
    test_input = np.array([input_rate(tt, time_peak, tau_p, 1.0, amp_stim, plat) for tt in t2])
    input_stim = TimedArray(test_input * Hz, dt=dt_ms * ms)

    duration = time_ms * ms

    C = params["Cm"] * pF
    I = 0.0 * nA
    Ee = params["E_e"] * mV
    Ei = params["E_i"] * mV
    tauw = params["tau_w"] * ms

    if not split_leak:
        gL = params["Gl"] * nS

        sim_name = (
            f"_b_{params['b_e']}_tau_e_{params['tau_e']}_tau_i_{params['tau_i']}"
            f"_eli_{int(params['EL_i'])}_ele_{int(params['EL_e'])}_iext_{iext_hz}"
        )

        eqs = """
        dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
        dw/dt=(a*(vm-EL)-w)/tauw : amp
        dGsynI/dt = -GsynI/TsynI : siemens
        dGsynE/dt = -GsynE/TsynE : siemens
        TsynI:second
        TsynE:second
        Vr:volt
        b:amp
        a:siemens
        DeltaT:volt
        Vcut:volt
        VT:volt
        EL:volt
        """

        g_inh = NeuronGroup(n1, model=eqs, threshold="vm > Vcut", refractory=5 * ms, reset="vm = Vr; w += b", method="heun")
        g_inh.vm = params["V_m"] * mV
        g_inh.a = params["a_i"] * nS
        g_inh.EL = params["EL_i"] * mV
        g_inh.w = g_inh.a * (g_inh.vm - g_inh.EL)
        g_inh.Vr = params["V_r"] * mV
        g_inh.TsynI = params["tau_i"] * ms
        g_inh.TsynE = params["tau_e"] * ms
        g_inh.b = params["b_i"] * pA
        g_inh.DeltaT = params["delta_i"] * mV
        g_inh.VT = params["V_th"] * mV
        g_inh.Vcut = params["V_cut"] * mV

        g_exc = NeuronGroup(n2, model=eqs, threshold="vm > Vcut", refractory=5 * ms, reset="vm = Vr; w += b", method="heun")
        g_exc.vm = params["V_m"] * mV
        g_exc.a = params["a_e"] * nS
        g_exc.EL = params["EL_e"] * mV
        g_exc.w = g_exc.a * (g_exc.vm - g_exc.EL)
        g_exc.Vr = params["V_r"] * mV
        g_exc.TsynI = params["tau_i"] * ms
        g_exc.TsynE = params["tau_e"] * ms
        g_exc.b = params["b_e"] * pA
        g_exc.DeltaT = params["delta_e"] * mV
        g_exc.VT = params["V_th"] * mV
        g_exc.Vcut = params["V_cut"] * mV

    else:
        gk_e, gna_e = conversion(50, -90, params["EL_e"], g_l=params["Gl"])
        gk_i, gna_i = conversion(50, -90, params["EL_i"], g_l=params["Gl"])

        if psych:
            gk_e, gna_e = conversion(50, -90, e_l_e_end_mv, g_na=gna_e)
            gk_i, gna_i = conversion(50, -90, e_l_i_end_mv, g_na=gna_i)

        ek = -90.0
        ena = 50.0

        gl_e = gk_e + gna_e
        gl_i = gk_i + gna_i
        el_e_eff = ((gna_e * ena) + (gk_e * ek)) / gl_e
        el_i_eff = ((gna_i * ena) + (gk_i * ek)) / gl_i

        ENa = ena * mV
        EK = ek * mV

        sim_name = (
            f"_b_{params['b_e']}_tau_e_{params['tau_e']}_tau_i_{params['tau_i']}"
            f"_g_K_e_{gk_e}_g_Na_e_{gna_e}_g_K_i_{gk_i}_g_Na_i_{gna_i}"
            f"_eli_{round(el_i_eff)}_ele_{round(el_e_eff)}_iext_{iext_hz}"
        )

        eqs = """
        dvm/dt=(gK*(EK-vm) + gNa*(ENa-vm)+ (gK+gNa)*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
        dw/dt=(a*(vm-EL)-w)/tauw : amp
        dGsynI/dt = -GsynI/TsynI : siemens
        dGsynE/dt = -GsynE/TsynE : siemens
        TsynI:second
        TsynE:second
        Vr:volt
        b:amp
        a:siemens
        DeltaT:volt
        Vcut:volt
        VT:volt
        EL:volt
        gK:siemens
        gNa:siemens
        """

        g_inh = NeuronGroup(n1, model=eqs, threshold="vm > Vcut", refractory=5 * ms, reset="vm = Vr; w += b", method="heun")
        g_inh.vm = params["V_m"] * mV
        g_inh.a = params["a_i"] * nS
        g_inh.EL = el_i_eff * mV
        g_inh.w = g_inh.a * (g_inh.vm - g_inh.EL)
        g_inh.Vr = params["V_r"] * mV
        g_inh.TsynI = params["tau_i"] * ms
        g_inh.TsynE = params["tau_e"] * ms
        g_inh.b = params["b_i"] * pA
        g_inh.DeltaT = params["delta_i"] * mV
        g_inh.VT = params["V_th"] * mV
        g_inh.Vcut = params["V_cut"] * mV
        g_inh.gK = gk_i * nS
        g_inh.gNa = gna_i * nS

        g_exc = NeuronGroup(n2, model=eqs, threshold="vm > Vcut", refractory=5 * ms, reset="vm = Vr; w += b", method="heun")
        g_exc.vm = params["V_m"] * mV
        g_exc.a = params["a_e"] * nS
        g_exc.EL = el_e_eff * mV
        g_exc.w = g_exc.a * (g_exc.vm - g_exc.EL)
        g_exc.Vr = params["V_r"] * mV
        g_exc.TsynI = params["tau_i"] * ms
        g_exc.TsynE = params["tau_e"] * ms
        g_exc.b = params["b_e"] * pA
        g_exc.DeltaT = params["delta_e"] * mV
        g_exc.VT = params["V_th"] * mV
        g_exc.Vcut = params["V_cut"] * mV
        g_exc.gK = gk_e * nS
        g_exc.gNa = gna_e * nS

    ext_rate_trace = None
    ext_rate_dt_ms = float(dt_ms)
    if external_rate_hz_trace is not None:
        ext_rate_trace = np.asarray(external_rate_hz_trace, dtype=float).reshape(-1)
        if ext_rate_trace.size == 0:
            raise ValueError("external_rate_hz_trace must be non-empty when provided.")
        if not np.all(np.isfinite(ext_rate_trace)):
            raise ValueError("external_rate_hz_trace contains non-finite values.")
        ext_rate_trace = np.maximum(ext_rate_trace, 0.0)
        ext_rate_dt_ms = float(dt_ms if external_rate_dt_ms is None else external_rate_dt_ms)
        if ext_rate_dt_ms <= 0.0:
            raise ValueError("external_rate_dt_ms must be > 0.")
        external_input = TimedArray(ext_rate_trace * Hz, dt=ext_rate_dt_ms * ms)
        p_ed = PoissonGroup(n2, rates="external_input(t)")
    elif amp_stim > 0:
        p_ed = PoissonGroup(n2, rates="input_stim(t)")
    else:
        p_ed = PoissonGroup(n2, rates=float(iext_hz) * Hz)

    qi = params["Q_i"] * nS
    qe = params["Q_e"] * nS
    prbc = params["p_con"]

    s_12 = Synapses(g_inh, g_exc, on_pre="GsynI_post+=qi")
    s_12.connect("i!=j", p=prbc)

    s_11 = Synapses(g_inh, g_inh, on_pre="GsynI_post+=qi")
    s_11.connect("i!=j", p=prbc)

    s_21 = Synapses(g_exc, g_inh, on_pre="GsynE_post+=qe")
    s_21.connect("i!=j", p=prbc)

    s_22 = Synapses(g_exc, g_exc, on_pre="GsynE_post+=qe")
    s_22.connect("i!=j", p=prbc)

    s_ed_in = Synapses(p_ed, g_inh, on_pre="GsynE_post+=qe")
    s_ed_in.connect(p=prbc)

    s_ed_ex = Synapses(p_ed, g_exc, on_pre="GsynE_post+=qe")
    s_ed_ex.connect(p=prbc)

    pgroup_e = NeuronGroup(1, "P:amp", method="heun")
    pe = Synapses(g_exc, pgroup_e, "P_post = w_pre : amp (summed)")
    pe.connect(p=1)
    p2mon = StateMonitor(pgroup_e, "P", record=0)

    sp_inh = SpikeMonitor(g_inh)
    fr_inh = PopulationRateMonitor(g_inh)
    sp_exc = SpikeMonitor(g_exc)
    fr_exc = PopulationRateMonitor(g_exc)

    run(duration)

    ras_inh = np.array([np.asarray(sp_inh.t / ms), np.asarray([i + n2 for i in sp_inh.i])], dtype=object)
    ras_exc = np.array([np.asarray(sp_exc.t / ms), np.asarray(sp_exc.i)], dtype=object)

    if ext_rate_trace is not None:
        time_array_ext = np.arange(ext_rate_trace.size, dtype=float) * ext_rate_dt_ms
        input_bin = bin_array(ext_rate_trace, float(bin_width_ms), time_array_ext)
    elif amp_stim > 0:
        time_array = np.arange(int(float(time_ms) / float(dt_ms))) * float(dt_ms)
        input_bin = bin_array(np.asarray(test_input), float(bin_width_ms), time_array)
    else:
        input_bin = np.full(1, np.nan)

    rates: PopulationRates = prepare_population_rates(
        total_time_ms=float(time_ms),
        dt_ms=float(dt_ms),
        fr_exc_monitor=fr_exc,
        fr_inh_monitor=fr_inh,
        adaptation_monitor=p2mon,
        bin_width_ms=float(bin_width_ms),
    )

    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_mean:
            np.save(
                out_dir / f"{cells}_mean_exc_amp_{amp_stim}.npy",
                np.array([np.mean(rates.exc_hz[int(len(rates.exc_hz) / 2) :]), amp_stim, params], dtype=object),
            )
            np.save(
                out_dir / f"{cells}_mean_inh_amp_{amp_stim}.npy",
                np.array([np.mean(rates.inh_hz[int(len(rates.inh_hz) / 2) :]), amp_stim, params], dtype=object),
            )

        if save_all:
            np.save(out_dir / f"{cells}_inh_amp_{amp_stim}.npy", np.array([rates.inh_hz, amp_stim, params], dtype=object))
            np.save(out_dir / f"{cells}_exc_amp_{amp_stim}.npy", np.array([rates.exc_hz, amp_stim, params], dtype=object))

    return NetworkSimulationResult(
        time_ms=np.asarray(rates.time_ms),
        rate_exc_hz=np.asarray(rates.exc_hz),
        rate_inh_hz=np.asarray(rates.inh_hz),
        adaptation=np.asarray(rates.adaptation) if rates.adaptation is not None else None,
        raster_exc=ras_exc,
        raster_inh=ras_inh,
        input_binned=np.asarray(input_bin),
        parameters=params,
        split_leak=split_leak,
        sim_name=sim_name,
    )


def run_snn_split_leak(
    *,
    seed_value: int,
    time_ms: float,
    iext_hz: float,
    input_hz: float = 0.0,
    plat_dur_ms: float = 0.0,
    psych: bool = False,
    e_l_e_start_mv: float = -65.0,
    e_l_i_start_mv: float = -65.0,
    e_l_e_end_mv: float = -59.0,
    e_l_i_end_mv: float = -63.0,
    parameter_overrides: dict[str, Any] | None = None,
    save_path: str | Path | None = None,
    save_mean: bool = False,
    save_all: bool = False,
) -> NetworkSimulationResult:
    """Compatibility wrapper mirroring legacy `run_SNN` behaviour (`adex_gK_gNa.py`)."""

    return run_adex_network_simulation(
        cells="FS-RS_10",
        seed_value=seed_value,
        time_ms=time_ms,
        iext_hz=iext_hz,
        input_hz=input_hz,
        plat_dur_ms=plat_dur_ms,
        parameter_overrides=parameter_overrides,
        split_leak=True,
        psych=psych,
        e_l_e_start_mv=e_l_e_start_mv,
        e_l_i_start_mv=e_l_i_start_mv,
        e_l_e_end_mv=e_l_e_end_mv,
        e_l_i_end_mv=e_l_i_end_mv,
        save_path=save_path,
        save_mean=save_mean,
        save_all=save_all,
    )
