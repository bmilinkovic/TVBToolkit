"""Single-cell AdEx simulations ported from legacy `brian_MF` scripts.

Ported/adapted from:
- `brian_MF/single_cell_sim.py`
- `brian_MF/single_cell_sim_modified.py`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from brian2 import (
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    mV,
    ms,
    nA,
    nS,
    pA,
    pF,
    run,
    start_scope,
)

from tvbtoolkit.brian_mf.mean_field.tf_calc import get_neuron_params_double_cell
from tvbtoolkit.brian_mf.receptors import conversion


@dataclass
class SingleCellResult:
    """Single-cell voltage/spike output.

    Attributes
    ----------
    time_ms
        Time axis (ms).
    voltage_mv
        Membrane potential trace (mV).
    spike_times_ms
        Spike times (ms).
    parameters
        Effective parameter dictionary used for simulation.
    split_leak
        Whether the split `gK/gNa` leak model was used.
    gk_ns
        Potassium conductance in nS when `split_leak=True`.
    gna_ns
        Sodium conductance in nS when `split_leak=True`.
    effective_el_mv
        Effective leak reversal in mV for split-leak mode.
    """

    time_ms: np.ndarray
    voltage_mv: np.ndarray
    spike_times_ms: np.ndarray
    parameters: dict[str, Any]
    split_leak: bool
    gk_ns: float | None = None
    gna_ns: float | None = None
    effective_el_mv: float | None = None


def _legacy_defaults(cell: str) -> dict[str, Any]:
    params = get_neuron_params_double_cell(cell)
    return params.copy()


def _apply_parameter_overrides(params: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return params
    out = params.copy()

    use_flag = overrides.get("use", True)
    if not use_flag:
        return out

    for key, val in overrides.items():
        if key == "use":
            continue
        if key not in out:
            raise KeyError(f"Unknown override key '{key}'. Valid keys: {sorted(out.keys())}")
        out[key] = val
    return out


def run_single_cell_adex(
    *,
    cell: str = "RS",
    iext_na: float = 0.3,
    time_ms: float = 200.0,
    parameter_overrides: dict[str, Any] | None = None,
    split_leak: bool = False,
    psych: bool = False,
    e_na_mv: float = 50.0,
    e_k_mv: float = -90.0,
    baseline_el_mv: float = -65.0,
    psych_el_mv: float = -63.0,
) -> SingleCellResult:
    """Run a single-cell AdEx simulation with legacy parity defaults.

    Parameters
    ----------
    cell
        Cell preset, typically ``"RS"`` or ``"FS"``.
    iext_na
        Injected current in nA.
    time_ms
        Simulation duration in ms.
    parameter_overrides
        Optional parameter overrides using legacy keys. If the dict contains
        ``{"use": False}``, overrides are ignored.
    split_leak
        If ``False``, use the standard leak form from `single_cell_sim.py`.
        If ``True``, use split `gK/gNa` leak form from
        `single_cell_sim_modified.py`.
    psych
        Only used when ``split_leak=True``; applies the shifted leak reversal
        potential conversion (`baseline_el_mv -> psych_el_mv`).
    e_na_mv, e_k_mv
        Reversal potentials used for split-leak conversion.
    baseline_el_mv, psych_el_mv
        Baseline and psych leak reversal values used in split-leak mode.

    Returns
    -------
    SingleCellResult
        Voltage and spike outputs with effective model parameters.
    """

    start_scope()

    params = _apply_parameter_overrides(_legacy_defaults(cell), parameter_overrides)

    cm = params["Cm"] * pF
    tauw_const = params["tau_w"] * ms
    a_const = params["a"] * nS
    ee_const = params["E_e"] * mV
    ei_const = params["E_i"] * mV
    i_ext = iext_na * nA

    if not split_leak:
        gl = params["Gl"] * nS
        eqs = """
        dvm/dt=(gL*(El-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
        dw/dt=(a*(vm-El)-w)/tauw : amp
        dGsynI/dt = -GsynI/TsynI : siemens
        dGsynE/dt = -GsynE/TsynE : siemens
        TsynI:second
        TsynE:second
        Vr:volt
        b_e:amp
        DeltaT:volt
        Vcut:volt
        VT:volt
        El:volt
        """

        # External constants in equation namespace (legacy variable names).
        C = cm
        gL = gl
        tauw = tauw_const
        a = a_const
        Ee = ee_const
        Ei = ei_const
        I = i_ext

        group = NeuronGroup(
            1,
            model=eqs,
            threshold="vm > Vcut",
            refractory=5 * ms,
            reset="vm = Vr; w += b_e",
            method="heun",
        )
        group.vm = params["V_m"] * mV
        group.El = params["EL"] * mV
        group.w = a_const * (group.vm - group.El)
        group.Vr = params["V_r"] * mV
        group.b_e = params["b"] * pA
        group.DeltaT = params["delta"] * mV
        group.VT = params["V_th"] * mV
        group.Vcut = params["V_cut"] * mV
        group.TsynI = params["tau_e"] * ms
        group.TsynE = params["tau_i"] * ms

        statemon = StateMonitor(group, "vm", record=0)
        spikemon = SpikeMonitor(group)

        run(time_ms * ms)

        return SingleCellResult(
            time_ms=np.asarray(statemon.t / ms),
            voltage_mv=np.asarray(statemon.vm[0] / mV),
            spike_times_ms=np.asarray(spikemon.t / ms),
            parameters=params,
            split_leak=False,
        )

    ena = e_na_mv
    ek = e_k_mv

    gk, gna = conversion(e_na=ena, e_k=ek, e_l=baseline_el_mv, g_l=params["Gl"])
    if psych:
        gk, gna = conversion(e_na=ena, e_k=ek, e_l=psych_el_mv, g_na=gna)
    gl_eff = gna + gk
    el_eff = (gna * ena + gk * ek) / gl_eff

    eqs = """
    dvm/dt=(gNa*(Ena-vm) + gK*(Ek-vm) + (gNa+gK)*DeltaT*exp((vm-VT)/DeltaT) - GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
    dw/dt=(a*(vm-El)-w)/tauw : amp
    dGsynI/dt = -GsynI/TsynI : siemens
    dGsynE/dt = -GsynE/TsynE : siemens
    TsynI:second
    TsynE:second
    Vr:volt
    b_e:amp
    DeltaT:volt
    Vcut:volt
    VT:volt
    El:volt
    """

    # External constants in equation namespace (legacy variable names).
    C = cm
    I = i_ext
    Ee = ee_const
    Ei = ei_const
    tauw = tauw_const
    a = a_const
    gNa = gna * nS
    gK = gk * nS
    Ena = ena * mV
    Ek = ek * mV

    group = NeuronGroup(
        1,
        model=eqs,
        threshold="vm > Vcut",
        refractory=5 * ms,
        reset="vm = Vr; w += b_e",
        method="heun",
    )
    group.vm = params["V_m"] * mV
    group.El = el_eff * mV
    group.w = a_const * (group.vm - group.El)
    group.Vr = params["V_r"] * mV
    group.b_e = params["b"] * pA
    group.DeltaT = params["delta"] * mV
    group.VT = params["V_th"] * mV
    group.Vcut = params["V_cut"] * mV
    group.TsynI = params["tau_e"] * ms
    group.TsynE = params["tau_i"] * ms

    statemon = StateMonitor(group, "vm", record=0)
    spikemon = SpikeMonitor(group)

    run(time_ms * ms)

    return SingleCellResult(
        time_ms=np.asarray(statemon.t / ms),
        voltage_mv=np.asarray(statemon.vm[0] / mV),
        spike_times_ms=np.asarray(spikemon.t / ms),
        parameters=params,
        split_leak=True,
        gk_ns=float(gk),
        gna_ns=float(gna),
        effective_el_mv=float(el_eff),
    )
