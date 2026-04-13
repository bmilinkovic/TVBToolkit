"""Single-region AdEx E/I simulation engine built directly on Brian2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from brian2 import (
    Hz,
    Network,
    NeuronGroup,
    PoissonGroup,
    PopulationRateMonitor,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    defaultclock,
    mV,
    ms,
    nS,
    pA,
    pF,
    amp,
    farad,
    second,
    seed,
    siemens,
    volt,
)

from tvbtoolkit.core.config import SingleRegionConfig


@dataclass
class SingleRegionResult:
    """Container for AdEx single-region outputs."""

    time_ms: np.ndarray
    exc_rate_hz: np.ndarray
    inh_rate_hz: np.ndarray
    exc_mean_vm_mv: np.ndarray
    inh_mean_vm_mv: np.ndarray
    exc_spike_times_ms: np.ndarray | None = None
    exc_spike_indices: np.ndarray | None = None
    inh_spike_times_ms: np.ndarray | None = None
    inh_spike_indices: np.ndarray | None = None


def run_single_region_simulation(cfg: SingleRegionConfig, seed_value: int = 0) -> SingleRegionResult:
    """Run a two-population AdEx network simulation using Brian2.

    The model follows a conductance-based adaptive exponential IF formulation for
    both excitatory and inhibitory populations.
    """
    seed(seed_value)
    defaultclock.dt = cfg.dt_ms * ms

    n_inh = int(cfg.n_total * cfg.inhibitory_fraction)
    n_exc = cfg.n_total - n_inh

    eqs = """
    dvm/dt = (
        g_l*(e_l-vm)
        + g_l*delta_t*exp((vm-v_t)/delta_t)
        - w
        + g_e*(e_e-vm)
        + g_i*(e_i-vm)
    )/c_m : volt (unless refractory)
    dw/dt = (a*(vm-e_l)-w)/tau_w : amp
    dg_e/dt = -g_e/tau_e : siemens
    dg_i/dt = -g_i/tau_i : siemens
    c_m : farad
    g_l : siemens
    e_l : volt
    v_t : volt
    delta_t : volt
    a : siemens
    b : amp
    tau_w : second
    tau_e : second
    tau_i : second
    e_e : volt
    e_i : volt
    v_reset : volt
    v_cut : volt
    """

    exc = NeuronGroup(
        n_exc,
        model=eqs,
        threshold="vm > v_cut",
        reset="vm = v_reset; w += b",
        refractory=cfg.refractory_ms * ms,
        method="euler",
    )
    inh = NeuronGroup(
        n_inh,
        model=eqs,
        threshold="vm > v_cut",
        reset="vm = v_reset; w += b",
        refractory=cfg.refractory_ms * ms,
        method="euler",
    )

    # Shared constants
    for pop in (exc, inh):
        pop.c_m = cfg.c_m_pf * pF
        pop.g_l = cfg.g_l_ns * nS
        pop.v_t = cfg.v_thresh_mv * mV
        pop.tau_e = cfg.tau_e_ms * ms
        pop.tau_i = cfg.tau_i_ms * ms
        pop.e_e = cfg.e_e_mv * mV
        pop.e_i = cfg.e_i_mv * mV
        pop.v_reset = cfg.v_reset_mv * mV
        pop.v_cut = cfg.v_cut_mv * mV
        pop.g_e = 0.0 * nS
        pop.g_i = 0.0 * nS

    # Excitatory (RS-like)
    exc.e_l = cfg.e_l_e_mv * mV
    exc.delta_t = cfg.delta_t_e_mv * mV
    exc.a = cfg.a_e_ns * nS
    exc.b = cfg.b_e_pa * pA
    exc.tau_w = cfg.tau_w_e_ms * ms
    exc.vm = cfg.e_l_e_mv * mV
    exc.w = 0.0 * pA

    # Inhibitory (FS-like)
    inh.e_l = cfg.e_l_i_mv * mV
    inh.delta_t = cfg.delta_t_i_mv * mV
    inh.a = cfg.a_i_ns * nS
    inh.b = cfg.b_i_pa * pA
    inh.tau_w = cfg.tau_w_i_ms * ms
    inh.vm = cfg.e_l_i_mv * mV
    inh.w = 0.0 * pA

    # Recurrent connectivity
    see = Synapses(exc, exc, on_pre=f"g_e_post += {cfg.q_e_ns}*nS")
    sei = Synapses(exc, inh, on_pre=f"g_e_post += {cfg.q_e_ns}*nS")
    sie = Synapses(inh, exc, on_pre=f"g_i_post += {cfg.q_i_ns}*nS")
    sii = Synapses(inh, inh, on_pre=f"g_i_post += {cfg.q_i_ns}*nS")
    for syn in (see, sei, sie, sii):
        syn.connect(p=cfg.p_connect)

    # External drive
    ext_e = PoissonGroup(n_exc, rates=cfg.external_rate_e_hz * Hz)
    ext_i = PoissonGroup(n_inh, rates=cfg.external_rate_i_hz * Hz)
    s_ext_ee = Synapses(ext_e, exc, on_pre=f"g_e_post += {cfg.q_e_ns}*nS")
    s_ext_ei = Synapses(ext_e, inh, on_pre=f"g_e_post += {cfg.q_e_ns}*nS")
    s_ext_ie = Synapses(ext_i, exc, on_pre=f"g_i_post += {cfg.q_i_ns}*nS")
    s_ext_ii = Synapses(ext_i, inh, on_pre=f"g_i_post += {cfg.q_i_ns}*nS")
    for syn in (s_ext_ee, s_ext_ei, s_ext_ie, s_ext_ii):
        syn.connect(p=cfg.p_external)

    m_exc = PopulationRateMonitor(exc)
    m_inh = PopulationRateMonitor(inh)
    sm_exc = StateMonitor(exc, variables="vm", record=[0], dt=cfg.dt_ms * ms)
    sm_inh = StateMonitor(inh, variables="vm", record=[0], dt=cfg.dt_ms * ms)
    sp_exc = SpikeMonitor(exc) if cfg.record_spikes else None
    sp_inh = SpikeMonitor(inh) if cfg.record_spikes else None

    net = Network(
        exc,
        inh,
        see,
        sei,
        sie,
        sii,
        ext_e,
        ext_i,
        s_ext_ee,
        s_ext_ei,
        s_ext_ie,
        s_ext_ii,
        m_exc,
        m_inh,
        sm_exc,
        sm_inh,
    )
    if sp_exc is not None:
        net.add(sp_exc)
    if sp_inh is not None:
        net.add(sp_inh)
    net.run(cfg.duration_ms * ms, report=None)

    t = np.asarray(m_exc.t / ms)
    return SingleRegionResult(
        time_ms=t,
        exc_rate_hz=np.asarray(m_exc.smooth_rate(window="flat", width=5 * ms) / Hz),
        inh_rate_hz=np.asarray(m_inh.smooth_rate(window="flat", width=5 * ms) / Hz),
        exc_mean_vm_mv=np.asarray(sm_exc.vm[0] / mV),
        inh_mean_vm_mv=np.asarray(sm_inh.vm[0] / mV),
        exc_spike_times_ms=np.asarray(sp_exc.t / ms) if sp_exc is not None else None,
        exc_spike_indices=np.asarray(sp_exc.i, dtype=int) if sp_exc is not None else None,
        inh_spike_times_ms=np.asarray(sp_inh.t / ms) if sp_inh is not None else None,
        inh_spike_indices=np.asarray(sp_inh.i, dtype=int) if sp_inh is not None else None,
    )
