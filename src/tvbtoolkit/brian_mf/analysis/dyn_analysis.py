"""Dynamic-analysis helpers ported from `brian_MF/Dyn_Analysis`.

Ported/adapted from:
- `Dyn_Analysis/calculate_b_crit.py`
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from pathlib import Path

import numpy as np
from brian2 import (
    Hz,
    NeuronGroup,
    PoissonGroup,
    PopulationRateMonitor,
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
from scipy.optimize import fixed_point

from tvbtoolkit.brian_mf.adex.brian_utils import bin_array, input_rate
from tvbtoolkit.brian_mf.analysis.survival_time import calculate_survival_time


@dataclass(frozen=True)
class BCriticalResult:
    """Critical adaptation sweep output."""

    tau_e_values_ms: np.ndarray
    tau_i_values_ms: np.ndarray
    b_critical_pa: np.ndarray
    table: np.ndarray


@dataclass(frozen=True)
class DynamicSweepResult:
    """Result metadata for the dynamic-network sweep."""

    output_dir: Path
    n_simulations: int
    saved_time_file: Path
    survival_array_file: Path | None = None


def parse_np_arange_csv(value: str) -> np.ndarray:
    """Parse a legacy CSV range string into an array.

    Accepted forms:
    - `\"x\"` -> `[x]`
    - `\"start,stop,step\"` -> `np.arange(start, stop, step)`
    """

    parts = [float(i) for i in value.split(",")]
    if len(parts) == 1:
        return np.array(parts)
    if len(parts) == 3:
        return np.arange(*parts)
    raise ValueError("Provide CSV list of 1 or 3 floats.")


def _tf2(
    finh: float,
    fexc: float,
    fext: float,
    fextin: float,
    p: np.ndarray,
    adapt: float,
    e_l: float,
    *,
    gei: float,
    pconnec: float,
    ntot: float,
    qi: float,
    qe: float,
    ti: float,
    te: float,
    ee: float,
    ei: float,
    gl: float,
    cm: float,
) -> float:
    fe = (fexc + fext) * (1.0 - gei) * pconnec * ntot
    fi = (finh + fextin) * gei * pconnec * ntot

    mu_gi = qi * ti * fi
    mu_ge = qe * te * fe
    mu_g = gl + mu_ge + mu_gi
    mu_v = (mu_ge * ee + mu_gi * ei + gl * e_l - adapt) / mu_g

    tm = cm / mu_g
    ue = qe / mu_g * (ee - mu_v)
    ui = qi / mu_g * (ei - mu_v)

    s_v = np.sqrt(
        fe * (ue * te) * (ue * te) / (2.0 * (te + tm))
        + fi * (ui * ti) * (ui * ti) / (2.0 * (ti + tm))
    )

    fe += 1e-9
    fi += 1e-9
    tv = (
        (fe * (ue * te) * (ue * te) + fi * (qi * ui) * (qi * ui))
        / (
            fe * (ue * te) * (ue * te) / (te + tm)
            + fi * (qi * ui) * (qi * ui) / (ti + tm)
        )
    )
    tvn = tv * gl / cm

    mu_v0, dmu_v0 = -60e-3, 10e-3
    s_v0, ds_v0 = 4e-3, 6e-3
    tvn0, dtvn0 = 0.5, 1.0

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

    return float(0.5 / tvn * gl / cm * (1.0 - erf((vthr - mu_v) / np.sqrt(2.0) / s_v)))


def compute_b_critical_grid(
    prs: np.ndarray,
    pfs: np.ndarray,
    *,
    b_values_pa: np.ndarray | None = None,
    tau_e_values_ms: np.ndarray | None = None,
    tau_i_values_ms: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> BCriticalResult:
    """Compute critical `b_e` over tau sweeps (legacy parity workflow).

    Notes
    -----
    Mirrors logic from `calculate_b_crit.py`:
    for each `(tau_i, tau_e)` combination, sweep `b_e` until the fixed-point
    transfer curve crosses below identity.
    """

    bvals = np.arange(0, 60, 1) if b_values_pa is None else np.asarray(b_values_pa, dtype=float)
    tau_ev = np.array([5.0]) if tau_e_values_ms is None else np.asarray(tau_e_values_ms, dtype=float)
    tau_iv = np.arange(3.0, 9.0, 0.1) if tau_i_values_ms is None else np.asarray(tau_i_values_ms, dtype=float)

    # Legacy constants from calculate_b_crit.py
    gl = 10e-9
    cm = 200e-12
    qe = 1.5e-9
    qi = 5.0e-9
    ee = 0.0
    ei = -80e-3
    tw_rs = 0.5
    pconnec = 0.05
    gei = 0.2
    ntot = 10000
    el_e = -64e-3
    el_i = -65e-3

    nuev = np.arange(1e-8, 10.0, 0.1)
    combos = [(tau_i, tau_e) for tau_i in tau_iv for tau_e in tau_ev]
    rows: list[list[float]] = []

    for tau_i, tau_e in combos:
        nuext = 0.0
        nuextin = 0.0
        critical_b = bvals[-1]

        for b_e in bvals:
            b_rs = b_e * 1e-12
            ti = tau_i * 1e-3
            te = tau_e * 1e-3

            lsfe = []
            for nue in nuev:
                w = nue * b_rs * tw_rs
                nui_fix = fixed_point(
                    lambda x: _tf2(
                        float(x),
                        nue,
                        nuext,
                        nuextin,
                        pfs,
                        0.0,
                        el_i,
                        gei=gei,
                        pconnec=pconnec,
                        ntot=ntot,
                        qi=qi,
                        qe=qe,
                        ti=ti,
                        te=te,
                        ee=ee,
                        ei=ei,
                        gl=gl,
                        cm=cm,
                    ),
                    [1.0],
                )
                tf_e = _tf2(
                    float(nui_fix),
                    nue,
                    nuext,
                    0.0,
                    prs,
                    w,
                    el_e,
                    gei=gei,
                    pconnec=pconnec,
                    ntot=ntot,
                    qi=qi,
                    qe=qe,
                    ti=ti,
                    te=te,
                    ee=ee,
                    ei=ei,
                    gl=gl,
                    cm=cm,
                )
                lsfe.append(tf_e)

            delta = nuev - np.asarray(lsfe)
            if np.all(delta >= 1e-11):
                critical_b = b_e
                break

        rows.append([tau_e, tau_i, critical_b])

    table = np.asarray(rows, dtype=float)
    result = BCriticalResult(
        tau_e_values_ms=tau_ev,
        tau_i_values_ms=tau_iv,
        b_critical_pa=table[:, 2],
        table=table,
    )

    if save_path is not None:
        out = Path(save_path)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "b_thresh.npy", table)

    return result


def run_dynamic_sweep(
    *,
    b_e_range: np.ndarray | None = None,
    tau_e_range: np.ndarray | None = None,
    tau_i_range: np.ndarray | None = None,
    n_seeds: np.ndarray | None = None,
    time_ms: float = 2000.0,
    save_path: str | Path = "./Dyn_Analysis/trials",
    overwrite: bool = False,
    compute_survival: bool = False,
    dt_ms: float = 0.1,
    amp_stim_hz: float = 1.0,
    plateau_ms: float = 100.0,
    tau_p_ms: float = 20.0,
    time_peak_ms: float = 200.0,
    n_inh: int = 2000,
    n_exc: int = 8000,
) -> DynamicSweepResult:
    """Run the legacy dynamic parameter sweep from `net_sims_dyn_analysis.py`.

    Parameters are kept close to the legacy script defaults, with additional
    hooks (`n_inh`, `n_exc`) to allow light-weight test runs.
    """

    bvals = np.arange(0, 30, 1) if b_e_range is None else np.asarray(b_e_range, dtype=float)
    tau_ev = np.array([5.0]) if tau_e_range is None else np.asarray(tau_e_range, dtype=float)
    tau_iv = np.arange(3.0, 9.0, 0.1) if tau_i_range is None else np.asarray(tau_i_range, dtype=float)
    seeds_v = np.arange(0, 100, 5) if n_seeds is None else np.asarray(n_seeds, dtype=int)

    if len(tau_iv) > 1 and len(tau_ev) > 1:
        raise ValueError("Iterate either tau_e or tau_i, not both simultaneously.")

    out_root = Path(save_path)
    sims_dir = out_root / "network_sims"
    sims_dir.mkdir(parents=True, exist_ok=True)

    t2 = np.arange(0.0, float(time_ms), float(dt_ms))
    test_input = np.array([input_rate(tt, time_peak_ms, tau_p_ms, 1.0, amp_stim_hz, plateau_ms) for tt in t2])
    input_stim = TimedArray(test_input * Hz, dt=dt_ms * ms)

    defaultclock.dt = dt_ms * ms
    duration = time_ms * ms

    C = 200 * pF
    gL = 10 * nS
    tauw = 500 * ms
    a = 0.0 * nS
    I = 0.0 * nA
    Ee = 0.0 * mV
    Ei = -80.0 * mV
    eli = -65.0
    ele = -64.0

    eqs = """
    dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
    dw/dt=(a*(vm-EL)-w)/tauw : amp
    dGsynI/dt = -GsynI/TsynI : siemens
    dGsynE/dt = -GsynE/TsynE : siemens
    TsynI:second
    TsynE:second
    Vr:volt
    b:amp
    DeltaT:volt
    Vcut:volt
    VT:volt
    EL:volt
    """

    combinations = [(seed_v, tau_i, b_ad, tau_e) for b_ad in bvals for tau_i in tau_iv for tau_e in tau_ev for seed_v in seeds_v]
    n_done = 0
    last_time = None

    for seed_v, tau_i, b_ad, tau_e in combinations:
        sim_name = f"b_{b_ad}_tau_i_{round(tau_i,1)}_tau_e_{round(tau_e,1)}_ampst_{amp_stim_hz}_seed_{seed_v}"
        str_exc = sims_dir / f"{sim_name}_exc.npy"
        str_inh = sims_dir / f"{sim_name}_inh.npy"

        if str_exc.exists() and str_inh.exists() and not overwrite:
            continue

        start_scope()
        seed(int(seed_v))
        defaultclock.dt = dt_ms * ms

        g_inh = NeuronGroup(n_inh, model=eqs, threshold="vm > Vcut", refractory=5 * ms, reset="vm = Vr; w += b", method="heun")
        g_inh.vm = -60 * mV
        g_inh.EL = eli * mV
        g_inh.w = a * (g_inh.vm - g_inh.EL)
        g_inh.Vr = -65 * mV
        g_inh.TsynI = tau_i * ms
        g_inh.TsynE = tau_e * ms
        g_inh.b = 0 * pA
        g_inh.DeltaT = 0.5 * mV
        g_inh.VT = -50.0 * mV
        g_inh.Vcut = g_inh.VT + 5 * g_inh.DeltaT

        g_exc = NeuronGroup(n_exc, model=eqs, threshold="vm > Vcut", refractory=5 * ms, reset="vm = Vr; w += b", method="heun")
        g_exc.vm = -60 * mV
        g_exc.EL = ele * mV
        g_exc.w = a * (g_exc.vm - g_exc.EL)
        g_exc.Vr = -65 * mV
        g_exc.TsynI = tau_i * ms
        g_exc.TsynE = tau_e * ms
        g_exc.b = b_ad * pA
        g_exc.DeltaT = 2 * mV
        g_exc.VT = -50.0 * mV
        g_exc.Vcut = g_exc.VT + 5 * g_exc.DeltaT

        p_ed = PoissonGroup(n_exc, rates="input_stim(t)")

        qi = 5.0 * nS
        qe = 1.5 * nS
        prbc = 500 / (n_inh + n_exc)

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

        fr_inh = PopulationRateMonitor(g_inh)
        fr_exc = PopulationRateMonitor(g_exc)

        run(duration)

        time_array = np.arange(int(time_ms / dt_ms)) * dt_ms
        lfr_exc = np.array(fr_exc.rate / Hz)
        tim_binned = bin_array(time_array, 5, time_array)
        pop_exc = bin_array(lfr_exc, 5, time_array)

        lfr_inh = np.array(fr_inh.rate / Hz)
        pop_inh = bin_array(lfr_inh, 5, time_array)

        np.save(str_exc, pop_exc)
        np.save(str_inh, pop_inh)
        last_time = tim_binned
        n_done += 1

    if last_time is None:
        # Nothing simulated this run; load a previous time array if present.
        time_candidates = sorted(sims_dir.glob("*_time.npy"))
        if time_candidates:
            saved_time = time_candidates[-1]
        else:
            saved_time = sims_dir / "time.npy"
            np.save(saved_time, np.array([], dtype=float))
    else:
        saved_time = sims_dir / f"binned_time_{int(time_ms)}ms.npy"
        if overwrite or not saved_time.exists():
            np.save(saved_time, last_time)

    survival_file = None
    if compute_survival:
        offset_index = int((plateau_ms + time_peak_ms + 5) / 5)
        load_until = int((time_ms / 5) - 5)

        if len(tau_ev) == 1 and len(tau_iv) > 1:
            tau_i_iter = True
            tau_values = tau_iv
            tau_str = "tau_i"
        elif len(tau_ev) > 1 and len(tau_iv) == 1:
            tau_i_iter = False
            tau_values = tau_ev
            tau_str = "tau_e"
        else:
            tau_i_iter = True
            tau_values = tau_iv
            tau_str = "tau_i"

        calculate_survival_time(
            bvals,
            tau_values,
            tau_i_iter,
            seeds_v,
            save_path=str(out_root),
            bin_ms=5,
            amp_stim=amp_stim_hz,
            offset_index=offset_index,
            load_until=load_until,
        )
        survival_file = out_root / f"{tau_str}_mean_array.npy"

    return DynamicSweepResult(
        output_dir=out_root,
        n_simulations=n_done,
        saved_time_file=saved_time,
        survival_array_file=survival_file,
    )
