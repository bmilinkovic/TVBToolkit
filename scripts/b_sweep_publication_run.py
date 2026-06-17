"""Run a publication-grade b_e sweep across three model scales.

For each b_e value in :data:`B_VALUES_PA` this script runs three simulations
that share an identical OU-correlated external drive trace:

1. **Spiking neural network (SNN)** — :func:`tvbtoolkit.brian_mf.adex.network.run_adex_network_simulation`
   (Brian2, faithful port of legacy ``adex_simulation_network.py`` used in
   TVBSim). The drive trace is passed via ``external_rate_hz_trace=``.

2. **Single-region mean field** — TVB Zerlaut **second-order** via
   :func:`tvbtoolkit.whole_brain.simulation.run_whole_brain_simulation` on a
   1-region zero-coupling connectivity. TVB's internal stochastic noise is
   disabled (``weight_noise=0``, ``stochastic_integrator=False``) and the
   shared OU trace is injected into ``external_input_ex_ex`` /
   ``external_input_in_ex`` at every ``dfun`` call via a small subclass
   :class:`_Zerlaut2OUDrive` (no toolbox modification).

3. **Whole-brain mean field (DK-68)** — same TVB Zerlaut second-order +
   ``_Zerlaut2OUDrive`` subclass with DK-68 connectivity. WB sees the same
   shared OU trace as the SNN and single-region MF.

Shared biophysics (lab v2 defaults from
``notebooks/brain_act_hybrid_common.BASE_PARAMETER_MODEL_NEW``)
--------------------------------------------------------------

* ``N_tot = 10000``, ``g = 0.2``, ``p_connect = 0.05``.
* ``E_L_e = -63 mV``, ``E_L_i = -65 mV``, ``tau_w_e = 500 ms``.
* MF/WB: ``T = 20 ms``, ``zerlaut_order = 2`` (WB) or 1st-order (single-region MF).
* MF/WB: v2 ``P_e`` / ``P_i`` polynomial coefficients.
* MF/WB initial conditions: ``E = 0.004 Hz``, ``I = 0.010 Hz``, ``W_e = 50 pA``.

External-drive convention — paper Fig 4a (per scale)
----------------------------------------------------

Paper Eq. (20):  ``v_aff(t) = v_drive + σ · ξ(t)``,  ``dξ = -(ξ/τ_OU) dt + dW``
with ``σ = 3.5 Hz``, ``τ_OU = 5 ms`` (paper Eqs. 20–21).

The Sacha et al. Fig 4a caption assigns *different* ``v_drive`` baselines to
the SNN/MF scales versus the whole-brain scale, because the WB benefits from
long-range coupling and operates with a lower mean drive:

* **SNN + single-region MF**:  ``v_drive = 0.4 Hz``     (:data:`SNN_MF_DRIVE_HZ`)
* **Whole brain (DK-68)**   :  ``v_drive = 0.315 Hz``   (:data:`WB_DRIVE_HZ`)

Both traces use the **same OU noise realization** (same seed → same ``ξ(t)``);
only the baseline differs. Effective per-neuron afferent rate:
``K_eff × v_aff(t)`` ≈ 160 Hz (SNN/MF) or 126 Hz (WB) ± OU fluctuation,
where ``K_eff = N_tot·(1-g)·p_con = K_ext_e = 400``.

Modes
-----
::

    python scripts/b_sweep_publication_run.py --all                 # full sweep
    python scripts/b_sweep_publication_run.py --b 25                # one b value
    python scripts/b_sweep_publication_run.py --all --force         # overwrite pickles
    python scripts/b_sweep_publication_run.py --all --snn-only      # update SNN only
    python scripts/b_sweep_publication_run.py --all --mf-only       # update MF only
    python scripts/b_sweep_publication_run.py --all --wb-only       # update WB only

Requires Brian2 + TVB + scipy + numba (toolkit runtime deps). Run from your
local Python environment, not the sandbox.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tvbtoolkit.brian_mf.adex.network import run_adex_network_simulation
from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.whole_brain import simulation as _wb_sim_module
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation
from tvbtoolkit.whole_brain.legacy_engine.src import Zerlaut as _zerlaut_mod


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

B_VALUES_PA: tuple[int, ...] = (5, 25, 45, 65, 85, 105, 125)
# Paper Fig 4a reference: AI uses b_e=5 pA, deep NREM/UD uses b_e=120 pA.
# This 7-point sweep walks from AI (5) through transition (25–65) to the UD
# regime (105–125), one even step (20 pA) apart.

SIM_DURATION_MS: float = 22_000.0
CUT_TRANSIENT_MS: float = 4_000.0
DT_MS: float = 0.1
SEED: int = 1

SNN_NTOT: int = 10_000
SNN_GEI: float = 0.2
SNN_P_CON: float = 0.05
SNN_BIN_MS: float = 5.0

# Paper Fig 4a v_drive per scale (Methods caption Fig 4):
#   SNN + single-region MF: 0.4 Hz/fiber (chosen near b_crit on the Fig 3 yellow line)
#   Whole-brain        : 0.315 Hz/fiber (long-range coupling supports activity)
SNN_MF_DRIVE_HZ: float = 0.4
WB_DRIVE_HZ:     float = 0.315

# Lab v2 defaults from notebooks/brain_act_hybrid_common.BASE_PARAMETER_MODEL_NEW
EL_E_MV: float = -63.0
EL_I_MV: float = -65.0
TAU_E_MS: float = 5.0
TAU_I_MS: float = 5.0

# Sacha 2025 Eq. (20) OU drive
PAPER_SIGMA_HZ: float = 3.5
TAU_OU_MS: float = 5.0
DRIVE_TRACE_DT_MS: float = 0.1   # OU trace resolution (shared by SNN + MF)

# v2 transfer-function polynomial coefficients
P_E_V2: tuple[float, ...] = (
    -0.04983106, 0.00506355, -0.02347012, 0.00229515, -0.00041053,
    0.01054705, -0.03659253, 0.00743749, 0.00126506, -0.04072161,
)
P_I_V2: tuple[float, ...] = (
    -0.05149122, 0.00400369, -0.00835201, 0.00024142, -0.00050706,
    0.00143454, -0.01468669, 0.00450271, 0.00284722, -0.01535780,
)

MF_T_MS: float = 20.0

# WB-specific
WB_INIT_COND: dict[str, list[float]] = {
    "E":    [0.004, 0.004],
    "I":    [0.010, 0.010],
    "C_ee": [0.0, 0.0],
    "C_ei": [0.0, 0.0],
    "C_ii": [0.0, 0.0],
    "W_e":  [50.0, 50.0],
    "W_i":  [0.0, 0.0],
    "noise":[0.0, 0.0],
}
WB_COUPLING_STRENGTH: float = 0.3
DK68_CONNECTIVITY_ZIP = REPO_ROOT / "data" / "connectivity" / "connectivity_68.zip"

PER_B_DIR = REPO_ROOT / "notebooks" / "outputs" / "b_sweep" / "per_b"
PER_B_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# OU drive trace — ported verbatim from notebooks/01_mean_field_sim_test.py
# ---------------------------------------------------------------------------

def paper_sigma_to_runner_sigma_hz(paper_sigma: float, tau_ou_ms: float, dt_ms: float) -> float:
    """Map paper Eq. (20) sigma to the discrete OU-update sigma used below."""
    tau_s = float(tau_ou_ms) / 1000.0
    target_std_hz = float(paper_sigma) * np.sqrt(max(tau_s, 1e-12) / 2.0)
    dt_over_tau = float(dt_ms) / max(float(tau_ou_ms), 1e-12)
    return float(target_std_hz * np.sqrt(max(2.0 - dt_over_tau, 1e-9)))


def build_shared_afferent_trace_hz(
    *,
    duration_ms: float,
    dt_ms: float,
    seed: int,
    v_drive_hz: float,
    sigma_ou_hz: float,
    tau_ou_ms: float,
) -> np.ndarray:
    """Generate one OU-correlated afferent drive trace (Hz). Clipped at 0."""
    n = int(float(duration_ms) / float(dt_ms))
    rng = np.random.default_rng(int(seed))
    xi = 0.0
    out = np.empty(n, dtype=float)
    tau = max(float(tau_ou_ms), 1e-12)
    scale = float(sigma_ou_hz) * np.sqrt(float(dt_ms) / tau)
    decay = 1.0 - float(dt_ms) / tau
    base = float(v_drive_hz)
    for i in range(n):
        xi = xi * decay + scale * rng.standard_normal()
        out[i] = max(base + xi, 0.0)
    return out


def _trim_transient_snn(time_ms: np.ndarray, *arrays: np.ndarray, cut_ms: float):
    mask = time_ms >= cut_ms
    t = time_ms[mask] - cut_ms
    return (t, *(a[mask] for a in arrays))


def _trim_transient_tvb(time_ms: np.ndarray, raw_data: np.ndarray, cut_ms: float):
    mask = time_ms >= cut_ms
    return time_ms[mask] - cut_ms, raw_data[mask]


# ---------------------------------------------------------------------------
# 2nd-order Zerlaut subclass with shared OU drive injection
# ---------------------------------------------------------------------------
#
# The TVB Zerlaut second-order ``dfun`` reads two scalar/array model
# attributes — ``external_input_ex_ex`` and ``external_input_in_ex`` — at
# every call. By overriding ``dfun`` and mutating these attributes from a
# precomputed OU trace just before delegating to ``super().dfun(...)``, the
# shared OU drive enters the rate equation through the same transfer-function
# path as a fluctuating afferent rate would. This is functionally equivalent
# to what ``run_adex_network_simulation`` does on the SNN side via
# ``external_rate_hz_trace=``.
#
# TVB's Heun integrator (deterministic or stochastic) calls ``dfun`` twice
# per simulator step (predictor + corrector). We count dfun calls and divide
# by 2 to get the simulator-time index into the OU trace.
#
# To disable TVB's internal noise we set ``weight_noise=0`` (the noise state
# variable still evolves but contributes nothing to the rate equation) and
# ``stochastic_integrator=False`` (HeunDeterministic — saves compute).

class _Zerlaut2OUDrive(_zerlaut_mod.Zerlaut_adaptation_second_order):
    """TVB Zerlaut 2nd-order whose external afferent rate is read from a shared OU trace.

    Instance attributes are attached externally by :func:`_attach_ou_drive`:

    * ``_ou_trace_khz``  : np.ndarray, OU trace in KHz (Hz/1000)
    * ``_dfun_call_count``: int, number of dfun calls so far
    """

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        trace = getattr(self, "_ou_trace_khz", None)
        if trace is not None and trace.size:
            # Heun calls dfun twice per integrator step → divide by 2.
            idx = min(self._dfun_call_count // 2, trace.size - 1)
            drive_khz = float(trace[idx])
            n_regions = int(state_variables.shape[1])
            shape = np.full(n_regions, drive_khz, dtype=float)
            self.external_input_ex_ex = shape
            self.external_input_in_ex = shape.copy()
            self._dfun_call_count += 1
        return super().dfun(state_variables, coupling, local_coupling)


def _attach_ou_drive(model, ou_trace_hz: np.ndarray) -> object:
    """Swap a stock 2nd-order Zerlaut model to :class:`_Zerlaut2OUDrive` in-place."""
    if not isinstance(model, _zerlaut_mod.Zerlaut_adaptation_second_order):
        raise TypeError(
            f"OU drive injection requires Zerlaut_adaptation_second_order, "
            f"got {type(model).__name__}"
        )
    model.__class__ = _Zerlaut2OUDrive
    model._ou_trace_khz = np.asarray(ou_trace_hz, dtype=float) * 1e-3
    model._dfun_call_count = 0
    return model


@contextmanager
def _patched_zerlaut_with_ou(ou_trace_hz: np.ndarray):
    """Temporarily monkey-patch ``_select_zerlaut_model`` to inject OU drive."""
    original = _wb_sim_module._select_zerlaut_model

    def patched(pm):
        model = original(pm)
        _attach_ou_drive(model, ou_trace_hz)
        return model

    _wb_sim_module._select_zerlaut_model = patched
    try:
        yield
    finally:
        _wb_sim_module._select_zerlaut_model = original


# ---------------------------------------------------------------------------
# Shared OU trace — called by SNN and MF/WB runners
# ---------------------------------------------------------------------------

def make_shared_ou_traces() -> tuple[np.ndarray, np.ndarray]:
    """Build the two OU drive traces — one for SNN+MF, one for WB.

    Both use the identical OU **noise realization** (same seed → same ξ(t))
    but **different baselines** per paper Fig 4a:

    * SNN + single-region MF baseline = :data:`SNN_MF_DRIVE_HZ` (0.4 Hz)
    * Whole-brain baseline           = :data:`WB_DRIVE_HZ`     (0.315 Hz)

    Returns ``(snn_mf_trace_hz, wb_trace_hz)``.
    """
    sigma_runner_hz = paper_sigma_to_runner_sigma_hz(
        paper_sigma=PAPER_SIGMA_HZ, tau_ou_ms=TAU_OU_MS, dt_ms=DRIVE_TRACE_DT_MS
    )
    common = dict(
        duration_ms=SIM_DURATION_MS,
        dt_ms=DRIVE_TRACE_DT_MS,
        seed=SEED,                # SAME seed → same OU noise realization in both
        sigma_ou_hz=sigma_runner_hz,
        tau_ou_ms=TAU_OU_MS,
    )
    snn_mf_trace = build_shared_afferent_trace_hz(v_drive_hz=SNN_MF_DRIVE_HZ, **common)
    wb_trace     = build_shared_afferent_trace_hz(v_drive_hz=WB_DRIVE_HZ,    **common)
    return snn_mf_trace, wb_trace


# ---------------------------------------------------------------------------
# Component runners
# ---------------------------------------------------------------------------

def run_snn_block(b_e_pa: int, *, snn_mf_trace_hz: np.ndarray) -> tuple[dict, float]:
    """Spiking AdEx network (Brian2) with the shared OU drive at baseline = SNN_MF_DRIVE_HZ."""
    t0 = time.time()
    snn = run_adex_network_simulation(
        cells="FS-RS_10",
        seed_value=SEED,
        time_ms=SIM_DURATION_MS,
        iext_hz=SNN_MF_DRIVE_HZ,    # label only — overridden by trace below
        input_hz=0.0,
        external_rate_hz_trace=snn_mf_trace_hz,
        external_rate_dt_ms=DRIVE_TRACE_DT_MS,
        dt_ms=DT_MS,
        bin_width_ms=SNN_BIN_MS,
        parameter_overrides={
            "b_e": b_e_pa,
            "EL_e": EL_E_MV,
            "EL_i": EL_I_MV,
            "tau_e": TAU_E_MS,
            "tau_i": TAU_I_MS,
            "Ntot": SNN_NTOT,
            "gei": SNN_GEI,
            "p_con": SNN_P_CON,
            "tau_w": 500,
            "Cm": 200,
            "Gl": 10,
            "Q_e": 1.5,
            "Q_i": 5.0,
        },
        split_leak=False,
    )
    t_snn = time.time() - t0
    t_kept, exc_kept, inh_kept, adapt_kept = _trim_transient_snn(
        snn.time_ms, snn.rate_exc_hz, snn.rate_inh_hz,
        snn.adaptation if snn.adaptation is not None else np.zeros_like(snn.time_ms),
        cut_ms=CUT_TRANSIENT_MS,
    )
    out = {
        "time_ms": t_kept,
        "rate_exc_hz": exc_kept,
        "rate_inh_hz": inh_kept,
        "adaptation_pa": adapt_kept,
        "raster_exc": snn.raster_exc,
        "raster_inh": snn.raster_inh,
        "n_exc": int((1.0 - SNN_GEI) * SNN_NTOT),
        "n_inh": int(SNN_GEI * SNN_NTOT),
        "parameters": snn.parameters,
        "afferent_trace_hz": snn_mf_trace_hz,
        "afferent_dt_ms": DRIVE_TRACE_DT_MS,
    }
    print(
        f"  SNN done in {t_snn:5.1f}s  "
        f"(exc {exc_kept.mean():.2f}±{exc_kept.std():.2f} Hz, "
        f"inh {inh_kept.mean():.2f}±{inh_kept.std():.2f} Hz)"
    )
    return out, t_snn


def _shared_zerlaut_overrides(b_e_pa: float, *, baseline_hz: float) -> dict:
    """TVB Zerlaut 2nd-order model overrides shared by single-region MF + WB.

    The ``external_input_*`` baselines below are the *config-time* values; the
    per-step OU drive in :class:`_Zerlaut2OUDrive` overwrites them every
    integrator call, so the baseline matters only for the brief window before
    the first dfun invocation. Setting them to the trace mean keeps
    documentation tidy.

    Parameters
    ----------
    b_e_pa : adaptation b_e (pA) — the sweep variable
    baseline_hz : SNN_MF_DRIVE_HZ (0.4) for single-region MF; WB_DRIVE_HZ
        (0.315) for the whole-brain run. Both values come from the Fig 4a
        caption.
    """
    return {
        "b_e": float(b_e_pa),
        "E_L_e": EL_E_MV,
        "E_L_i": EL_I_MV,
        "tau_e_e": TAU_E_MS,
        "tau_e_i": TAU_I_MS,
        "T": MF_T_MS,
        "N_tot": SNN_NTOT,
        "g": SNN_GEI,
        "p_connect_e": SNN_P_CON,
        "p_connect_i": SNN_P_CON,
        "external_input_ex_ex": float(baseline_hz) * 1e-3,    # KHz baseline (OU adds fluctuation)
        "external_input_ex_in": 0.0,
        "external_input_in_ex": float(baseline_hz) * 1e-3,
        "external_input_in_in": 0.0,
        "K_ext_e": 400,
        "K_ext_i": 0,
        "tau_OU": TAU_OU_MS,
        "weight_noise": 0.0,                            # disable internal-noise contribution to rate
        "P_e": list(P_E_V2),
        "P_i": list(P_I_V2),
        "initial_condition": WB_INIT_COND,
    }


def run_mf_block(b_e_pa: int, *, snn_mf_trace_hz: np.ndarray) -> tuple[dict, float]:
    """Single-region MF (TVB Zerlaut 2nd-order) with OU drive at the SNN+MF baseline.

    Uses :func:`run_whole_brain_simulation` on a 1×1 zero-coupling
    connectivity so the same TVB 2nd-order code path runs as for the whole
    brain. OU drive is injected by :class:`_Zerlaut2OUDrive` (active inside
    the :func:`_patched_zerlaut_with_ou` context manager).

    Baseline drive: ``SNN_MF_DRIVE_HZ`` (0.4 Hz, paper Fig 4a).
    """
    t0 = time.time()
    cfg = WholeBrainConfig(
        simulation_length_ms=SIM_DURATION_MS,
        dt_ms=DT_MS,
        zerlaut_order=2,
        stochastic_integrator=False,                    # OU is external (deterministic Heun)
        coupling_strength=0.0,
        monitor_mode="raw",
        monitor_variables=(0, 1, 2, 3, 4, 5, 6, 7),    # E, I, C_ee, C_ei, C_ii, W_e, W_i, noise
        weights=np.zeros((1, 1), dtype=float),
        tract_lengths=np.zeros((1, 1), dtype=float),
        parameter_overrides=_shared_zerlaut_overrides(b_e_pa, baseline_hz=SNN_MF_DRIVE_HZ),
    )
    with _patched_zerlaut_with_ou(snn_mf_trace_hz):
        result = run_whole_brain_simulation(cfg, seed=SEED)
    t_mf = time.time() - t0
    _t, _data = result.full_monitor_output[0]
    _t = np.asarray(_t).reshape(-1)
    _data = np.asarray(_data)
    t_kept, data_kept = _trim_transient_tvb(_t, _data, CUT_TRANSIENT_MS)
    ve_hz = data_kept[:, 0, 0, 0] * 1e3
    vi_hz = data_kept[:, 1, 0, 0] * 1e3
    W_pa  = data_kept[:, 5, 0, 0]
    out = {
        "time_ms": t_kept,
        "ve_hz": ve_hz,
        "vi_hz": vi_hz,
        "W_pa":  W_pa,
    }
    print(f"  MF  done in {t_mf:5.1f}s  (ve {ve_hz.mean():.2f} Hz, W {W_pa.mean():.1f} pA)")
    return out, t_mf


def run_wb_block(b_e_pa: int, *, wb_trace_hz: np.ndarray) -> tuple[dict, float]:
    """Whole-brain DK-68 (TVB Zerlaut 2nd-order) with OU drive at the WB baseline.

    Baseline drive: ``WB_DRIVE_HZ`` (0.315 Hz, paper Fig 4a).
    """
    t0 = time.time()
    if not DK68_CONNECTIVITY_ZIP.exists():
        raise FileNotFoundError(f"DK-68 atlas missing at {DK68_CONNECTIVITY_ZIP}")
    cfg = WholeBrainConfig(
        simulation_length_ms=SIM_DURATION_MS,
        dt_ms=DT_MS,
        zerlaut_order=2,
        stochastic_integrator=False,                    # OU is external
        coupling_strength=WB_COUPLING_STRENGTH,
        connectivity_zip=DK68_CONNECTIVITY_ZIP,
        monitor_mode="raw",
        monitor_variables=(0, 1, 5),                    # E, I, W_e
        parameter_overrides=_shared_zerlaut_overrides(b_e_pa, baseline_hz=WB_DRIVE_HZ),
    )
    with _patched_zerlaut_with_ou(wb_trace_hz):
        result = run_whole_brain_simulation(cfg, seed=SEED)
    t_wb = time.time() - t0
    _t, _data = result.full_monitor_output[0]
    _t = np.asarray(_t).reshape(-1)
    _data = np.asarray(_data)
    t_kept, data_kept = _trim_transient_tvb(_t, _data, CUT_TRANSIENT_MS)
    ve_hz = data_kept[:, 0, :, 0] * 1e3
    vi_hz = data_kept[:, 1, :, 0] * 1e3
    W_pa  = data_kept[:, 2, :, 0]
    out = {
        "time_ms": t_kept,
        "ve_hz": ve_hz,
        "vi_hz": vi_hz,
        "W_pa": W_pa,
        "region_labels": np.asarray(result.region_labels),
    }
    print(f"  WB  done in {t_wb:5.1f}s  (ve {ve_hz.mean():.2f} Hz)")
    return out, t_wb


# ---------------------------------------------------------------------------
# Per-b runner with mode flags
# ---------------------------------------------------------------------------

def run_one_b(
    b_e_pa: int,
    *,
    force: bool = False,
    snn_only: bool = False,
    mf_only: bool = False,
    wb_only: bool = False,
) -> Path:
    out_path = PER_B_DIR / f"b_{b_e_pa}.pkl"
    partial = snn_only or mf_only or wb_only

    if out_path.exists() and not force and not partial:
        print(f"[b={b_e_pa:>3d}] already present at {out_path.name} — skipping")
        return out_path
    if partial and not out_path.exists():
        raise FileNotFoundError(
            f"--snn-only/--mf-only/--wb-only require an existing pickle; missing: {out_path}"
        )

    existing: dict = {}
    if partial:
        with open(out_path, "rb") as f:
            existing = pickle.load(f)
        print(f"[b={b_e_pa:>3d}] updating existing pickle ({list(existing.keys())})")

    timings = existing.get("timings_s", {})
    payload = existing if partial else {
        "b_e_pa": b_e_pa,
        "sim_duration_ms": SIM_DURATION_MS,
        "cut_transient_ms": CUT_TRANSIENT_MS,
    }

    # Two OU traces per b — same noise realization, different baselines
    # (paper Fig 4a: SNN+MF at 0.4 Hz, WB at 0.315 Hz).
    snn_mf_trace_hz, wb_trace_hz = make_shared_ou_traces()

    print(f"[b={b_e_pa:>3d}] starting at {time.strftime('%H:%M:%S')}")
    if not partial or snn_only:
        snn_dict, t_snn = run_snn_block(b_e_pa, snn_mf_trace_hz=snn_mf_trace_hz)
        payload["snn"] = snn_dict
        timings["snn"] = t_snn
    if not partial or mf_only:
        mf_dict, t_mf = run_mf_block(b_e_pa, snn_mf_trace_hz=snn_mf_trace_hz)
        payload["mf"] = mf_dict
        timings["mf"] = t_mf
    if not partial or wb_only:
        wb_dict, t_wb = run_wb_block(b_e_pa, wb_trace_hz=wb_trace_hz)
        payload["wb"] = wb_dict
        timings["wb"] = t_wb

    timings["total"] = sum(v for k, v in timings.items() if k != "total")
    payload["timings_s"] = timings

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[b={b_e_pa:>3d}] WROTE {out_path.name}  (total {timings['total']:.1f}s)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--b", type=int, default=None, help="Run a single b_e value (pA).")
    parser.add_argument("--all", action="store_true", help="Run the full sweep sequentially.")
    parser.add_argument("--force", action="store_true", help="Overwrite full pickle.")
    parser.add_argument("--snn-only", action="store_true",
                        help="Re-run SNN only; load existing pickle, replace snn block.")
    parser.add_argument("--mf-only", action="store_true",
                        help="Re-run single-region MF only; keep existing snn+wb blocks.")
    parser.add_argument("--wb-only", action="store_true",
                        help="Re-run whole-brain only; keep existing snn+mf blocks.")
    args = parser.parse_args()

    if args.b is not None and args.all:
        parser.error("Use --b OR --all, not both.")
    if args.b is None and not args.all:
        parser.error("Either --b <int> or --all is required.")
    if sum([args.snn_only, args.mf_only, args.wb_only]) > 1:
        parser.error("Pick at most one of --snn-only / --mf-only / --wb-only.")

    b_values = [args.b] if args.b is not None else list(B_VALUES_PA)
    for b in b_values:
        run_one_b(
            int(b),
            force=args.force,
            snn_only=args.snn_only,
            mf_only=args.mf_only,
            wb_only=args.wb_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
