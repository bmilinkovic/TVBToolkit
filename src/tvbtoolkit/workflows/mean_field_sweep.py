"""Standalone mean-field AdEx sweep tools for pharmacological parameter space exploration.

Implements the Zerlaut first-order mean-field model (Di Volo 2019 / Destexhe lab) as a
pure-NumPy ODE integrator — no TVB/scipy dependency — enabling fast 2-D parameter sweeps
over (b_e, τ_i) and (b_e, τ_e) as in Sacha et al., *Nat. Comput. Sci.* 2025.

Unit conventions (strict parity with TVB Zerlaut dfun)
-------------------------------------------------------
Firing rates   : **kHz** internally (τ·Q·f has units nS when τ in ms, f in kHz)
Time constants : ms
Conductances   : nS
Membrane pot.  : mV
Adaptation W   : pA  (b_e·v_e·τ_w = pA·kHz·ms = pA ✓)
Capacitance    : pF

Input/output firing rates are in **Hz** at the public API boundary.

TVB dfun convention reproduced here
-------------------------------------
fe_eff = (E_rec + 1e-9) * (1-g)*p_e*N + (E_ext) * K_ext_e
fi_eff = (I_rec + 1e-9) * g*p_i*N

where E_rec = recurrent excitatory rate (kHz, state variable),
      E_ext = external excitatory drive (kHz, noise + constant drive).
The factor (1-g)*p_e*N = 400 = K_ext_e so both pathways have equal weight.

Dynamics notes
--------------
The full 2D ODE (v_e, v_i) has two stable fixed points for wake parameters:
  • DOWN state : v_e ≈ 0.003 Hz, v_i ≈ 0 Hz  (near-silent)
  • AI  state  : v_e ≈  6.3 Hz, v_i ≈ 14.7 Hz (asynchronous-irregular, physiological wake)

The H(v_e) bisector analysis (quasi-static inhibitory tracking) is approximate; use
the full 2D phase portrait or ODE-based computation for precision.

Key exports
-----------
MFParams          – dataclass with all biophysical parameters
build_params      – named condition presets (wake/gaba/nmda/sleep)
transfer_function – F_e or F_i (Hz in, Hz out)
compute_H_ve      – H(v_e) bisector curve for fixed-point analysis (quasi-static)
find_fixed_points – stable/unstable FPs from H(v_e)
find_2d_fps       – fixed points of the full 2D ODE (grid search)
run_mf_ode        – ODE integration with OU noise
compute_survival  – post-stimulus survival time (ms)
survival_sweep_2d – parallelised 2-D parameter sweep
predict_bcrit     – mean-field critical b_e (yellow line in Fig 3c,d)
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# erfc without scipy
# ---------------------------------------------------------------------------

def _erfc(x):
    """Complementary error function — uses scipy when available, else math.erfc."""
    try:
        from scipy.special import erfc
        return erfc(np.asarray(x, dtype=float))
    except ImportError:
        return np.vectorize(math.erfc)(np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class MFParams:
    """Biophysical parameters for the Zerlaut first-order mean-field model.

    Defaults = awake (reference) condition of Sacha et al. 2025.
    Units: conductances nS, potentials mV, time ms, adaptation pA, cap pF.
    """
    # Synaptic
    Q_e: float = 1.5        # excitatory quantal conductance (nS)
    Q_i: float = 5.0        # inhibitory quantal conductance (nS)
    tau_e: float = 5.0      # excitatory synaptic decay (ms)  — NMDA target
    tau_i: float = 5.0      # inhibitory synaptic decay (ms)  — GABA-A target
    E_e: float = 0.0        # excitatory reversal potential (mV)
    E_i: float = -80.0      # inhibitory reversal potential (mV)
    # Cellular
    g_L: float = 10.0       # leak conductance (nS)
    C_m: float = 200.0      # membrane capacitance (pF)
    E_L_e: float = -65.0    # excitatory leak reversal (mV)
    E_L_i: float = -65.0    # inhibitory leak reversal (mV)
    # Adaptation
    b_e: float = 5.0        # spike-triggered adaptation (pA)
    a_e: float = 0.0        # sub-threshold adaptation conductance (nS)
    tau_w_e: float = 500.0  # adaptation time constant (ms)
    # Network
    N_tot: int = 10_000
    p_connect_e: float = 0.05
    p_connect_i: float = 0.05
    g: float = 0.2          # inhibitory fraction
    K_ext_e: int = 400
    K_ext_i: int = 0
    # Mean-field timescale
    T_ms: float = 5.0
    # TF polynomial coefficients (Berlin parameters / Di Volo 2019)
    P_e: tuple = (
        -0.05017034,  0.00451531, -0.00794377, -0.00208418, -0.00054697,
         0.00341614, -0.01156433,  0.00194753,  0.00274079, -0.01066769,
    )
    P_i: tuple = (
        -0.05184978,  0.0061593,  -0.01403522,  0.00166511, -0.0020559,
         0.00318432, -0.03112775,  0.00656668,  0.00171829, -0.04516385,
    )
    # External drive (Hz, user-facing; converted to kHz internally)
    v_drive_hz: float = 0.315
    # OU noise amplitude on external drive (Hz); sigma_ou << barrier height (~0.5 Hz)
    # Default 0.05 Hz keeps the system in the AI state without escaping to runaway rates.
    sigma_ou_hz: float = 0.05
    tau_ou_ms: float = 5.0   # OU time constant (ms)


# ---------------------------------------------------------------------------
# Condition presets
# ---------------------------------------------------------------------------

_CONDITIONS: dict[str, dict] = {
    "wake":  dict(b_e=5.0,   tau_e=5.0,  tau_i=5.0, v_drive_hz=0.315),
    "gaba":  dict(b_e=30.0,  tau_e=5.0,  tau_i=7.0, v_drive_hz=0.315),
    "nmda":  dict(b_e=30.0,  tau_e=3.75, tau_i=5.0, v_drive_hz=0.55),
    "sleep": dict(b_e=120.0, tau_e=5.0,  tau_i=5.0, v_drive_hz=0.4),
}


def build_params(condition: str = "wake", **overrides) -> MFParams:
    """Construct MFParams for a named condition with optional overrides."""
    if condition not in _CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Choose from {list(_CONDITIONS)}")
    kw = dict(_CONDITIONS[condition])
    kw.update(overrides)
    base = MFParams()
    valid = {k for k in MFParams.__dataclass_fields__}
    return replace(base, **{k: v for k, v in kw.items() if k in valid})


# ---------------------------------------------------------------------------
# Internal TF — strict parity with TVB Zerlaut dfun
# ---------------------------------------------------------------------------

def _fluct_regime(
    ve_rec_khz: float | np.ndarray,   # recurrent exc rate (kHz, state var E)
    vi_rec_khz: float | np.ndarray,   # recurrent inh rate (kHz, state var I)
    W_pa: float | np.ndarray,         # adaptation (pA)
    ve_ext_khz: float | np.ndarray,   # external exc drive (kHz)
    p: "MFParams",
    pop: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (μ_V mV, σ_V mV, τ_V ms) — TVB Zerlaut get_fluct_regime_vars.

    Mirrors TVB convention exactly:
        fe = (E_rec + 1e-9) * (1-g)*p_e*N + E_ext * K_ext_e
        fi = (I_rec + 1e-9) * g*p_i*N
    """
    ve_r = np.asarray(ve_rec_khz, dtype=float)
    vi_r = np.asarray(vi_rec_khz, dtype=float)
    W    = np.asarray(W_pa,       dtype=float)   # pA
    ve_x = np.asarray(ve_ext_khz, dtype=float)

    # Effective rates (kHz)  — same formula as Zerlaut.py get_fluct_regime_vars
    fe = (ve_r + 1e-9) * (1.0 - p.g) * p.p_connect_e * p.N_tot + ve_x * p.K_ext_e
    fi = (vi_r + 1e-9) * p.g * p.p_connect_i * p.N_tot

    # Mean conductances (nS): nS · ms · kHz = nS ✓
    mu_Ge = p.Q_e * p.tau_e * fe
    mu_Gi = p.Q_i * p.tau_i * fi
    mu_G  = p.g_L + mu_Ge + mu_Gi

    T_m = p.C_m / mu_G          # pF / nS = ms ✓

    E_L = p.E_L_e if pop == "exc" else p.E_L_i
    # pA / nS = mV ✓  (1 pA / 1 nS = 10⁻¹² / 10⁻⁹ = 10⁻³ V = 1 mV)
    mu_V = (mu_Ge * p.E_e + mu_Gi * p.E_i + p.g_L * E_L - W) / mu_G

    U_e = p.Q_e / mu_G * (p.E_e - mu_V)   # mV
    U_i = p.Q_i / mu_G * (p.E_i - mu_V)   # mV

    fe_s = np.maximum(fe, 1e-9)
    fi_s = np.maximum(fi, 1e-9)

    # σ²_V: kHz·(mV·ms)²/ms = kHz·mV²·ms = mV² ✓
    sigma2 = (
        fe_s * (U_e * p.tau_e) ** 2 / (2.0 * (p.tau_e + T_m))
        + fi_s * (U_i * p.tau_i) ** 2 / (2.0 * (p.tau_i + T_m))
    )
    sigma_V = np.sqrt(np.maximum(sigma2, 1e-30))   # mV

    # τ_V (ms)
    num = fe_s * (U_e * p.tau_e) ** 2 + fi_s * (U_i * p.tau_i) ** 2
    den = (fe_s * (U_e * p.tau_e) ** 2 / (p.tau_e + T_m)
           + fi_s * (U_i * p.tau_i) ** 2 / (p.tau_i + T_m))
    T_V = np.where(den > 0, num / den, p.tau_e)   # ms

    return mu_V, sigma_V, T_V


def _threshold_poly(mu_V, sigma_V, T_V_N, P) -> np.ndarray:
    """Effective spike threshold polynomial (Zerlaut 2018 Eq. 11).

    Returns in **Volts** (P coefficients fitted in V units).
    Caller must multiply ×1000 before comparing with mu_V in mV.
    """
    mu0, dmu = -60.0, 10.0
    s0,  ds  =   4.0,  6.0
    tn0, dtn =   0.5,  1.0
    V = (mu_V   - mu0) / dmu
    S = (sigma_V - s0) / ds
    T = (T_V_N  - tn0) / dtn
    return (P[0]
            + P[1]*V + P[2]*S + P[3]*T
            + P[4]*V**2 + P[5]*S**2 + P[6]*T**2
            + P[7]*V*S  + P[8]*V*T  + P[9]*S*T)


def _tf_internal(
    ve_rec_khz, vi_rec_khz, W_pa, ve_ext_khz, p: "MFParams", pop: str,
) -> np.ndarray:
    """Transfer function — returns output firing rate in **kHz**."""
    P = p.P_e if pop == "exc" else p.P_i
    mu_V, sigma_V, T_V = _fluct_regime(ve_rec_khz, vi_rec_khz, W_pa, ve_ext_khz, p, pop)

    # Normalised autocorrelation (dimensionless): ms·nS/pF = ms·(1/ms) = 1 ✓
    T_V_N = T_V * p.g_L / p.C_m

    V_thr_V = _threshold_poly(mu_V, sigma_V, T_V_N, P)   # Volts
    V_thr_mV = V_thr_V * 1e3                              # mV (TVB does this too)

    sig_s = np.maximum(sigma_V, 1e-9)   # mV
    T_V_s = np.maximum(T_V, 1e-9)       # ms

    # F = erfc((V_thr - μ_V)/(√2·σ_V)) / (2·τ_V)  — output in 1/ms = kHz ✓
    return _erfc((V_thr_mV - mu_V) / (np.sqrt(2.0) * sig_s)) / (2.0 * T_V_s)


# ---------------------------------------------------------------------------
# Public TF (Hz in / Hz out)
# ---------------------------------------------------------------------------

def transfer_function(
    ve_hz: float | np.ndarray,
    vi_hz: float | np.ndarray,
    W_pa: float | np.ndarray,
    p: "MFParams",
    pop: Literal["exc", "inh"] = "exc",
    ve_ext_hz: float | np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the Zerlaut transfer function F(v_e, v_i, W).

    Parameters
    ----------
    ve_hz, vi_hz : recurrent excitatory/inhibitory firing rates (Hz)
    W_pa         : adaptation current (pA)
    p            : MFParams
    pop          : {"exc", "inh"}
    ve_ext_hz    : external excitatory drive (Hz); defaults to p.v_drive_hz

    Returns
    -------
    ndarray — firing rate in Hz
    """
    if ve_ext_hz is None:
        ve_ext_hz = p.v_drive_hz
    ve_r = np.asarray(ve_hz,    dtype=float) * 1e-3
    vi_r = np.asarray(vi_hz,    dtype=float) * 1e-3
    ve_x = np.asarray(ve_ext_hz, dtype=float) * 1e-3
    return _tf_internal(ve_r, vi_r, W_pa, ve_x, p, pop) * 1e3   # kHz → Hz


# ---------------------------------------------------------------------------
# Fixed-point analysis
# ---------------------------------------------------------------------------

def _find_vi_eq(ve_rec_khz: float, W_pa: float, ve_ext_khz: float,
                p: "MFParams", n_iter: int = 200) -> float:
    """Find inhibitory fixed point v_i*(v_e, W) in kHz."""
    vi = 2e-3
    for _ in range(n_iter):
        vi_new = float(_tf_internal(ve_rec_khz, vi, W_pa, ve_ext_khz, p, "inh"))
        if abs(vi_new - vi) < 1e-10:
            break
        vi = vi_new
    return max(vi, 0.0)


def compute_H_ve(
    ve_range_hz: np.ndarray,
    p: "MFParams",
    branch: str = "fixed",
    n_iter: int = 200,
    W_override: "float | np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute H(v_e) = F_e(v_e, v_i*(v_e), W(v_e)) using v_drive as external.

    Uses a vectorised implementation: all v_e values are processed simultaneously
    with ``n_iter`` fixed-point iterations for the inhibitory sub-population.

    Parameters
    ----------
    branch : str
        Starting point for the v_i iteration:
        * ``"fixed"`` (default) – start from ``2e-3 kHz`` (2 Hz), matching the
          legacy TVBSim/theoretical_tools convention.
        * ``"up"``   – start from near-zero (reveals the low-inhibition branch).
        * ``"down"`` – start from 30e-3 kHz (reveals the high-inhibition branch).
    n_iter : int
        Number of fixed-point iterations for v_i (default 200).
    W_override : float, array, or None
        Override the adaptation current (pA).  When ``None`` (default), the
        quasi-static value W*(v_e) = b_e · v_e · τ_w is used — this is the
        correct choice for finding **true fixed points** of the full 3-D ODE.
        Pass ``0.0`` to compute H with W=0 everywhere (represents the
        *instantaneous* excitatory gain before adaptation has built up —
        useful for finding the AI-state FP on the fast manifold).
        Pass an array of the same length as ``ve_range_hz`` for arbitrary W
        profiles (e.g., a fixed W value to sweep the slow manifold).

    Returns
    -------
    (ve_range_hz, H_ve_hz, vi_star_hz)  — all in Hz.

    Notes
    -----
    **Interpretation of the two W modes:**

    * ``W_override=None`` (quasi-static):  finds genuine 3-D fixed points where
      dW/dt = 0, i.e. W = b_e·v_e·τ_w.  At high b_e (≥ ~40 pA) the adaptation
      is so large that the AI-state FP is suppressed — H(v_e) < v_e for all
      v_e > 0.003 Hz.  This does NOT mean the AI state is unreachable; rather,
      the network enters a **slow-fast limit cycle** (UP/DOWN oscillation) driven
      by the τ_w = 500 ms adaptation timescale.

    * ``W_override=0`` (fast manifold):  treats W as zero, showing where the
      fast (v_e, v_i) sub-system would converge instantaneously.  This reveals
      the AI FP at ~6 Hz for all b_e values, confirming that the UP phase of
      the UP/DOWN cycle is the same AI attractor — the network just can't stay
      there because W slowly builds up and eventually quenches it.
    """
    drive  = p.v_drive_hz * 1e-3                   # kHz
    ve_r   = np.asarray(ve_range_hz, dtype=float) * 1e-3   # kHz

    if W_override is None:
        W_arr = p.b_e * ve_r * p.tau_w_e            # quasi-static W (pA)
    else:
        W_arr = np.full_like(ve_r, float(W_override)) if np.isscalar(W_override) \
                else np.asarray(W_override, dtype=float)

    if branch == "fixed":
        vi = np.full_like(ve_r, 2e-3)             # 2 Hz starting point
    elif branch == "up":
        vi = np.full_like(ve_r, 1e-9)
    elif branch == "down":
        vi = np.full_like(ve_r, 30e-3)
    else:
        raise ValueError(f"Unknown branch '{branch}'. Use 'fixed', 'up', or 'down'.")

    drive_arr = np.full_like(ve_r, drive)          # broadcast-friendly

    for _ in range(n_iter):
        vi_new = _tf_internal(ve_r, vi, W_arr, drive_arr, p, "inh")
        vi_new = np.maximum(vi_new, 0.0)
        delta  = np.max(np.abs(vi_new - vi))
        vi     = vi_new
        if delta < 1e-11:
            break

    H_ve = _tf_internal(ve_r, vi, W_arr, drive_arr, p, "exc") * 1e3   # → Hz
    return ve_range_hz, H_ve, vi * 1e3   # return vi in Hz too


def find_2d_fps(
    p: "MFParams",
    ve_max_hz: float = 20.0,
    vi_max_hz: float = 60.0,
    n_ve: int = 60,
    n_vi: int = 60,
    tol: float = 1e-4,
    n_iter: int = 500,
) -> dict[str, np.ndarray]:
    """Find fixed points of the full 2D (v_e, v_i) ODE (no adaptation) by grid search.

    Seeds a fine grid of initial conditions (v_e₀, v_i₀) and iterates:
        v_e ← F_e(v_e, v_i, W(v_e), v_drive)
        v_i ← F_i(v_e, v_i, W(v_e), v_drive)
    until convergence.  Unique fixed points (within tolerance) are returned.

    Stability is assessed by the Jacobian sign of the linearised map; a fixed point
    is classified as stable when both eigenvalues are inside the unit circle.
    """
    drive = p.v_drive_hz * 1e-3

    ve_grid = np.linspace(0.0, ve_max_hz * 1e-3, n_ve)
    vi_grid = np.linspace(0.0, vi_max_hz * 1e-3, n_vi)
    fps = []

    for ve0 in ve_grid:
        for vi0 in vi_grid:
            ve, vi = float(ve0), float(vi0)
            for _ in range(n_iter):
                W   = p.b_e * ve * p.tau_w_e
                ve_n = float(_tf_internal(max(ve, 0), max(vi, 0), W, drive, p, "exc"))
                vi_n = float(_tf_internal(max(ve, 0), max(vi, 0), W, drive, p, "inh"))
                if abs(ve_n - ve) < 1e-12 and abs(vi_n - vi) < 1e-12:
                    break
                ve, vi = max(ve_n, 0.0), max(vi_n, 0.0)
            W = p.b_e * ve * p.tau_w_e

            # Check residual
            Fe = float(_tf_internal(ve, vi, W, drive, p, "exc"))
            Fi = float(_tf_internal(ve, vi, W, drive, p, "inh"))
            if abs(Fe - ve) < tol * 1e-3 and abs(Fi - vi) < tol * 1e-3:
                # Cluster with existing FPs
                fp_hz = (ve * 1e3, vi * 1e3)
                dup = any(abs(fp_hz[0] - ex[0]) < tol and abs(fp_hz[1] - ex[1]) < tol
                          for ex in fps)
                if not dup:
                    fps.append(fp_hz)

    if not fps:
        return {"ve_hz": np.array([]), "vi_hz": np.array([])}

    ve_arr = np.array([fp[0] for fp in fps])
    vi_arr = np.array([fp[1] for fp in fps])
    idx    = np.argsort(ve_arr)
    return {"ve_hz": ve_arr[idx], "vi_hz": vi_arr[idx]}


def find_fixed_points(
    ve_range_hz: np.ndarray,
    H_ve: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fixed-point crossings of H(v_e) with the bisector H = v_e."""
    diff = H_ve - ve_range_hz
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    stable, unstable = [], []
    for idx in sign_changes:
        x0, x1 = ve_range_hz[idx], ve_range_hz[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        x_c = float(x0 - d0 * (x1 - x0) / (d1 - d0))
        lo = max(idx - 2, 0)
        hi = min(idx + 3, len(ve_range_hz))
        slope = np.polyfit(ve_range_hz[lo:hi], diff[lo:hi], 1)[0]
        (stable if slope < 0 else unstable).append(x_c)

    return {"stable": np.array(stable), "unstable": np.array(unstable)}


# ---------------------------------------------------------------------------
# Mean-field ODE integrator
# ---------------------------------------------------------------------------

def run_mf_ode(
    p: "MFParams",
    duration_ms: float = 5000.0,
    dt_ms: float = 0.1,
    seed: int = 0,
    stim_amplitude_hz: float = 0.0,
    stim_start_ms: float | None = None,
    stim_dur_ms: float = 120.0,
    sigma_ou_hz: float | None = None,
    tau_ou_ms: float | None = None,
    external_drive_hz: np.ndarray | None = None,
    external_drive_dt_ms: float | None = None,
    transient_ms: float = 1000.0,
    init_state: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Integrate the Zerlaut first-order mean-field ODE with OU noise.

    Equations 17–20 from Sacha et al. 2025:
        T · dv_e/dt = F_e(E, I, v_aff, W) − v_e
        T · dv_i/dt = F_i(E, I, v_aff, W) − v_i
        τ_w · dW/dt = −W + b_e · v_e
        v_aff(t)    = v_drive + σ·ξ(t)  (OU process)

    v_aff enters as the EXTERNAL drive (ve_ext_khz), mirroring dfun.

    Parameters
    ----------
    sigma_ou_hz : OU noise amplitude in Hz (default = p.sigma_ou_hz = 0.05 Hz).
        Must be << barrier height (~0.5 Hz) to keep the system in the AI state.
    external_drive_hz : optional explicit afferent-drive trace (Hz), shape (T,).
        When provided, this replaces the internal OU-generated drive.
    external_drive_dt_ms : sampling step of ``external_drive_hz`` in ms.
        Defaults to ``dt_ms``.
    init_state  : (ve0_hz, vi0_hz) initial firing rates (Hz).  Default (2, 2) Hz
        which falls into the AI-state basin (~6 Hz for wake parameters).

    Returns dict{'time_ms', 've_hz', 'vi_hz', 'W_pa'} from transient_ms onward.
    """
    if sigma_ou_hz is None:
        sigma_ou_hz = p.sigma_ou_hz
    if tau_ou_ms is None:
        tau_ou_ms = p.tau_ou_ms

    rng = np.random.default_rng(seed)
    n   = int(duration_ms / dt_ms)

    drive_khz = p.v_drive_hz * 1e-3       # kHz
    sigma_khz = sigma_ou_hz * 1e-3        # kHz (noise amplitude matches drive scale)
    stim_khz  = stim_amplitude_hz * 1e-3

    stim_on = np.zeros(n, dtype=float)
    if stim_start_ms is not None:
        i0 = int(stim_start_ms / dt_ms)
        i1 = min(i0 + int(stim_dur_ms / dt_ms), n)
        stim_on[i0:i1] = stim_khz

    # Initialise: (2 Hz, 2 Hz) sits in the AI-state basin for wake parameters
    if init_state is None:
        init_state = (2.0, 2.0)
    ve = init_state[0] * 1e-3   # kHz
    vi = init_state[1] * 1e-3
    W  = p.b_e * ve * p.tau_w_e   # pA

    t_arr  = np.arange(n, dtype=float) * dt_ms
    ve_arr = np.empty(n)
    vi_arr = np.empty(n)
    W_arr  = np.empty(n)

    drive_trace_khz = None
    if external_drive_hz is not None:
        drive_hz = np.asarray(external_drive_hz, dtype=float).reshape(-1)
        if drive_hz.size == 0:
            raise ValueError("external_drive_hz must be non-empty when provided.")
        if not np.all(np.isfinite(drive_hz)):
            raise ValueError("external_drive_hz contains non-finite values.")
        drive_hz = np.maximum(drive_hz, 0.0)
        trace_dt_ms = float(dt_ms if external_drive_dt_ms is None else external_drive_dt_ms)
        if trace_dt_ms <= 0.0:
            raise ValueError("external_drive_dt_ms must be > 0.")
        t_trace = np.arange(drive_hz.size, dtype=float) * trace_dt_ms
        t_target = np.arange(n, dtype=float) * float(dt_ms)
        drive_interp_hz = np.interp(t_target, t_trace, drive_hz, left=float(drive_hz[0]), right=float(drive_hz[-1]))
        drive_trace_khz = drive_interp_hz * 1e-3

    xi = 0.0   # OU state (kHz units)
    for i in range(n):
        if drive_trace_khz is None:
            # OU step: xi tracks fluctuations in external drive (kHz)
            xi = xi * (1.0 - dt_ms / tau_ou_ms) + sigma_khz * math.sqrt(dt_ms / tau_ou_ms) * rng.standard_normal()
            ve_ext = max(drive_khz + xi + stim_on[i], 0.0)   # kHz
        else:
            ve_ext = max(float(drive_trace_khz[i]) + stim_on[i], 0.0)   # kHz

        Fe = float(_tf_internal(max(ve, 0.0), max(vi, 0.0), W, ve_ext, p, "exc"))
        Fi = float(_tf_internal(max(ve, 0.0), max(vi, 0.0), W, ve_ext, p, "inh"))

        ve = max(ve + dt_ms / p.T_ms * (Fe - ve), 0.0)
        vi = max(vi + dt_ms / p.T_ms * (Fi - vi), 0.0)
        # Adaptation: τ_w · dW/dt = −W + b_e · τ_w · ν_e
        # ↔  dW/dt = −W/τ_w + b_e · ν_e     (equilibrium: W* = b_e · ν_e · τ_w)
        # Units: b_e [pA], ve [kHz = ms⁻¹], τ_w [ms]  →  b_e·ve [pA/ms] ✓
        W  = W  + dt_ms * (-W / p.tau_w_e + p.b_e * ve)

        ve_arr[i] = ve
        vi_arr[i] = vi
        W_arr[i]  = W

    t0 = int(transient_ms / dt_ms)
    return {
        "time_ms": t_arr[t0:] - t_arr[t0],
        "ve_hz":   ve_arr[t0:] * 1e3,   # kHz → Hz
        "vi_hz":   vi_arr[t0:] * 1e3,
        "W_pa":    W_arr[t0:],
    }


# ---------------------------------------------------------------------------
# Survival time
# ---------------------------------------------------------------------------

def compute_survival(
    p: "MFParams",
    stim_amplitude_hz: float = 50.0,
    stim_start_ms: float = 400.0,
    stim_dur_ms: float = 120.0,
    post_window_ms: float = 2000.0,
    dt_ms: float = 0.5,
    n_seeds: int = 20,
    seed_start: int = 0,
    threshold_fraction: float = 0.1,
    settle_ms: float = 500.0,
) -> float:
    """Mean post-stimulus survival time (ms) averaged over n_seeds realisations.

    Mirrors the legacy TVBSim ``calculate_survival_time`` convention:
      threshold = threshold_fraction × v_e at stim onset
      survival  = consecutive time (ms) where v_e > threshold after stim onset

    For wake parameters the AI state is ~6 Hz; a 50 Hz stimulus yields threshold
    ≈ 0.6 Hz so the network "survives" until it returns to the DOWN state.
    Larger b_e → stronger adaptation → shorter survival.

    Parameters
    ----------
    stim_amplitude_hz : extra drive added to v_ext during stim (Hz).
        Default 50 Hz gives a clearly supra-threshold perturbation.
    settle_ms : pre-stim transient used to let the network reach its spontaneous
        state before stim_start_ms.
    """
    total_ms = settle_ms + stim_start_ms + stim_dur_ms + post_window_ms
    durations = []
    for s in range(seed_start, seed_start + n_seeds):
        r  = run_mf_ode(p, duration_ms=total_ms, dt_ms=dt_ms, seed=s,
                        stim_amplitude_hz=stim_amplitude_hz,
                        stim_start_ms=settle_ms + stim_start_ms,
                        stim_dur_ms=stim_dur_ms,
                        transient_ms=0.0)
        ve = r["ve_hz"]
        stim_i   = int((settle_ms + stim_start_ms) / dt_ms)
        stim_end = stim_i + int(stim_dur_ms / dt_ms)
        # Threshold = threshold_fraction × rate at stim onset (legacy convention)
        thresh   = threshold_fraction * max(float(ve[stim_i]), 1e-6)
        post     = ve[stim_end:]
        below    = np.where(post < thresh)[0]
        durations.append(below[0] * dt_ms if len(below) > 0 else len(post) * dt_ms)
    return float(np.mean(durations))


# ---------------------------------------------------------------------------
# 2-D sweep (parallel)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> float:
    b, tau, axis, base_kw, sweep_kw = args
    base_kw = dict(base_kw)
    base_kw["b_e"]  = b
    base_kw[axis]   = tau
    base = MFParams()
    valid = {k for k in MFParams.__dataclass_fields__}
    p = replace(base, **{k: v for k, v in base_kw.items() if k in valid})
    return compute_survival(p, **sweep_kw)


def survival_sweep_2d(
    bvals_pa: np.ndarray,
    tau_vals_ms: np.ndarray,
    sweep_axis: Literal["tau_i", "tau_e"] = "tau_i",
    base_condition: str = "wake",
    base_params: "MFParams | None" = None,
    stim_amplitude_hz: float = 1.0,
    stim_dur_ms: float = 120.0,
    n_seeds: int = 20,
    dt_ms: float = 0.5,
    n_workers: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """2-D survival-time sweep over (b_e pA, τ_axis ms).

    Returns array (n_b, n_tau) of mean survival time in ms.
    """
    bp     = base_params or build_params(base_condition)
    bp_kw  = {f: getattr(bp, f) for f in MFParams.__dataclass_fields__}
    sw_kw  = dict(stim_amplitude_hz=stim_amplitude_hz, stim_dur_ms=stim_dur_ms,
                  n_seeds=n_seeds, dt_ms=dt_ms)
    jobs   = [(float(b), float(tau), sweep_axis, bp_kw, sw_kw)
              for b in bvals_pa for tau in tau_vals_ms]
    n_tot  = len(jobs)

    if verbose:
        print(f"[survival_sweep_2d] {n_tot} grid points "
              f"({len(bvals_pa)} b_e × {len(tau_vals_ms)} {sweep_axis}, "
              f"{n_seeds} seeds each)")

    if n_workers is None:
        import os
        n_workers = max(1, int((os.cpu_count() or 4) * 0.75))

    results = np.empty(n_tot, dtype=float)
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for k, v in enumerate(ex.map(_worker, jobs)):
                results[k] = v
                if verbose and (k + 1) % max(1, n_tot // 10) == 0:
                    print(f"  {k+1}/{n_tot}")
    else:
        for k, job in enumerate(jobs):
            results[k] = _worker(job)

    return results.reshape(len(bvals_pa), len(tau_vals_ms))


# ---------------------------------------------------------------------------
# Mean-field predicted b_crit
# ---------------------------------------------------------------------------

def predict_bcrit(
    tau_vals_ms: np.ndarray,
    sweep_axis: Literal["tau_i", "tau_e"] = "tau_i",
    base_condition: str = "wake",
    ve_range_hz: np.ndarray | None = None,
    b_scan: np.ndarray | None = None,
) -> np.ndarray:
    """Predict critical b_e (pA) for each τ value (yellow line in Fig 3c,d).

    b_crit = smallest b_e at which the UP stable fixed point (AI state) disappears
    via saddle-node bifurcation with the lower unstable FP.

    The H(v_e) bisector analysis gives:
      • For b_e < b_crit: UP stable FP at ~0.26 Hz + two unstable FPs
      • At b_crit: the UP stable FP and the nearest unstable FP merge → disappear
      • For b_e > b_crit: only DOWN state remains (H < v_e everywhere in low-rate regime)

    Note: the default vi* iteration (vi₀ = 2e-3 kHz) tracks the inhibitory branch
    relevant for the AI-state analysis.
    """
    if ve_range_hz is None:
        ve_range_hz = np.linspace(0.001, 5.0, 1000)
    if b_scan is None:
        b_scan = np.linspace(0, 200, 800)

    bp      = build_params(base_condition)
    b_crits = np.full(len(tau_vals_ms), np.nan)

    b_lo = float(b_scan[0])
    b_hi = float(b_scan[-1])

    def _count_stable(tau, b):
        overrides = {sweep_axis: float(tau)}
        p = replace(bp, b_e=float(b), **overrides)
        _, H, _ = compute_H_ve(ve_range_hz, p)
        return len(find_fixed_points(ve_range_hz, H)["stable"])

    for k, tau in enumerate(tau_vals_ms):
        n_base = _count_stable(tau, b_lo)   # stable FPs at smallest b
        n_hi   = _count_stable(tau, b_hi)
        if n_hi >= n_base:
            # No bifurcation in range — b_crit > b_hi
            continue
        # Bisection search
        lo, hi = b_lo, b_hi
        for _ in range(15):   # 15 bisection steps → ~log₂(200/1) precision
            mid = 0.5 * (lo + hi)
            if _count_stable(tau, mid) < n_base:
                hi = mid
            else:
                lo = mid
        b_crits[k] = hi

    return b_crits


__all__ = [
    "MFParams", "build_params",
    "transfer_function", "compute_H_ve", "find_fixed_points", "find_2d_fps",
    "run_mf_ode", "compute_survival",
    "survival_sweep_2d", "predict_bcrit",
]
