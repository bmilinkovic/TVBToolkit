#!/usr/bin/env python3
"""
Pure-numpy parity simulation: first-order vs second-order Zerlaut mean-field.

Implements the full ODE math from scratch (no TVB, no Brian2, no scipy).
Uses Euler integration on a single uncoupled node for speed.
This validates that:
  1. First-order produces a reasonable AI-regime fixed point.
  2. Second-order adds covariance corrections (C_ee, C_ei, C_ii) and
     converges to the same fixed point as first-order for small populations
     (finite-size corrections are O(1/N)).
  3. Coefficient ordering in the polynomial is correct.

Usage:
    python3 parity_numpy.py
"""

import numpy as np
from math import erfc, sqrt, exp

# ─── Parameters (from parameter_M_Berlin_new.py / TVBSim) ───────────────────
# Berlin/Fede config — CORRECT coefficient ordering:
#   threshold_func = P0 + P1*V + P2*S + P3*T + P4*V² + P5*S² + P6*T² + P7*VS + P8*VT + P9*ST
P_e = np.array([-0.05017034,  0.00451531, -0.00794377, -0.00208418, -0.00054697,
                  0.00341614, -0.01156433,  0.00194753,  0.00274079, -0.01066769])
P_i = np.array([-0.05184978,  0.0061593,  -0.01403522,  0.00166511, -0.0020559,
                  0.00318432, -0.03112775,  0.00656668,  0.00171829, -0.04516385])

# Incorrect/scrambled ordering from old notebook (for comparison)
P_e_WRONG = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00194753, 0.00274079, 0.00341614, -0.01066769, -0.01156433])
P_i_WRONG = np.array([-0.05184978, 0.0061593, -0.01403522, 0.00166511, -0.0020559,
                       0.00656668, 0.00171829, 0.00318432, -0.04516385, -0.03112775])

PARAMS = dict(
    g_L     = 10.0,    # nS
    E_L_e   = -64.0,   # mV  (excitatory leak reversal)
    E_L_i   = -65.0,   # mV  (inhibitory leak reversal)
    C_m     = 200.0,   # pF
    b_e     = 5.0,     # pA  (spike-triggered adaptation kick, excitatory)
    b_i     = 0.0,     # pA
    a_e     = 0.0,     # nS  (subthreshold adaptation conductance)
    a_i     = 0.0,     # nS
    tau_w_e = 500.0,   # ms
    tau_w_i = 1.0,     # ms
    E_e     = 0.0,     # mV  (exc reversal)
    E_i     = -80.0,   # mV  (inh reversal)
    Q_e     = 1.5,     # nS  (excitatory quantal conductance)
    Q_i     = 5.0,     # nS  (inhibitory quantal conductance)
    tau_e   = 5.0,     # ms  (excitatory synaptic decay)
    tau_i   = 5.0,     # ms  (inhibitory synaptic decay)
    N_tot   = 10000,   # total neuron count
    p_e     = 0.05,    # connection probability (excitatory)
    p_i     = 0.05,    # connection probability (inhibitory)
    g       = 0.2,     # fraction of inhibitory cells
    T       = 20.0,    # ms  (integration window)
    K_ext_e = 400,     # external excitatory connections
    K_ext_i = 0,       # external inhibitory connections
    # External drive (Hz → kHz conversion happens inside get_fluct)
    nu_ext_ee = 0.315e-3,   # kHz
    nu_ext_ei = 0.000,
    nu_ext_ie = 0.315e-3,
    nu_ext_ii = 0.000,
)


# ─── Core math (mirrors the JIT-compiled functions exactly) ──────────────────

def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W, Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                          g_L, C_m, E_L, N_tot, p_e, p_i, g, K_ext_e, K_ext_i):
    """
    Compute mean membrane potential μ_V, std σ_V, and autocorr time T_V
    for a given excitatory/inhibitory population.

    Equations from di Volo et al. 2018 / Neural Computation 2019.
    """
    fe = (Fe + 1e-6) * (1 - g) * p_e * N_tot + Fe_ext * K_ext_e   # total exc firing rate input
    fi = (Fi + 1e-6) *       g  * p_i * N_tot + Fi_ext * K_ext_i   # total inh firing rate input

    # Mean conductances (Eq. 5)
    mu_Ge = Q_e * tau_e * fe
    mu_Gi = Q_i * tau_i * fi
    mu_G  = g_L + mu_Ge + mu_Gi    # total mean conductance (Eq. 6)
    T_m   = C_m / mu_G             # effective membrane time constant (Eq. 6)

    # Mean membrane potential (Eq. 7)
    mu_V  = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G

    # Post-synaptic amplitude (conductance-based)
    U_e = Q_e / mu_G * (E_e - mu_V)
    U_i = Q_i / mu_G * (E_i - mu_V)

    # Voltage std (Eq. 8)
    sigma_V = sqrt(max(
        fe * (U_e * tau_e)**2 / (2 * (tau_e + T_m)) +
        fi * (U_i * tau_i)**2 / (2 * (tau_i + T_m)),
        1e-20
    ))

    # Autocorrelation time (Eq. 9)
    num = fe * (U_e * tau_e)**2 + fi * (U_i * tau_i)**2
    den = fe * (U_e * tau_e)**2 / (tau_e + T_m) + fi * (U_i * tau_i)**2 / (tau_i + T_m)
    T_V = num / max(den, 1e-20)

    return mu_V, sigma_V, T_V


def threshold_func(muV, sigmaV, TvN, P):
    """
    Phenomenological effective threshold (Eq. 11 in di Volo 2018).
    Polynomial in normalized (V, S, T) coordinates.

    P[0..9] maps to: const, V, S, T, V², S², T², VS, VT, ST
    where:
      V = (muV   + 60)  / 10
      S = (sigmaV -  4) /  6
      T = (TvN   - 0.5) /  1.0
    """
    muV0, DmuV0  = -60.0, 10.0
    sV0,  DsV0   =   4.0,  6.0
    TvN0, DTvN0  =   0.5,  1.0
    V = (muV   - muV0) / DmuV0
    S = (sigmaV - sV0) / DsV0
    T = (TvN   - TvN0) / DTvN0
    return (P[0] + P[1]*V + P[2]*S + P[3]*T
            + P[4]*V**2 + P[5]*S**2 + P[6]*T**2
            + P[7]*V*S  + P[8]*V*T  + P[9]*S*T)


def estimate_firing_rate(muV, sigmaV, Tv, V_thre):
    """
    Firing rate from erfc model (Eq. 10 in di Volo 2018).
    Returns rate in kHz.
    """
    arg = (V_thre - muV) / (sqrt(2) * max(sigmaV, 1e-12))
    return erfc(arg) / (2 * max(Tv, 1e-12))


def TF(fe, fi, fe_ext, fi_ext, W, P, E_L, is_excitatory, p):
    """Transfer function for one population."""
    tau_e = p['tau_e']
    if not is_excitatory:
        tau_e = p['tau_e']   # same tau_e for inhibitory projection onto inhibitory cells

    mu_V, sigma_V, T_V = get_fluct_regime_vars(
        fe, fi, fe_ext, fi_ext, W,
        p['Q_e'], tau_e, p['E_e'],
        p['Q_i'], p['tau_i'], p['E_i'],
        p['g_L'], p['C_m'], E_L,
        p['N_tot'], p['p_e'], p['p_i'], p['g'],
        p['K_ext_e'], p['K_ext_i']
    )
    # Normalize T_V by g_L/C_m (converts to dimensionless time)
    TvN = T_V * p['g_L'] / p['C_m']
    V_thre = threshold_func(mu_V, sigma_V, TvN, P)
    V_thre *= 1e3   # Volts → mV
    return estimate_firing_rate(mu_V, sigma_V, T_V, V_thre)


def TF_e(fe, fi, fe_ext, fi_ext, W_e, P, p):
    return TF(fe, fi, fe_ext, fi_ext, W_e, P, p['E_L_e'], True, p)

def TF_i(fe, fi, fe_ext, fi_ext, W_i, P, p):
    return TF(fe, fi, fe_ext, fi_ext, W_i, P, p['E_L_i'], False, p)


# ─── First-order ODE derivatives ─────────────────────────────────────────────

def dfun_first_order(state, fe_ext, fi_ext, P_e_arr, P_i_arr, p):
    """
    First-order mean-field system:
      T · dE/dt  = TF_e(E, I, ...) - E
      T · dI/dt  = TF_i(E, I, ...) - I
      dW_e/dt    = -W_e/τ_we + b_e·E + a_e·(μV - E_L_e)/τ_we
      dW_i/dt    = -W_i/τ_wi + b_i·I + a_i·(μV - E_L_i)/τ_wi
    """
    E, I, W_e, W_i = state
    T   = p['T']

    F_e = TF_e(E, I, fe_ext, fi_ext, W_e, P_e_arr, p)
    F_i = TF_i(E, I, fe_ext, fi_ext, W_i, P_i_arr, p)

    dE  = (F_e - E) / T
    dI  = (F_i - I) / T

    # Adaptation — excitatory
    mu_V_e, _, _ = get_fluct_regime_vars(
        E, I, fe_ext + p['nu_ext_ee'], fi_ext + p['nu_ext_ei'], W_e,
        p['Q_e'], p['tau_e'], p['E_e'],
        p['Q_i'], p['tau_i'], p['E_i'],
        p['g_L'], p['C_m'], p['E_L_e'],
        p['N_tot'], p['p_e'], p['p_i'], p['g'],
        p['K_ext_e'], p['K_ext_i']
    )
    dWe = -W_e / p['tau_w_e'] + p['b_e'] * E + p['a_e'] * (mu_V_e - p['E_L_e']) / p['tau_w_e']

    # Adaptation — inhibitory
    mu_V_i, _, _ = get_fluct_regime_vars(
        E, I, fe_ext + p['nu_ext_ie'], fi_ext + p['nu_ext_ii'], W_i,
        p['Q_e'], p['tau_e'], p['E_e'],
        p['Q_i'], p['tau_i'], p['E_i'],
        p['g_L'], p['C_m'], p['E_L_i'],
        p['N_tot'], p['p_e'], p['p_i'], p['g'],
        p['K_ext_e'], p['K_ext_i']
    )
    dWi = -W_i / p['tau_w_i'] + p['b_i'] * I + p['a_i'] * (mu_V_i - p['E_L_i']) / p['tau_w_i']

    return np.array([dE, dI, dWe, dWi])


# ─── Second-order ODE derivatives ────────────────────────────────────────────

def dfun_second_order(state, fe_ext, fi_ext, P_e_arr, P_i_arr, p):
    """
    Second-order mean-field system (adds C_ee, C_ei, C_ii covariances).

    Equations from Zerlaut 2018 (Eq. 6) / di Volo 2018:
      T · dE/dt    = (F_e - E) + ½C_ee·∂²F_e/∂E² + C_ei·∂²F_e/∂E∂I + ½C_ii·∂²F_e/∂I²
      T · dI/dt    = (F_i - I) + ½C_ee·∂²F_i/∂E² + C_ei·∂²F_i/∂E∂I + ½C_ii·∂²F_i/∂I²
      T · dC_ee/dt = F_e(1/T - F_e)/N_e + (F_e-E)² + 2C_ee·∂F_e/∂E + 2C_ei·∂F_e/∂I - 2C_ee
      T · dC_ei/dt = (F_e-E)(F_i-I) + C_ee·∂F_e/∂E + C_ei·∂F_e/∂I + C_ei·∂F_i/∂E + C_ii·∂F_i/∂I - 2C_ei
      T · dC_ii/dt = F_i(1/T - F_i)/N_i + (F_i-I)² + 2C_ii·∂F_i/∂I + 2C_ei·∂F_i/∂E - 2C_ii
      dW_e/dt      = -W_e/τ_we + b_e·E
      dW_i/dt      = -W_i/τ_wi + b_i·I
    """
    E, I, C_ee, C_ei, C_ii, W_e, W_i = state
    T   = p['T']
    N_e = p['N_tot'] * (1 - p['g'])
    N_i = p['N_tot'] * p['g']

    F_e = TF_e(E, I, fe_ext, fi_ext, W_e, P_e_arr, p)
    F_i = TF_i(E, I, fe_ext, fi_ext, W_i, P_i_arr, p)

    # Numerical first derivatives (central difference, spacing df)
    df = 1e-7
    scale = 2 * df * 1e3   # factor in denominator (rates in kHz, derivatives in 1/(kHz·ms))

    dF_e_dE = (TF_e(E+df, I, fe_ext, fi_ext, W_e, P_e_arr, p) -
               TF_e(E-df, I, fe_ext, fi_ext, W_e, P_e_arr, p)) / scale
    dF_e_dI = (TF_e(E, I+df, fe_ext, fi_ext, W_e, P_e_arr, p) -
               TF_e(E, I-df, fe_ext, fi_ext, W_e, P_e_arr, p)) / scale
    dF_i_dE = (TF_i(E+df, I, fe_ext, fi_ext, W_i, P_i_arr, p) -
               TF_i(E-df, I, fe_ext, fi_ext, W_i, P_i_arr, p)) / scale
    dF_i_dI = (TF_i(E, I+df, fe_ext, fi_ext, W_i, P_i_arr, p) -
               TF_i(E, I-df, fe_ext, fi_ext, W_i, P_i_arr, p)) / scale

    # Second derivatives
    d2F_e_dE2 = (TF_e(E+df, I, fe_ext, fi_ext, W_e, P_e_arr, p) - 2*F_e +
                 TF_e(E-df, I, fe_ext, fi_ext, W_e, P_e_arr, p)) / (df*1e3)**2
    d2F_e_dI2 = (TF_e(E, I+df, fe_ext, fi_ext, W_e, P_e_arr, p) - 2*F_e +
                 TF_e(E, I-df, fe_ext, fi_ext, W_e, P_e_arr, p)) / (df*1e3)**2
    d2F_e_dEdI = (
        (TF_e(E+df, I+df, fe_ext, fi_ext, W_e, P_e_arr, p) -
         TF_e(E+df, I-df, fe_ext, fi_ext, W_e, P_e_arr, p)) -
        (TF_e(E-df, I+df, fe_ext, fi_ext, W_e, P_e_arr, p) -
         TF_e(E-df, I-df, fe_ext, fi_ext, W_e, P_e_arr, p))
    ) / (4 * (df*1e3)**2)

    d2F_i_dE2 = (TF_i(E+df, I, fe_ext, fi_ext, W_i, P_i_arr, p) - 2*F_i +
                 TF_i(E-df, I, fe_ext, fi_ext, W_i, P_i_arr, p)) / (df*1e3)**2
    d2F_i_dI2 = (TF_i(E, I+df, fe_ext, fi_ext, W_i, P_i_arr, p) - 2*F_i +
                 TF_i(E, I-df, fe_ext, fi_ext, W_i, P_i_arr, p)) / (df*1e3)**2
    d2F_i_dEdI = (
        (TF_i(E+df, I+df, fe_ext, fi_ext, W_i, P_i_arr, p) -
         TF_i(E+df, I-df, fe_ext, fi_ext, W_i, P_i_arr, p)) -
        (TF_i(E-df, I+df, fe_ext, fi_ext, W_i, P_i_arr, p) -
         TF_i(E-df, I-df, fe_ext, fi_ext, W_i, P_i_arr, p))
    ) / (4 * (df*1e3)**2)

    # ── Firing rate ODEs (with second-order correction) ──────────────────────
    dE = (F_e - E
          + 0.5 * C_ee * d2F_e_dE2
          + C_ei * d2F_e_dEdI           # symmetric: ∂²F_e/∂E∂I = ∂²F_e/∂I∂E
          + 0.5 * C_ii * d2F_e_dI2
          ) / T

    dI = (F_i - I
          + 0.5 * C_ee * d2F_i_dE2
          + C_ei * d2F_i_dEdI
          + 0.5 * C_ii * d2F_i_dI2
          ) / T

    # ── Covariance ODEs ──────────────────────────────────────────────────────
    # dC_ee/dt: A_ee + (F_e - E)² + 2C_ee·∂F_e/∂E + 2C_ei·∂F_e/∂I - 2C_ee
    dC_ee = (F_e * (1/T - F_e) / N_e
             + (F_e - E)**2
             + 2 * C_ee * dF_e_dE
             + 2 * C_ei * dF_e_dI      # ∂F_e/∂I  (CORRECT — excitatory TF)
             - 2 * C_ee
             ) / T

    # dC_ei/dt: (F_e-E)(F_i-I) + C_ee·∂F_e/∂E + C_ei·∂F_e/∂I + C_ei·∂F_i/∂E + C_ii·∂F_i/∂I - 2C_ei
    dC_ei = ((F_e - E) * (F_i - I)
             + C_ee * dF_e_dE
             + C_ei * dF_e_dI
             + C_ei * dF_i_dE
             + C_ii * dF_i_dI
             - 2 * C_ei
             ) / T

    # dC_ii/dt: A_ii + (F_i - I)² + 2C_ii·∂F_i/∂I + 2C_ei·∂F_i/∂E - 2C_ii
    dC_ii = (F_i * (1/T - F_i) / N_i
             + (F_i - I)**2
             + 2 * C_ii * dF_i_dI
             + 2 * C_ei * dF_i_dE      # ∂F_i/∂E  (CORRECT — inhibitory TF)
             - 2 * C_ii
             ) / T

    # ── Adaptation ODEs ──────────────────────────────────────────────────────
    mu_V_e, _, _ = get_fluct_regime_vars(
        E, I, fe_ext, fi_ext, W_e,
        p['Q_e'], p['tau_e'], p['E_e'],
        p['Q_i'], p['tau_i'], p['E_i'],
        p['g_L'], p['C_m'], p['E_L_e'],
        p['N_tot'], p['p_e'], p['p_i'], p['g'],
        p['K_ext_e'], p['K_ext_i']
    )
    mu_V_i, _, _ = get_fluct_regime_vars(
        E, I, fe_ext, fi_ext, W_i,
        p['Q_e'], p['tau_e'], p['E_e'],
        p['Q_i'], p['tau_i'], p['E_i'],
        p['g_L'], p['C_m'], p['E_L_i'],
        p['N_tot'], p['p_e'], p['p_i'], p['g'],
        p['K_ext_e'], p['K_ext_i']
    )
    dWe = -W_e / p['tau_w_e'] + p['b_e'] * E + p['a_e'] * (mu_V_e - p['E_L_e']) / p['tau_w_e']
    dWi = -W_i / p['tau_w_i'] + p['b_i'] * I + p['a_i'] * (mu_V_i - p['E_L_i']) / p['tau_w_i']

    return np.array([dE, dI, dC_ee, dC_ei, dC_ii, dWe, dWi])


# ─── Euler integrator ─────────────────────────────────────────────────────────

def euler_integrate(dfun, state0, dt, n_steps, fe_ext, fi_ext, P_e_arr, P_i_arr, p,
                    record_every=100):
    """Integrate ODE system using Euler method. Returns (times, trajectory)."""
    state = state0.copy()
    times  = []
    traj   = []
    for i in range(n_steps):
        if i % record_every == 0:
            times.append(i * dt)
            traj.append(state.copy())
        deriv = dfun(state, fe_ext, fi_ext, P_e_arr, P_i_arr, p)
        state = state + dt * deriv
        # Clamp firing rates to [0, ∞) — biological lower bound
        state[0] = max(state[0], 0.0)
        state[1] = max(state[1], 0.0)
    return np.array(times), np.array(traj)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import time as _time

    p = PARAMS

    DT       = 0.1     # ms
    T_SIM    = 5000.0  # ms — long enough to reach steady state
    N_STEPS  = int(T_SIM / DT)
    REC_EVERY = 10     # record every 10 steps = every 1 ms

    # External inputs (passed straight through — no long-range coupling in single-node)
    fe_ext = p['nu_ext_ee']
    fi_ext = 0.0

    # Initial conditions
    state0_1st = np.array([0.001, 0.001, 100.0, 0.0])   # [E, I, W_e, W_i]  (kHz, kHz, pA, pA)
    state0_2nd = np.array([0.001, 0.001, 0.0, 0.0, 0.0, 100.0, 0.0])  # [E,I,Cee,Cei,Cii,We,Wi]

    print("=" * 72)
    print("PARITY CHECK: First-order vs Second-order Zerlaut mean-field")
    print("(single-node, no long-range coupling, pure-numpy Euler integration)")
    print("=" * 72)
    print(f"  Simulation: {T_SIM} ms, dt={DT} ms, {N_STEPS} steps")
    print()

    # ── Run 1: first-order with CORRECT P coefficients ────────────────────────
    print("Running order=1 (correct P_e/P_i) ... ", end="", flush=True)
    t0 = _time.perf_counter()
    times_1, traj_1 = euler_integrate(
        dfun_first_order, state0_1st, DT, N_STEPS, fe_ext, fi_ext,
        P_e, P_i, p, record_every=REC_EVERY
    )
    t_elapsed_1 = _time.perf_counter() - t0
    print(f"done in {t_elapsed_1:.2f}s")

    # ── Run 2: second-order with CORRECT P coefficients ───────────────────────
    print("Running order=2 (correct P_e/P_i) ... ", end="", flush=True)
    t0 = _time.perf_counter()
    times_2, traj_2 = euler_integrate(
        dfun_second_order, state0_2nd, DT, N_STEPS, fe_ext, fi_ext,
        P_e, P_i, p, record_every=REC_EVERY
    )
    t_elapsed_2 = _time.perf_counter() - t0
    print(f"done in {t_elapsed_2:.2f}s  (×{t_elapsed_2/t_elapsed_1:.2f} vs order=1)")

    # ── Run 3: first-order with WRONG/scrambled P coefficients (old notebook) ─
    print("Running order=1 (WRONG P_e/P_i — old notebook) ... ", end="", flush=True)
    t0 = _time.perf_counter()
    times_1w, traj_1w = euler_integrate(
        dfun_first_order, state0_1st, DT, N_STEPS, fe_ext, fi_ext,
        P_e_WRONG, P_i_WRONG, p, record_every=REC_EVERY
    )
    print(f"done in {_time.perf_counter()-t0:.2f}s")

    # ── Extract steady-state values (last 20% of simulation) ──────────────────
    cut = int(0.8 * len(times_1))

    def ss(traj, idx, cut):
        return float(np.mean(traj[cut:, idx]))

    E1  = ss(traj_1,  0, cut) * 1e3   # kHz → Hz
    I1  = ss(traj_1,  1, cut) * 1e3
    W1e = ss(traj_1,  2, cut)
    W1i = ss(traj_1,  3, cut)

    E2  = ss(traj_2,  0, cut) * 1e3
    I2  = ss(traj_2,  1, cut) * 1e3
    Cee = ss(traj_2,  2, cut)
    Cei = ss(traj_2,  3, cut)
    Cii = ss(traj_2,  4, cut)
    W2e = ss(traj_2,  5, cut)
    W2i = ss(traj_2,  6, cut)

    E1w = ss(traj_1w, 0, cut) * 1e3
    I1w = ss(traj_1w, 1, cut) * 1e3

    print()
    print("=" * 72)
    print("STEADY-STATE RESULTS")
    print("-" * 72)
    print(f"{'Variable':<18} {'Order=1 (correct)':>18}  {'Order=2 (correct)':>18}  {'Order=1 (WRONG P)':>18}")
    print(f"{'--------':<18} {'------------------':>18}  {'------------------':>18}  {'------------------':>18}")
    print(f"{'E (Hz)':<18} {E1:>18.4f}  {E2:>18.4f}  {E1w:>18.4f}")
    print(f"{'I (Hz)':<18} {I1:>18.4f}  {I2:>18.4f}  {I1w:>18.4f}")
    print(f"{'W_e (pA)':<18} {W1e:>18.4f}  {W2e:>18.4f}  {'n/a':>18}")
    print(f"{'W_i (pA)':<18} {W1i:>18.4f}  {W2i:>18.4f}  {'n/a':>18}")
    print(f"{'C_ee (kHz²)':<18} {'n/a':>18}  {Cee:>18.6e}  {'n/a':>18}")
    print(f"{'C_ei (kHz²)':<18} {'n/a':>18}  {Cei:>18.6e}  {'n/a':>18}")
    print(f"{'C_ii (kHz²)':<18} {'n/a':>18}  {Cii:>18.6e}  {'n/a':>18}")

    print()
    print("=" * 72)
    print("COMPARISON METRICS")
    print("-" * 72)

    # 1st vs 2nd order (correct P)
    n = min(len(traj_1), len(traj_2))
    E_diff = traj_1[:n, 0] - traj_2[:n, 0]
    mae_12 = float(np.mean(np.abs(E_diff))) * 1e3
    rel_12 = mae_12 / (float(np.mean(np.abs(traj_1[:n, 0]))) * 1e3 + 1e-12) * 100
    print(f"\n  Order=1 vs Order=2 (correct P):    MAE = {mae_12:.4f} Hz  ({rel_12:.2f}% of mean E)")

    # 1st correct vs 1st wrong P
    E_diff_w = traj_1[:n, 0] - traj_1w[:n, 0]
    mae_1w = float(np.mean(np.abs(E_diff_w))) * 1e3
    rel_1w = mae_1w / (float(np.mean(np.abs(traj_1[:n, 0]))) * 1e3 + 1e-12) * 100
    print(f"  Correct P vs WRONG P (order=1):    MAE = {mae_1w:.4f} Hz  ({rel_1w:.2f}% of mean E)")

    # Final steady-state deviation
    dE_ss = abs(E1 - E2)
    print(f"\n  Steady-state |E_1 - E_2|           = {dE_ss:.4f} Hz")
    print(f"  Steady-state |E_correct - E_wrong| = {abs(E1-E1w):.4f} Hz")

    # ── Covariance magnitude (signal of finite-size correction) ───────────────
    print()
    print("=" * 72)
    print("SECOND-ORDER COVARIANCE DYNAMICS")
    print("-" * 72)
    print(f"  C_ee (excitatory variance)   = {Cee:.3e} kHz²  ≡ {Cee*1e6:.3f} Hz²")
    print(f"  C_ei (exc-inh covariance)    = {Cei:.3e} kHz²")
    print(f"  C_ii (inhibitory variance)   = {Cii:.3e} kHz²  ≡ {Cii*1e6:.3f} Hz²")
    print(f"  Expected C_ee ~ F_e/T/N_e    = {E1/1e3 / p['T'] / (p['N_tot']*(1-p['g'])):.3e} kHz²  (Poisson shot noise estimate)")
    print()
    print("  Interpretation: C_ee ≈ 1/N · F_e/T — the finite-size correction")
    print("  is tiny for N=10000. Order=1 and order=2 should agree closely.")

    print()
    print("=" * 72)
    print("TIMING")
    print("-" * 72)
    print(f"  Order=1 Euler : {t_elapsed_1:.3f}s  for {T_SIM} ms at dt={DT} ms")
    print(f"  Order=2 Euler : {t_elapsed_2:.3f}s  (×{t_elapsed_2/t_elapsed_1:.2f} overhead from numerical Jacobian)")
    print(f"  Note: TVB framework adds ODE→monitor overhead. The ×{t_elapsed_2/t_elapsed_1:.1f} ratio")
    print(f"  is dominated by the 10 extra TF evaluations per step (second derivatives).")

    print()
    print("=" * 72)
    print("VERDICT")
    print("-" * 72)
    if rel_12 < 5.0:
        print(f"  ✓ PASS  — Order=2 ≈ order=1 within {rel_12:.2f}% MAE (as expected for N={p['N_tot']})")
    else:
        print(f"  ✗ CHECK — Order=2 deviates {rel_12:.2f}% from order=1 (unexpected for this N)")
    if rel_1w > 10.0:
        print(f"  ✓ P BUG CONFIRMED — Scrambled coefficients produce {rel_1w:.1f}% error vs correct P")
    else:
        print(f"  ℹ P difference is only {rel_1w:.1f}% — effects may be parameter-regime dependent")
    print()


if __name__ == "__main__":
    main()
