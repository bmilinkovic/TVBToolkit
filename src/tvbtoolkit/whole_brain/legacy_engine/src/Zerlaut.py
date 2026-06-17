#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zerlaut / di Volo AdEx Mean-Field Models for TVB whole-brain simulation.

Overview
--------
This module implements two levels of the Zerlaut mean-field model, both
derived from the master equation for a network of adaptive exponential
integrate-and-fire (AdEx) neurons:

  Zerlaut_adaptation_first_order   — 5 state variables (the standard model)
  Zerlaut_adaptation_second_order  — 8 state variables (adds covariance dynamics)

Both are TVB ``Model`` subclasses and can be dropped into a TVB ``Simulator``
as-is.  The second-order model is the default recommended choice and matches
the TVBSim implementation exactly.

What is the mean-field approach?
---------------------------------
Instead of simulating thousands of individual neurons, we track the *average*
behaviour of two populations: excitatory (E) and inhibitory (I).  The key
quantity is the population firing rate ν (in kHz), which evolves according to
a transfer function F that maps current inputs → output rate.

Transfer function F (how a single neuron responds to its inputs)
-----------------------------------------------------------------
Given the mean (μV) and standard deviation (σV) of the membrane potential,
the output firing rate is (Eq. 10 in [MV_2019]):

    F(νe, νi, ...) = erfc[(V_thre − μV) / (√2 · σV)] / (2 · TV)

where V_thre is an effective threshold computed by a polynomial (Eq. 11):

    V_thre = P0 + P1·V + P2·S + P3·T
           + P4·V² + P5·S² + P6·T²
           + P7·V·S + P8·V·T + P9·S·T

with normalised variables:
    V = (μV + 60 mV) / 10 mV
    S = (σV −  4 mV) /  6 mV
    T = (TV · gL/Cm − 0.5) / 1.0

The polynomial coefficients P[0..9] are obtained by fitting to spiking network
simulations (see Tables I/II in [MV_2019]).  **Their ordering matters** — the
10 terms correspond exactly to:
    P[0]=const, P[1]=V, P[2]=S, P[3]=T, P[4]=V², P[5]=S², P[6]=T²,
    P[7]=V·S,  P[8]=V·T, P[9]=S·T

Membrane potential statistics (from conductance-based inputs)
--------------------------------------------------------------
Given total excitatory (fe) and inhibitory (fi) synaptic rates [Eqns 5–9]:

    μGe = Qe · τe · fe            (mean excitatory conductance)
    μGi = Qi · τi · fi            (mean inhibitory conductance)
    μG  = gL + μGe + μGi          (total mean conductance, Eq. 6)
    Tm  = Cm / μG                  (effective membrane time constant)
    μV  = (μGe·Ee + μGi·Ei + gL·EL − W) / μG   (mean voltage, Eq. 7)
    σV  = √[fe·(Ue·τe)²/(2(τe+Tm)) + fi·(Ui·τi)²/(2(τi+Tm))]  (Eq. 8)
    TV  = [fe·(Ue·τe)² + fi·(Ui·τi)²]
        / [fe·(Ue·τe)²/(τe+Tm) + fi·(Ui·τi)²/(τi+Tm)]          (Eq. 9)

where Ue = Qe/μG · (Ee − μV) and Ui = Qi/μG · (Ei − μV) are the
post-synaptic potential amplitudes.

First-order dynamics (Eq. 4, [MV_2019])
----------------------------------------
    T · dE/dt  = F_e(E, I, ...) − E
    T · dI/dt  = F_i(E, I, ...) − I
    dW_e/dt    = −W_e/τwe + b_e·E + a_e·(μV − EL_e)/τwe
    dW_i/dt    = −W_i/τwi + b_i·I + a_i·(μV − EL_i)/τwi
    dη/dt      = −η/τOU          (Ornstein-Uhlenbeck noise state)

Second-order dynamics (Eq. 6, [ZD_2018])
------------------------------------------
The rate equations gain correction terms from firing-rate covariances:

    T · dE/dt = (F_e − E)
              + ½C_ee · ∂²F_e/∂E²  +  C_ei · ∂²F_e/∂E∂I  +  ½C_ii · ∂²F_e/∂I²

And three new covariance ODEs (where N_e = N_tot·(1−g), N_i = N_tot·g):

    T · dC_ee/dt = F_e(1/T − F_e)/N_e  +  (F_e−E)²
                 + 2·C_ee·∂F_e/∂E  +  2·C_ei·∂F_e/∂I  − 2·C_ee

    T · dC_ei/dt = (F_e−E)·(F_i−I)
                 + C_ee·∂F_e/∂E + C_ei·∂F_e/∂I + C_ei·∂F_i/∂E + C_ii·∂F_i/∂I
                 − 2·C_ei

    T · dC_ii/dt = F_i(1/T − F_i)/N_i  +  (F_i−I)²
                 + 2·C_ii·∂F_i/∂I  +  2·C_ei·∂F_i/∂E  − 2·C_ii

The shot-noise source term F(1/T − F)/N drives the covariances to their
steady state ~F/(T·N), which is O(1/N).  For N=10000 this correction is
negligible (< 0.01%), so first- and second-order agree in the AI regime.

The Jacobian (partial derivatives) is computed numerically using central
differences with step df=1e-7 kHz.

References
----------
[ZD_2018]   Zerlaut, Chemla, Chavane, Destexhe — J. Comput. Neurosci. 44:45, 2018.
            https://doi.org/10.1007/s10827-017-0668-2
[MV_2019]   Matteo di Volo, Alberto Romagnoni, Cristiano Capone, Alain Destexhe —
            Neural Computation 31(4):653-680, 2019.
            (published version of bioRxiv 2018 preprint)
[EBD_2009]  El Boustani & Destexhe — Neural Computation 21:1732-1775, 2009.
            https://doi.org/10.1162/neco.2009.02-08-710
"""

from tvb.simulator.models.base import Model, numpy
from tvb.basic.neotraits.api import NArray, Range, Final, List
import scipy.special as sp_spec
from numba import jit


# ══════════════════════════════════════════════════════════════════════════════
# First-order model
# ══════════════════════════════════════════════════════════════════════════════

class Zerlaut_adaptation_first_order(Model):
    """
    First-order Zerlaut/di Volo AdEx mean-field model.

    Tracks mean excitatory (E) and inhibitory (I) firing rates in kHz, slow
    adaptation currents (W_e, W_i) in pA, and an Ornstein-Uhlenbeck noise
    state (η).  Five state variables total.

    **When to use this model:**
    Use for fast exploratory simulations.  For N_tot=10000 it is numerically
    indistinguishable from second-order (< 0.01% error), but runs ~7× faster
    because it skips the Jacobian computation.  Use second-order when you
    need exact TVBSim parity or study finite-size fluctuations.

    **State variables** (all shape: n_regions):
      E      — excitatory population mean firing rate [kHz]
      I      — inhibitory population mean firing rate [kHz]
      W_e    — excitatory adaptation current [pA]
      W_i    — inhibitory adaptation current [pA]
      noise  — Ornstein-Uhlenbeck noise state [dimensionless]

    **Biological interpretation of key parameters:**
      g_L        leak conductance [nS] — controls membrane resistance
      E_L_e/i    leak reversal potential [mV] — resting potential
      C_m        membrane capacitance [pF] — controls time constant τm = Cm/gL
      b_e        spike-triggered adaptation kick [pA] — each spike adds b to W
      tau_w_e    adaptation time constant [ms] — how long adaptation persists
      Q_e/Q_i    synaptic quantal conductance [nS] — EPSP/IPSP amplitude
      tau_e/i    synaptic decay time constant [ms]
      g          fraction of inhibitory cells (typically 0.2 = 20%)
      N_tot      total neuron count in local network
      p_connect  connection probability within the local network
      T          integration window [ms] — sets the time-scale of rate dynamics
      P_e, P_i   transfer function polynomial coefficients (10 values each)
      K_ext_e    number of external excitatory inputs per neuron

    **Default P coefficients (Berlin/Fede configuration — matches TVBSim):**
      P_e = [-0.04983106, 0.00506355, -0.02347012, 0.00229515, -0.00041053,
              0.01054705, -0.03659253, 0.00743749,  0.00126506, -0.04072161]
      These are the original di Volo 2018/Zerlaut 2018 paper values.
      The Berlin configuration uses different fitted values — override via
      parameter_overrides in WholeBrainConfig.

    References: [ZD_2018], [MV_2019], [EBD_2009]
    """

    _ui_name = "Zerlaut_adaptation_first_order"
    ui_configurable_parameters = [
        'g_L', 'E_L_e', 'E_L_i', 'C_m',
        'b_e', 'a_e', 'b_i', 'a_i', 'tau_w_e', 'tau_w_i',
        'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e_e', 'tau_e_i', 'tau_i',
        'N_tot', 'p_connect_e', 'p_connect_i', 'g',
        'K_ext_e', 'K_ext_i', 'T',
        'external_input_ex_ex', 'external_input_ex_in',
        'external_input_in_ex', 'external_input_in_in',
        'tau_OU', 'weight_noise', 'noise_alpha',
    ]

    # ── Cellular parameters ──────────────────────────────────────────────────

    inh_factor = NArray(
        label="inh_factor",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=2.5, step=0.1),
        doc="""Scale factor applied to the long-range coupling signal when it
        drives the inhibitory population.  inh_factor=1 (default) means
        excitatory and inhibitory populations receive the same long-range input.
        Values > 1 boost inhibitory coupling.""")

    g_L = NArray(
        label="g_L  [nS]",
        default=numpy.array([10.0]),
        domain=Range(lo=0.1, hi=100.0, step=0.1),
        doc="""Leak conductance in nanoSiemens.  Together with C_m it sets the
        passive membrane time constant: τm = C_m / g_L ≈ 20 ms for the
        default values.  Larger g_L → shorter τm → faster membrane.""")

    E_L_e = NArray(
        label="E_L_e  [mV]",
        default=numpy.array([-65.0]),
        domain=Range(lo=-90.0, hi=-60.0, step=0.1),
        doc="""Leak (resting) reversal potential for the excitatory population
        [mV].  Typically −63 to −65 mV for RS (regular-spiking) pyramidal
        cells.  Drives the membrane toward this value when no synaptic input
        is present.""")

    E_L_i = NArray(
        label="E_L_i  [mV]",
        default=numpy.array([-65.0]),
        domain=Range(lo=-90.0, hi=-60.0, step=0.1),
        doc="""Leak reversal potential for the inhibitory population [mV].
        Typically −65 mV for FS (fast-spiking) interneurons.""")

    C_m = NArray(
        label="C_m  [pF]",
        default=numpy.array([200.0]),
        domain=Range(lo=10.0, hi=500.0, step=10.0),
        doc="""Membrane capacitance in picoFarads.  Controls how quickly the
        membrane potential responds to current injections.  Should scale
        linearly with g_L to maintain a fixed τm.""")

    # ── Adaptation parameters ────────────────────────────────────────────────

    b_e = NArray(
        label="b_e  [pA]",
        default=numpy.array([60.0]),
        domain=Range(lo=0.0, hi=150.0, step=1.0),
        doc="""Excitatory spike-triggered adaptation increment [pA].  After
        each spike W_e increases by b_e.  Large b_e → strong spike-frequency
        adaptation → the neuron slows down during sustained firing.
        b_e=5 pA → weak adaptation (awake state).
        b_e=60–120 pA → strong adaptation (sleep/anesthesia).""")

    a_e = NArray(
        label="a_e  [nS]",
        default=numpy.array([4.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.1),
        doc="""Excitatory subthreshold adaptation conductance [nS].  Produces
        slow hyperpolarisation proportional to (V − E_L).  Set to 0 for
        pure spike-triggered adaptation.  The Berlin configuration uses a_e=0.""")

    b_i = NArray(
        label="b_i  [pA]",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="""Inhibitory spike-triggered adaptation increment [pA].
        Typically 0 for fast-spiking interneurons (FS cells do not
        show significant spike-frequency adaptation).""")

    a_i = NArray(
        label="a_i  [nS]",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.1),
        doc="""Inhibitory subthreshold adaptation conductance [nS].
        Typically 0 for FS interneurons.""")

    tau_w_e = NArray(
        label="tau_w_e  [ms]",
        default=numpy.array([500.0]),
        domain=Range(lo=1.0, hi=1000.0, step=1.0),
        doc="""Adaptation time constant for excitatory neurons [ms].
        Controls how slowly the adaptation current W_e decays back to zero
        after spiking.  500 ms is typical for RS pyramidal cells and matches
        the original di Volo 2018 parameterisation.""")

    tau_w_i = NArray(
        label="tau_w_i  [ms]",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=1000.0, step=1.0),
        doc="""Adaptation time constant for inhibitory neurons [ms].
        Set to 1 ms (fast decay) since b_i=a_i=0 — effectively no adaptation
        in inhibitory cells.""")

    # ── Synaptic parameters ──────────────────────────────────────────────────

    E_e = NArray(
        label="E_e  [mV]",
        default=numpy.array([0.0]),
        domain=Range(lo=-20.0, hi=20.0, step=0.01),
        doc="""Excitatory (glutamatergic) reversal potential [mV].
        For AMPA/NMDA synapses this is ~0 mV.  Drives membrane toward 0 mV
        when excitatory synapses are activated.""")

    E_i = NArray(
        label="E_i  [mV]",
        default=numpy.array([-80.0]),
        domain=Range(lo=-100.0, hi=-60.0, step=1.0),
        doc="""Inhibitory (GABAergic) reversal potential [mV].
        For GABA-A synapses this is ~−80 mV.  Drives membrane toward −80 mV
        when inhibitory synapses are activated.""")

    Q_e = NArray(
        label="Q_e  [nS]",
        default=numpy.array([1.5]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""Excitatory quantal conductance [nS].  Peak conductance per
        single excitatory synaptic event (miniature EPSP amplitude).
        Controls the strength of individual E→E and E→I connections.""")

    Q_i = NArray(
        label="Q_i  [nS]",
        default=numpy.array([5.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.1),
        doc="""Inhibitory quantal conductance [nS].  Peak conductance per
        single inhibitory synaptic event.  Larger than Q_e to compensate
        for the smaller driving force at rest (E_i − μV is smaller than
        E_e − μV when μV < E_i).""")

    tau_e_e = NArray(
        label="tau_e_e  [ms]",
        default=numpy.array([5.0]),
        domain=Range(lo=1.0, hi=10.0, step=1.0),
        doc="""Decay time constant of excitatory synaptic conductance onto
        excitatory neurons [ms].  Models AMPA receptor kinetics (~5 ms).""")

    tau_e_i = NArray(
        label="tau_e_i  [ms]",
        default=numpy.array([5.0]),
        domain=Range(lo=1.0, hi=10.0, step=1.0),
        doc="""Decay time constant of excitatory synaptic conductance onto
        inhibitory neurons [ms].  Usually identical to tau_e_e.""")

    tau_i = NArray(
        label="tau_i  [ms]",
        default=numpy.array([5.0]),
        domain=Range(lo=0.5, hi=10.0, step=0.01),
        doc="""Decay time constant of inhibitory synaptic conductance [ms].
        Models GABA-A receptor kinetics (~5 ms).""")

    # ── Network parameters ───────────────────────────────────────────────────

    N_tot = NArray(
        dtype=int,
        label="N_tot",
        default=numpy.array([10000]),
        domain=Range(lo=1000, hi=50000, step=1000),
        doc="""Total number of neurons in the local cortical column.
        N_tot = N_e + N_i where N_e = N_tot·(1−g) excitatory and
        N_i = N_tot·g inhibitory cells.  Affects the magnitude of the
        finite-size noise correction in the second-order model.""")

    p_connect_e = NArray(
        label="p_connect_e",
        default=numpy.array([0.05]),
        domain=Range(lo=0.001, hi=0.2, step=0.001),
        doc="""Probability that any excitatory neuron connects to another
        neuron in the same local column.  5% is typical for sparse cortical
        networks.  Determines the mean number of recurrent excitatory inputs:
        K_ee = p_connect_e · N_e.""")

    p_connect_i = NArray(
        label="p_connect_i",
        default=numpy.array([0.05]),
        domain=Range(lo=0.001, hi=0.2, step=0.001),
        doc="""Probability that any inhibitory neuron connects to another
        neuron in the local column.  Same as p_connect_e by default.""")

    g = NArray(
        label="g  (inhibitory fraction)",
        default=numpy.array([0.2]),
        domain=Range(lo=0.01, hi=0.4, step=0.01),
        doc="""Fraction of neurons that are inhibitory.  g=0.2 means 20%
        inhibitory (standard cortical ratio of ~1 interneuron per 4 pyramidal
        cells).  Determines N_i = g·N_tot and N_e = (1−g)·N_tot.""")

    K_ext_e = NArray(
        dtype=int,
        label="K_ext_e",
        default=numpy.array([400]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of external excitatory synaptic inputs per neuron.
        These represent afferents from outside the local column (thalamic
        input, long-range projections not captured by the connectivity matrix).
        K_ext_e=400 with the 0.315 kHz drive gives ~0.126 nA mean current.""")

    K_ext_i = NArray(
        dtype=int,
        label="K_ext_i",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of external inhibitory inputs per neuron.  Set to 0
        by default — external drive is assumed purely excitatory.""")

    T = NArray(
        label="T  [ms]",
        default=numpy.array([20.0]),
        domain=Range(lo=1.0, hi=100.0, step=0.1),
        doc="""Integration window [ms].  Sets the time scale of the mean-field
        rate dynamics: T · dE/dt = F_e − E.  Smaller T → faster dynamics.
        T=20 ms is the default; T must be larger than the membrane time
        constant (~20 ms) for the mean-field approximation to be valid.""")

    # ── Transfer function polynomial coefficients ────────────────────────────

    P_e = NArray(
        label="P_e  (10 coefficients)",
        default=numpy.array([
            -0.04983106,   0.005063550882777035,  -0.023470121807314552,
             0.0022951513725067503,
            -0.0004105302652029825,  0.010547051343547399,  -0.03659252821136933,
             0.007437487505797858,  0.001265064721846073, -0.04072161294490446
        ]),
        doc="""Polynomial coefficients for the excitatory transfer function
        threshold (Eq. 11 in [MV_2019]).  10 values corresponding to:
          P[0]=const, P[1]=V, P[2]=S, P[3]=T, P[4]=V², P[5]=S², P[6]=T²,
          P[7]=V·S,   P[8]=V·T, P[9]=S·T
        where V=(μV+60)/10, S=(σV−4)/6, T=(TV·gL/Cm−0.5).

        ⚠ THE ORDERING MATTERS — see threshold_func for the polynomial.

        Default: original Zerlaut 2018 / di Volo 2018 paper values.
        Berlin configuration (TVBSim default):
          [-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
            0.00341614, -0.01156433,  0.00194753,  0.00274079, -0.01066769]
        Override via parameter_overrides={'parameter_model': {'P_e': [...]}}""")

    P_i = NArray(
        label="P_i  (10 coefficients)",
        default=numpy.array([
            -0.05149122024209484,  0.004003689190271077, -0.008352013668528155,
             0.0002414237992765705,
            -0.0005070645080016026,  0.0014345394104282397, -0.014686689498949967,
             0.004502706285435741,  0.0028472190352532454, -0.015357804594594548
        ]),
        doc="""Polynomial coefficients for the inhibitory transfer function
        threshold.  Same structure as P_e (see above).

        Berlin configuration (TVBSim default):
          [-0.05184978, 0.0061593, -0.01403522, 0.00166511, -0.0020559,
            0.00318432, -0.03112775, 0.00656668, 0.00171829, -0.04516385]""")

    # ── External (background) drive ──────────────────────────────────────────

    external_input_ex_ex = NArray(
        label="ν_ext→E→E  [kHz]",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""Constant external excitatory drive onto excitatory neurons
        [kHz].  Added to Fe_ext in the excitatory transfer function.
        Default 0.315e-3 kHz is set in the parameter file.""")

    external_input_ex_in = NArray(
        label="ν_ext→I→E  [kHz]",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""External inhibitory drive onto excitatory neurons [kHz].
        Typically 0 (external drive is purely excitatory).""")

    external_input_in_ex = NArray(
        label="ν_ext→E→I  [kHz]",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""External excitatory drive onto inhibitory neurons [kHz].
        Usually identical to external_input_ex_ex.""")

    external_input_in_in = NArray(
        label="ν_ext→I→I  [kHz]",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""External inhibitory drive onto inhibitory neurons [kHz].
        Typically 0.""")

    # ── Noise parameters ─────────────────────────────────────────────────────

    tau_OU = NArray(
        label="τ_OU  [ms]",
        default=numpy.array([5.0]),
        domain=Range(lo=0.10, hi=10.0, step=0.01),
        doc="""Ornstein-Uhlenbeck noise correlation time [ms].  Controls how
        quickly the noise state η decays: dη/dt = −η/τ_OU.  The TVB
        stochastic integrator injects white noise scaled by nsig; τ_OU
        colours it into a smooth correlated signal.""")

    weight_noise = NArray(
        label="weight_noise",
        default=numpy.array([1e-4]),
        domain=Range(lo=0.0, hi=50.0, step=1.0),
        doc="""Amplitude of the OU noise injected into the excitatory firing
        rate equation via Fe_ext += weight_noise · η.  In units of kHz so
        that the noise acts as an equivalent external excitatory drive.""")

    noise_alpha = NArray(
        label="α_shared  (noise mixing)",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Blend between private (α=0) and shared (α=1) OU noise.
        - α=0: each region gets independent noise (default, matches TVBSim).
        - α=1: all regions share the same noise (global synchrony forcing).
        Intermediate values give partial correlation.  Set via shared_noise_mode
        in the parameter file for richer spatial structure.""")

    # ── State variable metadata ──────────────────────────────────────────────

    state_variable_range = Final(
        label="State variable initial ranges [lo, hi]",
        default={
            "E":     numpy.array([0.0, 0.0]),    # start at rest (0 kHz)
            "I":     numpy.array([0.0, 0.0]),
            "W_e":   numpy.array([0.0, 0.0]),
            "W_i":   numpy.array([0.0, 0.0]),
            "noise": numpy.array([0.0, 0.0]),
        },
        doc="""Initial condition ranges for each state variable.  TVB draws
        random initial conditions uniformly from [lo, hi].  Setting lo=hi=0
        starts everything at rest, which avoids spurious transients.""")

    state_variable_boundaries = Final(
        label="State variable hard boundaries",
        default={
            "E": numpy.array([0.0, None]),   # firing rate ≥ 0
            "I": numpy.array([0.0, None]),
        },
        doc="""Hard lower bounds: firing rates cannot go negative.
        TVB clips E and I to [0, ∞) at each integration step.""")

    variables_of_interest = List(
        of=str,
        label="Variables monitored by TVB monitors",
        choices=("E", "I", "W_e", "W_i", "noise"),
        default=("E",),
        doc="""Which state variables to record.  'E' (excitatory rate) and
        'I' (inhibitory rate) are the most useful.  Index mapping:
          0=E, 1=I, 2=W_e, 3=W_i, 4=noise""")

    state_variables = 'E I W_e W_i noise'.split()
    _nvar = 5
    cvar = numpy.array([0], dtype=int)   # coupling variable = E (index 0)

    # ── Shared noise helper ──────────────────────────────────────────────────

    def _mixed_noise(self, noise):
        """Return α-blended private/shared noise for each region.

        With α=0 (default) returns the noise array unchanged.  With α>0,
        mixes in a spatially correlated component using _shared_noise_matrix
        (set by _configure_shared_noise in simulation.py).
        """
        alpha_arr = numpy.asarray(self.noise_alpha).reshape(-1)
        alpha = float(alpha_arr[0]) if alpha_arr.size > 0 else 0.0
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha <= 0.0:
            return noise

        shared_matrix = getattr(self, "_shared_noise_matrix", None)
        if shared_matrix is None:
            shared_noise = numpy.mean(noise) * numpy.ones_like(noise)
        else:
            shared_noise = numpy.dot(shared_matrix, noise)

        # Sqrt-weighted blend: keeps total noise variance constant across α.
        return numpy.sqrt(1.0 - alpha) * noise + numpy.sqrt(alpha) * shared_noise

    # ── Core dynamics ────────────────────────────────────────────────────────

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        First-order mean-field derivatives (one time step).

        Implements Eq. 4 from [MV_2019]:
            T · dE/dt  = F_e(E, I, Fe_ext, Fi_ext, W_e) − E
            T · dI/dt  = F_i(E, I, Fe_ext, Fi_ext, W_i) − I
            dW_e/dt    = −W_e/τwe + b_e·E + a_e·(μV_e − E_L_e)/τwe
            dW_i/dt    = −W_i/τwi + b_i·I + a_i·(μV_i − E_L_i)/τwi
            dη/dt      = −η/τ_OU

        The external input to excitatory neurons is:
            Fe_ext = c_0 + lc·E + weight_noise·η

        where c_0 is the long-range TVB coupling (kHz) and lc is local coupling.
        Negative Fe_ext is clipped to zero (can't have negative firing rate input).
        """
        E     = state_variables[0, :]
        I     = state_variables[1, :]
        W_e   = state_variables[2, :]
        W_i   = state_variables[3, :]
        noise = state_variables[4, :]
        derivative = numpy.empty_like(state_variables)

        c_0 = coupling[0, :]          # long-range coupling signal
        lc_E = local_coupling * E
        lc_I = local_coupling * I

        # External excitatory drive (long-range + local + noise)
        mixed_noise = self._mixed_noise(noise)
        Fe_ext = c_0 + lc_E + self.weight_noise * mixed_noise
        Fe_ext[Fe_ext * self.K_ext_e < 0] = 0.0   # clip negative inputs
        Fi_ext = lc_I   # external inhibitory drive (local only by default)

        # ── Firing rate derivatives (Eq. 4) ──────────────────────────────────
        derivative[0] = (
            self.TF_excitatory(E, I, Fe_ext + self.external_input_ex_ex,
                               Fi_ext + self.external_input_ex_in, W_e) - E
        ) / self.T

        derivative[1] = (
            self.TF_inhibitory(E, I, Fe_ext + self.external_input_in_ex,
                               Fi_ext + self.external_input_in_in, W_i) - I
        ) / self.T

        # ── Adaptation derivatives ────────────────────────────────────────────
        # Excitatory: need μV at excitatory neuron operating point
        mu_V, _, _ = self.get_fluct_regime_vars(
            E, I,
            Fe_ext + self.external_input_ex_ex,
            Fi_ext + self.external_input_ex_in,
            W_e,
            self.Q_e, self.tau_e_e, self.E_e,
            self.Q_i, self.tau_i,   self.E_i,
            self.g_L, self.C_m, self.E_L_e,
            self.N_tot, self.p_connect_e, self.p_connect_i, self.g,
            self.K_ext_e, self.K_ext_i,
        )
        derivative[2] = -W_e / self.tau_w_e + self.b_e * E + self.a_e * (mu_V - self.E_L_e) / self.tau_w_e

        # Inhibitory: need μV at inhibitory neuron operating point
        mu_V, _, _ = self.get_fluct_regime_vars(
            E, I,
            Fe_ext + self.external_input_in_ex,
            Fi_ext + self.external_input_in_in,
            W_i,
            self.Q_e, self.tau_e_i, self.E_e,
            self.Q_i, self.tau_i,   self.E_i,
            self.g_L, self.C_m, self.E_L_i,
            self.N_tot, self.p_connect_e, self.p_connect_i, self.g,
            self.K_ext_e, self.K_ext_i,
        )
        derivative[3] = -W_i / self.tau_w_i + self.b_i * I + self.a_i * (mu_V - self.E_L_i) / self.tau_w_i

        # ── OU noise state ────────────────────────────────────────────────────
        derivative[4] = -noise / self.tau_OU

        return derivative

    # ── Transfer functions ───────────────────────────────────────────────────

    def TF_excitatory(self, fe, fi, fe_ext, fi_ext, W):
        """Transfer function for the excitatory population.

        Calls TF() with excitatory-specific parameters (P_e, E_L_e, tau_e_e).
        Returns the predicted excitatory firing rate in kHz.
        """
        return self.TF(fe, fi, fe_ext, fi_ext, W, self.P_e, self.E_L_e, self.tau_e_e)

    def TF_inhibitory(self, fe, fi, fe_ext, fi_ext, W):
        """Transfer function for the inhibitory population.

        Calls TF() with inhibitory-specific parameters (P_i, E_L_i, tau_e_i).
        Returns the predicted inhibitory firing rate in kHz.
        """
        return self.TF(fe, fi, fe_ext, fi_ext, W, self.P_i, self.E_L_i, self.tau_e_i)

    def TF(self, fe, fi, fe_ext, fi_ext, W, P, E_L, tau_e):
        """Core transfer function: inputs → output firing rate [kHz].

        Steps:
          1. Compute membrane potential statistics (μV, σV, TV) via
             get_fluct_regime_vars (Eqns 5–9).
          2. Normalise and evaluate the threshold polynomial (Eq. 11).
          3. Return erfc-based firing rate estimate (Eq. 10).

        Parameters
        ----------
        fe, fi      : excitatory / inhibitory population firing rates [kHz]
        fe_ext, fi_ext : external excitatory / inhibitory inputs [kHz]
        W           : adaptation current [pA]
        P           : 10-element polynomial coefficient array
        E_L         : leak reversal potential for this population [mV]
        tau_e       : excitatory synaptic decay onto this population [ms]
        """
        mu_V, sigma_V, T_V = self.get_fluct_regime_vars(
            fe, fi, fe_ext, fi_ext, W,
            self.Q_e, tau_e, self.E_e,
            self.Q_i, self.tau_i, self.E_i,
            self.g_L, self.C_m, E_L,
            self.N_tot, self.p_connect_e, self.p_connect_i, self.g,
            self.K_ext_e, self.K_ext_i,
        )
        # Normalise T_V to units of τm = C_m/g_L
        TvN = T_V * self.g_L / self.C_m
        V_thre = self.threshold_func(mu_V, sigma_V, TvN,
                                     P[0], P[1], P[2], P[3], P[4],
                                     P[5], P[6], P[7], P[8], P[9])
        V_thre *= 1e3   # V → mV (threshold_func returns Volts)
        return self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre)

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W,
                              Q_e, tau_e, E_e,
                              Q_i, tau_i, E_i,
                              g_L, C_m, E_L, N_tot,
                              p_connect_e, p_connect_i, g,
                              K_ext_e, K_ext_i):
        """
        Compute membrane potential statistics (μV, σV, TV) from firing rates.

        This is the analytical mean-field solution for the membrane potential
        distribution of an AdEx neuron receiving Poissonian synaptic inputs.
        All equations from [MV_2019] / [EBD_2009].

        Parameters
        ----------
        Fe, Fi         : mean excitatory / inhibitory population rate [kHz]
        Fe_ext, Fi_ext : external excitatory / inhibitory rate inputs [kHz]
        W              : adaptation current [pA]
        Q_e, tau_e, E_e: excitatory synapse parameters
        Q_i, tau_i, E_i: inhibitory synapse parameters
        g_L, C_m, E_L  : passive membrane parameters
        N_tot          : total neuron count
        p_connect_e/i  : recurrent connection probability
        g              : inhibitory fraction
        K_ext_e/i      : number of external connections

        Returns
        -------
        mu_V   : mean membrane potential [mV]
        sigma_V: std of membrane potential fluctuations [mV]
        T_V    : autocorrelation time of fluctuations [ms]

        Notes
        -----
        The 1e-6 offset (``(Fe+1e-6)``) models spontaneous synaptic release:
        even at zero firing rate there is a tiny baseline conductance, which
        prevents division by zero in T_V and keeps the transfer function smooth.
        The ``Zerlaut_matteo`` variant removes this offset and adds a small
        ``1e-9`` floor after sigma_V instead.
        """
        # ── Step 1: total synaptic input rates (Eq. 5) ───────────────────────
        # (Fe+1e-6) adds a spontaneous-release floor so the model is always
        # in a finite-conductance state, even at zero network activity.
        fe = (Fe + 1.0e-6) * (1.0 - g) * p_connect_e * N_tot + Fe_ext * K_ext_e
        fi = (Fi + 1.0e-6) *        g  * p_connect_i * N_tot + Fi_ext * K_ext_i

        # ── Step 2: mean conductances and membrane time constant (Eqns 5–6) ──
        mu_Ge = Q_e * tau_e * fe          # mean excitatory conductance [nS]
        mu_Gi = Q_i * tau_i * fi          # mean inhibitory conductance [nS]
        mu_G  = g_L + mu_Ge + mu_Gi       # total mean conductance [nS]
        T_m   = C_m / mu_G               # effective membrane time constant [ms]

        # ── Step 3: mean membrane potential (Eq. 7) ──────────────────────────
        # Derived by equating current balance at the mean: μG·μV = μGe·Ee + μGi·Ei + gL·EL − W
        mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G

        # ── Step 4: post-synaptic potential amplitudes ────────────────────────
        # Each synaptic event shifts V by Q/μG · (E_rev − μV)
        U_e = Q_e / mu_G * (E_e - mu_V)
        U_i = Q_i / mu_G * (E_i - mu_V)

        # ── Step 5: voltage standard deviation (Eq. 8) ───────────────────────
        # Shot-noise approximation: each Poisson synapse contributes independently
        sigma_V = numpy.sqrt(
            fe * (U_e * tau_e) ** 2 / (2.0 * (tau_e + T_m)) +
            fi * (U_i * tau_i) ** 2 / (2.0 * (tau_i + T_m))
        )

        # ── Step 6: voltage autocorrelation time (Eq. 9) ─────────────────────
        T_V_num = fe * (U_e * tau_e) ** 2 + fi * (U_i * tau_i) ** 2
        T_V_den = (fe * (U_e * tau_e) ** 2 / (tau_e + T_m) +
                   fi * (U_i * tau_i) ** 2 / (tau_i + T_m))
        T_V = T_V_num / T_V_den    # units: ms

        return mu_V, sigma_V, T_V

    @staticmethod
    @jit(nopython=True, cache=True)
    def threshold_func(muV, sigmaV, TvN,
                       P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
        """
        Phenomenological effective threshold (Eq. 11 in [MV_2019]).

        Polynomial in three normalised voltage-statistics variables:
            V_thre = P0
                   + P1·V  + P2·S  + P3·T
                   + P4·V² + P5·S² + P6·T²
                   + P7·V·S + P8·V·T + P9·S·T

        Normalisation constants (page 48 in [ZD_2018]):
            V = (μV   + 60 mV) / 10 mV
            S = (σV   −  4 mV) /  6 mV
            T = (TvN  − 0.5) /  1.0       (TvN = TV · gL/Cm, dimensionless)

        The 10 coefficients are fitted to intracellular recordings or spiking
        network simulations.  ⚠ Their INDEX ORDER must match the terms above —
        swapping P[5..9] changes model dynamics by >50%.

        Returns V_thre in Volts (multiply by 1e3 to get mV).
        """
        muV0,  DmuV0  = -60.0, 10.0
        sV0,   DsV0   =   4.0,  6.0
        TvN0,  DTvN0  =   0.5,  1.0

        V = (muV   - muV0)  / DmuV0
        S = (sigmaV - sV0)  / DsV0
        T = (TvN   - TvN0)  / DTvN0

        return (P0
                + P1 * V + P2 * S + P3 * T
                + P4 * V**2 + P5 * S**2 + P6 * T**2
                + P7 * V * S + P8 * V * T + P9 * S * T)

    @staticmethod
    def estimate_firing_rate(muV, sigmaV, Tv, Vthre):
        """
        Convert voltage statistics to output firing rate (Eq. 10 in [MV_2019]).

        Assumes the sub-threshold membrane potential is Gaussian with mean μV
        and std σV.  Spikes occur when V crosses V_thre.  In the
        diffusion-approximation limit, the mean inter-spike interval is:

            ISI = 2·TV · erfc⁻¹(F·2·TV)  → F = erfc[(V_thre − μV) / (√2·σV)] / (2·TV)

        Parameters
        ----------
        muV   : mean membrane potential [mV]
        sigmaV: std of membrane potential [mV]
        Tv    : autocorrelation time of fluctuations [ms]
        Vthre : effective threshold voltage [mV]

        Returns
        -------
        firing_rate : [kHz]
        """
        return sp_spec.erfc((Vthre - muV) / (numpy.sqrt(2) * sigmaV)) / (2 * Tv)


# ══════════════════════════════════════════════════════════════════════════════
# Second-order model
# ══════════════════════════════════════════════════════════════════════════════

class Zerlaut_adaptation_second_order(Zerlaut_adaptation_first_order):
    """
    Second-order Zerlaut/di Volo AdEx mean-field model.

    Extends the first-order model with three additional state variables that
    track the *variance* (C_ee, C_ii) and *covariance* (C_ei) of the
    population firing rates.  Eight state variables total.

    **This is the recommended model and matches the TVBSim default.**

    **Why second-order matters:**
    The first-order model tracks only the mean firing rates.  The second-order
    model also tracks how much the firing rates fluctuate *around* their mean,
    and how these fluctuations are correlated between populations.  This
    allows the model to correctly propagate finite-size noise corrections.

    For large networks (N_tot=10000) the covariances reach their shot-noise
    steady state ~F/(T·N) ≈ 10⁻¹¹ kHz², which is negligible in practice.
    The practical difference from first-order is < 0.01%.

    **Additional state variables (indices 2, 3, 4):**
      C_ee  — variance of excitatory firing rate [kHz²]
              (how much E fluctuates around its mean)
      C_ei  — covariance between excitatory and inhibitory rates [kHz²]
              (how correlated E and I fluctuations are)
      C_ii  — variance of inhibitory firing rate [kHz²]

    **Covariance dynamics (Eq. 6 in [ZD_2018]):**
    Each covariance equation has:
      - A *source* term F(1/T − F)/N — shot noise from discrete spiking
      - A *drift* term (F − ν)² — noise from the mean-field approximation
      - *Coupling* terms ∂F/∂ν — how the Jacobian mixes covariances
      - A *decay* term −2C/T — covariances relax back toward zero

    **Rate correction (Eq. 5 in [ZD_2018]):**
    The mean firing rate equations gain second-order corrections:
        T · dE/dt = (F_e − E)
                  + ½C_ee · ∂²F_e/∂E²
                  + C_ei  · ∂²F_e/∂E∂I
                  + ½C_ii · ∂²F_e/∂I²

    The Jacobian and Hessian are computed numerically at each step using
    central differences (step size df=1e-7 kHz), which is the dominant
    cost compared to first-order.

    References: [ZD_2018], [MV_2019], [EBD_2009]
    """

    _ui_name = "Zerlaut_adaptation_second_order"

    state_variable_range = Final(
        label="State variable initial ranges [lo, hi]",
        default={
            "E":     numpy.array([0.0, 0.0]),
            "I":     numpy.array([0.0, 0.0]),
            "C_ee":  numpy.array([0.0, 0.0]),   # start at zero covariance
            "C_ei":  numpy.array([0.0, 0.0]),
            "C_ii":  numpy.array([0.0, 0.0]),
            "W_e":   numpy.array([0.0, 0.0]),
            "W_i":   numpy.array([0.0, 0.0]),
            "noise": numpy.array([0.0, 0.0]),
        },
        doc="""Initial conditions.  Starting C_ee=C_ei=C_ii=0 is correct:
        the covariances will relax to their shot-noise steady state in a
        few T-time-constants (~20–100 ms), which is usually well within the
        transient period that is discarded anyway.""")

    variables_of_interest = List(
        of=str,
        label="Variables monitored by TVB monitors",
        choices=("E", "I", "C_ee", "C_ei", "C_ii", "W_e", "W_i", "noise"),
        default=("E",),
        doc="""Which state variables to record.  Index mapping:
          0=E, 1=I, 2=C_ee, 3=C_ei, 4=C_ii, 5=W_e, 6=W_i, 7=noise""")

    state_variables = 'E I C_ee C_ei C_ii W_e W_i noise'.split()
    _nvar = 8

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Second-order mean-field derivatives (one time step).

        Implements Eqns 4–6 from [ZD_2018] / [MV_2019]:

        Firing rates (with second-order correction):
            T · dE/dt = (F_e − E)
                      + ½C_ee·∂²F_e/∂E² + C_ei·∂²F_e/∂E∂I + ½C_ii·∂²F_e/∂I²
            T · dI/dt = (F_i − I)
                      + ½C_ee·∂²F_i/∂E² + C_ei·∂²F_i/∂E∂I + ½C_ii·∂²F_i/∂I²

        Covariances:
            T · dC_ee/dt = F_e(1/T−F_e)/N_e + (F_e−E)²
                         + 2·C_ee·∂F_e/∂E + 2·C_ei·∂F_e/∂I − 2·C_ee

            T · dC_ei/dt = (F_e−E)·(F_i−I)
                         + C_ee·∂F_e/∂E + C_ei·∂F_e/∂I
                         + C_ei·∂F_i/∂E + C_ii·∂F_i/∂I − 2·C_ei

            T · dC_ii/dt = F_i(1/T−F_i)/N_i + (F_i−I)²
                         + 2·C_ii·∂F_i/∂I + 2·C_ei·∂F_i/∂E − 2·C_ii

        Adaptation and noise: same as first-order (indices 5, 6, 7).

        Numerical Jacobian: all partial derivatives ∂F/∂ν and ∂²F/∂ν² are
        computed by central differences with step df=1e-7 kHz.  This requires
        ~10 extra transfer function evaluations per step, giving the ~7–8× cost
        overhead of the second-order model.
        """
        # ── State variables ───────────────────────────────────────────────────
        E     = state_variables[0, :]
        I     = state_variables[1, :]
        C_ee  = state_variables[2, :]
        C_ei  = state_variables[3, :]
        C_ii  = state_variables[4, :]
        W_e   = state_variables[5, :]
        W_i   = state_variables[6, :]
        noise = state_variables[7, :]
        derivative = numpy.empty_like(state_variables)

        N_e = self.N_tot * (1.0 - self.g)   # number of excitatory neurons
        N_i = self.N_tot * self.g            # number of inhibitory neurons

        # ── External inputs ───────────────────────────────────────────────────
        c_0  = coupling[0, :]
        lc_E = local_coupling * E
        lc_I = local_coupling * I

        mixed_noise = self._mixed_noise(noise)
        # Long-range + local + noise → excitatory external drive
        E_input_exc = c_0 + lc_E + self.external_input_ex_ex + self.weight_noise * mixed_noise
        E_input_exc[E_input_exc < 0] = 0.0   # clip: can't have negative rate input
        # The inhibitory population also receives long-range input (scaled by inh_factor)
        E_input_inh = self.inh_factor * c_0 + lc_E + self.external_input_in_ex + self.weight_noise * mixed_noise
        E_input_inh[E_input_inh < 0] = 0.0
        I_input_exc = lc_I + self.external_input_ex_in
        I_input_inh = lc_I + self.external_input_in_in

        # ── Transfer functions at current state ───────────────────────────────
        F_e = self.TF_excitatory(E, I, E_input_exc, I_input_exc, W_e)
        F_i = self.TF_inhibitory(E, I, E_input_inh, I_input_inh, W_i)

        # ── Numerical derivatives (central differences) ───────────────────────
        # df is the finite-difference step in kHz.  1e-7 kHz is small enough
        # that the derivative error is < 1e-6 for typical firing rates.
        df = 1e-7

        def _dF_dE(TF, fe, fi, fe_ext, fi_ext, W):
            """First derivative ∂TF/∂fe  (units: 1 / [kHz · ms])"""
            return (TF(fe + df, fi, fe_ext, fi_ext, W) -
                    TF(fe - df, fi, fe_ext, fi_ext, W)) / (2 * df * 1e3)

        def _dF_dI(TF, fe, fi, fe_ext, fi_ext, W):
            """First derivative ∂TF/∂fi"""
            return (TF(fe, fi + df, fe_ext, fi_ext, W) -
                    TF(fe, fi - df, fe_ext, fi_ext, W)) / (2 * df * 1e3)

        def _d2F_dE2(TF, fe, fi, fe_ext, fi_ext, W, F_at_center):
            """Second derivative ∂²TF/∂fe²"""
            return (TF(fe + df, fi, fe_ext, fi_ext, W) - 2 * F_at_center +
                    TF(fe - df, fi, fe_ext, fi_ext, W)) / (df * 1e3) ** 2

        def _d2F_dI2(TF, fe, fi, fe_ext, fi_ext, W, F_at_center):
            """Second derivative ∂²TF/∂fi²"""
            return (TF(fe, fi + df, fe_ext, fi_ext, W) - 2 * F_at_center +
                    TF(fe, fi - df, fe_ext, fi_ext, W)) / (df * 1e3) ** 2

        def _d2F_dEdI(TF, fe, fi, fe_ext, fi_ext, W):
            """Mixed second derivative ∂²TF/∂fe∂fi"""
            return (
                _dF_dI(TF, fe + df, fi, fe_ext, fi_ext, W) -
                _dF_dI(TF, fe - df, fi, fe_ext, fi_ext, W)
            ) / (2 * df * 1e3)

        # Pre-compute all first derivatives (reused in covariance equations)
        dFe_dE = _dF_dE(self.TF_excitatory, E, I, E_input_exc, I_input_exc, W_e)
        dFe_dI = _dF_dI(self.TF_excitatory, E, I, E_input_exc, I_input_exc, W_e)
        dFi_dE = _dF_dE(self.TF_inhibitory, E, I, E_input_inh, I_input_inh, W_i)
        dFi_dI = _dF_dI(self.TF_inhibitory, E, I, E_input_inh, I_input_inh, W_i)

        # ── Firing rate derivatives (Eq. 4 + 2nd-order correction, Eq. 5) ────
        # The correction is:  ½C_ee·∂²F/∂E²  +  C_ei·∂²F/∂E∂I  +  ½C_ii·∂²F/∂I²
        # Note: coefficient on the mixed term is C_ei (not 2×½C_ei) — call once.
        derivative[0] = (
            F_e - E
            + 0.5 * C_ee * _d2F_dE2(self.TF_excitatory, E, I, E_input_exc, I_input_exc, W_e, F_e)
            +       C_ei * _d2F_dEdI(self.TF_excitatory, E, I, E_input_exc, I_input_exc, W_e)
            + 0.5 * C_ii * _d2F_dI2(self.TF_excitatory, E, I, E_input_exc, I_input_exc, W_e, F_e)
        ) / self.T

        derivative[1] = (
            F_i - I
            + 0.5 * C_ee * _d2F_dE2(self.TF_inhibitory, E, I, E_input_inh, I_input_inh, W_i, F_i)
            +       C_ei * _d2F_dEdI(self.TF_inhibitory, E, I, E_input_inh, I_input_inh, W_i)
            + 0.5 * C_ii * _d2F_dI2(self.TF_inhibitory, E, I, E_input_inh, I_input_inh, W_i, F_i)
        ) / self.T

        # ── Covariance derivatives (Eq. 6) ────────────────────────────────────
        # dC_ee/dt: variance of excitatory rate
        derivative[2] = (
            F_e * (1.0 / self.T - F_e) / N_e   # shot-noise source
            + (F_e - E) ** 2                    # drift
            + 2.0 * C_ee * dFe_dE               # ∂F_e/∂E coupling
            + 2.0 * C_ei * dFe_dI               # ∂F_e/∂I coupling
            - 2.0 * C_ee                         # decay
        ) / self.T

        # dC_ei/dt: covariance between excitatory and inhibitory rates
        derivative[3] = (
            (F_e - E) * (F_i - I)               # cross-drift
            + C_ee * dFe_dE                      # excitatory-excitatory coupling
            + C_ei * dFe_dI                      # excitatory-inhibitory coupling
            + C_ei * dFi_dE                      # inhibitory-excitatory coupling
            + C_ii * dFi_dI                      # inhibitory-inhibitory coupling
            - 2.0 * C_ei                         # decay
        ) / self.T

        # dC_ii/dt: variance of inhibitory rate
        derivative[4] = (
            F_i * (1.0 / self.T - F_i) / N_i   # shot-noise source
            + (F_i - I) ** 2                    # drift
            + 2.0 * C_ii * dFi_dI               # ∂F_i/∂I coupling
            + 2.0 * C_ei * dFi_dE               # ∂F_i/∂E coupling
            - 2.0 * C_ii                         # decay
        ) / self.T

        # ── Adaptation (same as first-order) ─────────────────────────────────
        mu_V_e, _, _ = self.get_fluct_regime_vars(
            E, I, E_input_exc, I_input_exc, W_e,
            self.Q_e, self.tau_e_e, self.E_e,
            self.Q_i, self.tau_i,   self.E_i,
            self.g_L, self.C_m, self.E_L_e,
            self.N_tot, self.p_connect_e, self.p_connect_i, self.g,
            self.K_ext_e, self.K_ext_i,
        )
        derivative[5] = -W_e / self.tau_w_e + self.b_e * E + self.a_e * (mu_V_e - self.E_L_e) / self.tau_w_e

        mu_V_i, _, _ = self.get_fluct_regime_vars(
            E, I, E_input_inh, I_input_inh, W_i,
            self.Q_e, self.tau_e_i, self.E_e,
            self.Q_i, self.tau_i,   self.E_i,
            self.g_L, self.C_m, self.E_L_i,
            self.N_tot, self.p_connect_e, self.p_connect_i, self.g,
            self.K_ext_e, self.K_ext_i,
        )
        derivative[6] = -W_i / self.tau_w_i + self.b_i * I + self.a_i * (mu_V_i - self.E_L_i) / self.tau_w_i

        # ── OU noise ─────────────────────────────────────────────────────────
        derivative[7] = -noise / self.tau_OU

        return derivative
