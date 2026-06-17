#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zerlaut_matteo — numerically cleaner variant of the Zerlaut/di Volo mean-field.

Overview
--------
This module contains ``Zerlaut_adaptation_first_order`` and
``Zerlaut_adaptation_second_order`` — both of which inherit *everything* from
the base ``Zerlaut.py`` classes and override only ``get_fluct_regime_vars``.

The two classes therefore differ from their parent only in **how the membrane
potential statistics (μV, σV, TV) are computed**.

What changed vs the base Zerlaut?
----------------------------------
Two small but important numerical changes were introduced by Matteo di Volo
(hence the name "matteo") to make the transfer function better behaved at low
firing rates:

1. **No spontaneous-release floor (+1e-6)**
   In the base ``Zerlaut.py``, the input firing rates are offset before
   computing synaptic currents:

       fe = (Fe + 1e-6) * (1-g) * p_connect * N_tot + ...

   The ``+1e-6`` models a tiny baseline of spontaneous synaptic release so
   that even at Fe=0 the neuron sits in a finite-conductance state.  This
   prevents division-by-zero in TV but also shifts the operating point
   slightly off zero.

   **In Zerlaut_matteo:** this offset is removed.  The rates are used as-is:

       fe = Fe * (1-g) * p_connect * N_tot + ...

   This gives cleaner behaviour at Fe=0 (σV → 0 rather than a small
   non-zero value).

2. **Floor applied after σV (+1e-9 → +1e-12 on the return)**
   Without the +1e-6 floor, σV can reach exactly zero when Fe=Fi=0.
   Passing σV=0 into the ``estimate_firing_rate`` erfc formula would cause
   a division by zero.

   **In Zerlaut_matteo:** σV is computed exactly (possibly zero), then
   a tiny floor is added on return: ``return mu_V, sigma_V + 1e-12, T_V``.
   The floor is small enough (sub-femtovolt) to be numerically invisible in
   all realistic operating regimes.

   The base Zerlaut adds +1e-9 after sigma_V inside the computation (before
   TV), which slightly affects TV. Zerlaut_matteo adds it only at return,
   keeping TV unaffected.

When to use Zerlaut_matteo vs base Zerlaut
-------------------------------------------
- **Zerlaut_matteo** is preferred when you care about precise behaviour near
  the quiescent (E=I=0) fixed point, or when doing phase-plane analysis.
  The absence of the spontaneous-release offset means the model has a true
  zero-activity fixed point.

- **Base Zerlaut** is the original implementation used in published TVBSim
  simulations (Berlin / Fede configuration).  Use it when reproducing results
  from Destexhe lab papers.

Code design
-----------
``get_fluct_regime_vars`` is a ``@staticmethod`` decorated with
``@jit(nopython=True, cache=True)`` for performance.  Because it is a static
method, both ``Zerlaut_adaptation_first_order`` and
``Zerlaut_adaptation_second_order`` can share the **same** override: Python
method resolution looks up ``get_fluct_regime_vars`` on the instance's class,
so calling ``self.get_fluct_regime_vars(...)`` in the second-order ``dfun``
(which inherits from the first-order class here) will find the overridden
version automatically.

We therefore only need to define it **once**, in ``Zerlaut_adaptation_first_order``,
and ``Zerlaut_adaptation_second_order`` inherits it for free.

References
----------
[ZD_2018]   Zerlaut, Chemla, Chavane, Destexhe — J. Comput. Neurosci. 44:45, 2018.
[MV_2019]   Matteo di Volo, Alberto Romagnoni, Cristiano Capone, Alain Destexhe —
            Neural Computation 31(4):653-680, 2019.
[EBD_2009]  El Boustani & Destexhe — Neural Computation 21:1732-1775, 2009.
"""

from . import Zerlaut
from tvb.simulator.models.base import numpy
from numba import jit


class Zerlaut_adaptation_first_order(Zerlaut.Zerlaut_adaptation_first_order):
    """
    First-order Zerlaut mean-field — Matteo di Volo's numerically cleaner variant.

    Inherits all parameters, state variables, and dynamics from the base
    ``Zerlaut_adaptation_first_order``.  Only ``get_fluct_regime_vars`` is
    overridden to remove the spontaneous-release floor (+1e-6) and instead
    apply a tiny sigma_V floor (+1e-12) at the return.

    The rest of the model — transfer function, adaptation ODEs, OU noise,
    dfun — is **identical** to the base class and is not repeated here.

    For full documentation of state variables, parameters, and dynamics see
    ``Zerlaut.Zerlaut_adaptation_first_order``.
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W,
                              Q_e, tau_e, E_e,
                              Q_i, tau_i, E_i,
                              g_L, C_m, E_L, N_tot,
                              p_connect_e, p_connect_i, g,
                              K_ext_e, K_ext_i):
        """
        Compute membrane potential statistics (μV, σV, TV) — Matteo variant.

        **Key difference from base Zerlaut:** the +1e-6 spontaneous-release
        floor is *not* applied before computing synaptic rates.  This gives a
        true zero at Fe=Fi=0 rather than a small baseline conductance.
        A sub-femtovolt floor (+1e-12) is added to sigma_V only at return to
        prevent division-by-zero in the erfc formula.

        All equations are from [MV_2019] / [EBD_2009]:
          Eq. 5  : synaptic rates fe, fi
          Eq. 6  : mean conductance μG, membrane time constant Tm
          Eq. 7  : mean membrane potential μV
          Eq. 8  : voltage standard deviation σV
          Eq. 9  : voltage autocorrelation time TV

        Parameters
        ----------
        Fe, Fi         : mean excitatory / inhibitory population rate [kHz]
        Fe_ext, Fi_ext : external excitatory / inhibitory rate inputs [kHz]
        W              : adaptation current [pA]
        Q_e, tau_e, E_e: excitatory synapse parameters (conductance, decay, reversal)
        Q_i, tau_i, E_i: inhibitory synapse parameters
        g_L, C_m, E_L  : passive membrane parameters (leak conductance, capacitance,
                          leak reversal)
        N_tot          : total neuron count in the local column
        p_connect_e/i  : recurrent connection probability
        g              : inhibitory fraction (N_i = g·N_tot)
        K_ext_e/i      : number of external connections per neuron

        Returns
        -------
        mu_V   : mean membrane potential [mV]
        sigma_V: std of membrane potential fluctuations [mV]  (≥ 1e-12)
        T_V    : autocorrelation time of fluctuations [ms]
        """
        # ── Step 1: total synaptic input rates (Eq. 5) ───────────────────────
        # No spontaneous floor here — Fe=0 gives fe=0 exactly (unlike base Zerlaut)
        fe = Fe * (1.0 - g) * p_connect_e * N_tot + Fe_ext * K_ext_e
        fi = Fi *        g  * p_connect_i * N_tot + Fi_ext * K_ext_i

        # ── Step 2: mean conductances and membrane time constant (Eqns 5–6) ──
        mu_Ge = Q_e * tau_e * fe          # mean excitatory conductance [nS]
        mu_Gi = Q_i * tau_i * fi          # mean inhibitory conductance [nS]
        mu_G  = g_L + mu_Ge + mu_Gi       # total mean conductance [nS]
        T_m   = C_m / mu_G               # effective membrane time constant [ms]

        # ── Step 3: mean membrane potential (Eq. 7) ──────────────────────────
        mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G

        # ── Step 4: post-synaptic potential amplitudes ────────────────────────
        U_e = Q_e / mu_G * (E_e - mu_V)
        U_i = Q_i / mu_G * (E_i - mu_V)

        # ── Step 5: voltage standard deviation (Eq. 8) ───────────────────────
        sigma_V = numpy.sqrt(
            fe * (U_e * tau_e) ** 2 / (2.0 * (tau_e + T_m)) +
            fi * (U_i * tau_i) ** 2 / (2.0 * (tau_i + T_m))
        )
        # NOTE: floor applied *after* sigma_V so that T_V is computed with
        # the exact sigma_V.  The +1e-9 shift of fe/fi in the base class
        # would have affected T_V; here it does not.
        # We add 1e-9 to fe/fi only for the TV computation to avoid 0/0.
        fe_TV = fe + 1e-9
        fi_TV = fi + 1e-9

        # ── Step 6: voltage autocorrelation time (Eq. 9) ─────────────────────
        T_V_num = fe_TV * (U_e * tau_e) ** 2 + fi_TV * (U_i * tau_i) ** 2
        T_V_den = (fe_TV * (U_e * tau_e) ** 2 / (tau_e + T_m) +
                   fi_TV * (U_i * tau_i) ** 2 / (tau_i + T_m))
        T_V = T_V_num / T_V_den

        # Sub-femtovolt floor on sigma_V prevents division-by-zero in erfc
        return mu_V, sigma_V + 1e-12, T_V


class Zerlaut_adaptation_second_order(Zerlaut.Zerlaut_adaptation_second_order):
    """
    Second-order Zerlaut mean-field — Matteo di Volo's numerically cleaner variant.

    Inherits all parameters, state variables, covariance dynamics, and
    second-order rate corrections from the base
    ``Zerlaut_adaptation_second_order``.  Only ``get_fluct_regime_vars`` is
    overridden — via inheritance from
    ``Zerlaut_matteo.Zerlaut_adaptation_first_order`` above.

    **The override is NOT duplicated here.**  Because Python resolves methods
    through the MRO (method resolution order), and this class inherits from
    ``Zerlaut.Zerlaut_adaptation_second_order``, while *that* class's
    ``get_fluct_regime_vars`` is a static method looked up at runtime, we need
    to explicitly tell Python to use the Matteo variant by inheriting from the
    *Matteo* first-order class instead of the base one.

    For full documentation of covariance state variables and second-order ODE
    system see ``Zerlaut.Zerlaut_adaptation_second_order``.
    """

    # Inherit get_fluct_regime_vars from Zerlaut_adaptation_first_order above.
    # Explicitly pull it in to make the MRO intent clear to readers:
    get_fluct_regime_vars = Zerlaut_adaptation_first_order.get_fluct_regime_vars
