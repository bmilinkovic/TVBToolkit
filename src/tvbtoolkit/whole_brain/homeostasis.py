"""Slow inhibitory homeostasis for calibrating AdEx operating points.

The update is a dimensionless, positivity-preserving translation of the
rate-based inhibitory plasticity rule used by Coronel-Oliveros et al. (2026):
presynaptic inhibitory activity gates a postsynaptic excitatory rate error.
It learns the regional inhibitory-to-excitatory quantal conductance ``Q_i_e``
between simulation epochs. This module supplies the pre-fitted/frozen control;
the online within-trial implementation is in ``Zerlaut_homeostatic``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def baseline_relative_activation_threshold(
    baseline_rate_khz: np.ndarray,
    *,
    n_sd: float = 5.0,
) -> np.ndarray:
    r"""Return each region's baseline mean plus ``n_sd`` sample SDs."""
    baseline = np.asarray(baseline_rate_khz, dtype=float)
    if baseline.ndim != 2 or baseline.shape[0] < 2:
        raise ValueError("baseline_rate_khz must have shape (time, regions).")
    if not np.isfinite(baseline).all():
        raise ValueError("Baseline rates must be finite.")
    if n_sd <= 0.0:
        raise ValueError("n_sd must be positive.")
    return np.mean(baseline, axis=0) + float(n_sd) * np.std(
        baseline, axis=0, ddof=1
    )


@dataclass(frozen=True)
class InhibitoryHomeostasisConfig:
    """Parameters for one epoch-wise inhibitory homeostasis update."""

    tau_s: float = 2.0
    beta: float = 1.0
    min_scale: float = 0.25
    max_scale: float = 4.0
    max_log_step: float = 0.15

    def validate(self) -> None:
        if self.tau_s <= 0.0:
            raise ValueError("tau_s must be positive.")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if not 0.0 < self.min_scale < self.max_scale:
            raise ValueError("Require 0 < min_scale < max_scale.")
        if self.max_log_step <= 0.0:
            raise ValueError("max_log_step must be positive.")


def update_inhibitory_conductance(
    q_i_ns: np.ndarray,
    excitatory_rate_khz: np.ndarray,
    inhibitory_rate_khz: np.ndarray,
    target_rate_khz: np.ndarray,
    *,
    base_q_i_ns: float,
    epoch_ms: float,
    config: InhibitoryHomeostasisConfig | None = None,
) -> np.ndarray:
    r"""Return the inhibitory conductance after one learning epoch.

    For regional inhibitory scale :math:`h=Q_{i\to e}/Q_{i\to e,0}`, the Euler
    update is

    .. math::

       \Delta\log h = \frac{\Delta t}{\tau}
       \frac{r_I}{\rho}\frac{r_E-\rho}{\rho}h^{\beta-1}.

    This retains the cited rule's three essential properties: inhibitory
    activity gates learning, excitatory error determines its direction, and a
    multiplicative soft bound keeps inhibition non-negative.  Rates are
    normalized by the target so the equation is dimensionally valid for the
    AdEx model's kHz rate convention.  A capped log step makes coarse epoch
    updates numerically conservative.
    """
    if config is None:
        config = InhibitoryHomeostasisConfig()
    config.validate()
    if base_q_i_ns <= 0.0 or epoch_ms <= 0.0:
        raise ValueError("base_q_i_ns and epoch_ms must be positive.")

    q_i = np.asarray(q_i_ns, dtype=float)
    rate_e = np.asarray(excitatory_rate_khz, dtype=float)
    rate_i = np.asarray(inhibitory_rate_khz, dtype=float)
    target = np.asarray(target_rate_khz, dtype=float)
    q_i, rate_e, rate_i, target = np.broadcast_arrays(q_i, rate_e, rate_i, target)
    if not all(np.isfinite(x).all() for x in (q_i, rate_e, rate_i, target)):
        raise ValueError("Homeostasis inputs must be finite.")
    if np.any(q_i <= 0.0) or np.any(target <= 0.0):
        raise ValueError("Q_i_e and target rates must be strictly positive.")

    scale = q_i / float(base_q_i_ns)
    error = (rate_e - target) / target
    gate = np.maximum(rate_i, 0.0) / target
    soft_bound = np.power(np.maximum(scale, 1e-12), config.beta - 1.0)
    delta_log = (float(epoch_ms) / 1000.0 / config.tau_s) * gate * error * soft_bound
    delta_log = np.clip(delta_log, -config.max_log_step, config.max_log_step)
    new_scale = np.clip(scale * np.exp(delta_log), config.min_scale, config.max_scale)
    return float(base_q_i_ns) * new_scale
