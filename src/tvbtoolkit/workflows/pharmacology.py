"""Pharmacology helpers for ketamine/psilocybin parameterization."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Hill / Emax PK-PD model
# ---------------------------------------------------------------------------

def hill_occupancy(
    C: float | np.ndarray,
    EC50: float = 1.95,
    n: float = 1.0,
    occ_max: float = 0.766,
) -> float | np.ndarray:
    """Receptor occupancy from plasma drug concentration via the Hill / Emax equation.

    Models the fraction of receptors occupied at plasma concentration ``C``::

        occ(C) = occ_max × C^n / (EC50^n + C^n)

    The ``occ_max`` term reflects that even at saturating drug concentrations,
    not all receptors become occupied — endogenous serotonin competes at baseline
    and the agonist radioligand preferentially labels high-affinity states.

    Default parameters are empirically fitted values from Madsen et al. (2019),
    *Neuropsychopharmacology* (n=8 humans, [¹¹C]Cimbi-36 PET, R²=0.92):
      - EC50    = 1.95 ng/mL  [95% CI: 1.17, 3.15]
      - occ_max = 0.766       [95% CI: 0.673, 0.880]
      - n       = 1           (single-site binding, no cooperativity)

    Parameters
    ----------
    C : float or array-like
        Plasma drug concentration (ng/mL).  Pass 0.0 for placebo.
    EC50 : float
        Concentration at half-maximum occupancy (ng/mL).
    n : float
        Hill cooperativity exponent.  n=1 is empirically justified for
        psilocin at 5-HT2A.
    occ_max : float
        Maximum attainable occupancy ∈ (0, 1].  Default 0.766 from Madsen 2019.

    Returns
    -------
    float or ndarray
        Occupancy in [0, occ_max].  Returns 0.0 for C ≤ 0.
    """
    C = np.asarray(C, dtype=float)
    occ = np.where(C > 0, occ_max * C**n / (EC50**n + C**n), 0.0)
    return float(occ) if occ.ndim == 0 else occ


def dose_sensitive_gK_profile(
    emax_ng_ml: float,
    receptor_map: np.ndarray,
    g_K_ctrl: float,
    g_K_drug_max: float,
    EC50: float = 1.95,
    hill_n: float = 1.0,
    occ_max: float = 0.766,
) -> np.ndarray:
    """Build a per-region g_K vector scaled by plasma drug concentration.

    Combines a spatial receptor-density map with a scalar Hill occupancy to
    produce the region-wise potassium leak conductance for a single subject::

        g_K(r) = g_K_ctrl − occ(C) · rec_norm(r) · (g_K_ctrl − g_K_drug_max)

    where ``rec_norm(r) ∈ [0, 1]`` is the normalised receptor density at
    region *r* and ``occ(C) ∈ [0, 1]`` is the Hill occupancy for plasma
    concentration *C*.

    - Low dose (small C) → occ ≈ 0 → g_K ≈ g_K_ctrl everywhere (near-baseline)
    - High dose + high receptor region → occ ≈ 1 & rec_norm ≈ 1 → g_K ≈ g_K_drug_max
    - High dose + low receptor region → large occ but small rec_norm → modest ΔgK

    Parameters
    ----------
    emax_ng_ml : float
        Peak plasma psilocin concentration (ng/mL) for this subject.
        Pass 0.0 for placebo.
    receptor_map : np.ndarray, shape (n_regions,)
        Per-region 5-HT2A density (any consistent unit; will be normalised).
    g_K_ctrl : float
        Baseline g_K [nS] (control / placebo condition).
    g_K_drug_max : float
        Minimum g_K [nS] at maximum drug effect in the highest-receptor region.
        Must be < g_K_ctrl (reducing g_K depolarises E_L_eff).
    EC50 : float
        Hill EC50 in ng/mL.
    hill_n : float
        Hill cooperativity exponent.

    Returns
    -------
    np.ndarray, shape (n_regions,)
        Per-region g_K values [nS] for use in Zerlaut_gK_gNa.
    """
    occ = hill_occupancy(emax_ng_ml, EC50=EC50, n=hill_n, occ_max=occ_max)
    rec = np.asarray(receptor_map, dtype=float).ravel()
    rec_min, rec_max = rec.min(), rec.max()
    rec_norm = (rec - rec_min) / (rec_max - rec_min + 1e-12)
    delta = g_K_ctrl - g_K_drug_max          # always > 0
    return g_K_ctrl - float(occ) * rec_norm * delta


def el_eff_from_gK_gNa(g_K: float | np.ndarray, g_Na: float, E_K: float = -90.0, E_Na: float = 50.0) -> float | np.ndarray:
    """Compute the effective leak reversal potential E_L_eff [mV].

    E_L_eff = (g_Na · E_Na + g_K · E_K) / (g_Na + g_K)

    This is the weighted-average resting potential that determines where the
    mean membrane voltage drifts in the absence of synaptic input.
    """
    return (g_Na * E_Na + g_K * E_K) / (g_Na + g_K)


def leak_to_conductances(E_Na: float, E_K: float, E_L: float, g_L: float | None = None, g_Na: float | None = None):
    """Convert leak potential constraints into `(g_K, g_Na)`.

    Mirrors the relation used in legacy TVBSim receptor workflows.
    """
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        g_Na = g_L - g_K
        return float(g_K), float(g_Na)
    if g_Na is None:
        raise ValueError("Provide either g_L or g_Na.")
    g_L = g_Na * (E_Na - E_K) / (E_L - E_K)
    g_K = g_L - g_Na
    return float(g_K), float(g_Na)


def receptor_to_gk_profile(gk_start: float, gk_end: float, receptors: np.ndarray) -> np.ndarray:
    """Interpolate region-wise g_K values from receptor density profile."""
    receptors = np.asarray(receptors, dtype=float).reshape(-1)
    if np.allclose(receptors.max(), receptors.min()):
        return np.full_like(receptors, fill_value=float(gk_start))
    return np.interp(receptors, [receptors.min(), receptors.max()], [gk_start, gk_end])

