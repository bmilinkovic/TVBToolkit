"""Pharmacology helpers for ketamine/psilocybin parameterization."""

from __future__ import annotations

import numpy as np


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

