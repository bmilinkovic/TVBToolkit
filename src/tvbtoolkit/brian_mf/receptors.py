"""Receptor helpers ported from legacy `brian_MF/receptors.py`."""

from __future__ import annotations

import numpy as np


def get_5ht2a_receptors() -> np.ndarray:
    """Return legacy 5-HT2A receptor density map (Desikan-like cortical order)."""

    return np.array(
        [
            5.96,
            4.48,
            3.59,
            4.27,
            3.93,
            4.26,
            4.24,
            3.89,
            3.73,
            4.38,
            3.72,
            4.56,
            3.98,
            4.49,
            4.31,
            4.81,
            4.65,
            4.18,
            3.9,
            4.14,
            4.15,
            4.39,
            3.5,
            4.77,
            4.02,
            4.34,
            4.2,
            4.52,
            4.26,
            4.55,
            4.52,
            4.22,
            4.13,
            4.53,
            4.42,
            4.21,
            4.17,
            3.95,
            4.36,
            3.55,
            4.17,
            3.77,
            4.31,
            3.69,
            4.73,
            4.56,
            4.48,
            4.28,
            4.64,
            4.28,
            4.3,
            3.92,
            4.17,
            4.13,
            4.58,
            4.03,
            4.42,
            4.47,
            4.22,
            4.37,
            4.17,
            4.33,
            3.93,
            4.06,
            3.88,
            4.21,
            4.25,
            4.26,
        ]
    )


def get_5ht1a_receptors() -> np.ndarray:
    """Return legacy 5-HT1A receptor density map placeholder (same order as 5-HT2A)."""

    # Legacy code did not include a dedicated alternative map.
    return get_5ht2a_receptors().copy()


def get_g_k_values(
    g_k_max: float,
    g_k_min: float,
    include_5ht1a: bool = False,
    fht1a_effect: float = 1.0,
    receptors: np.ndarray | None = None,
) -> np.ndarray:
    """Map receptor densities to potassium conductance values.

    This mirrors the simple min-max normalization used in the legacy module.
    """

    rec = np.asarray(receptors) if receptors is not None else get_5ht2a_receptors()
    rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-12)
    g_k = g_k_min + rec_norm * (g_k_max - g_k_min)
    if include_5ht1a:
        rec1 = get_5ht1a_receptors()
        rec1_norm = (rec1 - rec1.min()) / (rec1.max() - rec1.min() + 1e-12)
        g_k = g_k * (1.0 - fht1a_effect * rec1_norm)
    return g_k


def conversion(e_na: float, e_k: float, e_l: float, g_l: float | None = None, g_na: float | None = None) -> tuple[float, float]:
    """Convert leak reversal potential into equivalent `(g_K, g_Na)` values.

    Parameters are legacy-compatible and unit-agnostic; the returned values follow
    the same unit system as `g_l`/`g_na`.
    """

    if g_l is None and g_na is None:
        raise ValueError("At least one of g_l or g_na must be provided.")
    if g_l is not None:
        g_na_eff = (g_l * (e_l - e_k)) / (e_na - e_k)
        g_k_eff = g_l - g_na_eff
    else:
        g_na_eff = float(g_na)
        g_k_eff = g_na_eff * (e_na - e_l) / (e_l - e_k)
    return float(g_k_eff), float(g_na_eff)
