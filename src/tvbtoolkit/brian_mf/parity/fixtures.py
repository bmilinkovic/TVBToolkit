"""Deterministic fixtures for brian_MF parity testing."""

from __future__ import annotations

import numpy as np


def fixed_fit_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic RS/FS coefficient vectors for parity tests."""

    prs = np.array([-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407], dtype=float)
    pfs = np.array([-0.0514, 0.0040, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153], dtype=float)
    return prs, pfs


def fixed_rate_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic (ve, vi, FF, adapt) arrays for TF fitting checks."""

    ve = np.linspace(0.1, 30.0, 20)
    vi = np.linspace(0.1, 30.0, 20)
    vve, vvi = np.meshgrid(ve, vi)
    ff = 0.02 * vve + 0.01 * vvi + 0.2 * np.sin(vve / 5.0) + 0.05
    ff = np.clip(ff, 1e-4, None)
    adapt = 5e-12 + 1e-13 * vve
    return ve, vi, ff, adapt
