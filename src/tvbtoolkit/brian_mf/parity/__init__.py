"""Parity fixtures and comparison helpers against legacy TVBSim brian_MF."""

from tvbtoolkit.brian_mf.parity.compare import (
    compare_mf_with_legacy,
    compare_subthreshold_with_legacy,
)
from tvbtoolkit.brian_mf.parity.fixtures import (
    fixed_fit_coefficients,
    fixed_rate_grid,
)

__all__ = [
    "fixed_fit_coefficients",
    "fixed_rate_grid",
    "compare_mf_with_legacy",
    "compare_subthreshold_with_legacy",
]
