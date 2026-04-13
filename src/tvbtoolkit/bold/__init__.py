"""BOLD processing and FC-SC coupling utilities.

Exports include legacy-compatible helpers ported from TVBSim and a deterministic
BOLD transform from mean-field rates.
"""

from .bold import (
    BOLDParams,
    bold_from_firing_rates,
    butter_filtering,
    corr_FC_SC,
    corr_fc_sc,
    first_order_volterra_hrf,
    plot_FC_SC,
    plot_fc_sc,
    preprocess_bold_signal,
)

__all__ = [
    "BOLDParams",
    "butter_filtering",
    "preprocess_bold_signal",
    "corr_fc_sc",
    "corr_FC_SC",
    "plot_fc_sc",
    "plot_FC_SC",
    "first_order_volterra_hrf",
    "bold_from_firing_rates",
]
