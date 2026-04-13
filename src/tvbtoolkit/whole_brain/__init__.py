"""Whole-brain simulation APIs."""

from tvbtoolkit.whole_brain.analysis import (
    correlation_fc,
    fcsc_seedwise_from_saved_batch,
    plot_region_timeseries,
)
from tvbtoolkit.whole_brain.simulation import WholeBrainResult, run_whole_brain_simulation

__all__ = [
    "WholeBrainResult",
    "run_whole_brain_simulation",
    "plot_region_timeseries",
    "correlation_fc",
    "fcsc_seedwise_from_saved_batch",
]
