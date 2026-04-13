"""Figure generation APIs."""

from tvbtoolkit.visualization.plotting import (
    plot_brain_state_occupancy,
    plot_cohort_subject_metrics,
    plot_example_timeseries,
    plot_metric_summary,
    plot_sfc_vs_occupancy,
    plot_single_region_timeseries,
    plot_timeseries,
    set_publication_style,
)

__all__ = [
    "set_publication_style",
    "plot_example_timeseries",
    "plot_single_region_timeseries",
    "plot_timeseries",
    "plot_metric_summary",
    "plot_cohort_subject_metrics",
    "plot_brain_state_occupancy",
    "plot_sfc_vs_occupancy",
]
