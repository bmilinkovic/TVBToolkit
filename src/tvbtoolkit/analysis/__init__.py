"""Analysis utilities for TVBToolkit."""

from tvbtoolkit.analysis.brain_states import (
    BrainStateSummary,
    brain_state_metrics_dict,
    centers_to_matrices,
    cluster_brain_states,
    phase_patterns,
    sfc_sort_centroids,
    summarize_brain_states,
)
from tvbtoolkit.analysis.spectral import (
    PSDResult,
    ValidityResult,
    dominant_frequency,
    phase_coherence_validity,
    psd_per_region,
)
from tvbtoolkit.analysis.state_alignment import (
    AlignmentResult,
    align_states_to_templates,
    centroid_similarity_matrix,
    fit_state_templates,
    safe_pearson,
)
from tvbtoolkit.analysis.dynamics import load_survival_arrays, plot_survival_heatmap

__all__ = [
    # brain_states
    "BrainStateSummary",
    "phase_patterns",
    "cluster_brain_states",
    "summarize_brain_states",
    "centers_to_matrices",
    "brain_state_metrics_dict",
    "sfc_sort_centroids",
    # spectral
    "PSDResult",
    "ValidityResult",
    "psd_per_region",
    "dominant_frequency",
    "phase_coherence_validity",
    # state_alignment
    "AlignmentResult",
    "safe_pearson",
    "centroid_similarity_matrix",
    "fit_state_templates",
    "align_states_to_templates",
    # dynamics
    "load_survival_arrays",
    "plot_survival_heatmap",
]
