"""TVBToolkit public API."""

from tvbtoolkit.analysis import (
    AlignmentResult,
    BrainStateSummary,
    LEGACY_REFERENCE_PATHS,
    PHIID_ATOMS,
    PRIMARY_ATOMS,
    PUBLICATION_COHORT_ORDER,
    average_atom_matrices_by_group,
    align_states_to_templates,
    brain_state_metrics_dict,
    build_matlab_batch_command,
    centers_to_matrices,
    centroid_similarity_matrix,
    cluster_brain_states,
    default_atom_cmap,
    build_annotation_template,
    compute_fc_matrix,
    centroid_connectivity_similarity,
    compare_centroids_to_connectomes,
    connectivity_vectors_from_records,
    edge_rank_gradient,
    edge_vector_to_matrix,
    export_phiid_subject_inputs,
    fit_dynamic_state_model,
    fit_incremental_pca,
    fit_state_templates,
    iter_subject_feature_blocks,
    load_survival_arrays,
    load_local_phiid_index,
    load_local_phiid_subject,
    load_phiid_index,
    load_phiid_matrix,
    matrix_spearman_similarity,
    nodal_strength,
    parse_local_phiid_name,
    parse_phiid_output_name,
    phase_patterns,
    pool_reduced_dynamic_features,
    plot_publication_cohort_grid,
    plot_group_average_grid,
    plot_phiid_matrix,
    plot_survival_heatmap,
    publication_atom_cmap,
    redundancy_synergy_rank_gradient,
    save_annotation_template,
    save_group_average_outputs,
    safe_pearson,
    sanitize_subject_stub,
    score_kmeans_range,
    split_reconstructed_features,
    summarize_within_between,
    subject_records_to_frame,
    summarize_brain_states,
    threshold_top_density,
    upper_triangle_values,
    weighted_global_efficiency,
    weighted_modularity,
)
from tvbtoolkit.bold import (
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
from tvbtoolkit.complexity.measures import (
    ace,
    lzc_multichannel,
    lzc_single_channel,
    pci_casali_like,
    pci_ratio_proxy,
    sce,
)
from tvbtoolkit.core.config import OutputConfig, SingleRegionConfig, WholeBrainConfig
from tvbtoolkit.core.system import SystemSpecs, detect_system_specs, recommend_parallel_workers
from tvbtoolkit.datasets import (
    AAL90Atlas,
    StructuralMetadata,
    TractLengthSanity,
    convert_brain_act_dataset,
    list_subjects,
    load_aal90_atlas,
    load_subject_structural,
    normalize_connectivity,
    threshold_connectivity,
    validate_structural_matrices,
)
try:
    from tvbtoolkit.single_region.simulation import SingleRegionResult, run_single_region_simulation
except Exception as _single_region_import_error:
    SingleRegionResult = None
    run_single_region_simulation = None

from tvbtoolkit.single_region.analysis import (
    bin_array,
    calculate_psd_fmax,
    heaviside,
    input_rate,
    prepare_population_rates,
)
try:
    from tvbtoolkit.whole_brain.analysis import fcsc_seedwise_from_saved_batch
    from tvbtoolkit.whole_brain.simulation import WholeBrainResult, run_whole_brain_simulation
except Exception as _whole_brain_import_error:
    fcsc_seedwise_from_saved_batch = None
    WholeBrainResult = None
    run_whole_brain_simulation = None

from tvbtoolkit.visualization import (
    plot_brain_state_occupancy,
    plot_cohort_subject_metrics,
    plot_example_timeseries,
    plot_metric_summary,
    plot_sfc_vs_occupancy,
    plot_single_region_timeseries,
    plot_timeseries,
    set_publication_style,
)

try:
    from tvbtoolkit.workflows.pipelines import (
        ComplexitySummary,
        run_single_region_with_complexity,
        run_whole_brain_with_complexity,
    )
except Exception as _pipelines_import_error:
    ComplexitySummary = None
    run_single_region_with_complexity = None
    run_whole_brain_with_complexity = None
try:
    from tvbtoolkit.workflows.experiments import ConditionSpec, run_condition_batch
except Exception as _experiments_import_error:
    ConditionSpec = None
    run_condition_batch = None

try:
    from tvbtoolkit.workflows.brain_act_subjects import (
        BrainActSubjectConfig,
        run_brain_act_all_cohorts,
        run_cohort_batch,
        run_subject_simulation,
    )
except Exception as _brain_act_subjects_import_error:
    BrainActSubjectConfig = None
    run_brain_act_all_cohorts = None
    run_cohort_batch = None
    run_subject_simulation = None
try:
    from tvbtoolkit.workflows.presets import (
        build_stimulation_override,
        ketamine_depth_conditions,
        maria_sacha_nature_conditions,
        psilocybin_receptor_conditions,
        stimulation_schedule,
    )
except Exception as _presets_import_error:
    build_stimulation_override = None
    ketamine_depth_conditions = None
    maria_sacha_nature_conditions = None
    psilocybin_receptor_conditions = None
    stimulation_schedule = None
from tvbtoolkit.workflows.pharmacology import leak_to_conductances, receptor_to_gk_profile
try:
    from tvbtoolkit.brian_mf import (
        TransferFunctionFitConfig as BrianMFTransferFunctionFitConfig,
        TransferFunctionFitResult as BrianMFTransferFunctionFitResult,
        calculate_mf_difference as brian_mf_calculate_mf_difference,
        fit_adex_transfer_function as brian_mf_fit_adex_transfer_function,
        get_neuron_params_double_cell as brian_mf_get_neuron_params_double_cell,
        make_fit_from_data as brian_mf_make_fit_from_data,
        run_adex_network_simulation as brian_mf_run_adex_network_simulation,
        run_mean_field_simulation as brian_mf_run_mean_field_simulation,
        run_single_cell_adex as brian_mf_run_single_cell_adex,
        run_snn_split_leak as brian_mf_run_snn_split_leak,
    )
except Exception as _brian_mf_import_error:
    BrianMFTransferFunctionFitConfig = None
    BrianMFTransferFunctionFitResult = None
    brian_mf_calculate_mf_difference = None
    brian_mf_fit_adex_transfer_function = None
    brian_mf_get_neuron_params_double_cell = None
    brian_mf_make_fit_from_data = None
    brian_mf_run_adex_network_simulation = None
    brian_mf_run_mean_field_simulation = None
    brian_mf_run_single_cell_adex = None
    brian_mf_run_snn_split_leak = None

__all__ = [
    "WholeBrainConfig",
    "WholeBrainResult",
    "run_whole_brain_simulation",
    "SingleRegionConfig",
    "SingleRegionResult",
    "run_single_region_simulation",
    "bin_array",
    "heaviside",
    "input_rate",
    "prepare_population_rates",
    "calculate_psd_fmax",
    "OutputConfig",
    "BOLDParams",
    "butter_filtering",
    "preprocess_bold_signal",
    "corr_fc_sc",
    "corr_FC_SC",
    "plot_fc_sc",
    "plot_FC_SC",
    "first_order_volterra_hrf",
    "bold_from_firing_rates",
    "lzc_multichannel",
    "lzc_single_channel",
    "ace",
    "sce",
    "pci_casali_like",
    "pci_ratio_proxy",
    "ComplexitySummary",
    "run_whole_brain_with_complexity",
    "run_single_region_with_complexity",
    "ConditionSpec",
    "run_condition_batch",
    "BrainActSubjectConfig",
    "run_subject_simulation",
    "run_cohort_batch",
    "run_brain_act_all_cohorts",
    "set_publication_style",
    "plot_example_timeseries",
    "plot_single_region_timeseries",
    "plot_timeseries",
    "plot_metric_summary",
    "plot_cohort_subject_metrics",
    "plot_brain_state_occupancy",
    "plot_sfc_vs_occupancy",
    "ketamine_depth_conditions",
    "maria_sacha_nature_conditions",
    "psilocybin_receptor_conditions",
    "stimulation_schedule",
    "build_stimulation_override",
    "fcsc_seedwise_from_saved_batch",
    "leak_to_conductances",
    "receptor_to_gk_profile",
    "BrianMFTransferFunctionFitConfig",
    "BrianMFTransferFunctionFitResult",
    "brian_mf_get_neuron_params_double_cell",
    "brian_mf_run_mean_field_simulation",
    "brian_mf_run_single_cell_adex",
    "brian_mf_run_adex_network_simulation",
    "brian_mf_run_snn_split_leak",
    "brian_mf_calculate_mf_difference",
    "brian_mf_fit_adex_transfer_function",
    "brian_mf_make_fit_from_data",
    "SystemSpecs",
    "detect_system_specs",
    "recommend_parallel_workers",
    "AAL90Atlas",
    "StructuralMetadata",
    "TractLengthSanity",
    "convert_brain_act_dataset",
    "load_aal90_atlas",
    "list_subjects",
    "load_subject_structural",
    "validate_structural_matrices",
    "normalize_connectivity",
    "threshold_connectivity",
    "BrainStateSummary",
    "AlignmentResult",
    "safe_pearson",
    "phase_patterns",
    "cluster_brain_states",
    "summarize_brain_states",
    "centers_to_matrices",
    "brain_state_metrics_dict",
    "centroid_similarity_matrix",
    "fit_state_templates",
    "align_states_to_templates",
    "load_survival_arrays",
    "plot_survival_heatmap",
    "PHIID_ATOMS",
    "PRIMARY_ATOMS",
    "PUBLICATION_COHORT_ORDER",
    "LEGACY_REFERENCE_PATHS",
    "sanitize_subject_stub",
    "subject_records_to_frame",
    "export_phiid_subject_inputs",
    "build_matlab_batch_command",
    "parse_phiid_output_name",
    "load_phiid_matrix",
    "load_phiid_index",
    "average_atom_matrices_by_group",
    "save_group_average_outputs",
    "default_atom_cmap",
    "publication_atom_cmap",
    "plot_phiid_matrix",
    "plot_group_average_grid",
    "plot_publication_cohort_grid",
    "build_annotation_template",
    "compute_fc_matrix",
    "parse_local_phiid_name",
    "load_local_phiid_subject",
    "load_local_phiid_index",
    "edge_vector_to_matrix",
    "build_subject_dynamic_features",
    "iter_subject_feature_blocks",
    "fit_incremental_pca",
    "pool_reduced_dynamic_features",
    "score_kmeans_range",
    "fit_dynamic_state_model",
    "split_reconstructed_features",
    "reconstruct_state_centroids",
    "connectivity_vectors_from_records",
    "centroid_connectivity_similarity",
    "compare_centroids_to_connectomes",
    "upper_triangle_values",
    "matrix_spearman_similarity",
    "nodal_strength",
    "threshold_top_density",
    "redundancy_synergy_rank_gradient",
    "edge_rank_gradient",
    "weighted_global_efficiency",
    "weighted_modularity",
    "summarize_within_between",
    "save_annotation_template",
]
