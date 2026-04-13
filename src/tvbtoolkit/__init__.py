"""TVBToolkit public API."""

from tvbtoolkit.analysis import (
    AlignmentResult,
    BrainStateSummary,
    align_states_to_templates,
    brain_state_metrics_dict,
    centers_to_matrices,
    centroid_similarity_matrix,
    cluster_brain_states,
    fit_state_templates,
    load_survival_arrays,
    phase_patterns,
    plot_survival_heatmap,
    safe_pearson,
    summarize_brain_states,
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
]
