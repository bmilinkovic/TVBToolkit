"""Brian2/mean-field tools ported from legacy TVBSim `brian_MF`.

This namespace contains parity-oriented implementations of the legacy
AdEx single-cell/network and transfer-function fitting pipeline, with a cleaner API.
"""

from tvbtoolkit.brian_mf.mean_field.mf import (
    calculate_mf_difference,
    run_mean_field_simulation,
)
from tvbtoolkit.brian_mf.adex import (
    NetworkSimulationResult,
    SingleCellResult,
    run_adex_network_simulation,
    run_single_cell_adex,
    run_snn_split_leak,
)
from tvbtoolkit.brian_mf.analysis import (
    BCriticalResult,
    DynamicSweepResult,
    compute_b_critical_grid,
    parse_np_arange_csv,
    run_dynamic_sweep,
)
from tvbtoolkit.brian_mf.mean_field.tf_calc import (
    TransferFunctionFitConfig,
    TransferFunctionFitResult,
    fit_adex_transfer_function,
    get_connectivity_and_synapses_matrix,
    get_neuron_params,
    get_neuron_params_double_cell,
    list_param_sets,
    load_param_set,
    make_fit_from_data,
    save_param_set,
)

__all__ = [
    "get_neuron_params_double_cell",
    "get_neuron_params",
    "get_connectivity_and_synapses_matrix",
    "run_mean_field_simulation",
    "calculate_mf_difference",
    "TransferFunctionFitConfig",
    "TransferFunctionFitResult",
    "fit_adex_transfer_function",
    "make_fit_from_data",
    "SingleCellResult",
    "run_single_cell_adex",
    "NetworkSimulationResult",
    "run_adex_network_simulation",
    "run_snn_split_leak",
    "parse_np_arange_csv",
    "BCriticalResult",
    "DynamicSweepResult",
    "compute_b_critical_grid",
    "run_dynamic_sweep",
    "save_param_set",
    "load_param_set",
    "list_param_sets",
]
