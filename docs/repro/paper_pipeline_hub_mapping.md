# paper_pipeline_hub -> TVBToolkit Mapping

This document maps the key analysis blocks in
`/Users/borjan/CNRS/projects/paper_pipeline_hub/whole_pipeline.ipynb`
to integrated TVBToolkit modules.

## Whole-brain conditions + simulation
- paper: `functions.py::run_simulation_all`, notebook cells 15/18/24/33
- TVBToolkit:
  - `tvbtoolkit.workflows.presets.maria_sacha_nature_conditions`
  - `tvbtoolkit.workflows.experiments.run_condition_batch`
  - `tvbtoolkit.whole_brain.simulation.run_whole_brain_simulation`

## PCI computation and condition summary plot
- paper: `functions.py::calculate_PCI`, `plot_raincloud_with_stats`, notebook cells 25/26
- TVBToolkit:
  - `tvbtoolkit.complexity.pci_casali_like` (Casali-style)
  - metrics generated through `run_condition_batch`
  - summary figure via `tvbtoolkit.visualization.plot_metric_summary`

## BOLD + FC-SC coupling
- paper: `functions.py::corr_sc_fc`, `plot_FC_SC`, notebook cell 36
- TVBToolkit:
  - `tvbtoolkit.bold.bold_from_firing_rates`
  - `tvbtoolkit.bold.preprocess_bold_signal`
  - `tvbtoolkit.bold.corr_fc_sc`
  - `tvbtoolkit.whole_brain.analysis.fcsc_seedwise_from_saved_batch`

## Dynamical survival heatmap
- paper: `functions.py::load_survival`, `plot_heatmap_survival`, notebook cells 47/49
- TVBToolkit:
  - `tvbtoolkit.analysis.load_survival_arrays`
  - `tvbtoolkit.analysis.plot_survival_heatmap`

## Single-region utilities used in paper scripts
- paper: `functions.py::bin_array`, `heaviside`, `input_rate`, `prepare_FR`, `calculate_psd_fmax`
- TVBToolkit:
  - `tvbtoolkit.single_region.bin_array`
  - `tvbtoolkit.single_region.heaviside`
  - `tvbtoolkit.single_region.input_rate`
  - `tvbtoolkit.single_region.prepare_population_rates`
  - `tvbtoolkit.single_region.calculate_psd_fmax`

## Notes
- TVBToolkit implementation intentionally avoids runtime imports from
  `paper_pipeline_hub`.
- Data paths for precomputed dynamical arrays are configurable through
  `load_survival_arrays(..., paper_repo_root=...)`.
