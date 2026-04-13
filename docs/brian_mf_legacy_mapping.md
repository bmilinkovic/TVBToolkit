# brian_MF Legacy Audit and Mapping (TVBSim -> TVBToolkit)

This document records a parity-first migration plan for legacy `brian_MF` in:
`/Users/borjan/CNRS/projects/TVBSim/brian_MF`

Target package root:
`/Users/borjan/CNRS/projects/TVBToolkit/src/tvbtoolkit/brian_mf`

## Audit summary

- Legacy `brian_MF` is script-heavy: many files execute simulations directly via CLI side effects.
- Core reusable numerics live mainly in:
  - `Tf_calc/theoretical_tools.py`
  - `Tf_calc/cell_library.py`
  - `MF.py`
  - `brian_functions.py`
- Major side-effect folders are:
  - `Tf_calc/data/` (TF and fit arrays)
  - `Tf_calc/net_compar/` (network comparison arrays)
  - `Dyn_Analysis/` + `Dyn_Analysis/dynamical_precalc/`
  - `result/`

## Mapping table (legacy -> TVBToolkit)

| Legacy path | Provides | Inputs/outputs (shape, units) | TVBToolkit target | Status |
|---|---|---|---|---|
| `brian_MF/__init__.py` | package marker | none | `brian_mf/__init__.py` | **port** |
| `brian_MF/MF.py` | `calculate_mf_difference` wrapper over MF simulation | input: `fr_both[:,(inh,exc,input)]`, `inputs`, fit coeffs `PRS/PFS`; output: scalar mean abs diff | `brian_mf/mean_field/mf.py` | **port (core)** |
| `brian_MF/brian_functions.py` | PSD helpers, binning, Heaviside, injected input profile, raster+rate plotting, FR prep | arrays in ms/Hz, binned outputs | `brian_mf/adex/brian_utils.py` | **port (core utilities)** |
| `brian_MF/adex_simulation_network.py` | CLI network SNN simulation (RS/FS populations), save mean/all arrays | args: model/network params; outputs: rate traces (Hz), raster arrays, optional `.npy` saves | `brian_mf/adex/network.py` | **port (core)** |
| `brian_MF/adex_gK_gNa.py` | network SNN variant with leak split into `gK/gNa`, `run_SNN` callable | input: seed/time/input and optional psych leak shift; output: raster/rates/adaptation arrays + optional save | `brian_mf/adex/network.py` + `brian_mf/receptors.py` | **port (core)** |
| `brian_MF/single_cell_sim.py` | CLI single-cell AdEx simulation | input: single-cell params + current/time; output: membrane voltage and spikes plot | `brian_mf/adex/single_cell.py` | **port (core)** |
| `brian_MF/single_cell_sim_modified.py` | single-cell variant using receptor-based leak conversion | same as above + conversion helper | `brian_mf/adex/single_cell.py` + `brian_mf/receptors.py` | **port (core)** |
| `brian_MF/receptors.py` | receptor maps + conductance conversion (`conversion`) | receptor vectors (68 cortical), conductance/leak conversion values | `brian_mf/receptors.py` | **port** |
| `brian_MF/survival_time.py` | survival-time loading/compute/plot, network mean loader | input: binned network `.npy`; outputs: heatmaps, survival arrays | `brian_mf/analysis/survival_time.py` | **port** |
| `brian_MF/MF_script_with_OS.py` | monolithic CLI MF simulation using fitted TFs | input: CLI params; output: time-series FR/adaptation plot | `brian_mf/mean_field/mf.py` + notebook | **notebook-only pipeline** |
| `brian_MF/gK_gNa_MF_script_with_OS.py` | monolithic CLI MF with `gK/gNa` + psych mode | same pattern, with leak conversion and optional save | `brian_mf/mean_field/mf.py` + notebook | **notebook-only pipeline** |
| `brian_MF/Tf_calc/cell_library.py` | neuron/cell parameter presets, SI conversion | dict presets in mV/ms/nS/pA or SI | `brian_mf/mean_field/tf_calc.py` | **port (core)** |
| `brian_MF/Tf_calc/model_library.py` | AdEx equation string factory | input model name -> Brian2 equations | `brian_mf/adex/single_cell.py` (internal eq builder) | **port (integrated in simulation equations)** |
| `brian_MF/Tf_calc/syn_and_connec_library.py` | network synapse/connectivity preset matrix | output object matrix with synaptic params | `brian_mf/mean_field/tf_calc.py` | **port** |
| `brian_MF/Tf_calc/theoretical_tools.py` | TF fitting maths + MF simulator + diagnostics + fit workflow | arrays of rates/adaptations; outputs fitted coefficients `P(10,)`, MF estimates, fit diagnostics | `brian_mf/mean_field/tf_calc.py`, `brian_mf/mean_field/mf.py` | **port (core)** |
| `brian_MF/Tf_calc/tf_simulation.py` | script generating TF grid from SNN two-pop simulations | outputs `ExpTF_*`, adapt, muV arrays and params ranges | `brian_mf/mean_field/tf_calc.py` + notebook | **notebook-only pipeline** |
| `brian_MF/Tf_calc/tf_simulation_single_cell.py` | script generating single-cell TF grids | outputs `ExpTF_*` and params arrays | `brian_mf/mean_field/tf_calc.py` + notebook | **notebook-only pipeline** |
| `brian_MF/Tf_calc/setup.py` | environment bootstrap script (imports/clear output) | side effects only | **drop** (superseded by package deps) | **deprecated** |
| `brian_MF/Dyn_Analysis/calculate_b_crit.py` | critical `b_e` sweep using fixed-point MF curves | output `b_thresh_tau_*.npy` | `brian_mf/analysis/dyn_analysis.py` | **port (core)** |
| `brian_MF/Dyn_Analysis/net_sims_dyn_analysis.py` | large SNN parameter sweep for dynamic survival maps | output binned network sims + optional survival summaries | `brian_mf/analysis/dyn_analysis.py` + notebook | **port (core)** |
| `brian_MF/Dyn_Analysis/dynamical_precalc/*.npy` | precalculated sweep products | arrays for direct plotting/loading | `data/brian_mf/dynamical_precalc/` + loader | **wrapper/data copy** |
| `brian_MF/Dyn_Analysis/trials/*.npy` | trial sweep arrays | arrays | `data/brian_mf/trials/` + loader | **wrapper/data copy** |
| `brian_MF/result/*.npy` | MF-vs-SNN comparison artifacts | arrays | `brian_mf/io/storage.py` managed outputs | **wrapper migration** |
| `brian_MF/Tf_calc/data/*.npy` | TF experiments + fits + parameter arrays | arrays used by fitting scripts | `data/brian_mf/tf_calc/` + loader | **wrapper/data copy** |
| `brian_MF/Tf_calc/net_compar/*.npy` | network-vs-MF comparison arrays | arrays | `data/brian_mf/net_compar/` + loader | **wrapper/data copy** |

## Minimal parity fixtures (phase 1)

Core fixtures needed first (before refactor):

1. **Transfer-function analytic parity**
- Compare `eff_thresh`, `output_rate`, `mu_sig_tau_func` on deterministic synthetic grids.
- Inputs: fixed `ve`, `vi`, `FF`, `adapt` arrays + fixed parameter set (`FS-RS`).
- Checks: `np.allclose` with strict tolerances (`rtol<=1e-10`, `atol<=1e-12` where numerically stable).

2. **MF simulation parity (`run_MF`)**
- Compare legacy and TVBToolkit `run_MF` outputs with identical seeds and TF coefficients (`PRS/PFS`) from legacy files.
- Inputs: deterministic seed, `AmpStim`, `Iext`, short simulation (`TotTime=2s`) for test runtime.
- Checks: relative error and absolute error on mean exc/inh rates.

3. **MF difference wrapper parity (`calculate_mf_difference`)**
- Compare scalar diff against legacy function for a fixed synthetic `fr_both` matrix and `inputs` vector.

## Notes on side effects and deprecations

- Legacy scripts relied on cwd-relative paths (`./Tf_calc/data/...`, `./Dyn_Analysis/...`).
- TVBToolkit will route all persistence through `brian_mf/io/storage.py` with explicit paths.
- Monolithic legacy scripts become notebook workflows + callable APIs; CLI wrappers may be added as thin frontends only.
- Dedicated parity demo notebook for legacy-style figures:
  - `/Users/borjan/CNRS/projects/TVBToolkit/notebooks/brian_mf_network_single_cell_parity_demo.ipynb`
