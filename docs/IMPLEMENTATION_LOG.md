# Implementation Log

## Scope

Created a new standalone package (`tvbtoolkit`) in a separate repository:

- Path: `/Users/borjan/CNRS/projects/TVBToolkit`
- Goal: publishable toolbox architecture, independent of legacy TVBSim internals.
- Constraint respected: simulation engines rely on TVB and Brian2.

## Architectural Decisions

1. Fresh package layout under `src/tvbtoolkit`.
2. Explicit core configuration objects (`WholeBrainConfig`, `SingleRegionConfig`).
3. AdEx-centered single-region module implemented directly in Brian2.
4. Complexity module implemented from scratch (no legacy wrappers).
5. Workflow utilities to combine simulations + metrics.
6. Whole-brain strict-parity path now defaults to legacy Zerlaut AdEx family.
7. Added system-aware parallel recommendations and notebook hardware stamping.

## Files Added

- `pyproject.toml`
- `README.md`
- `src/tvbtoolkit/__init__.py`
- `src/tvbtoolkit/core/config.py`
- `src/tvbtoolkit/core/io.py`
- `src/tvbtoolkit/core/__init__.py`
- `src/tvbtoolkit/whole_brain/simulation.py`
- `src/tvbtoolkit/whole_brain/analysis.py`
- `src/tvbtoolkit/whole_brain/__init__.py`
- `src/tvbtoolkit/single_region/simulation.py` (AdEx primary)
- `src/tvbtoolkit/single_region/__init__.py`
- `src/tvbtoolkit/complexity/measures.py`
- `src/tvbtoolkit/complexity/__init__.py`
- `src/tvbtoolkit/workflows/pipelines.py`
- `src/tvbtoolkit/workflows/__init__.py`
- `src/tvbtoolkit/core/system.py`
- `src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut.py`
- `src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_gK_gNa.py`
- `src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_matteo.py`
- `src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_matteo_gK_gNa.py`
- `src/tvbtoolkit/whole_brain/legacy_engine/parameter/parameter_M_Berlin_new.py`

## AdEx Priority

The single-region core is now a two-population AdEx network:

- Exponential IF membrane equation with adaptation variable `w`.
- Separate excitatory/inhibitory populations.
- Conductance-based synapses with recurrent and external Poisson input.
- Population-rate outputs plus membrane-potential traces.

The whole-brain core now defaults to AdEx mean-field neural mass (Zerlaut family) with:

- Legacy model-selection flags (`matteo`, `gK_gNa`, `order`).
- Legacy parameter schema (`parameter_M_Berlin_new`) with override support.
- Legacy-style coupling/integrator/monitor/stimulation configuration path.

Parallel runtime updates:

- `run_condition_batch(...)` supports multi-process execution (`n_jobs`).
- CPU worker recommendation utility added (`recommend_parallel_workers`).
- Notebook cells now capture and display host specs for reproducibility.
- Explicitly documented that current implementation is CPU-based (no default iGPU acceleration).

## Next Iteration Recommended

1. Add benchmark notebooks reproducing legacy AdEx scenarios one-to-one.
2. Add regression tests against known legacy output statistics.
3. Add richer stimulation protocol API (pulses, ramps, patterned perturbations).

## Patch: Notebook + Legacy Override Reliability (Feb 2026)

Changes applied to address notebook runtime failures and improve execution UX:

- Fixed legacy override handling in `/Users/borjan/CNRS/projects/TVBToolkit/src/tvbtoolkit/whole_brain/simulation.py`:
  - `_apply_parameter_overrides(...)` now accepts direct top-level legacy blocks such as `parameter_model`, `parameter_monitor`, etc.
  - This resolves errors like `KeyError: Unknown override key 'parameter_model' for legacy parameter schema.`

- Verified and retained process-parallel and progress support in `/Users/borjan/CNRS/projects/TVBToolkit/src/tvbtoolkit/workflows/experiments.py`:
  - `run_condition_batch(...)` exposes `n_jobs`, `use_processes`, and `show_progress`.
  - Includes runtime status prints and tqdm progress bar when available.

- Cleaned rendering issues in first notebook and standardized execution parameters:
  - `/Users/borjan/CNRS/projects/TVBToolkit/notebooks/control_ketamine_psilocybin_lzc_pci.ipynb`
  - Removed literal escaped `\\n` cell content artifacts so code/text render normally.
  - Added explicit `n_jobs=N_JOBS`, `use_processes=True`, `show_progress=True` in batch run cell.

- Updated second notebook to use explicit parallel/progress arguments in all condition-batch calls:
  - `/Users/borjan/CNRS/projects/TVBToolkit/notebooks/legacy_parity_ketamine_psilocybin_sweeps.ipynb`

Notes:
- In notebook sessions, restart kernel before rerun so worker processes import the latest edited code.
- For editable installs (`pip install -e .`), source edits are immediately reflected after kernel restart.

## Patch: FAST_MODE Toggle for Iteration vs Final Runs (Feb 2026)

Added a one-cell runtime mode switch to both notebooks:

- `/Users/borjan/CNRS/projects/TVBToolkit/notebooks/control_ketamine_psilocybin_lzc_pci.ipynb`
- `/Users/borjan/CNRS/projects/TVBToolkit/notebooks/legacy_parity_ketamine_psilocybin_sweeps.ipynb`

What the toggle controls:

- Total simulation duration (fast mode shortened)
- Number of seeds (fast mode reduced)
- Number of stimulation times in sweep notebook (fast mode reduced)
- Timeseries persistence (`save_timeseries=False` in fast mode to reduce I/O)
- Parallel worker cap (`n_jobs_cap`) to avoid oversubscription on laptop workloads

Behavior:

- `FAST_MODE=True` for iterative development and quick checks
- `FAST_MODE=False` for final-quality parity figures

This preserves model equations and core solver settings (e.g., `dt_ms=0.1`) while reducing run multiplicity and I/O in fast mode.

## Patch: Brain-Act AAL90 Structural Dataset Integration (Feb 2026)

Added a full Brain-Act data integration path with conversion, loaders, validation, and examples.

### New modules and APIs

- Added dataset package:
  - `/Users/borjan/CNRS/projects/TVBToolkit/src/tvbtoolkit/datasets/__init__.py`
  - `/Users/borjan/CNRS/projects/TVBToolkit/src/tvbtoolkit/datasets/brain_act.py`

- Public API exports added in:
  - `/Users/borjan/CNRS/projects/TVBToolkit/src/tvbtoolkit/__init__.py`

New user-facing functions:
- `convert_brain_act_dataset(...)`
- `load_aal90_atlas(...)`
- `list_subjects(...)`
- `load_subject_structural(...)`
- `validate_structural_matrices(...)`
- `normalize_connectivity(...)`
- `threshold_connectivity(...)`

### One-shot conversion script

- Added:
  - `/Users/borjan/CNRS/projects/TVBToolkit/scripts/convert_brain_act_dataset.py`

Conversion output format:
- `atlas.npz`
- `subjects_control.npz`
- `subjects_mcs.npz`
- `subjects_uws.npz`
- `index.json` (subjects/cohorts/shapes/checksums)

### Documentation and audit

- Added audit and mapping document:
  - `/Users/borjan/CNRS/projects/TVBToolkit/docs/BRAIN_ACT_DATA_AUDIT.md`

- Updated README with:
  - conversion commands
  - loader examples
  - subject-specific simulation example
  - repo layout updates including `datasets/` and `scripts/`

### Example usage artifacts

- Added script:
  - `/Users/borjan/CNRS/projects/TVBToolkit/examples/brain_act_structural_demo.py`

- Added notebook:
  - `/Users/borjan/CNRS/projects/TVBToolkit/notebooks/brain_act_subject_loading_and_simulation.ipynb`

### Tests

- Added:
  - `/Users/borjan/CNRS/projects/TVBToolkit/tests/test_brain_act_dataset.py`

Coverage includes:
- conversion of synthetic organised data into cohort NPZ bundles
- cohort-aware subject listing
- ambiguous subject-id handling when cohort is omitted
- matrix validation and cleaning behaviour (symmetry/non-finite handling)

## Patch: Brain-Act Notebook Documentation + Publication Figures + Local Raw Mirror (Feb 2026)

### Notebook upgrade

Updated:
- `/Users/borjan/CNRS/projects/TVBToolkit/notebooks/brain_act_subject_loading_and_simulation.ipynb`

Changes:
- Expanded text/markdown documentation cells describing:
  - project context,
  - cohort definitions,
  - analysis rationale,
  - reproducibility/scaling notes.
- Switched to a pilot sampling strategy of **3 subjects per cohort**.
- Added publication-oriented figure cells:
  - cohort mean SC heatmaps,
  - structural QC distributions (weights, delays, degree),
  - simulated mean activity trajectories with cohort overlays,
  - complexity metric cohort comparison (LZc, PCI Casali-like).

### Local raw data mirror support

Added:
- `/Users/borjan/CNRS/projects/TVBToolkit/scripts/sync_brain_act_source_data.py`

Purpose:
- Mirror Brain-Act data into `TVBToolkit/data/brain_act/source` so workflows are not tied to an external checkout.
- Includes `raw/` by default (optional `--no-raw` to skip).

Also updated:
- `/Users/borjan/CNRS/projects/TVBToolkit/scripts/convert_brain_act_dataset.py`

Change:
- `--source-root` is now optional and defaults to local `data/brain_act/source`.

### README updates

Updated:
- `/Users/borjan/CNRS/projects/TVBToolkit/README.md`

Added instructions for:
- syncing local source data,
- converting local source to fast bundle format,
- running the upgraded Brain-Act notebook.

## Patch: Subject-Specific Brain-Act AAL90 Workflow + Brain-State Analysis (Feb 2026)

Implemented a dedicated subject-level workflow in `TVBToolkit` for Brain-Act structural cohorts while keeping the existing toolbox simulation core unchanged.

### New modules

- `src/tvbtoolkit/workflows/brain_act_subjects.py`
  - `BrainActSubjectConfig`
  - `run_subject_simulation(subject_id, cohort, cfg)`
  - `run_cohort_batch(cohort, subjects, cfg, n_jobs, use_processes)`
  - `run_brain_act_all_cohorts(...)`

- `src/tvbtoolkit/analysis/brain_states.py`
  - Hilbert-phase pattern extraction
  - k-means clustering of phase-coherence patterns
  - occupancy and transition summaries

- `src/tvbtoolkit/analysis/__init__.py`

### API exports updated

- `src/tvbtoolkit/workflows/__init__.py`
- `src/tvbtoolkit/visualization/__init__.py`
- `src/tvbtoolkit/__init__.py`

### New plotting helpers

Added in `src/tvbtoolkit/visualization/plotting.py`:

- `plot_cohort_subject_metrics(...)`
- `plot_brain_state_occupancy(...)`

### Brain-Act parity and QC

- Added explicit parity handling and QC reporting for lesion/damage zeros in subject SC/TL.
- Added QC script:
  - `scripts/brain_act_damage_mask_qc.py`
- Added parity documentation:
  - `docs/BRAIN_ACT_MASK_PARITY.md`

### Notebook update

Reworked:

- `notebooks/brain_act_subject_loading_and_simulation.ipynb`

Notebook now demonstrates:

- one-subject-per-cohort AAL90 runs using the new workflow API,
- temporal-average monitor settings,
- LZc/PCI Casali-like extraction,
- phase-state occupancy plotting,
- full-cohort batch entry point.

### Notes on model parity

- Subject-specific runs use the same AdEx/Zerlaut family used by ketamine/psilocybin workflows.
- No receptor modulation is applied in this Brain-Act subject pipeline.
- Simulation duration default set to 5000 ms for parity with prior notebook runs.
