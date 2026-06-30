# TVBToolkit

`TVBToolkit` is a clean, standalone toolbox for whole-brain and single-region brain simulations, designed for reproducible research and publication workflows.

It provides:

- **Whole-brain AdEx mean-field simulation** (Zerlaut family) using TVB, with strict-parity options aligned to the legacy TVBSim model family.
- **Surface-based AdEx mean-field simulation** using TVB Cortex/local-connectivity objects.
- **Single-region AdEx E/I simulation** using Brian2.
- **Complexity measures** (including LZc and Casali-style PCI) implemented in a modern, scriptable API.
- **Condition-batch workflows** for control/ketamine/psilocybin experiments.
- **Publication-ready plotting utilities** and an intuitive output directory structure.

This repository is intentionally separate from legacy TVBSim so that both projects can be compared side-by-side.

## Installation

Recommended fresh-clone setup:

```bash
git clone https://github.com/bmilinkovic/TVBToolkit.git
cd TVBToolkit
conda env create -f environment.yml
conda activate tvbtoolkit
python -m ipykernel install --user --name tvbtoolkit --display-name "Python (TVBToolkit)"
```

Quick sanity check:

```bash
python -c "import tvbtoolkit; print(tvbtoolkit.__file__)"
```

If that prints a path ending in `src/tvbtoolkit`, your editable install is configured correctly.

If you already have a suitable Python environment and only want to install the toolbox:

```bash
cd /path/to/TVBToolkit
python -m pip install -e ".[dev,notebooks]"
```

Optional neuroimaging helpers used by some downstream PHIID/visualization scripts can be installed with:

```bash
python -m pip install -e ".[neuro]"
```

On the HPC, after transferring or cloning the repository:

```bash
cd /path/to/TVBToolkit
bash hpc/create_conda_env.sh
conda activate tvbtoolkit
bash hpc/smoke_test_login.sh
```

## First 10 Minutes (macOS + conda)

If you are setting up from scratch, these commands should get you running quickly.

```bash
# 1) Clone or move to the repository
git clone https://github.com/bmilinkovic/TVBToolkit.git
cd TVBToolkit

# 2) Create and activate the reproducible environment
conda env create -f environment.yml
conda activate tvbtoolkit

# 3) Register the environment as a Jupyter kernel (one-time)
python -m ipykernel install --user --name tvbtoolkit --display-name "Python (TVBToolkit)"

# 4) Launch Jupyter
jupyter lab
```

In Jupyter:

1. Open one of the notebooks in `notebooks/`.
2. Set kernel to **Python (TVBToolkit)**.
3. Run all cells from top to bottom.
4. Keep `FAST_MODE=True` for the first pass.

Quick sanity check in a Python cell:

```python
import tvbtoolkit
print(tvbtoolkit.__file__)
```

If that prints a path ending in `src/tvbtoolkit`, your environment is configured correctly.

## Quick Start

### 1) Whole-brain AdEx simulation

```python
from tvbtoolkit import WholeBrainConfig, run_whole_brain_simulation

cfg = WholeBrainConfig(
    model_family="adex_zerlaut",
    simulation_length_ms=1000.0,
    dt_ms=0.1,
)

result = run_whole_brain_simulation(cfg, seed=1)
print(result.raw.shape)         # (time, region)
print(result.region_labels[:5])
```

## Packaged Reference Assets

Small, non-sensitive reference assets are included for reusable whole-brain
simulation workflows:

- `data/receptors/hansen_receptors_aal90.csv`: AAL90 receptor/transporter
  density maps derived from the Hansen et al. PET atlas.
- `data/brain_act/source/atlases/`: AAL90 lookup tables defining the region
  labels/order used by the AAL90 workflows.
- `data/connectivity/average_aal90/`: a generic/average AAL90 structural
  connectome with `weights.txt`, `tract_lengths.txt`, `centres.txt`, and a
  TVB-style zip archive.

Sensitive participant data and generated outputs remain ignored and are not
tracked in Git.

### 2) Single-region AdEx simulation (Brian2)

```python
from tvbtoolkit import SingleRegionConfig, run_single_region_simulation

cfg = SingleRegionConfig(duration_ms=500.0)
out = run_single_region_simulation(cfg, seed_value=1)

print(out.exc_rate_hz.shape, out.inh_rate_hz.shape)
```

### Surface-based AdEx simulation

Surface workflows live under `tvbtoolkit.surface` and reuse the same
AdEx/Zerlaut parameter machinery as whole-brain runs. They require aligned TVB
surface, region-mapping, and connectivity assets; see
`docs/surface_adex_simulations.md`.

### 3) Complexity metrics

```python
import numpy as np
from tvbtoolkit.complexity.measures import lzc_multichannel, pci_casali_like

x = np.random.randn(2000, 32)
print("LZc:", lzc_multichannel(x))
print("PCI (Casali-like):", pci_casali_like(x, stimulation_index=800, t_analysis_ms=300.0, dt_ms=1.0))
```

## Notebook Workflows

Two notebooks are included:

- `notebooks/control_ketamine_psilocybin_lzc_pci.ipynb`
- `notebooks/legacy_parity_ketamine_psilocybin_sweeps.ipynb`
- `notebooks/brain_act_subject_loading_and_simulation.ipynb`

### Which notebook should I run first?

Start with:

- `control_ketamine_psilocybin_lzc_pci.ipynb` (faster; simpler condition batch)

Then run:

- `legacy_parity_ketamine_psilocybin_sweeps.ipynb` (heavier; broader parity sweeps)

## FAST_MODE vs FINAL mode

Both notebooks now include:

```python
FAST_MODE = True
```

- `FAST_MODE=True`: quicker iteration (fewer seeds/stim points, shorter runtime, reduced I/O)
- `FAST_MODE=False`: full-quality outputs for final figures

Recommended workflow:

1. Develop and debug with `FAST_MODE=True`.
2. Switch to `FAST_MODE=False` once everything is stable.
3. Restart the kernel before final runs.

## Output Structure

Results are written under each notebook's output root, for example:

```text
outputs/
  control_ketamine_psilocybin/
    simulations/
      <condition>/
        seed_XXX.npz
    metrics/
      <condition>_metrics.npz
    figures/
      *.png
```

This keeps raw time series, metric arrays, and figures clearly separated.

## Parallel Execution

Use `run_condition_batch(...)` with explicit parallel settings:

- `n_jobs`
- `use_processes=True`
- `show_progress=True`

Helper functions are provided:

- `detect_system_specs()`
- `recommend_parallel_workers(task="whole_brain_tvb")`

On laptops, a moderate worker count is typically faster than saturating all cores.

## Monitor Mode (Performance)

Whole-brain monitor mode is configurable via `WholeBrainConfig`:

- `monitor_mode='temporal_average'` for downsampled/averaged output (recommended for batch speed)
- `monitor_mode='raw'` for full-resolution raw monitor output
- `temporal_average_period_ms` controls effective sampling interval

Batch workflows default to temporal averaging unless you explicitly override:

```python
metrics = run_condition_batch(
    ...,
    monitor_mode_default='temporal_average',
    temporal_average_period_ms=1.0,
)
```

To force legacy raw output:

```python
base_cfg.monitor_mode = 'raw'
metrics = run_condition_batch(..., monitor_mode_default='raw')
```

## Strict-Parity Whole-Brain Model Notes

For whole-brain runs, strict-parity support includes:

- `zerlaut_order=1` (first-order mean-field)
- `zerlaut_order=2` (second-order mean-field, including covariance terms)
- `zerlaut_matteo` and `zerlaut_gk_gna` variants
- legacy-style `parameter_overrides`, including top-level groups such as `parameter_model`

## Brain-Act AAL90 Dataset Integration

`TVBToolkit` now includes a dedicated Brain-Act structural dataset pipeline for:

- AAL90 atlas lookup handling
- subject-level structural connectivity (`C`) and tract lengths (`L`)
- cohort-aware loading (`control`, `mcs`, `uws`)
- optional structural validation and pre-simulation cleaning

### 1) Mirror Brain-Act source data into this repository (including `raw/`)

```bash
cd /path/to/TVBToolkit
python scripts/sync_brain_act_source_data.py \
  --source /Users/borjan/code/Brain-Act/brain-act/data \
  --dest data/brain_act/source
```

### 2) Convert once to fast-loading bundles

```bash
cd /path/to/TVBToolkit
python scripts/convert_brain_act_dataset.py \
  --output-dir data/brain_act/converted \
  --overwrite
```

This writes:

- `data/brain_act/converted/atlas.npz`
- `data/brain_act/converted/subjects_control.npz`
- `data/brain_act/converted/subjects_mcs.npz`
- `data/brain_act/converted/subjects_uws.npz`
- `data/brain_act/converted/index.json`

By default, conversion reads from `data/brain_act/source` if `--source-root` is not provided.

### 3) Load atlas and subject structure

```python
from tvbtoolkit import load_aal90_atlas, list_subjects, load_subject_structural

dataset_root = "data/brain_act/converted"
atlas = load_aal90_atlas(dataset_root)
subjects_control = list_subjects(dataset_root, cohort="control")

C, L, atlas, meta = load_subject_structural(
    subject_id=subjects_control[0],
    cohort="control",
    dataset_root=dataset_root,
    validate=True,
    normalize="max",
)
```

### 4) Run with subject-specific `C` and `L`

```python
from tvbtoolkit import WholeBrainConfig, run_whole_brain_simulation

cfg = WholeBrainConfig(
    model_family="adex_zerlaut",
    zerlaut_order=1,
    simulation_length_ms=400.0,
    monitor_mode="temporal_average",
    temporal_average_period_ms=1.0,
    weights=C,
    tract_lengths=L,
)
sim = run_whole_brain_simulation(cfg, seed=0)
```

## Brain-Act Notes

- The Brain-Act source cohorts are `CNT`, `MCS`, `UWS`; in `TVBToolkit` they map to `control`, `mcs`, `uws`.
- ROI ordering is fixed by the atlas lookup used at conversion time.
- No implicit ROI remapping is applied. Matrix shape must match atlas size exactly.
- End-to-end example notebook: `notebooks/brain_act_subject_loading_and_simulation.ipynb`

## Troubleshooting

### `ModuleNotFoundError: No module named 'tvbtoolkit'`

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Then restart the notebook kernel.

### Override key errors (for example `parameter_model`)

This has been patched in the current codebase. If you still see old behaviour, restart the kernel so worker processes load updated source.

### Simulations are very slow

Use notebook `FAST_MODE=True` while iterating. The largest time costs are:

- many seeds/conditions/stimulation points
- long simulations with small `dt_ms`
- saving all raw time series for every job

## Repository Layout

```text
src/tvbtoolkit/
  core/
  datasets/
  whole_brain/
  single_region/
  complexity/
  workflows/
  visualization/
scripts/
notebooks/
docs/
data/
examples/
tests/
```

## Documentation

- Implementation/edit log: `docs/IMPLEMENTATION_LOG.md`
- Repo bootstrap notes: `docs/REPO_BOOTSTRAP_GUIDE.md`

## Citation and Versioning

If you publish with this toolbox, tag a release and cite that release commit/DOI to ensure full reproducibility.

## Brain-Act Subject-Specific Batch Workflow (AAL90)

For full subject-level runs with the same toolbox metric interface:

```python
from tvbtoolkit import BrainActSubjectConfig, run_subject_simulation, run_cohort_batch

cfg = BrainActSubjectConfig(
    dataset_root="data/brain_act/converted",
    output_root="notebooks/outputs/brain_act_subject_specific",
    seeds=(0, 1, 2),
    simulation_length_ms=5000.0,   # matches ketamine/psilocybin parity runs
    monitor_mode="temporal_average",
    temporal_average_period_ms=1.0,
    zerlaut_order=1,
)

# One subject
res = run_subject_simulation(subject_id="sub-01", cohort="control", cfg=cfg)

# Full cohort
res_control = run_cohort_batch(
    cohort="control",
    subjects=None,  # all subjects
    cfg=cfg,
    n_jobs=5,
    use_processes=True,
    show_progress=True,
)
```

This writes:

- `simulations/<cohort>/<subject>/seed_*.npz`
- `metrics/<cohort>/<subject>_metrics.npz`
- `metrics/<cohort>_cohort_metrics.npz`

### Brain-state (k-means) analysis

The subject workflow includes phase-pattern clustering summaries (Brain-Act-style brain states):

- state labels
- state occupancy
- transition matrices

via `tvbtoolkit.analysis` helpers.

## Brain-Act Damage-Mask Parity QC

To verify lesion/damage zero-mask behaviour against Brain-Act conventions:

```bash
python scripts/brain_act_damage_mask_qc.py \
  --dataset-root data/brain_act/converted \
  --output-root notebooks/outputs/brain_act_mask_qc
```

See:

- `docs/BRAIN_ACT_MASK_PARITY.md`
