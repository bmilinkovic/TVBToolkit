# Empirical BOLD PhiID Audit

This note records the lineage of the old PhiID workflow and the corresponding TVBToolkit plan for the current AAL90 empirical BOLD dataset.

## Legacy lineage

### 1. Where the saved figures in point 3 came from

- Figure notebook:
  - `/Users/borjan/code/python/AnesthesiaProjectEmergence/phiid_plot.ipynb`
- Saved figures directory:
  - `/Users/borjan/code/python/AnesthesiaProjectEmergence/results/phiid/used-figures`

That notebook points to raw per-subject PhiID matrices here:

- `/Users/borjan/code/python/TVBEmergence/results/phiid/Idep_xtb/`

Its plotting logic is simple and specific:

- collect files whose names contain `sts` or `rtr`
- separate them by condition
- average matrices per participant
- then average across participants
- draw heatmaps for the averaged synergy (`sts`) and redundancy (`rtr`) matrices

### 2. Where those raw PhiID matrices were generated

- Legacy MATLAB generator:
  - `/Users/borjan/code/python/TVBEmergence/test/matlab/emergence_measures.m`

Core historical loop:

- load one subject timeseries
- reshape it into a 2D matrix with shape `regions x time`
- for every ordered ROI pair `(row1, row2)` with `row1 ~= row2`
- call:
  - `PhiIDFull([time_series(row1,:); time_series(row2,:)], 1, 'idep_xtb')`
- write one output file per atom, e.g.:
  - `<subject>_sts_mat_Idep_xtb.mat`
  - `<subject>_rtr_mat_Idep_xtb.mat`
  - plus the other 14 atoms and the `sr_gradient`

Important historical details:

- the old EEG script truncates to `20000` timepoints because the source-reconstructed EEG files varied in length
- the plotting notebook used only `sts` and `rtr`, even though all 16 atoms were saved

## Current TVBToolkit BOLD source

The present empirical BOLD source in this repo is the new DoC AAL90 dataset loader used by:

- `/Users/borjan/CNRS/projects/TVBToolkit/scripts/brain_states_new_doc_bold_audited.py`

Relevant facts from that audited loader:

- subject timeseries are loaded from the raw `DoC_*.mat` FC files
- each subject is reshaped to `(time, 90)`
- the correct reorder mode for this dataset is `aal90_fc`
  - FC/BOLD starts in interleaved AAL90 order
  - SC is already in symmetric left-then-right-reverse order
- after reordering, each subject has a 90-region BOLD matrix ready for pairwise ROI analysis

## TVBToolkit implementation plan

The repo-local workflow added for this project is:

1. Python notebook exports each subject’s reordered BOLD timeseries to MATLAB input files.
2. MATLAB batch runner loops over all ROI pairs and calls `PhiIDFull(..., 1, 'idep_xtb')`.
3. One file per atom is saved per subject, mirroring the historical workflow.
4. Python reloads those outputs and averages `sts` and `rtr` by cohort, or by any grouping columns you choose.

Added files in this repo:

- Python helpers:
  - `src/tvbtoolkit/analysis/phiid.py`
- MATLAB runner:
  - `scripts/phiid_empirical_bold_aal90.m`
- Notebook:
  - `notebooks/empirical_bold_phiid_aal90.ipynb`

## Key methodological choice

For the new DoC BOLD data there is no need to reproduce the old EEG-specific `1:20000` truncation unless you want forced parity with an EEG-length window.

Default TVBToolkit plan:

- keep the full matched BOLD length per subject
- export the reordered `90 x T` matrix directly
- compute all 16 PhiID atoms
- use `sts` and `rtr` as the primary cohort-average matrices, because that is what the old figure workflow actually visualized
