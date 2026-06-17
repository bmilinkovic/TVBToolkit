# Luppi 2022 Figure 1 to 4 Reproduction Plan for DoC AAL90 BOLD

This note turns the methodology in Luppi et al. (2022), "A synergistic core for human brain evolution and cognition", into a practical plan for the current disorders-of-consciousness AAL90 BOLD dataset in TVBToolkit.

## Bottom line

The core PhiID computation is already in place for the DoC dataset:

- subject-level reordered AAL90 BOLD export
- pairwise `PhiIDFull(..., 1, 'mmi')`
- per-subject `sts` and `rtr` atom matrices
- cohort- and condition-level averages and heatmaps

This is enough to reproduce the main matrix products behind Figure 1a,b on the DoC dataset and to extend them cohort-wise.

The rest of Figures 1 to 4 splits into three levels:

1. Ready now from current data:
   - group-average redundancy and synergy matrices
   - subject-level redundancy/synergy vs FC similarity
   - nodal redundancy-to-synergy rank gradient
   - edge-wise redundancy-to-synergy rank gradient
   - graph metrics on synergy and redundancy networks
   - structure-function similarity against subject structural connectivity
2. Ready after adding lightweight region annotations:
   - within- vs between-network summaries
   - RSN-pair edge-gradient matrices
   - Figure 2 style summaries by network and cytoarchitectonic class
3. Not directly reproducible without extra external resources or atlas harmonization:
   - NeuroSynth term meta-analysis in the exact paper style
   - exact Von Economo mapping used in the paper
   - HCP/Schaefer-232 specific comparisons
   - human-vs-macaque analyses

## What the paper actually does

### Figure 1

- Compute pairwise integrated information decomposition between all ROI pairs.
- Use MMI-PID for Gaussian data.
- Focus on persistent redundancy (`rtr`) and persistent synergy (`sts`).
- Average redundancy and synergy matrices across subjects.
- Compare each participant's redundancy and synergy matrices with traditional FC.
- Compute a nodal redundancy-to-synergy gradient:
  - nodal strength in synergy matrix
  - nodal strength in redundancy matrix
  - rank each separately
  - `synergy_rank - redundancy_rank`
- Relate that regional gradient to NeuroSynth cognitive-topic maps.

### Figure 2

- Summarize the nodal redundancy-to-synergy gradient by:
  - canonical resting-state networks
  - cytoarchitectonic classes

### Figure 3

- Compare redundancy and synergy networks as graphs.
- Show that synergy is more globally integrative.
- Show that redundancy is more segregated/modular.
- Test within-network vs between-network concentration.
- Build an edge-rank gradient matrix and summarize it by RSN pairs.

### Figure 4

- Compare redundancy and synergy with structural connectivity.
- Threshold synergy/redundancy to the same density as structural connectivity.
- Quantify structural-functional similarity with Spearman correlation on upper triangles.
- Test whether redundancy is stronger for directly connected pairs and synergy for indirectly connected pairs.

## Mapping this onto the current DoC dataset

### Dataset and atlas differences

The paper used a Schaefer-232 plus subcortical parcellation for the main human analysis. Our current dataset uses AAL90. That changes interpretation, but it does not block the core workflow.

Main implications:

- Matrix-level PhiID is still fully valid on AAL90.
- Structural connectivity comparison is easier because the DoC dataset already includes subject structural matrices in matching AAL90 order.
- Figure 2 and Figure 3 RSN analyses require an AAL90-to-network annotation table.
- The exact paper NeuroSynth and cytoarchitectonic overlays cannot be reproduced faithfully until we define the AAL90 mapping choices.

### What is already wired in this repo

- PhiID helpers:
  - `src/tvbtoolkit/analysis/phiid.py`
- Figure-style helper functions:
  - `src/tvbtoolkit/analysis/luppi2022.py`
- MATLAB runner with optional parallel `parfor`:
  - `scripts/phiid_empirical_bold_aal90.m`
- End-to-end empirical runner:
  - `scripts/run_empirical_bold_phiid.py`
- Core audit for old EEG-to-BOLD lineage:
  - `docs/phiid_empirical_bold_audit.md`
- Current starter notebook:
  - `notebooks/empirical_bold_phiid_aal90.ipynb`

## Dependency audit

### Already available locally

- PhiID MATLAB toolbox at `/Users/borjan/code/matlab/elph/PhiID`
- DoC empirical BOLD loader and AAL90 reorder audit
- subject structural connectivity in the same dataset

### Prepared in this repo

- bootstrap script for GitHub-based external resources:
  - `scripts/bootstrap_luppi2022_external.sh`

### External resources likely needed

- NeuroSynth meta-analysis helper:
  - `gpreti/GSP_StructuralDecouplingIndex`
- spin / surface permutation utilities if we want spatial nulls:
  - `frantisekvasa/rotate_parcellation`

### External resources not yet automated

- rsHRF toolbox
  - cited by the paper via NITRC rather than GitHub
  - needed only if we want the paper's HRF-deconvolved path rather than the simpler direct-BOLD version
- exact paper atlas-side annotations
  - AAL90 RSN assignments
  - AAL90 cytoarchitectonic class assignments

## Recommended staged plan

### Stage A: direct DoC reproduction with current data

Goal: reproduce the paper's core analysis logic on the current dataset before adding external overlays.

1. Run subject-level PhiID with `redundancy='mmi'`.
2. Save per-subject `sts` and `rtr` matrices.
3. Average by:
   - cohort
   - cohort + stage + sedation
4. Build subject-level FC matrices from the same BOLD data.
5. For each subject, compute:
   - `corr(FC, redundancy)`
   - `corr(FC, synergy)`
6. For each cohort/condition average, compute:
   - nodal rank gradient
   - edge-rank gradient
   - global efficiency for synergy and redundancy networks
   - modularity for synergy and redundancy networks
   - structural similarity using subject SC density matching

Deliverables:

- cohort-average redundancy and synergy matrices
- condition-average redundancy and synergy matrices
- FC-vs-synergy and FC-vs-redundancy summary plots
- nodal gradient vectors and heatmaps
- graph metric summary tables
- SC similarity summary tables

### Stage B: paper-style network summaries on AAL90

Goal: recover the Figure 2 and Figure 3 style summaries on the DoC dataset.

1. Create an AAL90 annotation table with:
   - ROI label
   - hemisphere
   - RSN assignment
   - cytoarchitectonic class
2. Use that table to compute:
   - Figure 2 style network-wise gradient summaries
   - within- vs between-RSN summaries
   - RSN-pair edge-rank gradient matrices

The scaffold for this is now ready in:

- `src/tvbtoolkit/analysis/luppi2022.py`

### Stage C: optional paper-closer extensions

Goal: move toward a closer methodological reproduction where scientifically justified.

Options:

- add rsHRF deconvolution before PhiID
- add NeuroSynth term meta-analysis
- add spatial null models
- harmonize AAL90 with external cortical annotation maps

## Important methodological decisions for our adaptation

### Redundancy function

Use `mmi` first.

This matches the paper's main Gaussian MMI-PID analysis and is the right first pass for the current BOLD workflow.

### HRF deconvolution

The paper's main human analysis used HRF-deconvolved BOLD. However, the paper also reports that the main synergy/redundancy identification is robust without deconvolution.

Recommendation for our dataset:

- first run without deconvolution to establish the DoC cohort effects
- add an rsHRF branch later only if we want a closer methodological parity analysis

### Atlas annotations

The biggest non-PhiID bottleneck is not computation but annotation:

- AAL90 RSN mapping
- AAL90 cytoarchitectonic mapping

Until those are explicitly defined, any Figure 2 or RSN-pair summary will be provisional.

## Parallelization and compute strategy

The current MATLAB runner is parallelizable over ROI rows via `parfor`.

Recommended strategy:

- wait for the other CPU-heavy analysis to finish
- then run `mmi` with a moderate worker count first, for example 4 workers
- if memory and MATLAB stability are good, increase worker count gradually

Why this is safe:

- each subject is independent
- within each subject, each ROI row is independent
- outputs are written one atom-file per subject after row completion

## Immediate next step

The most sensible next move is:

1. finish the `mmi` DoC PhiID generation
2. create the AAL90 annotation table
3. compute Figure 1, Figure 3, and Figure 4 style outputs first
4. add Figure 2-style summaries once the annotation table is filled
