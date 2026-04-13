# New DoC BOLD Brain-State Audit (TVBToolkit)

## Scope
This audit compares the new-data script
`/Users/borjan/CNRS/projects/TVBToolkit/scripts/brain_states_new_doc_bold.py`
against the reference Brain-Act logic in:

- `00_01_complete_data_setup.py`
- `01_01_load_and_convert_doc_data.py`
- `01_02_reorder_and_verify_bold_data.py`
- `03_02_calculate_functional_connectomes.py`
- `03_03_fc_vs_sc.py`
- `04_02_phase_coherence.py`
- `04_04_analyse_phase_states.py`
- `07_01_markov_states.py`
- `legacy_phase_coherence_new.py`
- `05_01_phase_coherence_visualisation.ipynb`
- `07_02_markov_plotting.ipynb`

The main goal was parity with Brain-Act state-analysis assumptions while preserving subject-wise comparability and subject-local `k=5` state fitting.

## Reference Pipeline Summary (Brain-Act)

## Data handling
- BOLD/SC are handled subject-wise.
- Brain-Act includes explicit BOLD ROI reordering to symmetric AAL90 (`01_02_reorder_and_verify_bold_data.py`) using index:
  - left: `0,2,4,...,88`
  - right reversed: `89,87,...,1`
- FC-SC comparisons are subject-level (subject FC against corresponding subject SC).

## Phase-state extraction
- Legacy path (not the lightweight default path):
  - z-score ROI signals over time
  - demean across ROIs per timepoint
  - Butterworth bandpass (`0.01-0.20 Hz`, order `3`, TR-aware)
  - Hilbert transform + phase unwrapping
  - phase-coherence features `cos(Δphase)`
- KMeans is run with fixed `k` for an analysis block; per-subject analyses are also used in Brain-Act scripts.

## State comparability
- Brain-Act code includes both pooled/group clustering and per-subject analyses.
- Label alignment steps (e.g., SFC-based sorting/relabeling) are post-hoc operations on already-fitted states.

## Markov / transitions
- Transition summaries include row-stochastic matrices.
- Legacy Markov script also computes no-self transitions and entropy metrics, often with run-collapsing.

## New Dataset Verification

## New data loaded (before exclusions)
From `data/doc_patients_new_data`:
- Total: `179` subjects
- Cohort counts:
  - `control=35`
  - `emcs=18`
  - `mcs=75`
  - `uws=51`
- All BOLD arrays standardized to subject x 90 ROI x 297 timepoints.
- Subject IDs are unique in the audited loader (`179/179` unique).

## Sedation structure (before exclusions)
- `control`: non-sedated `35`
- `emcs`: non-sedated `7`, sedated `11`
- `mcs`: non-sedated `29`, sedated `46`
- `uws`: non-sedated `29`, sedated `22`

## Non-finite BOLD subjects
Detected `5` subjects with non-finite BOLD values (NaNs) in raw data:
- MCS chronic non-sedated: `2`
- MCS chronic sedated: `1`
- UWS chronic non-sedated: `2`

Per user requirement, these are excluded (not imputed).
Final analyzed sample: `174` subjects.

## FC-SC ROI-order parity check
Static FC-SC coupling was explicitly tested under different ROI-order assumptions across all 179 loaded subjects:
- identity: mean `0.0768`
- reorder FC (AAL90 symmetric): mean `0.2794`
- reorder SC only: mean `0.0538`

Conclusion: BOLD/FC side required AAL90 symmetric reordering for subject-level FC-SC parity (`applied_mode=aal90_fc`).

## What Was Wrong in the Previous New Script

## Confirmed mismatches / bugs
1. **Wrong default state-extraction pipeline for parity**
   - Previous script defaulted to `pipeline="standard"` (`brain_states_new_doc_bold.py:935-939`).
   - Brain-Act parity requires legacy preprocessing (`brain_act_legacy` path).

2. **No explicit ROI reorder parity step before FC-SC coupling**
   - Previous script had no AAL90 symmetric reorder diagnostic or enforcement.
   - This strongly reduced coupling consistency and likely distorted SC-linked conclusions.

3. **Canonical template alignment as primary output path**
   - Previous script fit pooled templates and mapped occupancy/transitions to template states as the main outputs (`brain_states_new_doc_bold.py:726-902`).
   - This can impose discrete template bins on results and obscure subject-local variability.

4. **No explicit exclusion/QC for non-finite BOLD subjects**
   - Previous script did not isolate subjects with NaN BOLD, leading to unstable clustering behavior.

## Intended differences in previous script (not necessarily bugs)
- Inclusion of EMCS and sedation strata is expected for the new dataset.
- Subject-level extraction existed, but downstream reporting emphasized template-aligned outputs.

## Corrective Implementation
A new audited pipeline was created:
- `scripts/brain_states_new_doc_bold_audited.py`

## Key fixes made
1. **Subject-local states are primary**
   - `k=5` fitted per subject from that subject’s own BOLD data.

2. **Brain-Act legacy parity defaults**
   - Default `pipeline="brain_act_legacy"`
   - Default `clustering_backend="sklearn"`
   - Legacy TR/bandpass/filter parameters exposed and logged.

3. **Explicit ROI-order QC and enforced correction**
   - Static FC-SC coupling is tested across reorder hypotheses.
   - Chosen mode is logged (`logs/roi_reorder_decision.json`).
   - For this dataset, `aal90_fc` is selected.

4. **Non-finite BOLD exclusion**
   - Subjects with any non-finite BOLD values are excluded and listed in:
     - `tables/excluded_subjects_nonfinite_bold.csv`

5. **Post-hoc alignment is separate**
   - Primary reporting uses subject-local states with within-subject SFC-rank alignment only.
   - Optional canonical template fitting remains available only as explicit post-hoc mode (`--run-posthoc-template`).

6. **Subject-level SC-FC coupling preserved**
   - Static FC-SC is computed per subject using that subject’s FC and SC.
   - State-level SFC is computed per local state vs that same subject SC.

7. **Transition/Markov parity-style summaries added**
   - Rank-aligned transitions, no-self transitions, entropy rate outputs per subject.

## Output Structure (audited run)
`/Users/borjan/CNRS/projects/TVBToolkit/results/doc_patients_new_bold_brain_states_audited`

- `tables/`: subject-level and group-level CSVs
- `figures/`: publication-style PNG/SVG
- `npz/`: compact arrays for centroids/labels/transitions
- `logs/`: run metadata and ROI reorder decision

## Remaining Uncertainties / Decisions for PI Review
1. **Edge trimming parity choice (`trim_edge_samples`)**
   - Current default is `9` (matching recent Brain-Act phase scripts).
   - Some older legacy scripts ran without trimming (`0`).

2. **Template usage policy**
   - Post-hoc templates are optional and off by default.
   - If template-level reporting is desired for a manuscript panel, keep it clearly secondary to local-state results.

3. **Identity mapping across acute/chronic datasets**
   - Source files are treated as unique subjects as requested.
   - If longitudinal identity linkage exists externally, that can be integrated later for mixed-effects modeling.
