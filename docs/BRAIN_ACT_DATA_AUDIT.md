# Brain-Act Structural Data Audit And Mapping

## Scope
Audit target: `/Users/borjan/code/Brain-Act/brain-act/data`

Focus:
- AAL90 atlas lookup tables
- Subject-level structural connectivity matrices
- Subject-level tract-length matrices
- Cohort and subject indexing metadata

## Observed Data Layout

### Atlas files
- `atlases/custom_lookuptable_AAL.txt`
- `atlases/symmetric_lookuptable.txt`

Both files contain one `Unknown` row (`index=0`) followed by 90 ROI entries.

### Structural and tract data
- `organized/structural_connectomes/{CNT,MCS,UWS}/sub-XX_structural_connectome.mat`
- `organized/tract_lengths/{CNT,MCS,UWS}/sub-XX_tract_lengths.txt`

### Metadata
- `organized/metadata/subjects_info.csv`
- `organized/metadata/file_mapping.csv`
- `organized/metadata/individual_file_mapping.csv`
- `organized/metadata/data_summary.csv`
- `organized/metadata/validation_report.csv`

## File Format Audit

### Structural connectivity `.mat`
Expected keys in per-subject files:
- `structural_connectome` (`90x90`, float64)
- `subject_id` (char/string)
- `condition` (char/string)

### Tract lengths `.txt`
- Dense numeric matrix (`90x90`, float)

### Integrity checks on source data
Across all cohorts:
- Shapes are consistent (`90x90`).
- Matrices are symmetric.
- Diagonals are zero.

Source cohort counts:
- `CNT`: 35 subjects
- `MCS`: 26 subjects
- `UWS`: 19 subjects

## Cohort Mapping

Toolbox canonical cohort labels:
- `CNT -> control`
- `MCS -> mcs`
- `UWS -> uws`

## Canonical Loader Data Model

Implemented in `tvbtoolkit.datasets.brain_act`:

- `AAL90Atlas`
  - `labels`
  - `region_codes`
  - `region_indices`
  - `ordering`
  - `source`
- `StructuralMetadata`
  - `subject_id`
  - `cohort` (canonical)
  - `source_cohort` (raw cohort code)
  - `dataset_index`
  - matrix shapes
  - optional `validation_report`

Subject payload:
- `connectivity` (`N x N`)
- `tract_lengths` (`N x N`)
- `cohort`
- checksums (index-level)

## Fast On-Disk Format

Converted dataset (output directory) contains:
- `atlas.npz`
- `subjects_control.npz`
- `subjects_mcs.npz`
- `subjects_uws.npz`
- `index.json`

### `index.json` includes
- format/version
- source root
- atlas metadata + checksum
- cohort index:
  - subject ids
  - cohort file name
  - cohort file checksum
  - matrix shape
- per-subject checksums for connectivity and tract lengths

## File-to-Loader Mapping

- `atlases/custom_lookuptable_AAL.txt`
  - Parsed by `load_aal90_atlas(...)` and conversion step.
- `organized/structural_connectomes/*/sub-*_structural_connectome.mat`
  - Loaded in conversion by `convert_brain_act_dataset(...)`.
- `organized/tract_lengths/*/sub-*_tract_lengths.txt`
  - Loaded in conversion by `convert_brain_act_dataset(...)`.
- Converted `subjects_*.npz`
  - Accessed by `list_subjects(...)` and `load_subject_structural(...)`.

## ROI Ordering Guarantee

ROI ordering is anchored to the chosen atlas lookup table at conversion time.
Every subject matrix is required to match atlas dimensionality exactly (`N x N`).
Loader validation enforces shape consistency between:
- atlas labels/order
- connectivity matrix
- tract-length matrix

No implicit ROI remapping is applied.

