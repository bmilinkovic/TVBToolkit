# CNRS External Data Layout

This project keeps bulky raw data, generated results, legacy outputs, and
temporary artifacts outside the repository under:

```text
/Volumes/ex_data/cnrs
```

Override this root with `CNRS_DATA_ROOT` on another machine or cluster.

## Dataset Roots

- Drugs Maastricht: `/Volumes/ex_data/cnrs/data_drugs_maastricht`
- DOC Liege: `/Volumes/ex_data/cnrs/data_doc_liege`
- Stimulation Liege: `/Volumes/ex_data/cnrs/data_stimulation_liege`
- Unclear or mixed provenance: `/Volumes/ex_data/cnrs/legacy_unsorted`

Each dataset root uses:

```text
raw/
results/
legacy/
tmp/
```

## Migration Classification

- `data/drugs_data` -> `data_drugs_maastricht/raw/drugs_data`
- `data/doc_data` -> `data_doc_liege/raw/doc_data`
- `data/brain_act` -> `data_doc_liege/raw/brain_act`
- `data/stim_data` -> `data_stimulation_liege/raw/stim_data`
- Maas drug result folders -> `data_drugs_maastricht/results`
- DOC/PhiID/Brain-Act result folders -> `data_doc_liege/results`
- tDCS/TMS-EEG stimulation result folders -> `data_stimulation_liege/results`
- Generic two-node, three-node, bivariate, nonlinear, and calibration sweeps
  -> `legacy_unsorted/results`
- Third-party external checkouts, old local project folders, notebook scratch
  data, and copied local caches -> `legacy_unsorted/legacy` or
  `legacy_unsorted/tmp`

`legacy_unsorted` is used where an output is generic model exploration or its
dataset identity is mixed/unclear, so it is not silently assigned to one of the
three named datasets.

## Code Access

Use `tvbtoolkit.core.paths` for new code. For example:

```python
from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results

data_root = doc_liege_raw("doc_data")
output_root = doc_liege_results("phiid_empirical_bold")
```
