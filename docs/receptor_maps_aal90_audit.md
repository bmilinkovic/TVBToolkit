# AAL90 receptor audit — 2026-10-05

## Current simulation finding

`hpc/submit_first_order_gnak_tms50.sh` invokes
`experiments/first_order_gnak/simulate.py`. Its configuration calls
`tvbtoolkit.workflows.pharmacology.get_5ht2a_aal90(target_labels=labels)`.
The default selects `5HT2a_cimbi_hc29_beliveau` by name from
`data/receptors/hansen_receptors_aal90.csv`, then correctly joins by AAL label.
The simulation min-max scales this vector and uses
`g_K = g_K_baseline - occupancy * rho * (g_K_baseline - g_K_drug)`.

**The former source table had a one-region displacement (now corrected).** The saved
116-value PET extraction already excludes background. The table matches
entries `[1:91] / max(entries[1:91])`, rather than the correct first 90 entries.
Thus the first row labelled Precentral_L carries Precentral_R's value; the
last row includes the next, cerebellar parcel. The code's label join cannot
repair values that were mislabelled earlier.

Re-extracting Cimbi directly from its original PET NIfTI with the local AAL
atlas and the original NiftiLabelsMasker settings reproduced the saved raw
extraction exactly (maximum absolute error zero). The XML label sequence
matches the production CSV's labels for the first 90 regions. Corrected and
old regional Cimbi values correlate at 0.5626328928. Runtime loader tests
confirmed named-column selection and label permutation.

This is separate from the incorrect tracer-column assignments in the old
Brain-Act 19-column plotting script. Production does not load that 19-column
matrix, but its 37-column CSV shares the one-region displacement.

The production 37-tracer table has now been corrected, and the unscaled
90-region data are also tracked in Git. Both 5-HT2A APIs share the checked
loader. Trial cache signatures and SBI provenance include the receptor CSV.
No model equations, running HPC job, or existing simulation result was changed.
The remote cluster file and running process were not inspected. If the cluster
uses this same input, nonzero-occupancy runs require rerunning after a versioned
input correction. Zero-occupancy runs are unaffected by this receptor error.
Do not replace inputs during an ongoing job or mix corrected and old trials.

## New figures and numerical data

Run `python scripts/audit_receptor_maps_aal90.py` in the repository with the
local scientific Python environment. Source/atlas/output locations are CLI
arguments. Outputs are separate in `results/receptor_maps_aal90_audit/`.
The script now compares with the corrected production table by default;
pass `--comparison-csv` with an archived old table to reproduce the historical
alignment comparison (old tables can be retrieved from Git history).

- Corrected 90 × 19 heatmap: PNG and editable SVG.
- Brain panels: three axial slices per map (MNI z = -12, 6, 30 mm), including
  subcortical structures; these are not cortical surface renderings.
- Cimbi old/new alignment comparison.
- Corrected raw and max-scaled 37-tracer tables, unscaled 19-map composites,
  0–1 19-map table, and JSON provenance with source hashes and selections.

Selection is by explicit tracer name, never column position. The named
selection follows the original Hansen `make_receptor_matrix.py` for direct
maps and the repeated-study groups. Repeated studies are individually
z-scored over the 90 regions, then averaged with sample-size weights. VAChT
uses the available studies n=4,5,18; the Spreng n=3 AAL extraction is absent.
This is an explicitly documented three-study variant, not an exact copy of
the original four-study implementation.

For each final map, `rho = (x - min(x)) / (max(x) - min(x))`. All 19 maps are
checked finite, nonconstant, with minimum zero and maximum one. AAL rows stay
in alternating left/right order; there is no false hemisphere divider at 45.

These are **within-map relative weights**, not absolute receptor densities.
Negative z scores mean below-average values; some individual PET-derived maps
also contain negative estimates before combining studies. Min-max scaling
does not cure uncertainty or bias in PET quantification. Zero denotes the
regional minimum, not demonstrated receptor absence. Scaling preserves ranks
and relative differences within each map, not ratios or cross-receptor
absolute density comparisons. It is the same scaling convention as the
current gNa/gK simulation, without its negligible denominator epsilon.
