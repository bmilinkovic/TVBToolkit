# Brain-Act Damage-Mask Parity (AAL90)

## Objective

Ensure the subject-specific pipeline in `TVBToolkit` reproduces the lesion/damage masking convention used in the original Brain-Act analyses.

## What the original Brain-Act code does

Inspection of the original repository scripts shows:

1. Structural-connectivity lesion edges are represented directly as exact zeros in subject SC matrices.
2. Lesion visualisation/statistics (`03_01_plot_lesion_heatmaps_improved.py`) operate on those loaded zeros (`A == 0.0`), without applying an additional synthetic mask transform.
3. Tract-length matrices are loaded from subject files and used as provided.

Therefore, the parity rule is:

- Preserve source SC/TL matrices as loaded (with diagonal handling), and do not invent a new lesion mask.

## TVBToolkit implementation

`src/tvbtoolkit/workflows/brain_act_subjects.py` applies the following:

- Diagonal values set to zero for SC and TL.
- Source zero-edge patterns are preserved.
- QC report metrics are computed on upper-triangle edges:
  - SC zero-edge fraction
  - TL zero-edge fraction
  - count of `SC==0` and `TL!=0` mismatches
- For patient cohorts (`mcs`, `uws`), optional robust fallback can enforce `TL=0` where `SC=0` if mismatches are detected.

## QC utility

Use:

```bash
python scripts/brain_act_damage_mask_qc.py \
  --dataset-root data/brain_act/converted \
  --output-root notebooks/outputs/brain_act_mask_qc
```

Outputs:

- `brain_act_damage_mask_per_subject.tsv`
- `brain_act_damage_mask_summary.json`
- `brain_act_damage_mask_qc.png`

## Expected pattern

In the current converted dataset:

- `mcs` and `uws` should exhibit strong SC/TL lesion-mask consistency (`SC==0 => TL==0`).
- `control` can contain benign SC-zero/TL-nonzero edges (non-lesion context), which are preserved for fidelity.
