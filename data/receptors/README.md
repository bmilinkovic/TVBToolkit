# Receptor Density Maps

This folder contains public/reference receptor-density maps used by TVBToolkit
for receptor-informed whole-brain simulations.

## `hansen_receptors_aal90.csv`

AAL90-parcellated neurotransmitter receptor/transporter maps derived from the
Hansen et al. PET receptor atlas.

- Rows: 90 AAL regions, cerebellum excluded.
- Columns: 37 PET tracer maps.
- Values: divided by the maximum of each tracer map. Some PET estimates remain
  negative; this is not a guarantee of the interval [0, 1]. The current gNa/gK
  simulation applies min–max scaling to its selected 5-HT2A map afterward.
- Region order: AAL90 order used by the Brain-Act/TVBToolkit AAL workflows,
  beginning with `Precentral_L` and ending with `Temporal_Inf_R`.

Primary citation:

Hansen JY et al. (2022). Mapping neurotransmitter systems to the structural and
functional organization of the human neocortex. Nature Neuroscience, 25,
1569-1581.

Useful API:

```python
from tvbtoolkit.brian_mf.receptors import get_hansen_receptors_aal90, get_5ht2a_aal90

receptors = get_hansen_receptors_aal90()
ht2a = get_5ht2a_aal90(tracer="cimbi")
```

## Regional alignment correction (2026-10-05)

The former table mistakenly used entries `[1:91]` of a 116-region extraction
that already excluded background. This shifted every tracer by one region.
The current table uses `[:90]`, with names verified against the AAL XML.
`hansen_receptors_aal90_raw.csv` preserves those corrected, unscaled PET
values; dividing its columns by their maxima reproduces the production table.
Cimbi extraction was independently repeated from the source PET NIfTI and
matched exactly. See `docs/receptor_maps_aal90_audit.md`.

The old table SHA256 was
`552f017bf97315c55589a090e8bedaa21cdc43e5aebaea4f43ea70421c415364`.
The 5-HT2A loader rejects that known file even via an explicit override.
Use a new output directory after updating the cluster; never mix old and
corrected receptor trials. Zero-occupancy dynamics are unaffected by this
particular error. Legacy DK68/AAL116 approximation APIs are separate and
have not been changed into AAL90 APIs.
