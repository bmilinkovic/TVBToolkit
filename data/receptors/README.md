# Receptor Density Maps

This folder contains public/reference receptor-density maps used by TVBToolkit
for receptor-informed whole-brain simulations.

## `hansen_receptors_aal90.csv`

AAL90-parcellated neurotransmitter receptor/transporter maps derived from the
Hansen et al. PET receptor atlas.

- Rows: 90 AAL regions, cerebellum excluded.
- Columns: 37 PET tracer maps.
- Values: max-scaled receptor density values per tracer map.
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
