# Average AAL90 Structural Connectome

This folder contains a reusable AAL90 structural connectome for whole-brain TVB
simulations when no subject-specific structural connectome is available.

## Files

- `weights.txt`: 90 x 90 symmetric adjacency/connection-weight matrix.
- `tract_lengths.txt`: 90 x 90 symmetric tract-length matrix.
- `centres.txt`: AAL90 region labels and MNI-like coordinates.
- `connectivity_average_aal90.zip`: TVB-style connectivity archive containing
  the three files above.

## Provenance

The weights and tract lengths were ported from the legacy TVBSim reference data:

`TVBSim/tvbsim/TVB/tvb_model_reference/data/connectivity/AAL90/`

The original source archive was named `connectivity_AAL90.zip` and contained
`weights.txt`, `tract_lengths.txt`, and `centers.txt`.

The legacy `centers.txt` coordinates were preserved, but region labels were
regenerated from `data/brain_act/source/atlases/custom_lookuptable_AAL.txt`
because several labels in the legacy centers file were truncated. Matrix values
were not modified.

## Caveat

This should be treated as a generic/average AAL90 structural scaffold for
simulation examples and shared toolbox workflows. For patient-specific or
cohort-specific analyses, use the subject-specific structural connectomes
instead.
