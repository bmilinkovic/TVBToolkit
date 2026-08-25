# Subject-specific Momi graph-Laplacian AdEx/PCI protocol

## Scope

This is a two-subject robustness experiment, not a cohort comparison. It asks
whether a control and a UWS subject produce different perturbational responses
when diagnosis-level physiology is held fixed and only their native structural
connectome and matching tract lengths differ.

The prespecified subjects are `control:c0015` and `uws:u0020`, matching the
outcome-blind subjects already used in the repository's structural calibration
panel. The target is atlas-resolved `Supp_Motor_Area_L` (AAL90 zero-based index
9); no numeric target is assumed during loading.

## Laplacian sign and the released Momi implementation

The conventional graph Laplacian is

```text
D_ii = sum_j A_ij
L = D - A
L_normalized = L / ||L||_F
```

Thus `L` has a positive diagonal, negative off-diagonal edges, zero row sums,
and a non-negative spectrum. This is the matrix saved, hashed, tested, and
shown in project QC figures.

The Momi et al. paper's prose creates a sign ambiguity by referring to a
Laplacian while also describing a non-positive spectrum. Their released
PyTepFit implementation resolves how the operator enters the dynamics. It
computes positive delayed input `A*x_delayed` and adds the negative degree term
`-D*x_current`, i.e. it applies `-L`. Because TVB applies the supplied matrix
directly in its linear coupling, this experiment supplies `-L` to TVB while
retaining `D-A` as the mathematical Laplacian in all provenance.

The released code also applies `log1p` and fitted edge gains before its
normalization. Those operations are deliberately omitted here: the requested
test is a fixed-parameter transformation of native inverse-node-volume
connectomes, and fitting edge gains would destroy the structure-only contrast.
This divergence is explicit rather than silently described as an exact
reproduction of the fitted Jansen-Rit pipeline.

## Structural inputs and tract lengths

- Dataset scheme: `native_invnodevol`.
- Weight metric: MRtrix SIFT2 streamline weights scaled by inverse endpoint
  node volume.
- Subject/cohort rescaling before Laplacian: none.
- Damage mask: union of raw and inverse-node-volume zero masks, already applied
  to weights and matching tract lengths during dataset conversion.
- Matrix preparation: validated, symmetric, finite, non-negative adjacency with
  a zero diagonal.
- Laplacian scaling: per-subject Frobenius norm, following the paper-level
  description.
- Tract lengths: never multiplied into weights or the Laplacian.
- Delays: subject-specific `delay_ms = length_mm / 4 m/s`, as in the Momi
  separation of connection weights and length-derived delays.

For the selected pair, the control has 3963 non-zero undirected edges (1.05%
absent), while the UWS subject has 3390 (15.36% absent). Neither has an isolated
node. All non-zero weighted edges have non-zero matched tract lengths.

## Parameters held constant across subjects

| Component | Selection |
|---|---|
| Model | Second-order Zerlaut AdEx, split `gK/gNa` baseline implementation |
| Excitatory adaptation `b_e` | 10 pA for both subjects |
| Noise | private stochastic noise only; `noise_alpha=0` |
| Integration step | 0.1 ms |
| Rate monitor | exact 3 ms temporal average (333.33 Hz) |
| Transient | 4000 ms in the robustness run |
| Pre/post analysis window | 300 ms each |
| PCI response start | 8 ms after pulse onset |
| Target | left supplementary motor area |
| Pulse | square, 10 ms, 0.0003 kHz external input |
| Global coupling `G` | 3.5 for both subjects |
| Conduction speed | 4 m/s for both subjects |
| Trials | matched seeds 0-9 (10 trials per subject) |
| PCI-LZ | Baseline trial-bootstrap maximum-statistic null, 500 resamples, alpha 0.01; source entropy reported only |
| PCI-ST | `k=1.2`, minimum SNR 1.1, maximum variance 99%, 100 threshold steps |

`G=3.5` is a declared starting point rather than a fitted result. It preserves
approximately the off-diagonal scale of the existing native-weight reference
for the selected control: `0.0025 * ||D-A||_F = 3.52`. Both subjects receive
the same value; no per-subject recoupling is allowed. A `G` sensitivity grid is
required before interpreting cohort-scale effects.

## Outputs and execution

The runner saves each trial, the matched-trial mean 90-region time courses,
PCI-LZ/PCI-ST and response metrics, adjacency/Laplacian QC, evoked time-course
figures, and a manifest containing matrix hashes and every design selection.

```bash
python experiments/momi_laplacian_pci/run_two_subject_robustness.py --dry-run
python experiments/momi_laplacian_pci/run_two_subject_robustness.py --workers 8
```

On Slurm:

```bash
sbatch hpc/submit_momi_laplacian_pci_robustness.sh
```

The two-seed, 100-ms-transient smoke run is an implementation check only. Its
baseline has not settled sufficiently for scientific interpretation, and its
zero PCI-LZ values must not be reported as subject findings.

## First 10-trial robustness result

The full local run completed all 20 simulations with the parameters above.
Both subjects settled near the same mean baseline rate (control 3.086 Hz; UWS
3.089 Hz), supporting the intended fixed-physiology comparison. Peak left-SMA
responses were also similar (9.04 and 9.30 Hz above baseline, respectively).

PCI-ST was 95.39 for the control and 78.52 for UWS (17 and 15 retained
components). This is a directional separation consistent with the proposed
structural hypothesis, but it is one prespecified subject pair and is not an
inferential result.

PCI-LZ is calculated with the repository's simulation-validated significance
route: subtract each trial's prestimulus mean, scale by pooled prestimulus
variation, bootstrap prestimulus trials with replacement, and threshold the
trial-average response against the 99th percentile of the global
source-by-time null maximum. Pre/post whole-block exchange is not used for the
primary result; it remains available only through
`--pci-significance-method pre_post_swap` as a sensitivity analysis.

The PCI-LZ values below are regenerated from the cached trials whenever the
analysis configuration changes, without rerunning the AdEx simulations.
The earlier analysis forced both PCI-LZ values to zero when response entropy
was below 0.08. That rule is no longer applied to simulations. Entropy and
active fraction remain visible diagnostics; the calculated sparse-response
PCI values are retained rather than erased.

The spread heuristic called 10 control regions and 17 UWS regions active. The
larger UWS count despite more missing edges is a warning that per-subject
Frobenius normalization can compensate for lower total connection mass by
upscaling preserved edges. A follow-up sensitivity should compare (a) the
paper-style per-subject norm, (b) a common control-derived norm, and (c) native
adjacency at matched effective coupling before scaling to the full cohort.

## Three-subject operator and PCI-method comparison

The entropy cutoff was removed before this comparison. Entropy and significant
cell fraction are still saved, but never replace the calculated PCI with zero.
The third subject, `uws:u0001`, was selected before inspecting PCI because its
3.52% missing-edge fraction lies between control `c0015` (1.05%) and the more
damaged `uws:u0020` (15.36%). Ten distinct stochastic seeds (0--9) are matched
across every subject and operator.

| Operator | Subject | Current trial-average PCI-LZ | Legacy TVBSim PCI-LZ | PCI-ST |
|---|---|---:|---:|---:|
| Native A, G=0.0025 | c0015 | 0.5120 | 0.8162 | 84.96 |
| Native A, G=0.0025 | u0001 | 0.5120 | 0.8078 | 108.36 |
| Native A, G=0.0025 | u0020 | 0.5120 | 0.7615 | 71.62 |
| Normalized -L, G=3.5 | c0015 | 0.4661 | 0.8005 | 95.39 |
| Normalized -L, G=3.5 | u0001 | 0.4648 | 0.7858 | 95.82 |
| Normalized -L, G=3.5 | u0020 | 0.3986 | 0.7730 | 78.52 |

The current native PCI is identical because every subject produced the same
minimal significant matrix: only the stimulated left SMA was active, for the
same nine 3-ms bins from 9 through 33 ms (0.1031% of cells; entropy 0.01172).
It therefore contains no resolved structural ordering at
this stimulation/significance setting. The legacy TVBSim estimator does show a
decline, but it thresholds each noisy trial and averages ten separately
normalized PCI values, so it is not the empirical trial-average calculation.

With the Laplacian, the current method separates the high-damage subject
(`u0020`, 14.5% below control) while leaving the moderate-damage subject
essentially equal to control (0.27% below). PCI-ST independently shows the same
pattern for the Laplacian. This is a robustness result for three prespecified
subjects, not a cohort inference or a monotonic edge-loss law.

The two operator rows use their established numerical scales. Native `G=0.0025`
is the existing adjacency-model reference; Laplacian `G=3.5` compensates for
Frobenius normalization. Because `A` and `D-A` are different operators, these
values are not an exact operator-norm match. Absolute native-versus-Laplacian
PCI differences must therefore not be attributed solely to the transform;
the defensible comparison here is the subject contrast within each operator.

## Four-subject 5-HT2A occupancy run

The receptor-occupancy extension uses one structurally prespecified subject
from each clinical condition: control `c0015` (1.05% absent edges), EMCS
`e0003` (3.40%), MCS `m0075` (9.94%), and UWS `u0020` (15.36%). All settings
in the table above remain fixed, including `b_e=10` pA, `G=3.5`, individual
tract-length delays at 4 m/s, the left-SMA pulse, and matched stochastic seeds
0--9. Only 5-HT2A occupancy changes: 0, 0.25, 0.50, and 0.766. The Cimbi AAL90
receptor map modulates the established split `gK/gNa` conductance/leak route
toward `E_L,e=-61.2` mV and `E_L,i=-64.4` mV. No 0.08 entropy rule or other
response-amplitude floor is applied.

| Subject | Occupancy 0: current / legacy / PCI-ST | Occupancy 0.766: current / legacy / PCI-ST |
|---|---:|---:|
| CNT c0015 | 0.4661 / 0.8005 / 95.39 | 0.4928 / 0.8157 / 99.85 |
| EMCS e0003 | 0.3686 / 0.7929 / 100.99 | 0.3884 / 0.8191 / 102.09 |
| MCS m0075 | 0.3509 / 0.7731 / 60.95 | 0.3686 / 0.8035 / 59.58 |
| UWS u0020 | 0.3986 / 0.7730 / 78.52 | 0.4648 / 0.7943 / 96.22 |

Legacy TVBSim PCI-LZ increases at every tested subject's highest occupancy.
The current trial-average PCI-LZ also finishes above baseline in all four, but
is visibly quantized because its significant matrix is extremely sparse
(0.14--0.21% active cells). UWS shows the largest endpoint change: +16.6% for
current PCI-LZ and +22.5% for PCI-ST. MCS PCI-ST falls slightly (-2.2%), so the
result does not support a universal monotonic drug response across all
estimators. Nor does the four-case baseline establish a monotonic relation
between missing-edge fraction and PCI. These are construction/robustness
results that motivate a full-cohort, multi-seed sensitivity analysis.

The main butterfly plots show unthresholded change from each region's own
prestimulus mean. Statistical baseline thresholds appear only in the separate
PCI audit output; they are not drawn on, or used to censor, the propagation
figures.

## Primary sources

- Momi D, Wang Z, Griffiths JD (2023), *TMS-evoked responses are driven by
  recurrent large-scale network dynamics*, eLife 12:e83232,
  https://doi.org/10.7554/eLife.83232
- Authors' released implementation, https://github.com/GriffithsLab/PyTepFit
