# PCI-LZ method comparison and simulation adaptation

## Scope

This document separates EEG acquisition, PCI preprocessing and the final
complexity calculation. It also distinguishes literal empirical replication
from a justified adaptation for regional firing-rate simulations.

## Side-by-side comparison

| Choice | Casali et al. 2013 | Casarotto et al. 2016 | Farnes et al. 2020 | Current serotonergic simulation | Proposed next simulation |
|---|---|---|---|---|---|
| Perturbation | Single-pulse TMS | Single-pulse TMS at multiple viable sites | Single-pulse TMS over right BA7 | 10-ms direct input to left `Supp_Motor_Area_L` | Decide whether the model pulse should end by 8 ms |
| Repetitions | Figure example averaged 150 trials | Sessions with fewer than 80 good trials were excluded | Approximately 300 pulses per condition in the displayed example | 100 independent simulated trials | 100 trials, subject to convergence analysis |
| Measured signal | 60-channel TMS-compatible EEG followed by cortical source reconstruction | Same PCI source-reconstruction lineage | 60-channel EEG followed by MNI source reconstruction | Ninety AAL excitatory firing-rate signals | Same AAL signals unless an EEG observation model is added |
| Raw acquisition | TMS-compatible EEG commonly recorded at 1,450 Hz in this lineage | Hardware acquisition precedes standardized preprocessing | 5,000 Hz | AdEx integration step 0.1 ms | AdEx integration step 0.1 ms |
| PCI sampling rate | 362.5 Hz in the documented implementation lineage | 362.5 Hz | 312.5 Hz | Nominally 128 Hz (`dt=7.8125` ms) | Requested 340 Hz; model-compatible recommendation 333.33 Hz (`dt=3.0` ms) |
| Filter before PCI | 0.1--45 Hz empirical TMS-EEG pipeline | 0.1--45 Hz | 1--45 Hz | No EEG forward model or equivalent band-pass | Must be described as regional-rate PCI, not EEG PCI |
| Alignment | TMS-locked trial average | TMS-locked trial average | TMS-locked trial average | Each simulation is cut around its own recorded onset and aligned before averaging | Same |
| Significance | Nonparametric bootstrap at cortical-source level, `alpha=0.01` | Procedure stated to comply with Casali 2013 | 99th percentile of maximum amplitudes from bootstrap-resampled baseline activity | Corrected result table used baseline trial bootstrap, 500 resamples, `alpha=0.01`; repository production default currently remains pre/post swap, 1,000 permutations, `alpha=0.05` | Baseline-bootstrap primary; pre/post swap only as sensitivity analysis |
| Response used for LZ | First 300 ms; the accessible main paper does not state the precise initial artifact boundary | Same validated PCI pipeline | 8--300 ms | Requested 8--300 ms begins at approximately 15.6 ms on the nominal 128-Hz grid | Requested 8--300 ms becomes 9--297 ms at 333.33 Hz |
| Active-matrix floor | Source entropy normalization; empirical software lineage uses an activity/entropy floor | PCI set to zero when significant activity is below 1% | Entropy must exceed 0.08 | Current code can apply entropy floor 0.08 | Retain and report both entropy and pre-floor PCI |
| Complexity | Sort sources, LZ76 compression, entropy normalization | Same | Same scripts supplied by the PCI developers | Same structural calculation on AAL regions | Same, explicitly called model PCI-LZ |

The large differences in raw acquisition rates are not differences in the
sampling used by Lempel--Ziv PCI. Casali/Casarotto used 362.5 Hz for the PCI
matrix and Farnes used 312.5 Hz. A proposed 340-Hz rate lies almost exactly
between them, but TVB's temporal-average monitor requires an integer number of
0.1-ms integration steps. Requesting a 2.9412-ms period is rounded internally
to 29 steps, producing 2.9 ms or 344.83 Hz rather than 340 Hz. A 3-ms period
gives exactly 333.33 Hz, lies between the empirical rates, and produces 97
samples from the first included sample at 9 ms through 297 ms. It is the cleaner
production setting. Alternatively, a 2.9-ms period gives 344.83 Hz and 100
response samples.

## What the 8-ms boundary means

Farnes removed the physical TMS artifact from -2 to +5 ms and then calculated
PCI from 8 to 300 ms. In an EEG experiment, this avoids giving the electromagnetic
discharge and amplifier artifact to the complexity algorithm.

The simulation has no amplifier artifact, but it should still avoid treating
the imposed input as internally generated propagation. With the recommended
333.33-Hz monitor, an 8-ms requested boundary advances to the first available
sample at 9 ms. The configured model stimulus currently lasts 10 ms. Thus the
proposed PCI window would begin 1 ms before that stimulus ends. There are three coherent
protocols:

1. Keep the 10-ms stimulus and begin PCI at the first sample after it ends
   (12 ms at 333.33 Hz).
2. Keep the empirical 8-ms PCI boundary and shorten the model input so that it
   ends no later than 8 ms.
3. Keep both 10 ms and 8 ms, but state explicitly that the first PCI sample
   overlaps the imposed input and show a post-input sensitivity analysis.

Option 2 gives the cleanest empirical-style timing. Option 1 gives the cleanest
interpretation of the existing model perturbation. Option 3 is usable as a
sensitivity analysis but is not the preferred primary definition.

## Step-by-step simulation-adapted PCI-LZ

For each subject, condition and receptor occupancy:

1. **Generate repeated perturbations.** Run the same model configuration 100
   times with independent noise realizations and a recorded perturbation onset.
2. **Cut each trial.** Extract 300 ms before and 300 ms after that trial's own
   perturbation onset.
3. **Time-lock.** Put the perturbation onset at the same matrix column in every
   trial. Do not average trials before this alignment.
4. **Remove the baseline offset.** For every trial and region, subtract that
   trial's mean prestimulus firing rate.
5. **Express activity relative to baseline variability.** Divide each region by
   its pooled prestimulus standard deviation. This prevents high-variance
   regions from determining the threshold for every other region.
6. **Average trials.** Average the standardized, aligned trials to obtain one
   deterministic region-by-time response.
7. **Construct the null distribution.** Resample prestimulus trials with
   replacement. Average each resample and retain the largest absolute baseline
   value across all regions and baseline times.
8. **Choose the threshold.** Use the 99th percentile of these bootstrap maxima
   (`alpha=0.01`). This is a family-wise maximum-statistic threshold.
9. **Binarize.** Set a region-time cell to 1 when the absolute trial-averaged
   response exceeds the threshold and to 0 otherwise.
10. **Select the response interval.** Retain the matrix from the first available
    sample at or after 8 ms until 300 ms.
11. **Sort regions.** Sort matrix rows in descending order of the number of
    active samples. This makes the Lempel--Ziv scan deterministic and follows
    the empirical PCI construction.
12. **Calculate source entropy.** If `p1` is the fraction of active cells,
    calculate `H = -p1 log2(p1) - (1-p1) log2(1-p1)`.
13. **Compress the matrix.** Apply two-dimensional LZ76 parsing to count new
    spatiotemporal binary patterns, producing `cL`.
14. **Normalize.** For `L = n_regions * n_times`, calculate
    `PCI = cL * log2(L) / (L * H)`.
15. **Quality control.** Save the threshold, active fraction, entropy, raw LZ
    count, normalization factor, effective sample boundaries and final PCI.
    If the entropy floor is used, also save the PCI before that floor is applied.

## Minimal explanatory implementation

Run:

```bash
python scripts/plot_pci_method_walkthrough.py
```

The script deliberately uses a synthetic response so it is a methods
illustration rather than a result. It generates separate alignment,
thresholding and complexity figures plus a six-panel composite. Its summary
JSON records the requested and effective timing and warns when the PCI window
starts before the configured model input has ended.

## Primary sources

- Casali AG et al. (2013), *A theoretically based index of consciousness
  independent of sensory processing and behavior*.
  https://doi.org/10.1126/scitranslmed.3006294
- Casarotto S et al. (2016), *Stratification of unresponsive patients by an
  independently validated index of brain complexity*.
  https://doi.org/10.1002/ana.24779
- Farnes N et al. (2020), *Increased signal diversity/complexity of spontaneous
  EEG, but not evoked EEG responses, in ketamine-induced psychedelic state in
  humans*. https://doi.org/10.1371/journal.pone.0242056
