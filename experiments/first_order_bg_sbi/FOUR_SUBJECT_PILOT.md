# Four-subject 40,000-simulation pilot

**Use [CLUSTER_RUN.md](CLUSTER_RUN.md) for deployment.** Existing local pilot
manifests are superseded. Transfer only their four exported empirical NPZ files
and subject list; run `prepare_cluster.py` after `git pull` to fingerprint the
actual committed cluster code. This avoids unrelated local research changes.

| Condition | Subject | Stage | Sedation | BOLD volumes | Prior draws |
|---|---|---|---|---:|---:|
| Wake/control | c0005 | control | none | 297 | 10,000 |
| EMCS | e0003 | chronic | none | 297 | 10,000 |
| MCS | m0011 | chronic | none | 297 | 10,000 |
| UWS | u0033 | chronic | none | 297 | 10,000 |

Control/UWS retain earlier example IDs. EMCS/MCS use the first eligible chronic,
non-sedated subject in the source map, not the earlier sedated examples. This is
a computational pilot, not a representative four-subject population estimate.

## Locked design

- First-order stochastic gNa/gK, receptor occupancy zero, no TMS.
- b uniform 0–80 pA; G uniform 0.025–0.075; tau_w=500 ms; speed=4 m/s.
- G interval comes from `docs/first_order_gNa_gK_HPC_and_G_sweep_plan.txt`:
  earlier grid 0.025, 0.0375, 0.05, 0.0625, 0.075. It is a candidate prior,
  not established physiological bounds. Broaden later if supported by diagnostics.
- Personal inverse-volume SC divided by one shared control maximum; personal delays.
- Confirmed TR=2.4 s. **120 s retained + 60 s warm-up = 180 s per draw**.
  Fifty BOLD samples enter each simulated summary. All 297 empirical volumes
  are covered by six 50-volume segments, including an end-anchored segment.
  Segment-conditioned posteriors and their coverage-weighted mixture are saved;
  the mixture is not a joint all-segment likelihood posterior. Simulated time
  series are NOT saved; only features, diagnostics, parameters and seeds.
- Native TVB BOLD monitor with double-gamma convolution; 4-ms internal stock,
  explicit SPM-style shapes 6/16, unit scale, 1/6 undershoot, 32-s kernel support.
  This is not the mathematical digamma function. TVB's default MixtureOfGammas
  parameters differ, so all parameters are supplied explicitly.
- HRF source: [SPM implementation](https://github.com/spm/spm/blob/main/spm_hrf.m).
  A plausible conventional observation model, not proof it is optimal in DoC.
- **119 features:** FC distributions, regional FC means, ACF, four spectral
  summaries and nine explicit short-recording FCD statistics. **No state
  occupancy or state SC–FC coupling features.**

## Data provenance

The matched raw BOLD MAT files contain 297×90 time series but no acquisition TR or
filtering history in the inspected variables/attributes. The earlier audited
analysis log at
`results/doc_patients_new_bold_brain_states_audited/legacy_style_repro/logs/run_metadata.json`
on the external disk explicitly says **2.4 s**, with analysis filtering 0.01–0.2 Hz.
The user subsequently confirmed **TR=2.4 s**; it is no longer provisional.
Upstream filtering remains undocumented in the inspected MAT metadata.

Current banks use confirmed TR=2.4 s and the common 0.01–0.10 Hz feature filter.
The prior TR-confirmation environment gate was removed; the launcher instead
checks that the updated, state-free pilot manifest is supplied.

Subject matching uses condition, stage, sedation and original within-file index
from `source_subject_map.csv`. Counts are checked against the exact inverse-volume
SC file. The supplied matched-file ordering is relied on where independent names
are absent. `input_audit.json` records this limitation rather than claiming that
names were verified for every participant. AAL labels explicitly map interleaved
BOLD order to the SC atlas.

## State features excluded from fitting

At the user's request, state occupancy and SC–FC coupling are excluded from the
single-subject fit. The existing cohort-pooled state analysis can be applied
downstream as a separate group validation, but it will require new recordings
because raw simulated BOLD is not saved. The optional utility is disabled in all four
new manifests, which contain no state template. This does not remove ordinary
static FC or FCD features. Degenerate b/G values remain acceptable; inspect joint
posteriors and posterior prediction, not just separate medians.

## Why not one minute with these features?

At TR=2.4 s, 60 s is only 25 volumes. A 60-s FCD window then occupies the entire
record, so there is no sequence of windows for the current FCD distribution.
A 0.01-Hz cycle lasts 100 s; one minute is less than one cycle. Filtering does
not create the missing information. FC is calculable but very noisy and its
sample matrix has rank at most 24 before allowing for temporal autocorrelation.
That rank does not prohibit FC-summary fitting, but cautions against treating
the estimate as a stable 90-region connectivity representation.

The user selected **two minutes**, allowing FCD calculation. The 25-volume
windows slide by one volume over 50 volumes, giving 26 windows and 325 distinct
off-diagonal FCD similarities. These overlapping pairs are NOT independent
observations. The distribution is summarized by mean, SD, skewness, excess
kurtosis and five quantiles. Only the diagonal is excluded: the old k=25 rule
would leave one pair, insufficient for these summaries.

Two short-recording implementation issues were caught before simulation:
VBI 0.4.3's FCD length check mixes seconds and samples; the legacy BOLD filter
can transpose data when timepoints < regions. The modular adapter explicitly
converts seconds to samples and filters on time axis 0, without changing library
or production code. Regression tests enforce 50×90 shape and correct window-pair
calculations. An intermediate preparation with misoriented features is superseded
and cannot pass current source-hash checks; it was never used for simulations.

Matched-length empirical/simulated summaries avoid comparing different
finite-sample distributions. Two minutes is still only 1.2 cycles at 0.01 Hz;
computability does not establish reliability. Use the saved validation segments
and posterior prediction to assess robustness. More training simulations do not
remove finite-recording uncertainty.

## Cluster submission

Transfer the repository, the complete pilot folder (including manifests), and the
shared-max-normalised structural dataset. Do not transfer or hardcode Mac paths.
The script verifies content hashes of the subject SC/tracts at the supplied root.

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/cluster/path/converted_structural_invnodevol_control_max
export BG_SBI_PILOT=/cluster/path/first_order_bg_sbi_four_2min
cd "$TVB_REPO"
mkdir -p hpc/logs
# Recommended timing/QC preflight: 16 draws per subject, retained on resume.
BG_SBI_STOP=16 sbatch hpc/submit_first_order_bg_sbi_four.sh
# After inspecting that benchmark, extend to 10,000 per subject:
sbatch hpc/submit_first_order_bg_sbi_four.sh
# Separate inference submission after simulations and QC finish:
BG_STAGE=train sbatch hpc/submit_first_order_bg_sbi_four.sh
```

The array has four subject tasks, limited to one running 48-CPU node by default,
matching the previous per-job resource request. Each subject uses 48 simulation
workers. Adjust concurrency only after memory/runtime benchmarking. No job has
been submitted. Neural integration remains dt=0.1 ms (1.8 million steps per
draw). See RUNTIME_ESTIMATES.md for measured local timing and conditional
cluster extrapolations; these are not cluster benchmarks.

Each 10k bank reserves 10% outside network training; the current recovery check
samples up to 100 of those held-out draws. Save joint samples and repeat training
seeds; no requirement that the posterior collapse to a single point.
