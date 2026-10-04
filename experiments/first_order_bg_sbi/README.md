# First-order subject-specific b–G SBI

**Current cluster instructions: [CLUSTER_RUN.md](CLUSTER_RUN.md).** Version 4
saves summaries only (no raw simulated recordings), uses every empirical volume
through matched-length segments, and adds starred b–G descriptive fit maps.

This is a separate, **provisional inference workflow**, not a validated fit to
participant data. Four Liege participant banks are now prepared: see
[FOUR_SUBJECT_PILOT.md](FOUR_SUBJECT_PILOT.md). Nothing here changes the production
model or automatically submits a campaign.

## Scientific question

Can regional BOLD connectivity and slow temporal persistence constrain local
spike-triggered adaptation **b** separately from long-range coupling **G**, given
each participant's measured structural connectome?

b and tau_w are not interchangeable: in the simple spike-triggered adaptation
equation, the equilibrium contribution scales with b*tau_w*E, whereas recovery
depends on tau_w. Fixing tau_w=500 ms reduces confounding, but does not prove it is
the true value. A later sensitivity analysis should repeat inference at nearby
fixed tau_w values. b and G can still trade off; broad/correlated posteriors are
scientific results, not failures to hide.

## What is fixed

- First-order split-leak gNa/gK AdEx, **occupancy=0**, no TMS.
- tau_w=500 ms; conduction speed=4 m/s; existing stochastic drive/noise settings.
- Production background drive 0.315 Hz and dt=0.1 ms, with the existing stochastic
  Heun implementation. Neural E/I and TF remain in native kHz; W is pA.
- Subject-specific tract lengths and SC divided by the **shared control maximum
  single edge**, c=122.63607788085938. No per-subject or row normalization.
- b~Uniform(0,80) pA, G~Uniform(g_min,g_max), independent. The four-subject pilot
  uses G=0.025–0.075 from the earlier candidate sweep (production G=0.05).
- Use identical priors across diagnostic groups; COMA is disallowed.

The imported production configuration supplies all remaining parameters. Source
hashes freeze that implementation. A bank must be recreated if the code changes.
Existing stochastic-noise installation and no-rate-clipping behaviour are reused.

## Features and observation requirements

Input is an NPZ with **bold[time,90]** and **labels[90]**, plus an explicit TR in
seconds. Labels must match the structural atlas exactly; the script reorders by
labels rather than assuming left/right hemisphere order. Missing or constant
regions are errors, not silently imputed values. The current pilot retains
120 seconds per simulation and uses matched 50-volume empirical segments.

Supply motion/confound-corrected, dummy-volume-removed BOLD on a **regular time
grid**, before additional temporal filtering. Do not concatenate across scrubbed
gaps or apply this filter a second time to already filtered empirical data.
Motion processing, global-signal regression choices and missing/damaged ROI
handling need agreement before the empirical campaign. Record them using
`--preprocessing-note`; that note is provenance, not automatic preprocessing.

The same native extractor applies z-scoring and a second-order 0.01–0.10 Hz BOLD
bandpass to observations and simulations (this is not the TMS firing-rate filter).
Features: native VBI FC/FCD moments and quantiles; 60-second sliding-window FCD;
90 labelled FC row means; across-region mean/SD autocorrelation at 1,2,4 TRs.
Regional FC row means retain only a spatial summary, **not full edgewise FC**.
Version 2 also adds mean/SD BOLD slow-band power fraction and spectral peak.
Optional frozen five-state templates can add occupancy and conditional SC–phase-FC
coupling, but are **disabled in the current four-subject pilot** at the user's
request. Version 3's 119 features contain no brain-state metrics, with explicit
short-recording filtering and FCD (see FOUR_SUBJECT_PILOT.md). This set need not identify
a unique b/G: degeneracy is represented by the joint posterior. Full FC and state
durations remain useful independent checks, not already implemented targets.

Version 2 uses the TVB BOLD convolution driven by E with an explicit SPM-style
double-gamma HRF (shapes 6 and 16, unit scale, undershoot coefficient 1/6,
32-second support, unit discrete sum). The earlier Volterra default is no longer
used for newly prepared banks. This is an observation-model change only.
Matching BOLD summaries does not establish cellular ground truth. Haemodynamic
differences across brain injury can bias b/G inference and need sensitivity tests.

## Commands

Run from the repository root with the existing `.[inference]` environment
(maintained VBI >=0.4.3; existing installation is 0.4.3).

```bash
python experiments/first_order_bg_sbi/workflow.py prepare \
  --bold /path/to/c0005_bold_labelled.npz --subject c0005 --cohort control \
  --dataset-root /path/to/converted_structural_invnodevol_control_max \
  --tr 2.4 --g-min 0.025 --g-max 0.075 \
  --preprocessing-note 'REPLACE with actual preprocessing provenance' \
  --output /path/to/bg_c0005
```

**0.025–0.075 is the previously discussed candidate interval, not proven
physiological bounds.** Check rates and prior-predictive coverage across it.
No adaptation gradient is imposed by cohort. Preparation runs no simulation.

```bash
# Initial timing/validity benchmark, not an inferentially adequate sample:
python experiments/first_order_bg_sbi/workflow.py simulate \
  --bank /path/to/bg_c0005 --dataset-root /path/to/normalized_sc --stop 16 --workers 4

# Extend the identical indexed bank, skipping existing records:
python experiments/first_order_bg_sbi/workflow.py simulate \
  --bank /path/to/bg_c0005 --dataset-root /path/to/normalized_sc --stop 1000 --workers 48

# Separate training process, once all requested draws are complete and pass QC:
python experiments/first_order_bg_sbi/workflow.py train \
  --bank /path/to/bg_c0005 --budget 1000 --training-seed 1 --threads 8

# New noise realizations at JOINT posterior samples:
python experiments/first_order_bg_sbi/workflow.py predict \
  --bank /path/to/bg_c0005 --dataset-root /path/to/normalized_sc \
  --fit /path/to/bg_c0005/fit_1000_seed1 --count 20 --workers 4
```

There is one predictive set per bank; it cannot be overwritten accidentally.
Prediction currently saves feature vectors/metrics for comparison, not recordings or an automated
pass/fail judgment on empirical goodness of fit. Never load untrusted
`posterior.pt` files: Python posterior serialization uses pickle.

Cluster launchers are `hpc/submit_first_order_bg_sbi.sh` (simulation) and
`hpc/submit_first_order_bg_sbi_train.sh` (inference), using the existing cluster
bootstrap, workq, 48 CPUs and 14-day limit. Set `TVB_REPO`, `TVB_DATASET_ROOT`,
`BG_SBI_BANK`; explicitly increase `BG_SBI_STOP` for simulations. Training requires
`BG_SBI_BUDGET`. Submit from the repository root after creating `hpc/logs`.
No cluster submission has been made. Inspect RAM/storage from the pilot before
launching 48 simultaneous long simulations. Dataset paths are supplied at run
time; the manifest checks actual SC/tract content rather than Mac-specific paths.

Simulations last **120 retained seconds plus 60 seconds warm-up**. Default
warm-up is 60 s rounded upward to whole BOLD volumes. This is a starting setting,
not a demonstrated settling time: compare a longer warm-up before production.
Exact empirical sample count and TR are checked. Ten-ms neural averages are used
in memory for E/I/W QC. No simulated neural or BOLD time series are written for
any draw, including predictions or failed draws. Save parameters/seeds to permit
reruns; arbitrary future time-series analyses WILL require resimulation.

## Files saved per participant

- `manifest.json`: subject/group, common priors, protocol, labels, TR, SC/tract and
  empirical hashes, code provenance, feature order.
- `observation.npz`: reordered empirical BOLD, all segment feature vectors/weights.
- `draws/0000000/{features.npz,metadata.json}`: theta, seed,
  features, timing, rate/adaptation summaries or failure. No simulated raw signals.
- `run_START_STOP.json`: all statuses for a submitted range.
- `fit_BUDGET_seedN/`: training data, posterior object and 10,000 joint samples,
  marginal 90% intervals, b–G correlation, held-out recovery samples/coverage,
  `posterior_and_recovery.pdf`.
- `predictive/`: new stochastic posterior-predictive features and QC.
- `fit_map_BUDGET/`: b–G scatter coloured by feature-fit score, star at best valid
  draw, numerical scores and best-fit parameter/seed record. Not a posterior MAP.

All 297 empirical volumes enter six contiguous 50-volume segments (starts
0,50,100,150,200,247); overlap at the end is downweighted. Inference saves a
posterior conditional on each segment and a coverage-weighted **mixture** over
segments. This uses all observed data but is NOT the posterior for a joint
all-segment likelihood and does not pretend correlated segments are independent.
The mixture deliberately retains between-segment uncertainty. Keep these segment
posteriors for inspection; stricter joint sequence inference is separate work.

Nonfinite/feature failures are kept. Negative or >100 Hz **10-ms averaged** rates
are flagged for review; this is conservative QC, not a physiological cutoff or a
formal PFP diagnosis. Training refuses any flagged draw in the requested bank.
It does not silently drop cases, clip rates or renormalize SC. If the prior needs
revision, document it and generate a new bank with the same revised priors across
subjects. Current QC does not replace trajectory/attractor validation.

## How many simulations? Evidence, not a universal minimum

| Primary source | Inference example | Reported budget |
|---|---|---:|
| Hashemi et al. 2024, Fig. 4 | Six homogeneous MPR parameters | 100,000 |
| Hashemi et al. 2024, heterogeneous example | Regional excitability plus other parameters | 1,000,000 |
| Ziaeemehr et al., VBI 2025 reviewed preprint v1 | Single G from FC/FCD | Around 100 |
| Same VBI paper, pDMF example | Ten parameters | 50,000 |

Sources: [Hashemi et al.](https://doi.org/10.1088/2632-2153/ad6230),
[VBI paper, explicit reviewed-preprint version](https://elifesciences.org/reviewed-preprints/106194v1).
These are different models/observations and are **not an AdEx b–G benchmark**.
The 100k result supports your recollection of their practice, but not a rule that
every two-parameter problem needs at least that number.

Proposed progression: 16–32 full-duration runs for timing/QC; 1k pilot; then 5k,
10k, 50k, and 100k **if diagnostics justify expansion**. Each budget reserves 10%
outside neural-network training for recovery (up to 100 test cases by default).
The VBI/SBI backend also makes its own training/validation split. Repeat training
with several seeds, inspect joint posterior contraction/correlation, marginal
coverage and synthetic recovery across the prior. Compare predictive FC/FCD and
temporal statistics with observed values. Larger budgets cannot cure structurally
uninformative summaries or a wrong observation model.

For rigorous budget comparisons use a separate fixed synthetic test bank;
current held-out membership changes with budget, so its coverage estimates are
not paired comparisons or a full simulation-based calibration study.

This implementation trains **one posterior per participant/connectome**. Do not
reuse a posterior trained on one SC for another subject or pool banks blindly.
At 179 subjects, 100k EACH means 17.9 million full-duration simulations. Benchmark
CPU seconds and storage first; derive costs from actual full-duration runs. A
connectome-conditioned amortized network could share training, but is a different
method and is not implemented here. Start with representative participants.

After individual validation, summarize subject posteriors by cohort with
uncertainty. Such summaries are not a hierarchical group posterior. TMS/PCI
predictions remain a separate downstream application of validated parameters.
