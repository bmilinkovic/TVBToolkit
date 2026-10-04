# Implementation checks — 2026-10-04

Scope: software checks, **not scientific validation of an empirical b–G fit**.

- New tests check indexed seeds/priors, b/G mapping without mutating the source
  config, explicit AAL90 alignment, feature reproducibility, actual native
  first-order gNa/gK monitor configuration, and preparation of a synthetic bank.
- Existing inference parameter/dataset/backend tests also pass, including a
  small synthetic VBI training and posterior-sampling test.
- A 4-second actual stochastic first-order gNa/gK configuration smoke run
  completed with 90 regions, five states and BOLD monitoring. It exposed that TVB
  uses `temporal_average_period_ms`, not `monitor_period_ms`, for the neural
  monitor. The new driver was corrected and a regression assertion added.
- CLI help, Python compilation and both shell launchers pass syntax checks.
- No empirical fitting, full-length BOLD simulation bank, HPC submission, group
  comparison, convergence proof or posterior-predictive validation has occurred.

Local test command (capture disabled to avoid an unrelated local readline crash):

```bash
TVB_USER_HOME=/private/tmp/tvb_bg_home MPLCONFIGDIR=/private/tmp/tvb_bg_mpl \
OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/borjan/miniconda3/bin/python -m pytest -p no:capture \
tests/test_first_order_bg_sbi.py tests/test_inference_parameters.py tests/test_inference_sbi.py -q
```

Before empirical inference: supply BOLD/TR/preprocessing provenance, approve a
common G prior, compare longer warm-up, benchmark full-duration simulations,
inspect prior predictive physiological validity/feature coverage, then perform
synthetic parameter recovery and posterior-predictive checks.

## Subsequent four-subject extension (same date)

The earlier missing-data status above is superseded: four real matched Liege BOLD
banks were prepared, with historical-analysis TR=2.4 s recorded as provisional.
See FOUR_SUBJECT_PILOT.md for acquisition/preprocessing limitations.

All **eight** current workflow tests pass, including the double-gamma analytical
formula and fixed-state occupancy checks. A separate 4.8-second real first-order
gNa/gK smoke run with the double-gamma monitor produced 480 ten-ms neural samples
and two 2.4-second BOLD samples; kernel support was 32 s and its discrete sum 1.
This is a wiring check, not a baseline-convergence or haemodynamic-validation run.
Four 10k banks are planned (40k total); none of those long simulations has run.
Full-duration timing/QC, posterior inference and prediction remain outstanding.

## Two-minute revision — 2026-10-05

User confirmed TR=2.4 s and selected 120 retained seconds, excluding all brain-state
features. Current banks are `results/first_order_bg_sbi_four_2min/`: 50 observed
volumes, 119 features, four other 50-volume validation segments per participant.
All **nine** workflow tests pass. New regression checks cover explicit 50×90 time
axis filtering, feature/name length equality, and the 26-window/325-pair FCD
calculation. Earlier intermediate banks are superseded and source-hash checks
prevent reuse. No simulation bank or posterior was trained on those preparations.
The local neural integration benchmark is documented in RUNTIME_ESTIMATES.md;
no cluster job has been submitted.

## Cluster release — summaries-only policy

User explicitly withdrew raw-data saving after reviewing storage requirements.
Version 4 saves no simulated neural/BOLD recordings, only features and diagnostics.
Four empirical inputs successfully pass `prepare_cluster.py`, with six segments
covering all 297 empirical volumes. Local banks fingerprint local research code;
production banks must be prepared on the cluster after pulling committed code.

The **14 workflow tests** pass, including an end-to-end mocked monitor run that
asserts only feature/metadata files are saved, analytical two-minute FCD checks,
complete empirical coverage, numerical fit-score maximisation and PNG/PDF map
generation with a starred best fit. Shell syntax and Python compilation pass.
These are software checks; no 40k simulation campaign, warm-up comparison or
scientific validation of the window-marginalized posterior has been completed.
