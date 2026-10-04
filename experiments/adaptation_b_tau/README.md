# First-order adaptation screen

This separate experiment reuses the native first-order simulator and checkpoint
forking. It changes only excitatory b and tau_w. No model equations, second-order
code, transfer functions, weights, delays or production defaults are changed.

The manifest records the protocol, subjects, seeds, normalization and source
hashes. Each cell retains raw E/I/W recordings, regional metrics, PCI masks,
execution metadata and both condition rows. Software failures and physiological
invalidity are distinct. Pending combinations remain explicit in metrics.csv.

The grid has 179 subjects (35 controls, 18 EMCS, 75 MCS, 51 UWS), 54 pairs each.
Each pair uses 8 s equilibration, 6 s spontaneous recording, then matched 2 s
control and stimulated branches. Branches share state, delay history and future
noise. Different subjects have independent seeds; parameter pairs within a
subject share the seed. The pulse is 5 ms at 0.35 Hz/ms into excitatory rate in
left supplementary motor cortex (zero-based AAL index 9). Connectivity is divided
by one reference-control maximum edge, with G=0.05 and original conduction delays.

## Definitions and limitations

See protocol.json and metrics.py for all numeric criteria. UP/DOWN classification
requires two separated log-rate modes, minimum dwell time and completed cycles;
unimodal fluctuations do not automatically become UP/DOWN states. An OFF proxy
requires an initial positive response and sustained suppression below both a
baseline quantile and half its mean. Boundary durations are censored. This is a
firing-rate proxy, not the empirical EEG high-frequency suppression measurement.
Observed switching does not prove mathematical bistability. Six seconds gives
only preliminary duration/frequency estimates, especially near slow transitions.

PCI uses the existing implementations, alpha=0.10, baseline -400 to -100 ms,
response 0 to 300 ms from onset, 3 ms analysis sampling, without high-pass or
entropy floor. Control recordings are analyzed separately, not subtracted from
the PCI input. A single paired trial is insufficient for native trial-bootstrap
PCI-LZ: its column is missing, and baseline-resampled single-trial LZ is explicitly
exploratory. Repeated independent trials and longer baselines remain necessary
to validate shortlisted candidates and recruitment estimates.

No individual empirical targets were supplied. Selection is a declared ordinal
phenotype heuristic, NOT an individual fit or clinical validation. Diagnostic
labels enter this illustrative ranking only, never the simulations. Parameter
preference weakly favors the existing b=5 pA, tau=500 ms regime. PCI-ST/90 in the
score is only a bounded heuristic, not a published PCI normalization. Raw metric
contrasts at common parameter values must be inspected separately from ranking;
choosing group-specific scores cannot demonstrate a diagnostic distinction.
Invalid cells and cells without initial evoked responses cannot be selected.
Near-optimal sets and subject bootstrap selections quantify ranking dispersion,
not uncertainty of a measured empirical fit. All choices require sensitivity
testing before manuscript use.

## Commands

Use /Users/borjan/miniconda3/bin/python for these commands:

    python experiments/adaptation_b_tau/test_metrics.py
    python experiments/adaptation_b_tau/run.py plan
    python experiments/adaptation_b_tau/run.py run --workers 6
    python experiments/adaptation_b_tau/run.py report

Default output: results/adaptation_b_tau_v1. `launch` runs the full resumable
campaign in the background and records its PID/log. Existing completed cells
are reused; analysis failures with saved recordings can be recovered without
re-simulation. Do not change the protocol in an existing output directory.

Physiological motivation: Rosanova et al. (2018),
https://pmc.ncbi.nlm.nih.gov/articles/PMC6200777/ ; Goldman et al. (2023),
https://doi.org/10.3389/fncom.2022.1058957 . Neither establishes patient-specific
b/tau values or validates the heuristic selection score used here.
