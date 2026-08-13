# AdEx parameter inference with VBI

## Scope and backend decision

TVBToolkit uses the maintained
[Virtual Brain Inference (VBI)](https://github.com/ins-amu/vbi) package as an
optional inference backend. The integration was developed and tested against
VBI `0.4.3`, `sbi` `0.23.2`, and PyTorch `2.5.0`.

The older editable checkout at `/Users/borjan/code/python/vbi` reports version
`0.3.1` and predates the current `ins-amu/vbi` package layout. It is not used by
this integration. VBI owns neural density estimation (SNPE, SNLE, and SNRE) and
its FC/FCD summary statistics. TVBToolkit owns the AdEx simulator, parameter
mapping, empirical-data alignment, state-template logic, and validation.

## Installation

For a new environment:

```bash
SKIP_CPP=1 conda env create -f environment.yml
conda activate tvbtoolkit
```

`SKIP_CPP=1` prevents VBI from compiling model backends that TVBToolkit does not
need. TVBToolkit's AdEx model continues to run through TVB.

To add inference support to an existing environment:

```bash
SKIP_CPP=1 python -m pip install -e ".[inference,dev,notebooks]"
python -c "import vbi; print(vbi.__version__)"
```

The last command must report the reproducibly pinned VBI version `0.4.3`.

## Starting prior

`AdExPrior.default()` defines independent uniform priors in this exact order:

| Inference name | AdEx/TVB target | Range | Unit |
|---|---|---:|---|
| `adaptation_b_e` | `parameter_model.b_e` | 0–120 | pA |
| `global_coupling` | `WholeBrainConfig.coupling_strength` | 0.05–0.50 | dimensionless |
| `conduction_speed` | `WholeBrainConfig.conduction_speed` | 1–20 | mm/ms, numerically m/s |
| `noise_amplitude` | `parameter_model.weight_noise` | 5e-5–2e-4 | model input units |

The requested “beta” is interpreted as `b_e`, the spike-triggered excitatory
adaptation increment. The implemented Zerlaut/AdEx model has no parameter
literally named `beta`. Use `include_external_drive=True` to add a fifth prior
over the external excitatory input.

These are defensible starting ranges, not universal physiological bounds.
Prior-predictive checks must reject regimes that diverge, saturate, or produce
constant BOLD before a scientific fit is run.

## Empirical and simulated features

The observation and every simulation use the same fitted
`BOLDFeatureExtractor`:

1. ROI-wise z-score and Butterworth band-pass for connectivity features.
2. VBI 0.4.3 static-FC distribution statistics.
3. VBI 0.4.3 sliding-window FCD distribution statistics.
4. Brain-Act legacy phase preprocessing.
5. Occupancy and self-transition probabilities relative to one fixed state
   template fitted to the empirical observation.
6. Mean and standard deviation of phase synchrony.

The fixed template is essential. Independently clustering each simulation would
make state labels permutation-dependent, so occupancy values would not be
comparable inputs to SBI.

For the default configuration, the feature vector has 32 entries: 10 FC, 10
FCD, 5 occupancy, 5 state-persistence, and 2 synchrony features.

## Brain Act alignment

`load_brain_act_bold_mat()` expects subject files containing `SC` and `BOLD`.
The audited DoC exports store BOLD in interleaved AAL90 order and SC in the
symmetric AAL90 order. The default loader applies the validated BOLD-only
permutation:

```text
0, 2, 4, ..., 88, 89, 87, ..., 1
```

Non-finite BOLD is rejected rather than imputed. SC is checked for square shape,
finiteness, non-negativity, and symmetry, then max-normalized.

The shared paired SC+BOLD files do not contain tract lengths. Consequently,
conduction speed cannot be inferred from those files alone. The example attaches
the packaged average AAL90 tract lengths, whose matrix ordering was checked
against the shared SC matrices. Prefer subject-specific, order-matched tract
lengths whenever available.

## End-to-end API

```python
from pathlib import Path

from tvbtoolkit import WholeBrainConfig
from tvbtoolkit.inference import (
    AdExBOLDSimulator,
    AdExPrior,
    BOLDFeatureConfig,
    BOLDFeatureExtractor,
    load_brain_act_bold_mat,
    simulate_prior,
    train_vbi_posterior,
)

record = load_brain_act_bold_mat(
    "subject.mat",
    tract_lengths="data/connectivity/average_aal90/tract_lengths.txt",
)

feature_extractor = BOLDFeatureExtractor(
    BOLDFeatureConfig(tr_seconds=record.tr_seconds)
)
x_observed = feature_extractor.fit_transform(
    record.bold,
    structural_connectivity=record.structural_connectivity,
)

prior = AdExPrior.default()
transient_ms = 20_000.0
base = WholeBrainConfig(
    simulation_length_ms=transient_ms + record.bold.shape[0] * record.tr_seconds * 1000.0,
    dt_ms=0.1,
    zerlaut_order=2,
    stochastic_integrator=True,
    monitor_mode="temporal_average",
    temporal_average_period_ms=1.0,
    weights=record.structural_connectivity,
    tract_lengths=record.tract_lengths,
)
simulator = AdExBOLDSimulator(
    base,
    prior,
    feature_extractor,
    transient_ms=transient_ms,
)

dataset = simulate_prior(
    prior,
    simulator,
    num_simulations=1000,
    feature_names=feature_extractor.feature_names_,
    seed=42,
)
dataset.save("outputs/adex_vbi/simulations.npz")
posterior = train_vbi_posterior(dataset, prior, method="SNPE")
```

See `notebooks/adex_vbi_brain_act_inference.ipynb` for plotting and guarded
simulation/training cells.

## What has and has not been demonstrated

The empirical Brain Act path has been exercised on a control record with 297
volumes and 90 regions. ROI alignment, filtering, VBI FC/FCD extraction, state
template fitting, and the 32-element finite observation vector all succeed.
Short real AdEx runs also confirm that TVB returns the configured BOLD monitor.

A scientifically valid Brain Act posterior has not yet been trained locally.
Matching 297 volumes at TR=2.4 s after a 20 s transient requires a 732.8 s AdEx
simulation. At `dt=0.1 ms`, that is 7.328 million integration steps per prior
sample; 1,000 simulations require 7.328 billion integration steps before neural
density-estimator training. This is an HPC workload, not a notebook smoke test.

Do not shorten simulated BOLD while retaining the full empirical observation:
that changes the sampling distribution of FC/FCD and state occupancy. For a
publishable result, run at matched duration and then perform:

1. prior-predictive rejection/QC;
2. simulation-based calibration on synthetic ground truths;
3. held-out parameter-recovery tests;
4. posterior-predictive simulations against FC, FCD, occupancy, and features
   not used during training;
5. sensitivity analysis over feature definitions and prior bounds;
6. repeated inference seeds and density-estimator architectures.

Until these checks pass, the framework is operational but an empirical
parameter estimate must not be presented as validated.
