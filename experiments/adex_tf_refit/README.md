# AdEx transfer-function refit experiment

This is an **isolated experimental workspace**. No production coefficients or equations are changed.

## Purpose
Test whether the original and split-leak AdEx single-neuron responses can be better
approximated by the existing 10-coefficient erfc transfer function over low and high
input-rate regimes. A fit is **not** considered successful simply because it removes
a high-rate mean-field equilibrium.

## Run
From the repository root, in an environment with `brian2 numpy scipy pandas matplotlib`:

```bash
python experiments/adex_tf_refit/run.py --mode both --n-grid 15 --seeds 3 --duration-s 12 --output experiments/adex_tf_refit/results
```

For a quick smoke test (NOT adequate for scientific inference):
```bash
python experiments/adex_tf_refit/run.py --n-grid 4 --seeds 1 --duration-s 2
```

## Model alignment
- Two independent Poisson input streams, 400 excitatory and 100 inhibitory afferents,
  corresponding to N=10000, inhibitory fraction 0.2, p=0.05.
- Excitatory synapses Qe=1.5 nS, inhibitory Qi=5 nS, both 5 ms decay.
- RS and FS AdEx cells are fit separately. Original leak uses gL(EL-V).
- Split leak uses gK(EK-V)+gNa(ENa-V), with gK+gNa=gL and
  gK*EK+gNa*ENa=gL*EL. These are **electrically equivalent at fixed conductances**.
  This experiment is an equivalence control, not a model of receptor-dependent
  leak modulation. Additional split-leak conditions require separate validation.
- The code uses mean adaptation from the spiking simulation as the TF input.
- Output-rate training uses a log1p residual, which retains finite zero-rate samples.
- Validation is by **held-out Poisson seeds** at the same input locations.
  This is not out-of-domain validation.
- A 5-ms refractory period implies an upper rate limit of 200 Hz in the spiking cell.
  A predicted ~193 Hz rate may therefore be physiologically/numerically near
  refractory saturation, not necessarily just a fitting defect.

## Interpretation
The output contains observations, predictions, coefficients, and a JSON report
with held-out metrics by rate regime. Do not install the coefficients into TVB
without additionally checking Jacobian eigenvalues, equilibrium branches, and
whole-brain pulse recovery using the original integration timestep and connectomes.
The fit objective does not enforce the absence of high-rate attractors.

## Limitations
- The original legacy script used one RS and one FS cell per grid location and
  averaged the final 20% of a 10 s recording. This experiment averages over
  the latter half of each run, and repeats independent seeds.
- For low rates, long duration and multiple seeds are essential.
- For strong input, distinguish high *input* rates from high *output* rates.
- The 10-parameter polynomial is not guaranteed to extrapolate beyond its
  sampled voltage-statistics domain.
- Compare these newly fitted coefficients against the **actual coefficient
  arrays loaded by the specific TVB run**, not just defaults in a parameter file.
- No numerical results are included until the experiments are executed.
