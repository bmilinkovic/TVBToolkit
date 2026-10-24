# AdEx receiving-node gain propagation experiment

Isolated experimental branch; **main and production TF coefficients unchanged**.

## Motivation

Alter susceptibility of receiving regions while preserving structural weights.
`gain_vector[i]` is dimensionless; `saturation_khz` is a rate (kHz).

The repository's first-order `Zerlaut_adaptation_first_order.dfun` receives
`coupling[0,:]` as long-range excitation in kHz. The experimental
`ReceivingGainMixin` transforms only that coupling input and calls production
`dfun` unchanged. The directly injected pulse is not automatically limited.

## Use with TVB installed

```python
from tvbtoolkit.whole_brain.legacy_engine.src.Zerlaut import Zerlaut_adaptation_first_order
from experiments.adex_gain_propagation.gain import ReceivingGainMixin

class ExperimentalGainZerlaut(ReceivingGainMixin, Zerlaut_adaptation_first_order):
    pass

model = ExperimentalGainZerlaut()
model.gain_vector = gamma  # shape (n_regions,)
model.reference_coupling = baseline_coupling  # shape (n_regions,), kHz
model.saturation_khz = 0.01
```

Run the base model first, measure its stable baseline coupling, then use the
same connectome, coupling function, delays, initial conditions and stimulus
on each gain condition. Do not apply a different source-conditioned gain map
to different stimulated nodes and call it a fixed regional parameter.

### Gain assignment

`heterogeneous_gain(W, stim_node, amount=1)` is an **exploratory hypothesis**:
gamma_i = 1 + alpha * [W[i,s] + (W @ W)[i,s]] / max_k(...).
It mixes direct and 2-hop directed weighted reachability. It does **not**
measure regional excitability independently from connectivity and is prone
to double-counting network topology. Compare against homogeneous gain and
shuffled gains; ideally infer gamma from empirical regional excitability
measurements independent of the connectome.

Tract lengths belong in TVB's delay kernel, not in gamma as a proxy for
anatomical proximity. If you use a distance-dependent gain prior, distinguish
direct anatomical tract lengths from Euclidean distance, and explicitly test
whether it adds explanatory value beyond structural weights.

### Critical checks

1. Zero-stimulus exact-baseline invariance after transformation.
2. Whole-brain Jacobian stability: local stability alone does not suffice.
3. Dose-response, 10-ms pulse, 0.1–1 s response and >3 s recovery.
4. Region x time butterfly/heatmap and downstream latency/amplitude metrics.
5. Persistent ~193 Hz state must be reported, not silently clipped.
6. E/I output in Hz is a simulated firing-rate source signal, **not scalp TEP**.

## Executed synthetic 90-node proof of concept

An independent NumPy/SciPy parity prototype was executed separately with
the Berlin first-order TF parameters. It used synthetic 90-node connectivity,
not a patient SC and not the native TVB integrator. At coupling gain 0.02,
15-Hz-per-afferent direct pulse, 10 ms: original gain recruited 0 downstream
regions above 1 Hz; heterogeneous linear and bounded gain recruited 2 each.
All three left the stimulated region at ~192.8 Hz, establishing that the
coupling gain does not itself remove the high-rate attractor.

The executable standalone prototype and figures were provided as conversation
artifacts alongside this experimental implementation.

This experiment is not a validated patient-specific whole-brain result.
