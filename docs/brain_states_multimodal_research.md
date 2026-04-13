# Brain-State Algorithm for Firing Rates vs fMRI BOLD
## Research Report & Implementation Reference

**Date**: 2026-04-10  
**Status**: Approved for implementation

---

## 1. The Core Problem — An Intuitive Overview

Think of the brain-state pipeline as a machine that answers one question at every moment in time: *"How synchronized are different brain regions right now, and how does that pattern of synchronization look?"* It does this by extracting the instantaneous phase of each region's signal and computing how similar neighboring phases are across all region pairs.

The machine was designed for a very specific kind of signal: **fMRI BOLD**, which is slow, smooth, and band-limited between about 0.01 and 0.2 Hz. Applying it unchanged to **simulated firing rates** is like trying to run diesel fuel through a petrol engine — the basic mechanism works but you will get the wrong output because the fuel is completely different in its composition and energy profile.

There are four concrete mismatches. Each is explained below, first intuitively, then technically.

---

## 2. Mismatch 1 — BOLD is a Blurry Echo; Firing Rates are the Actual Signal

### Intuitive explanation

Imagine you have a live music concert (neural firing). You record it and then play the recording through an extremely reverberant cave with 6 seconds of echo. The sound you hear coming out of the cave — smeared, delayed, and smoothed — is BOLD. It still tells you something about the music, but with heavy distortion in time. Two instruments that played at different moments might now sound simultaneous inside the cave. Two instruments that played simultaneously might sound slightly different because of where they were sitting relative to the cave walls.

The HRF (haemodynamic response function) is that cave. BOLD is a linear convolution of neural activity with a function that peaks around 6 seconds and takes 20–30 seconds to settle. This means:
- Fast neural events (gamma oscillations at 30–80 Hz) are completely invisible in BOLD — averaged completely away
- What BOLD tracks is the very slow *envelope* of average activity over seconds
- Two regions can appear perfectly phase-synchronized in BOLD even if their moment-to-moment neural oscillations are anti-phase, as long as they are both "broadly co-active" at the same slow timescale

### Technical implication

Phase-coherence brain states computed from BOLD represent **infra-slow co-activation patterns** (the neural envelope at 0.01–0.2 Hz). Phase-coherence brain states computed directly from firing rates represent **neural oscillation synchronization** (at whatever frequency band you analyze). These are not the same quantity. They are not directly comparable without HRF convolution.

TVBToolkit already has `bold_from_firing_rates()` which applies the Volterra HRF kernel to convert firing rates into BOLD-like signals. If your goal is comparison with fMRI resting-state data, use this conversion and then run the standard BOLD pipeline. If your goal is understanding the underlying neural dynamics themselves, run the firing-rate pipeline described here.

---

## 3. Mismatch 2 — The Hilbert Transform Only Gives Meaningful Phase for Narrow-Band Signals

### Intuitive explanation

The Hilbert transform is a way of asking: *"At this exact moment, what is the cycle position — the phase — of the oscillation in this signal?"* This question only has a clean answer if the signal is dominated by one clear oscillation at any given time. Think of it like trying to read the time on a clock face. If there is exactly one clock on the wall, you can read it perfectly. If there are twelve different clocks all showing different times and all overlapping each other, you cannot determine what "the time" is — you just see noise.

A raw, unfiltered firing rate signal from a spiking neural network is like twelve overlapping clocks: it contains energy at many frequencies simultaneously (delta, theta, alpha, beta, gamma), all mixed together. The Hilbert transform on this broadband signal produces a phase estimate that jumps erratically, dominated by whichever frequency happens to be momentarily strongest. This is sometimes called "phase slippage" and makes the resulting brain-state patterns unreliable.

Filtering the signal to a specific band before applying the Hilbert transform is like putting on a lens that only lets you see one clock face at a time. You get a clean, stable, interpretable phase.

### Technical implication

For the phase-coherence pipeline to give meaningful brain states from firing rates, the signal **must** be narrowband-filtered before the Hilbert transform. The filter cutoffs should be chosen based on the dominant oscillation produced by the model, which varies by simulation type (see Section 5).

There is one important exception: for **simulated data with no noise**, a relatively wide band (e.g., 2–80 Hz) can still work reasonably well. In real electrophysiology recordings, a wide band would also pick up power-line noise (50 or 60 Hz) and muscle artifacts, making the phase estimates meaningless. In simulation, the signal is clean, so the Hilbert transform on 2–80 Hz will be dominated by whatever oscillation the model actually produces (typically gamma for the AdEx/MF models). This is why your suggested range is defensible for TVBToolkit's use case.

---

## 4. Mismatch 3 — Sampling Rate and Filter Cutoffs Are Completely Different

### Intuitive explanation

A bandpass filter for BOLD designed for signals sampled every 2 seconds (0.5 Hz) would pass frequencies between 0.01 and 0.2 Hz. If you apply that same filter to a firing rate signal sampled every 0.1 milliseconds (10,000 Hz), it would work mathematically — but it would zero out essentially the entire signal, because 0.01–0.2 Hz is a tiny sliver of the 0–5,000 Hz space the firing rate occupies. You would get almost nothing out. It is like asking a fire hose to deliver exactly one droplet per second — the filter is technically applicable but produces the wrong thing entirely.

The filter cutoffs must be designed in the context of the signal's sampling rate and the frequency range of interest. For firing rates, the relevant oscillations live 100 to 10,000 times faster than the BOLD oscillations, so the filter cutoffs need to shift by the same factor.

### Technical implication

TVBToolkit's AdEx and mean-field simulations bin their outputs at `bin_width_ms = 5.0 ms` by default (in `brian_utils.prepare_population_rates`). A 5 ms rectangular bin acts as a low-pass filter with:
- A -3 dB point at approximately 1/(π × 0.005 s) ≈ **64 Hz**
- A first spectral null at 1/0.005 s = **200 Hz**

This means the maximum usable bandwidth in the binned output is effectively 0–64 Hz (where energy is preserved) or generously 0–100 Hz (where some energy remains). Choosing an analysis band that extends above 100 Hz would be analyzing smoothing artefacts, not neural signal. This constrains the sensible upper cutoff to **≤80 Hz** for the AdEx/MF outputs, which is well within the usable range.

---

## 5. Mismatch 4 — Signal Stationarity and Normalization

### Intuitive explanation

BOLD at rest is a relatively stationary signal — it meanders slowly around a stable baseline with no dramatic jumps or silence. A single z-score over the whole recording is a fair normalization.

Firing rates from spiking networks can behave very differently. The AdEx adaptation current has a time constant of `tau_w = 500 ms`, which is long enough to produce "up-state / down-state" transitions — the network can alternate between periods of strong activity and near-silence. If you z-score the whole signal across one of these transitions, you are computing the mean and standard deviation of a bimodal distribution, which gives a poor normalization for either state separately.

### Technical implication

Firing-rate preprocessing should discard an initial transient (the network settling time, typically 200–1000 ms) before z-scoring. The z-score itself is appropriate, but if the simulation shows strong bursting or state transitions, consider epoch-wise normalization (z-score each continuous active period separately). For TVBToolkit's current use case (tonic input, looking for steady-state oscillations), a single global z-score after transient removal is sufficient.

---

## 6. The Frequency Band: Your Suggestion and Why It Is (Mostly) Right

**Your suggestion: 2–80 Hz. Assessment: correct for AdEx SNN and mean-field; needs clarification for TVB whole-brain.**

### What these models actually produce

**AdEx SNN** (from the actual TVBToolkit parameters):
- Membrane time constant: τ_m = Cm/Gl = 200 pF / 10 nS = **20 ms** → natural frequency ~50 Hz
- Synaptic time constants: τ_e = τ_i = **5 ms** → E-I loop resonance typically 30–60 Hz
- Adaptation time constant: τ_w = **500 ms** → very slow adaptation, can produce state transitions at 0.5–2 Hz
- Summary oscillatory content: **gamma (30–60 Hz) dominant**, with slower dynamics (2–10 Hz) from adaptation and network-level modulation

**Mean-field (Di Volo)**:
- Analytically approximates the AdEx SNN, so carries the same frequency content
- Typically slightly smoother (fewer high-frequency artefacts) than the raw binned SNN rates

**TVB whole-brain (Zerlaut, resting state)**:
- The *individual node* dynamics are in the gamma/beta range (same model family as above)
- The *inter-regional coupling* through structural connectivity creates slow envelope modulations in the **0.05–1 Hz** range — these are the resting-state fluctuations that give rise to BOLD-like correlations
- This split is important: there are two distinct phenomena at two very different timescales in the same signal

### Band recommendation per model

| Model | Relevant oscillation | Recommended band | Rationale |
|-------|---------------------|-----------------|-----------|
| AdEx SNN | E-I gamma + slow adaptation | **2–80 Hz** | Captures gamma (30–60 Hz), beta (13–30 Hz), and slower adaptation modulations (2–13 Hz); stays within binned Nyquist |
| Mean-field (Di Volo) | Same as AdEx | **2–80 Hz** | Same reasoning |
| TVB Zerlaut — local oscillations | Node gamma/beta | **2–80 Hz** | Same as above, applied per-region before network coupling |
| TVB Zerlaut — network resting state | Slow envelope modulations | **0.05–1.0 Hz** (after downsampling) | This is what drives BOLD-like SC-FC coupling; needs separate slow-band analysis |

### Why 2 Hz is the right lower bound

You asked specifically whether 2 Hz is the right lower cutoff. The answer is yes, and here is the reasoning:

- The adaptation current (τ_w = 500 ms) can drive oscillations or state transitions at approximately 1/τ_w ≈ **2 Hz**. Setting the lower cutoff at 2 Hz captures this and excludes slower drifts that are not oscillatory neural dynamics (baseline drift, thermal drift in real recordings — irrelevant for simulation but useful convention).
- Going lower, to say 0.5 Hz, risks picking up very slow transient effects from network settling that are not true sustained oscillations.
- Going higher, to say 5 Hz, would miss any adaptation-driven slow modulations and the lower end of alpha oscillations if the model produces them.

**2 Hz is the right lower bound.**

### Why 80 Hz is the right upper bound

- The AdEx SNN bins outputs at 5 ms → effective bandwidth ≤ 64–100 Hz
- Gamma oscillations in E-I networks with τ_i = 5 ms peak around 30–60 Hz
- 80 Hz is well within the reliable energy range of the binned signal and captures the full gamma band
- Going higher (e.g., 100+ Hz) adds nothing for the binned output and risks artefacts from the binning itself

**80 Hz is the right upper bound.**

### The one wrinkle: TVB Zerlaut whole-brain

For the Zerlaut model specifically, you will need to make a deliberate choice about which timescale you are studying:

**Option A — Local oscillation states (2–80 Hz)**: What synchrony patterns exist at the fast, intrinsic neural oscillation level within and between nodes? These brain states reflect fast neural coordination and are closer to what you would see in LFP recordings.

**Option B — Slow network states (0.05–1 Hz, after downsampling to e.g. 10 Hz)**: What slow co-activation patterns does the coupled whole-brain network produce? These are the patterns that correlate with resting-state BOLD networks and structural connectivity. This is what the brain-state algorithm was originally designed to capture at the BOLD level.

For comparing with empirical fMRI data, Option B (or converting via `bold_from_firing_rates()` first) is more appropriate. For studying intrinsic network oscillations at the neural level, Option A is more appropriate.

---

## 7. Implementation Plan (Approved)

### Changes to `brain_states.py`

Add a `"firing_rate"` pipeline branch to `phase_patterns()` with these steps:

1. **Transient removal** — drop first `transient_ms / dt_ms` samples
2. **Z-score** — per region over time, ddof=1 (same as standard)
3. **Narrowband filter** — Butterworth bandpass at `bandpass_hz`, default `(2.0, 80.0)` Hz, order 4
4. **Hilbert** — along time axis
5. **Phase** — `np.angle(analytic)` (no unwrap; gamma oscillations are cyclostationary)
6. **Edge trim** — `trim_edge_samples = 0` by default (filter handles its own edge effects)

New parameters added to `phase_patterns()` and `summarize_brain_states()`:
- `dt_ms: float = 5.0` — sampling interval of the firing-rate signal in milliseconds (default matches AdEx bin width)
- `transient_ms: float = 500.0` — initial samples to discard before analysis

### New file: `analysis/spectral.py`

Utility functions:
- `psd_per_region(x, dt_ms)` — Welch PSD per region, returns frequencies and power
- `dominant_frequency(x, dt_ms, bandpass_hz)` — peak frequency within band per region
- `phase_coherence_validity(x, dt_ms, bandpass_hz)` — mean analytic amplitude and Kuramoto order parameter; warns if signal is not band-limited

### No changes to

- `cluster_brain_states()` — modality-agnostic
- `sfc_sort_centroids()` — modality-agnostic
- `_compute_transition_matrix()` — modality-agnostic
- `corr_fc_sc()` — modality-agnostic

---

## 8. Quick-Reference Usage After Implementation

```python
from tvbtoolkit.analysis.brain_states import summarize_brain_states, sfc_sort_centroids

# From AdEx SNN / mean-field (binned at dt=5 ms)
summary = summarize_brain_states(
    firing_rates,          # shape (time, regions), binned at 5 ms
    n_states=5,
    pipeline="firing_rate",
    dt_ms=5.0,
    bandpass_hz=(2.0, 80.0),
    transient_ms=500.0,
)

# From TVB Zerlaut whole-brain — fast local oscillations
summary_fast = summarize_brain_states(
    tvb_rates,             # shape (time, regions), dt=0.1 ms
    n_states=5,
    pipeline="firing_rate",
    dt_ms=0.1,
    bandpass_hz=(2.0, 80.0),
    transient_ms=1000.0,   # longer transient for whole-brain settling
)

# From TVB Zerlaut whole-brain — slow network states
# Downsample to 10 Hz first (100 ms bins), then:
summary_slow = summarize_brain_states(
    tvb_rates_downsampled,  # shape (time, regions), dt=100 ms
    n_states=5,
    pipeline="firing_rate",
    dt_ms=100.0,
    bandpass_hz=(0.05, 1.0),
    transient_ms=1000.0,
)

# From BOLD (unchanged — existing pipeline)
summary_bold = summarize_brain_states(
    bold_signal,
    n_states=5,
    pipeline="standard",   # or "brain_act_legacy"
)
```
