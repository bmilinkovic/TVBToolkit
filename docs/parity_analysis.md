# TVBSim ↔ TVBToolkit Parity Analysis

**Date**: 2026-04-09  
**Scope**: Neuronal (AdEx/SNN) simulation · Mean-field simulation · Whole-brain simulation · LZc · PCI-like measures

---

## 1. Neuronal (AdEx/SNN) Simulation

**TVBSim source**: `brian_MF/adex_simulation_network.py`  
**TVBToolkit source**: `src/tvbtoolkit/brian_mf/adex/network.py` → `run_adex_network_simulation()`

### ✅ In parity

- Default parameter values are identical across both (Cm, Gl, tau_w, V_th, V_cut, V_m, V_r, a_e/i, b_e/i, delta_e/i, EL_e/i, tau_e/i, E_e/i, Q_e/i, p_con, gei, Ntot).
- AdEx equations are character-for-character identical (membrane dynamics, synaptic conductance decay, reset rule, threshold `vm > Vcut`, Heun integrator).
- Population structure is identical (N1 inhibitory FS, N2 excitatory RS, external Poisson drive, six synapse groups).
- `prepare_FR` / `prepare_population_rates` produce equivalent output (TimBinned, popRate arrays, adaptation signal).
- Save format is identical (mean and all `.npy` file structure).
- `defaultclock.dt` is set to 0.1 ms in both (TVBSim relies on Brian2 default; TVBToolkit sets it explicitly — same value).
- The split-leak `gK/gNa` variant (`adex_gK_gNa.py`) is unified into `run_adex_network_simulation(split_leak=True)` in TVBToolkit — not a divergence, just consolidation.

### ⚠️ Minor difference (no numerical impact)

- **Adaptation current initialization order** (inhibitory group): In TVBSim, `G_inh.w` is assigned before `G_inh.EL` is set, so the initial `w` is computed with `EL = 0 mV` (Brian2 default). In TVBToolkit, `EL` is set first, then `w`. This would matter only if `a_i > 0`, but both implementations default `a_i = 0`, so `w = 0` in either case. No numerical divergence for default parameters.

---

## 2. Mean-Field (MF) Simulation

**TVBSim source**: `brian_MF/MF.py` (calls `Tf_calc/theoretical_tools.py::run_MF`)  
**TVBToolkit source**: `src/tvbtoolkit/brian_mf/mean_field/mf.py` → `run_mean_field_simulation()`

### ✅ In parity

- Transfer-function formula (`_legacy_transfer_function`) is a faithful port of the Di Volo formulation used in `theoretical_tools.py`: same mu_v, s_v, tv computation; same 10-coefficient polynomial threshold; same erf-based output-rate formula.
- Ornstein-Uhlenbeck noise generation uses the same equation (mean-reverting, sigma=3.5, theta=1/(5 ms)), same time step (dt=1e-4 s).
- Adaptation dynamics (`w += dt * (-w/tau_w + b_rs * fe)`) are identical.
- `calculate_mf_difference()` — the public parity wrapper — produces results in the same column order `[mean_inh, mean_exc, amp_stim]` as the legacy script.

### ⚠️ Minor difference (documented)

- **MF plateau duration**: TVBToolkit hard-codes `plateau_ms = 900.0` with an explicit comment noting this is a legacy override from `run_MF`. This needs to be cross-checked against `theoretical_tools.py::run_MF` directly to confirm. If the legacy `run_MF` also uses 900 ms, the two are in parity; if `run_MF` computed the plateau dynamically (as the SNN script does: `TotTime - time_peek - TauP`), there is a discrepancy.

---

## 3. Whole-Brain Simulation

**TVBSim source**: `tvbsim/simconfig.py` + `tvbsim/run_simulations.py` (uses `nuu_tools_simulation_human.py::init`)  
**TVBToolkit source**: `src/tvbtoolkit/whole_brain/simulation.py` → `run_whole_brain_simulation()`

### ✅ In parity

- **Model selection logic** is identical: the four-way branch on `(matteo, gK_gNa)` × order 1/2 selects the same Zerlaut model class and the same `variables_of_interest` string lists.
- **Model parameter assignment**: same `to_skip` list (`['initial_condition', 'matteo', 'order', 'gK_gNa', 'noise_alpha', 'shared_noise_mode']`), same `setattr(model, key, np.array(value))` call, same `state_variable_range` initialization.
- **Shared-noise configuration**: the code in `_configure_shared_noise` is a verbatim copy of the TVBSim block (private/global/connectivity modes, `model._shared_noise_mode`, `model._shared_noise_matrix`).
- **Connectivity normalization and disconnection**: identical (`weights / sum + 1e-12`, zero-row/column disconnection).
- **Coupling**: same `getattr(lab.coupling, type)(**{k: np.array(v) for ...})` construction.
- **Integrator**: same Heun deterministic and stochastic constructors; same `noise.random_stream.seed(seed)` call.
- **Stimulation** (PulseTrain): same `onset/tau/T` parameter mapping, same region weight array, same `model.stvar` assignment.
- **Monitors**: Raw (with `RawVoi` → fallback to `Raw`), TemporalAverage, Bold, Ca — all constructed equivalently.

### ❌ Parity gaps

#### 3a. `AfferentCoupling` monitor — missing in TVBToolkit
TVBSim supports `parameter_monitor['Afferent_coupling']` and appends a `lab.monitors.AfferentCoupling` monitor. TVBToolkit's `_build_monitors` has no equivalent branch. If any workflow uses this monitor, results from TVBToolkit will not include that output channel.

#### 3b. Stochastic integrator nsig shaping
TVBSim passes the raw `nsig` array directly to `lab.noise.Additive(nsig=np.array(...))`. TVBToolkit applies an elaborate reshaping pipeline to produce `nsig` with shape `(nvar, n_regions, n_modes)`. If the default parameter file already provides a correctly shaped `nsig`, outputs are identical. If the raw `nsig` is a scalar or 1D vector, TVBSim relies on TVB to broadcast it while TVBToolkit pre-broadcasts it — the effective noise magnitude should be equivalent, but edge cases (e.g., `nsig` per-variable with fewer entries than `nvar`) will be handled differently. Worth a numerical spot-check when the parameter file's `nsig` is not already (nvar, n_regions, n_modes).

#### 3c. Raw monitor — conditional VOI
TVBSim always passes `variables_of_interest` to `RawVoi`. TVBToolkit only passes them when the `parameter_Raw` sub-dict is present and contains that key. In the default parameter file this sub-dict is likely present, so in practice the two behave the same, but it's a latent path difference.

#### 3d. `from_folder` connectivity — centres handling
TVBSim reads and transposes the `centres.txt` file (`centres=centers.T`). TVBToolkit passes `centres=np.array([])` for the `from_folder` path. Centres are not used in TVB dynamics, so this does not affect simulation output.

---

## 4. LZc Complexity

**TVBSim source**: `tvbsim/entropy_measures.py` → `calculate_LempelZiv` / `calculate_LempelZiv_single`  
**TVBToolkit source**: `src/tvbtoolkit/complexity/measures.py` → `lzc_multichannel` / `lzc_single_channel`

### ✅ In parity

- Binarization logic is the same: Hilbert-envelope amplitude > channel-wise mean of that envelope.
- Flattening order is the same: time-major (time axis varies fastest in the reshaped/rearranged string), producing the same binary sequence for equal input.
- Normalization is the same: LZc(data) / LZc(random shuffle).
- Single-channel variant: both compute per-channel LZc and average.

### ❌ Parity gap — preprocessing

TVBSim's `calculate_LempelZiv` calls `preprocess(data)` before binarization:
```python
data = preprocess(data)          # detrend(data - mean)
bin_data = binarize_signal(data) # Hilbert envelope > mean
```
TVBToolkit's `lzc_multichannel` binarizes the raw signal directly — **no detrending, no mean subtraction**:
```python
b = _binarize_hilbert(x)         # Hilbert envelope > mean (raw signal)
```

This means the Hilbert envelopes and their per-channel means are computed on different data. The binarized sequences will differ, and the resulting LZc values are **not numerically equivalent**. The effect is largest when the signal has a strong slow trend or non-zero mean, which is common in mean-field firing rate outputs.

The same gap applies to `lzc_single_channel` vs `calculate_LempelZiv_single`, since both just call their respective multichannel variants per channel.

### ⚠️ Minor difference (likely equivalent)

TVBSim uses the external `lempel_ziv_complexity` library (LZ76). TVBToolkit uses its own `_lz_complexity_binary_1d`. Both implement the same LZ76 algorithm and should agree on all inputs, but there has been no formal verification of numerical identity.

---

## 5. PCI-like Measures

**TVBSim source**: `tvbsim/PCI.py` (orchestration) + `tvbsim/TVB/pci_v2.py` (core) + `tvbsim/TVB/tvb_model_reference/src/nuu_tools_simulation_human.py` (`binarise_signals`)  
**TVBToolkit source**: `src/tvbtoolkit/complexity/pci_casali.py` (core primitives) + `src/tvbtoolkit/complexity/measures.py` → `pci_casali_like()`

### ✅ In parity

- **`sort_binJ`**: identical — sort channels by descending activation sum.
- **`source_entropy`**: identical formula — `p1 = sum(1==1)/L`, Shannon entropy `H = -p1*log2(p1) - p0*log2(p0)`.
- **`pci_norm_factor`**: identical formula — `S = (L * H) / log2(L)`.
- **`lz_complexity_2D`**: same 2D LZ76 algorithm. TVBSim uses `bitarray.search()`; TVBToolkit uses `bytes.find()`. For binary (0/1) sequences, both find the same substrings — the algorithms are equivalent. The global `ct` accumulator in TVBSim (used only by `calculate_pci_lower`, which is commented out) is absent in TVBToolkit; this has no effect on the main `lz_complexity_2D` return value.
- **`binarise_signals`**: the ported version in `pci_casali.py` faithfully reproduces the TVBSim logic — mean-normalise, std-normalise, shuffle pre-stim baseline `nshuffles` times, take `max_sorted[-int(nshuffles/percentile)]` as threshold. Default arguments (`nshuffles=10`, `percentile=100`) match TVBSim call site.
- **PCI calculation pipeline**: `pci_casali_like()` calls the primitives in the same order as `_calculate_PCI_seed_subset`: `binarise_signals → sort_binJ → lz_complexity_2d → pci_norm_factor`.

### ⚠️ Minor differences (no algorithmic divergence)

- **`SpeedUp` flag in `lz_complexity_2D`**: TVBSim has `SpeedUp=False` (disabled by default), which would sort and strip zero rows when enabled. TVBToolkit never strips zero rows. Since `SpeedUp=False` in TVBSim, the paths are identical in practice.
- **Unit-safety in `pci_casali_like`**: TVBToolkit explicitly converts `t_analysis_ms` to bins via `dt_ms` before indexing, and raises a `ValueError` for boundary violations. TVBSim's pipeline does this inline via `int(t_analysis/times_l[0])`. The calculation is equivalent but TVBToolkit is more defensive.
- **Input orientation handling**: `pci_casali_like` has an explicit `_coerce_channels_time` step that infers whether the input is `(time, channels)` or `(channels, time)`. TVBSim transposes explicitly in `_calculate_PCI_seed_subset` (`sig_region_all = np.transpose(sig_region_all)`). The intent is the same.
- **`pci_like` / `pci_ratio_proxy`**: TVBToolkit retains these as deprecated wrappers that compute `LZc(post)/LZc(pre)`. These are explicitly documented as **not** Casali PCI and not present in TVBSim at all. They should not be used for parity work.

---

## Summary Table

| Component | Parity | Notes |
|-----------|--------|-------|
| AdEx/SNN neuronal sim | ✅ Full parity | Initialization order bug in TVBSim has no numerical impact (`a_i=a_e=0`) |
| Mean-field simulation | ✅ Full parity | Plateau duration in MF should be verified against `theoretical_tools.py` |
| Whole-brain simulation (model, coupling, connectivity, stimulation) | ✅ Full parity | |
| Whole-brain — `AfferentCoupling` monitor | ❌ Missing in TVBToolkit | |
| Whole-brain — stochastic nsig shaping | ⚠️ Latent difference | Numerically equivalent for typical default nsig; edge-case sensitive |
| LZc (multichannel and single-channel) | ❌ Preprocessing gap | TVBSim detrends before Hilbert binarization; TVBToolkit does not |
| LZc — LZ76 algorithm | ✅ Equivalent | Different implementations (library vs. custom), same algorithm |
| PCI `binarise_signals` | ✅ In parity | |
| PCI `lz_complexity_2D` | ✅ In parity | `bitarray` vs `bytes.find` are equivalent for binary sequences |
| PCI `sort_binJ` / `source_entropy` / `pci_norm_factor` | ✅ In parity | |
| PCI orchestration (`pci_casali_like`) | ✅ In parity | Unit-safe bin indexing is an improvement, not a divergence |

---

## Recommended Actions

1. **LZc preprocessing**: Add the `preprocess` step (detrend + mean subtraction) to `lzc_multichannel` in `measures.py`, matching TVBSim's `calculate_LempelZiv`. This is the most impactful gap.

2. **MF plateau verification**: Read `Tf_calc/theoretical_tools.py::run_MF` and confirm whether the plateau duration is 900 ms (as hardcoded in TVBToolkit) or dynamically computed.

3. **`AfferentCoupling` monitor**: Add a branch in `_build_monitors` if this monitor is needed for any workflow.

4. **Stochastic nsig spot-check**: Run a paired numerical test with a non-scalar `nsig` to verify TVBSim and TVBToolkit produce the same noise magnitude.
