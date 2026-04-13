# TVBToolkit ↔ Brain-Act Parity Analysis

**Date**: 2026-04-10  
**Scope**: Phase-pattern brain-state extraction · K-means clustering · SC-FC coupling  
**Brain-Act root**: `/Users/borjan/code/Brain-Act/brain-act`

Brain-Act contains three relevant analysis scripts that differ in their preprocessing philosophy:

| Script | Role |
|--------|------|
| `code/04_02_phase_coherence.py` | Standard (modern) phase-coherence pipeline |
| `code/04_04_analyse_phase_states.py` | Extended version with data-driven k selection |
| `code/legacy_phase_coherence_new.py` | Legacy pipeline with bandpass filtering and SFC sorting |

TVBToolkit exposes both pipelines through `src/tvbtoolkit/analysis/brain_states.py` via the `pipeline` argument:
- `pipeline="standard"` → targets parity with `04_02` / `04_04`
- `pipeline="brain_act_legacy"` → targets parity with `legacy_phase_coherence_new.py`

---

## 1. Phase-Pattern Extraction

### 1a. "Standard" pipeline — TVBToolkit vs Brain-Act `04_02` / `04_04`

**Brain-Act `04_02` / `04_04`** (`compute_patterns_one_subject`):
```python
# Input: X_rt shape (R, T)
Xz = stats.zscore(X_rt, axis=1, ddof=1)   # z-score over time, per region
analytic = spsg.hilbert(Xz, axis=1)
phases = np.angle(analytic)               # no unwrap
t_idx = np.arange(T_shift, T - T_shift)  # T_shift=9 edge guard (both ends)
P[k] = np.cos(phases[iu, t] - phases[ju, t])
Sglob[k] = np.abs(np.mean(np.exp(1j * phases[:, t])))
```

**TVBToolkit** (`phase_patterns`, pipeline="standard"):
```python
# Input: x shape (T, R)
xz = zscore(x, axis=0, ddof=1)           # z-score over time, per region
analytic = hilbert(xz, axis=0)
phase = np.angle(analytic)               # no unwrap
trim_edge_samples = 9                    # same edge guard (both ends)
patterns[k] = np.cos(phase[ti, iu] - phase[ti, ju])
global_sync[k] = np.abs(np.mean(np.exp(1j * phase[ti, :])))
```

### ✅ In parity

- Z-score: `ddof=1` over the time axis — identical (transposed orientation is equivalent).
- No bandpass, no demean across ROIs.
- Hilbert applied along the time axis.
- Phase: `np.angle()` — no unwrapping.
- Edge guard: 9 samples trimmed from both ends — identical.
- Phase-difference formula: `cos(phase_i - phase_j)` — identical.
- Global synchrony: `|mean(exp(i·φ))|` — identical.
- Upper-triangle index convention: `np.triu_indices(R, k=1)` — identical.

---

### 1b. "Legacy" pipeline — TVBToolkit vs `legacy_phase_coherence_new.py`

**Brain-Act legacy** (`build_phase_coherence`):
```python
# Input: [N, T] (ROI × Time)
ts_z = zscore_over_time(ts)              # ddof=0 (np.std default)
ts_z = demean_over_rois_per_time(ts_z)  # subtract ROI mean at each timepoint
ts_f = butterworth_filtering_legacy(ts_z, TR)  # iirfilter 3rd-order, 0.01–0.20 Hz, filtfilt
hilb = spsg.hilbert(ts_f, axis=-1)
inst_phase = np.unwrap(np.angle(hilb), axis=-1)  # unwrapped phase
# NO T_shift trim
val = np.cos(adif(inst_phase[i, t], inst_phase[j, t]))  # upper-tri, i<j
```

**TVBToolkit** (`phase_patterns`, pipeline="brain_act_legacy") via `_legacy_preprocess`:
```python
# Input: (T, R) — transposed internally to (R, T) for preprocessing
xz = (xr - mu) / sd  # ddof=0 (np.std default), per-ROI over time
xz = xz - np.mean(xz, axis=0)  # demean across ROIs at each timepoint
xf = filtfilt(b, a, xz, axis=-1)  # iirfilter 3rd-order, 0.01–0.20 Hz
analytic = hilbert(xl, axis=0)  # transposed back to (T, R)
phase = np.unwrap(np.angle(analytic), axis=0)  # unwrapped
trim_edge_samples = 9 (default)  # ⚠️ Brain-Act legacy does NOT trim
```

### ✅ In parity (preprocessing core)

- Z-score: `ddof=0`, per-ROI over time — identical.
- Demean across ROIs per timepoint — identical.
- Bandpass: 3rd-order Butterworth iirfilter, 0.01–0.20 Hz, `filtfilt` — identical.
- Phase: `np.unwrap(np.angle(...))` — identical.
- Phase-difference formula: `cos(adif(φ_i, φ_j)) ≡ cos(φ_i - φ_j)` — identical (both compute the shortest angular distance then apply cosine).

### ❌ Parity gap — edge trimming in legacy mode

TVBToolkit's `"brain_act_legacy"` pipeline trims `trim_edge_samples=9` from both ends by default. Brain-Act's legacy script applies **no edge trimming**. For exact legacy parity, call TVBToolkit with `trim_edge_samples=0`.

---

### 1c. Internal Brain-Act inconsistency (informational)

Brain-Act `04_02`/`04_04` use `ddof=1` in `scipy.stats.zscore`, while `legacy_phase_coherence_new.py` uses `ddof=0` (via `np.std`). TVBToolkit's "standard" uses `ddof=1` and "brain_act_legacy" uses `ddof=0`, so each TVBToolkit mode correctly mirrors its corresponding Brain-Act script.

---

## 2. K-means Clustering

### Hyperparameter comparison

| Parameter | Brain-Act `04_02`/`04_04` | Brain-Act legacy | TVBToolkit (`sklearn` backend) |
|-----------|--------------------------|------------------|-------------------------------|
| `random_state` | `42` | `1` (`KM_SEED`) | `0` (default `random_seed`) |
| `n_init` | `10` | `200` (`KM_INIT`) | `20` (default) |
| `max_iter` | `300` (sklearn default) | `300` (`KM_MAXITER`) | `100` (default) |
| `init` | `'k-means++'` (sklearn default) | `'k-means++'` | `'k-means++'` |

### ❌ Parity gap — hyperparameters

**None of the three sources agree on defaults.** TVBToolkit's defaults (`random_seed=0`, `n_init=20`, `max_iter=100`) match neither Brain-Act script. The number of initializations and the random seed both affect the final cluster assignment in ambiguous landscapes.

**Recommended remediation:**

To match Brain-Act `04_02` / `04_04`:
```python
cluster_brain_states(patterns, n_states=k, backend="sklearn",
                     random_seed=42, n_init=10, max_iter=300)
```

To match the Brain-Act legacy script:
```python
cluster_brain_states(patterns, n_states=k, backend="sklearn",
                     random_seed=1, n_init=200, max_iter=300)
```

Consider documenting these parameter sets in `BrainStateSummary` or exposing them as named presets.

---

### 2a. Scipy backend

TVBToolkit's `"scipy"` backend wraps `scipy.cluster.vq.kmeans2` with `minit="points"`. Brain-Act uses sklearn `KMeans` exclusively. The two algorithms implement the same Lloyd's algorithm but may produce different label assignments due to initialization differences. For strict parity with Brain-Act always use `backend="sklearn"`.

---

## 3. Transition Matrix

**Brain-Act `04_02`/`04_04`** (`plot_state_transitions`):
```python
for i in range(len(idx) - 1):
    trans_mat[idx[i], idx[i + 1]] += 1
row_sums = trans_mat.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
trans_mat = trans_mat / row_sums
```
Self-transitions are included; runs are **not** collapsed.

**Brain-Act legacy** (`markov_transition_no_self`):
```python
seq = remove_redundancies(seq)  # collapse runs (1,1,2→1,2)
for a, b in zip(seq[:-1], seq[1:]):
    if a == b: continue         # skip self-transitions (shouldn't occur after collapse)
P[a, b] += 1
```
Runs are collapsed; self-transitions are excluded.

**TVBToolkit** (`_compute_transition_matrix`):
```python
for i in range(labels.size - 1):
    tm[a, b] += 1.0
row_sum[row_sum == 0.0] = 1.0
return tm / row_sum
```
Self-transitions are included; runs are **not** collapsed.

### ✅ TVBToolkit matches Brain-Act `04_02`/`04_04`

Both include self-transitions and do not collapse runs.

### ❌ TVBToolkit differs from Brain-Act legacy

Brain-Act legacy collapses runs and excludes self-transitions (Markov no-self). TVBToolkit does not expose this variant. This should be noted when comparing legacy-derived occupancy or entropy results.

---

## 4. SC-FC Coupling

Brain-Act and TVBToolkit compute SC-FC relationships in different contexts and at different levels of analysis. The table below maps each implementation:

| Implementation | What is correlated | Upper-tri or full? |
|----------------|--------------------|--------------------|
| Brain-Act legacy `sfc_sort_and_relabel` | Each brain-state centroid vector vs SC upper-tri | Upper-tri only |
| Brain-Act legacy per-subject SFC | Each centroid vs per-subject SC upper-tri | Upper-tri only |
| TVBToolkit `corr_fc_sc` (bold.py) | Full Pearson FC matrix vs full SC matrix | Full (diagonal included) |
| TVBToolkit `fcsc_seedwise_from_saved_batch` | Three variants: full signed, masked upper-tri abs, masked upper-tri signed | Upper-tri and full |

### ❌ Parity gap — TVBToolkit `corr_fc_sc` uses full matrix

TVBToolkit's `corr_fc_sc` flattens the full `(N, N)` FC and SC matrices:
```python
fc = np.corrcoef(sig)                            # full N×N
coef = np.corrcoef(fc.reshape(-1), sc.reshape(-1))[0, 1]
```
Brain-Act legacy uses the **upper-triangle only** (i < j), excluding the diagonal:
```python
v_sc = [sc[i, j] for i < j]                     # upper-tri, no diagonal
r, _ = pearsonr(centroid_vec_upper_tri, v_sc)
```
Including the diagonal and redundant lower triangle inflates the correlation artificially (diagonal is always 1 in Pearson FC). The masked upper-triangle variant in `fcsc_seedwise_from_saved_batch` is the correct equivalent.

### ❌ Parity gap — TVBToolkit lacks centroid-SFC sorting

Brain-Act legacy sorts brain-state centroids by their Pearson correlation with SC (ascending), relabelling all labels accordingly. TVBToolkit has no equivalent of this `sfc_sort_and_relabel` function. This means TVBToolkit brain-state labels are not SC-ordered and **cannot be directly compared** to Brain-Act legacy outputs without post-hoc reordering.

### ⚠️ Conceptual difference (not a bug)

Brain-Act's SFC is computed per centroid (state-level structure-function coupling); TVBToolkit's `corr_fc_sc` is a classic global FC-SC metric computed from the full time-averaged correlation matrix. These are complementary metrics, not the same quantity.

---

## 5. K-Selection (Brain-Act `04_04` only)

Brain-Act `04_04_analyse_phase_states.py` adds automatic k-selection using Silhouette, Calinski–Harabasz, Davies–Bouldin, Inertia, and Gap statistic. TVBToolkit has no equivalent k-selection module. This is a missing feature rather than a parity gap in existing functionality.

---

## Summary Table

| Component | Parity | Notes |
|-----------|--------|-------|
| Phase extraction — standard pipeline | ✅ Full parity | zscore(ddof=1), Hilbert, angle, T_shift=9, cos(Δφ), global synchrony |
| Phase extraction — legacy pipeline (preprocessing) | ✅ Full parity | zscore(ddof=0), ROI demean, bandpass 0.01–0.20 Hz order 3, Hilbert, unwrap |
| Phase extraction — legacy edge trimming | ❌ Parity gap | TVBToolkit trims 9 samples (default); Brain-Act legacy trims 0. Fix: `trim_edge_samples=0` |
| K-means random_state | ❌ Parity gap | Brain-Act 04_02/04_04: 42; legacy: 1; TVBToolkit: 0 |
| K-means n_init | ❌ Parity gap | Brain-Act 04_02/04_04: 10; legacy: 200; TVBToolkit: 20 |
| K-means max_iter | ⚠️ Minor gap | Brain-Act 04_02/04_04 and legacy: 300; TVBToolkit: 100 |
| K-means init method | ✅ In parity | All use `'k-means++'` (sklearn backend) |
| Transition matrix | ✅ Matches 04_02/04_04 | Both include self-transitions, no run collapse |
| Transition matrix vs legacy | ❌ Differs | Legacy collapses runs and excludes self-transitions |
| Global synchrony formula | ✅ Full parity | `\|mean(exp(i·φ))\|` |
| SC-FC coupling — upper-tri vs full | ❌ Parity gap | `corr_fc_sc` uses full matrix; Brain-Act uses upper-tri only |
| SC-FC coupling — centroid SFC sorting | ❌ Missing in TVBToolkit | Brain-Act legacy sorts states by SC correlation; TVBToolkit has no equivalent |
| Automatic k-selection | ❌ Missing in TVBToolkit | Present in Brain-Act 04_04 (Silhouette, CH, DB, Inertia, Gap) |

---

## Recommended Actions

1. **Legacy edge trimming**: When using `pipeline="brain_act_legacy"`, pass `trim_edge_samples=0` for exact parity with Brain-Act's legacy script.

2. **K-means hyperparameters**: Expose preset kwargs (e.g., `preset="brain_act_04"` → `random_seed=42, n_init=10, max_iter=300` and `preset="brain_act_legacy"` → `random_seed=1, n_init=200, max_iter=300`) in `cluster_brain_states()` to make parity straightforward.

3. **SC-FC coupling — use upper triangle**: Replace `corr_fc_sc`'s full-matrix flatten with an upper-triangle masked version (equivalent to the existing masked variant in `fcsc_seedwise_from_saved_batch`). This aligns with Brain-Act's convention and avoids diagonal inflation.

4. **Centroid SFC sorting**: Add an `sfc_sort_centroids(centers, sc)` utility that sorts brain-state centroids by ascending Pearson correlation with the SC upper-triangle vector. This is a one-function gap but required for cross-study label comparability.

5. **Transition matrix legacy variant**: Add a `collapse_runs=True` / `exclude_self=True` option to `_compute_transition_matrix` for parity with Brain-Act legacy's Markov analysis.

6. **Automatic k-selection** (optional): Port Brain-Act `04_04`'s `choose_k_by_metrics` logic (Silhouette, CH, DB, Inertia, Gap) into TVBToolkit as a convenience function.
