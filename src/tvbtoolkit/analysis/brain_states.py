"""Brain-state analysis utilities based on phase-coherence pattern clustering.

This module exposes three analysis pipelines via the ``pipeline`` argument:

- ``standard`` (default): lightweight Hilbert phase extraction used in earlier
  TVBToolkit runs and matching Brain-Act ``04_02``/``04_04``.
- ``brain_act_legacy``: parity-oriented preprocessing/clustering settings that
  mirror the Brain-Act legacy script (``legacy_phase_coherence_new.py``).
  Applies ROI-wise z-scoring, ROI-mean removal, Butterworth band-pass filtering,
  Hilbert phase unwrapping, and KMeans-style clustering.
- ``firing_rate``: designed for population firing rates from AdEx SNN,
  mean-field, or TVB whole-brain simulations. Discards an initial transient,
  z-scores per region, applies a narrowband Butterworth filter (default
  2–80 Hz for AdEx/MF; use 0.05–1 Hz for slow TVB whole-brain states), then
  runs the same Hilbert + cos(Δφ) core as the other pipelines. Requires
  ``dt_ms`` to be set correctly for the input signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.signal import filtfilt, hilbert, iirfilter
from scipy.spatial.distance import squareform
from scipy.stats import zscore

try:  # optional dependency for Brain-Act legacy clustering parity
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None


@dataclass(frozen=True)
class BrainStateSummary:
    """Summary outputs from phase-pattern clustering."""

    labels: np.ndarray
    centers: np.ndarray
    occupancy: np.ndarray
    transition_matrix: np.ndarray
    global_synchrony: np.ndarray
    edge_index_i: np.ndarray
    edge_index_j: np.ndarray



def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected shape (time, regions), got {x.shape}.")
    return x



def _legacy_preprocess(
    x: np.ndarray,
    *,
    tr_seconds: float,
    bandpass_hz: tuple[float, float],
    filter_order: int,
) -> np.ndarray:
    """Apply Brain-Act legacy preprocessing on a time-by-region signal.

    Parameters
    ----------
    x : np.ndarray
        Array of shape ``(time, regions)``.
    tr_seconds : float
        Sampling period (seconds) used for filter design.
    bandpass_hz : tuple[float, float]
        Low/high cut frequencies in Hz.
    filter_order : int
        Butterworth order.

    Returns
    -------
    np.ndarray
        Preprocessed signal in shape ``(time, regions)``.
    """
    xr = np.asarray(x, dtype=float).T  # regions x time

    # Legacy-style ROI-wise z-score over time (ddof=0 equivalent).
    mu = np.mean(xr, axis=-1, keepdims=True)
    sd = np.std(xr, axis=-1, keepdims=True) + 1e-12
    xz = (xr - mu) / sd

    # Legacy demean across ROIs at each time point.
    xz = xz - np.mean(xz, axis=0, keepdims=True)

    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be strictly positive.")

    nyq = 0.5 / float(tr_seconds)
    low_hz, high_hz = float(bandpass_hz[0]), float(bandpass_hz[1])
    if not (0.0 < low_hz < high_hz < nyq):
        raise ValueError(
            "bandpass_hz must satisfy 0 < low < high < Nyquist. "
            f"Got low={low_hz}, high={high_hz}, nyquist={nyq}."
        )

    low = low_hz / nyq
    high = high_hz / nyq
    b, a = iirfilter(int(filter_order), (low, high), btype="bandpass", ftype="butter", output="ba")

    # If input is too short for filtfilt padding, skip filtering gracefully.
    try:
        xf = filtfilt(b, a, xz, axis=-1)
    except ValueError:
        xf = xz

    return xf.T  # back to time x regions



def _firing_rate_preprocess(
    x: np.ndarray,
    *,
    dt_ms: float,
    transient_ms: float,
    bandpass_hz: tuple[float, float],
    filter_order: int,
) -> np.ndarray:
    """Preprocess a firing-rate signal for phase-coherence analysis.

    Pipeline (per column / region):
    1. Discard the initial transient (first ``transient_ms / dt_ms`` samples).
    2. Z-score over time (ddof=1).
    3. Apply a narrowband Butterworth bandpass filter.

    Parameters
    ----------
    x : np.ndarray
        Array of shape ``(time, regions)`` in Hz. Must already be at the
        resolution described by ``dt_ms`` (e.g., the 5 ms binned output from
        ``prepare_population_rates``).
    dt_ms : float
        Sampling interval in milliseconds.
    transient_ms : float
        Initial duration to discard (milliseconds).
    bandpass_hz : tuple[float, float]
        Low/high cut frequencies in Hz. Default recommendation: ``(2.0, 80.0)``
        for AdEx SNN / mean-field; ``(0.05, 1.0)`` for slow TVB whole-brain
        network states (requires downsampling to ≥2 Hz first).
    filter_order : int
        Butterworth order. Recommended ``4`` for a clean narrowband response.

    Returns
    -------
    np.ndarray
        Filtered signal with shape ``(time_after_transient, regions)``.
    """
    x = np.asarray(x, dtype=float)
    transient_samples = int(round(transient_ms / dt_ms))
    if transient_samples >= x.shape[0]:
        raise ValueError(
            f"transient_ms={transient_ms} ms discards {transient_samples} samples "
            f"but the signal only has {x.shape[0]} samples."
        )
    xc = x[transient_samples:]  # drop transient

    # Z-score per region over time (ddof=1, same as standard pipeline)
    xz = zscore(xc, axis=0, ddof=1)
    xz = np.nan_to_num(xz, nan=0.0)

    # Narrowband Butterworth filter
    dt_s = dt_ms / 1000.0
    nyq = 0.5 / dt_s
    low_hz, high_hz = float(bandpass_hz[0]), float(bandpass_hz[1])
    if not (0.0 < low_hz < high_hz < nyq):
        raise ValueError(
            f"bandpass_hz must satisfy 0 < low < high < Nyquist={nyq:.3f} Hz. "
            f"Got low={low_hz}, high={high_hz}. "
            f"Check dt_ms={dt_ms} and your filter cutoffs."
        )
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = iirfilter(int(filter_order), (low, high), btype="bandpass", ftype="butter", output="ba")
    try:
        xf = filtfilt(b, a, xz, axis=0)
    except ValueError:
        # Signal too short for default padding — use Gustafsson's method
        xf = filtfilt(b, a, xz, axis=0, method="gust")
    return xf


def phase_patterns(
    x: np.ndarray,
    trim_edge_samples: int | None = None,
    *,
    pipeline: Literal["standard", "brain_act_legacy", "firing_rate"] = "standard",
    tr_seconds: float = 2.4,
    bandpass_hz: tuple[float, float] = (0.01, 0.20),
    filter_order: int = 3,
    dt_ms: float = 5.0,
    transient_ms: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute phase-coherence patterns from a regional time series matrix.

    Parameters
    ----------
    x : np.ndarray
        Array of shape ``(time, regions)``.
    trim_edge_samples : int | None, default=None
        Number of samples trimmed from both ends for the returned patterns.
        When ``None`` (default), the value is set automatically per pipeline:
        ``9`` for ``"standard"`` (matching Brain-Act ``04_02``/``04_04``),
        ``0`` for ``"brain_act_legacy"`` (Brain-Act legacy applies no edge
        trimming), and ``0`` for ``"firing_rate"`` (the narrowband filter
        handles edge effects internally). Pass an explicit integer to override.
    pipeline : {"standard", "brain_act_legacy", "firing_rate"}, default="standard"
        Preprocessing/phase extraction path.

        - ``"standard"``: z-score (ddof=1) → Hilbert → angle. Matches
          Brain-Act ``04_02``/``04_04``. For fMRI BOLD.
        - ``"brain_act_legacy"``: z-score (ddof=0) + ROI demean + Butterworth
          bandpass (0.01–0.20 Hz) + Hilbert + unwrap. Matches Brain-Act
          ``legacy_phase_coherence_new.py``. For fMRI BOLD.
        - ``"firing_rate"``: transient removal + z-score (ddof=1) + narrowband
          Butterworth (default 2–80 Hz) + Hilbert + angle. For population
          firing rates from AdEx SNN, mean-field, or TVB whole-brain sims.
          Requires ``dt_ms`` to be set to the signal's actual sampling interval.

    tr_seconds : float, default=2.4
        Sampling period (seconds) for ``"brain_act_legacy"`` band-pass filter.
    bandpass_hz : tuple[float, float], default=(0.01, 0.20)
        Low/high filter cutoffs in Hz.
        - ``"brain_act_legacy"``: 0.01–0.20 Hz (BOLD infra-slow range)
        - ``"firing_rate"``: override to ``(2.0, 80.0)`` for AdEx/MF, or
          ``(0.05, 1.0)`` for slow TVB whole-brain network states
    filter_order : int, default=3
        Butterworth order. Recommended ``4`` for ``"firing_rate"``
        narrowband use; ``3`` for BOLD legacy.
    dt_ms : float, default=5.0
        Sampling interval in milliseconds. Only used by ``"firing_rate"``.
        Must match the actual resolution of ``x``. For AdEx SNN / mean-field
        outputs from ``prepare_population_rates`` with default settings, this
        is ``5.0`` ms. For raw TVB whole-brain rates, use ``0.1`` ms.
    transient_ms : float, default=500.0
        Duration (ms) of the initial transient to discard before analysis.
        Only used by ``"firing_rate"``. A typical SNN/MF settling time is
        200–500 ms; use 500–2000 ms for whole-brain simulations.

    Returns
    -------
    patterns : np.ndarray
        ``(time_valid, n_edges)`` cosine phase-difference patterns.
    global_sync : np.ndarray
        ``(time_valid,)`` global synchrony trajectory.
    iu : np.ndarray
        Upper-triangle row indices for region pairs.
    ju : np.ndarray
        Upper-triangle column indices for region pairs.
    """
    x = _ensure_2d(x)
    t, n_regions = x.shape
    if n_regions < 2:
        raise ValueError("Need at least 2 regions for phase patterns.")

    # Pipeline-aware default for edge trimming.
    if trim_edge_samples is None:
        trim_edge_samples = 9 if pipeline == "standard" else 0

    if pipeline == "standard":
        # Standard TVBToolkit path — parity with Brain-Act 04_02/04_04.
        xz = zscore(x, axis=0, ddof=1)
        xz = np.nan_to_num(xz, nan=0.0)
        analytic = hilbert(xz, axis=0)
        phase = np.angle(analytic)
        start = max(int(trim_edge_samples), 1)
    elif pipeline == "brain_act_legacy":
        # Legacy Brain-Act parity path — parity with legacy_phase_coherence_new.py.
        xl = _legacy_preprocess(
            x,
            tr_seconds=tr_seconds,
            bandpass_hz=bandpass_hz,
            filter_order=filter_order,
        )
        analytic = hilbert(xl, axis=0)
        phase = np.unwrap(np.angle(analytic), axis=0)
        start = max(int(trim_edge_samples), 0)
    elif pipeline == "firing_rate":
        # Firing-rate pipeline — for AdEx SNN, mean-field, and TVB whole-brain rates.
        # Preprocessing returns a shorter array (transient removed); x is replaced.
        x = _firing_rate_preprocess(
            x,
            dt_ms=dt_ms,
            transient_ms=transient_ms,
            bandpass_hz=bandpass_hz,
            filter_order=filter_order,
        )
        t, n_regions = x.shape  # update t after transient removal
        analytic = hilbert(x, axis=0)
        phase = np.angle(analytic)    # no unwrap — narrowband signal is cyclostationary
        start = max(int(trim_edge_samples), 0)
    else:
        raise ValueError("pipeline must be one of: 'standard', 'brain_act_legacy', 'firing_rate'.")

    iu, ju = np.triu_indices(n_regions, k=1)

    stop = max(t - int(trim_edge_samples), 1)
    valid_idx = np.arange(start, stop)
    if valid_idx.size == 0:
        return np.empty((0, iu.size), dtype=float), np.empty((0,), dtype=float), iu, ju

    patterns = np.empty((valid_idx.size, iu.size), dtype=float)
    global_sync = np.empty(valid_idx.size, dtype=float)

    for k, ti in enumerate(valid_idx):
        global_sync[k] = np.abs(np.mean(np.exp(1j * phase[ti, :])))
        patterns[k] = np.cos(phase[ti, iu] - phase[ti, ju])

    return patterns, global_sync, iu, ju



def _compute_occupancy(labels: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=n_states)
    total = max(int(labels.size), 1)
    return counts.astype(float) / float(total)



def _compute_transition_matrix(
    labels: np.ndarray,
    n_states: int,
    *,
    collapse_runs: bool = False,
    exclude_self: bool = False,
) -> np.ndarray:
    """Row-stochastic transition matrix.

    Parameters
    ----------
    collapse_runs : bool, default=False
        If ``True``, consecutive identical labels are collapsed before counting
        transitions (Brain-Act legacy ``remove_redundancies`` behaviour).
    exclude_self : bool, default=False
        If ``True``, self-transitions are not counted (Brain-Act legacy
        ``markov_transition_no_self`` behaviour). Implies ``collapse_runs`` in
        the legacy script; set both if you need exact parity.
    """
    seq = labels.astype(int)
    if collapse_runs and seq.size > 0:
        out = [seq[0]]
        for v in seq[1:]:
            if v != out[-1]:
                out.append(v)
        seq = np.array(out, dtype=int)

    tm = np.zeros((n_states, n_states), dtype=float)
    if seq.size < 2:
        return tm
    for i in range(seq.size - 1):
        a = int(seq[i])
        b = int(seq[i + 1])
        if exclude_self and a == b:
            continue
        if 0 <= a < n_states and 0 <= b < n_states:
            tm[a, b] += 1.0
    row_sum = tm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return tm / row_sum



def cluster_brain_states(
    patterns: np.ndarray,
    n_states: int = 5,
    *,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    backend: Literal["scipy", "sklearn"] = "sklearn",
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster phase patterns and return ``(labels, centers)``.

    Parameters
    ----------
    patterns : np.ndarray
        ``(time, edges)`` matrix.
    n_states : int, default=5
        Target number of clusters.
    random_seed : int, default=42
        RNG seed. Matches Brain-Act ``04_02``/``04_04`` default (``42``).
        Use ``random_seed=1`` with ``n_init=200`` for Brain-Act legacy parity.
    n_init : int, default=10
        Number of restarts. Matches Brain-Act ``04_02``/``04_04`` default.
        Use ``200`` for Brain-Act legacy parity.
    max_iter : int, default=300
        Max iterations per run. Matches both Brain-Act ``04_02``/``04_04`` and
        Brain-Act legacy (both use sklearn's 300-iteration default).
    backend : {"scipy", "sklearn"}, default="sklearn"
        Clustering backend. ``"sklearn"`` matches Brain-Act behaviour and is
        the recommended default.
    """
    patterns = np.asarray(patterns)
    if patterns.ndim != 2:
        raise ValueError(f"Expected patterns shape (time, edges), got {patterns.shape}")
    if patterns.shape[0] == 0:
        return np.empty((0,), dtype=int), np.empty((0, patterns.shape[1]), dtype=float)

    k_eff = max(1, min(int(n_states), int(patterns.shape[0])))

    if backend == "sklearn":
        if KMeans is None:
            raise ImportError("scikit-learn is required for backend='sklearn'.")
        # Preserve float32/float64 and C-order when possible so disk-backed
        # memmaps can be consumed without an eager float64 copy.
        if patterns.dtype not in (np.float32, np.float64):
            patterns = patterns.astype(np.float32, copy=False)
        if not patterns.flags.c_contiguous:
            patterns = np.ascontiguousarray(patterns)
        km = KMeans(
            n_clusters=k_eff,
            random_state=int(random_seed),
            n_init=max(1, int(n_init)),
            max_iter=max(1, int(max_iter)),
            init="k-means++",
        )
        labels = km.fit_predict(patterns).astype(int)
        centers = np.asarray(km.cluster_centers_, dtype=float)
        return labels, centers

    if backend == "scipy":
        patterns = np.asarray(patterns, dtype=float)
        best_labels = None
        best_centers = None
        best_inertia = np.inf

        rng = np.random.default_rng(random_seed)
        for _ in range(max(1, int(n_init))):
            seed = int(rng.integers(0, 2**31 - 1))
            centers, labels = kmeans2(patterns, k=k_eff, minit="points", iter=max(1, int(max_iter)), seed=seed)
            inertia = float(np.sum((patterns - centers[labels]) ** 2))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers

        labels = np.asarray(best_labels, dtype=int)
        centers = np.asarray(best_centers, dtype=float)
        return labels, centers

    raise ValueError("backend must be one of: 'scipy', 'sklearn'.")



def sfc_sort_centroids(
    centers: np.ndarray,
    labels: np.ndarray,
    sc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort brain-state centroids by ascending SC-FC coupling (Pearson correlation).

    Mirrors Brain-Act legacy ``sfc_sort_and_relabel``: each centroid's upper-triangle
    vector is correlated with the SC upper-triangle, then centroids are ordered from
    least to most SC-like (ascending Pearson r). Labels are relabelled accordingly.

    Parameters
    ----------
    centers : np.ndarray
        ``(k, n_edges)`` centroid matrix from :func:`cluster_brain_states`.
        ``n_edges`` must equal ``n_regions * (n_regions - 1) // 2``.
    labels : np.ndarray
        ``(time,)`` integer cluster labels from :func:`cluster_brain_states`.
    sc : np.ndarray
        ``(n_regions, n_regions)`` structural connectivity matrix.

    Returns
    -------
    centers_sorted : np.ndarray
        Centroids reordered ascending by SFC (``k, n_edges``).
    labels_sorted : np.ndarray
        Labels remapped to the new centroid ordering (``time,``).
    sort_order : np.ndarray
        Integer permutation applied to rows of ``centers`` (``k,``).
    sfc_values : np.ndarray
        Pearson r of each sorted centroid with the SC upper-triangle (``k,``).

    Notes
    -----
    The SC upper-triangle (``k=1``, ``i < j``) is extracted using
    ``np.triu_indices``. The centroid vector is assumed to be in the same
    lexicographic upper-triangle order, as produced by :func:`phase_patterns`.
    """
    sc = np.asarray(sc, dtype=float)
    n_regions = sc.shape[0]
    iu, ju = np.triu_indices(n_regions, k=1)
    sc_vec = sc[iu, ju]

    k = centers.shape[0]
    if centers.shape[1] != sc_vec.size:
        raise ValueError(
            f"centers has {centers.shape[1]} edge features but SC upper-triangle "
            f"has {sc_vec.size} elements (expected n_regions*(n_regions-1)//2 = "
            f"{n_regions * (n_regions - 1) // 2})."
        )

    sfc_raw = np.empty(k, dtype=float)
    for i in range(k):
        r = float(np.corrcoef(centers[i], sc_vec)[0, 1])
        sfc_raw[i] = r

    order = np.argsort(sfc_raw)  # ascending — Brain-Act legacy convention
    inv = np.empty_like(order)
    inv[order] = np.arange(k)

    centers_sorted = centers[order]
    labels_sorted = inv[labels.astype(int)]
    sfc_sorted = sfc_raw[order]

    return centers_sorted, labels_sorted, order, sfc_sorted



def summarize_brain_states(
    x: np.ndarray,
    n_states: int = 5,
    *,
    trim_edge_samples: int | None = None,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    pipeline: Literal["standard", "brain_act_legacy", "firing_rate"] = "standard",
    clustering_backend: Literal["scipy", "sklearn"] | None = None,
    tr_seconds: float = 2.4,
    bandpass_hz: tuple[float, float] = (0.01, 0.20),
    filter_order: int = 3,
    collapse_runs: bool = False,
    exclude_self: bool = False,
    dt_ms: float = 5.0,
    transient_ms: float = 500.0,
) -> BrainStateSummary:
    """Run complete phase-pattern brain-state summarisation for one time series.

    Parameters
    ----------
    trim_edge_samples : int | None, default=None
        Passed to :func:`phase_patterns`. When ``None``, defaults to ``9``
        for ``"standard"``, ``0`` for ``"brain_act_legacy"`` and
        ``"firing_rate"``.
    random_seed : int, default=42
        Matches Brain-Act ``04_02``/``04_04``. Use ``1`` for legacy parity.
    n_init : int, default=10
        Matches Brain-Act ``04_02``/``04_04``. Use ``200`` for legacy parity.
    max_iter : int, default=300
        Matches both Brain-Act scripts.
    collapse_runs : bool, default=False
        If ``True``, transition matrix is computed after collapsing consecutive
        identical labels (Brain-Act legacy behaviour).
    exclude_self : bool, default=False
        If ``True``, self-transitions are excluded from the transition matrix
        (Brain-Act legacy ``markov_transition_no_self`` behaviour).
    dt_ms : float, default=5.0
        Firing-rate pipeline only. Sampling interval in milliseconds.
        Set to match the actual resolution of ``x`` (e.g. ``5.0`` for
        AdEx/MF default bin width; ``0.1`` for raw TVB whole-brain rates).
    transient_ms : float, default=500.0
        Firing-rate pipeline only. Initial duration (ms) to discard.

    Notes
    -----
    For Brain-Act legacy parity use:
    ``pipeline='brain_act_legacy'``, ``random_seed=1``, ``n_init=200``,
    ``collapse_runs=True``, ``exclude_self=True``.

    For AdEx SNN / mean-field firing rates use:
    ``pipeline='firing_rate'``, ``bandpass_hz=(2.0, 80.0)``,
    ``filter_order=4``, ``dt_ms=5.0``, ``transient_ms=500.0``.

    For slow TVB whole-brain network states use:
    ``pipeline='firing_rate'``, ``bandpass_hz=(0.05, 1.0)``,
    ``filter_order=4``, ``dt_ms=<downsample_dt_ms>``.
    """
    if clustering_backend is None:
        clustering_backend = "sklearn"

    patterns, global_sync, iu, ju = phase_patterns(
        x,
        trim_edge_samples=trim_edge_samples,
        pipeline=pipeline,
        tr_seconds=tr_seconds,
        bandpass_hz=bandpass_hz,
        filter_order=filter_order,
        dt_ms=dt_ms,
        transient_ms=transient_ms,
    )
    labels, centers = cluster_brain_states(
        patterns,
        n_states=n_states,
        random_seed=random_seed,
        n_init=n_init,
        max_iter=max_iter,
        backend=clustering_backend,
    )

    n_states_eff = int(centers.shape[0]) if centers.ndim == 2 else 0
    occupancy = _compute_occupancy(labels, n_states=max(1, n_states_eff))
    transitions = _compute_transition_matrix(
        labels,
        n_states=max(1, n_states_eff),
        collapse_runs=collapse_runs,
        exclude_self=exclude_self,
    )

    return BrainStateSummary(
        labels=labels,
        centers=centers,
        occupancy=occupancy,
        transition_matrix=transitions,
        global_synchrony=np.asarray(global_sync, dtype=float),
        edge_index_i=np.asarray(iu, dtype=int),
        edge_index_j=np.asarray(ju, dtype=int),
    )



def centers_to_matrices(summary: BrainStateSummary, n_regions: int) -> np.ndarray:
    """Convert flattened upper-triangle centers back to square matrices.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_states, n_regions, n_regions)``.
    """
    if summary.centers.size == 0:
        return np.empty((0, n_regions, n_regions), dtype=float)

    mats = []
    for row in summary.centers:
        mats.append(squareform(row))
    return np.stack(mats, axis=0)



def brain_state_metrics_dict(summary: BrainStateSummary) -> dict[str, Any]:
    """Convert summary dataclass to plain numpy dictionary for saving."""
    return {
        "brain_state_labels": np.asarray(summary.labels, dtype=int),
        "brain_state_centers": np.asarray(summary.centers, dtype=float),
        "brain_state_occupancy": np.asarray(summary.occupancy, dtype=float),
        "brain_state_transition_matrix": np.asarray(summary.transition_matrix, dtype=float),
        "global_synchrony": np.asarray(summary.global_synchrony, dtype=float),
        "edge_index_i": np.asarray(summary.edge_index_i, dtype=int),
        "edge_index_j": np.asarray(summary.edge_index_j, dtype=int),
    }
