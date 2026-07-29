"""Casali-style PCI helper routines.

This module provides a minimal, self-contained port of the routines required by
TVBSim's PCI pipeline:

- ``binarise_signals``
- ``sort_binJ``
- ``source_entropy``
- ``lz_complexity_2d``
- ``pci_norm_factor``

It also provides ``binarise_signals_casali``, whose production default is an
independent implementation of the within-trial pre/post-block permutation
described by Pantazis et al. (2005) and adopted for source-level TMS/EEG
analysis by Casali et al. (2010). The older TVBSim and baseline-only nulls are
retained for explicit sensitivity analyses.

Attribution
-----------
Ported/adapted from TVBSim reference implementations:

- ``tvbsim/TVB/pci_v2.py``
- ``tvbsim/TVB/tvb_model_reference/src/nuu_tools_simulation_human.py``

Original TVBSim PCI implementation based on Casali et al. (2013).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class CasaliSignificanceResult:
    """Result of trial-averaged source-significance estimation.

    The compatibility fields at the start of this dataclass are retained for
    existing callers.  Canonical pre/post permutations additionally expose the
    time-resolved thresholds, corrected P values, normalization statistics and
    randomization provenance required to audit the inferential result.
    """

    binary: np.ndarray
    averaged_response: np.ndarray
    threshold: float
    null_maxima: np.ndarray
    significance_method: str
    n_surrogates: int
    alpha: float
    two_sided: bool
    threshold_by_time: np.ndarray
    corrected_p_values: np.ndarray
    observed_statistic: np.ndarray
    baseline_mean: np.ndarray
    baseline_sd: np.ndarray
    fwer_scope: str
    seed: int | None
    rng_bit_generator: str
    swap_matrix_sha256: str | None
    swap_fraction: np.ndarray
    n_trials: int
    n_sources: int
    n_pre_bins: int
    n_post_bins: int
    chunk_size: int
    quantile_method: str
    active_count: int
    active_fraction: float
    entropy: float
    below_one_percent_activation: bool


def _ensure_binary_2d(x: np.ndarray) -> np.ndarray:
    """Return a 2D binary matrix with shape ``(channels, time)``.

    Parameters
    ----------
    x : np.ndarray
        Input matrix expected to be binary-like (0/1 or bool).

    Returns
    -------
    np.ndarray
        ``uint8`` array with values in ``{0, 1}``.
    """
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix (channels, time), got shape {arr.shape}.")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8, copy=False)
    if np.issubdtype(arr.dtype, np.number):
        return (arr > 0).astype(np.uint8)
    raise TypeError("Binary matrix must be numeric or boolean.")



def sort_binJ(binJ: np.ndarray) -> np.ndarray:
    """Sort binary channel matrix by descending channel activation.

    This mirrors Casali/TVBSim ordering where channels are ranked by the number
    of active bins before Lempel-Ziv complexity evaluation.

    Parameters
    ----------
    binJ : np.ndarray
        Binary matrix with shape ``(channels, time)``.

    Returns
    -------
    np.ndarray
        Sorted binary matrix with the same shape.
    """
    b = _ensure_binary_2d(binJ)
    rank = np.sum(b, axis=1).argsort()[::-1]
    return b[rank, :]



def source_entropy(binJ: np.ndarray) -> float:
    """Compute source entropy used in Casali-style PCI diagnostics.

    Parameters
    ----------
    binJ : np.ndarray
        Binary matrix with shape ``(channels, time)``.

    Returns
    -------
    float
        Shannon entropy of the Bernoulli source distribution over all entries.
    """
    b = _ensure_binary_2d(binJ)
    total = float(b.size)
    if total <= 0:
        return 0.0

    p1 = float(np.sum(b == 1)) / total
    p0 = 1.0 - p1
    if p0 * p1:
        return float(-p1 * np.log2(p1) - p0 * np.log2(p0))
    return 0.0



def pci_norm_factor(binJ: np.ndarray) -> float:
    """Compute Casali normalization factor for PCI.

    Parameters
    ----------
    binJ : np.ndarray
        Binary matrix with shape ``(channels, time)``.

    Returns
    -------
    float
        Normalization term ``S = (L * H) / log2(L)`` with
        ``L = channels * time`` and Bernoulli entropy ``H``.
    """
    b = _ensure_binary_2d(binJ)
    L = int(b.shape[0] * b.shape[1])
    if L <= 1:
        return 0.0

    p1 = float(np.sum(b == 1)) / float(L)
    p0 = 1.0 - p1
    if p0 * p1:
        H = -p1 * np.log2(p1) - p0 * np.log2(p0)
    else:
        H = 0.0
    return float((L * H) / np.log2(L))



def lz_complexity_2d(binJ: np.ndarray) -> int:
    """Compute 2D Lempel-Ziv complexity used by TVBSim Casali PCI.

    Notes
    -----
    This is a faithful structural port of the TVBSim routine from
    ``pci_v2.lz_complexity_2D`` with binary substring search performed on byte
    sequences (rather than `bitarray`) to avoid additional dependencies.

    Parameters
    ----------
    binJ : np.ndarray
        Binary matrix with shape ``(channels, time)``.

    Returns
    -------
    int
        2D Lempel-Ziv complexity count.
    """
    D = _ensure_binary_2d(binJ)
    if D.size == 0:
        return 0

    L1, L2 = D.shape
    if L1 <= 0 or L2 <= 0:
        return 0

    # Reference initial state in TVBSim implementation.
    c = 1
    r = 1
    q = 1
    k = 1
    i = 1
    stop = False

    # Each time-column encoded as bytes over channels.
    cols = [bytes(D[:, y].tolist()) for y in range(L2)]

    def _end_of_column(r_: int, c_: int, i_: int, q_: int, k_: int, stop_: bool):
        r_ += 1
        if r_ > L2:
            c_ += 1
            stop_ = True
        else:
            i_ = 0
            q_ = r_ - 1
            k_ = 1
        return r_, c_, i_, q_, k_, stop_

    while not stop:
        if q == r:
            a = i + k - 1
        else:
            a = L1

        haystack = cols[q - 1][0:a]
        needle = cols[r - 1][i : i + k]
        found = haystack.find(needle) != -1

        if found:
            k += 1
            if i + k > L1:
                r, c, i, q, k, stop = _end_of_column(r, c, i, q, k, stop)
        else:
            q -= 1
            if q < 1:
                c += 1
                i = i + k
                if i + 1 > L1:
                    r, c, i, q, k, stop = _end_of_column(r, c, i, q, k, stop)
                else:
                    q = r
                    k = 1

    return int(c)



def binarise_signals(
    signal_m: np.ndarray,
    t_stim: int,
    nshuffles: int = 10,
    percentile: float = 100.0,
) -> np.ndarray:
    """Binarize trial signals using baseline-centred surrogate thresholding.

    This ports TVBSim's ``binarise_signals`` logic used before Casali PCI
    calculation.

    Parameters
    ----------
    signal_m : np.ndarray
        Real-valued signal with shape ``(n_trials, n_sources, n_bins)``.
    t_stim : int
        Stimulation onset in **bins** within each trial window.
        Baseline is ``:t_stim``.
    nshuffles : int, default=10
        Number of surrogate shuffles for baseline threshold estimation.
    percentile : float, default=100.0
        TVBSim-style percentile parameter in threshold indexing:
        ``signalThresh = sorted_surrogates[-int(nshuffles / percentile)]``.

    Returns
    -------
    np.ndarray
        Boolean array of same shape as ``signal_m``.

    Notes
    -----
    - Axis conventions follow TVBSim exactly: ``(trial, source, time)``.
    - ``t_stim`` must already be in bins (not milliseconds).
    """
    s = np.asarray(signal_m, dtype=float)
    if s.ndim != 3:
        raise ValueError(
            f"Expected shape (n_trials, n_sources, n_bins), got {s.shape}."
        )
    if nshuffles < 1:
        raise ValueError("nshuffles must be >= 1.")
    if percentile <= 0:
        raise ValueError("percentile must be > 0.")

    n_trials, n_sources, n_bins = s.shape
    if not (1 <= t_stim < n_bins):
        raise ValueError(
            f"t_stim must be in [1, n_bins-1]. Got t_stim={t_stim}, n_bins={n_bins}."
        )

    means_prestim = np.mean(s[:, :, :t_stim], axis=2)
    means_safe = np.where(np.abs(means_prestim) < _EPS, _EPS, means_prestim)

    signal_centre = s / means_safe[:, :, np.newaxis] - 1.0

    std_prestim = np.std(signal_centre[:, :, :t_stim], axis=2)
    std_safe = np.where(std_prestim < _EPS, 1.0, std_prestim)
    signal_centre_norm = signal_centre / std_safe[:, :, np.newaxis]

    signal_prestim_shuffle = signal_centre_norm[:, :, :t_stim].copy()
    max_absval_surrogates = np.zeros(int(nshuffles), dtype=float)

    for i_shuffle in range(int(nshuffles)):
        for i_source in range(n_sources):
            for i_trial in range(n_trials):
                signal_curr = signal_prestim_shuffle[i_trial, i_source]
                np.random.shuffle(signal_curr)
                signal_prestim_shuffle[i_trial, i_source] = signal_curr

        shuffle_avg = np.mean(signal_prestim_shuffle, axis=0)
        max_absval_surrogates[i_shuffle] = np.max(np.abs(shuffle_avg))

    max_sorted = np.sort(max_absval_surrogates)
    threshold_index = -int(nshuffles / percentile)
    signal_thresh = max_sorted[threshold_index]

    return signal_centre_norm > signal_thresh


def _source_baseline_statistics(
    signal: np.ndarray,
    t_stim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled prestimulus mean and sample SD for every source."""
    baseline = signal[:, :, :t_stim]
    n_observations = int(baseline.shape[0] * baseline.shape[2])
    if n_observations < 2:
        raise ValueError(
            "At least two prestimulus observations are required to estimate "
            "source variability."
        )

    mean = baseline.mean(axis=(0, 2))
    centred = baseline - mean[np.newaxis, :, np.newaxis]
    sd = np.sqrt(
        np.square(centred).sum(axis=(0, 2)) / float(n_observations - 1)
    )
    source_scale = np.max(np.abs(centred), axis=(0, 2))
    tolerance = (
        32.0
        * np.finfo(float).eps
        * np.maximum(source_scale, np.finfo(float).tiny)
    )
    invalid = (~np.isfinite(sd)) | (sd <= tolerance)
    if np.any(invalid):
        indices = np.flatnonzero(invalid)
        preview = ", ".join(str(int(index)) for index in indices[:12])
        suffix = "..." if indices.size > 12 else ""
        raise ValueError(
            "Prestimulus SD is zero or numerically unresolved for source "
            f"indices [{preview}{suffix}]. Statistical normalization is "
            "undefined; add stochastic baseline variability or remove the "
            "invalid sources explicitly."
        )
    return mean, sd


def _swap_matrix_sha256(swap_matrix: np.ndarray) -> str:
    """Return a stable digest of a trial-swap schedule and its shape."""
    swaps = np.asarray(swap_matrix, dtype=np.uint8, order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(swaps.shape, dtype="<i8").tobytes())
    digest.update(swaps.tobytes(order="C"))
    return digest.hexdigest()


def _timewise_corrected_p_values(
    observed: np.ndarray,
    null_maxima: np.ndarray,
) -> np.ndarray:
    """Compute plus-one Monte Carlo P values corrected across sources."""
    obs = np.asarray(observed, dtype=float)
    maxima = np.asarray(null_maxima, dtype=float)
    if obs.ndim != 2 or maxima.ndim != 2 or obs.shape[1] != maxima.shape[1]:
        raise ValueError(
            "Expected observed=(sources,time) and null_maxima=(surrogates,time)."
        )
    n_surrogates = int(maxima.shape[0])
    corrected = np.empty(obs.shape, dtype=float)
    for time_index in range(obs.shape[1]):
        comparison_scale = max(
            1.0,
            float(np.max(np.abs(maxima[:, time_index]))),
            float(np.max(np.abs(obs[:, time_index]))),
        )
        # Affine source transformations can introduce a few dozen ulps while
        # baseline means are subtracted. Treat numerically identical
        # permutation statistics as ties so their Monte Carlo ranks are stable.
        tie_tolerance = 512.0 * np.finfo(float).eps * comparison_scale
        exceedances = np.count_nonzero(
            maxima[:, time_index, np.newaxis]
            >= obs[np.newaxis, :, time_index] - tie_tolerance,
            axis=0,
        )
        corrected[:, time_index] = (1.0 + exceedances) / float(
            n_surrogates + 1
        )
    return corrected


def _global_corrected_p_values(
    observed: np.ndarray,
    null_maxima: np.ndarray,
) -> np.ndarray:
    """Compute plus-one Monte Carlo P values against a global maximum null."""
    obs = np.asarray(observed, dtype=float)
    maxima = np.asarray(null_maxima, dtype=float).reshape(-1)
    corrected = np.empty(obs.shape, dtype=float)
    for time_index in range(obs.shape[1]):
        comparison_scale = max(
            1.0,
            float(np.max(np.abs(maxima))),
            float(np.max(np.abs(obs[:, time_index]))),
        )
        tie_tolerance = 512.0 * np.finfo(float).eps * comparison_scale
        exceedances = np.count_nonzero(
            maxima[:, np.newaxis]
            >= obs[np.newaxis, :, time_index] - tie_tolerance,
            axis=0,
        )
        corrected[:, time_index] = (1.0 + exceedances) / float(
            maxima.size + 1
        )
    return corrected


def binarise_signals_casali(
    signal_m: np.ndarray,
    t_stim: int,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.01,
    two_sided: bool = True,
    seed: int | None = 0,
    significance_method: str = "pre_post_swap",
    single_trial: str = "raise",
    return_details: bool = False,
    chunk_size: int = 64,
    swap_matrix: np.ndarray | None = None,
) -> np.ndarray | CasaliSignificanceResult:
    """Estimate significant activity in one trial-averaged source response.

    The production default implements the explicitly documented
    Casali/Pantazis source-significance randomization:

    1. retain equal-duration prestimulus and poststimulus blocks;
    2. independently decide, for every trial, whether those two complete
       blocks are exchanged (one decision shared by all sources and times);
    3. recompute the permuted prestimulus mean and SD;
    4. average trials and normalize the response by its prestimulus standard
       error;
    5. at each response time, retain the largest absolute statistic over
       sources; and
    6. threshold with plus-one Monte Carlo P values at ``alpha``.

    This controls the family-wise error rate over sources separately at each
    response time (Pantazis permutation method 3).  It preserves spatial and
    temporal covariance within a trial, unlike independently shuffling every
    source trace.

    ``"temporal_shuffle"`` and ``"trial_bootstrap"`` retain the previous
    baseline-only nulls as explicit sensitivity analyses.  Neither should be
    described as the canonical Casali/Pantazis pre/post permutation.

    Parameters
    ----------
    signal_m : np.ndarray
        Real-valued data with shape ``(n_trials, n_sources, n_bins)``.
    t_stim : int
        Stimulation onset in bins. For ``"pre_post_swap"``, the input must have
        exactly ``2 * t_stim`` bins so the two exchanged blocks are equal.
    n_bootstrap : int, default=1000
        Historical API name for the number of Monte Carlo permutations.
    alpha : float, default=0.01
        Family-wise significance level.
    two_sided : bool, default=True
        Use absolute statistics when true.
    seed : int or None, default=0
        Seed for a dedicated PCG64 generator. The same seed and trial order
        reproduce the same swap schedule.
    significance_method : {"pre_post_swap", "temporal_shuffle",
        "trial_bootstrap"}, default="pre_post_swap"
        Null construction. The latter two choices are sensitivity analyses.
    single_trial : {"raise", "baseline_resample"}, default="raise"
        A single trial is rejected by default. The explicit fallback uses the
        legacy baseline-resampling sensitivity analysis.
    return_details : bool, default=False
        Return :class:`CasaliSignificanceResult` instead of only ``binary``.
    chunk_size : int, default=64
        Number of pre/post swap schedules evaluated together.
    swap_matrix : np.ndarray, optional
        Prescribed Boolean matrix with shape ``(n_permutations, n_trials)``.
        This is primarily intended for exact reference tests and audited
        reruns. When provided, its row count supersedes ``n_bootstrap``.

    Returns
    -------
    np.ndarray or CasaliSignificanceResult
        A single ``uint8`` source-by-time matrix. For the canonical method,
        prestimulus entries are zero and only poststimulus entries contain
        inferential decisions.

    Notes
    -----
    This is exact to the explicit source-level randomization described by
    Pantazis et al. and adopted by Casali et al. (2010). The historic 2013 PCI
    supplementary MATLAB implementation is not publicly inspectable here, so
    byte-for-byte equivalence to that software is not asserted.
    """
    s = np.asarray(signal_m, dtype=float)
    if s.ndim == 2:
        s = s[np.newaxis, :, :]
    if s.ndim != 3:
        raise ValueError(
            "Expected (n_trials, n_sources, n_bins) or "
            f"(n_sources, n_bins), got {s.shape}."
        )
    if not np.all(np.isfinite(s)):
        raise ValueError("signal_m contains NaN or infinite values.")

    n_trials, n_sources, n_bins = (int(value) for value in s.shape)
    if n_sources < 1:
        raise ValueError("signal_m must contain at least one source.")
    if not (1 <= int(t_stim) < n_bins):
        raise ValueError(
            f"t_stim must be in [1, n_bins-1]. Got {t_stim}, n_bins={n_bins}."
        )
    t_stim = int(t_stim)
    n_surrogates_requested = int(n_bootstrap)
    if n_surrogates_requested < 1 or n_surrogates_requested != n_bootstrap:
        raise ValueError("n_bootstrap must be an integer >= 1.")
    if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be finite and between 0 and 1.")
    if int(chunk_size) < 1 or int(chunk_size) != chunk_size:
        raise ValueError("chunk_size must be an integer >= 1.")
    chunk_size = int(chunk_size)

    aliases = {
        "pre_post_swap": "pre_post_swap",
        "pre_post_permutation": "pre_post_swap",
        "within_trial_pre_post": "pre_post_swap",
        "casali": "pre_post_swap",
        "canonical": "pre_post_swap",
        "temporal_shuffle": "temporal_shuffle",
        "prestimulus_shuffle": "temporal_shuffle",
        "time_shuffle": "temporal_shuffle",
        "trial_bootstrap": "trial_bootstrap",
        "bootstrap_trials": "trial_bootstrap",
    }
    requested_method = (
        str(significance_method).strip().lower().replace("-", "_")
    )
    if requested_method not in aliases:
        raise ValueError(
            "significance_method must be 'pre_post_swap', "
            "'temporal_shuffle', or 'trial_bootstrap'. "
            f"Got {significance_method!r}."
        )
    method_key = aliases[requested_method]

    single_trial_key = str(single_trial).strip().lower()
    if n_trials == 1 and single_trial_key in ("raise", "error", "strict"):
        raise ValueError(
            "Casali/Pantazis significance estimation requires more than one "
            "trial. Pass trial-level data, or set "
            "single_trial='baseline_resample' for a non-canonical sensitivity "
            "analysis."
        )
    if n_trials == 1 and single_trial_key != "baseline_resample":
        raise ValueError(
            "single_trial must be 'raise' or 'baseline_resample'. "
            f"Got {single_trial!r}."
        )
    if n_trials == 1:
        method_key = "single_trial_baseline_resample"

    if swap_matrix is not None and method_key != "pre_post_swap":
        raise ValueError(
            "swap_matrix is only valid with significance_method='pre_post_swap'."
        )
    if method_key == "pre_post_swap" and n_bins != 2 * t_stim:
        raise ValueError(
            "pre_post_swap requires equal prestimulus and poststimulus "
            f"durations: expected n_bins={2 * t_stim}, got {n_bins}."
        )

    baseline_mean, baseline_sd = _source_baseline_statistics(s, t_stim)
    centred = s - baseline_mean[np.newaxis, :, np.newaxis]
    averaged_response = centred.mean(axis=0) / baseline_sd[:, np.newaxis]
    rng = np.random.Generator(np.random.PCG64(seed))
    rng_name = type(rng.bit_generator).__name__

    if method_key == "pre_post_swap":
        if swap_matrix is None:
            swaps = rng.integers(
                0,
                2,
                size=(n_surrogates_requested, n_trials),
                dtype=np.uint8,
            ).astype(bool)
        else:
            raw_swaps = np.asarray(swap_matrix)
            if raw_swaps.ndim != 2 or raw_swaps.shape[1] != n_trials:
                raise ValueError(
                    "swap_matrix must have shape "
                    f"(n_permutations, {n_trials}), got {raw_swaps.shape}."
                )
            if raw_swaps.shape[0] < 1 or not np.all(
                (raw_swaps == 0) | (raw_swaps == 1)
            ):
                raise ValueError(
                    "swap_matrix must contain at least one row and only 0/1 "
                    "or Boolean values."
                )
            swaps = raw_swaps.astype(bool, copy=True)

        n_surrogates = int(swaps.shape[0])
        n_post_bins = int(n_bins - t_stim)
        n_baseline_observations = int(n_trials * t_stim)
        sqrt_trials = float(np.sqrt(n_trials))
        pre = centred[:, :, :t_stim]
        post = centred[:, :, t_stim:]

        observed_statistic = (
            averaged_response * sqrt_trials
        )
        observed_post = observed_statistic[:, t_stim:]
        observed_for_test = (
            np.abs(observed_post) if two_sided else observed_post
        )

        pre_sum_by_trial = pre.sum(axis=2)
        post_sum_by_trial = post.sum(axis=2)
        pre_square_sum_by_trial = np.square(pre).sum(axis=2)
        post_square_sum_by_trial = np.square(post).sum(axis=2)
        pre_sum = pre_sum_by_trial.sum(axis=0)
        pre_square_sum = pre_square_sum_by_trial.sum(axis=0)
        pre_sum_delta = post_sum_by_trial - pre_sum_by_trial
        pre_square_sum_delta = (
            post_square_sum_by_trial - pre_square_sum_by_trial
        )
        post_mean = post.mean(axis=0)
        swapped_post_delta = np.ascontiguousarray(
            (pre - post).reshape(n_trials, n_sources * n_post_bins)
        )

        null_maxima = np.empty(
            (n_surrogates, n_post_bins),
            dtype=float,
        )
        for start in range(0, n_surrogates, chunk_size):
            stop = min(start + chunk_size, n_surrogates)
            weights = swaps[start:stop].astype(float, copy=False)

            permuted_pre_sum = (
                pre_sum[np.newaxis, :] + weights @ pre_sum_delta
            )
            permuted_pre_square_sum = (
                pre_square_sum[np.newaxis, :]
                + weights @ pre_square_sum_delta
            )
            permuted_pre_mean = (
                permuted_pre_sum / float(n_baseline_observations)
            )
            variance_numerator = (
                permuted_pre_square_sum
                - float(n_baseline_observations)
                * np.square(permuted_pre_mean)
            )
            variance_tolerance = (
                64.0
                * np.finfo(float).eps
                * np.maximum(
                    np.abs(permuted_pre_square_sum),
                    np.finfo(float).tiny,
                )
            )
            invalid_variance = (
                (~np.isfinite(variance_numerator))
                | (variance_numerator <= variance_tolerance)
            )
            if np.any(invalid_variance):
                local_permutation, source_index = np.argwhere(
                    invalid_variance
                )[0]
                raise ValueError(
                    "A pre/post permutation produced zero or numerically "
                    "unresolved prestimulus SD at permutation "
                    f"{start + int(local_permutation)}, source "
                    f"{int(source_index)}."
                )
            permuted_pre_sd = np.sqrt(
                variance_numerator
                / float(n_baseline_observations - 1)
            )

            permuted_post_mean = (
                post_mean[np.newaxis, :, :]
                + (
                    weights @ swapped_post_delta
                ).reshape(stop - start, n_sources, n_post_bins)
                / float(n_trials)
            )
            permuted_statistic = (
                permuted_post_mean
                - permuted_pre_mean[:, :, np.newaxis]
            ) / (
                permuted_pre_sd[:, :, np.newaxis] / sqrt_trials
            )
            values = (
                np.abs(permuted_statistic)
                if two_sided
                else permuted_statistic
            )
            null_maxima[start:stop] = values.max(axis=1)

        threshold_by_time = np.quantile(
            null_maxima,
            1.0 - float(alpha),
            axis=0,
            method="higher",
        )
        corrected_post = _timewise_corrected_p_values(
            observed_for_test,
            null_maxima,
        )
        binary = np.zeros((n_sources, n_bins), dtype=np.uint8)
        binary[:, t_stim:] = (corrected_post <= float(alpha)).astype(
            np.uint8
        )
        corrected_p_values = np.ones((n_sources, n_bins), dtype=float)
        corrected_p_values[:, t_stim:] = corrected_post
        threshold = float(np.max(threshold_by_time))
        fwer_scope = "sources_per_response_time"
        swap_digest = _swap_matrix_sha256(swaps)
        swap_fraction = swaps.mean(axis=1, dtype=float)
    else:
        # Explicit sensitivity analyses retained for backwards comparison.
        trial_base_mean = s[:, :, :t_stim].mean(axis=2, keepdims=True)
        trial_centred = s - trial_base_mean
        sensitivity_sd = np.sqrt(
            np.square(trial_centred[:, :, :t_stim]).mean(axis=(0, 2))
        )
        sensitivity_scale = np.max(
            np.abs(trial_centred[:, :, :t_stim]),
            axis=(0, 2),
        )
        sensitivity_tolerance = (
            32.0
            * np.finfo(float).eps
            * np.maximum(sensitivity_scale, np.finfo(float).tiny)
        )
        invalid = sensitivity_sd <= sensitivity_tolerance
        if np.any(invalid):
            indices = ", ".join(
                str(int(index)) for index in np.flatnonzero(invalid)[:12]
            )
            raise ValueError(
                "Prestimulus SD is zero or numerically unresolved for "
                f"sensitivity-analysis source indices [{indices}]."
            )
        normalized = (
            trial_centred
            / sensitivity_sd[np.newaxis, :, np.newaxis]
        )
        averaged_response = normalized.mean(axis=0)
        observed_statistic = averaged_response.copy()
        baseline_only = normalized[:, :, :t_stim]
        null_maxima = np.empty(n_surrogates_requested, dtype=float)
        for surrogate_index in range(n_surrogates_requested):
            if method_key == "single_trial_baseline_resample":
                indices = rng.integers(0, t_stim, t_stim)
                surrogate = baseline_only[0, :, indices]
            elif method_key == "temporal_shuffle":
                surrogate = rng.permuted(
                    baseline_only,
                    axis=2,
                ).mean(axis=0)
            else:
                indices = rng.integers(0, n_trials, n_trials)
                surrogate = baseline_only[indices].mean(axis=0)
            values = np.abs(surrogate) if two_sided else surrogate
            null_maxima[surrogate_index] = float(values.max())

        threshold = float(
            np.quantile(
                null_maxima,
                1.0 - float(alpha),
                method="higher",
            )
        )
        observed_for_test = (
            np.abs(observed_statistic)
            if two_sided
            else observed_statistic
        )
        corrected_p_values = _global_corrected_p_values(
            observed_for_test,
            null_maxima,
        )
        binary = (corrected_p_values <= float(alpha)).astype(np.uint8)
        threshold_by_time = np.full(n_bins, threshold, dtype=float)
        n_surrogates = n_surrogates_requested
        n_post_bins = int(n_bins - t_stim)
        fwer_scope = "sources_and_prestimulus_time_global"
        swap_digest = None
        swap_fraction = np.empty(0, dtype=float)

    binary_post = binary[:, t_stim:]
    active_count = int(binary_post.sum())
    active_fraction = (
        float(binary_post.mean()) if binary_post.size else 0.0
    )
    entropy = float(source_entropy(binary_post))
    result = CasaliSignificanceResult(
        binary=binary,
        averaged_response=averaged_response,
        threshold=threshold,
        null_maxima=null_maxima,
        significance_method=method_key,
        n_surrogates=int(n_surrogates),
        alpha=float(alpha),
        two_sided=bool(two_sided),
        threshold_by_time=np.asarray(threshold_by_time, dtype=float),
        corrected_p_values=corrected_p_values,
        observed_statistic=observed_statistic,
        baseline_mean=baseline_mean,
        baseline_sd=baseline_sd,
        fwer_scope=fwer_scope,
        seed=None if seed is None else int(seed),
        rng_bit_generator=rng_name,
        swap_matrix_sha256=swap_digest,
        swap_fraction=np.asarray(swap_fraction, dtype=float),
        n_trials=n_trials,
        n_sources=n_sources,
        n_pre_bins=t_stim,
        n_post_bins=n_post_bins,
        chunk_size=chunk_size,
        quantile_method="higher",
        active_count=active_count,
        active_fraction=active_fraction,
        entropy=entropy,
        below_one_percent_activation=bool(active_fraction < 0.01),
    )
    if return_details:
        return result
    return result.binary


def binarise(
    signal_m: np.ndarray,
    t_stim: int,
    *,
    method: str = "tvbsim",
    **kwargs,
) -> np.ndarray:
    """Binarize a continuous-valued signal via a selectable route.

    Parameters
    ----------
    signal_m : np.ndarray
        Real-valued signal, ``(n_trials, n_sources, n_bins)``.
    t_stim : int
        Stimulation onset in bins.
    method : {"tvbsim", "casali"}, default="tvbsim"
        - ``"tvbsim"`` : :func:`binarise_signals` — the existing shuffle-based
          route (per-trial output). Accepts ``nshuffles``, ``percentile``.
        - ``"casali"`` : :func:`binarise_signals_casali` — a single
          trial-averaged output using the within-trial pre/post permutation by
          default. Accepts ``n_bootstrap``, ``alpha``, ``two_sided``, ``seed``,
          ``significance_method``, ``chunk_size`` and ``swap_matrix``.
    **kwargs
        Forwarded to the selected route.

    Returns
    -------
    np.ndarray
        Binary matrix. Shape is route-dependent:
        ``(n_trials, n_sources, n_bins)`` for ``"tvbsim"``;
        ``(n_sources, n_bins)`` for ``"casali"``.
    """
    key = method.lower()
    if key in ("tvbsim", "shuffle", "current"):
        return binarise_signals(signal_m, t_stim, **kwargs)
    if key in ("casali", "paper", "bootstrap"):
        return binarise_signals_casali(signal_m, t_stim, **kwargs)
    raise ValueError(f"Unknown binarisation method {method!r}; use 'tvbsim' or 'casali'.")


__all__ = [
    "CasaliSignificanceResult",
    "binarise_signals",
    "binarise_signals_casali",
    "binarise",
    "sort_binJ",
    "source_entropy",
    "lz_complexity_2d",
    "pci_norm_factor",
]
