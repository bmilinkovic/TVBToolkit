"""Casali-style PCI helper routines.

This module provides a minimal, self-contained port of the routines required by
TVBSim's PCI pipeline:

- ``binarise_signals``
- ``sort_binJ``
- ``source_entropy``
- ``lz_complexity_2d``
- ``pci_norm_factor``

Attribution
-----------
Ported/adapted from TVBSim reference implementations:

- ``tvbsim/TVB/pci_v2.py``
- ``tvbsim/TVB/tvb_model_reference/src/nuu_tools_simulation_human.py``

Original TVBSim PCI implementation based on Casali et al. (2013).
"""

from __future__ import annotations

import numpy as np


_EPS = np.finfo(float).eps


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


def binarise_signals_casali(
    signal_m: np.ndarray,
    t_stim: int,
    *,
    n_bootstrap: int = 500,
    alpha: float = 0.01,
    two_sided: bool = True,
    seed: int | None = 0,
    single_trial: str = "raise",
) -> np.ndarray:
    """Binarize signals with the canonical Casali et al. (2013) procedure.

    This is the method used to produce the empirical ``binJ`` matrices stored in
    the tDCS/TMS-EEG dataset (``soglia = alpha = 0.01``). It differs from
    :func:`binarise_signals` (the TVBSim/shuffle route) in five ways:

    1. **Baseline subtraction then per-source z-scoring** by each source's own
       baseline SD (significance in units of baseline variability), not relative
       change to the baseline mean.
    2. **Bootstrap** trial resampling for the null, not a temporal shuffle.
    3. **Two-sided** significance (``|response| > thresh``), not one-sided.
    4. Operates on the **trial-averaged** response → a single ``binJ``.
    5. Threshold = the ``(1 - alpha)`` percentile of the bootstrap **max
       statistic** over sources × baseline-time (multiple-comparison control).

    Parameters
    ----------
    signal_m : np.ndarray
        Real-valued signal, shape ``(n_trials, n_sources, n_bins)``. A 2D
        ``(n_sources, n_bins)`` array is treated as a single trial, which is
        rejected by default because the canonical bootstrap needs trials.
    t_stim : int
        Stimulation onset in **bins**; baseline is ``:t_stim``.
    n_bootstrap : int, default=500
        Number of bootstrap resamples for the null distribution.
    alpha : float, default=0.01
        Significance level (Casali ``soglia``); threshold is the
        ``100 * (1 - alpha)`` percentile of the bootstrap max statistic.
    two_sided : bool, default=True
        If ``True`` threshold ``|response|`` (captures negative deflections).
    seed : int or None, default=0
        Seed for the bootstrap RNG.
    single_trial : {"raise", "baseline_resample"}, default="raise"
        Behavior when only one trial is supplied. ``"raise"`` refuses to
        compute a trial-bootstrap threshold from a single averaged trace.
        ``"baseline_resample"`` uses a non-canonical baseline time-resampling
        surrogate and should be used only as an explicit sensitivity analysis.

    Returns
    -------
    np.ndarray
        ``uint8`` matrix of shape ``(n_sources, n_bins)`` — the trial-averaged
        significant-source matrix (note: **not** per-trial, unlike
        :func:`binarise_signals`).
    """
    s = np.asarray(signal_m, dtype=float)
    if s.ndim == 2:
        s = s[np.newaxis, :, :]
    if s.ndim != 3:
        raise ValueError(
            f"Expected (n_trials, n_sources, n_bins) or (n_sources, n_bins), got {s.shape}."
        )
    n_trials, _, n_bins = s.shape
    if not (1 <= t_stim < n_bins):
        raise ValueError(f"t_stim must be in [1, n_bins-1]. Got {t_stim}, n_bins={n_bins}.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    single_trial_key = str(single_trial).lower()
    if n_trials == 1 and single_trial_key in ("raise", "error", "strict"):
        raise ValueError(
            "Casali bootstrap binarisation requires more than one trial. "
            "Pass trial-level data with shape (n_trials, n_sources, n_bins), "
            "or set single_trial='baseline_resample' for a non-canonical "
            "single-trace sensitivity analysis."
        )
    if n_trials == 1 and single_trial_key not in ("baseline_resample",):
        raise ValueError(
            "single_trial must be 'raise' or 'baseline_resample'. "
            f"Got {single_trial!r}."
        )

    rng = np.random.default_rng(seed)

    # 1. Baseline subtraction per trial & source, then per-source normalization
    #    by baseline SD. Casali significance is expressed in units of each
    #    source's own baseline variability; without this a single global
    #    threshold is dominated by the loudest sources (source-current baseline
    #    SD spans 1-2 orders of magnitude across vertices), leaving quiet
    #    sources permanently sub-threshold and PCI strongly under-estimated.
    base_mean = s[:, :, :t_stim].mean(axis=2, keepdims=True)
    bc = s - base_mean
    base_sd = bc[:, :, :t_stim].std(axis=(0, 2))  # (n_sources,) pooled SD
    base_sd = np.where(base_sd < _EPS, 1.0, base_sd)
    bc = bc / base_sd[np.newaxis, :, np.newaxis]
    avg = bc.mean(axis=0)  # (n_sources, n_bins) z-scored trial-averaged response
    base_bc = bc[:, :, :t_stim]  # (n_trials, n_sources, t_stim)

    # 2/5. Bootstrap null of the max statistic over baseline.
    maxstat = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        if n_trials > 1:
            idx = rng.integers(0, n_trials, n_trials)
            boot = base_bc[idx].mean(axis=0)  # (n_sources, t_stim)
        else:
            # Non-canonical fallback: resample baseline time points with
            # replacement while preserving the spatial pattern in each column.
            idx = rng.integers(0, t_stim, t_stim)
            boot = base_bc[0, :, idx]
        vals = np.abs(boot) if two_sided else boot
        maxstat[b] = float(vals.max())

    thresh = float(np.quantile(maxstat, 1.0 - alpha))

    # 3. Two-sided thresholding of the averaged response.
    resp = np.abs(avg) if two_sided else avg
    return (resp > thresh).astype(np.uint8)


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
        - ``"casali"`` : :func:`binarise_signals_casali` — the paper-faithful
          bootstrap route (single trial-averaged output). Accepts
          ``n_bootstrap``, ``alpha``, ``two_sided``, ``seed``.
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
    "binarise_signals",
    "binarise_signals_casali",
    "binarise",
    "sort_binJ",
    "source_entropy",
    "lz_complexity_2d",
    "pci_norm_factor",
]
