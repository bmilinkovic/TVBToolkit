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


__all__ = [
    "binarise_signals",
    "sort_binJ",
    "source_entropy",
    "lz_complexity_2d",
    "pci_norm_factor",
]
