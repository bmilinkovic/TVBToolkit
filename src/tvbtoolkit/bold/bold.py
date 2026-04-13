"""BOLD signal utilities.

This module provides two layers of functionality:

1) Legacy-compatible helpers ported from TVBSim ``tvbsim/BOLD.py``:
   - :class:`BOLDParams`
   - :func:`butter_filtering`
   - :func:`corr_fc_sc` / :func:`corr_FC_SC`

2) A deterministic offline BOLD transform from mean-field activity using the
   TVB first-order Volterra kernel commonly used for Balloon-Windkessel-style
   BOLD monitoring.

Notes
-----
Ported/adapted from TVBSim ``tvbsim/BOLD.py`` (original TVBSim contributors).
TVBToolkit version is self-contained and has no runtime TVBSim imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsg
from scipy.stats import zscore


@dataclass(frozen=True)
class BOLDParams:
    """Band-pass preprocessing parameters for BOLD analyses.

    Parameters
    ----------
    TR : float, default=2.0
        Sampling interval in seconds (fMRI repetition time).
    n_order : int, default=2
        Butterworth filter order.
    low_f_num : float, default=0.01
        Low cutoff frequency in Hz.
    high_f_num : float, default=0.1
        High cutoff frequency in Hz.

    Notes
    -----
    This matches TVBSim defaults from ``tvbsim/BOLD.py``.
    """

    TR: float = 2.0
    n_order: int = 2
    low_f_num: float = 0.01
    high_f_num: float = 0.1


def _as_time_regions(signal: np.ndarray, *, n_regions_hint: int | None = None) -> np.ndarray:
    """Return signal as ``(time, regions)``.

    Parameters
    ----------
    signal : ndarray
        Input array with shape ``(time, regions)`` or ``(regions, time)``.
    n_regions_hint : int | None, optional
        If provided, used to disambiguate orientation.

    Returns
    -------
    ndarray
        Array of shape ``(time, regions)``.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}.")

    if n_regions_hint is not None:
        if x.shape[1] == n_regions_hint:
            return x
        if x.shape[0] == n_regions_hint:
            return x.T

    # Heuristic fallback: time axis is usually longer than region axis.
    if x.shape[0] >= x.shape[1]:
        return x
    return x.T


def butter_filtering(signal: np.ndarray, bp: BOLDParams) -> np.ndarray:
    """Apply Butterworth band-pass filtering to a regional time series.

    Parameters
    ----------
    signal : ndarray
        Array of shape ``(time, regions)``.
    bp : BOLDParams
        Filtering parameters.

    Returns
    -------
    ndarray
        Filtered signal with shape ``(time, regions)``.

    Notes
    -----
    This implementation is a direct port of TVBSim ``butter_filtering``:
    ``scipy.signal.iirfilter(..., ftype='butter')`` + ``scipy.signal.filtfilt``
    along axis 0.

    For short inputs where default padding would fail (`len(time) <= padlen`),
    the function falls back to ``method='gust'`` for numerical robustness.
    """
    x = _as_time_regions(signal)
    nyquist_freq = 0.5 / float(bp.TR)
    low_f = float(bp.low_f_num) / nyquist_freq
    high_f = float(bp.high_f_num) / nyquist_freq
    b, a = spsg.iirfilter(
        int(bp.n_order),
        [low_f, high_f],
        btype="bandpass",
        ftype="butter",
        output="ba",
    )
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        warnings.warn(
            "Input is short for default filtfilt padding; using method='gust' for band-pass filtering.",
            RuntimeWarning,
            stacklevel=2,
        )
        return spsg.filtfilt(b, a, x, axis=0, method="gust")
    return spsg.filtfilt(b, a, x, axis=0)


def preprocess_bold_signal(
    signal: np.ndarray,
    *,
    params: BOLDParams | None = None,
    apply_zscore: bool = True,
    apply_bandpass: bool = True,
    n_regions_hint: int | None = None,
) -> np.ndarray:
    """Preprocess a BOLD-like signal with z-scoring and optional band-pass.

    Parameters
    ----------
    signal : ndarray
        Input signal with shape ``(time, regions)`` or ``(regions, time)``.
    params : BOLDParams | None, optional
        Filter settings. If ``None``, defaults to :class:`BOLDParams`.
    apply_zscore : bool, default=True
        If ``True``, z-score each region over time.
    apply_bandpass : bool, default=True
        If ``True``, apply :func:`butter_filtering`.
    n_regions_hint : int | None, optional
        If provided, enforces axis interpretation so output remains
        `(time, regions)` even when `time < regions`.

    Returns
    -------
    ndarray
        Preprocessed signal with shape ``(time, regions)``.
    """
    x = _as_time_regions(signal, n_regions_hint=n_regions_hint)
    if apply_zscore:
        x = zscore(x, axis=0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if apply_bandpass:
        x = butter_filtering(x, params if params is not None else BOLDParams())
    return x


def corr_fc_sc(signal: np.ndarray, structural_connectivity: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute functional connectivity and FC-SC coupling.

    Parameters
    ----------
    signal : ndarray
        Regional signal matrix. Accepted shapes are ``(regions, time)`` or
        ``(time, regions)``.
    structural_connectivity : ndarray
        Structural connectivity matrix with shape ``(regions, regions)``.

    Returns
    -------
    fc_abs : ndarray
        Absolute Pearson FC matrix (``regions x regions``).
    coupling_coef : float
        Pearson correlation between flattened FC and SC entries.

    Notes
    -----
    TVBSim ``corr_FC_SC`` expects ``(regions, time)``. This function accepts
    both orientations and internally normalizes to that convention.
    """
    sc = np.asarray(structural_connectivity, dtype=float)
    if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
        raise ValueError("structural_connectivity must be square (regions x regions).")

    x = np.asarray(signal, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"signal must be 2D, got {x.shape}.")
    if x.shape[0] == sc.shape[0]:
        sig = x
    elif x.shape[1] == sc.shape[0]:
        sig = x.T
    else:
        raise ValueError(
            f"signal shape {x.shape} is incompatible with SC shape {sc.shape}; "
            "one axis must equal number of regions."
        )

    fc = np.corrcoef(sig)
    # Use upper triangle only (i < j) — excludes diagonal (self-correlation = 1)
    # and redundant lower triangle, matching Brain-Act's convention.
    n = fc.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    fc_vec = fc[iu, ju]
    sc_vec = sc[iu, ju]
    pearson_fcsc = np.corrcoef(fc_vec, sc_vec)
    coef = float(pearson_fcsc[0, 1])
    return np.abs(fc), coef


def corr_FC_SC(signal: np.ndarray, SC: np.ndarray) -> tuple[np.ndarray, float]:
    """Legacy-name alias for :func:`corr_fc_sc`.

    This keeps API parity with TVBSim ``BOLD.py``.
    """
    return corr_fc_sc(signal, SC)


def plot_fc_sc(
    signal: np.ndarray,
    structural_connectivity: np.ndarray,
    *,
    preprocess: bool = True,
    params: BOLDParams | None = None,
    figsize: tuple[float, float] = (8.5, 4.0),
) -> tuple[plt.Figure, np.ndarray, np.ndarray, float]:
    """Plot FC and SC matrices with FC-SC coupling annotation.

    Parameters
    ----------
    signal : ndarray
        Regional signal with shape ``(time, regions)`` or ``(regions, time)``.
    structural_connectivity : ndarray
        Structural connectivity matrix of shape ``(regions, regions)``.
    preprocess : bool, default=True
        If ``True``, apply z-score and band-pass preprocessing before FC.
    params : BOLDParams | None, optional
        BOLD preprocessing parameters. Ignored if ``preprocess=False``.
    figsize : tuple of float, default=(8.5, 4.0)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    axes : ndarray
        Axes array with FC and SC subplots.
    fc_abs : ndarray
        Absolute FC matrix.
    coupling_coef : float
        Pearson coupling coefficient between flattened FC and SC entries.
    """
    x = _as_time_regions(signal, n_regions_hint=np.asarray(structural_connectivity).shape[0])
    if preprocess:
        tr = params.TR if params is not None else 2.0
        pp = preprocess_bold_signal(x, params=params if params is not None else BOLDParams(TR=tr))
    else:
        pp = x

    fc_abs, coef = corr_fc_sc(pp, structural_connectivity)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im0 = axes[0].imshow(fc_abs, cmap="seismic", vmin=0.0, vmax=1.0, origin="lower")
    im1 = axes[1].imshow(structural_connectivity, cmap="viridis", origin="lower")
    axes[0].set_title("|FC|")
    axes[1].set_title(f"SC\nFC-SC r = {coef:.3f}")
    for ax in axes:
        ax.set_xlabel("Region")
        ax.set_ylabel("Region")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()

    return fig, axes, fc_abs, coef


def plot_FC_SC(
    signal: np.ndarray,
    SC: np.ndarray,
    *,
    preprocess: bool = True,
    bp: BOLDParams | None = None,
    figsize: tuple[float, float] = (8.5, 4.0),
) -> tuple[plt.Figure, np.ndarray, np.ndarray, float]:
    """Legacy-name wrapper for :func:`plot_fc_sc`.

    Parameters follow the TVBSim naming convention (``SC``, ``bp``).
    """
    return plot_fc_sc(signal, SC, preprocess=preprocess, params=bp, figsize=figsize)


def first_order_volterra_hrf(
    t_s: np.ndarray,
    *,
    tau_s: float = 0.8,
    tau_f: float = 0.4,
) -> np.ndarray:
    """Evaluate the first-order Volterra HRF kernel used by TVB's BOLD monitor.

    The kernel follows TVB's default equation:

    .. math::
       G(t) = \frac{1}{3}\,\exp\left(-\frac{t}{2\tau_s}\right)
              \frac{\sin\left(\sqrt{\frac{1}{\tau_f} - \frac{1}{4\tau_s^2}}\,t\right)}
                   {\sqrt{\frac{1}{\tau_f} - \frac{1}{4\tau_s^2}}}

    Parameters
    ----------
    t_s : ndarray
        Time vector in seconds.
    tau_s : float, default=0.8
        Exponential decay parameter.
    tau_f : float, default=0.4
        Oscillatory parameter.

    Returns
    -------
    ndarray
        HRF kernel values with shape matching ``t_s``.

    References
    ----------
    Friston et al., NeuroImage 12:466-477 (2000).
    """
    t = np.asarray(t_s, dtype=float)
    denom_term = (1.0 / float(tau_f)) - (1.0 / (4.0 * float(tau_s) ** 2))
    if denom_term <= 0:
        raise ValueError("Invalid (tau_s, tau_f) combination: kernel denominator is non-positive.")
    w = np.sqrt(denom_term)
    return (1.0 / 3.0) * np.exp(-0.5 * (t / float(tau_s))) * (np.sin(w * t) / w)


def bold_from_firing_rates(
    rates: np.ndarray,
    *,
    dt_ms: float,
    tr_ms: float = 2000.0,
    hrf_length_ms: float = 20000.0,
    tau_s: float = 0.8,
    tau_f: float = 0.4,
    k_1: float = 5.6,
    V_0: float = 0.02,
    interim_period_ms: float = 4.0,
    return_debug: bool = False,
) -> np.ndarray | dict[str, Any]:
    """Generate BOLD-like signals from mean-field firing rates.

    Parameters
    ----------
    rates : ndarray
        Neural activity matrix with shape ``(time, regions)`` or
        ``(regions, time)``.
    dt_ms : float
        Sampling interval of ``rates`` in milliseconds.
    tr_ms : float, default=2000.0
        Output BOLD sampling period in milliseconds (fMRI TR).
    hrf_length_ms : float, default=20000.0
        Length of HRF history used for convolution.
    tau_s : float, default=0.8
        First-order Volterra kernel parameter.
    tau_f : float, default=0.4
        First-order Volterra kernel parameter.
    k_1 : float, default=5.6
        Volterra scaling parameter.
    V_0 : float, default=0.02
        Resting blood volume fraction.
    interim_period_ms : float, default=4.0
        Inner averaging period in milliseconds. TVB's BOLD monitor uses 4 ms
        (`2**-2 / ms` stock sampling rate).
    return_debug : bool, default=False
        If ``True``, return a dictionary with intermediate arrays.

    Returns
    -------
    bold : ndarray
        BOLD signal with shape ``(time_bold, regions)``.
    or dict
        If ``return_debug=True``, returns a dictionary with keys:
        ``bold``, ``time_ms``, ``stock_signal``, ``hrf_kernel``.

    Notes
    -----
    This is an offline deterministic transform aligned with TVB's first-order
    Volterra BOLD monitor assumptions:

    1. Average neural drive into fixed-length interim bins.
    2. Convolve each region with a causal first-order Volterra HRF.
    3. Apply scaling ``(conv - 1) * (k_1 * V_0)``.
    4. Downsample to ``tr_ms``.

    This function provides a practical way to derive BOLD-like trajectories from
    mean-field rates when only rate monitor outputs are available.
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive.")
    if tr_ms <= 0:
        raise ValueError("tr_ms must be positive.")
    if interim_period_ms <= 0:
        raise ValueError("interim_period_ms must be positive.")

    x = _as_time_regions(rates)
    t_len, n_regions = x.shape
    if t_len < 2:
        raise ValueError("Need at least 2 time samples to compute BOLD.")

    interim_steps = max(1, int(round(interim_period_ms / dt_ms)))
    n_bins = t_len // interim_steps
    if n_bins < 2:
        raise ValueError(
            "Signal is too short for the chosen interim_period_ms; "
            f"need >= {2 * interim_steps} samples, got {t_len}."
        )

    trimmed = x[: n_bins * interim_steps]
    stock = trimmed.reshape(n_bins, interim_steps, n_regions).mean(axis=1)

    stock_steps = int(np.ceil(float(hrf_length_ms) / float(interim_period_ms)))
    stock_time_max_s = float(hrf_length_ms) / 1000.0
    stock_time_step_s = stock_time_max_s / float(stock_steps)
    stock_time_s = np.arange(0.0, stock_time_max_s, stock_time_step_s)
    hrf = first_order_volterra_hrf(stock_time_s, tau_s=tau_s, tau_f=tau_f)

    conv_full = spsg.fftconvolve(stock, hrf[:, None], mode="full", axes=0)
    conv = conv_full[:n_bins]
    bold_stock = (conv - 1.0) * (float(k_1) * float(V_0))

    tr_bins = max(1, int(round(float(tr_ms) / float(interim_period_ms))))
    bold = bold_stock[tr_bins - 1 :: tr_bins]
    time_ms = (np.arange(bold.shape[0], dtype=float) + 1.0) * float(tr_ms)

    if return_debug:
        return {
            "bold": bold,
            "time_ms": time_ms,
            "stock_signal": stock,
            "hrf_kernel": hrf,
            "interim_steps": np.array([interim_steps], dtype=int),
            "tr_bins": np.array([tr_bins], dtype=int),
        }
    return bold


__all__ = [
    "BOLDParams",
    "butter_filtering",
    "preprocess_bold_signal",
    "corr_fc_sc",
    "corr_FC_SC",
    "plot_fc_sc",
    "plot_FC_SC",
    "first_order_volterra_hrf",
    "bold_from_firing_rates",
]
