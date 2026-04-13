"""Spectral utility functions for validating and characterising firing-rate signals.

Before running the phase-coherence brain-state pipeline on firing rates it is
worth confirming that the signal actually contains a meaningful oscillation in
the chosen frequency band.  Applying a Hilbert transform to a signal that has
no spectral peak produces phase estimates that are essentially random, and the
resulting brain-state patterns would be uninterpretable.

This module provides three lightweight checks:

- :func:`psd_per_region` — Welch power spectral density for every region.
- :func:`dominant_frequency` — the frequency with the most power inside the
  target band, per region.
- :func:`phase_coherence_validity` — a scalar summary of whether the narrowband
  signal is strong enough and coherent enough to justify phase analysis.

All functions accept signals in ``(time, regions)`` format and require the
sampling interval ``dt_ms`` in milliseconds (e.g. ``5.0`` for AdEx/MF outputs
binned at the default 5 ms bin width, ``0.1`` for raw TVB whole-brain rates).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy.signal import filtfilt, hilbert, iirfilter, welch
from scipy.stats import zscore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PSDResult(NamedTuple):
    """Output of :func:`psd_per_region`.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency axis in Hz, shape ``(n_freqs,)``.
    power : np.ndarray
        Power spectral density per region, shape ``(n_freqs, n_regions)``.
        Units: (signal units)² / Hz.
    """
    frequencies: np.ndarray
    power: np.ndarray


class ValidityResult(NamedTuple):
    """Output of :func:`phase_coherence_validity`.

    Attributes
    ----------
    mean_amplitude : np.ndarray
        Mean analytic amplitude of the narrowband signal per region,
        shape ``(n_regions,)``.  Near-zero values indicate that the
        band contains very little signal energy.
    kuramoto_order : float
        Mean Kuramoto order parameter |mean(exp(iφ))| averaged over all
        time points and region pairs.  Ranges 0–1; values close to 0 mean
        the inter-regional phases are uniformly distributed (no coherent
        oscillation), values close to 1 mean all regions are phase-locked.
        Practical rule of thumb: values above ~0.1 suggest there is
        enough coherent oscillatory activity to cluster meaningfully.
    has_spectral_peak : np.ndarray
        Boolean array of shape ``(n_regions,)``; ``True`` if the PSD inside
        the target band exceeds ``peak_snr_threshold`` times the out-of-band
        floor, indicating a genuine spectral peak rather than broadband noise.
    warnings : list[str]
        Human-readable warning strings raised during the check.
    """
    mean_amplitude: np.ndarray
    kuramoto_order: float
    has_spectral_peak: np.ndarray
    warnings: list[str]


def psd_per_region(
    x: np.ndarray,
    dt_ms: float,
    *,
    nperseg: int | None = None,
) -> PSDResult:
    """Compute Welch power spectral density for every region.

    Parameters
    ----------
    x : np.ndarray
        Signal array of shape ``(time, regions)`` in arbitrary units (Hz for
        firing rates).
    dt_ms : float
        Sampling interval in milliseconds.
    nperseg : int | None, default=None
        Length of each Welch segment.  When ``None``, defaults to
        ``min(256, n_samples // 4)`` so it works on short simulations too.

    Returns
    -------
    PSDResult
        Named tuple with ``frequencies`` (Hz) and ``power`` arrays.

    Examples
    --------
    >>> result = psd_per_region(firing_rates, dt_ms=5.0)
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogy(result.frequencies, result.power[:, 0])
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("PSD")
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError(f"Expected (time, regions) array, got shape {x.shape}.")

    n_samples = x.shape[0]
    fs_hz = 1000.0 / float(dt_ms)  # sampling frequency in Hz

    seg = nperseg if nperseg is not None else max(4, min(256, n_samples // 4))

    freqs, pxx_first = welch(x[:, 0], fs=fs_hz, nperseg=seg)
    power_all = np.empty((len(freqs), x.shape[1]), dtype=float)
    power_all[:, 0] = pxx_first
    for r in range(1, x.shape[1]):
        _, power_all[:, r] = welch(x[:, r], fs=fs_hz, nperseg=seg)

    return PSDResult(frequencies=freqs, power=power_all)


def dominant_frequency(
    x: np.ndarray,
    dt_ms: float,
    bandpass_hz: tuple[float, float],
    *,
    nperseg: int | None = None,
) -> np.ndarray:
    """Return the frequency with maximum power inside the band, per region.

    Parameters
    ----------
    x : np.ndarray
        Signal array of shape ``(time, regions)``.
    dt_ms : float
        Sampling interval in milliseconds.
    bandpass_hz : tuple[float, float]
        ``(low_hz, high_hz)`` frequency range to search within.
    nperseg : int | None, default=None
        Welch segment length (passed to :func:`psd_per_region`).

    Returns
    -------
    np.ndarray
        Dominant frequency in Hz per region, shape ``(n_regions,)``.

    Notes
    -----
    For the AdEx SNN / mean-field pipeline (``bandpass_hz=(2.0, 80.0)``)
    this typically returns values in the 30–60 Hz range (gamma), reflecting
    the E-I loop resonance driven by the 5 ms synaptic time constants.
    For TVB whole-brain slow states (``bandpass_hz=(0.05, 1.0)``) expect
    values in the 0.05–0.5 Hz range.

    Examples
    --------
    >>> dom_f = dominant_frequency(firing_rates, dt_ms=5.0,
    ...                            bandpass_hz=(2.0, 80.0))
    >>> print(f"Mean dominant frequency: {dom_f.mean():.1f} Hz")
    """
    low_hz, high_hz = float(bandpass_hz[0]), float(bandpass_hz[1])
    result = psd_per_region(x, dt_ms, nperseg=nperseg)
    mask = (result.frequencies >= low_hz) & (result.frequencies <= high_hz)
    if not mask.any():
        raise ValueError(
            f"No frequency bins found inside bandpass_hz={bandpass_hz}. "
            f"Frequency resolution is {result.frequencies[1] - result.frequencies[0]:.4f} Hz. "
            "Try a longer signal or adjust bandpass_hz."
        )
    in_band_power = result.power[mask, :]          # (n_band_freqs, n_regions)
    in_band_freqs = result.frequencies[mask]       # (n_band_freqs,)
    peak_idx = np.argmax(in_band_power, axis=0)   # (n_regions,)
    return in_band_freqs[peak_idx]


def phase_coherence_validity(
    x: np.ndarray,
    dt_ms: float,
    bandpass_hz: tuple[float, float],
    *,
    filter_order: int = 4,
    peak_snr_threshold: float = 3.0,
    amplitude_threshold: float = 0.05,
    kuramoto_threshold: float = 0.05,
    nperseg: int | None = None,
) -> ValidityResult:
    """Check whether a signal is suitable for phase-coherence brain-state analysis.

    Runs three complementary checks:

    1. **Spectral peak check** — is there a clear spectral peak inside the
       target band, or is the band just broadband noise?
    2. **Amplitude check** — does the narrowband-filtered signal have
       non-negligible amplitude in each region?
    3. **Kuramoto coherence check** — are there at least brief moments of
       inter-regional phase synchronisation, or are the phases completely
       random across all pairs?

    Parameters
    ----------
    x : np.ndarray
        Signal array of shape ``(time, regions)``.
    dt_ms : float
        Sampling interval in milliseconds.
    bandpass_hz : tuple[float, float]
        Target analysis band ``(low_hz, high_hz)``.
    filter_order : int, default=4
        Butterworth order for the narrowband filter used in the amplitude and
        Kuramoto checks.
    peak_snr_threshold : float, default=3.0
        A region is considered to have a spectral peak when its in-band
        maximum power is at least this many times the median out-of-band power.
    amplitude_threshold : float, default=0.05
        Minimum mean analytic amplitude (in z-score units) for a region to be
        considered active.  Regions below this are flagged.
    kuramoto_threshold : float, default=0.05
        Minimum mean Kuramoto order parameter across all region pairs.  Below
        this the phases are essentially random and clustering is not meaningful.
    nperseg : int | None, default=None
        Welch segment length.

    Returns
    -------
    ValidityResult
        Named tuple; see :class:`ValidityResult` for field descriptions.

    Examples
    --------
    >>> result = phase_coherence_validity(firing_rates, dt_ms=5.0,
    ...                                  bandpass_hz=(2.0, 80.0))
    >>> if result.kuramoto_order < 0.05:
    ...     print("Warning: no coherent oscillation detected — brain states may not be meaningful.")
    >>> print("Dominant region amplitudes:", result.mean_amplitude)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError(f"Expected (time, regions), got {x.shape}.")

    warn_msgs: list[str] = []
    n_regions = x.shape[1]
    fs_hz = 1000.0 / float(dt_ms)
    nyq = fs_hz / 2.0
    low_hz, high_hz = float(bandpass_hz[0]), float(bandpass_hz[1])

    if high_hz >= nyq:
        warn_msgs.append(
            f"bandpass_hz upper cutoff {high_hz} Hz exceeds or equals the Nyquist "
            f"frequency {nyq:.1f} Hz for dt_ms={dt_ms}. Reduce the upper cutoff or "
            "decrease dt_ms (increase sampling rate)."
        )

    # ---- 1. Spectral peak check ------------------------------------------------
    psd_result = psd_per_region(x, dt_ms, nperseg=nperseg)
    freqs = psd_result.frequencies
    power = psd_result.power

    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    out_band = ~in_band

    has_peak = np.zeros(n_regions, dtype=bool)
    for r in range(n_regions):
        in_band_max = power[in_band, r].max() if in_band.any() else 0.0
        out_band_median = np.median(power[out_band, r]) if out_band.any() else np.nan
        if np.isnan(out_band_median) or out_band_median <= 0:
            has_peak[r] = True  # can't compute SNR — assume ok
        else:
            has_peak[r] = (in_band_max / out_band_median) >= peak_snr_threshold

    n_no_peak = int(np.sum(~has_peak))
    if n_no_peak > 0:
        warn_msgs.append(
            f"{n_no_peak}/{n_regions} region(s) have no clear spectral peak in "
            f"{bandpass_hz} Hz (in-band max < {peak_snr_threshold}× out-of-band median). "
            "The Hilbert phase in these regions may be unreliable."
        )

    # ---- 2. Amplitude check (narrowband filter + Hilbert) ----------------------
    xz = zscore(x, axis=0, ddof=1)
    xz = np.nan_to_num(xz, nan=0.0)

    low_norm = low_hz / nyq
    high_norm = high_hz / nyq
    b, a = iirfilter(
        int(filter_order), (low_norm, high_norm),
        btype="bandpass", ftype="butter", output="ba",
    )
    try:
        xf = filtfilt(b, a, xz, axis=0)
    except ValueError:
        xf = filtfilt(b, a, xz, axis=0, method="gust")

    analytic = hilbert(xf, axis=0)
    amplitude = np.abs(analytic)          # (time, regions)
    mean_amplitude = amplitude.mean(axis=0)  # (n_regions,)

    low_amp_regions = int(np.sum(mean_amplitude < amplitude_threshold))
    if low_amp_regions > 0:
        warn_msgs.append(
            f"{low_amp_regions}/{n_regions} region(s) have very low mean analytic "
            f"amplitude (< {amplitude_threshold} in z-score units) after narrowband "
            f"filtering to {bandpass_hz} Hz. These regions may have no oscillatory "
            "content in the target band."
        )

    # ---- 3. Kuramoto order parameter -------------------------------------------
    phase = np.angle(analytic)            # (time, regions)
    # Mean resultant length across all region pairs at each time step
    order_param = np.abs(np.mean(np.exp(1j * phase), axis=1))  # (time,)
    kuramoto_order = float(order_param.mean())

    if kuramoto_order < kuramoto_threshold:
        warn_msgs.append(
            f"Mean Kuramoto order parameter = {kuramoto_order:.4f} < {kuramoto_threshold}. "
            "Inter-regional phases are nearly uniformly distributed — there is very little "
            "coherent oscillatory synchrony. Brain-state clustering on these patterns is "
            "unlikely to produce physiologically meaningful results. Consider checking: "
            "(1) whether the model produces oscillations in this band, "
            "(2) whether dt_ms is set correctly, "
            "(3) whether the simulation ran long enough."
        )

    if warn_msgs:
        for msg in warn_msgs:
            warnings.warn(msg, UserWarning, stacklevel=2)

    return ValidityResult(
        mean_amplitude=mean_amplitude,
        kuramoto_order=kuramoto_order,
        has_spectral_peak=has_peak,
        warnings=warn_msgs,
    )


__all__ = [
    "PSDResult",
    "ValidityResult",
    "psd_per_region",
    "dominant_frequency",
    "phase_coherence_validity",
]
