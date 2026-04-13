"""Single-region analysis utilities.

These utilities integrate core signal-processing helpers used in the
paper_pipeline_hub single-cell/network sections into TVBToolkit's single-region
namespace.
"""

from __future__ import annotations

import numpy as np


def bin_array(array: np.ndarray, bin_width: float, time_array: np.ndarray) -> np.ndarray:
    """Bin a time series by averaging values in fixed-width bins.

    Parameters
    ----------
    array : ndarray
        One-dimensional signal array.
    bin_width : float
        Bin width in the same units as ``time_array`` spacing.
    time_array : ndarray
        Monotonic time vector associated with ``array``.

    Returns
    -------
    ndarray
        Mean value in each bin.
    """
    x = np.asarray(array, dtype=float).reshape(-1)
    t = np.asarray(time_array, dtype=float).reshape(-1)
    if x.size != t.size:
        raise ValueError("array and time_array must have the same length.")
    if x.size < 2:
        return x.copy()

    dt = float(t[1] - t[0])
    if dt <= 0:
        raise ValueError("time_array must be strictly increasing.")
    n0 = int(float(bin_width) / dt)
    if n0 <= 0:
        raise ValueError("bin_width is too small for the provided time step.")
    n1 = int((float(t[-1]) - float(t[0])) / float(bin_width))
    if n1 <= 0:
        return np.array([float(np.mean(x))], dtype=float)
    return x[: n0 * n1].reshape((n1, n0)).mean(axis=1)


def heaviside(x: np.ndarray) -> np.ndarray:
    """Compute the Heaviside step function ``0.5 * (1 + sign(x))``."""
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.sign(arr))


def input_rate(
    t: np.ndarray,
    t1_exc: float,
    tau1_exc: float,
    tau2_exc: float,
    ampl_exc: float,
    plateau: float,
) -> np.ndarray:
    """Generate the piecewise-Gaussian/plateau input profile used in paper scripts.

    This mirrors the form used in ``paper_pipeline_hub/functions.py``.
    """
    tt = np.asarray(t, dtype=float)
    inp = float(ampl_exc) * (
        np.exp(-(tt - float(t1_exc)) ** 2 / (2.0 * float(tau1_exc) ** 2)) * heaviside(-(tt - float(t1_exc)))
        + heaviside(-(tt - (float(t1_exc) + float(plateau)))) * heaviside(tt - float(t1_exc))
        + np.exp(-(tt - (float(t1_exc) + float(plateau))) ** 2 / (2.0 * float(tau2_exc) ** 2))
        * heaviside(tt - (float(t1_exc) + float(plateau)))
    )
    return np.asarray(inp, dtype=float)


def prepare_population_rates(
    total_time: float,
    dt: float,
    pop_rate_exc: np.ndarray,
    pop_rate_inh: np.ndarray,
    adaptation: np.ndarray,
    *,
    bin_width: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin excitatory/inhibitory population rates and adaptation traces.

    Parameters
    ----------
    total_time : float
        Total simulation time (same unit as ``dt`` and ``bin_width``).
    dt : float
        Simulation time step.
    pop_rate_exc : ndarray
        Excitatory population rate trace.
    pop_rate_inh : ndarray
        Inhibitory population rate trace.
    adaptation : ndarray
        Adaptation variable trace.
    bin_width : float, default=5.0
        Width for binning/averaging.

    Returns
    -------
    tim_binned : ndarray
    rate_exc_binned : ndarray
    rate_inh_binned : ndarray
    adaptation_binned : ndarray
    """
    t = np.arange(int(float(total_time) / float(dt)), dtype=float) * float(dt)
    re = np.asarray(pop_rate_exc, dtype=float).reshape(-1)
    ri = np.asarray(pop_rate_inh, dtype=float).reshape(-1)
    ad = np.asarray(adaptation, dtype=float).reshape(-1)

    n = min(t.size, re.size, ri.size, ad.size)
    t = t[:n]
    re = re[:n]
    ri = ri[:n]
    ad = ad[:n]

    tim_binned = bin_array(t, bin_width, t)
    exc_b = bin_array(re, bin_width, t)
    inh_b = bin_array(ri, bin_width, t)
    ad_b = bin_array(ad, bin_width, t)
    return tim_binned, exc_b, inh_b, ad_b


def calculate_psd_fmax(
    pop_rate_exc: np.ndarray,
    pop_rate_inh: np.ndarray,
    time_binned: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute PSD for excitatory/inhibitory rates and peak frequency.

    Parameters
    ----------
    pop_rate_exc : ndarray
        Excitatory population rate.
    pop_rate_inh : ndarray
        Inhibitory population rate.
    time_binned : ndarray
        Time vector corresponding to rates, in milliseconds.

    Returns
    -------
    frq_max : float
        Peak frequency (Hz) from excitatory PSD.
    frq_pos : ndarray
        Positive frequency axis.
    pwr_exc_pos : ndarray
        Excitatory PSD values on ``frq_pos``.
    pwr_inh_pos : ndarray
        Inhibitory PSD values on ``frq_pos``.
    """
    t_ms = np.asarray(time_binned, dtype=float).reshape(-1)
    exc = np.asarray(pop_rate_exc, dtype=float).reshape(-1)
    inh = np.asarray(pop_rate_inh, dtype=float).reshape(-1)

    n = min(t_ms.size, exc.size, inh.size)
    if n < 3:
        raise ValueError("Need at least 3 samples for PSD analysis.")

    t_s = t_ms[:n] * 1e-3
    exc = exc[:n]
    inh = inh[:n]

    f_sampling = float(n) / float(t_s[-1])
    frq = np.fft.fftfreq(n, 1.0 / f_sampling)

    pwr_exc = np.abs(np.fft.fft(exc)) ** 2
    pwr_inh = np.abs(np.fft.fft(inh)) ** 2

    good = frq > 0.0
    frq_pos = frq[good]
    pwr_exc_pos = pwr_exc[good]
    pwr_inh_pos = pwr_inh[good]

    if frq_pos.size == 0:
        raise ValueError("No positive frequencies available for PSD.")
    frq_max = float(frq_pos[int(np.argmax(pwr_exc_pos))])
    return frq_max, frq_pos, pwr_exc_pos, pwr_inh_pos


__all__ = [
    "bin_array",
    "heaviside",
    "input_rate",
    "prepare_population_rates",
    "calculate_psd_fmax",
]
