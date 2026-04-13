"""Utility functions ported from legacy TVBSim `brian_MF/brian_functions.py`.

Ported/adapted from TVBSim brian_MF (legacy local repository) for parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class PopulationRates:
    """Binned population-rate outputs.

    Attributes
    ----------
    time_ms
        Binned time axis in milliseconds.
    exc_hz
        Excitatory population firing rate in Hz.
    inh_hz
        Inhibitory population firing rate in Hz.
    adaptation
        Binned adaptation trace (legacy `P2mon`) when available.
    """

    time_ms: np.ndarray
    exc_hz: np.ndarray
    inh_hz: np.ndarray
    adaptation: np.ndarray | None = None


def calculate_psd_fmax(pop_rate_exc: np.ndarray, pop_rate_inh: np.ndarray, time_binned_ms: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute basic FFT power spectra and excitatory peak frequency.

    This mirrors the legacy implementation exactly (plain FFT on binned rates).

    Parameters
    ----------
    pop_rate_exc
        Excitatory rate trace (Hz), shape ``(T,)``.
    pop_rate_inh
        Inhibitory rate trace (Hz), shape ``(T,)``.
    time_binned_ms
        Binned time axis in milliseconds, shape ``(T,)``.

    Returns
    -------
    tuple
        ``(frq_max, frq_positive, pwr_exc_positive, pwr_inh_positive)``.
    """

    time_s = np.asarray(time_binned_ms, dtype=float) * 1e-3
    fr_exc = np.asarray(pop_rate_exc, dtype=float)
    fr_inh = np.asarray(pop_rate_inh, dtype=float)

    f_sampling = len(time_s) / time_s[-1]
    frq = np.fft.fftfreq(len(time_s), 1.0 / f_sampling)

    pwr_exc = np.abs(np.fft.fft(fr_exc.T)) ** 2
    pwr_inh = np.abs(np.fft.fft(fr_inh.T)) ** 2

    good = frq > 0
    frq_good = frq[good]
    pwr_exc_good = pwr_exc[good]
    pwr_inh_good = pwr_inh[good]
    frq_max = float(frq_good[pwr_exc_good == pwr_exc_good.max()][0])

    return frq_max, frq_good, pwr_exc_good, pwr_inh_good


def plot_psd(frq_max: float, frq_good: np.ndarray, pwr_exc: np.ndarray, pwr_inh: np.ndarray) -> None:
    """Plot PSD curves using the legacy visual style."""

    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    plt.rcParams.update({"font.size": 14})

    axes.loglog(frq_good, pwr_inh, "-", color="darkred", alpha=0.9, label="Inh.")
    axes.loglog(frq_good, pwr_exc, "-", color="SteelBlue", label="Exc.")
    axes.axvline(x=frq_max, color="b", label=f"fmax = {frq_max:.2f}")

    axes.set_xlabel("Frequency (Hz)")
    axes.set_ylabel("Power")
    axes.legend()
    plt.tight_layout()
    plt.show()


def bin_array(array: np.ndarray, bin_width: float, time_array: np.ndarray) -> np.ndarray:
    """Bin an array into equally spaced bins and take per-bin mean.

    Parameters
    ----------
    array
        Input samples to bin.
    bin_width
        Bin width in the same units as ``time_array``.
    time_array
        Time support matching ``array``.
    """

    arr = np.asarray(array)
    t = np.asarray(time_array)
    n0 = int(bin_width / (t[1] - t[0]))
    n1 = int((t[-1] - t[0]) / bin_width)
    return arr[: n0 * n1].reshape((n1, n0)).mean(axis=1)


def heaviside(x: np.ndarray | float) -> np.ndarray | float:
    """Legacy Heaviside approximation ``0.5 * (1 + sign(x))``."""

    return 0.5 * (1 + np.sign(x))


def input_rate(t: np.ndarray | float, t1_exc: float, tau1_exc: float, tau2_exc: float, ampl_exc: float, plateau: float) -> np.ndarray:
    """Legacy external-input pulse profile used by AdEx simulations.

    Time is interpreted in milliseconds (matching legacy scripts).
    """

    t_arr = np.asarray(t)
    inp = ampl_exc * (
        np.exp(-((t_arr - t1_exc) ** 2) / (2.0 * tau1_exc**2)) * heaviside(-(t_arr - t1_exc))
        + heaviside(-(t_arr - (t1_exc + plateau))) * heaviside(t_arr - t1_exc)
        + np.exp(-((t_arr - (t1_exc + plateau)) ** 2) / (2.0 * tau2_exc**2)) * heaviside(t_arr - (t1_exc + plateau))
    )
    return inp


def plot_raster_mean_fr(
    ras_inh: np.ndarray,
    ras_exc: np.ndarray,
    time_binned_ms: np.ndarray,
    pop_rate_inh: np.ndarray,
    pop_rate_exc: np.ndarray,
    adaptation: np.ndarray | None = None,
    input_binned: np.ndarray | None = None,
    title: str = "",
    figsize: tuple[float, float] = (10, 16),
    save: int = 0,
    save_path: str = "",
    save_name: str = "plot.pdf",
) -> None:
    """Legacy raster + mean firing-rate panel."""

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    ax_raster, ax_rate = axes

    ax_raster.plot(ras_inh[0] / 1000.0, ras_inh[1], ",r")
    ax_raster.plot(ras_exc[0] / 1000.0, ras_exc[1], ",b")
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Neuron index")

    ax_rate.plot(time_binned_ms / 1000.0, pop_rate_inh, "r", label="Inh")
    ax_rate.plot(time_binned_ms / 1000.0, pop_rate_exc, "b", label="Exc")
    if input_binned is not None and not np.isnan(input_binned).all():
        ax_rate.plot(time_binned_ms / 1000.0, input_binned, "green", label="input")

    ax_rate.set_xlabel("Time (s)")
    ax_rate.set_ylabel("Population Firing Rate")
    ax_raster.set_title(title)

    if save > 0:
        from pathlib import Path

        out_dir = Path(save_path) if save_path else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / save_name, format="pdf", dpi=300, bbox_inches="tight")
    if save < 2:
        fig.show()


def prepare_population_rates(
    total_time_ms: float,
    dt_ms: float,
    fr_exc_monitor: Any,
    fr_inh_monitor: Any,
    adaptation_monitor: Any,
    bin_width_ms: float = 5.0,
) -> PopulationRates:
    """Bin Brian2 monitors into the legacy firing-rate representation.

    Parameters
    ----------
    total_time_ms
        Total simulation duration in milliseconds.
    dt_ms
        Simulation timestep in milliseconds.
    fr_exc_monitor
        Brian2 population-rate monitor for excitatory neurons.
    fr_inh_monitor
        Brian2 population-rate monitor for inhibitory neurons.
    adaptation_monitor
        Legacy `P2mon` monitor (summed adaptation) or compatible object.
    bin_width_ms
        Temporal bin width.
    """

    from brian2 import Hz, arange, array

    time_array = arange(int(total_time_ms / dt_ms)) * dt_ms

    fr_exc = array(fr_exc_monitor.rate / Hz)
    t_binned = bin_array(time_array, bin_width_ms, time_array)
    pop_exc = bin_array(fr_exc, bin_width_ms, time_array)

    fr_inh = array(fr_inh_monitor.rate / Hz)
    pop_inh = bin_array(fr_inh, bin_width_ms, time_array)

    adaptation = None
    if adaptation_monitor is not None:
        adaptation = bin_array(adaptation_monitor[0].P, bin_width_ms, time_array)

    return PopulationRates(time_ms=t_binned, exc_hz=pop_exc, inh_hz=pop_inh, adaptation=adaptation)
