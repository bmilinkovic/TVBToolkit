"""Perturbational Complexity Index based on state transitions (PCI-ST).

This module is an independent implementation of the algorithm described by
Comolatti et al. (2019) and exposed by the authors' reference implementation:

https://github.com/renzocom/PCIst

PCI-ST is calculated from one trial-averaged evoked response.  It first uses
singular-value decomposition (SVD) to retain strong, spatially independent
response components.  It then counts response state transitions that cannot be
explained by baseline transitions.

Unlike Lempel-Ziv PCI, PCI-ST is **not bounded between zero and one**.  Values
must therefore be compared only when signal representation, sampling rate,
analysis windows, and algorithm parameters are held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np


_DEFAULT_BASELINE_WINDOW_MS = (-400.0, -50.0)
_DEFAULT_RESPONSE_WINDOW_MS = (0.0, 300.0)


@dataclass(frozen=True)
class PCIStResult:
    """Detailed result of a PCI-ST calculation.

    Array-valued fields use the order of the retained SVD components.  The
    threshold-dependent arrays have shape ``(n_steps, n_components)``.

    Notes
    -----
    ``pci_st`` is not normalized to the interval ``[0, 1]``.
    """

    pci_st: float
    component_contributions: np.ndarray
    n_components: int
    n_variance_components: int
    selected_component_indices: np.ndarray
    singular_values: np.ndarray
    explained_variance_percent: np.ndarray
    cumulative_explained_variance_percent: np.ndarray
    snr_before_filter: np.ndarray
    retained_snrs: np.ndarray
    component_signals: np.ndarray
    thresholds: np.ndarray
    nst_response: np.ndarray
    nst_baseline: np.ndarray
    nst_difference: np.ndarray
    optimal_threshold_indices: np.ndarray
    optimal_thresholds: np.ndarray
    optimal_nst_response: np.ndarray
    optimal_nst_baseline: np.ndarray
    baseline_window_ms: tuple[float, float]
    response_window_ms: tuple[float, float]
    n_baseline_samples: int
    n_response_samples: int
    baseline_sample_times_ms: np.ndarray
    response_sample_times_ms: np.ndarray
    k: float
    min_snr: float
    max_var_percent: float
    n_steps: int
    embedded: bool
    embedding_dimension: int
    embedding_delay_samples: int
    average_referenced: bool
    baseline_corrected: bool
    n_trials_averaged: int = 1
    trial_baseline_centered: bool = False
    sampling_interval_ms: float = float("nan")
    sampling_rate_hz: float = float("nan")
    effective_baseline_sample_range_ms: tuple[float, float] = (
        float("nan"),
        float("nan"),
    )
    effective_response_sample_range_ms: tuple[float, float] = (
        float("nan"),
        float("nan"),
    )

    @property
    def score(self) -> float:
        """Alias for :attr:`pci_st`."""

        return self.pci_st


def _validate_window(
    value: Sequence[float],
    *,
    name: str,
) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    start, end = float(value[0]), float(value[1])
    if not np.isfinite([start, end]).all():
        raise ValueError(f"{name} must contain finite values.")
    if start >= end:
        raise ValueError(f"{name} start must be smaller than its end.")
    return start, end


def _validate_evoked_and_times(
    evoked: np.ndarray,
    times_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(evoked, dtype=float)
    times = np.asarray(times_ms, dtype=float)

    if signal.ndim != 2:
        raise ValueError(
            "evoked must have shape (channels, time); "
            f"received array with shape {signal.shape}."
        )
    if signal.shape[0] < 1 or signal.shape[1] < 2:
        raise ValueError("evoked must contain at least one channel and two samples.")
    if times.ndim != 1:
        raise ValueError("times_ms must be one-dimensional.")
    if times.size != signal.shape[1]:
        raise ValueError(
            "times_ms length must equal the evoked time dimension; "
            f"got {times.size} and {signal.shape[1]}."
        )
    if not np.isfinite(signal).all() or not np.isfinite(times).all():
        raise ValueError("evoked and times_ms must contain only finite values.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_ms must be strictly increasing.")
    return signal, times


def _window_mask(
    times_ms: np.ndarray,
    window_ms: tuple[float, float],
) -> np.ndarray:
    """Return the official-Python-compatible half-open window mask."""

    start, end = window_ms
    return (times_ms >= start) & (times_ms < end)


def _uniform_sampling_interval_ms(times_ms: np.ndarray) -> float:
    """Return the sampling interval, rejecting irregular sample times."""

    differences = np.diff(np.asarray(times_ms, dtype=float))
    interval = float(np.median(differences))
    tolerance = max(1e-9, abs(interval) * 1e-7)
    if not np.allclose(differences, interval, rtol=1e-7, atol=tolerance):
        maximum_deviation = float(np.max(np.abs(differences - interval)))
        raise ValueError(
            "PCI-ST requires uniformly sampled times. "
            f"Median interval={interval:.12g} ms; maximum deviation="
            f"{maximum_deviation:.12g} ms."
        )
    return interval


def _validate_window_coverage(
    times_ms: np.ndarray,
    window_ms: tuple[float, float],
    *,
    sampling_interval_ms: float,
    name: str,
) -> np.ndarray:
    """Return a window mask and reject silently truncated requested windows."""

    mask = _window_mask(times_ms, window_ms)
    selected = times_ms[mask]
    if selected.size == 0:
        raise ValueError(f"{name} contains no samples.")

    interval = float(sampling_interval_ms)
    tolerance = max(1e-9, abs(interval) * 1e-7)
    start_gap = float(selected[0] - window_ms[0])
    end_gap = float(window_ms[1] - selected[-1])
    if start_gap < -tolerance or start_gap > interval + tolerance:
        raise ValueError(
            f"times_ms does not cover the requested {name} start "
            f"{window_ms[0]:g} ms; first included sample is "
            f"{selected[0]:g} ms."
        )
    if end_gap <= 0.0 or end_gap > interval + tolerance:
        raise ValueError(
            f"times_ms does not cover the requested {name} end "
            f"{window_ms[1]:g} ms; last included sample is "
            f"{selected[-1]:g} ms at dt={interval:g} ms."
        )
    return mask


def _sample_range(values: np.ndarray) -> tuple[float, float]:
    """Return first/last sample times, or NaNs for an empty selection."""

    samples = np.asarray(values, dtype=float).reshape(-1)
    if samples.size == 0:
        return (float("nan"), float("nan"))
    return (float(samples[0]), float(samples[-1]))


def _delay_embed(
    components: np.ndarray,
    *,
    dimension: int,
    delay_samples: int,
) -> np.ndarray:
    """Embed component time series using the reference lag convention."""

    n_components, n_times = components.shape
    cut = (dimension - 1) * delay_samples
    n_embedded_times = n_times - cut
    if n_embedded_times < 1:
        raise ValueError(
            "Time-delay embedding leaves no samples; reduce "
            "embedding_dimension or embedding_delay_samples."
        )

    embedded = np.empty(
        (n_components, dimension, n_embedded_times),
        dtype=float,
    )
    embedded[:, 0, :] = components[:, cut:]
    for lag_index in range(1, dimension):
        start = (dimension - lag_index - 1) * delay_samples
        stop = -lag_index * delay_samples
        embedded[:, lag_index, :] = components[:, start:stop]
    return embedded


def _pairwise_distances(states: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between columns of ``states``."""

    columns = np.asarray(states, dtype=float).T
    differences = columns[:, np.newaxis, :] - columns[np.newaxis, :, :]
    return np.linalg.norm(differences, ord=2, axis=2)


def _transition_rates(
    distances: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Count recurrence-state changes for every candidate threshold."""

    recurrence = distances[np.newaxis, :, :] <= thresholds[:, np.newaxis, np.newaxis]
    transitions = recurrence[:, :, 1:] != recurrence[:, :, :-1]
    n_samples = int(distances.shape[0])
    return transitions.sum(axis=(1, 2), dtype=float) / float(n_samples**2)


def _zero_result(
    *,
    singular_values: np.ndarray,
    explained_variance_percent: np.ndarray,
    cumulative_explained_variance_percent: np.ndarray,
    n_variance_components: int,
    snr_before_filter: np.ndarray,
    component_signals: np.ndarray,
    state_times_ms: np.ndarray,
    baseline_window_ms: tuple[float, float],
    response_window_ms: tuple[float, float],
    k: float,
    min_snr: float,
    max_var_percent: float,
    n_steps: int,
    embedded: bool,
    embedding_dimension: int,
    embedding_delay_samples: int,
    average_referenced: bool,
    baseline_corrected: bool,
    sampling_interval_ms: float,
) -> PCIStResult:
    base_mask = _window_mask(state_times_ms, baseline_window_ms)
    response_mask = _window_mask(state_times_ms, response_window_ms)
    baseline_times = state_times_ms[base_mask]
    response_times = state_times_ms[response_mask]
    empty_components = np.empty(0, dtype=float)
    empty_indices = np.empty(0, dtype=np.int64)
    empty_search = np.empty((n_steps, 0), dtype=float)
    return PCIStResult(
        pci_st=0.0,
        component_contributions=empty_components,
        n_components=0,
        n_variance_components=int(n_variance_components),
        selected_component_indices=empty_indices,
        singular_values=np.asarray(singular_values, dtype=float),
        explained_variance_percent=np.asarray(
            explained_variance_percent,
            dtype=float,
        ),
        cumulative_explained_variance_percent=np.asarray(
            cumulative_explained_variance_percent,
            dtype=float,
        ),
        snr_before_filter=np.asarray(snr_before_filter, dtype=float),
        retained_snrs=empty_components,
        component_signals=np.asarray(component_signals, dtype=float),
        thresholds=empty_search,
        nst_response=empty_search.copy(),
        nst_baseline=empty_search.copy(),
        nst_difference=empty_search.copy(),
        optimal_threshold_indices=empty_indices,
        optimal_thresholds=empty_components,
        optimal_nst_response=empty_components,
        optimal_nst_baseline=empty_components,
        baseline_window_ms=baseline_window_ms,
        response_window_ms=response_window_ms,
        n_baseline_samples=int(np.count_nonzero(base_mask)),
        n_response_samples=int(np.count_nonzero(response_mask)),
        baseline_sample_times_ms=baseline_times.copy(),
        response_sample_times_ms=response_times.copy(),
        k=float(k),
        min_snr=float(min_snr),
        max_var_percent=float(max_var_percent),
        n_steps=int(n_steps),
        embedded=bool(embedded),
        embedding_dimension=int(embedding_dimension),
        embedding_delay_samples=int(embedding_delay_samples),
        average_referenced=bool(average_referenced),
        baseline_corrected=bool(baseline_corrected),
        sampling_interval_ms=float(sampling_interval_ms),
        sampling_rate_hz=float(1000.0 / sampling_interval_ms),
        effective_baseline_sample_range_ms=_sample_range(baseline_times),
        effective_response_sample_range_ms=_sample_range(response_times),
    )


def pci_st(
    evoked: np.ndarray,
    times_ms: np.ndarray,
    *,
    baseline_window_ms: Sequence[float] = _DEFAULT_BASELINE_WINDOW_MS,
    response_window_ms: Sequence[float] = _DEFAULT_RESPONSE_WINDOW_MS,
    k: float = 1.2,
    min_snr: float = 1.1,
    max_var_percent: float = 99.0,
    n_steps: int = 100,
    max_threshold_fraction: float = 1.0,
    n_components: int | None = None,
    embed: bool = False,
    embedding_dimension: int = 1,
    embedding_delay_samples: int = 2,
    average_reference: bool = False,
    baseline_correct: bool = False,
    baseline_correction_end_ms: float = -50.0,
    return_details: bool = False,
) -> float | PCIStResult:
    """Calculate PCI-ST from a trial-averaged evoked response.

    Parameters
    ----------
    evoked
        Trial-averaged response with canonical shape ``(channels, time)``.
    times_ms
        Strictly increasing, uniformly spaced sample times in milliseconds.
        The requested baseline and response windows must be covered to within
        one sampling interval; materially truncated windows are rejected.
    baseline_window_ms
        Half-open baseline interval ``[start, end)``.  The default matches the
        authors' TMS/EEG reference implementation.
    response_window_ms
        Half-open response interval ``[start, end)``.  The default matches the
        authors' TMS/EEG reference implementation.
    k
        Baseline-transition penalty.  The published/reference default is 1.2.
    min_snr
        Retain response components with SNR strictly greater than this value.
    max_var_percent
        Retain the smallest leading set of SVD components explaining at least
        this percentage of response variance.
    n_steps
        Number of recurrence thresholds searched for each component.
    max_threshold_fraction
        Multiplier applied to the maximum response distance when defining the
        upper end of the threshold search.
    n_components
        Optional cap on SVD components before variance and SNR selection.
    embed
        Whether to use time-delay embedding for state-transition counting.
        The reference TMS/EEG default is ``False``.
    embedding_dimension
        Number of delay coordinates when ``embed=True``.
    embedding_delay_samples
        Delay between embedding coordinates, in samples.
    average_reference
        Subtract the instantaneous channel mean before calculation.  Disabled
        by default, matching the reference implementation.
    baseline_correct
        Subtract each channel's mean before ``baseline_correction_end_ms``.
        Disabled by default because empirical evoked inputs are normally
        preprocessed before PCI-ST.
    baseline_correction_end_ms
        End of the half-open interval used for optional baseline correction.
    return_details
        Return :class:`PCIStResult` instead of only the scalar score.

    Returns
    -------
    float or PCIStResult
        PCI-ST score or detailed diagnostics.  The score is nonnegative but is
        **not bounded above by one**.

    References
    ----------
    Comolatti, R. et al. (2019). A fast and general method to empirically
    estimate the complexity of brain responses to transcranial and
    intracranial stimulations. Brain Stimulation, 12, 1280-1289.
    https://doi.org/10.1016/j.brs.2019.05.013
    """

    signal, times = _validate_evoked_and_times(evoked, times_ms)
    sampling_interval_ms = _uniform_sampling_interval_ms(times)
    baseline_window = _validate_window(
        baseline_window_ms,
        name="baseline_window_ms",
    )
    response_window = _validate_window(
        response_window_ms,
        name="response_window_ms",
    )
    if baseline_window[1] > response_window[0]:
        raise ValueError("baseline and response windows must not overlap.")
    _validate_window_coverage(
        times,
        baseline_window,
        sampling_interval_ms=sampling_interval_ms,
        name="baseline_window_ms",
    )
    _validate_window_coverage(
        times,
        response_window,
        sampling_interval_ms=sampling_interval_ms,
        name="response_window_ms",
    )
    if not np.isfinite(k) or float(k) <= 1.0:
        raise ValueError("k must be finite and > 1.")
    if not np.isfinite(min_snr) or float(min_snr) < 0.0:
        raise ValueError("min_snr must be finite and >= 0.")
    if (
        not np.isfinite(max_var_percent)
        or not 0.0 < float(max_var_percent) <= 100.0
    ):
        raise ValueError("max_var_percent must be in (0, 100].")
    if int(n_steps) != n_steps or int(n_steps) < 2:
        raise ValueError("n_steps must be an integer >= 2.")
    n_steps = int(n_steps)
    if (
        not np.isfinite(max_threshold_fraction)
        or float(max_threshold_fraction) <= 0.0
    ):
        raise ValueError("max_threshold_fraction must be finite and > 0.")
    if n_components is not None:
        if int(n_components) != n_components or int(n_components) < 1:
            raise ValueError("n_components must be a positive integer or None.")
        n_components = int(n_components)
    if int(embedding_dimension) != embedding_dimension or int(embedding_dimension) < 1:
        raise ValueError("embedding_dimension must be a positive integer.")
    if (
        int(embedding_delay_samples) != embedding_delay_samples
        or int(embedding_delay_samples) < 1
    ):
        raise ValueError("embedding_delay_samples must be a positive integer.")
    embedding_dimension = int(embedding_dimension)
    embedding_delay_samples = int(embedding_delay_samples)

    prepared = signal.copy()
    if average_reference:
        prepared -= prepared.mean(axis=0, keepdims=True)
    if baseline_correct:
        correction_end = float(baseline_correction_end_ms)
        if not np.isfinite(correction_end):
            raise ValueError("baseline_correction_end_ms must be finite.")
        correction_mask = times < correction_end
        if not np.any(correction_mask):
            raise ValueError("Optional baseline-correction interval is empty.")
        prepared -= prepared[:, correction_mask].mean(axis=1, keepdims=True)

    crop_mask = (times >= baseline_window[0]) & (times < response_window[1])
    if np.count_nonzero(crop_mask) < 2:
        raise ValueError("Combined baseline/response crop contains too few samples.")
    prepared = prepared[:, crop_mask]
    cropped_times = times[crop_mask]

    baseline_mask = _window_mask(cropped_times, baseline_window)
    response_mask = _window_mask(cropped_times, response_window)
    n_baseline = int(np.count_nonzero(baseline_mask))
    n_response = int(np.count_nonzero(response_mask))
    if n_baseline <= 1 or n_response <= 1:
        raise ValueError(
            "Baseline and response windows must each contain at least two samples."
        )

    response_for_svd = prepared[:, response_mask].T
    _, all_singular_values, right_vectors_t = np.linalg.svd(
        response_for_svd,
        full_matrices=False,
    )
    available_components = int(all_singular_values.size)
    component_cap = (
        available_components
        if n_components is None
        else min(int(n_components), available_components)
    )
    singular_values = all_singular_values[:component_cap]
    spatial_basis = right_vectors_t.T[:, :component_cap]

    response_variance = np.square(singular_values)
    total_response_variance = float(response_variance.sum())
    # Do not use an arbitrary epsilon here: PCI-ST is invariant to a global
    # nonzero amplitude scaling, including signals expressed in very small
    # physical units.
    if total_response_variance == 0.0:
        state_times = cropped_times[
            (embedding_dimension - 1) * embedding_delay_samples :
        ] if embed else cropped_times
        result = _zero_result(
            singular_values=singular_values,
            explained_variance_percent=np.zeros_like(singular_values),
            cumulative_explained_variance_percent=np.zeros_like(singular_values),
            n_variance_components=0,
            snr_before_filter=np.empty(0, dtype=float),
            component_signals=np.empty((0, cropped_times.size), dtype=float),
            state_times_ms=state_times,
            baseline_window_ms=baseline_window,
            response_window_ms=response_window,
            k=float(k),
            min_snr=float(min_snr),
            max_var_percent=float(max_var_percent),
            n_steps=n_steps,
            embedded=bool(embed),
            embedding_dimension=embedding_dimension,
            embedding_delay_samples=embedding_delay_samples,
            average_referenced=bool(average_reference),
            baseline_corrected=bool(baseline_correct),
            sampling_interval_ms=sampling_interval_ms,
        )
        return result if return_details else result.pci_st

    explained_variance = 100.0 * response_variance / total_response_variance
    cumulative_variance = np.cumsum(explained_variance)
    if float(max_var_percent) == 100.0:
        n_variance_components = component_cap
    else:
        n_variance_components = int(
            np.searchsorted(
                cumulative_variance,
                float(max_var_percent),
                side="left",
            )
            + 1
        )

    projected = (prepared.T @ spatial_basis).T
    variance_components = projected[:n_variance_components]
    response_power = np.mean(
        np.square(variance_components[:, response_mask]),
        axis=1,
    )
    baseline_power = np.mean(
        np.square(variance_components[:, baseline_mask]),
        axis=1,
    )
    power_ratio = np.full(response_power.shape, np.nan, dtype=float)
    positive_baseline = baseline_power > 0.0
    np.divide(
        response_power,
        baseline_power,
        out=power_ratio,
        where=positive_baseline,
    )
    power_ratio[(~positive_baseline) & (response_power > 0.0)] = np.inf
    snr_before_filter = np.sqrt(power_ratio)
    snr_mask = snr_before_filter > float(min_snr)
    selected_component_indices = np.flatnonzero(snr_mask).astype(np.int64)
    retained_snrs = snr_before_filter[snr_mask]
    selected_components = variance_components[snr_mask]

    if selected_components.shape[0] == 0:
        state_times = cropped_times[
            (embedding_dimension - 1) * embedding_delay_samples :
        ] if embed else cropped_times
        result = _zero_result(
            singular_values=singular_values,
            explained_variance_percent=explained_variance,
            cumulative_explained_variance_percent=cumulative_variance,
            n_variance_components=n_variance_components,
            snr_before_filter=snr_before_filter,
            component_signals=selected_components,
            state_times_ms=state_times,
            baseline_window_ms=baseline_window,
            response_window_ms=response_window,
            k=float(k),
            min_snr=float(min_snr),
            max_var_percent=float(max_var_percent),
            n_steps=n_steps,
            embedded=bool(embed),
            embedding_dimension=embedding_dimension,
            embedding_delay_samples=embedding_delay_samples,
            average_referenced=bool(average_reference),
            baseline_corrected=bool(baseline_correct),
            sampling_interval_ms=sampling_interval_ms,
        )
        return result if return_details else result.pci_st

    if embed:
        states = _delay_embed(
            selected_components,
            dimension=embedding_dimension,
            delay_samples=embedding_delay_samples,
        )
        cut = (embedding_dimension - 1) * embedding_delay_samples
        state_times = cropped_times[cut:]
    else:
        states = selected_components[:, np.newaxis, :]
        state_times = cropped_times

    state_baseline_mask = _window_mask(state_times, baseline_window)
    state_response_mask = _window_mask(state_times, response_window)
    n_state_baseline = int(np.count_nonzero(state_baseline_mask))
    n_state_response = int(np.count_nonzero(state_response_mask))
    if n_state_baseline <= 1 or n_state_response <= 1:
        raise ValueError(
            "Embedding leaves fewer than two baseline or response samples."
        )

    n_retained = int(selected_components.shape[0])
    thresholds = np.empty((n_steps, n_retained), dtype=float)
    nst_baseline = np.empty_like(thresholds)
    nst_response = np.empty_like(thresholds)

    for component_index in range(n_retained):
        component_states = states[component_index]
        baseline_distances = _pairwise_distances(
            component_states[:, state_baseline_mask]
        )
        response_distances = _pairwise_distances(
            component_states[:, state_response_mask]
        )
        minimum_threshold = float(np.median(baseline_distances.ravel()))
        maximum_threshold = float(
            np.max(response_distances) * float(max_threshold_fraction)
        )
        component_thresholds = np.linspace(
            minimum_threshold,
            maximum_threshold,
            n_steps,
        )
        thresholds[:, component_index] = component_thresholds
        nst_baseline[:, component_index] = _transition_rates(
            baseline_distances,
            component_thresholds,
        )
        nst_response[:, component_index] = _transition_rates(
            response_distances,
            component_thresholds,
        )

    nst_difference = nst_response - float(k) * nst_baseline
    optimal_indices = np.argmax(nst_difference, axis=0).astype(np.int64)
    component_indices = np.arange(n_retained, dtype=np.int64)
    optimal_differences = nst_difference[optimal_indices, component_indices]
    contributions = np.maximum(
        0.0,
        optimal_differences * float(n_state_response),
    )
    score = float(np.sum(contributions))

    result = PCIStResult(
        pci_st=score,
        component_contributions=contributions,
        n_components=n_retained,
        n_variance_components=n_variance_components,
        selected_component_indices=selected_component_indices,
        singular_values=singular_values,
        explained_variance_percent=explained_variance,
        cumulative_explained_variance_percent=cumulative_variance,
        snr_before_filter=snr_before_filter,
        retained_snrs=retained_snrs,
        component_signals=selected_components,
        thresholds=thresholds,
        nst_response=nst_response,
        nst_baseline=nst_baseline,
        nst_difference=nst_difference,
        optimal_threshold_indices=optimal_indices,
        optimal_thresholds=thresholds[optimal_indices, component_indices],
        optimal_nst_response=nst_response[optimal_indices, component_indices],
        optimal_nst_baseline=nst_baseline[optimal_indices, component_indices],
        baseline_window_ms=baseline_window,
        response_window_ms=response_window,
        n_baseline_samples=n_state_baseline,
        n_response_samples=n_state_response,
        baseline_sample_times_ms=state_times[state_baseline_mask].copy(),
        response_sample_times_ms=state_times[state_response_mask].copy(),
        k=float(k),
        min_snr=float(min_snr),
        max_var_percent=float(max_var_percent),
        n_steps=n_steps,
        embedded=bool(embed),
        embedding_dimension=embedding_dimension,
        embedding_delay_samples=embedding_delay_samples,
        average_referenced=bool(average_reference),
        baseline_corrected=bool(baseline_correct),
        sampling_interval_ms=sampling_interval_ms,
        sampling_rate_hz=float(1000.0 / sampling_interval_ms),
        effective_baseline_sample_range_ms=_sample_range(
            state_times[state_baseline_mask]
        ),
        effective_response_sample_range_ms=_sample_range(
            state_times[state_response_mask]
        ),
    )
    return result if return_details else result.pci_st


def pci_st_from_trials(
    trials: np.ndarray,
    times_ms: np.ndarray,
    *,
    baseline_center_trials: bool = True,
    return_details: bool = False,
    **pci_st_kwargs,
) -> float | PCIStResult:
    """Average aligned trials and calculate one PCI-ST value.

    Parameters
    ----------
    trials
        Aligned trial-level data with explicit shape
        ``(trials, channels, time)``.
    times_ms
        Relative, stimulation-locked, uniformly spaced sample times in
        milliseconds.
    baseline_center_trials
        Subtract each trial/channel's mean over ``baseline_window_ms`` before
        averaging.  This is enabled by default for model output.  Set it to
        ``False`` when trials have already undergone empirical preprocessing.
    return_details
        Return :class:`PCIStResult` rather than a scalar.
    **pci_st_kwargs
        Forwarded to :func:`pci_st`.

    Notes
    -----
    PCI-ST is evaluated once on the trial average.  It is not calculated
    separately for each trial and then averaged.
    """

    trial_array = np.asarray(trials, dtype=float)
    times = np.asarray(times_ms, dtype=float)
    if trial_array.ndim != 3:
        raise ValueError(
            "trials must have shape (trials, channels, time); "
            f"received {trial_array.shape}."
        )
    if trial_array.shape[0] < 1:
        raise ValueError("trials must contain at least one trial.")
    if trial_array.shape[1] < 1 or trial_array.shape[2] < 2:
        raise ValueError("trials must contain channels and at least two samples.")
    if not np.isfinite(trial_array).all():
        raise ValueError("trials must contain only finite values.")
    if times.ndim != 1 or times.size != trial_array.shape[2]:
        raise ValueError(
            "times_ms must be one-dimensional and match the trial time axis."
        )
    if baseline_center_trials and bool(
        pci_st_kwargs.get("baseline_correct", False)
    ):
        raise ValueError(
            "Choose either baseline_center_trials=True or "
            "baseline_correct=True, not both."
        )

    prepared_trials = trial_array.copy()
    if baseline_center_trials:
        baseline_window = _validate_window(
            pci_st_kwargs.get(
                "baseline_window_ms",
                _DEFAULT_BASELINE_WINDOW_MS,
            ),
            name="baseline_window_ms",
        )
        baseline_mask = _window_mask(times, baseline_window)
        if not np.any(baseline_mask):
            raise ValueError("Trial baseline-centering interval is empty.")
        baseline_means = prepared_trials[:, :, baseline_mask].mean(
            axis=2,
            keepdims=True,
        )
        prepared_trials -= baseline_means

    evoked = prepared_trials.mean(axis=0)
    output = pci_st(
        evoked,
        times,
        return_details=return_details,
        **pci_st_kwargs,
    )
    if not return_details:
        return output

    if not isinstance(output, PCIStResult):
        raise AssertionError("Detailed PCI-ST calculation did not return PCIStResult.")
    return replace(
        output,
        n_trials_averaged=int(trial_array.shape[0]),
        trial_baseline_centered=bool(baseline_center_trials),
    )


__all__ = ["PCIStResult", "pci_st", "pci_st_from_trials"]
