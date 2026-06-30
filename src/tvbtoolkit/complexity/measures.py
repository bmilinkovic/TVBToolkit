"""Complexity measures for whole-brain and single-region activity.

This module contains:
- Lempel-Ziv complexity (multichannel and single-channel variants)
- ACE/SCE entropy measures
- Casali-style PCI (ported workflow parity with TVBSim)
- A deprecated ratio proxy previously used as a lightweight PCI surrogate
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.signal import detrend, hilbert

from tvbtoolkit.complexity.pci_casali import (
    binarise_signals,
    binarise_signals_casali,
    lz_complexity_2d,
    pci_norm_factor,
    sort_binJ,
    source_entropy,
)


def _normalise_binarise_method(method: str) -> str:
    """Return the canonical binarisation route key or raise on typos."""
    key = str(method).lower()
    if key in ("tvbsim", "shuffle", "current"):
        return "tvbsim"
    if key in ("casali", "paper", "bootstrap"):
        return "casali"
    raise ValueError(f"Unknown binarise_method {method!r}; use 'tvbsim' or 'casali'.")


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure ``x`` is a real-valued 2D array.

    Parameters
    ----------
    x : np.ndarray
        Input array expected to represent time-series samples.

    Returns
    -------
    np.ndarray
        ``float`` array with exactly two dimensions.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected array with shape (time, channels).")
    return x



def _preprocess(x: np.ndarray) -> np.ndarray:
    """Mean-subtract then detrend along the time axis.

    Mirrors TVBSim's ``entropy_measures.preprocess``:
    ``signal.detrend(data - mean(data, axis=0), axis=0)``

    Parameters
    ----------
    x : np.ndarray
        Signal matrix with shape ``(time, channels)``.

    Returns
    -------
    np.ndarray
        Preprocessed float array of the same shape.
    """
    x = _ensure_2d(x)
    return detrend(x - x.mean(axis=0, keepdims=True), axis=0)



def _binarize_hilbert(x: np.ndarray) -> np.ndarray:
    """Binarize each channel by its mean Hilbert-envelope amplitude."""
    x = _ensure_2d(x)
    analytic = np.abs(hilbert(x, axis=0))
    thr = analytic.mean(axis=0, keepdims=True)
    return (analytic > thr).astype(np.uint8)



def _lz_complexity_binary_1d(bits: np.ndarray) -> int:
    """Lempel-Ziv 76 complexity for a 1D binary sequence."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return 0
    s = "".join(map(str, bits.tolist()))
    i = 0
    c = 0
    n = len(s)
    while i < n:
        l = 1
        while i + l <= n and s[i : i + l] in s[:i]:
            l += 1
        c += 1
        i += l
    return c



def lzc_multichannel(x: np.ndarray, *, shuffle_seed: int | None = 0) -> float:
    """Compute normalized multichannel Lempel-Ziv complexity.

    The pipeline mirrors TVBSim ``entropy_measures.calculate_LempelZiv``:

    1. **Preprocess**: mean-subtract then detrend each channel along the time
       axis (matches TVBSim ``preprocess``).
    2. **Binarize**: Hilbert-envelope amplitude > channel-wise mean of that
       envelope.
    3. **Flatten**: reshape ``(time, channels)`` → 1D in time-major order.
    4. **LZ76**: score the binary sequence with an LZ76 parser.
    5. **Normalize**: divide by the LZ76 score of a shuffled copy.

    Parameters
    ----------
    x : np.ndarray
        Signal matrix with shape ``(time, channels)``.
        Time is represented as discrete bins/samples; physical units depend on
        the simulation sampling interval used upstream.
    shuffle_seed : int or None, default=0
        Seed for the surrogate shuffle used in normalization.  The default is
        deterministic so HPC reruns are exactly reproducible.  Pass ``None``
        to use non-deterministic entropy from NumPy.

    Returns
    -------
    float
        Normalized multichannel LZ complexity.

    References
    ----------
    Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences.
    """
    b = _binarize_hilbert(_preprocess(x))
    seq = b.reshape(-1)
    c = _lz_complexity_binary_1d(seq)
    shuffled = np.copy(seq)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(shuffled)
    c_ref = max(_lz_complexity_binary_1d(shuffled), 1)
    return float(c / c_ref)



def lzc_single_channel(x: np.ndarray) -> float:
    """Compute mean single-channel Lempel-Ziv complexity.

    This applies :func:`lzc_multichannel` independently to each channel treated
    as a univariate sequence and returns the channel-average score.

    Parameters
    ----------
    x : np.ndarray
        Signal matrix with shape ``(time, channels)``.
        Time is represented as discrete bins/samples.

    Returns
    -------
    float
        Mean LZ complexity across channels.

    References
    ----------
    Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences.
    """
    x = _ensure_2d(x)
    vals = [lzc_multichannel(x[:, [k]]) for k in range(x.shape[1])]
    return float(np.mean(vals))



def _row_to_int(binary_row: np.ndarray) -> int:
    powers = (2 ** np.arange(binary_row.size)).astype(np.int64)
    return int((binary_row.astype(np.int64) * powers).sum())



def _shannon_entropy_from_symbols(symbols: np.ndarray) -> float:
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())



def ace(x: np.ndarray) -> float:
    """Compute Amplitude Coalition Entropy (ACE).

    ACE is computed from channel coalitions obtained by Hilbert-envelope
    binarization, followed by Shannon entropy over coalition symbols and
    normalization by a shuffled surrogate.

    Parameters
    ----------
    x : np.ndarray
        Signal matrix with shape ``(time, channels)``.
        Time is represented as discrete bins/samples.

    Returns
    -------
    float
        Normalized ACE score.

    References
    ----------
    Schartner, M. et al. (2015). Complexity of multi-dimensional spontaneous
    EEG decreases during propofol induced general anaesthesia.
    """
    b = _binarize_hilbert(x)
    symbols = np.array([_row_to_int(row) for row in b], dtype=np.int64)
    h_data = _shannon_entropy_from_symbols(symbols)
    shuffled = b.copy()
    for ch in range(shuffled.shape[1]):
        np.random.shuffle(shuffled[:, ch])
    symbols_ref = np.array([_row_to_int(row) for row in shuffled], dtype=np.int64)
    h_ref = max(_shannon_entropy_from_symbols(symbols_ref), 1e-12)
    return float(h_data / h_ref)



def sce(x: np.ndarray, threshold_rad: float = 0.8) -> float:
    """Compute Synchrony Coalition Entropy (SCE).

    Instantaneous phase is estimated by Hilbert transform. For each reference
    channel, a binary synchrony coalition is formed from phase differences to
    all other channels, entropy is computed over coalition symbols, and values
    are normalized by a random-coalition entropy reference.

    Parameters
    ----------
    x : np.ndarray
        Signal matrix with shape ``(time, channels)``.
        Time is represented as discrete bins/samples.
    threshold_rad : float, default=0.8
        Phase-difference threshold (radians) used for synchrony binarization.

    Returns
    -------
    float
        Normalized SCE score.

    References
    ----------
    Schartner, M. et al. (2015). Complexity of multi-dimensional spontaneous
    EEG decreases during propofol induced general anaesthesia.
    """
    x = _ensure_2d(x)
    phase = np.angle(hilbert(x, axis=0))
    n_time, n_ch = phase.shape
    sce_vals = []
    for i in range(n_ch):
        coalition = np.zeros((n_time, n_ch - 1), dtype=np.uint8)
        c = 0
        for j in range(n_ch):
            if i == j:
                continue
            dphi = np.abs(phase[:, i] - phase[:, j])
            dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
            coalition[:, c] = (dphi < threshold_rad).astype(np.uint8)
            c += 1
        symbols = np.array([_row_to_int(row) for row in coalition], dtype=np.int64)
        sce_vals.append(_shannon_entropy_from_symbols(symbols))

    rand = np.random.randint(0, 2, size=(n_time, max(n_ch - 1, 1)), dtype=np.uint8)
    rand_sym = np.array([_row_to_int(row) for row in rand], dtype=np.int64)
    h_ref = max(_shannon_entropy_from_symbols(rand_sym), 1e-12)
    return float(np.mean(sce_vals) / h_ref)



def _coerce_channels_time(x: np.ndarray, stimulation_index: int) -> np.ndarray:
    """Coerce input orientation to ``(channels, time)``.

    Parameters
    ----------
    x : np.ndarray
        Two-dimensional signal matrix that may be either ``(time, channels)`` or
        ``(channels, time)``.
    stimulation_index : int
        Stimulation onset index in bins. Used to infer likely time axis when
        shape-based inference is ambiguous.

    Returns
    -------
    np.ndarray
        Array with shape ``(channels, time)``.
    """
    arr = _ensure_2d(x)
    stim = int(stimulation_index)

    axis0_can_be_time = 0 <= stim < arr.shape[0]
    axis1_can_be_time = 0 <= stim < arr.shape[1]

    if axis0_can_be_time and not axis1_can_be_time:
        # Input most likely (time, channels).
        return arr.T
    if axis1_can_be_time and not axis0_can_be_time:
        # Input most likely (channels, time).
        return arr

    # Ambiguous: choose the longer axis as time.
    if arr.shape[0] >= arr.shape[1]:
        return arr.T
    return arr



def pci_casali_like(
    x: np.ndarray,
    stimulation_index: int,
    t_analysis_ms: float,
    *,
    dt_ms: float,
    nshuffles: int = 10,
    percentile: float = 100.0,
    use_post_only: bool = True,
    return_debug: bool = False,
    binarise_method: str = "tvbsim",
    binarise_kwargs: dict[str, Any] | None = None,
) -> float | dict[str, Any]:
    """Compute Casali-style PCI from source-level activity.

    This is the TVBToolkit parity implementation of the TVBSim Casali pipeline:

    ``binarise_signals -> sort_binJ -> lz_complexity_2d -> pci_norm_factor``

    The binarization step is selectable via ``binarise_method``:

    - ``"tvbsim"`` (default): the original shuffle-based route, unchanged.
    - ``"casali"`` : the paper-faithful bootstrap route
      (:func:`~tvbtoolkit.complexity.pci_casali.binarise_signals_casali`),
      matching how the empirical dataset's ``binJ`` was produced. Pass route
      parameters (``n_bootstrap``, ``alpha``, ``two_sided``, ``seed``) via
      ``binarise_kwargs``.

    Parameters
    ----------
    x : np.ndarray
        Source activity matrix, either ``(time, channels)`` or
        ``(channels, time)``.
    stimulation_index : int
        Stimulation onset index in **bins** on the time axis of ``x``.
    t_analysis_ms : float
        Duration of analysis window in **milliseconds** before and after onset.
        The extracted segment is ``[onset-nbins : onset+nbins]``.
    dt_ms : float
        Sampling interval in **milliseconds per bin**. Used for unit-safe
        conversion:
        ``nbins = int(round(t_analysis_ms / dt_ms))``.
    nshuffles : int, default=10
        Number of baseline shuffles used by ``binarise_signals``.
    percentile : float, default=100.0
        Percentile parameter used in TVBSim threshold indexing for surrogate
        maxima.
    use_post_only : bool, default=True
        If ``True``, compute PCI on post-stimulus bins only using
        ``binJ[:, nbins_analysis:]``.
    return_debug : bool, default=False
        If ``True``, return a diagnostic dictionary including entropy,
        normalization factor, sparsity, and intermediate shapes.

    Returns
    -------
    float or dict
        PCI scalar (default) or debug dictionary when ``return_debug=True``.

    Notes
    -----
    - This is a model-space Casali-style proxy intended for simulation outputs.
      It is not a complete clinical TMS-EEG PCI pipeline (no forward model,
      sensor-space reconstruction, or empirical preprocessing stack).
    - Units are handled explicitly: ``t_analysis_ms`` is never used directly as
      an array index; indexing is performed in bins via ``dt_ms``.

    References
    ----------
    Casali, A. G. et al. (2013). A theoretically based index of consciousness
    independent of sensory processing and behavior.
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0.")

    X = _coerce_channels_time(x, stimulation_index=stimulation_index)
    n_channels, n_time = X.shape
    onset = int(stimulation_index)

    nbins_analysis = int(round(float(t_analysis_ms) / float(dt_ms)))
    if nbins_analysis < 1:
        raise ValueError(
            "t_analysis_ms/dt_ms produced fewer than one analysis bin. "
            "Increase t_analysis_ms or decrease dt_ms."
        )

    start = onset - nbins_analysis
    stop = onset + nbins_analysis
    if start < 0 or stop > n_time:
        raise ValueError(
            "Invalid stimulation window bounds for signal length: "
            f"start={start}, stop={stop}, n_time={n_time}."
        )

    window = X[:, start:stop]
    bin_kw = dict(binarise_kwargs or {})
    method_key = _normalise_binarise_method(binarise_method)
    if method_key == "casali":
        # Paper-faithful route: returns a single (n_sources, n_bins) matrix.
        binJ = binarise_signals_casali(
            window[np.newaxis, :, :], t_stim=nbins_analysis, **bin_kw
        ).astype(np.uint8)
    elif method_key == "tvbsim":
        signal_binary = binarise_signals(
            window[np.newaxis, :, :],
            t_stim=nbins_analysis,
            nshuffles=int(nshuffles),
            percentile=float(percentile),
        )
        binJ = signal_binary.astype(np.uint8)[0]
    if use_post_only:
        # Important unit fix: slice using bins, not milliseconds.
        binJ = binJ[:, nbins_analysis:]

    binJs = sort_binJ(binJ)
    entropy_val = float(source_entropy(binJs))

    if np.any(binJs):
        lz_val = float(lz_complexity_2d(binJs))
        norm_val = float(pci_norm_factor(binJs))
        pci_val = float(lz_val / max(norm_val, np.finfo(float).eps))
    else:
        lz_val = 0.0
        norm_val = 0.0
        pci_val = 0.0

    if not return_debug:
        return pci_val

    return {
        "pci": pci_val,
        "lz": lz_val,
        "norm": norm_val,
        "entropy": entropy_val,
        "sparsity": float(np.mean(binJs)),
        "nbins_analysis": int(nbins_analysis),
        "dt_ms": float(dt_ms),
        "t_analysis_ms": float(t_analysis_ms),
        "shape_input": tuple(np.asarray(x).shape),
        "shape_channels_time": tuple(X.shape),
        "shape_window": tuple(window.shape),
        "shape_binary": tuple(binJs.shape),
        "use_post_only": bool(use_post_only),
        "binarise_method": method_key,
    }



def pci_casali_like_multi_trial(
    trials: "list[np.ndarray] | np.ndarray",
    stimulation_index: int,
    t_analysis_ms: float,
    *,
    dt_ms: float,
    nshuffles: int = 10,
    percentile: float = 100.0,
    binarise_method: str = "tvbsim",
    binarise_kwargs: dict[str, Any] | None = None,
) -> "tuple[float, np.ndarray]":
    """Compute Casali-style PCI from multiple stimulation trials.

    ``binarise_method`` selects the binarization route ("tvbsim" default, or
    "casali" for the paper-faithful bootstrap route). Under ``"casali"`` the
    trials are jointly reduced to a single trial-averaged ``binJ`` and one PCI
    value is returned (with ``pci_per_trial`` holding that single value),
    matching the empirical single-matrix convention.

    This is the exact multi-trial parity implementation matching TVBSim's
    ``_calculate_PCI_seed_subset`` and ``parallelized_PCI`` workflow.

    TVBSim uses ``n_trials=5`` (the ``n_trials`` argument, default 5) for each
    PCI estimate.  Each trial is a separate simulation run that shares the same
    stimulus time.  The trials are stacked into a 3D array of shape
    ``(n_trials, n_sources, 2*nbins_analysis)`` and jointly binarized using
    pre-stimulus baseline statistics pooled across all trials (matching
    ``binarise_signals``).  PCI is then computed for each trial individually on
    the post-stimulus window and the mean is returned.

    Key TVBSim parameter defaults replicated here:
    - ``t_analysis = 300`` ms  (half-window for pre/post)
    - ``n_trials   = 5``        (seeds per PCI call)
    - ``nshuffles  = 10``       (surrogate shuffles inside ``binarise_signals``)
    - ``percentile = 100``      (threshold percentile, TVBSim convention)

    Parameters
    ----------
    trials : list of np.ndarray or np.ndarray of shape (n_trials, ?, ?)
        One array per stimulation trial.  Each trial can be either
        ``(n_time, n_sources)`` or ``(n_sources, n_time)`` — the function
        auto-detects orientation using ``stimulation_index``.  Alternatively,
        pass a pre-stacked 3D array with shape ``(n_trials, n_sources, n_bins)``
        where the time window must already be cut to ``[onset-nbins:onset+nbins]``
        **and** ``stimulation_index`` should then be set to ``nbins_analysis``
        (i.e. the midpoint of the pre-cut window).
    stimulation_index : int
        Stimulus onset in **bins** on the time axis.  Must satisfy
        ``stimulation_index >= nbins_analysis`` and
        ``stimulation_index + nbins_analysis <= n_time`` for every trial.
    t_analysis_ms : float
        Half-window duration in **milliseconds** (identical pre and post).
        Converted to bins internally via ``nbins_analysis = int(round(t_analysis_ms / dt_ms))``.
    dt_ms : float
        Sampling interval in ms/bin (e.g. ``5.0`` for AdEx/MF outputs).
    nshuffles : int, default=10
        Number of pre-stimulus surrogate shuffles (TVBSim default: ``10``).
    percentile : float, default=100.0
        TVBSim-style threshold percentile for surrogate maxima indexing.

    Returns
    -------
    mean_pci : float
        Mean PCI across all trials (consistent with TVBSim reporting).
    pci_per_trial : np.ndarray
        Per-trial PCI values, shape ``(n_trials,)``.

    Notes
    -----
    For real Brain-Act / TVBSim parity each ``trial`` should come from a
    separate simulation run with the same stimulus injected at the same
    physical time.  If you only have spontaneous simulation outputs, you can
    approximate trials by sub-sampling non-overlapping windows or by re-running
    the simulation with different random seeds and the same external-input step
    applied at ``stimtime``.

    See Also
    --------
    pci_casali_like : single-trial version
    binarise_signals : baseline thresholding (TVBSim-ported)
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0.")

    nbins_analysis = int(round(float(t_analysis_ms) / float(dt_ms)))
    if nbins_analysis < 1:
        raise ValueError(
            "t_analysis_ms / dt_ms < 1 bin.  Increase t_analysis_ms or decrease dt_ms."
        )

    # ---- Normalise input to list of (n_sources, n_time) arrays ----
    if isinstance(trials, np.ndarray) and trials.ndim == 3:
        # Pre-stacked: (n_trials, n_sources, n_bins)
        trial_list = [trials[k] for k in range(trials.shape[0])]
    else:
        trial_list = list(trials)

    if len(trial_list) == 0:
        raise ValueError("trials must contain at least one trial array.")

    # Coerce each trial to (n_sources, n_time) using stimulation_index for orientation.
    coerced: list[np.ndarray] = []
    onset = int(stimulation_index)
    for t_arr in trial_list:
        t_arr = np.asarray(t_arr, dtype=float)
        if t_arr.ndim != 2:
            raise ValueError(
                f"Each trial must be 2D (time×sources or sources×time). Got shape {t_arr.shape}."
            )
        # If callers pass pre-cut peri-stimulus windows, the time axis has
        # exactly 2*nbins_analysis samples and onset is the midpoint.  Detect
        # that case explicitly because Brain-Act has 90 regions and only ~76
        # PCI bins; a "longer axis is time" heuristic would otherwise flip
        # source/time orientation incorrectly.
        if onset == nbins_analysis and t_arr.shape[1] == 2 * nbins_analysis:
            coerced.append(t_arr)
        elif onset == nbins_analysis and t_arr.shape[0] == 2 * nbins_analysis:
            coerced.append(t_arr.T)
        else:
            # _coerce_channels_time returns (channels, time) — reuse the
            # existing helper for full-length trials.
            coerced.append(_coerce_channels_time(t_arr, stimulation_index=onset))

    n_trials = len(coerced)
    n_sources = coerced[0].shape[0]

    # ---- Cut window [onset-nbins : onset+nbins] from each trial ----
    windows: list[np.ndarray] = []
    for k, src_time in enumerate(coerced):
        n_time = src_time.shape[1]
        start = onset - nbins_analysis
        stop = onset + nbins_analysis
        if start < 0 or stop > n_time:
            raise ValueError(
                f"Trial {k}: stimulus window [{start}, {stop}) out of bounds "
                f"for signal length {n_time}.  "
                f"onset={onset}, nbins_analysis={nbins_analysis}."
            )
        windows.append(src_time[:, start:stop])  # (n_sources, 2*nbins)

    # Stack into (n_trials, n_sources, 2*nbins_analysis) — matches TVBSim.
    stacked = np.stack(windows, axis=0)
    bin_kw = dict(binarise_kwargs or {})
    method_key = _normalise_binarise_method(binarise_method)

    if method_key == "casali":
        # Paper-faithful route: trials reduce to one trial-averaged binJ.
        binJ = binarise_signals_casali(
            stacked, t_stim=nbins_analysis, **bin_kw
        ).astype(np.uint8)[:, nbins_analysis:]
        binJs = sort_binJ(binJ)
        if np.any(binJs):
            pci = float(lz_complexity_2d(binJs) / max(pci_norm_factor(binJs), np.finfo(float).eps))
        else:
            pci = 0.0
        return pci, np.full(n_trials, pci, dtype=float)

    # ---- Joint binarization (pre-stimulus baseline pooled across trials) ----
    # t_stim = nbins_analysis  →  pre-stimulus is [:, :, :nbins_analysis]
    sig_all_binary = binarise_signals(
        stacked,
        t_stim=nbins_analysis,
        nshuffles=int(nshuffles),
        percentile=float(percentile),
    )

    # ---- PCI per trial on post-stimulus window ----
    pci_per_trial = np.empty(n_trials, dtype=float)
    for k in range(n_trials):
        # Exact TVBSim slice: binJ = sig_all_binary[k, :, nbins_analysis:]
        binJ = sig_all_binary.astype(np.uint8)[k, :, nbins_analysis:]

        binJs = sort_binJ(binJ)
        if not np.any(binJs):
            pci_per_trial[k] = 0.0
            continue

        lz_val = float(lz_complexity_2d(binJs))
        norm_val = float(pci_norm_factor(binJs))
        pci_per_trial[k] = float(lz_val / max(norm_val, np.finfo(float).eps))

    return float(np.mean(pci_per_trial)), pci_per_trial


def pci_ratio_proxy(
    x: np.ndarray,
    stimulation_index: int,
    post_window: int,
    pre_window: int | None = None,
) -> float:
    """Compute deprecated post/pre LZ ratio proxy.

    This metric is retained for backward compatibility only and should not be
    interpreted as Casali PCI.

    Parameters
    ----------
    x : np.ndarray
        Signal matrix with shape ``(time, channels)``.
        Time axis is discrete bins/samples.
    stimulation_index : int
        Time index (bins) of perturbation onset.
    post_window : int
        Number of post-onset bins used for complexity estimation.
    pre_window : int | None, default=None
        Number of pre-onset bins. If ``None``, uses ``post_window``.

    Returns
    -------
    float
        Ratio ``LZc(post) / LZc(pre)``.

    Notes
    -----
    This is **not** Casali PCI. It computes
    ``LZc(post) / LZc(pre)`` and can invert group ordering when denominator
    effects dominate. Prefer :func:`pci_casali_like`.
    """
    x = _ensure_2d(x)
    if pre_window is None:
        pre_window = post_window
    if stimulation_index - pre_window < 0 or stimulation_index + post_window > x.shape[0]:
        raise ValueError("Invalid stimulation window bounds for given signal length.")
    pre = x[stimulation_index - pre_window : stimulation_index]
    post = x[stimulation_index : stimulation_index + post_window]
    return float(lzc_multichannel(post) / max(lzc_multichannel(pre), 1e-12))



def pci_like(
    x: np.ndarray,
    stimulation_index: int,
    post_window: int,
    pre_window: int | None = None,
) -> float:
    """Backward-compatible alias for :func:`pci_ratio_proxy`.

    Deprecated
    ----------
    ``pci_like`` is a ratio proxy and not Casali PCI. Use
    :func:`pci_casali_like` for parity with the TVBSim PCI pipeline.
    """
    warnings.warn(
        "pci_like() is deprecated and remains a ratio-based proxy. "
        "Use pci_casali_like() for Casali-style PCI parity.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return pci_ratio_proxy(
        x=x,
        stimulation_index=stimulation_index,
        post_window=post_window,
        pre_window=pre_window,
    )
