"""Whole-brain plotting and analysis helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tvbtoolkit.bold import BOLDParams, bold_from_firing_rates, corr_fc_sc, preprocess_bold_signal
from tvbtoolkit.core.config import OutputConfig
from tvbtoolkit.whole_brain.simulation import WholeBrainResult


def plot_region_timeseries(
    result: WholeBrainResult,
    region_indices: list[int] | None = None,
    max_regions: int = 8,
    figsize: tuple[int, int] = (10, 5),
):
    """Plot selected regional time series from a `WholeBrainResult`."""
    if region_indices is None:
        region_indices = list(range(min(max_regions, result.raw.shape[1])))
    fig, ax = plt.subplots(figsize=figsize)
    for idx in region_indices:
        ax.plot(result.time_ms / 1000.0, result.raw[:, idx], label=str(result.region_labels[idx]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity (a.u.)")
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    ax.set_title("Whole-brain regional trajectories")
    fig.tight_layout()
    return fig


def correlation_fc(result: WholeBrainResult) -> np.ndarray:
    """Compute a simple Pearson FC matrix from regional time series."""
    return np.corrcoef(result.raw.T)


def fcsc_seedwise_from_saved_batch(
    output: OutputConfig,
    *,
    conditions: list[str],
    seeds: list[int],
    structural_connectivity: np.ndarray,
    cut_transient_ms: float,
    tr_ms: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute FC-SC coupling per condition/seed from saved condition-batch files.

    Parameters
    ----------
    output : OutputConfig
        Output configuration used during `run_condition_batch`.
    conditions : list[str]
        Condition names to analyze.
    seeds : list[int]
        Seed list to include.
    structural_connectivity : ndarray
        SC matrix of shape `(regions, regions)`.
    cut_transient_ms : float
        Initial transient removed before analysis.
    tr_ms : float
        Sampling interval for BOLD conversion in milliseconds.

    Returns
    -------
    dict[str, dict[str, ndarray]]
        Per-condition arrays:
        - `legacy_r_signed_full`
        - `masked_r_abs_upper`
        - `masked_r_signed_upper`
    """
    sc = np.asarray(structural_connectivity, dtype=float)
    if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
        raise ValueError("structural_connectivity must be square.")
    iu = np.triu_indices_from(sc, k=1)
    mask_sc = sc[iu] > 0

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        xa = np.asarray(a, dtype=float).reshape(-1)
        xb = np.asarray(b, dtype=float).reshape(-1)
        if xa.size != xb.size or xa.size < 3:
            return float("nan")
        if float(np.std(xa)) <= 0.0 or float(np.std(xb)) <= 0.0:
            return float("nan")
        return float(np.corrcoef(xa, xb)[0, 1])

    out: dict[str, dict[str, np.ndarray]] = {}
    bp = BOLDParams(TR=float(tr_ms) / 1000.0)

    for cond in conditions:
        cond_dir = output.simulations_dir / cond
        rows = []
        for seed in seeds:
            sf = cond_dir / f"seed_{int(seed):03d}.npz"
            if not sf.exists():
                continue
            d = np.load(sf, allow_pickle=True)
            t_ms = np.asarray(d["time_ms"], dtype=float)
            raw = np.asarray(d["raw"], dtype=float)

            keep = t_ms >= float(cut_transient_ms)
            t_ms = t_ms[keep]
            raw = raw[keep]
            if t_ms.size < 10:
                continue

            dt_ms = float(np.median(np.diff(t_ms))) if t_ms.size > 1 else 1.0
            bold = bold_from_firing_rates(raw, dt_ms=dt_ms, tr_ms=float(tr_ms))
            bold_pp = preprocess_bold_signal(
                bold,
                params=bp,
                apply_zscore=True,
                apply_bandpass=True,
                n_regions_hint=sc.shape[0],
            )
            if bold_pp.shape[1] != sc.shape[0] and bold_pp.shape[0] == sc.shape[0]:
                bold_pp = bold_pp.T
            if bold_pp.shape[1] != sc.shape[0]:
                raise ValueError(
                    f"BOLD shape {bold_pp.shape} incompatible with SC shape {sc.shape} after preprocessing."
                )

            fc_abs, legacy = corr_fc_sc(bold_pp, sc)
            fc_signed = np.corrcoef(bold_pp.T)
            masked_abs = _safe_corr(fc_abs[iu][mask_sc], sc[iu][mask_sc])
            masked_signed = _safe_corr(fc_signed[iu][mask_sc], sc[iu][mask_sc])
            rows.append((legacy, masked_abs, masked_signed))

        arr = np.asarray(rows, dtype=float) if rows else np.empty((0, 3), dtype=float)
        out[cond] = {
            "legacy_r_signed_full": arr[:, 0] if arr.size else np.array([], dtype=float),
            "masked_r_abs_upper": arr[:, 1] if arr.size else np.array([], dtype=float),
            "masked_r_signed_upper": arr[:, 2] if arr.size else np.array([], dtype=float),
        }

    return out
