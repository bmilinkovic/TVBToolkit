"""Publication-style plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "control": "#2E86AB",
    "mcs": "#E67E22",
    "uws": "#C0392B",
    "ketamine": "#F58518",
    "psilocybin": "#54A24B",
}


def _pci_metric_array(metrics: dict[str, Any]) -> np.ndarray:
    """Return Casali PCI array with backward-compatible fallback."""
    if "pci_casali_like" in metrics:
        return np.asarray(metrics["pci_casali_like"], dtype=float)
    return np.asarray(metrics["pci_like"], dtype=float)


def set_publication_style() -> None:
    """Set a clean, high-contrast plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F9FC",
            "axes.edgecolor": "#2A2A2A",
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.dpi": 200,
        }
    )


def _save_figure(fig, save_path: str | Path | None):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")


def plot_example_timeseries(
    metrics_by_condition: dict[str, dict[str, np.ndarray]],
    max_regions: int = 6,
    trim_first_ms: float = 0.0,
    save_path: str | Path | None = None,
):
    """Plot legacy-equivalent excitatory/inhibitory firing-rate trajectories.

    Uses `raw_example` as excitatory activity and `raw_inh_example` if present.
    Data are converted from kHz to Hz for consistency with legacy TVBSim plots.
    """
    set_publication_style()
    n = len(metrics_by_condition)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (cond, data) in zip(axes, metrics_by_condition.items()):
        t = np.asarray(data["time_ms_example"]) / 1000.0
        e_hz = 1000.0 * np.asarray(data["raw_example"], dtype=float)
        i_raw = np.asarray(data.get("raw_inh_example", np.array([])), dtype=float)
        i_hz = 1000.0 * i_raw if i_raw.size else None

        if trim_first_ms > 0:
            keep = t >= (trim_first_ms / 1000.0)
            t = t[keep]
            e_hz = e_hz[keep]
            if i_hz is not None and i_hz.ndim == 2 and i_hz.shape[0] == keep.shape[0]:
                i_hz = i_hz[keep]

        n_regions = min(max_regions, e_hz.shape[1])
        color_e = "#2B6CB0"
        color_i = "#C05621"

        for r in range(n_regions):
            ax.plot(t, e_hz[:, r], color=color_e, alpha=0.35, lw=1.0)
        ax.plot(t, e_hz[:, :n_regions].mean(axis=1), color=color_e, lw=2.2, label="Exc.")

        if i_hz is not None and i_hz.ndim == 2 and i_hz.shape[1] >= n_regions:
            for r in range(n_regions):
                ax.plot(t, i_hz[:, r], color=color_i, alpha=0.28, lw=0.9)
            ax.plot(t, i_hz[:, :n_regions].mean(axis=1), color=color_i, lw=2.0, label="Inh.")

        ax.set_title(f"{cond.capitalize()} firing-rate trajectories")
        ax.set_ylabel("Firing rate (Hz)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(bottom=min(-3.0, ax.get_ylim()[0]))
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def _coerce_single_region_payload(result: Any):
    """Extract canonical single-region timeseries arrays from object or dict."""
    if isinstance(result, dict):
        t = np.asarray(result["time_ms"], dtype=float)
        exc = np.asarray(result["exc_rate_hz"], dtype=float)
        inh = np.asarray(result["inh_rate_hz"], dtype=float)
        e_st = result.get("exc_spike_times_ms", None)
        e_si = result.get("exc_spike_indices", None)
        i_st = result.get("inh_spike_times_ms", None)
        i_si = result.get("inh_spike_indices", None)
    else:
        t = np.asarray(result.time_ms, dtype=float)
        exc = np.asarray(result.exc_rate_hz, dtype=float)
        inh = np.asarray(result.inh_rate_hz, dtype=float)
        e_st = getattr(result, "exc_spike_times_ms", None)
        e_si = getattr(result, "exc_spike_indices", None)
        i_st = getattr(result, "inh_spike_times_ms", None)
        i_si = getattr(result, "inh_spike_indices", None)

    return (
        t,
        exc,
        inh,
        None if e_st is None else np.asarray(e_st, dtype=float),
        None if e_si is None else np.asarray(e_si, dtype=int),
        None if i_st is None else np.asarray(i_st, dtype=float),
        None if i_si is None else np.asarray(i_si, dtype=int),
    )


def plot_single_region_timeseries(
    result: Any,
    source: str = "single_region_spiking",
    include_spikes: bool = True,
    max_spike_points: int = 60000,
    save_path: str | Path | None = None,
):
    """Plot single-region excitatory/inhibitory firing rates and optional spikes.

    Args:
        result: Output from `run_single_region_simulation()` or equivalent dict.
        source: `single_region_spiking` or `single_region_reduced`.
        include_spikes: If `True`, render rasters when spike arrays are available.
        max_spike_points: Maximum spike points per population to plot.
    """
    if source not in {"single_region_spiking", "single_region_reduced"}:
        raise ValueError("source must be 'single_region_spiking' or 'single_region_reduced'.")

    t_ms, exc_hz, inh_hz, e_st, e_si, i_st, i_si = _coerce_single_region_payload(result)
    t_s = t_ms / 1000.0

    has_spikes = (
        include_spikes
        and e_st is not None
        and e_si is not None
        and i_st is not None
        and i_si is not None
        and e_st.size > 0
        and i_st.size > 0
    )

    set_publication_style()
    if has_spikes:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
        )
        ax_rate, ax_exc_spk, ax_inh_spk = axes
    else:
        fig, ax_rate = plt.subplots(1, 1, figsize=(12, 4.3), sharex=True)
        ax_exc_spk = ax_inh_spk = None

    color_e = "#2B6CB0"
    color_i = "#C05621"
    ax_rate.plot(t_s, exc_hz, color=color_e, lw=2.1, label="Exc.")
    ax_rate.plot(t_s, inh_hz, color=color_i, lw=2.0, label="Inh.")
    ax_rate.set_ylabel("Firing rate (Hz)")
    ax_rate.set_title(
        "Single-Region Spiking Firing Rates"
        if source == "single_region_spiking"
        else "Single-Region Reduced Firing Rates"
    )
    ax_rate.legend(loc="upper right", fontsize=9)

    if has_spikes:
        if e_st.size > max_spike_points:
            step = int(np.ceil(e_st.size / max_spike_points))
            e_st_plot, e_si_plot = e_st[::step], e_si[::step]
        else:
            e_st_plot, e_si_plot = e_st, e_si
        if i_st.size > max_spike_points:
            step = int(np.ceil(i_st.size / max_spike_points))
            i_st_plot, i_si_plot = i_st[::step], i_si[::step]
        else:
            i_st_plot, i_si_plot = i_st, i_si

        ax_exc_spk.scatter(e_st_plot / 1000.0, e_si_plot, s=2.2, color=color_e, alpha=0.5, rasterized=True)
        ax_inh_spk.scatter(i_st_plot / 1000.0, i_si_plot, s=2.2, color=color_i, alpha=0.5, rasterized=True)
        ax_exc_spk.set_ylabel("Exc idx")
        ax_inh_spk.set_ylabel("Inh idx")
        ax_inh_spk.set_xlabel("Time (s)")
        ax_exc_spk.set_title("Excitatory Spike Raster", fontsize=11)
        ax_inh_spk.set_title("Inhibitory Spike Raster", fontsize=11)
    else:
        ax_rate.set_xlabel("Time (s)")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_timeseries(
    data: Any,
    source: str = "whole_brain",
    **kwargs,
):
    """Dispatch plotting across whole-brain and single-region simulation outputs.

    Args:
        data: Whole-brain batch metrics dict, or single-region simulation output.
        source: One of `whole_brain`, `single_region_spiking`, `single_region_reduced`.
    """
    if source == "whole_brain":
        return plot_example_timeseries(data, **kwargs)
    if source in {"single_region_spiking", "single_region_reduced"}:
        return plot_single_region_timeseries(data, source=source, **kwargs)
    raise ValueError(
        "Unsupported source. Use one of: "
        "'whole_brain', 'single_region_spiking', 'single_region_reduced'."
    )


def plot_metric_summary(
    metrics_by_condition: dict[str, dict[str, np.ndarray]],
    save_path: str | Path | None = None,
):
    """Plot LZc and Casali-style PCI distributions by condition."""
    set_publication_style()
    conditions = list(metrics_by_condition.keys())
    lzc_data = [np.asarray(metrics_by_condition[c]["lzc"]) for c in conditions]
    pci_data = [_pci_metric_array(metrics_by_condition[c]) for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [lzc_data, pci_data], ["LZc", "PCI (Casali-like)"]):
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(PALETTE.get(conditions[i], "#999999"))
            body.set_alpha(0.45)
            body.set_edgecolor("#1f1f1f")
        ax.scatter(
            np.repeat(np.arange(1, len(conditions) + 1), [len(v) for v in data]),
            np.concatenate(data),
            s=18,
            color="#1A1A1A",
            alpha=0.65,
            zorder=3,
        )
        ax.set_xticks(np.arange(1, len(conditions) + 1), [c.capitalize() for c in conditions], rotation=15)
        ax.set_title(title)
        ax.set_ylabel("Score")

    fig.suptitle("Complexity Metrics Across Conditions", y=1.02, fontsize=16)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_cohort_subject_metrics(
    cohort_results: dict[str, dict[str, dict[str, np.ndarray]]],
    save_path: str | Path | None = None,
):
    """Plot subject-level LZc and Casali-style PCI summaries across cohorts.

    Args:
        cohort_results: Mapping `{cohort: {subject_id: result_dict}}` as returned by
            repeated `run_cohort_batch(...)` calls.
    """
    set_publication_style()
    cohorts = list(cohort_results.keys())

    lzc_vals = []
    pci_vals = []
    for cohort in cohorts:
        subjects = cohort_results[cohort]
        lzc_vals.append(np.array([np.mean(np.asarray(v["lzc"], dtype=float)) for v in subjects.values()]))
        pci_vals.append(np.array([np.mean(_pci_metric_array(v)) for v in subjects.values()]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, title in zip(axes, [lzc_vals, pci_vals], ["LZc", "PCI (Casali-like)"]):
        parts = ax.violinplot(values, showmeans=True, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(PALETTE.get(cohorts[i], "#888888"))
            body.set_edgecolor("#111111")
            body.set_alpha(0.45)

        for i, arr in enumerate(values):
            x = np.full(arr.shape[0], i + 1, dtype=float)
            jitter = np.linspace(-0.08, 0.08, num=max(arr.shape[0], 1))
            ax.scatter(x + jitter[: arr.shape[0]], arr, s=16, color="#111111", alpha=0.65)

        ax.set_xticks(np.arange(1, len(cohorts) + 1), [c.upper() for c in cohorts])
        ax.set_title(title)
        ax.set_ylabel("Score")

    fig.suptitle("Subject-Level Complexity Across Brain-Act Cohorts", y=1.02)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig



def plot_brain_state_occupancy(
    cohort_results: dict[str, dict[str, dict[str, np.ndarray]]],
    save_path: str | Path | None = None,
):
    """Plot mean brain-state occupancy curves across cohorts."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for cohort, subjects in cohort_results.items():
        occ = []
        for res in subjects.values():
            arr = np.asarray(res.get("brain_state_occupancy", np.array([])), dtype=float)
            if arr.size == 0:
                continue
            occ.append(arr.mean(axis=0))
        if not occ:
            continue
        occ = np.asarray(occ, dtype=float)
        occ_mean = occ.mean(axis=0)
        occ_std = occ.std(axis=0)
        x = np.arange(1, occ_mean.size + 1)
        color = PALETTE.get(cohort, "#666666")
        ax.plot(x, occ_mean, marker="o", lw=2.0, color=color, label=cohort.upper())
        ax.fill_between(x, occ_mean - occ_std, occ_mean + occ_std, color=color, alpha=0.15)

    ax.set_xlabel("Brain state")
    ax.set_ylabel("Occupancy probability")
    ax.set_title("Brain-State Occupancy by Cohort")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_sfc_vs_occupancy(
    cohort_results: dict[str, dict[str, dict[str, np.ndarray]]],
    save_path: str | Path | None = None,
):
    """Plot brain-state occupancy as a function of state-level SF coupling.

    Uses per-seed state occupancies and per-seed state SF-coupling values from
    the Brain-Act subject workflow (`brain_state_occupancy_sfc_sorted`,
    `brain_state_sfc_sorted`).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cohort, subjects in cohort_results.items():
        xs = []
        ys = []
        for res in subjects.values():
            occ = np.asarray(res.get("brain_state_occupancy_sfc_sorted", np.array([])), dtype=float)
            sfc = np.asarray(res.get("brain_state_sfc_sorted", np.array([])), dtype=float)
            if occ.size == 0 or sfc.size == 0 or occ.shape != sfc.shape:
                continue
            xs.append(sfc.reshape(-1))
            ys.append(occ.reshape(-1))

        if not xs:
            continue

        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        color = PALETTE.get(cohort, "#666666")

        ax.scatter(x, y, s=18, alpha=0.35, color=color, edgecolors="none", label=f"{cohort.upper()} samples")
        if x.size >= 2 and np.std(x) > 0:
            p = np.polyfit(x, y, deg=1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            yy = p[0] * xx + p[1]
            ax.plot(xx, yy, color=color, lw=2.0, alpha=0.9, label=f"{cohort.upper()} trend")

    ax.set_xlabel("State SF coupling (Pearson r with subject SC)")
    ax.set_ylabel("State occupancy probability")
    ax.set_title("Brain-State Occupancy vs SF Coupling")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
