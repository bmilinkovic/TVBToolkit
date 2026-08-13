"""Create a transparent, synthetic walkthrough of simulation-adapted PCI-LZ.

This is an explanatory figure generator, not a scientific result.  It uses a
small synthetic firing-rate dataset so every intermediate object in the PCI
calculation can be inspected.  The significance step mirrors the
baseline-bootstrap estimator used for the corrected serotonergic PCI results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tvbtoolkit.complexity.pci_casali import (
    lz_complexity_2d,
    pci_norm_factor,
    sort_binJ,
    source_entropy,
)

COLORS = {
    "baseline": "#7A8796",
    "response": "#C75D26",
    "threshold": "#344765",
    "active": "#C75D26",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-rate-hz", type=float, default=1_000.0 / 3.0)
    parser.add_argument("--response-start-ms", type=float, default=8.0)
    parser.add_argument("--analysis-window-ms", type=float, default=300.0)
    parser.add_argument("--stim-duration-ms", type=float, default=10.0)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-sources", type=int, default=18)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pci_method_walkthrough"),
    )
    return parser.parse_args()


def configure_style() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in available else "Helvetica"
    mpl.rcParams.update(
        {
            "font.family": font,
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": f"{font}:italic",
            "mathtext.bf": f"{font}:bold",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    sns.set_theme(style="ticks", context="paper", font=font)


def make_synthetic_trials(
    *,
    n_trials: int,
    n_sources: int,
    sampling_rate_hz: float,
    analysis_window_ms: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return model-like trials shaped (trial, source, time)."""
    rng = np.random.default_rng(seed)
    dt_ms = 1_000.0 / sampling_rate_hz
    half_bins = round(analysis_window_ms / dt_ms)
    time_ms = (np.arange(2 * half_bins) - half_bins) * dt_ms

    innovations = rng.normal(0.0, 0.7, (n_trials, n_sources, 2 * half_bins))
    trials = np.empty_like(innovations)
    trials[..., 0] = innovations[..., 0]
    for index in range(1, innovations.shape[-1]):
        trials[..., index] = 0.72 * trials[..., index - 1] + innovations[..., index]

    # A reproducible sequence of local recruitment, propagation and recurrence.
    positive_time = np.maximum(time_ms, 0.0)
    for source in range(n_sources):
        latency = 12.0 + 7.5 * source
        width = 11.0 + 0.45 * source
        primary = np.exp(-0.5 * ((positive_time - latency) / width) ** 2)
        recurrence_latency = 118.0 + 3.0 * (source % 5)
        recurrence = np.exp(-0.5 * ((positive_time - recurrence_latency) / 24.0) ** 2)
        polarity = -1.0 if source % 4 == 3 else 1.0
        amplitude = 2.8 * np.exp(-source / 24.0)
        trial_gain = rng.normal(1.0, 0.10, (n_trials, 1))
        trials[:, source, :] += trial_gain * (
            polarity * amplitude * primary + 0.55 * amplitude * recurrence
        )

    return trials, time_ms, half_bins


def baseline_bootstrap_pci(
    trials: np.ndarray,
    *,
    onset_bin: int,
    sampling_rate_hz: float,
    response_start_ms: float,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    """Calculate PCI-LZ with a simple baseline maximum-statistic bootstrap."""
    baseline_mean = trials[:, :, :onset_bin].mean(axis=2, keepdims=True)
    centred = trials - baseline_mean
    baseline_sd = centred[:, :, :onset_bin].std(axis=(0, 2), ddof=1)
    baseline_sd = np.maximum(baseline_sd, np.finfo(float).eps)
    standardized = centred / baseline_sd[np.newaxis, :, np.newaxis]
    averaged = standardized.mean(axis=0)

    rng = np.random.default_rng(seed)
    baseline = standardized[:, :, :onset_bin]
    null_maxima = np.empty(n_bootstrap, dtype=float)
    for replicate in range(n_bootstrap):
        trial_indices = rng.integers(0, trials.shape[0], trials.shape[0])
        bootstrap_average = baseline[trial_indices].mean(axis=0)
        null_maxima[replicate] = np.abs(bootstrap_average).max()
    threshold = float(np.quantile(null_maxima, 1.0 - alpha))

    binary_full = (np.abs(averaged) > threshold).astype(np.uint8)
    dt_ms = 1_000.0 / sampling_rate_hz
    offset_bins = int(np.ceil(response_start_ms / dt_ms))
    effective_start_ms = offset_bins * dt_ms
    response = averaged[:, onset_bin + offset_bins :]
    binary = binary_full[:, onset_bin + offset_bins :]
    sorted_binary = sort_binJ(binary)
    entropy = float(source_entropy(sorted_binary))
    lz_count = int(lz_complexity_2d(sorted_binary)) if np.any(sorted_binary) else 0
    normalization = (
        float(pci_norm_factor(sorted_binary)) if np.any(sorted_binary) else 0.0
    )
    pci = lz_count / normalization if normalization > 0.0 else 0.0

    return {
        "standardized_trials": standardized,
        "trial_average": averaged,
        "response": response,
        "null_maxima": null_maxima,
        "threshold": threshold,
        "binary": binary,
        "sorted_binary": sorted_binary,
        "offset_bins": offset_bins,
        "effective_start_ms": effective_start_ms,
        "entropy": entropy,
        "lz_count": lz_count,
        "normalization": normalization,
        "pci": float(pci),
    }


def panel_letter(axis: mpl.axes.Axes, letter: str) -> None:
    axis.text(
        -0.15,
        1.08,
        letter,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def plot_alignment(
    axis: mpl.axes.Axes,
    trials: np.ndarray,
    time_ms: np.ndarray,
    onset_bin: int,
    stim_duration_ms: float,
    effective_start_ms: float,
) -> None:
    source = min(5, trials.shape[1] - 1)
    selected = trials[:20, source]
    for trace in selected:
        axis.plot(time_ms, trace, color=COLORS["baseline"], alpha=0.18, lw=0.55)
    axis.plot(
        time_ms,
        selected.mean(axis=0),
        color=COLORS["response"],
        lw=1.5,
        label="trial average",
    )
    axis.axvspan(0.0, stim_duration_ms, color="#E8B56D", alpha=0.28, lw=0)
    axis.axvline(time_ms[onset_bin], color="#222222", lw=0.8, ls="--")
    axis.axvline(
        effective_start_ms,
        color=COLORS["threshold"],
        lw=0.9,
        ls=":",
    )
    axis.set_xlim(-80, 220)
    axis.set_xlabel("Time from perturbation (ms)")
    axis.set_ylabel("Firing rate (a.u.)")
    axis.set_title("Trial alignment", loc="left")
    axis.legend(frameon=False, loc="upper right")
    axis.text(
        0.02,
        0.03,
        f"shading: {stim_duration_ms:g}-ms input; dotted: PCI starts {effective_start_ms:.2f} ms",
        transform=axis.transAxes,
        fontsize=5.5,
    )


def plot_average(
    axis: mpl.axes.Axes,
    result: dict[str, object],
    time_ms: np.ndarray,
    onset_bin: int,
    add_colorbar: bool = True,
) -> None:
    response = np.asarray(result["response"])
    effective_start = float(result["effective_start_ms"])
    response_time = time_ms[onset_bin + int(result["offset_bins"]) :]
    limit = float(np.quantile(np.abs(response), 0.99))
    image = axis.imshow(
        response,
        aspect="auto",
        origin="lower",
        extent=[response_time[0], response_time[-1], 1, response.shape[0]],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.axvline(effective_start, color="#222222", lw=0.8, ls="--")
    axis.set_xlabel("Time from perturbation (ms)")
    axis.set_ylabel("Model region")
    axis.set_title("Trial-averaged response", loc="left")
    if add_colorbar:
        colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        colorbar.set_label("Response (baseline SD)")
    else:
        axis.text(
            0.98,
            0.03,
            "colour: baseline SD",
            transform=axis.transAxes,
            ha="right",
            fontsize=5.5,
        )


def plot_threshold(
    axis: mpl.axes.Axes, result: dict[str, object], alpha: float
) -> None:
    maxima = np.asarray(result["null_maxima"])
    threshold = float(result["threshold"])
    sns.histplot(maxima, bins=30, color=COLORS["baseline"], ax=axis)
    axis.axvline(
        threshold,
        color=COLORS["threshold"],
        lw=1.4,
        label=f"{100 * (1 - alpha):g}th percentile",
    )
    axis.set_xlabel("Bootstrap max |response|")
    axis.set_ylabel("Bootstrap samples")
    axis.set_title("Baseline-bootstrap threshold", loc="left")
    axis.legend(frameon=False)


def plot_binary(
    axis: mpl.axes.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    duration_ms: float,
) -> None:
    cmap = mpl.colors.ListedColormap(["#F3F3F3", COLORS["active"]])
    axis.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[0, duration_ms, 1, matrix.shape[0]],
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_xlabel("Response time (ms)")
    axis.set_ylabel("Model region")
    axis.set_title(title, loc="left")


def plot_equation(axis: mpl.axes.Axes, result: dict[str, object]) -> None:
    axis.axis("off")
    entropy = float(result["entropy"])
    lz_count = int(result["lz_count"])
    normalization = float(result["normalization"])
    pci = float(result["pci"])
    active_fraction = float(np.asarray(result["sorted_binary"]).mean())
    axis.text(0.02, 0.92, "LZ complexity and normalization", fontsize=8, va="top")
    axis.text(
        0.02,
        0.72,
        r"$H=-p_1\log_2p_1-(1-p_1)\log_2(1-p_1)$",
        fontsize=8,
    )
    axis.text(
        0.02,
        0.52,
        r"$PCI=\dfrac{c_L\log_2 L}{L H}=\dfrac{c_L}{L H/\log_2 L}$",
        fontsize=9,
    )
    axis.text(
        0.02,
        0.27,
        (
            f"Active fraction = {active_fraction:.3f}\n"
            f"Source entropy H = {entropy:.3f}\n"
            f"Lempel–Ziv count cL = {lz_count}\n"
            f"Normalization = {normalization:.2f}\n"
            f"PCI = {pci:.3f}"
        ),
        va="top",
        linespacing=1.35,
    )


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def create_figures(
    trials: np.ndarray,
    time_ms: np.ndarray,
    onset_bin: int,
    result: dict[str, object],
    args: argparse.Namespace,
) -> None:
    configure_style()
    output = args.output_dir / "figures"
    response_duration = float(args.analysis_window_ms - result["effective_start_ms"])

    width = 183.0 / 25.4
    composite, axes = plt.subplots(2, 3, figsize=(width, 4.8))
    plot_alignment(
        axes[0, 0],
        trials,
        time_ms,
        onset_bin,
        args.stim_duration_ms,
        float(result["effective_start_ms"]),
    )
    plot_average(axes[0, 1], result, time_ms, onset_bin, False)
    plot_threshold(axes[0, 2], result, args.alpha)
    plot_binary(
        axes[1, 0],
        np.asarray(result["binary"]),
        title="Significant region–time matrix",
        duration_ms=response_duration,
    )
    plot_binary(
        axes[1, 1],
        np.asarray(result["sorted_binary"]),
        title="Regions sorted by activation",
        duration_ms=response_duration,
    )
    plot_equation(axes[1, 2], result)
    for letter, axis in zip("abcdef", axes.flat, strict=True):
        panel_letter(axis, letter)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    composite.subplots_adjust(hspace=0.55, wspace=0.42)
    save_figure(composite, output / "pci_method_walkthrough_composite")
    plt.close(composite)

    component_specs = [
        (
            "trial_alignment",
            plot_alignment,
            (
                trials,
                time_ms,
                onset_bin,
                args.stim_duration_ms,
                float(result["effective_start_ms"]),
            ),
        ),
        ("trial_average", plot_average, (result, time_ms, onset_bin)),
        ("bootstrap_threshold", plot_threshold, (result, args.alpha)),
    ]
    for name, function, function_args in component_specs:
        figure, axis = plt.subplots(figsize=(89.0 / 25.4, 2.4))
        function(axis, *function_args)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        save_figure(figure, output / f"pci_method_{name}")
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(width, 2.4))
    plot_binary(
        axes[0],
        np.asarray(result["binary"]),
        title="Significant activity",
        duration_ms=response_duration,
    )
    plot_binary(
        axes[1],
        np.asarray(result["sorted_binary"]),
        title="Sorted activity",
        duration_ms=response_duration,
    )
    plot_equation(axes[2], result)
    figure.subplots_adjust(wspace=0.45)
    save_figure(figure, output / "pci_method_binary_and_complexity")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.sampling_rate_hz <= 0.0:
        raise ValueError("--sampling-rate-hz must be positive.")
    if not 0.0 <= args.response_start_ms < args.analysis_window_ms:
        raise ValueError("--response-start-ms must lie inside the response window.")
    if args.n_trials < 2 or args.n_sources < 2:
        raise ValueError("The walkthrough requires at least two trials and sources.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie strictly between zero and one.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials, time_ms, onset_bin = make_synthetic_trials(
        n_trials=args.n_trials,
        n_sources=args.n_sources,
        sampling_rate_hz=args.sampling_rate_hz,
        analysis_window_ms=args.analysis_window_ms,
        seed=args.seed,
    )
    result = baseline_bootstrap_pci(
        trials,
        onset_bin=onset_bin,
        sampling_rate_hz=args.sampling_rate_hz,
        response_start_ms=args.response_start_ms,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed + 1,
    )
    create_figures(trials, time_ms, onset_bin, result, args)

    summary = {
        "warning": "Synthetic explanatory example; not a scientific result.",
        "significance_method": "baseline trial bootstrap maximum statistic",
        "sampling_rate_hz": args.sampling_rate_hz,
        "dt_ms": 1_000.0 / args.sampling_rate_hz,
        "analysis_window_ms": args.analysis_window_ms,
        "response_start_ms_requested": args.response_start_ms,
        "response_start_ms_effective": result["effective_start_ms"],
        "stim_duration_ms_shown_for_protocol_comparison": args.stim_duration_ms,
        "response_starts_before_stimulus_ends": bool(
            result["effective_start_ms"] < args.stim_duration_ms
        ),
        "n_trials": args.n_trials,
        "n_sources": args.n_sources,
        "n_response_samples": int(np.asarray(result["binary"]).shape[1]),
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "bootstrap_threshold": result["threshold"],
        "active_fraction": float(np.asarray(result["binary"]).mean()),
        "source_entropy": result["entropy"],
        "lz_count": result["lz_count"],
        "normalization": result["normalization"],
        "pci": result["pci"],
    }
    (args.output_dir / "walkthrough_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
