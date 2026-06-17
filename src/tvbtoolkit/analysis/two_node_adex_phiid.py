"""Helpers for 2-node AdEx PhiID noise-sweep analysis."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import scipy.io

MEASURE_ORDER: tuple[str, ...] = ("mmi", "ccs", "idep_xta", "idep_xtb")
MEASURE_LABELS: dict[str, str] = {
    "mmi": "MMI",
    "ccs": "CCS",
    "idep_xta": "Idep XtA",
    "idep_xtb": "Idep XtB",
}
MEASURE_COLORS: dict[str, str] = {
    "mmi": "#466C95",
    "ccs": "#C46A4A",
    "idep_xta": "#6E8B61",
    "idep_xtb": "#8A5A8C",
}
REASONABLE_WEIGHT_NOISE_INTERVAL: tuple[float, float] = (5e-5, 3e-4)

_OUTPUT_RE = re.compile(r"^(?P<stub>.+)__phiid_(?P<measure>[A-Za-z0-9_]+)$")


def parse_two_node_output_name(path: str | Path) -> dict[str, str] | None:
    match = _OUTPUT_RE.match(Path(path).stem)
    if match is None:
        return None
    return match.groupdict()


def load_two_node_phiid_index(output_dir: str | Path, manifest_path: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    out = Path(output_dir).expanduser().resolve()
    for path in sorted(out.glob("*.mat")):
        meta = parse_two_node_output_name(path)
        if meta is None:
            continue
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "stub": str(meta["stub"]),
                "measure": str(meta["measure"]).lower(),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if manifest_path is not None:
        manifest = pd.read_csv(Path(manifest_path).expanduser().resolve())
        if not manifest.empty and "stub" in manifest.columns:
            df = df.merge(manifest, how="left", on="stub")
    return df.sort_values(["measure", "noise_value", "seed"]).reset_index(drop=True)


def load_two_node_phiid_output(path: str | Path) -> dict[str, float]:
    mat = scipy.io.loadmat(str(Path(path).expanduser().resolve()))
    return {
        "sts": float(np.asarray(mat["sts_val"], dtype=float).reshape(-1)[0]),
        "rtr": float(np.asarray(mat["rtr_val"], dtype=float).reshape(-1)[0]),
    }


def summarize_two_node_outputs(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if index_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw_rows: list[dict[str, Any]] = []
    for _, row in index_df.iterrows():
        values = load_two_node_phiid_output(row["path"])
        raw_rows.append(
            {
                "stub": row["stub"],
                "measure": row["measure"],
                "noise_value": float(row["noise_value"]),
                "g_value": float(row["g_value"]),
                "seed": int(row["seed"]),
                "sts": float(values["sts"]),
                "rtr": float(values["rtr"]),
            }
        )

    raw_df = pd.DataFrame(raw_rows).sort_values(["measure", "noise_value", "seed"]).reset_index(drop=True)
    summary = (
        raw_df.groupby(["measure", "noise_value", "g_value"], dropna=False)
        .agg(
            sts_mean=("sts", "mean"),
            sts_sem=("sts", _sem),
            rtr_mean=("rtr", "mean"),
            rtr_sem=("rtr", _sem),
            n_replicates=("seed", "count"),
        )
        .reset_index()
    )
    measure_rank = {name: idx for idx, name in enumerate(MEASURE_ORDER)}
    summary["_measure_rank"] = summary["measure"].map(measure_rank).fillna(999)
    summary = summary.sort_values(["_measure_rank", "noise_value"]).drop(columns="_measure_rank").reset_index(drop=True)
    summary["measure_label"] = summary["measure"].map(MEASURE_LABELS).fillna(summary["measure"].str.upper())
    return raw_df, summary


def _sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _add_reasonable_noise_band(ax: plt.Axes) -> None:
    lo, hi = REASONABLE_WEIGHT_NOISE_INTERVAL
    ax.axvspan(lo, hi, color="#B8BDC7", alpha=0.18, zorder=0)


def _heat_cmap(atom: str) -> mcolors.Colormap:
    if atom == "sts":
        colors = ["#FBF5E9", "#EBCB8B", "#D88C4A", "#B65D4A", "#7C2F39"]
    else:
        colors = ["#F7F1E3", "#D9D4C7", "#B7C2C8", "#7B9AA6", "#3E5C67"]
    cmap = mcolors.LinearSegmentedColormap.from_list(f"two_node_{atom}", colors)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def plot_two_node_publication(summary_df: pd.DataFrame, *, g_value: float) -> tuple[plt.Figure, np.ndarray]:
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    atoms = (("sts", "Synergy (STS)"), ("rtr", "Redundancy (RTR)"))
    measures = [m for m in MEASURE_ORDER if m in summary_df["measure"].dropna().unique().tolist()]

    for ax, (atom, title) in zip(axes, atoms, strict=False):
        _add_reasonable_noise_band(ax)
        mean_col = f"{atom}_mean"
        sem_col = f"{atom}_sem"
        for measure in measures:
            sub = summary_df.loc[summary_df["measure"] == measure].copy()
            if sub.empty:
                continue
            color = MEASURE_COLORS.get(measure, "#4C566A")
            label = MEASURE_LABELS.get(measure, measure.upper())
            ax.fill_between(
                sub["noise_value"],
                sub[mean_col] - sub[sem_col],
                sub[mean_col] + sub[sem_col],
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
            ax.plot(
                sub["noise_value"],
                sub[mean_col],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.8,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Dynamical noise (weight_noise)")
        ax.set_ylabel("PhiID atom value")
        ax.grid(True, axis="y", color="#D8D2C4", linewidth=0.6, alpha=0.6)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Two-node AdEx PhiID under increasing dynamical noise (G = {g_value:g})", y=1.02, fontsize=12.0)
    fig.tight_layout()
    return fig, axes


def plot_two_node_sts_only(summary_df: pd.DataFrame, *, g_value: float) -> tuple[plt.Figure, plt.Axes]:
    _set_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    measures = [m for m in MEASURE_ORDER if m in summary_df["measure"].dropna().unique().tolist()]
    _add_reasonable_noise_band(ax)
    for measure in measures:
        sub = summary_df.loc[summary_df["measure"] == measure].copy()
        if sub.empty:
            continue
        color = MEASURE_COLORS.get(measure, "#4C566A")
        label = MEASURE_LABELS.get(measure, measure.upper())
        ax.fill_between(
            sub["noise_value"],
            sub["sts_mean"] - sub["sts_sem"],
            sub["sts_mean"] + sub["sts_sem"],
            color=color,
            alpha=0.20,
            linewidth=0.0,
        )
        ax.plot(
            sub["noise_value"],
            sub["sts_mean"],
            color=color,
            linewidth=2.6,
            marker="o",
            markersize=5.2,
            label=label,
        )
    ax.set_title(f"Synergy (STS) under increasing dynamical noise (G = {g_value:g})")
    ax.set_xlabel("Dynamical noise (weight_noise)")
    ax.set_ylabel("PhiID synergy (STS)")
    ax.grid(True, axis="y", color="#D8D2C4", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax


def plot_two_node_replicates(raw_df: pd.DataFrame, summary_df: pd.DataFrame, *, g_value: float) -> tuple[plt.Figure, np.ndarray]:
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    atoms = (("sts", "Synergy (STS)"), ("rtr", "Redundancy (RTR)"))
    measures = [m for m in MEASURE_ORDER if m in summary_df["measure"].dropna().unique().tolist()]

    for ax, (atom, title) in zip(axes, atoms, strict=False):
        _add_reasonable_noise_band(ax)
        for measure in measures:
            color = MEASURE_COLORS.get(measure, "#4C566A")
            raw = raw_df.loc[raw_df["measure"] == measure].copy()
            for _, rep in raw.groupby("seed", dropna=False):
                rep = rep.sort_values("noise_value")
                ax.plot(rep["noise_value"], rep[atom], color=color, alpha=0.14, linewidth=0.9)
            sub = summary_df.loc[summary_df["measure"] == measure].copy()
            if not sub.empty:
                label = MEASURE_LABELS.get(measure, measure.upper())
                ax.plot(
                    sub["noise_value"],
                    sub[f"{atom}_mean"],
                    color=color,
                    linewidth=2.2,
                    marker="o",
                    markersize=4.8,
                    label=label,
                )
        ax.set_title(title)
        ax.set_xlabel("Dynamical noise (weight_noise)")
        ax.set_ylabel("PhiID atom value")
        ax.grid(True, axis="y", color="#D8D2C4", linewidth=0.6, alpha=0.6)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Two-node AdEx replicate trajectories across dynamical noise (G = {g_value:g})", y=1.02, fontsize=12.0)
    fig.tight_layout()
    return fig, axes


def plot_two_node_g_noise_heatmaps(
    summary_df: pd.DataFrame,
    *,
    measure: str,
    g_values: list[float],
    noise_values: list[float],
) -> tuple[plt.Figure, np.ndarray]:
    _set_style()
    measure = str(measure).lower()
    subset = summary_df.loc[summary_df["measure"] == measure].copy()
    subset["g_round"] = subset["g_value"].round(12)
    subset["noise_round"] = subset["noise_value"].round(12)
    g_sorted = sorted({round(float(x), 12) for x in g_values} | set(subset["g_round"].tolist()))
    noise_sorted = sorted({round(float(x), 12) for x in noise_values} | set(subset["noise_round"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), squeeze=False, constrained_layout=True)
    atoms = (("sts_mean", "STS"), ("rtr_mean", "RTR"))

    def _tick_positions(values: list[float], n_target: int = 6) -> tuple[list[int], list[str]]:
        if len(values) <= n_target:
            idx = list(range(len(values)))
        else:
            idx = np.linspace(0, len(values) - 1, n_target, dtype=int).tolist()
        labels = [f"{values[i]:g}" for i in idx]
        return idx, labels

    for ax, (value_col, title) in zip(axes.ravel(), atoms, strict=False):
        pivot = (
            subset.pivot(index="noise_round", columns="g_round", values=value_col)
            .reindex(index=noise_sorted, columns=g_sorted)
        )
        data = pivot.to_numpy(dtype=float)
        valid = data[np.isfinite(data)]
        vmax = float(valid.max()) if valid.size else 1.0
        vmin = float(valid.min()) if valid.size else 0.0
        if title == "RTR":
            vmin = 0.0
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=_heat_cmap(title.lower()), vmin=vmin, vmax=vmax)
        ax.set_title(f"{MEASURE_LABELS.get(measure, measure.upper())} {title}")
        ax.set_xlabel("Global coupling G")
        ax.set_ylabel("Dynamical noise (weight_noise)")
        x_idx, x_labels = _tick_positions(g_sorted)
        y_idx, y_labels = _tick_positions(noise_sorted)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(y_idx)
        ax.set_yticklabels(y_labels)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{title} value")

    fig.suptitle(f"Two-node AdEx PhiID across global coupling and dynamical noise ({MEASURE_LABELS.get(measure, measure.upper())})", y=1.02, fontsize=12.0)
    return fig, axes


def plot_two_node_g_noise_heatmaps_publication(
    summary_df: pd.DataFrame,
    *,
    measure: str,
    g_values: list[float],
    noise_values: list[float],
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    _set_style()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )
    measure = str(measure).lower()
    subset = summary_df.loc[summary_df["measure"] == measure].copy()
    subset["g_round"] = subset["g_value"].round(12)
    subset["noise_round"] = subset["noise_value"].round(12)
    g_sorted = sorted({round(float(x), 12) for x in g_values} | set(subset["g_round"].tolist()))
    noise_sorted = sorted({round(float(x), 12) for x in noise_values} | set(subset["noise_round"].tolist()))
    fig = plt.figure(figsize=(11.0, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.055, 1.0, 0.055], wspace=0.06)
    axes = np.empty((1, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 2], sharex=axes[0, 0], sharey=axes[0, 0])
    atoms = (
        ("sts_mean", "Synergy (STS)", "sts"),
        ("rtr_mean", "Redundancy (RTR)", "rtr"),
    )

    def _tick_positions(values: list[float], n_target: int = 5) -> tuple[list[int], list[str]]:
        if len(values) <= n_target:
            idx = list(range(len(values)))
        else:
            idx = np.linspace(0, len(values) - 1, n_target, dtype=int).tolist()
        labels = [f"{values[i]:.2g}" for i in idx]
        return idx, labels

    x_idx, x_labels = _tick_positions(g_sorted)
    y_idx, y_labels = _tick_positions(noise_sorted)

    for ax, cax, (value_col, panel_title, atom_name) in zip(
        axes.ravel(),
        [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])],
        atoms,
        strict=False,
    ):
        pivot = (
            subset.pivot(index="noise_round", columns="g_round", values=value_col)
            .reindex(index=noise_sorted, columns=g_sorted)
        )
        data = pivot.to_numpy(dtype=float)
        valid = data[np.isfinite(data)]
        vmax = float(valid.max()) if valid.size else 1.0
        vmin = float(valid.min()) if valid.size else 0.0
        if atom_name == "rtr":
            vmin = 0.0
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=_heat_cmap(atom_name),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel_title, pad=10)
        ax.set_xlabel("Global coupling G")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_idx)
        ax.set_yticklabels(y_labels)
        ax.tick_params(length=3.5, width=0.9)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("STS" if atom_name == "sts" else "RTR", fontsize=10.5)
        cbar.ax.tick_params(labelsize=9.0)

    axes[0, 0].set_ylabel("Dynamical noise")
    axes[0, 1].set_ylabel("")
    axes[0, 1].tick_params(labelleft=False)

    if title:
        fig.suptitle(title, y=1.02, fontsize=14.0, fontweight="bold")
    return fig, axes
