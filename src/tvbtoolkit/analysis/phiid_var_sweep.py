"""Helpers for bivariate VAR PhiID noise-sweep simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def _measures_present(frame: pd.DataFrame) -> list[str]:
    present = [str(x).lower() for x in frame.get("measure", pd.Series(dtype=str)).dropna().unique().tolist()]
    ordered = [measure for measure in MEASURE_ORDER if measure in present]
    extras = sorted([measure for measure in present if measure not in ordered])
    return ordered + extras


def _to_string_list(value: Any) -> list[str]:
    arr = np.asarray(value, dtype=object).reshape(-1)
    out: list[str] = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
            continue
        if isinstance(item, np.ndarray):
            flat = np.asarray(item).reshape(-1)
            if flat.size == 1:
                out.append(str(flat.item()))
            else:
                out.append("".join(str(x) for x in flat.tolist()))
            continue
        out.append(str(item))
    return out


def load_var_noise_sweep(path: str | Path) -> dict[str, Any]:
    """Load the saved MATLAB results for the bivariate VAR PhiID sweep."""
    mat = scipy.io.loadmat(str(Path(path).expanduser().resolve()), squeeze_me=True, struct_as_record=False)

    cross_like = mat["cross_coef"] if "cross_coef" in mat else mat.get("interaction_coef", np.nan)
    innovation_like = mat["innovation_sd"] if "innovation_sd" in mat else np.nan
    shared_like = mat["common_noise_fraction"] if "common_noise_fraction" in mat else np.nan

    sts_values = np.asarray(mat["sts_values"], dtype=float)
    rtr_values = np.asarray(mat["rtr_values"], dtype=float)
    status_codes = np.asarray(mat["status_codes"], dtype=int)
    if sts_values.ndim == 2:
        sts_values = sts_values[:, :, np.newaxis]
    if rtr_values.ndim == 2:
        rtr_values = rtr_values[:, :, np.newaxis]
    if status_codes.ndim == 2:
        status_codes = status_codes[:, :, np.newaxis]

    return {
        "noise_levels": np.atleast_1d(np.asarray(mat["noise_levels"], dtype=float)).reshape(-1),
        "measures": _to_string_list(mat["measures"]),
        "sts_values": sts_values,
        "rtr_values": rtr_values,
        "status_codes": status_codes,
        "job_seed_matrix": np.asarray(mat["job_seed_matrix"], dtype=int),
        "self_coef": float(np.asarray(mat["self_coef"], dtype=float)),
        "cross_coef": float(np.asarray(cross_like, dtype=float)),
        "innovation_sd": float(np.asarray(innovation_like, dtype=float)),
        "n_timepoints": int(np.asarray(mat["n_timepoints"], dtype=int)),
        "burnin": int(np.asarray(mat["burnin"], dtype=int)),
        "tau": int(np.asarray(mat["tau"], dtype=int)),
        "common_noise_fraction": float(np.asarray(shared_like, dtype=float)),
    }


def sweep_long_form(results: dict[str, Any]) -> pd.DataFrame:
    """Return one row per measure, noise level, and replicate."""
    measures = [str(x).lower() for x in results["measures"]]
    noise_levels = np.asarray(results["noise_levels"], dtype=float).reshape(-1)
    sts_values = np.asarray(results["sts_values"], dtype=float)
    rtr_values = np.asarray(results["rtr_values"], dtype=float)
    status_codes = np.asarray(results["status_codes"], dtype=int)
    seed_matrix = np.asarray(results["job_seed_matrix"], dtype=int)

    rows: list[dict[str, Any]] = []
    for measure_idx, measure in enumerate(measures):
        for noise_idx, noise_level in enumerate(noise_levels):
            for rep_idx in range(seed_matrix.shape[1]):
                rows.append(
                    {
                        "measure": measure,
                        "measure_label": MEASURE_LABELS.get(measure, measure.upper()),
                        "noise_level": float(noise_level),
                        "replicate": int(rep_idx + 1),
                        "seed": int(seed_matrix[noise_idx, rep_idx]),
                        "sts": float(sts_values[noise_idx, rep_idx, measure_idx]),
                        "rtr": float(rtr_values[noise_idx, rep_idx, measure_idx]),
                        "status_code": int(status_codes[noise_idx, rep_idx, measure_idx]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate means and SEMs by measure and noise level."""
    if df.empty:
        return pd.DataFrame()

    def _sem(x: pd.Series) -> float:
        arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return 0.0
        return float(arr.std(ddof=1) / np.sqrt(arr.size))

    summary = (
        df.groupby(["measure", "measure_label", "noise_level"], dropna=False)
        .agg(
            sts_mean=("sts", "mean"),
            sts_sem=("sts", _sem),
            rtr_mean=("rtr", "mean"),
            rtr_sem=("rtr", _sem),
            n_replicates=("replicate", "count"),
        )
        .reset_index()
    )
    measure_rank = {name: idx for idx, name in enumerate(MEASURE_ORDER)}
    summary["_measure_rank"] = summary["measure"].map(measure_rank).fillna(999)
    summary = summary.sort_values(["_measure_rank", "noise_level"]).drop(columns="_measure_rank").reset_index(drop=True)
    return summary


def _set_publication_style() -> None:
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


def plot_noise_sweep_publication(summary_df: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    """Plot STS and RTR versus observation noise for MMI and CCS."""
    _set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    measures = _measures_present(summary_df)
    atoms = (
        ("sts", "Synergy (STS)"),
        ("rtr", "Redundancy (RTR)"),
    )

    for ax, (atom, title) in zip(axes, atoms, strict=False):
        for measure in measures:
            sub = summary_df.loc[summary_df["measure"] == measure].copy()
            if sub.empty:
                continue
            color = MEASURE_COLORS.get(measure, "#4C566A")
            mean_col = f"{atom}_mean"
            sem_col = f"{atom}_sem"
            label = MEASURE_LABELS.get(measure, measure.upper())
            ax.fill_between(
                sub["noise_level"],
                sub[mean_col] - sub[sem_col],
                sub[mean_col] + sub[sem_col],
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
            ax.plot(
                sub["noise_level"],
                sub[mean_col],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.8,
                label=label,
            )

        ax.set_title(title)
        ax.set_xlabel("Observation noise SD")
        ax.set_ylabel("PhiID atom value")
        ax.grid(True, axis="y", color="#D8D2C4", linewidth=0.6, alpha=0.6)

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Bivariate VAR(1) PhiID under increasing observation noise", y=1.02, fontsize=12.0)
    fig.tight_layout()
    return fig, axes


def plot_noise_sweep_replicates(df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    """Plot faint replicate trajectories with mean overlays."""
    _set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    measures = _measures_present(summary_df)
    atoms = (
        ("sts", "Synergy (STS)"),
        ("rtr", "Redundancy (RTR)"),
    )

    for ax, (atom, title) in zip(axes, atoms, strict=False):
        for measure in measures:
            color = MEASURE_COLORS.get(measure, "#4C566A")
            raw = df.loc[df["measure"] == measure].copy()
            for _, rep in raw.groupby("replicate", dropna=False):
                rep = rep.sort_values("noise_level")
                ax.plot(
                    rep["noise_level"],
                    rep[atom],
                    color=color,
                    alpha=0.17,
                    linewidth=0.9,
                )
            sub = summary_df.loc[summary_df["measure"] == measure].copy()
            if not sub.empty:
                ax.plot(
                    sub["noise_level"],
                    sub[f"{atom}_mean"],
                    color=color,
                    linewidth=2.4,
                    marker="o",
                    markersize=4.8,
                    label=MEASURE_LABELS.get(measure, measure.upper()),
                )
        ax.set_title(title)
        ax.set_xlabel("Observation noise SD")
        ax.set_ylabel("PhiID atom value")
        ax.grid(True, axis="y", color="#D8D2C4", linewidth=0.6, alpha=0.6)

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Replicate-level variability across the observation-noise sweep", y=1.02, fontsize=12.0)
    fig.tight_layout()
    return fig, axes
