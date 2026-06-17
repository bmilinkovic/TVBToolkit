"""Helpers for 3-node AdEx PhiID sweep analysis."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import scipy.io

MEASURE_ORDER: tuple[str, ...] = ("mmi", "ccs")
ATOM_ORDER: tuple[str, ...] = ("sts", "rtr")
PAIR_LABELS: dict[tuple[int, int], str] = {
    (0, 1): "Coupled pair (1-2)",
    (0, 2): "Uncoupled pair (1-3)",
    (1, 2): "Uncoupled pair (2-3)",
}

_OUTPUT_RE = re.compile(r"^(?P<stub>.+)__phiid_(?P<measure>[A-Za-z0-9_]+)$")


def parse_three_node_output_name(path: str | Path) -> dict[str, str] | None:
    match = _OUTPUT_RE.match(Path(path).stem)
    if match is None:
        return None
    return match.groupdict()


def load_three_node_phiid_index(output_dir: str | Path, manifest_path: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    out = Path(output_dir).expanduser().resolve()
    for path in sorted(out.glob("*.mat")):
        meta = parse_three_node_output_name(path)
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
    return df.sort_values(["measure", "g_value", "noise_value", "seed"]).reset_index(drop=True)


def load_three_node_phiid_output(path: str | Path) -> dict[str, np.ndarray]:
    mat = scipy.io.loadmat(str(Path(path).expanduser().resolve()))
    return {
        "sts": np.asarray(mat["sts_mat"], dtype=float),
        "rtr": np.asarray(mat["rtr_mat"], dtype=float),
    }


def summarize_three_node_outputs(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if index_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw_rows: list[dict[str, Any]] = []
    avg_rows: list[dict[str, Any]] = []
    group_cols = ["measure", "g_value", "noise_value"]

    for _, row in index_df.iterrows():
        mats = load_three_node_phiid_output(row["path"])
        for atom in ATOM_ORDER:
            mat = np.asarray(mats[atom], dtype=float)
            raw_rows.append(
                {
                    "stub": row["stub"],
                    "measure": row["measure"],
                    "g_value": float(row["g_value"]),
                    "noise_value": float(row["noise_value"]),
                    "seed": int(row["seed"]),
                    "atom": atom,
                    "matrix": mat,
                    "coupled_pair_value": float(mat[0, 1]),
                    "mixed_pair_mean": float(np.mean([mat[0, 2], mat[1, 2]])),
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    for (measure, g_value, noise_value, atom), group in raw_df.groupby(group_cols + ["atom"], dropna=False):
        mats = np.stack(group["matrix"].to_list(), axis=0)
        avg_rows.append(
            {
                "measure": measure,
                "g_value": float(g_value),
                "noise_value": float(noise_value),
                "atom": atom,
                "n_seeds": int(mats.shape[0]),
                "matrix": np.mean(mats, axis=0),
                "coupled_pair_mean": float(group["coupled_pair_value"].mean()),
                "mixed_pair_mean": float(group["mixed_pair_mean"].mean()),
                "coupled_pair_sem": _sem(group["coupled_pair_value"]),
                "mixed_pair_sem": _sem(group["mixed_pair_mean"]),
            }
        )

    avg_df = pd.DataFrame(avg_rows).sort_values(["measure", "atom", "noise_value", "g_value"]).reset_index(drop=True)
    raw_out = raw_df.drop(columns=["matrix"]).copy()
    return raw_out, avg_df


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
            "axes.titlesize": 10.0,
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


def _atom_cmap(atom: str) -> mcolors.Colormap:
    if atom == "sts":
        colors = ["#FBF5E9", "#EBCB8B", "#D88C4A", "#B65D4A", "#7C2F39"]
    else:
        colors = ["#F7F1E3", "#D9D4C7", "#B7C2C8", "#7B9AA6", "#3E5C67"]
    cmap = mcolors.LinearSegmentedColormap.from_list(f"three_node_{atom}", colors)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def plot_three_node_matrix_grid(
    avg_df: pd.DataFrame,
    *,
    measure: str,
    atom: str,
    g_values: list[float],
    noise_values: list[float],
    figsize_per_panel: tuple[float, float] = (2.5, 2.4),
) -> tuple[plt.Figure, np.ndarray]:
    _set_style()
    measure = str(measure).lower()
    atom = str(atom).lower()
    subset = avg_df.loc[(avg_df["measure"] == measure) & (avg_df["atom"] == atom)].copy()
    fig, axes = plt.subplots(
        len(noise_values),
        len(g_values),
        figsize=(figsize_per_panel[0] * len(g_values), figsize_per_panel[1] * len(noise_values)),
        squeeze=False,
        constrained_layout=True,
    )
    mats = [np.asarray(x, dtype=float) for x in subset["matrix"].tolist()]
    vmax = max(float(np.nanmax(np.abs(m))) for m in mats) if mats else 1.0
    if atom == "rtr":
        vmin = 0.0
    else:
        vmin = -vmax if any(np.nanmin(m) < 0 for m in mats) else 0.0

    for row_idx, noise_value in enumerate(noise_values):
        for col_idx, g_value in enumerate(g_values):
            ax = axes[row_idx, col_idx]
            hit = subset.loc[
                (subset["noise_value"] == noise_value)
                & (subset["g_value"] == g_value)
            ]
            if hit.empty:
                ax.axis("off")
                continue
            mat = np.asarray(hit.iloc[0]["matrix"], dtype=float)
            im = ax.imshow(mat, cmap=_atom_cmap(atom), vmin=vmin, vmax=vmax, origin="lower")
            if row_idx == 0:
                ax.set_title(f"G = {g_value:g}")
            if col_idx == 0:
                ax.set_ylabel(f"Noise\n{noise_value:g}")
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(["1", "2", "3"])
            ax.set_yticklabels(["1", "2", "3"])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
    cbar.set_label(f"{atom.upper()} value")
    fig.suptitle(f"{measure.upper()} {atom.upper()} matrices across G and noise", y=1.01, fontsize=12.0)
    return fig, axes


def plot_three_node_hypothesis_summary(avg_df: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), squeeze=False, constrained_layout=True)
    panels = [
        ("mmi", "sts", "mixed_pair_mean", "MMI STS: mixed-pair mean"),
        ("mmi", "rtr", "coupled_pair_mean", "MMI RTR: coupled pair"),
        ("ccs", "sts", "mixed_pair_mean", "CCS STS: mixed-pair mean"),
        ("ccs", "rtr", "coupled_pair_mean", "CCS RTR: coupled pair"),
    ]
    heat_cmap = mcolors.LinearSegmentedColormap.from_list(
        "three_node_summary",
        ["#FBF7EF", "#E8D8BE", "#B7C2C8", "#6E8797", "#324B5A"],
    )

    for ax, (measure, atom, value_col, title) in zip(axes.ravel(), panels, strict=False):
        sub = avg_df.loc[(avg_df["measure"] == measure) & (avg_df["atom"] == atom)].copy()
        pivot = sub.pivot(index="noise_value", columns="g_value", values=value_col).sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=heat_cmap)
        ax.set_title(title)
        ax.set_xlabel("G")
        ax.set_ylabel("Noise")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([f"{x:g}" for x in pivot.index])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Hypothesis-oriented summaries across the 3-node AdEx sweep", y=1.01, fontsize=12.0)
    return fig, axes


def plot_three_node_hypothesis_clean(
    avg_df: pd.DataFrame,
    *,
    normalize_sts: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    _set_style()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )
    fig = plt.figure(figsize=(16.2, 7.1), constrained_layout=True)
    if normalize_sts:
        gs = fig.add_gridspec(2, 6, width_ratios=[1.0, 1.0, 1.0, 0.06, 1.05, 0.06], wspace=0.06, hspace=0.04)
    else:
        gs = fig.add_gridspec(
            2,
            8,
            width_ratios=[1.0, 0.05, 1.0, 0.05, 1.0, 0.05, 1.05, 0.06],
            wspace=0.06,
            hspace=0.04,
        )
    axes = np.empty((2, 4), dtype=object)

    methods = ["mmi", "ccs"]
    stcmap = mcolors.LinearSegmentedColormap.from_list(
        "three_node_sts_clean",
        ["#FCF8EF", "#EED9B6", "#E1A35C", "#C36B47", "#7A3038"],
    )
    contrast_cmap = mcolors.LinearSegmentedColormap.from_list(
        "three_node_contrast",
        ["#3C6893", "#F5F1E8", "#C95E46"],
    )

    g_values = sorted(float(x) for x in avg_df["g_value"].dropna().unique().tolist())
    noise_values = sorted(float(x) for x in avg_df["noise_value"].dropna().unique().tolist())

    def _ticks(values: list[float], n_target: int = 5) -> tuple[list[int], list[str]]:
        if len(values) <= n_target:
            idx = list(range(len(values)))
        else:
            idx = np.linspace(0, len(values) - 1, n_target, dtype=int).tolist()
        labels = [f"{values[i]:.2g}" for i in idx]
        return idx, labels

    x_idx, x_labels = _ticks(g_values)
    y_idx, y_labels = _ticks(noise_values)
    col_titles = [
        "Uncoupled Pair 1-3",
        "Uncoupled Pair 2-3",
        "Coupled Pair 1-2",
        "STS Contrast\nUncoupled - Coupled",
    ]

    for row_idx, measure in enumerate(methods):
        if normalize_sts:
            for col_idx in range(3):
                axes[row_idx, col_idx] = fig.add_subplot(gs[row_idx, col_idx])
            axes[row_idx, 3] = fig.add_subplot(gs[row_idx, 4])
        else:
            axes[row_idx, 0] = fig.add_subplot(gs[row_idx, 0])
            axes[row_idx, 1] = fig.add_subplot(gs[row_idx, 2])
            axes[row_idx, 2] = fig.add_subplot(gs[row_idx, 4])
            axes[row_idx, 3] = fig.add_subplot(gs[row_idx, 6])

        sts = avg_df.loc[(avg_df["measure"] == measure) & (avg_df["atom"] == "sts")].copy()
        sts_pair_13 = sts.copy()
        sts_pair_13["pair_value"] = sts_pair_13["matrix"].apply(lambda x: float(np.asarray(x, dtype=float)[0, 2]))
        sts_pair_23 = sts.copy()
        sts_pair_23["pair_value"] = sts_pair_23["matrix"].apply(lambda x: float(np.asarray(x, dtype=float)[1, 2]))
        sts_pair_12 = sts.copy()
        sts_pair_12["pair_value"] = sts_pair_12["matrix"].apply(lambda x: float(np.asarray(x, dtype=float)[0, 1]))

        sts_uncoupled_13 = sts_pair_13.pivot(index="noise_value", columns="g_value", values="pair_value").reindex(index=noise_values, columns=g_values)
        sts_uncoupled_23 = sts_pair_23.pivot(index="noise_value", columns="g_value", values="pair_value").reindex(index=noise_values, columns=g_values)
        sts_coupled = sts_pair_12.pivot(index="noise_value", columns="g_value", values="pair_value").reindex(index=noise_values, columns=g_values)
        sts_uncoupled_mean = 0.5 * (sts_uncoupled_13 + sts_uncoupled_23)
        contrast = sts_uncoupled_mean - sts_coupled

        sts_arrays_raw = [
            sts_uncoupled_13.to_numpy(dtype=float),
            sts_uncoupled_23.to_numpy(dtype=float),
            sts_coupled.to_numpy(dtype=float),
        ]
        sts_valid = np.concatenate([arr[np.isfinite(arr)] for arr in sts_arrays_raw if np.isfinite(arr).any()])
        sts_vmin = float(sts_valid.min()) if sts_valid.size else 0.0
        sts_vmax = float(sts_valid.max()) if sts_valid.size else 1.0
        sts_span = sts_vmax - sts_vmin

        if normalize_sts and sts_span > 0:
            sts_arrays = [(arr - sts_vmin) / sts_span for arr in sts_arrays_raw]
        elif normalize_sts:
            sts_arrays = [np.zeros_like(arr, dtype=float) for arr in sts_arrays_raw]
        else:
            sts_arrays = sts_arrays_raw

        contrast_data = 0.5 * (sts_arrays[0] + sts_arrays[1]) - sts_arrays[2]
        contrast_valid = contrast_data[np.isfinite(contrast_data)]
        contrast_vmax = float(np.nanmax(np.abs(contrast_valid))) if contrast_valid.size else 1.0

        ims: list[Any] = []
        panel_data = sts_arrays + [contrast_data]
        panel_cmaps = [stcmap, stcmap, stcmap, contrast_cmap]
        if normalize_sts:
            panel_ranges = [
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (-contrast_vmax, contrast_vmax),
            ]
        else:
            panel_ranges = []
            for arr in sts_arrays_raw:
                valid = arr[np.isfinite(arr)]
                vmin = float(valid.min()) if valid.size else 0.0
                vmax = float(valid.max()) if valid.size else 1.0
                panel_ranges.append((vmin, vmax))
            panel_ranges.append((-contrast_vmax, contrast_vmax))

        for col_idx, (data, cmap, (vmin, vmax)) in enumerate(zip(panel_data, panel_cmaps, panel_ranges, strict=False)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], pad=10)
            if row_idx == 1:
                ax.set_xlabel("Global coupling G")
                ax.set_xticks(x_idx)
                ax.set_xticklabels(x_labels)
            else:
                ax.set_xticks(x_idx)
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel("Dynamical noise")
                ax.set_yticks(y_idx)
                ax.set_yticklabels(y_labels)
            else:
                ax.set_yticks(y_idx)
                ax.set_yticklabels([])
            ax.tick_params(length=3.5, width=0.9)
            ax.spines["left"].set_linewidth(0.9)
            ax.spines["bottom"].set_linewidth(0.9)
            if col_idx == 3:
                ax.contour(
                    data,
                    levels=[0.0],
                    colors=["#374151"],
                    linewidths=1.0,
                    alpha=0.75,
                    origin="lower",
                )

        row_label = "MMI" if measure == "mmi" else "CCS"
        axes[row_idx, 0].text(
            -0.34,
            0.5,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13.0,
            fontweight="bold",
            color="#2B2B2B",
        )

        if normalize_sts:
            cax_sts = fig.add_subplot(gs[row_idx, 3])
            cbar_sts = fig.colorbar(ims[0], cax=cax_sts)
            cbar_sts.set_label("Normalized STS", fontsize=10.5)
            cbar_sts.ax.tick_params(labelsize=9.0)
            cbar_sts.set_ticks([0.0, 0.5, 1.0])
            cax_contrast = fig.add_subplot(gs[row_idx, 5])
        else:
            for im_idx, gs_col in enumerate([1, 3, 5]):
                cax_sts = fig.add_subplot(gs[row_idx, gs_col])
                cbar_sts = fig.colorbar(ims[im_idx], cax=cax_sts)
                cbar_sts.set_label("STS", fontsize=10.0)
                cbar_sts.ax.tick_params(labelsize=8.5)
            cax_contrast = fig.add_subplot(gs[row_idx, 7])

        cbar_contrast = fig.colorbar(ims[3], cax=cax_contrast)
        cbar_contrast.set_label("Positive: uncoupled > coupled", fontsize=10.2)
        cbar_contrast.ax.tick_params(labelsize=9.0)

    fig.suptitle(
        "Three-node AdEx sweep: uncoupled-pair and coupled-pair synergy landscapes",
        y=1.02,
        fontsize=14.5,
        fontweight="bold",
    )
    return fig, axes
