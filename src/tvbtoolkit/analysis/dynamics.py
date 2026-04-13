"""Dynamical-analysis utilities.

This module integrates survival-time loading and heatmap plotting functions used
in the Maria Sacha paper pipeline, adapted to TVBToolkit conventions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_survival_arrays(
    *,
    load: str = "tau_e",
    precalc: bool = False,
    paper_repo_root: str | Path | None = None,
    save_path: str = "trials/",
) -> tuple[np.ndarray, list[float], list[float], np.ndarray, np.ndarray]:
    """Load survival-time arrays with paper-compatible path conventions.

    Parameters
    ----------
    load : {"tau_e", "tau_i"}, default="tau_e"
        Which parameter sweep to load.
    precalc : bool, default=False
        If ``True``, load from ``Dyn_Analysis/dynamical_precalc``.
        Otherwise load from ``Dyn_Analysis/<save_path>``.
    paper_repo_root : str | Path | None, optional
        Root path to ``paper_pipeline_hub``. If ``None``, defaults to the
        relative path ``/Users/borjan/CNRS/projects/paper_pipeline_hub``.
    save_path : str, default="trials/"
        Subfolder under ``Dyn_Analysis`` used for non-precalculated arrays.

    Returns
    -------
    mean_array : ndarray
        Mean survival-time matrix.
    taus : list[float]
        Tau values for threshold-curve x-axis.
    bthr : list[float]
        Threshold ``b_e`` values.
    tau_v : ndarray
        Tau grid used for heatmap x-axis.
    bvals : ndarray
        ``b_e`` grid used for heatmap y-axis.
    """
    if load not in {"tau_e", "tau_i"}:
        raise ValueError("load must be 'tau_e' or 'tau_i'.")

    if paper_repo_root is None:
        root = Path("/Users/borjan/CNRS/projects/paper_pipeline_hub")
    else:
        root = Path(paper_repo_root)

    if precalc:
        pre = root / "Dyn_Analysis" / "dynamical_precalc"
        if load == "tau_e":
            mean_array = np.load(pre / "tau_e_mean_array.npy")
            bth = np.load(pre / "b_thresh_tau_e.npy")
            taus = list(np.asarray(bth[:, 0], dtype=float))
            bthr = list(np.asarray(bth[:, -1], dtype=float))
            tau_v = np.load(pre / "tau_e_heatmap_taus.npy")
            bvals = np.load(pre / "tau_e_heatmap_bvals.npy")
        else:
            mean_array = np.load(pre / "mean_array_tau_i.npy")
            taus = list(np.asarray(np.load(pre / "tauis_bcrit.npy"), dtype=float))
            bthr = list(np.asarray(np.load(pre / "bthr_tauis_bcrit.npy"), dtype=float))
            bvals = np.arange(0.0, 25.0, 1.0)
            tau_v = np.arange(3.0, 9.0, 0.1)
    else:
        p = root / "Dyn_Analysis" / save_path
        mean_array = np.load(p / f"{load}_mean_array.npy")
        bthr_arr = np.load(p / f"b_thresh_{load}.npy")
        bthr = list(np.asarray(bthr_arr[:, -1], dtype=float))
        tau_v = np.load(p / f"{load}_heatmap_taus.npy")
        bvals = np.load(p / f"{load}_heatmap_bvals.npy")
        if load == "tau_e":
            taus = list(np.asarray(bthr_arr[:, 0], dtype=float))
        else:
            taus = list(np.asarray(bthr_arr[:, 1], dtype=float))
            taus = [i for i in taus if i <= float(np.max(tau_v))]

    return np.asarray(mean_array, dtype=float), taus, bthr, np.asarray(tau_v, dtype=float), np.asarray(bvals, dtype=float)


def plot_survival_heatmap(
    mean_array: np.ndarray,
    taus: list[float],
    bthr: list[float],
    tau_v: np.ndarray,
    bvals: np.ndarray,
    *,
    load: str,
    z_min: float | None = None,
    z_max: float | None = None,
    line_color: str = "white",
    cmap: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot survival-time heatmap with threshold line.

    Parameters
    ----------
    mean_array : ndarray
        Heatmap matrix.
    taus : list[float]
        Threshold-curve tau coordinates.
    bthr : list[float]
        Threshold-curve ``b_e`` coordinates.
    tau_v : ndarray
        Tau axis grid.
    bvals : ndarray
        ``b_e`` axis grid.
    load : {"tau_e", "tau_i"}
        Determines axis label defaults.
    z_min, z_max : float | None, optional
        Optional color limits.
    line_color : str, default="white"
        Color for threshold curve.
    cmap : str | None, optional
        Colormap override.
    save_path : str | Path | None, optional
        If provided, save the figure.
    """
    if load not in {"tau_e", "tau_i"}:
        raise ValueError("load must be 'tau_e' or 'tau_i'.")

    arr = np.asarray(mean_array, dtype=float)
    tau_v = np.asarray(tau_v, dtype=float)
    bvals = np.asarray(bvals, dtype=float)

    if cmap is None:
        cmap = "hot" if load == "tau_e" else "jet"

    if z_min is None:
        z_min = float(np.nanmin(arr))
    if z_max is None:
        z_max = float(np.nanmax(arr))

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[float(np.min(tau_v)), float(np.max(tau_v)), float(np.min(bvals)), float(np.max(bvals))],
        cmap=cmap,
        vmin=z_min,
        vmax=z_max,
    )

    ax.plot(np.asarray(taus, dtype=float), np.asarray(bthr, dtype=float), color=line_color, lw=2.4)
    ax.set_xlabel(r"$\tau_e$ (ms)" if load == "tau_e" else r"$\tau_i$ (ms)")
    ax.set_ylabel(r"$b_e$ (pA)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean survival time (ms)")
    fig.tight_layout()

    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, bbox_inches="tight")

    return fig


__all__ = ["load_survival_arrays", "plot_survival_heatmap"]
