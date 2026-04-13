"""Survival-time analysis helpers ported from legacy brian_MF."""

from __future__ import annotations

from pathlib import Path
import os
import numpy as np


def load_survival(load: str = "tau_e", precalc: bool = False, save_path: str = "./") -> tuple[np.ndarray, list[float], list[float], np.ndarray, np.ndarray]:
    """Load survival-time products in the same layout as legacy brian_MF.

    Parameters
    ----------
    load
        Which sweep axis to load: ``"tau_e"`` or ``"tau_i"``.
    precalc
        If ``True``, read from `Dyn_Analysis/dynamical_precalc` paths.
    save_path
        Relative suffix path used by non-precalculated legacy outputs.
    """

    if precalc:
        if load == "tau_e":
            mean_array = np.load("./Dyn_Analysis/dynamical_precalc/tau_e_mean_array.npy")
            taus = list(np.load("./Dyn_Analysis/dynamical_precalc/b_thresh_tau_e.npy")[:, 0])
            bthr = list(np.load("./Dyn_Analysis/dynamical_precalc/b_thresh_tau_e.npy")[:, -1])
            tau_v = np.load("./Dyn_Analysis/dynamical_precalc/tau_e_heatmap_taus.npy")
            bvals = np.load("./Dyn_Analysis/dynamical_precalc/tau_e_heatmap_bvals.npy")
        elif load == "tau_i":
            mean_array = np.load("./Dyn_Analysis/dynamical_precalc/mean_array_tau_i.npy")
            taus = list(np.load("./Dyn_Analysis/dynamical_precalc/tauis_bcrit.npy"))
            bthr = list(np.load("./Dyn_Analysis/dynamical_precalc/bthr_tauis_bcrit.npy"))
            bvals = np.arange(0, 25, 1)
            tau_v = np.arange(3.0, 9.0, 0.1)
        else:
            raise ValueError("load must be 'tau_e' or 'tau_i'.")
    else:
        base = Path("./Dyn_Analysis") / save_path
        mean_array = np.load(base / f"{load}_mean_array.npy")
        bthr = list(np.load(base / f"b_thresh_{load}.npy")[:, -1])
        tau_v = np.load(base / f"{load}_heatmap_taus.npy")
        bvals = np.load(base / f"{load}_heatmap_bvals.npy")
        if load == "tau_e":
            taus = list(np.load(base / f"b_thresh_{load}.npy")[:, 0])
        elif load == "tau_i":
            taus = list(np.load(base / f"b_thresh_{load}.npy")[:, 1])
            taus = [i for i in taus if i <= tau_v.max()]
        else:
            raise ValueError("load must be 'tau_e' or 'tau_i'.")
    return mean_array, taus, bthr, tau_v, bvals


def calculate_survival_time(
    bvals: np.ndarray,
    tau_values: np.ndarray,
    tau_i_iter: bool,
    n_seeds: np.ndarray,
    save_path: str = "./network_sims/",
    bin_ms: int = 5,
    amp_stim: float = 1.0,
    offset_index: int = 61,
    load_until: int = 399,
) -> np.ndarray:
    """Compute survival durations from saved network simulation arrays.

    Ported/adapted from legacy `brian_MF/survival_time.py`.
    """

    tau_str = "tau_i" if tau_i_iter else "tau_e"
    all_seeds = []

    for seed in n_seeds:
        dur_dead_tau = []
        for tau in tau_values:
            dur_dead = []
            for b_ad in bvals:
                tau_e = 5.0
                tau_i = float(tau)
                if not tau_i_iter:
                    tau_e = float(tau)
                    tau_i = 5.0

                candidates = [
                    f"b_{b_ad}_tau_i_{round(tau_i,1)}_tau_e_{round(tau_e,1)}_ampst_{amp_stim}_seed_{seed}",
                    f"b_{float(b_ad)}_tau_i_{round(tau_i,1)}_tau_e_{round(tau_e,1)}_ampst_{amp_stim}_seed_{float(seed)}",
                    f"b_{float(b_ad)}_tau_i_{round(tau_i,1)}_tau_e_{round(tau_e,1)}_ampst_{amp_stim}_seed_{seed}",
                    f"b_{int(b_ad)}_tau_i_{int(tau_i)}_tau_e_{round(tau_e,1)}_ampst_{amp_stim}_seed_{int(seed)}",
                ]

                pop_exc = None
                for sim_name in candidates:
                    path = Path(save_path) / "network_sims" / f"{sim_name}_exc.npy"
                    path_alt = Path(save_path) / "network_sims" / f"{sim_name}_exc_vol2.npy"
                    if path.exists():
                        pop_exc = np.load(path)[:load_until]
                        break
                    if path_alt.exists():
                        pop_exc = np.load(path_alt)[:load_until]
                        break
                if pop_exc is None:
                    raise FileNotFoundError(f"No matching simulation found for b={b_ad}, tau={tau}, seed={seed}")

                thresh = pop_exc[offset_index] / 10.0
                consecutive_count = sum(1 for value in pop_exc[offset_index:] if value > thresh)
                dur_dead.append(consecutive_count * bin_ms)
            dur_dead_tau.append(dur_dead)
        all_seeds.append(np.array(dur_dead_tau).T)

    all_seeds_arr = np.array(all_seeds)
    mean_array = np.mean(all_seeds_arr, axis=0)

    out = Path(save_path)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{tau_str}_mean_array.npy", mean_array)
    np.save(out / f"{tau_str}_heatmap_bvals.npy", bvals)
    np.save(out / f"{tau_str}_heatmap_taus.npy", tau_values)

    return mean_array
