from __future__ import annotations

from pathlib import Path

import numpy as np

from tvbtoolkit.analysis.dynamics import load_survival_arrays


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def test_load_survival_arrays_precalc_tau_e(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    pre = root / "Dyn_Analysis" / "dynamical_precalc"

    _write(pre / "tau_e_mean_array.npy", np.ones((3, 4)))
    _write(pre / "b_thresh_tau_e.npy", np.array([[3.0, 10.0], [4.0, 12.0], [5.0, 15.0]]))
    _write(pre / "tau_e_heatmap_taus.npy", np.array([3.0, 4.0, 5.0, 6.0]))
    _write(pre / "tau_e_heatmap_bvals.npy", np.array([0.0, 5.0, 10.0]))

    mean, taus, bthr, tau_v, bvals = load_survival_arrays(load="tau_e", precalc=True, paper_repo_root=root)
    assert mean.shape == (3, 4)
    assert len(taus) == 3
    assert len(bthr) == 3
    assert tau_v.shape == (4,)
    assert bvals.shape == (3,)


def test_load_survival_arrays_trials_tau_i(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    trials = root / "Dyn_Analysis" / "trials"

    _write(trials / "tau_i_mean_array.npy", np.ones((4, 5)))
    _write(trials / "b_thresh_tau_i.npy", np.array([[8.0, 3.0, 11.0], [9.0, 4.0, 12.0]]))
    _write(trials / "tau_i_heatmap_taus.npy", np.array([3.0, 4.0, 5.0, 6.0, 7.0]))
    _write(trials / "tau_i_heatmap_bvals.npy", np.array([0.0, 5.0, 10.0, 15.0]))

    mean, taus, bthr, tau_v, bvals = load_survival_arrays(
        load="tau_i",
        precalc=False,
        paper_repo_root=root,
        save_path="trials/",
    )
    assert mean.shape == (4, 5)
    assert len(taus) >= 1
    assert len(bthr) == 2
    assert tau_v.shape == (5,)
    assert bvals.shape == (4,)
