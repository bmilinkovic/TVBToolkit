"""Side-by-side parity checks against legacy TVBSim brian_MF.

These helpers are optional and only run when the legacy checkout is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np

from tvbtoolkit.brian_mf.mean_field.mf import run_mean_field_simulation
from tvbtoolkit.brian_mf.mean_field.tf_calc import get_neuron_params_double_cell, mu_sig_tau_func


@dataclass(frozen=True)
class MeanFieldParityReport:
    """Numeric parity report for MF mean rates."""

    legacy_exc_hz: float
    legacy_inh_hz: float
    new_exc_hz: float
    new_inh_hz: float
    abs_err_exc: float
    abs_err_inh: float


@dataclass(frozen=True)
class SubthresholdParityReport:
    """Parity report for mu/sigma/tau subthreshold calculations."""

    max_abs_err_mu_v: float
    max_abs_err_sig_v: float
    max_abs_err_tau_v: float
    max_abs_err_tau_n_v: float


def _load_module(path: Path, module_name: str):
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_subthreshold_with_legacy(
    *,
    legacy_root: str | Path = "/Users/borjan/CNRS/projects/TVBSim/brian_MF",
    cell_type: str = "RS",
) -> SubthresholdParityReport:
    """Compare `mu_sig_tau_func` output against legacy theoretical tools."""

    root = Path(legacy_root)
    legacy_file = root / "Tf_calc" / "theoretical_tools.py"
    if not legacy_file.exists():
        raise FileNotFoundError(f"Legacy file not found: {legacy_file}")

    legacy = _load_module(legacy_file, "legacy_theoretical_tools_sub")
    params = get_neuron_params_double_cell("FS-RS", si_units=False)

    ve = np.linspace(0.1, 20.0, 16)
    vi = np.linspace(0.1, 20.0, 16)
    vve, vvi = np.meshgrid(ve, vi)
    ff = np.clip(0.02 * vve + 0.01 * vvi, 1e-5, None)
    adapt = 5e-12 + 2e-13 * vve

    mu_new, sig_new, tau_new, tau_n_new = mu_sig_tau_func(vve, vvi, ff, adapt, params, cell_type, w_prec=False)
    mu_old, sig_old, tau_old, tau_n_old = legacy.mu_sig_tau_func(vve, vvi, ff, adapt, params, cell_type, w_prec=False)

    return SubthresholdParityReport(
        max_abs_err_mu_v=float(np.max(np.abs(mu_new - mu_old))),
        max_abs_err_sig_v=float(np.max(np.abs(sig_new - sig_old))),
        max_abs_err_tau_v=float(np.max(np.abs(tau_new - tau_old))),
        max_abs_err_tau_n_v=float(np.max(np.abs(tau_n_new - tau_n_old))),
    )


def compare_mf_with_legacy(
    prs: np.ndarray,
    pfs: np.ndarray,
    *,
    legacy_root: str | Path = "/Users/borjan/CNRS/projects/TVBSim/brian_MF",
    cells: str = "FS-RS",
    amp_stim_hz: float = 0.0,
    iext_hz: float = 0.3,
    total_time_s: float = 2.0,
    seed: int = 123,
) -> MeanFieldParityReport:
    """Compare legacy `run_MF` output with the new TVBToolkit port.

    Raises
    ------
    FileNotFoundError
        If the legacy checkout is not available.
    """

    root = Path(legacy_root)
    legacy_file = root / "Tf_calc" / "theoretical_tools.py"
    if not legacy_file.exists():
        raise FileNotFoundError(f"Legacy file not found: {legacy_file}")

    legacy = _load_module(legacy_file, "legacy_theoretical_tools")

    np.random.seed(seed)
    legacy_exc, legacy_inh = legacy.run_MF(cells, amp_stim_hz, prs, pfs, Iext=iext_hz, TotTime=total_time_s)

    result = run_mean_field_simulation(
        cells,
        amp_stim_hz,
        prs,
        pfs,
        iext_hz=iext_hz,
        total_time_s=total_time_s,
        seed=seed,
    )

    return MeanFieldParityReport(
        legacy_exc_hz=float(legacy_exc),
        legacy_inh_hz=float(legacy_inh),
        new_exc_hz=float(result.mean_exc_hz),
        new_inh_hz=float(result.mean_inh_hz),
        abs_err_exc=float(abs(float(legacy_exc) - float(result.mean_exc_hz))),
        abs_err_inh=float(abs(float(legacy_inh) - float(result.mean_inh_hz))),
    )
