#!/usr/bin/env python3
"""Run a bivariate VAR PhiID noise sweep and render publication figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.analysis import (  # noqa: E402
    load_var_noise_sweep,
    plot_noise_sweep_publication,
    plot_noise_sweep_replicates,
    summarize_sweep,
    sweep_long_form,
)


def build_matlab_command(
    *,
    config_path: Path,
    output_path: Path,
    matlab_bin: str,
    matlab_toolbox_root: str,
    runner_path: Path,
    use_parallel: bool,
    n_workers: int,
) -> str:
    statements = [f"addpath(genpath('{Path(matlab_toolbox_root).expanduser().resolve().as_posix()}'))"]
    statements.append(f"addpath('{runner_path.parent.as_posix()}')")
    statements.append(
        "phiid_var_bivariate_noise_sweep("
        f"'{config_path.as_posix()}', "
        f"'{output_path.as_posix()}', "
        f"{str(bool(use_parallel)).lower()}, "
        f"{int(n_workers)})"
    )
    return f'{matlab_bin} -batch "' + "; ".join(statements) + '"'


def _save_figures(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
    fig.savefig(out_dir / f"{stem}_transparent.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def _write_note(note_path: Path, args: argparse.Namespace) -> None:
    text = f"""# Bivariate VAR PhiID Noise Sweep

This sweep simulates a latent symmetric bivariate VAR(1),

`z_t = A z_(t-1) + sigma_eps * eps_t`

with:

- `self_coef = {args.self_coef}`
- `cross_coef = {args.cross_coef}`
- `innovation_sd = {args.innovation_sd}`
- `n_timepoints = {args.n_timepoints}`
- `burnin = {args.burnin}`
- `tau = {args.tau}`
- `n_replicates = {args.n_replicates}`

The manipulated quantity is **observation noise SD**, not pure innovation amplitude.
That choice is deliberate: `PhiIDFull` standardizes each variable internally, so a simple
global rescaling of stationary innovation variance mainly rescales the process while leaving
its correlation structure nearly unchanged. Adding observation noise changes the effective
signal-to-noise ratio and therefore the lagged dependencies that drive `STS` and `RTR`.

`common_noise_fraction = {args.common_noise_fraction}` controls how much of the observation
noise is shared across the two channels.
"""
    note_path.write_text(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (0.0 <= float(args.common_noise_fraction) <= 1.0):
        raise ValueError("common_noise_fraction must lie in [0, 1].")

    eigvals = np.linalg.eigvals(np.array([[args.self_coef, args.cross_coef], [args.cross_coef, args.self_coef]], dtype=float))
    if np.max(np.abs(eigvals)) >= 1.0:
        raise ValueError(
            "The requested VAR(1) matrix is not stable. "
            f"Max abs eigenvalue = {float(np.max(np.abs(eigvals))):.4f}."
        )

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    logs_dir = output_root / "logs"
    for path in (figures_dir, tables_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    config_path = output_root / "config.mat"
    results_mat_path = output_root / "phiid_var_noise_sweep.mat"
    runner_path = Path(args.matlab_runner).expanduser().resolve()

    scipy.io.savemat(
        config_path,
        {
            "noise_levels": np.asarray(args.noise_levels, dtype=float).reshape(1, -1),
            "measures": np.asarray(list(args.measures), dtype=object).reshape(1, -1),
            "n_replicates": np.asarray([[int(args.n_replicates)]], dtype=np.int32),
            "n_timepoints": np.asarray([[int(args.n_timepoints)]], dtype=np.int32),
            "burnin": np.asarray([[int(args.burnin)]], dtype=np.int32),
            "tau": np.asarray([[int(args.tau)]], dtype=np.int32),
            "self_coef": np.asarray([[float(args.self_coef)]], dtype=float),
            "cross_coef": np.asarray([[float(args.cross_coef)]], dtype=float),
            "innovation_sd": np.asarray([[float(args.innovation_sd)]], dtype=float),
            "common_noise_fraction": np.asarray([[float(args.common_noise_fraction)]], dtype=float),
            "base_seed": np.asarray([[int(args.base_seed)]], dtype=np.int32),
        },
        do_compression=True,
    )

    matlab_cmd = build_matlab_command(
        config_path=config_path,
        output_path=results_mat_path,
        matlab_bin=args.matlab_bin,
        matlab_toolbox_root=args.matlab_toolbox_root,
        runner_path=runner_path,
        use_parallel=args.matlab_parallel,
        n_workers=args.matlab_workers,
    )
    (logs_dir / "matlab_command.txt").write_text(matlab_cmd + "\n")

    if args.run_matlab:
        subprocess.run(matlab_cmd, shell=True, cwd=_REPO_ROOT, check=True)

    if not results_mat_path.exists():
        summary = {
            "status": "prepared_only",
            "output_root": str(output_root),
            "matlab_command": matlab_cmd,
        }
        (logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    results = load_var_noise_sweep(results_mat_path)
    raw_df = sweep_long_form(results)
    summary_df = summarize_sweep(raw_df)
    raw_df.to_csv(tables_dir / "raw_atom_values.csv", index=False)
    summary_df.to_csv(tables_dir / "summary_atom_values.csv", index=False)
    failed_jobs = int((raw_df["status_code"] < 0).sum()) if "status_code" in raw_df.columns else 0

    fig_primary, _ = plot_noise_sweep_publication(summary_df)
    _save_figures(fig_primary, figures_dir, "phiid_var_noise_sweep_primary")

    fig_reps, _ = plot_noise_sweep_replicates(raw_df, summary_df)
    _save_figures(fig_reps, figures_dir, "phiid_var_noise_sweep_replicates")

    _write_note(output_root / "README.md", args)

    summary = {
        "status": "completed",
        "output_root": str(output_root),
        "results_mat": str(results_mat_path),
        "n_noise_levels": int(len(args.noise_levels)),
        "n_replicates": int(args.n_replicates),
        "measures": list(args.measures),
        "failed_jobs": failed_jobs,
        "matlab_command": matlab_cmd,
        "figure_primary": str(figures_dir / "phiid_var_noise_sweep_primary.png"),
        "summary_table": str(tables_dir / "summary_atom_values.csv"),
    }
    (logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/phiid_var_bivariate_noise_sweep",
    )
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2])
    parser.add_argument("--measures", nargs="+", default=["mmi", "ccs"])
    parser.add_argument("--n-replicates", type=int, default=32)
    parser.add_argument("--n-timepoints", type=int, default=1200)
    parser.add_argument("--burnin", type=int, default=300)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--self-coef", type=float, default=0.72)
    parser.add_argument("--cross-coef", type=float, default=0.18)
    parser.add_argument("--innovation-sd", type=float, default=1.0)
    parser.add_argument("--common-noise-fraction", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=31415)
    parser.add_argument("--run-matlab", action="store_true")
    parser.add_argument("--matlab-parallel", action="store_true", default=False)
    parser.add_argument("--matlab-workers", type=int, default=0)
    parser.add_argument("--matlab-bin", type=str, default="/Applications/MATLAB_R2023b.app/bin/matlab")
    parser.add_argument("--matlab-toolbox-root", type=str, default="/Users/borjan/code/matlab/elph")
    parser.add_argument(
        "--matlab-runner",
        type=str,
        default=str(_REPO_ROOT / "scripts" / "phiid_var_bivariate_noise_sweep.m"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))
