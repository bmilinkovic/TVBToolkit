#!/usr/bin/env python3
"""Run a nonlinear bivariate PhiID dynamical-noise sweep and render figures."""

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

from tvbtoolkit.analysis import (  # noqa: E402
    load_var_noise_sweep,
    plot_noise_sweep_publication,
    plot_noise_sweep_replicates,
    summarize_sweep,
    sweep_long_form,
)
from tvbtoolkit.core.paths import legacy_results  # noqa: E402


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
        "phiid_bivariate_nonlinear_noise_sweep("
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
    plt.close(fig)


def _plot_sts_only(summary_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    for measure in summary_df["measure"].dropna().unique():
        sub = summary_df.loc[summary_df["measure"] == measure].copy()
        if sub.empty:
            continue
        color = "#466C95"
        label = "Synergy (STS)"
        ax.fill_between(
            sub["noise_level"],
            sub["sts_mean"] - sub["sts_sem"],
            sub["sts_mean"] + sub["sts_sem"],
            color=color,
            alpha=0.20,
            linewidth=0.0,
        )
        ax.plot(
            sub["noise_level"],
            sub["sts_mean"],
            color=color,
            linewidth=2.6,
            marker="o",
            markersize=5.2,
            label=label,
        )

    ax.set_title("Synergy can increase once dynamical noise recruits the joint interaction")
    ax.set_xlabel("Dynamical noise SD")
    ax.set_ylabel("PhiID synergy (STS)")
    ax.grid(True, axis="y", color="#D8D2C4", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def _write_note(note_path: Path, args: argparse.Namespace) -> None:
    text = f"""# Nonlinear Bivariate PhiID Dynamical-Noise Sweep

This sweep uses the continuous-valued nonlinear dynamics

`x_(t+1) = tanh(a * x_t + b * x_t * y_t) + sigma * eps_x`

`y_(t+1) = tanh(a * y_t - b * x_t * y_t) + sigma * eps_y`

with:

- `self_coef = {args.self_coef}`
- `interaction_coef = {args.interaction_coef}`
- `n_timepoints = {args.n_timepoints}`
- `burnin = {args.burnin}`
- `tau = {args.tau}`
- `n_replicates = {args.n_replicates}`

Unlike the earlier linear VAR example, here the manipulated quantity is **dynamical
noise** inside the nonlinear state update itself. The multiplicative term is joint
in the two variables, so moderate noise can drive the system away from the quiet
fixed point and expose more genuinely synergistic predictive structure.

This does **not** imply that synergy must always increase with noise. The expected
pattern is model-dependent. In linear Gaussian systems, increasing noise often
reduces synergy. In this nonlinear toy model, however, rising dynamical noise can
increase measured synergy because it recruits the joint interaction term.
"""
    note_path.write_text(text)


def _retitle_noise_figures(fig: plt.Figure, *, title: str) -> None:
    axes = fig.axes[:2]
    for ax in axes:
        ax.set_xlabel("Dynamical noise SD")
    if fig._suptitle is not None:
        fig._suptitle.set_text(title)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    logs_dir = output_root / "logs"
    for path in (figures_dir, tables_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    config_path = output_root / "config.mat"
    results_mat_path = output_root / "phiid_nonlinear_noise_sweep.mat"
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
            "interaction_coef": np.asarray([[float(args.interaction_coef)]], dtype=float),
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
        summary = {"status": "prepared_only", "output_root": str(output_root), "matlab_command": matlab_cmd}
        (logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    results = load_var_noise_sweep(results_mat_path)
    raw_df = sweep_long_form(results)
    summary_df = summarize_sweep(raw_df)
    raw_df.to_csv(tables_dir / "raw_atom_values.csv", index=False)
    summary_df.to_csv(tables_dir / "summary_atom_values.csv", index=False)

    fig_primary, _ = plot_noise_sweep_publication(summary_df)
    _retitle_noise_figures(fig_primary, title="Nonlinear bivariate PhiID under increasing dynamical noise")
    _save_figures(fig_primary, figures_dir, "phiid_nonlinear_noise_sweep_primary")

    fig_reps, _ = plot_noise_sweep_replicates(raw_df, summary_df)
    _retitle_noise_figures(fig_reps, title="Replicate trajectories under increasing dynamical noise")
    _save_figures(fig_reps, figures_dir, "phiid_nonlinear_noise_sweep_replicates")

    fig_sts = _plot_sts_only(summary_df)
    _save_figures(fig_sts, figures_dir, "phiid_nonlinear_noise_sweep_sts_only")

    _write_note(output_root / "README.md", args)

    summary = {
        "status": "completed",
        "output_root": str(output_root),
        "results_mat": str(results_mat_path),
        "n_noise_levels": int(len(args.noise_levels)),
        "n_replicates": int(args.n_replicates),
        "measures": list(args.measures),
        "matlab_command": matlab_cmd,
        "figure_primary": str(figures_dir / "phiid_nonlinear_noise_sweep_primary.png"),
        "summary_table": str(tables_dir / "summary_atom_values.csv"),
    }
    (logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=str, default=str(legacy_results("results", "phiid_bivariate_nonlinear_noise_sweep")))
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25])
    parser.add_argument("--measures", nargs="+", default=["mmi"])
    parser.add_argument("--n-replicates", type=int, default=24)
    parser.add_argument("--n-timepoints", type=int, default=1600)
    parser.add_argument("--burnin", type=int, default=400)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--self-coef", type=float, default=0.10)
    parser.add_argument("--interaction-coef", type=float, default=2.5)
    parser.add_argument("--base-seed", type=int, default=27182)
    parser.add_argument("--run-matlab", action="store_true")
    parser.add_argument("--matlab-parallel", action="store_true", default=False)
    parser.add_argument("--matlab-workers", type=int, default=0)
    parser.add_argument("--matlab-bin", type=str, default="/Applications/MATLAB_R2023b.app/bin/matlab")
    parser.add_argument("--matlab-toolbox-root", type=str, default="/Users/borjan/code/matlab/elph")
    parser.add_argument("--matlab-runner", type=str, default=str(_REPO_ROOT / "scripts" / "phiid_bivariate_nonlinear_noise_sweep.m"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))
