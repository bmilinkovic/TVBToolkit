#!/usr/bin/env python3
"""Run a 3-node AdEx mean-field G/noise PhiID sweep and render tutorial figures."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))
os.environ.setdefault("TVB_USER_HOME", str(Path("/tmp/tvb-user-home").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.core.config import WholeBrainConfig  # noqa: E402
from tvbtoolkit.core.paths import legacy_results  # noqa: E402
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: E402
from tvbtoolkit.analysis import (  # noqa: E402
    load_three_node_phiid_index,
    plot_three_node_hypothesis_summary,
    plot_three_node_matrix_grid,
    summarize_three_node_outputs,
)


def _float_tag(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    text = text.replace("-", "m").replace(".", "p")
    return text


def _build_stub(g_value: float, noise_value: float, seed: int) -> str:
    return f"G{_float_tag(g_value)}__noise{_float_tag(noise_value)}__seed{int(seed):03d}"


def _simulate_single_job(job: dict[str, Any]) -> dict[str, Any]:
    g_value = float(job["g_value"])
    noise_value = float(job["noise_value"])
    seed = int(job["seed"])
    sim_length_ms = float(job["simulation_length_ms"])
    transient_ms = float(job["transient_ms"])
    dt_ms = float(job["dt_ms"])
    monitor_period_ms = float(job["monitor_period_ms"])
    zerlaut_order = int(job["zerlaut_order"])
    output_input_path = Path(job["output_input_path"])

    weights = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    tract_lengths = np.ones((3, 3), dtype=float)
    np.fill_diagonal(tract_lengths, 0.0)

    cfg = WholeBrainConfig(
        simulation_length_ms=sim_length_ms,
        dt_ms=dt_ms,
        monitor_period_ms=monitor_period_ms,
        coupling_strength=g_value,
        weights=weights,
        tract_lengths=tract_lengths,
        monitor_mode="raw",
        monitor_variables=(0, 1),
        zerlaut_order=zerlaut_order,
        parameter_overrides={
            "T": 20.0,
            "tau_OU": 5.0,
            "weight_noise": noise_value,
            "noise_alpha": 0.0,
            "shared_noise_mode": "none",
        },
    )

    result = run_whole_brain_simulation(cfg, seed=seed)
    keep = np.asarray(result.time_ms, dtype=float) >= transient_ms
    times = np.asarray(result.time_ms, dtype=float)[keep] - transient_ms
    rates = np.asarray(result.raw, dtype=float)[keep, :]
    if rates.ndim != 2 or rates.shape[1] != 3:
        raise ValueError(f"Expected raw excitatory output with shape (time, 3), got {rates.shape}.")
    time_series = rates.T
    if time_series.shape[1] < 10:
        raise ValueError(f"Too few post-transient samples retained: {time_series.shape[1]}")

    output_input_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        output_input_path,
        {
            "time_series": np.asarray(time_series, dtype=np.float64),
            "time_ms": np.asarray(times, dtype=np.float64).reshape(1, -1),
            "stub": np.asarray([str(job["stub"])], dtype=object),
            "g_value": np.asarray([[g_value]], dtype=float),
            "noise_value": np.asarray([[noise_value]], dtype=float),
            "seed": np.asarray([[seed]], dtype=np.int32),
            "simulation_length_ms": np.asarray([[sim_length_ms]], dtype=float),
            "transient_ms": np.asarray([[transient_ms]], dtype=float),
            "dt_ms": np.asarray([[dt_ms]], dtype=float),
            "monitor_period_ms": np.asarray([[monitor_period_ms]], dtype=float),
            "zerlaut_order": np.asarray([[zerlaut_order]], dtype=np.int32),
        },
        do_compression=True,
    )
    return {
        "stub": str(job["stub"]),
        "g_value": g_value,
        "noise_value": noise_value,
        "seed": seed,
        "n_timepoints": int(time_series.shape[1]),
        "input_path": str(output_input_path),
    }


def _run_simulation_jobs(jobs: list[dict[str, Any]], workers: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            rows.append(_simulate_single_job(job))
        return pd.DataFrame(rows)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_simulate_single_job, job): job for job in jobs}
        for fut in as_completed(futs):
            rows.append(fut.result())
    return pd.DataFrame(rows).sort_values(["g_value", "noise_value", "seed"]).reset_index(drop=True)


def build_matlab_command(
    *,
    input_dir: Path,
    output_dir: Path,
    measures: list[str],
    matlab_bin: str,
    matlab_toolbox_root: str,
    runner_path: Path,
    use_parallel: bool,
    n_workers: int,
) -> str:
    statements = [f"addpath(genpath('{Path(matlab_toolbox_root).expanduser().resolve().as_posix()}'))"]
    statements.append(f"addpath('{runner_path.parent.as_posix()}')")
    statements.append(
        "phiid_three_node_adex_sweep("
        f"'{input_dir.as_posix()}', "
        f"'{output_dir.as_posix()}', "
        f"'{','.join(measures)}', "
        f"{str(bool(use_parallel)).lower()}, "
        f"{int(n_workers)})"
    )
    return f'{matlab_bin} -batch "' + "; ".join(statements) + '"'


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
    fig.savefig(out_dir / f"{stem}_transparent.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def _write_tutorial_note(path: Path, args: argparse.Namespace) -> None:
    text = f"""# Three-Node AdEx PhiID Sweep

This tutorial sweep uses a 3-node Zerlaut AdEx mean-field network:

- node 1 <-> node 2 are coupled
- node 3 is uncoupled
- coupling weights are fixed structurally and scaled globally by `G`

Simulation settings:

- Zerlaut order: `{args.zerlaut_order}`
- mean-field timescale `T = 20 ms`
- simulation length: `{args.simulation_length_ms} ms`
- transient removed: `{args.transient_ms} ms`
- analyzed window: `{args.simulation_length_ms - args.transient_ms} ms`
- integration step: `{args.dt_ms} ms`
- monitor period: `{args.monitor_period_ms} ms`

Noise manipulation:

- the swept noise parameter is the model OU noise gain `weight_noise`
- `noise_alpha = 0.0`, so noise is private rather than shared
- `shared_noise_mode = 'none'`

PhiID:

- pairwise bivariate PhiID is computed for all 3 node-pairs
- this yields `3 x 3` `STS` and `RTR` matrices
- both `MMI` and `CCS` are evaluated separately
"""
    path.write_text(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    input_dir = output_root / "inputs"
    phiid_dir = output_root / "phiid"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    logs_dir = output_root / "logs"
    for path in (output_root, input_dir, phiid_dir, tables_dir, figures_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for g_value in args.g_values:
        for noise_value in args.noise_values:
            for seed in args.seeds:
                stub = _build_stub(g_value, noise_value, seed)
                jobs.append(
                    {
                        "stub": stub,
                        "g_value": float(g_value),
                        "noise_value": float(noise_value),
                        "seed": int(seed),
                        "simulation_length_ms": float(args.simulation_length_ms),
                        "transient_ms": float(args.transient_ms),
                        "dt_ms": float(args.dt_ms),
                        "monitor_period_ms": float(args.monitor_period_ms),
                        "zerlaut_order": int(args.zerlaut_order),
                        "output_input_path": input_dir / f"{stub}.mat",
                    }
                )

    manifest = _run_simulation_jobs(jobs, workers=int(args.python_workers))
    manifest.to_csv(output_root / "manifest.csv", index=False)

    runner_path = Path(args.matlab_runner).expanduser().resolve()
    matlab_cmd = build_matlab_command(
        input_dir=input_dir,
        output_dir=phiid_dir,
        measures=[str(x).lower() for x in args.measures],
        matlab_bin=args.matlab_bin,
        matlab_toolbox_root=args.matlab_toolbox_root,
        runner_path=runner_path,
        use_parallel=args.matlab_parallel,
        n_workers=args.matlab_workers,
    )
    (logs_dir / "matlab_command.txt").write_text(matlab_cmd + "\n")
    if args.run_matlab:
        subprocess.run(matlab_cmd, shell=True, cwd=_REPO_ROOT, check=True)
    elif not any(phiid_dir.glob("*.mat")):
        summary = {
            "status": "prepared_simulations_only",
            "output_root": str(output_root),
            "n_jobs": int(len(jobs)),
            "manifest_path": str(output_root / "manifest.csv"),
            "matlab_command": matlab_cmd,
        }
        (logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    index_df = load_three_node_phiid_index(phiid_dir, manifest_path=output_root / "manifest.csv")
    expected_outputs = int(manifest.shape[0]) * int(len(args.measures))
    if int(index_df.shape[0]) != expected_outputs:
        raise RuntimeError(
            f"Incomplete three-node PhiID output set: expected {expected_outputs} files, found {int(index_df.shape[0])}."
        )

    raw_df, avg_df = summarize_three_node_outputs(index_df)
    raw_df.to_csv(tables_dir / "raw_pairwise_values.csv", index=False)
    avg_export = avg_df.drop(columns=["matrix"]).copy()
    avg_export.to_csv(tables_dir / "averaged_pairwise_values.csv", index=False)
    avg_df.to_pickle(tables_dir / "averaged_matrices.pkl")

    g_values = sorted(float(x) for x in args.g_values)
    noise_values = sorted(float(x) for x in args.noise_values)
    for measure in args.measures:
        for atom in ("sts", "rtr"):
            fig, _ = plot_three_node_matrix_grid(
                avg_df,
                measure=str(measure).lower(),
                atom=atom,
                g_values=g_values,
                noise_values=noise_values,
            )
            _save_fig(fig, figures_dir, f"{str(measure).lower()}_{atom}_matrix_grid")

    fig_hyp, _ = plot_three_node_hypothesis_summary(avg_df)
    _save_fig(fig_hyp, figures_dir, "hypothesis_pair_summary")

    _write_tutorial_note(output_root / "README.md", args)

    summary = {
        "status": "completed",
        "output_root": str(output_root),
        "n_jobs": int(len(jobs)),
        "n_phiid_outputs": int(index_df.shape[0]),
        "g_values": [float(x) for x in g_values],
        "noise_values": [float(x) for x in noise_values],
        "seeds": [int(x) for x in args.seeds],
        "measures": [str(x).lower() for x in args.measures],
        "figure_example": str(figures_dir / "mmi_sts_matrix_grid.png"),
    }
    (logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=str, default=str(legacy_results("results", "three_node_adex_phiid_sweep")))
    parser.add_argument("--g-values", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument(
        "--noise-values",
        type=float,
        nargs="+",
        default=[0.0, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 1.25e-4, 1.5e-4, 2e-4, 3e-4],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--measures", nargs="+", default=["mmi", "ccs"])
    parser.add_argument("--simulation-length-ms", type=float, default=5000.0)
    parser.add_argument("--transient-ms", type=float, default=1000.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--monitor-period-ms", type=float, default=1.0)
    parser.add_argument("--zerlaut-order", type=int, default=1)
    parser.add_argument("--python-workers", type=int, default=1)
    parser.add_argument("--run-matlab", action="store_true")
    parser.add_argument("--matlab-parallel", action="store_true", default=False)
    parser.add_argument("--matlab-workers", type=int, default=0)
    parser.add_argument("--matlab-bin", type=str, default="/Applications/MATLAB_R2023b.app/bin/matlab")
    parser.add_argument("--matlab-toolbox-root", type=str, default="/Users/borjan/code/matlab/elph")
    parser.add_argument(
        "--matlab-runner",
        type=str,
        default=str(_REPO_ROOT / "scripts" / "phiid_three_node_adex_sweep.m"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))
