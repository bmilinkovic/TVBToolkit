"""Experiment workflows for condition-wise simulation and metric extraction."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from tvbtoolkit.complexity.measures import lzc_multichannel, pci_casali_like
from tvbtoolkit.core.config import OutputConfig, WholeBrainConfig
from tvbtoolkit.core.io import save_npz
from tvbtoolkit.core.system import recommend_parallel_workers
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation


@dataclass(frozen=True)
class ConditionSpec:
    """One experimental condition definition."""

    name: str
    description: str
    parameter_overrides: dict[str, Any]


def _merge_overrides(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_overrides(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _run_seed_condition_job(job: tuple) -> dict[str, Any]:
    """Worker entrypoint for one `(condition, seed)` simulation + metrics job."""
    (
        base_cfg,
        cond_name,
        cond_overrides,
        seed,
        post_stim_window,
        cond_dir,
        save_timeseries,
        monitor_mode_default,
        temporal_average_period_ms,
    ) = job

    cfg = deepcopy(base_cfg)
    cfg.parameter_overrides = _merge_overrides(cfg.parameter_overrides, cond_overrides)
    # Batch default: use temporal average monitor unless explicitly overridden.
    if cfg.monitor_mode is None and monitor_mode_default is not None:
        cfg.monitor_mode = monitor_mode_default
        if monitor_mode_default == "temporal_average" and temporal_average_period_ms is not None:
            cfg.temporal_average_period_ms = float(temporal_average_period_ms)
    sim_out = run_whole_brain_simulation(cfg, seed=seed)

    x = np.asarray(sim_out.raw, dtype=float)
    x_inh = None if sim_out.raw_inh is None else np.asarray(sim_out.raw_inh, dtype=float)
    t = np.asarray(sim_out.time_ms).reshape(-1)
    stimtime_ms = None
    p_stim = cfg.parameter_overrides.get("parameter_stimulus", {})
    if isinstance(p_stim, dict):
        stimtime_ms = p_stim.get("stimtime", None)
    if stimtime_ms is None:
        stim_idx = x.shape[0] // 2
    else:
        stim_idx = int(np.argmin(np.abs(t - float(stimtime_ms))))

    lzc_val = lzc_multichannel(x)
    # Interpret post_stim_window in milliseconds, then convert to samples using
    # the effective monitor sampling interval. This keeps behavior consistent
    # across raw and temporal-average monitor modes.
    if t.size > 1:
        sample_dt_ms = float(np.median(np.diff(t)))
    else:
        sample_dt_ms = 1.0
    sample_dt_ms = max(sample_dt_ms, 1e-9)
    target_window_samples = max(1, int(round(float(post_stim_window) / sample_dt_ms)))

    # Casali-style PCI uses a symmetric analysis window around stimulation.
    max_valid_window = min(stim_idx, x.shape[0] - stim_idx)
    window_samples = min(target_window_samples, max_valid_window)
    if window_samples < 1:
        # Edge-case fallback for very short outputs or boundary stim times.
        pci_val = float("nan")
    else:
        t_analysis_ms = window_samples * sample_dt_ms
        pci_val = pci_casali_like(
            x,
            stimulation_index=stim_idx,
            t_analysis_ms=t_analysis_ms,
            dt_ms=sample_dt_ms,
        )

    if save_timeseries:
        save_npz(
            Path(cond_dir) / f"seed_{seed:03d}.npz",
            time_ms=sim_out.time_ms,
            raw=x,
            raw_inh=x_inh if x_inh is not None else np.array([], dtype=float),
            region_labels=np.asarray(sim_out.region_labels),
            lzc=np.array([lzc_val], dtype=float),
            pci_casali_like=np.array([pci_val], dtype=float),
        )

    return {
        "condition": cond_name,
        "seed": seed,
        "lzc": float(lzc_val),
        "pci_casali_like": float(pci_val),
        "time_ms": np.asarray(sim_out.time_ms),
        "raw": x,
        "raw_inh": x_inh,
    }


def run_condition_batch(
    base_cfg: WholeBrainConfig,
    conditions: list[ConditionSpec],
    seeds: list[int],
    output: OutputConfig,
    post_stim_window: int = 300,
    save_timeseries: bool = True,
    n_jobs: int | None = None,
    use_processes: bool = True,
    show_progress: bool = True,
    monitor_mode_default: str | None = "temporal_average",
    temporal_average_period_ms: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Run multiple pharmacological conditions and compute LZc / Casali-style PCI.

    `post_stim_window` is specified in milliseconds and converted to monitor
    sample-count internally (raw or temporal-average).

    Saved layout:
      - `simulations/<condition>/seed_<N>.npz`
      - `metrics/<condition>_metrics.npz`

    Monitor behavior:
      - By default, batch runs use `monitor_mode='temporal_average'` for speed.
      - Set `base_cfg.monitor_mode` explicitly to override.
      - Or pass `monitor_mode_default='raw'` / `None`.
    """
    output.simulations_dir.mkdir(parents=True, exist_ok=True)
    output.metrics_dir.mkdir(parents=True, exist_ok=True)
    if monitor_mode_default not in {None, "raw", "temporal_average"}:
        raise ValueError("monitor_mode_default must be one of: None, 'raw', 'temporal_average'.")

    if n_jobs is None:
        n_jobs = recommend_parallel_workers(task="whole_brain_tvb")
    n_jobs = max(1, int(n_jobs))
    t0 = perf_counter()

    results: dict[str, dict[str, np.ndarray]] = {}
    per_cond = {}
    jobs = []
    for cond in conditions:
        cond_dir = output.simulations_dir / cond.name
        cond_dir.mkdir(parents=True, exist_ok=True)
        per_cond[cond.name] = {
            "lzc": [],
            "pci_casali_like": [],
            "seed_results": {},
        }
        for seed in seeds:
            jobs.append(
                (
                    base_cfg,
                    cond.name,
                    cond.parameter_overrides,
                    int(seed),
                    int(post_stim_window),
                    str(cond_dir),
                    bool(save_timeseries),
                    monitor_mode_default,
                    float(temporal_average_period_ms),
                )
            )

    def _progress(iterable, total: int, desc: str):
        if show_progress and tqdm is not None:
            return tqdm(iterable, total=total, desc=desc)
        return iterable

    if use_processes and n_jobs > 1:
        if show_progress:
            print(f"[run_condition_batch] dispatching {len(jobs)} jobs on {n_jobs} processes")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for out in _progress(ex.map(_run_seed_condition_job, jobs), total=len(jobs), desc="Simulating"):
                d = per_cond[out["condition"]]
                d["lzc"].append(out["lzc"])
                d["pci_casali_like"].append(out["pci_casali_like"])
                d["seed_results"][out["seed"]] = out
    else:
        if show_progress:
            print("[run_condition_batch] running sequentially")
        for job in _progress(jobs, total=len(jobs), desc="Simulating"):
            out = _run_seed_condition_job(job)
            d = per_cond[out["condition"]]
            d["lzc"].append(out["lzc"])
            d["pci_casali_like"].append(out["pci_casali_like"])
            d["seed_results"][out["seed"]] = out

    for cond in conditions:
        d = per_cond[cond.name]
        sorted_seeds = sorted(d["seed_results"].keys())
        first = d["seed_results"][sorted_seeds[0]]
        metrics = {
            "lzc": np.asarray(d["lzc"], dtype=float),
            "pci_casali_like": np.asarray(d["pci_casali_like"], dtype=float),
            "seeds": np.asarray(sorted_seeds, dtype=int),
            "time_ms_example": np.asarray(first["time_ms"]),
            "raw_example": np.asarray(first["raw"]),
            "raw_inh_example": np.asarray(first["raw_inh"]) if first["raw_inh"] is not None else np.array([], dtype=float),
        }
        save_npz(output.metrics_dir / f"{cond.name}_metrics.npz", **metrics)
        results[cond.name] = metrics

    if show_progress:
        dt = perf_counter() - t0
        print(f"[run_condition_batch] completed in {dt:.1f}s ({len(conditions)} conditions x {len(seeds)} seeds)")

    return results
