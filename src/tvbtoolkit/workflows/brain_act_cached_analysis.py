"""Parallel cached-analysis pipeline for Brain-Act dual-domain outputs.

This module computes downstream metrics from already-saved simulation `.npz` files
without re-running TVB simulations.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import multiprocessing as mp
import re
from typing import Any

import numpy as np
import pandas as pd

from tvbtoolkit.analysis.brain_states import cluster_brain_states, phase_patterns
from tvbtoolkit.complexity.measures import lzc_multichannel, pci_casali_like
from tvbtoolkit.datasets.brain_act import load_subject_structural
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import (
    _apply_damage_parity,
    _compute_domain_metrics,
    _safe_pearson,
    _upper_triangle_vector,
    _sedation_group,
)


_SEED_RE = re.compile(r"seed_(\d+)\.npz$")


DEFAULT_SCENARIO_LABELS: dict[str, str] = {
    "private_alpha0": "Private only (alpha=0.00)",
    "global_alpha_low": "Global shared, low alpha (0.15)",
    "global_alpha_med": "Global shared, medium alpha (0.40)",
    "global_alpha_high": "Global shared, high alpha (0.70)",
    "sc_alpha_med": "SC-shaped shared, medium alpha (0.40)",
}
_LEGACY_POOLED_BOLD_MAX_ROWS_PER_JOB = 12


def _subsample_rows(x: np.ndarray, max_rows: int | None) -> np.ndarray:
    """Uniformly subsample pooled rows to cap memory while preserving coverage."""
    if max_rows is None:
        return x
    k = int(max_rows)
    if k <= 0 or x.shape[0] <= k:
        return x
    idx = np.linspace(0, x.shape[0] - 1, k, dtype=int)
    return x[idx]


def _subsample_timeseries_rows(
    x: np.ndarray,
    t: np.ndarray,
    *,
    max_rows: int | None,
    trim_edge_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a timeseries before phase-pattern extraction to bound memory.

    Keeps enough samples so that, after edge trimming, approximately `max_rows`
    pattern rows remain.
    """
    if max_rows is None:
        return x, t
    k = int(max_rows)
    if k <= 0:
        return x, t

    keep = int(k + 2 * max(0, int(trim_edge_samples)))
    if x.shape[0] <= keep:
        return x, t

    idx = np.linspace(0, x.shape[0] - 1, keep, dtype=int)
    return x[idx], t[idx]


def _compute_basic_domain_metrics(
    x: np.ndarray,
    t_ms: np.ndarray,
    pci_window_ms: float,
    *,
    compute_pci: bool = True,
) -> dict[str, float]:
    """Compute domain metrics without brain-state clustering."""
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t_ms, dtype=float)
    lzc = float(lzc_multichannel(x_arr))

    if not compute_pci:
        return {"lzc": lzc, "pci": float("nan")}

    if x_arr.shape[0] < 8:
        return {"lzc": lzc, "pci": float("nan")}

    dt_ms = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 1.0
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        dt_ms = 1.0

    stim_idx = x_arr.shape[0] // 2
    max_half = min(stim_idx, x_arr.shape[0] - stim_idx)
    if max_half < 2:
        return {"lzc": lzc, "pci": float("nan")}

    target_bins = max(2, int(round(float(pci_window_ms) / max(dt_ms, 1e-9))))
    window_bins = min(max_half, target_bins)
    t_analysis_ms = float(window_bins * dt_ms)
    pci = float(
        pci_casali_like(
            x_arr,
            stimulation_index=stim_idx,
            t_analysis_ms=t_analysis_ms,
            dt_ms=dt_ms,
        )
    )
    return {"lzc": lzc, "pci": pci}


def _compute_pooled_state_rows(
    *,
    npz_paths: list[Path],
    sim_dir: Path,
    dataset_root: Path,
    n_states: int,
    scenario_labels: dict[str, str],
    rate_max_rows_per_job: int | None,
    bold_max_rows_per_job: int | None,
    random_seed: int,
    show_progress: bool = False,
    pooled_progress_every: int = 25,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build legacy-style pooled states from cached simulations, then score subject SC.

    For each scenario and domain, this function:
    1) pools phase-pattern rows across all subjects/jobs in the scenario,
    2) fits one KMeans state model on the pooled matrix,
    3) computes subject-specific SCFC coupling of pooled centroids,
    4) reports occupancy and SCFC after sorting states by SCFC (ascending).
    """

    jobs_by_scenario: dict[str, list[Path]] = {}
    for p in npz_paths:
        scenario, _cohort, _subject_id, _seed = _parse_job(p, sim_dir)
        jobs_by_scenario.setdefault(scenario, []).append(p)

    struct_cache: dict[tuple[str, str], dict[str, Any]] = {}
    out_rows: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}

    def _get_subject_struct(subject_id: str, cohort: str) -> dict[str, Any]:
        key = (subject_id, cohort)
        if key in struct_cache:
            return struct_cache[key]
        c, l, _atlas, meta = load_subject_structural(
            subject_id=subject_id,
            cohort=cohort,
            dataset_root=str(dataset_root),
            validate=True,
            enforce_symmetry=True,
            zero_diagonal=True,
            nonfinite="raise",
        )
        c, l, sc_zero_frac = _apply_damage_parity(c, l, cohort)
        rec = {
            "sc_vec": _upper_triangle_vector(c),
            "sc_zero_fraction_upper": float(sc_zero_frac),
            "stage": str(getattr(meta, "stage", "") or ""),
            "sedation": str(getattr(meta, "sedation", "") or ""),
        }
        struct_cache[key] = rec
        return rec

    domain_cfg = {
        "rate": {
            "x_key": "rate",
            "t_key": "time_rate_ms",
            "pipeline": "standard",
            "trim_edge_samples": 9,
            "pre_subsample_timeseries": True,
            "bandpass_hz": (0.01, 0.20),
            "filter_order": 3,
            "n_init": 3,
            "max_iter": 60,
            "backend": "scipy",
            "max_rows_per_job": rate_max_rows_per_job,
        },
        "bold": {
            "x_key": "bold",
            "t_key": "time_bold_ms",
            "pipeline": "brain_act_legacy",
            "trim_edge_samples": 0,
            "pre_subsample_timeseries": False,
            "bandpass_hz": (0.01, 0.20),
            "filter_order": 3,
            "n_init": 3,
            "max_iter": 60,
            "backend": "scipy",
            "max_rows_per_job": bold_max_rows_per_job,
        },
    }

    for scenario, paths in sorted(jobs_by_scenario.items()):
        for domain, cfg in domain_cfg.items():
            paths_sorted = sorted(paths)
            n_paths = len(paths_sorted)
            t_domain = perf_counter()
            if show_progress:
                _log_line(
                    f"[cached-analysis pooled] scenario={scenario} domain={domain} jobs={n_paths} max_rows_per_job={cfg['max_rows_per_job']}",
                    log_path,
                )
            pooled_blocks: list[np.ndarray] = []
            splits: list[tuple[Path, int, int]] = []
            offset = 0

            for i, p in enumerate(paths_sorted, start=1):
                d = np.load(p, allow_pickle=True)
                x = np.asarray(d[cfg["x_key"]], dtype=float)
                t = np.asarray(d[cfg["t_key"]], dtype=float)
                if x.ndim != 2 or x.shape[0] < max(10, n_states):
                    continue

                if bool(cfg.get("pre_subsample_timeseries", False)):
                    x_fit, t_fit = _subsample_timeseries_rows(
                        x,
                        t,
                        max_rows=cfg["max_rows_per_job"],
                        trim_edge_samples=int(cfg["trim_edge_samples"]),
                    )
                else:
                    x_fit, t_fit = x, t

                tr_s = float(np.median(np.diff(t_fit))) / 1000.0 if t_fit.size > 1 else 2.4
                patterns, _global_sync, _iu, _ju = phase_patterns(
                    x_fit,
                    trim_edge_samples=int(cfg["trim_edge_samples"]),
                    pipeline=str(cfg["pipeline"]),
                    tr_seconds=max(tr_s, 1e-6),
                    bandpass_hz=cfg["bandpass_hz"],
                    filter_order=int(cfg["filter_order"]),
                )
                if patterns.shape[0] < max(2, n_states):
                    continue

                block = _subsample_rows(patterns, cfg["max_rows_per_job"]).astype(np.float32, copy=False)
                if block.shape[0] < max(2, n_states):
                    continue

                pooled_blocks.append(block)
                start = offset
                offset += int(block.shape[0])
                splits.append((p, start, offset))

                if show_progress and (i % max(1, pooled_progress_every) == 0 or i == n_paths):
                    elapsed_s = perf_counter() - t_domain
                    eta_s = (elapsed_s / float(i)) * float(max(0, n_paths - i))
                    _log_line(
                        f"[cached-analysis pooled] scenario={scenario} domain={domain} phase_rows={offset} "
                        f"jobs_done={i}/{n_paths} elapsed_s={elapsed_s:.1f} eta_s={eta_s:.1f}",
                        log_path,
                    )

            if not pooled_blocks:
                if show_progress:
                    _log_line(f"[cached-analysis pooled] scenario={scenario} domain={domain} no pooled rows", log_path)
                continue

            pooled = np.concatenate(pooled_blocks, axis=0)
            try:
                labels, centers = cluster_brain_states(
                    pooled,
                    n_states=n_states,
                    random_seed=int(random_seed),
                    n_init=int(cfg["n_init"]),
                    max_iter=int(cfg["max_iter"]),
                    backend=str(cfg["backend"]),
                )
            except Exception:
                # Fallback for environments where sklearn stack is unavailable.
                labels, centers = cluster_brain_states(
                    pooled,
                    n_states=n_states,
                    random_seed=int(random_seed),
                    n_init=int(cfg["n_init"]),
                    max_iter=int(cfg["max_iter"]),
                    backend="scipy",
                )

            n_eff = int(centers.shape[0]) if centers.ndim == 2 else 0
            if n_eff <= 0:
                continue
            if show_progress:
                elapsed_s = perf_counter() - t_domain
                _log_line(
                    f"[cached-analysis pooled] scenario={scenario} domain={domain} clustered_rows={pooled.shape[0]} "
                    f"states={n_eff} elapsed_s={elapsed_s:.1f}",
                    log_path,
                )

            # ── Recompute occupancy from full (non-subsampled) timeseries ──────────
            # The pooled K-means used subsampled phase patterns (max_rows_per_job
            # rows per subject) for memory efficiency.  Using those same labels
            # to compute occupancy makes it a multiple of 1/max_rows_per_job,
            # producing artificial discrete clusters in the scatter plots.
            # Instead, we do a second pass: assign every timepoint in the full
            # signal to the nearest centroid and compute occupancy from that.
            centers_f64 = np.asarray(centers, dtype=float)
            full_occ: dict[Path, np.ndarray] = {}
            for p_occ in [s[0] for s in splits]:
                d_occ = np.load(p_occ, allow_pickle=True)
                x_occ = np.asarray(d_occ[cfg["x_key"]], dtype=float)
                t_occ = np.asarray(d_occ[cfg["t_key"]], dtype=float)
                if x_occ.ndim != 2 or x_occ.shape[0] < max(10, n_states):
                    full_occ[p_occ] = np.full(n_eff, float("nan"))
                    continue
                tr_s_occ = float(np.median(np.diff(t_occ))) / 1000.0 if t_occ.size > 1 else 2.4
                try:
                    pats_full, *_ = phase_patterns(
                        x_occ,
                        trim_edge_samples=int(cfg["trim_edge_samples"]),
                        pipeline=str(cfg["pipeline"]),
                        tr_seconds=max(tr_s_occ, 1e-6),
                        bandpass_hz=cfg["bandpass_hz"],
                        filter_order=int(cfg["filter_order"]),
                    )
                except Exception:
                    full_occ[p_occ] = np.full(n_eff, float("nan"))
                    continue
                if pats_full.shape[0] < 1:
                    full_occ[p_occ] = np.full(n_eff, float("nan"))
                    continue
                # Nearest centroid assignment via squared Euclidean distance
                diffs = pats_full[:, None, :].astype(float) - centers_f64[None, :, :]
                full_lab = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)
                full_occ[p_occ] = (
                    np.bincount(full_lab, minlength=n_eff).astype(float)
                    / float(max(1, full_lab.size))
                )

            for p, a, b in splits:
                scenario_k, cohort, subject_id, seed = _parse_job(p, sim_dir)
                if b <= a:
                    continue
                occ = full_occ.get(p, np.full(n_eff, float("nan")))

                subj = _get_subject_struct(subject_id, cohort)
                sc_vec = np.asarray(subj["sc_vec"], dtype=float)
                sfc = np.asarray([_safe_pearson(row, sc_vec) for row in np.asarray(centers, dtype=float)], dtype=float)
                order = np.argsort(np.nan_to_num(sfc, nan=np.inf))

                for j in range(n_eff):
                    idx = int(order[j])
                    key = (scenario_k, cohort, subject_id, int(seed), int(j + 1))
                    row = out_rows.get(key)
                    if row is None:
                        sed = str(subj["sedation"])
                        row = {
                            "scenario": scenario_k,
                            "scenario_label": scenario_labels.get(scenario_k, scenario_k),
                            "cohort": cohort,
                            "stage": str(subj["stage"]),
                            "sedation": sed,
                            "sedation_group": _sedation_group(sed),
                            "subject_id": subject_id,
                            "seed": int(seed),
                            "state_rank": int(j + 1),
                            "sfc_rate": float("nan"),
                            "occ_rate": float("nan"),
                            "sfc_bold": float("nan"),
                            "occ_bold": float("nan"),
                            "sc_zero_fraction_upper": float(subj["sc_zero_fraction_upper"]),
                        }
                    if domain == "rate":
                        row["sfc_rate"] = float(sfc[idx])
                        row["occ_rate"] = float(occ[idx])
                    else:
                        row["sfc_bold"] = float(sfc[idx])
                        row["occ_bold"] = float(occ[idx])
                    out_rows[key] = row

    rows = list(out_rows.values())
    rows.sort(key=lambda r: (r["scenario"], r["cohort"], r["subject_id"], int(r["seed"]), int(r["state_rank"])))
    return rows


def _parse_job(npz_path: Path, sim_dir: Path) -> tuple[str, str, str, int]:
    rel = npz_path.relative_to(sim_dir)
    scenario, cohort, subject_id, fname = rel.parts
    m = _SEED_RE.match(fname)
    seed = int(m.group(1)) if m else 0
    return scenario, cohort, subject_id, seed


def _analyze_cached_job(
    npz_path_s: str,
    sim_dir_s: str,
    dataset_root_s: str,
    n_states: int,
    pci_window_rate_ms: float,
    pci_window_bold_ms: float,
    scenario_labels: dict[str, str],
    compute_local_states: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Analyze one cached simulation file and compute dual-domain metrics."""
    t0 = perf_counter()
    npz_path = Path(npz_path_s)
    sim_dir = Path(sim_dir_s)
    dataset_root = Path(dataset_root_s)

    scenario, cohort, subject_id, seed = _parse_job(npz_path, sim_dir)
    prefix = {
        "scenario": scenario,
        "cohort": cohort,
        "subject_id": subject_id,
        "seed": int(seed),
    }

    try:
        d = np.load(npz_path, allow_pickle=True)
        t_rate_ms = np.asarray(d["time_rate_ms"], dtype=float)
        x_rate = np.asarray(d["rate"], dtype=float)
        t_bold_ms = np.asarray(d["time_bold_ms"], dtype=float)
        x_bold = np.asarray(d["bold"], dtype=float)

        c, l, _atlas, meta = load_subject_structural(
            subject_id=subject_id,
            cohort=cohort,
            dataset_root=str(dataset_root),
            validate=True,
            enforce_symmetry=True,
            zero_diagonal=True,
            nonfinite="raise",
        )
        c, l, sc_zero_frac = _apply_damage_parity(c, l, cohort)

        stage = str(getattr(meta, "stage", "") or "")
        sedation = str(getattr(meta, "sedation", "") or "")
        sed_group = _sedation_group(sedation)

        if compute_local_states:
            met_rate = _compute_domain_metrics(
                x_rate,
                t_rate_ms,
                c,
                n_states=n_states,
                pci_window_ms=pci_window_rate_ms,
                compute_pci=True,
                brain_state_pipeline="standard",
                brain_state_trim_edge_samples=9,
                brain_state_tr_seconds=float(np.median(np.diff(t_rate_ms))) / 1000.0 if t_rate_ms.size > 1 else 0.25,
                brain_state_bandpass_hz=(0.01, 0.20),
                brain_state_n_init=10,
            )
        else:
            met_rate = _compute_basic_domain_metrics(
                x_rate,
                t_rate_ms,
                pci_window_rate_ms,
                compute_pci=True,
            )

        met_bold = None
        if x_bold.shape[0] >= max(10, n_states):
            if compute_local_states:
                bold_tr_s = float(np.median(np.diff(t_bold_ms))) / 1000.0 if t_bold_ms.size > 1 else 2.4
                met_bold = _compute_domain_metrics(
                    x_bold,
                    t_bold_ms,
                    c,
                    n_states=n_states,
                    pci_window_ms=pci_window_bold_ms,
                    compute_pci=False,
                    brain_state_pipeline="brain_act_legacy",
                    brain_state_trim_edge_samples=0,
                    brain_state_tr_seconds=bold_tr_s,
                    brain_state_bandpass_hz=(0.01, 0.20),
                    brain_state_n_init=20,
                )
            else:
                met_bold = _compute_basic_domain_metrics(
                    x_bold,
                    t_bold_ms,
                    pci_window_bold_ms,
                    compute_pci=False,
                )

        metric_row = {
            "scenario": scenario,
            "scenario_label": scenario_labels.get(scenario, scenario),
            "cohort": cohort,
            "stage": stage,
            "sedation": sedation,
            "sedation_group": sed_group,
            "subject_id": subject_id,
            "seed": int(seed),
            "sc_zero_fraction_upper": float(sc_zero_frac),
            "n_rate_samples": int(x_rate.shape[0]),
            "n_bold_samples": int(x_bold.shape[0]),
            "lzc_rate": float(met_rate["lzc"]),
            "pci_rate": float(met_rate["pci"]),
            "lzc_bold": float(met_bold["lzc"]) if met_bold is not None else float("nan"),
            "pci_bold": float(met_bold["pci"]) if met_bold is not None else float("nan"),
        }

        state_rows: list[dict[str, Any]] = []
        if compute_local_states:
            n_state = len(met_rate["occupancy_sfc_sorted"])
            for j in range(n_state):
                sfc_b = (
                    float(met_bold["sfc_sorted"][j]) if met_bold is not None and j < len(met_bold["sfc_sorted"]) else float("nan")
                )
                occ_b = (
                    float(met_bold["occupancy_sfc_sorted"][j])
                    if met_bold is not None and j < len(met_bold["occupancy_sfc_sorted"])
                    else float("nan")
                )
                state_rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario_labels.get(scenario, scenario),
                        "cohort": cohort,
                        "stage": stage,
                        "sedation": sedation,
                        "sedation_group": sed_group,
                        "subject_id": subject_id,
                        "seed": int(seed),
                        "state_rank": j + 1,
                        "sfc_rate": float(met_rate["sfc_sorted"][j]),
                        "occ_rate": float(met_rate["occupancy_sfc_sorted"][j]),
                        "sfc_bold": sfc_b,
                        "occ_bold": occ_b,
                        "sc_zero_fraction_upper": float(sc_zero_frac),
                    }
                )

        metric_row["runtime_s"] = float(perf_counter() - t0)
        return metric_row, state_rows, None

    except Exception as e:  # pragma: no cover - pass-through for notebook/user debugging
        err = {
            **prefix,
            "path": str(npz_path),
            "error_type": type(e).__name__,
            "error": str(e),
            "runtime_s": float(perf_counter() - t0),
        }
        return None, [], err


def _derive_coma_subgroup(row: pd.Series) -> str:
    cohort = str(row.get("cohort", "")).strip().lower()
    sed = str(row.get("sedation_group", "")).strip().lower()
    src = str(row.get("source_sc_file", "")).strip().lower()

    has_coma = (cohort == "coma") or ("coma" in src)
    if not has_coma:
        return "non_coma"
    if ("sedated_coma" in src) or (sed == "sedated"):
        return "coma_sedated"
    if ("acute_coma" in src) or (sed == "non_sedated"):
        return "coma_non_sedated"
    return "coma_unknown"


def _log_line(msg: str, log_path: Path | None) -> None:
    print(msg, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")


def run_cached_dual_domain_analysis(
    *,
    sim_dir: str | Path,
    dataset_root: str | Path,
    n_states: int = 5,
    pci_window_rate_ms: float = 300.0,
    pci_window_bold_ms: float = 30000.0,
    n_jobs: int = 8,
    use_processes: bool = True,
    show_progress: bool = True,
    progress_every: int = 1,
    log_path: str | Path | None = None,
    scenario_labels: dict[str, str] | None = None,
    state_mode: str = "subject_local",
    pooled_rate_max_rows_per_job: int | None = 120,
    pooled_bold_max_rows_per_job: int | None = _LEGACY_POOLED_BOLD_MAX_ROWS_PER_JOB,
    pooled_random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute downstream metrics from cached dual-domain simulation files.

    Parameters
    ----------
    sim_dir : str | Path
        Root directory containing cached `.npz` files arranged as
        `scenario/cohort/subject_id/seed_XXX.npz`.
    dataset_root : str | Path
        Brain-Act converted dataset root used to load subject structural metadata.
    n_states : int, default=5
        Number of k-means states in `summarize_brain_states`.
    pci_window_rate_ms : float, default=300.0
        Casali-like PCI window for rate domain (ms).
    pci_window_bold_ms : float, default=30000.0
        Casali-like PCI window for BOLD domain (ms).
    n_jobs : int, default=8
        Maximum number of concurrent workers.
    use_processes : bool, default=True
        Use `ProcessPoolExecutor` (True) or `ThreadPoolExecutor` (False).
    show_progress : bool, default=True
        Emit progress lines while jobs complete.
    progress_every : int, default=1
        Log every N completed jobs.
    log_path : str | Path | None
        Optional path to append progress lines.
    scenario_labels : dict[str, str] | None
        Optional label map for scenario keys.
    state_mode : {"subject_local", "legacy_pooled"}, default="subject_local"
        Brain-state extraction mode for `states_df`.
        - `subject_local`: current per-subject/per-job local state fitting.
        - `legacy_pooled`: pooled-state fit per scenario/domain, then subject SCFC.
    pooled_rate_max_rows_per_job : int | None, default=120
        Only used when `state_mode='legacy_pooled'`. Caps pooled rate rows per
        job before clustering to keep memory bounded.
    pooled_bold_max_rows_per_job : int | None, default=12
        Only used when `state_mode='legacy_pooled'`. Caps pooled BOLD rows per
        job before clustering to bound pooled-memory/runtime.
    pooled_random_seed : int, default=0
        RNG seed used by pooled clustering restarts.

    Returns
    -------
    metrics_df : pandas.DataFrame
        One row per cached file with LZc/PCI for rates and BOLD.
    states_df : pandas.DataFrame
        One row per `(file, state_rank)` with occupancy and SCFC coupling per domain.
    errors_df : pandas.DataFrame
        One row per failed file analysis (empty if none).
    """
    sim_dir_p = Path(sim_dir).expanduser().resolve()
    dataset_root_p = Path(dataset_root).expanduser().resolve()
    log_path_p = Path(log_path).expanduser().resolve() if log_path is not None else None

    scenario_labels = dict(DEFAULT_SCENARIO_LABELS if scenario_labels is None else scenario_labels)
    state_mode = str(state_mode).strip().lower()
    if state_mode not in {"subject_local", "legacy_pooled"}:
        raise ValueError("state_mode must be one of: 'subject_local', 'legacy_pooled'.")

    npz_paths = sorted(sim_dir_p.glob("*/*/*/seed_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No cached simulation files found under: {sim_dir_p}")

    _log_line(
        f"[cached-analysis] start jobs={len(npz_paths)} backend={'process' if use_processes else 'thread'} "
        f"workers={n_jobs} local_states={'on' if state_mode == 'subject_local' else 'off'}",
        log_path_p,
    )

    t0 = perf_counter()
    metric_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    ex_kwargs: dict[str, Any] = {"max_workers": int(max(1, n_jobs))}
    if use_processes:
        ex_kwargs["mp_context"] = mp.get_context("spawn")

    with Executor(**ex_kwargs) as ex:
        compute_local_states = state_mode == "subject_local"
        futs = {
            ex.submit(
                _analyze_cached_job,
                str(p),
                str(sim_dir_p),
                str(dataset_root_p),
                int(n_states),
                float(pci_window_rate_ms),
                float(pci_window_bold_ms),
                scenario_labels,
                compute_local_states,
            ): p
            for p in npz_paths
        }

        done = 0
        total = len(futs)
        for fut in as_completed(futs):
            done += 1
            metric_row, rows_state, err = fut.result()

            if metric_row is not None:
                metric_rows.append(metric_row)
                if compute_local_states:
                    state_rows.extend(rows_state)
            if err is not None:
                error_rows.append(err)

            if show_progress and (done % max(1, progress_every) == 0 or done == total):
                elapsed_s = perf_counter() - t0
                eta_s = (elapsed_s / float(done)) * float(max(0, total - done))
                p = futs[fut]
                scenario, cohort, subject_id, seed = _parse_job(p, sim_dir_p)
                status = "error" if err is not None else "ok"
                _log_line(
                    f"[cached-analysis {done}/{total}] scenario={scenario} cohort={cohort} subject={subject_id} "
                    f"seed={seed} status={status} elapsed_s={elapsed_s:.1f} eta_s={eta_s:.1f}",
                    log_path_p,
                )

    if state_mode == "legacy_pooled":
        scenario_counts: dict[str, int] = {}
        for p in npz_paths:
            s, _c, _sid, _seed = _parse_job(p, sim_dir_p)
            scenario_counts[s] = scenario_counts.get(s, 0) + 1
        _log_line(
            "[cached-analysis] legacy-pooled mode: fitting states independently per scenario (no cross-scenario pooling). "
            f"scenario_jobs={scenario_counts}",
            log_path_p,
        )
        _log_line("[cached-analysis] building legacy-pooled state rows from cached simulations...", log_path_p)
        state_rows = _compute_pooled_state_rows(
            npz_paths=npz_paths,
            sim_dir=sim_dir_p,
            dataset_root=dataset_root_p,
            n_states=int(n_states),
            scenario_labels=scenario_labels,
            rate_max_rows_per_job=pooled_rate_max_rows_per_job,
            bold_max_rows_per_job=pooled_bold_max_rows_per_job,
            random_seed=int(pooled_random_seed),
            show_progress=show_progress,
            pooled_progress_every=25,
            log_path=log_path_p,
        )
        _log_line(f"[cached-analysis] legacy-pooled state rows: {len(state_rows)}", log_path_p)

    metrics_df = pd.DataFrame(metric_rows)
    states_df = pd.DataFrame(state_rows)
    if not metrics_df.empty:
        metrics_df["state_mode"] = state_mode
    if not states_df.empty:
        states_df["state_mode"] = state_mode
    errors_df = pd.DataFrame(error_rows)

    elapsed = perf_counter() - t0
    _log_line(
        f"[cached-analysis] completed metrics_rows={len(metrics_df)} state_rows={len(states_df)} errors={len(errors_df)} elapsed_s={elapsed:.2f}",
        log_path_p,
    )
    return metrics_df, states_df, errors_df


def finalize_cached_analysis_tables(
    *,
    metrics_df: pd.DataFrame,
    states_df: pd.DataFrame,
    dataset_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach source metadata + coma subgroup labels to cached-analysis tables."""
    dataset_root_p = Path(dataset_root).expanduser().resolve()
    source_map_path = dataset_root_p / "source_subject_map.csv"

    metrics_out = metrics_df.copy()
    states_out = states_df.copy()

    if source_map_path.exists() and not metrics_out.empty:
        src_df = pd.read_csv(source_map_path)
        keep = [
            c for c in ["subject_id", "source_sc_file", "source_tl_file", "source_subject_index"] if c in src_df.columns
        ]
        src_df = src_df[keep].drop_duplicates(["subject_id"])
        metrics_out = metrics_out.merge(src_df, on="subject_id", how="left")
        states_out = states_out.merge(src_df, on="subject_id", how="left")
    else:
        if "source_sc_file" not in metrics_out.columns:
            metrics_out["source_sc_file"] = ""
        if "source_sc_file" not in states_out.columns:
            states_out["source_sc_file"] = ""

    if not metrics_out.empty:
        metrics_out["coma_subgroup"] = metrics_out.apply(_derive_coma_subgroup, axis=1)
    if not states_out.empty:
        states_out["coma_subgroup"] = states_out.apply(_derive_coma_subgroup, axis=1)

    if metrics_out.empty:
        subject_groups = pd.DataFrame(columns=["cohort", "subject_id", "stage", "sedation_group", "coma_subgroup"])
    else:
        subject_groups = metrics_out[
            ["cohort", "subject_id", "stage", "sedation_group", "coma_subgroup"]
        ].drop_duplicates(["cohort", "subject_id"])

    return metrics_out, states_out, subject_groups


__all__ = [
    "DEFAULT_SCENARIO_LABELS",
    "run_cached_dual_domain_analysis",
    "finalize_cached_analysis_tables",
]
