"""Subject-specific Brain-Act AAL90 whole-brain simulation workflows."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from tvbtoolkit.analysis.brain_states import brain_state_metrics_dict, summarize_brain_states
from tvbtoolkit.complexity.measures import lzc_multichannel, pci_casali_like
from tvbtoolkit.core.config import OutputConfig, WholeBrainConfig
from tvbtoolkit.core.io import save_npz
from tvbtoolkit.core.system import recommend_parallel_workers
from tvbtoolkit.datasets.brain_act import list_subjects, load_subject_structural
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation


_DEF_GOLDMAN_PARAMETER_MODEL = {
    "T": 40.0,
    "P_e": [-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407],
    "P_i": [-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153],
    "E_L_e": -63.0,
    "E_L_i": -65.0,
    "b_e": 5.0,
    "tau_e_e": 5.0,
    "tau_e_i": 5.0,
    "initial_condition": {
        "E": [0.004, 0.004],
        "I": [0.010, 0.010],
        "C_ee": [0.0, 0.0],
        "C_ei": [0.0, 0.0],
        "C_ii": [0.0, 0.0],
        "W_e": [50.0, 50.0],
        "W_i": [0.0, 0.0],
        "noise": [0.0, 0.0],
    },
}


def _upper_triangle_vector(x: np.ndarray) -> np.ndarray:
    """Return upper-triangle vector (k=1) from a square matrix."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got shape={arr.shape}.")
    iu = np.triu_indices(arr.shape[0], k=1)
    return np.asarray(arr[iu], dtype=float)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson r with zero-variance / non-finite safeguards."""
    xa = np.asarray(a, dtype=float).reshape(-1)
    xb = np.asarray(b, dtype=float).reshape(-1)
    if xa.size != xb.size or xa.size == 0:
        return float("nan")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(xb)):
        return float("nan")
    sa = float(np.std(xa))
    sb = float(np.std(xb))
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


@dataclass(frozen=True)
class BrainActSubjectConfig:
    """Configuration for subject-specific AAL90 whole-brain simulations.

    Defaults match the AdEx/Zerlaut first-order setup used for ketamine/
    psilocybin notebooks, with temporal averaging enabled for speed.
    """

    dataset_root: str | Path | None = None
    output_root: str | Path = Path("notebooks/outputs/brain_act_subject_batches")
    seeds: tuple[int, ...] = (0, 1, 2)

    simulation_length_ms: float = 5000.0
    dt_ms: float = 0.1
    conduction_speed: float = 4.0
    coupling_strength: float = 0.25

    model_family: Literal["adex_zerlaut", "generic2d"] = "adex_zerlaut"
    zerlaut_matteo: bool = False
    zerlaut_gk_gna: bool = False
    zerlaut_order: Literal[1, 2] = 1
    stochastic_integrator: bool = True

    monitor_mode: Literal["raw", "temporal_average"] = "temporal_average"
    temporal_average_period_ms: float = 1.0
    monitor_variables: tuple[int, ...] = (0, 1)

    # No receptor modulation for this Brain-Act subject-specific pipeline.
    parameter_overrides: dict[str, Any] = field(
        default_factory=lambda: {"parameter_model": deepcopy(_DEF_GOLDMAN_PARAMETER_MODEL)}
    )

    post_stim_window_ms: float = 300.0
    compute_brain_states: bool = True
    brain_state_k: int = 5
    brain_state_pipeline: Literal["standard", "brain_act_legacy"] = "standard"
    brain_state_trim_edge_samples: int = 9
    brain_state_clustering_backend: Literal["scipy", "sklearn"] | None = None
    brain_state_n_init: int = 20
    brain_state_max_iter: int = 100
    brain_state_tr_seconds: float = 2.4
    brain_state_bandpass_hz: tuple[float, float] = (0.01, 0.20)
    brain_state_filter_order: int = 3

    save_timeseries: bool = True

    validate_structural: bool = True
    enforce_symmetry: bool = True
    zero_diagonal: bool = True
    nonfinite: str = "raise"
    normalize_connectivity: str | None = None
    threshold_connectivity: float | None = None
    percentile_connectivity: float | None = None

    # Brain-Act parity: source matrices already encode lesion damage as zeros.
    # If mismatches are found in patient cohorts, optionally hard-enforce TL=0
    # on damaged SC edges for strict reproducibility.
    enforce_patient_tl_mask_if_needed: bool = True



def _merge_overrides(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_overrides(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out



def _subject_output_dirs(output_root: Path, cohort: str, subject_id: str) -> tuple[Path, Path]:
    sim_dir = output_root / "simulations" / cohort / subject_id
    metrics_dir = output_root / "metrics" / cohort
    sim_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return sim_dir, metrics_dir



def _damage_mask_report(c: np.ndarray, l: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices_from(c, k=1)
    c_zero = c[iu] == 0.0
    l_zero = l[iu] == 0.0
    n_edges = int(c_zero.size)
    sc_zero_count = int(np.sum(c_zero))
    tl_zero_count = int(np.sum(l_zero))
    sc_zero_tl_nonzero = int(np.sum(c_zero & ~l_zero))
    return {
        "n_edges_upper": float(n_edges),
        "sc_zero_edges_upper": float(sc_zero_count),
        "tl_zero_edges_upper": float(tl_zero_count),
        "sc_zero_tl_nonzero_upper": float(sc_zero_tl_nonzero),
        "sc_zero_fraction_upper": float(sc_zero_count / max(n_edges, 1)),
        "tl_zero_fraction_upper": float(tl_zero_count / max(n_edges, 1)),
    }



def _apply_brain_act_damage_parity(
    c: np.ndarray,
    l: np.ndarray,
    cohort: str,
    *,
    enforce_patient_tl_mask_if_needed: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply Brain-Act lesion parity handling.

    Original Brain-Act pipeline uses SC/TL matrices as provided in data files;
    lesion edges are encoded as zeros and then visualised directly. We preserve
    that behaviour. For robustness, if patient cohorts contain SC-zero but TL-
    nonzero edges, we can enforce TL=0 on those edges.
    """
    c = np.asarray(c, dtype=float).copy()
    l = np.asarray(l, dtype=float).copy()
    np.fill_diagonal(c, 0.0)
    np.fill_diagonal(l, 0.0)

    report = _damage_mask_report(c, l)
    cohort_l = str(cohort).lower()
    is_patient = cohort_l in {"emcs", "mcs", "uws", "coma"}

    if is_patient and enforce_patient_tl_mask_if_needed:
        mismatch = (c == 0.0) & (l != 0.0)
        if np.any(mismatch):
            l[mismatch] = 0.0
            report = _damage_mask_report(c, l)
            report["patient_tl_mask_enforced"] = 1.0
        else:
            report["patient_tl_mask_enforced"] = 0.0
    else:
        report["patient_tl_mask_enforced"] = 0.0

    return c, l, report



def _build_subject_cfg(base: BrainActSubjectConfig, c: np.ndarray, l: np.ndarray) -> WholeBrainConfig:
    overrides = _merge_overrides({}, base.parameter_overrides)
    return WholeBrainConfig(
        simulation_length_ms=float(base.simulation_length_ms),
        dt_ms=float(base.dt_ms),
        conduction_speed=float(base.conduction_speed),
        coupling_strength=float(base.coupling_strength),
        model_family=base.model_family,
        zerlaut_matteo=bool(base.zerlaut_matteo),
        zerlaut_gk_gna=bool(base.zerlaut_gk_gna),
        zerlaut_order=int(base.zerlaut_order),
        stochastic_integrator=bool(base.stochastic_integrator),
        monitor_mode=base.monitor_mode,
        temporal_average_period_ms=float(base.temporal_average_period_ms),
        monitor_variables=tuple(base.monitor_variables),
        weights=np.asarray(c, dtype=float),
        tract_lengths=np.asarray(l, dtype=float),
        parameter_overrides=overrides,
    )



def _compute_pci_casali_from_timeseries(
    x: np.ndarray,
    t_ms: np.ndarray,
    post_stim_window_ms: float,
) -> float:
    if t_ms.size > 1:
        sample_dt_ms = float(np.median(np.diff(t_ms)))
    else:
        sample_dt_ms = 1.0
    sample_dt_ms = max(sample_dt_ms, 1e-9)

    stim_idx = x.shape[0] // 2
    target = max(1, int(round(float(post_stim_window_ms) / sample_dt_ms)))
    max_valid = min(stim_idx, x.shape[0] - stim_idx)
    window = min(target, max_valid)
    if window < 1:
        return float("nan")
    t_analysis_ms = window * sample_dt_ms
    return float(
        pci_casali_like(
            x,
            stimulation_index=stim_idx,
            t_analysis_ms=t_analysis_ms,
            dt_ms=sample_dt_ms,
        )
    )



def run_subject_simulation(
    subject_id: str,
    cohort: str,
    cfg: BrainActSubjectConfig,
) -> dict[str, Any]:
    """Run one subject-specific whole-brain batch and return metrics/results."""
    output_root = Path(cfg.output_root)
    output = OutputConfig(root=output_root)

    c, l, atlas, meta = load_subject_structural(
        subject_id=subject_id,
        cohort=cohort,
        dataset_root=cfg.dataset_root,
        validate=cfg.validate_structural,
        enforce_symmetry=cfg.enforce_symmetry,
        zero_diagonal=cfg.zero_diagonal,
        nonfinite=cfg.nonfinite,
        normalize=cfg.normalize_connectivity,
        threshold=cfg.threshold_connectivity,
        percentile=cfg.percentile_connectivity,
    )

    c, l, mask_report = _apply_brain_act_damage_parity(
        c,
        l,
        cohort=meta.cohort,
        enforce_patient_tl_mask_if_needed=cfg.enforce_patient_tl_mask_if_needed,
    )

    wb_cfg = _build_subject_cfg(cfg, c, l)
    sim_dir, metrics_dir = _subject_output_dirs(output.root, meta.cohort, subject_id)

    seeds = [int(s) for s in cfg.seeds]
    lzc_vals = []
    pci_vals = []
    brain_occ = []
    brain_transitions = []
    brain_centers = []
    brain_sfc = []
    brain_occ_sfc_sorted = []
    brain_sfc_sorted = []

    sc_vec = _upper_triangle_vector(c)

    time_example = None
    raw_example = None
    raw_inh_example = None
    region_labels = None

    for seed in seeds:
        sim = run_whole_brain_simulation(wb_cfg, seed=seed)
        t_ms = np.asarray(sim.time_ms, dtype=float)
        x = np.asarray(sim.raw, dtype=float)
        x_inh = None if sim.raw_inh is None else np.asarray(sim.raw_inh, dtype=float)

        if cfg.save_timeseries:
            save_npz(
                sim_dir / f"seed_{seed:03d}.npz",
                time_ms=t_ms,
                raw=x,
                raw_inh=np.array([], dtype=float) if x_inh is None else x_inh,
                region_labels=np.asarray(sim.region_labels),
            )

        lzc_vals.append(float(lzc_multichannel(x)))
        pci_vals.append(float(_compute_pci_casali_from_timeseries(x, t_ms, cfg.post_stim_window_ms)))

        if cfg.compute_brain_states:
            bs = summarize_brain_states(
                x,
                n_states=cfg.brain_state_k,
                trim_edge_samples=cfg.brain_state_trim_edge_samples,
                random_seed=seed,
                n_init=cfg.brain_state_n_init,
                max_iter=cfg.brain_state_max_iter,
            )
            occ = np.asarray(bs.occupancy, dtype=float)
            centers = np.asarray(bs.centers, dtype=float)
            sfc_vals = np.asarray([_safe_pearson(row, sc_vec) for row in centers], dtype=float)
            order = np.argsort(np.nan_to_num(sfc_vals, nan=np.inf))

            brain_occ.append(occ)
            brain_transitions.append(np.asarray(bs.transition_matrix, dtype=float))
            brain_centers.append(centers)
            brain_sfc.append(sfc_vals)
            brain_occ_sfc_sorted.append(occ[order])
            brain_sfc_sorted.append(sfc_vals[order])

        if time_example is None:
            time_example = t_ms
            raw_example = x
            raw_inh_example = np.array([], dtype=float) if x_inh is None else x_inh
            region_labels = np.asarray(sim.region_labels)

    metrics: dict[str, Any] = {
        "subject_id": subject_id,
        "cohort": meta.cohort,
        "stage": np.asarray([meta.stage if meta.stage is not None else ""], dtype="U32"),
        "sedation": np.asarray([meta.sedation if meta.sedation is not None else ""], dtype="U32"),
        "seeds": np.asarray(seeds, dtype=int),
        "lzc": np.asarray(lzc_vals, dtype=float),
        "pci_casali_like": np.asarray(pci_vals, dtype=float),
        "time_ms_example": np.asarray(time_example, dtype=float),
        "raw_example": np.asarray(raw_example, dtype=float),
        "raw_inh_example": np.asarray(raw_inh_example, dtype=float),
        "region_labels": np.asarray(region_labels),
        "atlas_labels": np.asarray(atlas.labels),
        "mask_n_edges_upper": np.array([mask_report["n_edges_upper"]], dtype=float),
        "mask_sc_zero_edges_upper": np.array([mask_report["sc_zero_edges_upper"]], dtype=float),
        "mask_tl_zero_edges_upper": np.array([mask_report["tl_zero_edges_upper"]], dtype=float),
        "mask_sc_zero_tl_nonzero_upper": np.array([mask_report["sc_zero_tl_nonzero_upper"]], dtype=float),
        "mask_sc_zero_fraction_upper": np.array([mask_report["sc_zero_fraction_upper"]], dtype=float),
        "mask_tl_zero_fraction_upper": np.array([mask_report["tl_zero_fraction_upper"]], dtype=float),
        "mask_patient_tl_mask_enforced": np.array([mask_report["patient_tl_mask_enforced"]], dtype=float),
    }

    if cfg.compute_brain_states and brain_occ:
        occ = np.stack(brain_occ, axis=0)
        transitions = np.stack(brain_transitions, axis=0)
        centers = np.stack(brain_centers, axis=0)
        sfc = np.stack(brain_sfc, axis=0)
        occ_sfc_sorted = np.stack(brain_occ_sfc_sorted, axis=0)
        sfc_sorted = np.stack(brain_sfc_sorted, axis=0)
        metrics["brain_state_occupancy"] = occ
        metrics["brain_state_transition_matrix_mean"] = transitions.mean(axis=0)
        metrics["brain_state_transition_matrix_std"] = transitions.std(axis=0)
        metrics["brain_state_centers"] = centers
        metrics["brain_state_sfc"] = sfc
        metrics["brain_state_occupancy_sfc_sorted"] = occ_sfc_sorted
        metrics["brain_state_sfc_sorted"] = sfc_sorted

    save_npz(metrics_dir / f"{subject_id}_metrics.npz", **metrics)

    out = {
        "subject_id": subject_id,
        "cohort": meta.cohort,
        "stage": meta.stage if meta.stage is not None else "",
        "sedation": meta.sedation if meta.sedation is not None else "",
        "lzc": np.asarray(metrics["lzc"], dtype=float),
        "pci_casali_like": np.asarray(metrics["pci_casali_like"], dtype=float),
        "seeds": np.asarray(metrics["seeds"], dtype=int),
        "time_ms_example": np.asarray(metrics["time_ms_example"], dtype=float),
        "raw_example": np.asarray(metrics["raw_example"], dtype=float),
        "raw_inh_example": np.asarray(metrics["raw_inh_example"], dtype=float),
        "metrics_path": str(metrics_dir / f"{subject_id}_metrics.npz"),
        "simulation_dir": str(sim_dir),
        "mask_report": mask_report,
    }
    if cfg.compute_brain_states and "brain_state_occupancy" in metrics:
        out["brain_state_occupancy"] = np.asarray(metrics["brain_state_occupancy"], dtype=float)
        out["brain_state_transition_matrix_mean"] = np.asarray(
            metrics["brain_state_transition_matrix_mean"], dtype=float
        )
        out["brain_state_centers"] = np.asarray(metrics["brain_state_centers"], dtype=float)
        out["brain_state_sfc"] = np.asarray(metrics["brain_state_sfc"], dtype=float)
        out["brain_state_occupancy_sfc_sorted"] = np.asarray(
            metrics["brain_state_occupancy_sfc_sorted"], dtype=float
        )
        out["brain_state_sfc_sorted"] = np.asarray(metrics["brain_state_sfc_sorted"], dtype=float)
    return out



def _run_subject_job(job: tuple[str, str, BrainActSubjectConfig]) -> dict[str, Any]:
    subject_id, cohort, cfg = job
    return run_subject_simulation(subject_id=subject_id, cohort=cohort, cfg=cfg)



def run_cohort_batch(
    cohort: str,
    subjects: list[str] | None,
    cfg: BrainActSubjectConfig,
    n_jobs: int | None = None,
    use_processes: bool = True,
    show_progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run all selected subjects in one cohort and return subject-keyed results."""
    if subjects is None:
        subjects = list_subjects(dataset_root=cfg.dataset_root, cohort=cohort)
    subjects = [str(s) for s in subjects]

    if n_jobs is None:
        n_jobs = recommend_parallel_workers(task="whole_brain_tvb")
    n_jobs = max(1, int(n_jobs))

    output = OutputConfig(root=Path(cfg.output_root))
    cohort_l = str(cohort).lower()
    jobs = [(sid, cohort, cfg) for sid in subjects]
    results_by_subject: dict[str, dict[str, Any]] = {}

    def _progress(iterable, total: int, desc: str):
        if show_progress and tqdm is not None:
            return tqdm(iterable, total=total, desc=desc)
        return iterable

    t0 = perf_counter()
    total_jobs = len(jobs)
    if use_processes and n_jobs > 1:
        if show_progress:
            print(f"[run_cohort_batch] dispatching {total_jobs} subjects on {n_jobs} processes")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_run_subject_job, job) for job in jobs]
            iterator = as_completed(futures)
            iterator = _progress(iterator, total=total_jobs, desc=f"{cohort_l} subjects")
            completed = 0
            for fut in iterator:
                out = fut.result()
                results_by_subject[out["subject_id"]] = out
                completed += 1
                print(
                    "[run_cohort_batch] "
                    f"cohort={cohort_l} {completed}/{total_jobs} done: "
                    f"{out.get('subject_id', '')} "
                    f"stage={out.get('stage', '')} sedation={out.get('sedation', '')}"
                )
    else:
        if show_progress:
            print("[run_cohort_batch] running sequentially")
        completed = 0
        for job in _progress(jobs, total=total_jobs, desc=f"{cohort_l} subjects"):
            out = _run_subject_job(job)
            results_by_subject[out["subject_id"]] = out
            completed += 1
            print(
                "[run_cohort_batch] "
                f"cohort={cohort_l} {completed}/{total_jobs} done: "
                f"{out.get('subject_id', '')} "
                f"stage={out.get('stage', '')} sedation={out.get('sedation', '')}"
            )

    # Cohort-level summary saved with same metrics style as condition-batch outputs.
    ordered = sorted(results_by_subject.keys())
    lzc_mean = np.asarray([np.mean(results_by_subject[s]["lzc"]) for s in ordered], dtype=float)
    pci_mean = np.asarray([np.mean(results_by_subject[s]["pci_casali_like"]) for s in ordered], dtype=float)
    sc_zero_frac = np.asarray(
        [results_by_subject[s]["mask_report"]["sc_zero_fraction_upper"] for s in ordered],
        dtype=float,
    )
    tl_zero_frac = np.asarray(
        [results_by_subject[s]["mask_report"]["tl_zero_fraction_upper"] for s in ordered],
        dtype=float,
    )

    summary = {
        "subjects": np.asarray(ordered),
        "stages": np.asarray([results_by_subject[s].get("stage", "") for s in ordered], dtype="U32"),
        "sedation": np.asarray([results_by_subject[s].get("sedation", "") for s in ordered], dtype="U32"),
        "lzc_subject_mean": lzc_mean,
        "pci_casali_like_subject_mean": pci_mean,
        "sc_zero_fraction_upper": sc_zero_frac,
        "tl_zero_fraction_upper": tl_zero_frac,
    }
    if ordered and "brain_state_occupancy_sfc_sorted" in results_by_subject[ordered[0]]:
        occ = np.asarray(
            [np.mean(results_by_subject[s]["brain_state_occupancy_sfc_sorted"], axis=0) for s in ordered],
            dtype=float,
        )
        sfc = np.asarray(
            [np.mean(results_by_subject[s]["brain_state_sfc_sorted"], axis=0) for s in ordered],
            dtype=float,
        )
        summary["brain_state_occupancy_sfc_sorted_subject_mean"] = occ
        summary["brain_state_sfc_sorted_subject_mean"] = sfc
    save_npz(output.metrics_dir / f"{cohort_l}_cohort_metrics.npz", **summary)

    if show_progress:
        dt = perf_counter() - t0
        print(f"[run_cohort_batch] completed cohort={cohort_l} in {dt:.1f}s ({len(ordered)} subjects)")

    return results_by_subject



def run_brain_act_all_cohorts(
    cfg: BrainActSubjectConfig,
    cohorts: tuple[str, ...] = ("control", "emcs", "mcs", "uws"),
    n_jobs: int | None = None,
    use_processes: bool = True,
    show_progress: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run the subject-specific workflow for each cohort."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cohort in cohorts:
        out[str(cohort).lower()] = run_cohort_batch(
            cohort=cohort,
            subjects=None,
            cfg=cfg,
            n_jobs=n_jobs,
            use_processes=use_processes,
            show_progress=show_progress,
        )
    return out
