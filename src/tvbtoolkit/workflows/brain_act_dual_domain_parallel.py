"""Process-safe helpers for dual-domain Brain-Act jobs (rates + TVB Bold)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from time import perf_counter

import numpy as np

from tvbtoolkit.analysis.brain_states import summarize_brain_states
from tvbtoolkit.complexity.measures import lzc_multichannel, pci_casali_like
from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.datasets.brain_act import load_subject_structural


def worker_initializer() -> None:
    """Prepare spawned worker processes for TVB imports.

    Handles two different llvmlite generations:

    * llvmlite <= 0.41 (paired with numba 0.58): ``initialize()``,
      ``initialize_native_target()``, and ``initialize_native_asmprinter()``
      must be called explicitly — LLVM does not auto-initialise on import.

    * llvmlite >= 0.42 (paired with numba 0.59+): ``initialize()`` is
      deprecated and raises ``RuntimeError``.  LLVM target registration now
      happens automatically on import, so we only no-op the deprecated call so
      that TVB's internal invocation does not crash.  Target-registration
      functions (``initialize_native_target`` etc.) must *not* be patched away
      in this case — doing so prevents targets from being registered and causes
      "no targets are registered" crashes.

    This function must live at **module level** so that ``ProcessPoolExecutor``
    with the ``spawn`` start method can pickle and transmit it to workers.
    Pass it as ``initializer=worker_initializer`` when constructing the executor.
    """
    try:
        import llvmlite.binding as llvmlib  # noqa: PLC0415
        try:
            llvmlib.initialize()
            # Succeeded → old llvmlite (<= 0.41): also register native target.
            for _fn in ("initialize_native_target", "initialize_native_asmprinter"):
                if hasattr(llvmlib, _fn):
                    getattr(llvmlib, _fn)()
        except RuntimeError:
            # Deprecated in >= 0.42; patch so TVB's internal call is silenced.
            # Do NOT patch target-registration functions — they still work and
            # are required (auto-registration only covers the core init step).
            setattr(llvmlib, "initialize", lambda *_a, **_kw: None)
    except Exception:
        pass  # llvmlite not installed — nothing to patch


def _upper_triangle_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    iu = np.triu_indices_from(arr, k=1)
    return np.asarray(arr[iu], dtype=float)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=float).reshape(-1)
    xb = np.asarray(b, dtype=float).reshape(-1)
    if xa.size != xb.size or xa.size < 3:
        return float("nan")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(xb)):
        return float("nan")
    if np.std(xa) <= 0.0 or np.std(xb) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def _apply_damage_parity(c: np.ndarray, l: np.ndarray, cohort: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply damage masking then max-normalise the SC weights.

    Order matters:
      1. Zero diagonal.
      2. For patient cohorts: zero tract lengths where SC weight is 0
         (damaged/absent connections cannot carry signal).
      3. Max-normalise SC weights over the surviving (non-zero) edges so that
         weights are in [0, 1] regardless of raw fibre-count scale.
         Normalisation is applied after masking so the reference maximum is
         taken from surviving connections only.
    """
    c = np.asarray(c, dtype=float).copy()
    l = np.asarray(l, dtype=float).copy()
    np.fill_diagonal(c, 0.0)
    np.fill_diagonal(l, 0.0)

    # ── Damage parity (patient cohorts only) ─────────────────────────────────
    if cohort.lower() in {"mcs", "uws", "emcs", "coma"}:
        mismatch = (c == 0.0) & (l != 0.0)
        if np.any(mismatch):
            l[mismatch] = 0.0

    # ── Max-normalise over surviving edges ────────────────────────────────────
    c_max = float(np.max(c))
    if c_max > 0.0:
        c = c / c_max

    iu = np.triu_indices_from(c, k=1)
    return c, l, float(np.mean(c[iu] == 0.0))


def _extract_rate_and_bold(full_output: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parsed: list[tuple[float, np.ndarray, np.ndarray]] = []
    for t, d in full_output:
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        d_arr = np.asarray(d)
        if d_arr.ndim == 4:
            x = np.asarray(d_arr[:, 0, :, 0], dtype=float)
        elif d_arr.ndim == 2:
            x = np.asarray(d_arr, dtype=float)
        else:
            x = np.asarray(d_arr).reshape(d_arr.shape[0], -1)
        dt = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else np.inf
        parsed.append((dt, t_arr, x))

    parsed.sort(key=lambda z: z[0])
    if len(parsed) < 2:
        raise ValueError("Expected at least two monitor outputs (rates + BOLD).")
    return parsed[0][1], parsed[0][2], parsed[-1][1], parsed[-1][2]


def _compute_pci(x: np.ndarray, t_ms: np.ndarray, post_window_ms: float) -> float:
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t_ms, dtype=float)
    if x_arr.shape[0] < 8:
        return float("nan")

    dt_ms = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 1.0
    stim_idx = x_arr.shape[0] // 2
    max_half = min(stim_idx, x_arr.shape[0] - stim_idx)
    if max_half < 2:
        return float("nan")

    target_bins = max(2, int(round(post_window_ms / max(dt_ms, 1e-9))))
    window_bins = min(max_half, target_bins)
    t_analysis_ms = float(window_bins * dt_ms)
    return float(
        pci_casali_like(
            x_arr,
            stimulation_index=stim_idx,
            t_analysis_ms=t_analysis_ms,
            dt_ms=dt_ms,
        )
    )


def _compute_domain_metrics(
    x: np.ndarray,
    t_ms: np.ndarray,
    sc: np.ndarray,
    *,
    n_states: int,
    pci_window_ms: float,
    compute_pci: bool = True,
    brain_state_pipeline: str = "standard",
    brain_state_trim_edge_samples: int = 9,
    brain_state_tr_seconds: float = 2.4,
    brain_state_bandpass_hz: tuple[float, float] = (0.01, 0.20),
    brain_state_n_init: int = 10,
) -> dict[str, Any]:
    lzc = float(lzc_multichannel(x))
    pci = _compute_pci(x, t_ms, pci_window_ms) if compute_pci else float("nan")

    bs = summarize_brain_states(
        np.asarray(x, dtype=float),
        n_states=n_states,
        random_seed=0,
        n_init=int(brain_state_n_init),
        trim_edge_samples=int(brain_state_trim_edge_samples),
        pipeline=str(brain_state_pipeline),
        tr_seconds=float(brain_state_tr_seconds),
        bandpass_hz=(float(brain_state_bandpass_hz[0]), float(brain_state_bandpass_hz[1])),
    )
    occ = np.asarray(bs.occupancy, dtype=float)
    centers = np.asarray(bs.centers, dtype=float)

    sc_vec = _upper_triangle_vector(sc)
    sfc = np.asarray([_safe_pearson(row, sc_vec) for row in centers], dtype=float)
    order = np.argsort(np.nan_to_num(sfc, nan=np.inf))

    return {
        "lzc": lzc,
        "pci": pci,
        "occupancy_sfc_sorted": occ[order],
        "sfc_sorted": sfc[order],
    }


def _sedation_group(sedation: str) -> str:
    sed_l = sedation.strip().lower()
    if sed_l in {"", "nan", "none", "unknown", "na", "n/a"}:
        return "unknown"
    if any(tok in sed_l for tok in ["no sedation", "non sedated", "non-sedated", "unsedated", "awake"]):
        return "non_sedated"
    return "sedated"


def run_dual_domain_job(
    *,
    scenario_key: str,
    scenario_label: str,
    noise_alpha: float,
    shared_noise_mode: str,
    cohort: str,
    subject_id: str,
    seed: int,
    dataset_root: str,
    sim_dir_root: str,
    simulation_length_ms: float,
    rate_monitor_period_ms: float,
    bold_period_ms: float,
    transient_ms: float,
    n_states: int,
    pci_window_rate_ms: float,
    pci_window_bold_ms: float,
    base_parameter_model: dict[str, Any],
    enable_bold: bool = True,
) -> dict[str, Any]:
    """Run one subject/scenario/seed job and return metric/state rows.

    This function is defined in a module (not notebook scope) so it can be used
    safely by `ProcessPoolExecutor` workers.
    """
    t0 = perf_counter()
    c, l, _atlas, meta = load_subject_structural(
        subject_id=subject_id,
        cohort=cohort,
        dataset_root=dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    c, l, sc_zero_frac = _apply_damage_parity(c, l, cohort)

    stage = str(getattr(meta, "stage", "") or "")
    sedation = str(getattr(meta, "sedation", "") or "")
    sed_group = _sedation_group(sedation)

    parameter_model = deepcopy(base_parameter_model)
    parameter_model.update(
        {
            "noise_alpha": float(noise_alpha),
            "shared_noise_mode": str(shared_noise_mode),
        }
    )

    overrides = {"parameter_model": parameter_model}
    if enable_bold:
        overrides["parameter_monitor"] = {
            "Bold": True,
            "parameter_Bold": {
                "variables_of_interest": [0],
                "period": float(bold_period_ms),
            },
        }

    wb_cfg = WholeBrainConfig(
        simulation_length_ms=float(simulation_length_ms),
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=0.25,
        model_family="adex_zerlaut",
        zerlaut_matteo=False,
        zerlaut_gk_gna=False,
        zerlaut_order=1,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(rate_monitor_period_ms),
        monitor_variables=(0, 1),
        weights=np.asarray(c, dtype=float),
        tract_lengths=np.asarray(l, dtype=float),
        parameter_overrides=overrides,
    )

    # Local import avoids TVB simulator initialization for analysis-only workflows.
    from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation

    sim = run_whole_brain_simulation(wb_cfg, seed=int(seed))

    if enable_bold:
        t_rate_ms, x_rate, t_bold_ms, x_bold = _extract_rate_and_bold(sim.full_monitor_output)
    else:
        t_rate_ms = np.asarray(sim.time_ms, dtype=float)
        x_rate = np.asarray(sim.raw, dtype=float)
        t_bold_ms = np.array([], dtype=float)
        x_bold = np.empty((0, x_rate.shape[1]), dtype=float)

    keep_rate = t_rate_ms >= float(transient_ms)
    t_rate_ms, x_rate = t_rate_ms[keep_rate], x_rate[keep_rate]
    if enable_bold and t_bold_ms.size:
        keep_bold = t_bold_ms >= float(transient_ms)
        t_bold_ms, x_bold = t_bold_ms[keep_bold], x_bold[keep_bold]

    save_dir = Path(sim_dir_root) / scenario_key / cohort / subject_id
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_dir / f"seed_{int(seed):03d}.npz",
        time_rate_ms=t_rate_ms,
        rate=x_rate,
        time_bold_ms=t_bold_ms,
        bold=x_bold,
        region_labels=np.asarray(sim.region_labels),
    )

    rate_dt_s = float(np.median(np.diff(t_rate_ms))) / 1000.0 if t_rate_ms.size > 1 else 0.25
    met_rate = _compute_domain_metrics(
        x_rate,
        t_rate_ms,
        c,
        n_states=n_states,
        pci_window_ms=pci_window_rate_ms,
        compute_pci=True,
        brain_state_pipeline="standard",
        brain_state_trim_edge_samples=9,
        brain_state_tr_seconds=max(rate_dt_s, 1e-6),
        brain_state_bandpass_hz=(0.01, 0.20),
        brain_state_n_init=10,
    )

    met_bold = None
    if enable_bold and x_bold.shape[0] >= max(10, n_states):
        bold_dt_s = float(np.median(np.diff(t_bold_ms))) / 1000.0 if t_bold_ms.size > 1 else float(bold_period_ms) / 1000.0
        met_bold = _compute_domain_metrics(
            x_bold,
            t_bold_ms,
            c,
            n_states=n_states,
            pci_window_ms=pci_window_bold_ms,
            compute_pci=False,
            brain_state_pipeline="brain_act_legacy",
            brain_state_trim_edge_samples=0,
            brain_state_tr_seconds=max(bold_dt_s, 1e-6),
            brain_state_bandpass_hz=(0.01, 0.20),
            brain_state_n_init=20,
        )

    metric_row = {
        "scenario": scenario_key,
        "scenario_label": scenario_label,
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

    state_rows = []
    n_state = len(met_rate["occupancy_sfc_sorted"])
    for j in range(n_state):
        sfc_b = float(met_bold["sfc_sorted"][j]) if met_bold is not None and j < len(met_bold["sfc_sorted"]) else float("nan")
        occ_b = float(met_bold["occupancy_sfc_sorted"][j]) if met_bold is not None and j < len(met_bold["occupancy_sfc_sorted"]) else float("nan")
        state_rows.append(
            {
                "scenario": scenario_key,
                "scenario_label": scenario_label,
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

    runtime_s = float(perf_counter() - t0)
    return {
        "metric_row": metric_row,
        "state_rows": state_rows,
        "runtime_s": runtime_s,
        "scenario": scenario_key,
        "cohort": cohort,
        "subject_id": subject_id,
        "seed": int(seed),
    }


def run_simulation_only_job(
    *,
    scenario_key: str,
    noise_alpha: float,
    shared_noise_mode: str,
    cohort: str,
    subject_id: str,
    seed: int,
    dataset_root: str,
    output_dir: str,
    simulation_length_ms: float,
    rate_monitor_period_ms: float,
    transient_ms: float,
    base_parameter_model: dict[str, Any],
    enable_bold: bool = False,
    bold_period_ms: float = 2400.0,
) -> dict[str, Any]:
    """Run one spontaneous whole-brain simulation and save NPZ.

    This is the simulation-only counterpart of ``run_dual_domain_job``.  No
    complexity metrics or brain-state analyses are computed here; all downstream
    analyses are performed by the companion analysis notebook
    (``brain_act_full_analysis_v2.ipynb``).

    Parameters
    ----------
    scenario_key : str
        Noise-scenario identifier (e.g. ``"private_alpha0"``).
    noise_alpha : float
        Correlated-noise mixing coefficient α ∈ [0, 1].
    shared_noise_mode : str
        ``"none"`` (private only), ``"global"`` (one shared process), or
        ``"connectivity"`` (SC-shaped mixing).
    cohort : str
        Dataset cohort (e.g. ``"control"``, ``"uws"``, ``"mcs"``).
    subject_id : str
        Subject identifier string.
    seed : int
        Random seed for the stochastic integrator.
    dataset_root : str
        Path to the converted structural dataset root.
    output_dir : str
        Directory where the NPZ file will be written as ``seed_NNN.npz``.
    simulation_length_ms : float
        Total simulation length in milliseconds.
    rate_monitor_period_ms : float
        Temporal-average monitor period (milliseconds).  Determines the
        effective sampling rate of the firing-rate output.
    transient_ms : float
        Initial transient to discard before saving (milliseconds).
    base_parameter_model : dict
        AdEx Zerlaut parameter overrides (T, P_e, P_i, etc.).
    enable_bold : bool
        Whether to also record a TVB Bold monitor output.  Default ``False``
        for PCI-trial runs; set ``True`` for spontaneous runs when BOLD is
        needed for downstream analysis.
    bold_period_ms : float
        BOLD monitor period (milliseconds).  Only used when ``enable_bold``
        is ``True``.

    Returns
    -------
    dict
        Metadata row: scenario, cohort, subject_id, seed, n_rate_samples,
        runtime_s, save_path, etc.
    """
    t0 = perf_counter()

    c, l, _atlas, meta = load_subject_structural(
        subject_id=subject_id,
        cohort=cohort,
        dataset_root=dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    c, l, sc_zero_frac = _apply_damage_parity(c, l, cohort)

    stage = str(getattr(meta, "stage", "") or "")
    sedation = str(getattr(meta, "sedation", "") or "")

    parameter_model = deepcopy(base_parameter_model)
    parameter_model.update(
        {"noise_alpha": float(noise_alpha), "shared_noise_mode": str(shared_noise_mode)}
    )

    overrides: dict[str, Any] = {"parameter_model": parameter_model}
    if enable_bold:
        overrides["parameter_monitor"] = {
            "Bold": True,
            "parameter_Bold": {"variables_of_interest": [0], "period": float(bold_period_ms)},
        }

    wb_cfg = WholeBrainConfig(
        simulation_length_ms=float(simulation_length_ms),
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=0.25,
        model_family="adex_zerlaut",
        zerlaut_matteo=False,
        zerlaut_gk_gna=False,
        zerlaut_order=1,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(rate_monitor_period_ms),
        monitor_variables=(0, 1),
        weights=np.asarray(c, dtype=float),
        tract_lengths=np.asarray(l, dtype=float),
        parameter_overrides=overrides,
    )

    from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation

    sim = run_whole_brain_simulation(wb_cfg, seed=int(seed))

    if enable_bold:
        t_rate_ms, x_rate, t_bold_ms, x_bold = _extract_rate_and_bold(sim.full_monitor_output)
    else:
        t_rate_ms = np.asarray(sim.time_ms, dtype=float)
        x_rate = np.asarray(sim.raw, dtype=float)
        t_bold_ms = np.array([], dtype=float)
        x_bold = np.empty((0, x_rate.shape[1] if x_rate.ndim > 1 else 0), dtype=float)

    # Strip transient from rate output.
    keep_rate = t_rate_ms >= float(transient_ms)
    t_rate_post = t_rate_ms[keep_rate]
    x_rate_post = x_rate[keep_rate]

    save_path = Path(output_dir) / f"seed_{int(seed):03d}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = dict(
        time_rate_ms=t_rate_post,
        rate=x_rate_post,
        region_labels=np.asarray(sim.region_labels),
        simulation_length_ms=np.array([float(simulation_length_ms)]),
        transient_ms=np.array([float(transient_ms)]),
        rate_monitor_period_ms=np.array([float(rate_monitor_period_ms)]),
        noise_alpha=np.array([float(noise_alpha)]),
        seed=np.array([int(seed)]),
    )
    if enable_bold and t_bold_ms.size:
        keep_bold = t_bold_ms >= float(transient_ms)
        save_dict["time_bold_ms"] = t_bold_ms[keep_bold]
        save_dict["bold"] = x_bold[keep_bold]

    np.savez_compressed(save_path, **save_dict)

    runtime_s = float(perf_counter() - t0)
    return {
        "scenario": scenario_key,
        "cohort": cohort,
        "subject_id": subject_id,
        "seed": int(seed),
        "stage": stage,
        "sedation": sedation,
        "sc_zero_fraction": float(sc_zero_frac),
        "n_rate_samples": int(t_rate_post.shape[0]),
        "n_bold_samples": int(save_dict.get("bold", np.array([])).shape[0]),
        "runtime_s": runtime_s,
        "save_path": str(save_path),
    }


def run_pci_trial_job(
    *,
    scenario_key: str,
    noise_alpha: float,
    shared_noise_mode: str,
    cohort: str,
    subject_id: str,
    trial_seed: int,
    dataset_root: str,
    output_dir: str,
    transient_ms: float,
    t_analysis_ms: float,
    rate_monitor_period_ms: float,
    base_parameter_model: dict[str, Any],
    stim_amplitude: float = 0.0005,
    stim_duration_ms: float = 50.0,
    stim_region: "list[int] | None" = None,
    stim_onset_ms: "float | None" = None,
    total_sim_ms: "float | None" = None,
) -> dict[str, Any]:
    """Run one short PCI-trial simulation with a genuine TMS-like pulse.

    Replicates TVBSim's ``parallelized_PCI`` / ``_calculate_PCI_seed_subset``
    approach: each trial is an independent whole-brain simulation in which a
    brief external-input pulse is injected at a fixed onset time via TVB's
    ``StimuliRegion`` + ``PulseTrain`` API (``parameter_stimulus``).

    Simulation design
    -----------------
    ::

        |─── transient_ms ─────────────────|── pre (T_analysis) ──|── post (T_analysis) ──|
        0                              stim_onset_ms − T_analysis  stim_onset_ms  sim_end
                                                                       ↑
                                                              TMS pulse injected here
                                                              (duration: stim_duration_ms)

    - ``simulation_length_ms = transient_ms + 2 × t_analysis_ms``
    - ``stim_onset_ms`` defaults to ``transient_ms + t_analysis_ms`` (centre of
      the analysis window) but can be randomised per trial (Hugo/Maria convention).

    The pre-stimulus window provides the spontaneous-activity baseline for
    ``binarise_signals``; the post-stimulus window captures the genuine
    perturbational response.

    Stimulus specification
    ----------------------
    The pulse is modelled as a rectangular ``PulseTrain`` with period ≫ simulation
    length (effectively a single shot).  It is applied to ``stim_region`` nodes
    on model variable 0 (excitatory firing rate E).

    Parameters
    ----------
    scenario_key, noise_alpha, shared_noise_mode, cohort, subject_id : str/float
        Condition identifiers (see ``run_simulation_only_job``).
    trial_seed : int
        Random seed for this trial.  Trials differ only in their noise
        realisation.
    dataset_root : str
        Path to the converted structural dataset root.
    output_dir : str
        Directory where the NPZ file is written as ``trial_NNN.npz``.
    transient_ms : float
        Initial transient to discard (milliseconds).
    t_analysis_ms : float
        One-sided PCI analysis window (milliseconds).  Matches TVBSim default
        ``t_analysis = 300``.
    rate_monitor_period_ms : float
        Temporal-average monitor period (milliseconds).
    base_parameter_model : dict
        AdEx Zerlaut parameter overrides (T, P_e, P_i, etc.).
    stim_amplitude : float
        Stimulus amplitude injected into variable E (kHz).
        Hugo/Maria convention: 0.0005 kHz (0.5 Hz).
    stim_duration_ms : float
        Duration of the rectangular pulse (milliseconds).
        Hugo/Maria convention: 50 ms.
    stim_region : list[int] or None
        Region indices (0-based) to stimulate.  ``None`` defaults to ``[0]``
        (first node in the SC matrix).  Mirror TVBSim's single-seed approach:
        the perturbation is local; complexity is measured globally.
    stim_onset_ms : float or None
        Absolute time (ms from t=0) at which the pulse fires.  When ``None``
        (default) falls back to ``transient_ms + t_analysis_ms``.  Pass a
        per-seed randomised value to implement the Hugo/Maria convention.
    total_sim_ms : float or None
        Total simulation length (ms).  When ``None`` (default) falls back to
        ``transient_ms + 2 × t_analysis_ms``.  Must be set to a value that
        accommodates the full post-stim window: at least
        ``stim_onset_ms + t_analysis_ms``.  Hugo/Maria convention: 8000 ms.

    Returns
    -------
    dict
        Metadata: scenario, cohort, subject_id, trial_seed, stim_onset_ms,
        stim_amplitude, stim_region, n_samples, runtime_s, save_path.
    """
    t0 = perf_counter()

    # Stimulus onset: use per-trial randomised value when provided; otherwise
    # default to the centre of the analysis window (transient + t_analysis).
    if stim_onset_ms is None:
        stim_onset_ms = float(transient_ms) + float(t_analysis_ms)
    else:
        stim_onset_ms = float(stim_onset_ms)

    # Total simulation length: use explicit value when provided (Hugo/Maria: 8000 ms).
    # Falls back to transient + 2×t_analysis, but enforces that the full post-stim
    # window (stim_onset + t_analysis) fits inside the simulation.
    if total_sim_ms is not None:
        simulation_length_ms = float(total_sim_ms)
    else:
        simulation_length_ms = float(transient_ms) + 2.0 * float(t_analysis_ms)
    # Safety guard: ensure post-stim window fits
    min_required = stim_onset_ms + float(t_analysis_ms)
    if simulation_length_ms < min_required:
        simulation_length_ms = min_required

    if stim_region is None:
        stim_region = [0]

    c, l, _atlas, _meta = load_subject_structural(
        subject_id=subject_id,
        cohort=cohort,
        dataset_root=dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    c, l, _sc_zero_frac = _apply_damage_parity(c, l, cohort)

    parameter_model = deepcopy(base_parameter_model)
    parameter_model.update(
        {"noise_alpha": float(noise_alpha), "shared_noise_mode": str(shared_noise_mode)}
    )

    # ── Genuine TMS-like stimulus ─────────────────────────────────────────────
    # Passed to _build_stimulation() inside run_whole_brain_simulation via
    # _apply_parameter_overrides(parameters, cfg.parameter_overrides).
    # stimperiod >> simulation_length ensures a single pulse (no repetition).
    parameter_stimulus = {
        "stimtime":   float(stim_onset_ms),
        "stimdur":    float(stim_duration_ms),
        "stimperiod": float(simulation_length_ms) * 10.0,   # >> sim length → single shot
        "stimval":    float(stim_amplitude),
        "stimregion": list(stim_region),
        "stimvariables": [0],   # variable 0 = excitatory firing rate E
    }

    wb_cfg = WholeBrainConfig(
        simulation_length_ms=float(simulation_length_ms),
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=0.25,
        model_family="adex_zerlaut",
        zerlaut_matteo=False,
        zerlaut_gk_gna=False,
        zerlaut_order=1,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(rate_monitor_period_ms),
        monitor_variables=(0, 1),
        weights=np.asarray(c, dtype=float),
        tract_lengths=np.asarray(l, dtype=float),
        parameter_overrides={
            "parameter_model":    parameter_model,
            "parameter_stimulus": parameter_stimulus,
        },
    )

    from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: PLC0415

    sim = run_whole_brain_simulation(wb_cfg, seed=int(trial_seed))

    t_ms = np.asarray(sim.time_ms, dtype=float)
    x    = np.asarray(sim.raw,     dtype=float)

    # Keep the full post-transient window (pre-stim baseline + post-stim response).
    keep   = t_ms >= float(transient_ms)
    t_post = t_ms[keep]
    x_post = x[keep]

    save_path = Path(output_dir) / f"trial_{int(trial_seed):03d}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        time_ms                = t_post,
        rate                   = x_post,
        region_labels          = np.asarray(sim.region_labels),
        stim_onset_ms          = np.array([float(stim_onset_ms)]),
        t_analysis_ms          = np.array([float(t_analysis_ms)]),
        rate_monitor_period_ms = np.array([float(rate_monitor_period_ms)]),
        trial_seed             = np.array([int(trial_seed)]),
        noise_alpha            = np.array([float(noise_alpha)]),
        stim_amplitude         = np.array([float(stim_amplitude)]),
        stim_duration_ms       = np.array([float(stim_duration_ms)]),
        stim_region            = np.array(stim_region, dtype=int),
    )

    runtime_s = float(perf_counter() - t0)
    return {
        "scenario":       scenario_key,
        "cohort":         cohort,
        "subject_id":     subject_id,
        "trial_seed":     int(trial_seed),
        "stim_onset_ms":  float(stim_onset_ms),
        "stim_amplitude": float(stim_amplitude),
        "stim_region":    list(stim_region),
        "n_samples":      int(t_post.shape[0]),
        "runtime_s":      runtime_s,
        "save_path":      str(save_path),
    }


__all__ = [
    "worker_initializer",
    "run_dual_domain_job",
    "run_simulation_only_job",
    "run_pci_trial_job",
]
