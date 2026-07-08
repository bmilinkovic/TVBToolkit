#!/usr/bin/env python3
"""Small subject-level PCI pilot with 5-HT2A-weighted split-leak modulation.

This script intentionally mirrors the existing condition-b PCI protocol while
running only a small subject/trial subset. Baseline PCI is read from the
existing condition-b cache when available; nonzero occupancy conditions are
simulated with the split gK/gNa Zerlaut model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(_REPO_ROOT / ".tvb-temp"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = _REPO_ROOT / "src"
_NOTEBOOKS = _REPO_ROOT / "notebooks"
for _p in (_SRC, _NOTEBOOKS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from brain_act_hybrid_common import (  # noqa: E402
    BASE_PARAMETER_MODEL_NEW,
    CONDITION_ORDER,
    COND_COLORS,
    DATASET_ROOT,
    RATE_MONITOR_PERIOD_MS_OLD,
    SCENARIOS,
    get_subject_jobs,
)
from tvbtoolkit.brian_mf.receptors import get_5ht2a_aal90  # noqa: E402
from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial  # noqa: E402
from tvbtoolkit.core.config import WholeBrainConfig  # noqa: E402
from tvbtoolkit.core.paths import doc_liege_results  # noqa: E402
from tvbtoolkit.datasets.brain_act import load_subject_structural  # noqa: E402
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: E402
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import _apply_damage_parity, worker_initializer  # noqa: E402
from tvbtoolkit.workflows.pharmacology import el_eff_from_gK_gNa, leak_to_conductances  # noqa: E402


CONDITION_B_GRADIENT = {
    "CNT": 10.0,
    "EMCS": 30.0,
    "MCS": 55.0,
    "UWS": 75.0,
    "COMA": 75.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=doc_liege_results("doc_simulation_run", "ba_sim_hybrid", "condition_b", "sims_pci"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "results" / "serotonergic_pci_pilot",
    )
    p.add_argument("--subjects-per-cohort", type=int, default=3)
    p.add_argument("--cohorts", nargs="+", default=["coma", "uws", "mcs", "emcs", "control"])
    p.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Explicit subject as cohort:subject_id. Can be passed multiple times.",
    )
    p.add_argument("--scenario", default="private_alpha0")
    p.add_argument("--trial-seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--occupancies", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.766])
    p.add_argument("--transient-ms", type=float, default=4000.0)
    p.add_argument("--t-analysis-ms", type=float, default=300.0)
    p.add_argument("--trial-sim-ms", type=float, default=8000.0)
    p.add_argument("--stim-amplitude", type=float, default=0.00030)
    p.add_argument("--stim-duration-ms", type=float, default=10.0)
    p.add_argument("--stim-region", type=int, nargs="+", default=[18])
    p.add_argument("--stim-onset-seed", type=int, default=0)
    p.add_argument("--e-l-e-drug", type=float, default=-61.2)
    p.add_argument("--e-l-i-drug", type=float, default=-64.4)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _occ_tag(occupancy: float) -> str:
    return f"occ_{int(round(float(occupancy) * 1000)):03d}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _select_subjects(dataset_root: Path, cohorts: list[str], subjects_per_cohort: int, explicit_subjects: list[str] | None = None):
    jobs = get_subject_jobs(dataset_root)
    if explicit_subjects:
        by_key = {(j.cohort, j.subject_id): j for j in jobs}
        selected = []
        for spec in explicit_subjects:
            try:
                cohort, subject_id = spec.split(":", 1)
            except ValueError as exc:
                raise ValueError(f"Explicit subject must be cohort:subject_id, got {spec!r}") from exc
            key = (cohort.strip().lower(), subject_id.strip())
            if key not in by_key:
                raise KeyError(f"Unknown subject spec {spec!r}.")
            selected.append(by_key[key])
        return selected

    out = []
    for cohort in cohorts:
        cohort_jobs = [j for j in jobs if j.cohort == cohort]
        out.extend(cohort_jobs[: max(0, int(subjects_per_cohort))])
    return out


def _stim_onsets(trial_seeds: list[int], *, transient_ms: float, t_analysis_ms: float, trial_sim_ms: float, seed: int) -> dict[int, float]:
    rng = np.random.default_rng(int(seed))
    all_onsets: dict[int, float] = {}
    max_seed = max(trial_seeds) if trial_seeds else -1
    for trial_seed in range(max_seed + 1):
        onset = int(
            rng.integers(
                int(transient_ms + t_analysis_ms),
                int(trial_sim_ms - t_analysis_ms),
            )
        )
        all_onsets[trial_seed] = float(onset)
    return {int(s): all_onsets[int(s)] for s in trial_seeds}


def _gk_profile_from_occupancy(
    *,
    occupancy: float,
    receptor_map: np.ndarray,
    e_l_start: float,
    e_l_drug: float,
    e_na: float = 50.0,
    e_k: float = -90.0,
) -> tuple[np.ndarray, float, float, float]:
    g_k_ctrl, g_na = leak_to_conductances(e_na, e_k, e_l_start, g_L=10.0)
    g_k_drug, _ = leak_to_conductances(e_na, e_k, e_l_drug, g_Na=g_na)
    rec = np.asarray(receptor_map, dtype=float).reshape(-1)
    rec_norm = (rec - float(np.min(rec))) / (float(np.max(rec)) - float(np.min(rec)) + 1e-12)
    g_k = float(g_k_ctrl) - float(occupancy) * rec_norm * (float(g_k_ctrl) - float(g_k_drug))
    eff_max_region = float(el_eff_from_gK_gNa(float(np.min(g_k)), g_na, E_K=e_k, E_Na=e_na))
    return np.asarray(g_k, dtype=float), float(g_na), float(g_k_ctrl), eff_max_region


def _build_parameter_model(condition: str, occupancy: float, receptor_map: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    base = deepcopy(BASE_PARAMETER_MODEL_NEW)
    base["b_e"] = float(CONDITION_B_GRADIENT[condition])

    if float(occupancy) <= 0.0:
        return base

    g_ke, g_na_e, _gke_ctrl, e_eff_e_max = _gk_profile_from_occupancy(
        occupancy=occupancy,
        receptor_map=receptor_map,
        e_l_start=float(base["E_L_e"]),
        e_l_drug=float(args.e_l_e_drug),
    )
    g_ki, g_na_i, _gki_ctrl, e_eff_i_max = _gk_profile_from_occupancy(
        occupancy=occupancy,
        receptor_map=receptor_map,
        e_l_start=float(base["E_L_i"]),
        e_l_drug=float(args.e_l_i_drug),
    )

    base.update(
        {
            "g_K_e": g_ke.tolist(),
            "g_Na_e": float(g_na_e),
            "g_K_i": g_ki.tolist(),
            "g_Na_i": float(g_na_i),
            # Stored for output provenance only. Model selection is controlled
            # by WholeBrainConfig.zerlaut_gk_gna.
            "serotonergic_occupancy": float(occupancy),
            "serotonergic_e_eff_e_highest_receptor": float(e_eff_e_max),
            "serotonergic_e_eff_i_highest_receptor": float(e_eff_i_max),
        }
    )
    return base


def _run_trial(
    *,
    scenario_key: str,
    scenario_cfg: dict[str, Any],
    cohort: str,
    condition: str,
    subject_id: str,
    trial_seed: int,
    occupancy: float,
    receptor_map: np.ndarray,
    output_dir: Path,
    stim_onset_ms: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    t0 = perf_counter()
    c, l, _atlas, meta = load_subject_structural(
        subject_id=subject_id,
        cohort=cohort,
        dataset_root=args.dataset_root,
        validate=True,
        enforce_symmetry=True,
        zero_diagonal=True,
        nonfinite="raise",
    )
    c, l, sc_zero_frac = _apply_damage_parity(c, l, cohort)

    parameter_model = _build_parameter_model(condition, occupancy, receptor_map, args)
    parameter_model.update(
        {
            "noise_alpha": float(scenario_cfg["noise_alpha"]),
            "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
        }
    )

    parameter_stimulus = {
        "stimtime": float(stim_onset_ms),
        "stimdur": float(args.stim_duration_ms),
        "stimperiod": float(args.trial_sim_ms) * 10.0,
        "stimval": float(args.stim_amplitude),
        "stimregion": [int(x) for x in args.stim_region],
        "stimvariables": [0],
    }

    wb_cfg = WholeBrainConfig(
        simulation_length_ms=float(args.trial_sim_ms),
        dt_ms=0.1,
        conduction_speed=4.0,
        coupling_strength=0.25,
        model_family="adex_zerlaut",
        zerlaut_matteo=False,
        zerlaut_gk_gna=bool(float(occupancy) > 0.0),
        zerlaut_order=2,
        stochastic_integrator=True,
        monitor_mode="temporal_average",
        temporal_average_period_ms=float(RATE_MONITOR_PERIOD_MS_OLD),
        monitor_variables=(0, 1),
        weights=np.asarray(c, dtype=float),
        tract_lengths=np.asarray(l, dtype=float),
        parameter_overrides={
            "parameter_model": parameter_model,
            "parameter_stimulus": parameter_stimulus,
        },
    )

    sim = run_whole_brain_simulation(wb_cfg, seed=int(trial_seed))
    t_ms = np.asarray(sim.time_ms, dtype=float)
    x = np.asarray(sim.raw, dtype=float)
    keep = t_ms >= float(args.transient_ms)
    t_post = t_ms[keep]
    x_post = x[keep]

    save_path = output_dir / f"trial_{int(trial_seed):03d}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        time_ms=t_post,
        rate=x_post,
        region_labels=np.asarray(sim.region_labels),
        stim_onset_ms=np.array([float(stim_onset_ms)]),
        t_analysis_ms=np.array([float(args.t_analysis_ms)]),
        rate_monitor_period_ms=np.array([float(RATE_MONITOR_PERIOD_MS_OLD)]),
        trial_seed=np.array([int(trial_seed)]),
        noise_alpha=np.array([float(scenario_cfg["noise_alpha"])]),
        stim_amplitude=np.array([float(args.stim_amplitude)]),
        stim_duration_ms=np.array([float(args.stim_duration_ms)]),
        stim_region=np.array(args.stim_region, dtype=int),
        occupancy=np.array([float(occupancy)]),
        sc_zero_fraction_upper=np.array([float(sc_zero_frac)]),
    )

    return {
        "cohort": cohort,
        "condition": condition,
        "subject_id": subject_id,
        "stage": str(getattr(meta, "stage", "") or ""),
        "sedation": str(getattr(meta, "sedation", "") or ""),
        "scenario": scenario_key,
        "occupancy": float(occupancy),
        "trial_seed": int(trial_seed),
        "runtime_s": float(perf_counter() - t0),
        "save_path": str(save_path),
    }


def _run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    return _run_trial(**job)


def _load_trials(paths: list[Path]) -> tuple[list[np.ndarray], int, float, float]:
    trials = []
    onset_bins = []
    dt_vals = []
    t_analysis_vals = []
    for p in paths:
        d = np.load(p)
        t = np.asarray(d["time_ms"], dtype=float)
        x = np.asarray(d["rate"], dtype=float)
        dt = float(np.median(np.diff(t)))
        onset_abs = float(np.ravel(d["stim_onset_ms"])[0])
        onset = int(np.argmin(np.abs(t - onset_abs)))
        trials.append(x)
        onset_bins.append(onset)
        dt_vals.append(dt)
        t_analysis_vals.append(float(np.ravel(d["t_analysis_ms"])[0]))

    if not trials:
        raise ValueError("No trial files provided.")
    onset_ref = int(round(float(np.median(onset_bins))))
    dt_ref = float(np.median(dt_vals))
    t_analysis_ref = float(np.median(t_analysis_vals))
    return trials, onset_ref, dt_ref, t_analysis_ref


def _compute_pci_for_condition(paths: list[Path]) -> tuple[float, np.ndarray]:
    trials, onset, dt_ms, t_analysis_ms = _load_trials(paths)
    return pci_casali_like_multi_trial(
        trials,
        stimulation_index=onset,
        t_analysis_ms=t_analysis_ms,
        dt_ms=dt_ms,
    )


def _condition_paths(root: Path, occ: float, scenario: str, cohort: str, subject_id: str, trial_seeds: list[int]) -> list[Path]:
    if float(occ) <= 0.0:
        base = root / "condb_doc_gradient" / scenario / cohort / subject_id
    else:
        base = root / "sims_pci" / _occ_tag(occ) / scenario / cohort / subject_id
    return [base / f"trial_{int(seed):03d}.npz" for seed in trial_seeds]


def _plot(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"

    subj = metrics.copy()
    subj["condition"] = pd.Categorical(subj["condition"], categories=CONDITION_ORDER, ordered=True)
    subj = subj.sort_values(["condition", "subject_id", "occupancy"])

    baseline = subj.loc[subj["occupancy"] == 0.0, ["cohort", "subject_id", "pci_mean"]].rename(
        columns={"pci_mean": "pci_baseline"}
    )
    subj = subj.merge(baseline, on=["cohort", "subject_id"], how="left")
    subj["pci_rescue"] = subj["pci_mean"] - subj["pci_baseline"]
    subj.to_csv(tables_dir / "serotonergic_pci_subject_metrics_with_rescue.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax = axes[0]
    for (condition, subject_id), g in subj.groupby(["condition", "subject_id"], observed=True):
        color = COND_COLORS.get(str(condition), "#555555")
        ax.plot(g["occupancy"], g["pci_mean"], marker="o", linewidth=1.8, color=color, alpha=0.9)
        ax.text(float(g["occupancy"].max()) + 0.01, float(g["pci_mean"].iloc[-1]), f"{condition}:{subject_id}", fontsize=7, color=color)
    ax.set_xlabel("5-HT2A occupancy")
    ax.set_ylabel("PCI")
    ax.set_title("Subject PCI")
    ax.grid(alpha=0.25)

    ax = axes[1]
    nz = subj[subj["occupancy"] > 0.0]
    for condition, g in nz.groupby("condition", observed=True):
        color = COND_COLORS.get(str(condition), "#555555")
        ax.scatter(g["occupancy"], g["pci_rescue"], label=str(condition), s=42, color=color, alpha=0.9)
        mean = g.groupby("occupancy", as_index=False)["pci_rescue"].mean()
        ax.plot(mean["occupancy"], mean["pci_rescue"], color=color, linewidth=2.0)
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xlabel("5-HT2A occupancy")
    ax.set_ylabel("PCI rescue vs baseline")
    ax.set_title("PCI Rescue")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"serotonergic_pci_pilot.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    if args.scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {args.scenario!r}.")
    scenario_cfg = SCENARIOS[args.scenario]
    subjects = _select_subjects(args.dataset_root, args.cohorts, args.subjects_per_cohort, args.subject)
    receptor_map = get_5ht2a_aal90()
    stim_onsets = _stim_onsets(
        [int(s) for s in args.trial_seeds],
        transient_ms=float(args.transient_ms),
        t_analysis_ms=float(args.t_analysis_ms),
        trial_sim_ms=float(args.trial_sim_ms),
        seed=int(args.stim_onset_seed),
    )

    manifest = {
        "script": "scripts/run_serotonergic_pci_pilot.py",
        "dataset_root": str(args.dataset_root),
        "baseline_root": str(args.baseline_root),
        "output_root": str(args.output_root),
        "scenario": args.scenario,
        "scenario_cfg": scenario_cfg,
        "subjects": [s.__dict__ for s in subjects],
        "trial_seeds": [int(s) for s in args.trial_seeds],
        "stim_onsets_ms_by_trial_seed": {str(k): float(v) for k, v in stim_onsets.items()},
        "occupancies": [float(o) for o in args.occupancies],
        "e_l_e_drug": float(args.e_l_e_drug),
        "e_l_i_drug": float(args.e_l_i_drug),
        "workers": int(args.workers),
    }
    _write_json(args.output_root / "logs" / "run_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    trial_jobs: list[dict[str, Any]] = []
    for occ in [float(o) for o in args.occupancies if float(o) > 0.0]:
        for sj in subjects:
            out_dir = args.output_root / "sims_pci" / _occ_tag(occ) / args.scenario / sj.cohort / sj.subject_id
            for trial_seed in [int(s) for s in args.trial_seeds]:
                save_path = out_dir / f"trial_{trial_seed:03d}.npz"
                if save_path.exists() and not args.overwrite:
                    continue
                trial_jobs.append(
                    {
                        "scenario_key": args.scenario,
                        "scenario_cfg": scenario_cfg,
                        "cohort": sj.cohort,
                        "condition": sj.condition,
                        "subject_id": sj.subject_id,
                        "trial_seed": trial_seed,
                        "occupancy": occ,
                        "receptor_map": receptor_map,
                        "output_dir": out_dir,
                        "stim_onset_ms": float(stim_onsets[trial_seed]),
                        "args": args,
                    }
                )

    completed_rows: list[dict[str, Any]] = []
    print(f"[sero-pci] queued {len(trial_jobs)} serotonergic trial simulations on {int(args.workers)} workers", flush=True)
    if trial_jobs:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=worker_initializer) as ex:
            futures = [ex.submit(_run_trial_job, job) for job in trial_jobs]
            total = len(futures)
            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                completed_rows.append(row)
                print(
                    "[sero-pci] "
                    f"{i}/{total} done occ={row['occupancy']:.3f} "
                    f"{row['condition']}/{row['subject_id']} trial={row['trial_seed']} "
                    f"runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    _write_csv(args.output_root / "logs" / "completed_trials.csv", completed_rows)

    metric_rows: list[dict[str, Any]] = []
    for sj in subjects:
        for occ in [float(o) for o in args.occupancies]:
            root = args.baseline_root if occ <= 0.0 else args.output_root
            paths = _condition_paths(root, occ, args.scenario, sj.cohort, sj.subject_id, [int(s) for s in args.trial_seeds])
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing trial files for {sj.subject_id} occ={occ}: {missing[:3]}")
            pci_mean, pci_per_trial = _compute_pci_for_condition(paths)
            metric_rows.append(
                {
                    "cohort": sj.cohort,
                    "condition": sj.condition,
                    "subject_id": sj.subject_id,
                    "scenario": args.scenario,
                    "occupancy": float(occ),
                    "n_trials": int(len(paths)),
                    "pci_mean": float(pci_mean),
                    "pci_std": float(np.std(pci_per_trial)),
                    "pci_per_trial": json.dumps([float(x) for x in pci_per_trial]),
                    "trial_paths": json.dumps([str(p) for p in paths]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_root / "tables" / "serotonergic_pci_subject_metrics.csv", index=False)
    _plot(metrics, args.output_root)
    print(f"[sero-pci] wrote {args.output_root}")


if __name__ == "__main__":
    main()
