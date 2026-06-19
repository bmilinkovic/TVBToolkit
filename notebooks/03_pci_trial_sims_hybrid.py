#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from brain_act_hybrid_common import (
    BASE_PARAMETER_MODEL_NEW,
    DATASET_ROOT,
    PROJECT_ROOT,
    RATE_MONITOR_PERIOD_MS_OLD,
    SCENARIOS,
    get_subject_jobs,
    save_json,
)

from tvbtoolkit.workflows.brain_act_dual_domain_parallel import (
    run_pci_trial_job,
    worker_initializer,
)


SHARED_B_PCI_SCENARIO_KEYS = (
    "private_alpha0",   # alpha = 0 baseline; no duplicated global/sc alpha=0
    "global_alpha_025",
    "global_alpha_045",
    "sc_alpha_025",
    "sc_alpha_045",
)
CONDITION_B_PCI_SCENARIO_KEYS = (
    "private_alpha0",
    "global_alpha_025",
    "sc_alpha_045",
)
DEFAULT_B_VALUES = (5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
CONDITION_B_GRADIENTS: dict[str, dict[str, float]] = {
    "doc_gradient": {
        "CNT": 10.0,
        "EMCS": 30.0,
        "MCS": 55.0,
        "UWS": 75.0,
        "COMA": 75.0,
    },
}


def _b_tag(b_val: float) -> str:
    v = float(b_val)
    if abs(v - round(v)) < 1e-12:
        return f"b{int(round(v)):03d}"
    return f"b{str(v).replace('.', 'p')}"


def _condition_b_tag(name: str) -> str:
    return f"condb_{name}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run perturbational PCI trial simulations for the reduced HPC PCI plan. "
            "Outputs are saved separately for shared-b and condition-specific-b branches."
        )
    )
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "notebooks" / "outputs" / "ba_sim_hybrid")
    p.add_argument("--workers", type=int, default=max(1, int(round((os.cpu_count() or 8) * 0.8))))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Build manifests and report queued PCI trial jobs without launching simulations.")
    p.add_argument("--sweep-mode", choices=("both", "shared_b", "condition_b"), default="both",
                   help="Run shared-b PCI branch, condition-specific-b PCI branch, or both.")
    p.add_argument("--scenario", action="append", dest="scenarios", default=None,
                   help="Optional manual scenario override applied to both branches.")
    p.add_argument("--b-values", type=float, nargs="+", default=list(DEFAULT_B_VALUES))
    p.add_argument("--condition-b-gradient", action="append", dest="condition_b_gradients", default=None,
                   choices=tuple(CONDITION_B_GRADIENTS.keys()),
                   help="Condition-specific b gradient to run. Defaults to doc_gradient.")

    # TVBSim-style multi-trial PCI convention: 100 independent perturbation trials.
    p.add_argument("--n-trials-pci", type=int, default=100)
    p.add_argument("--pci-transient-ms", type=float, default=4000.0)
    p.add_argument("--t-analysis-ms-pci", type=float, default=300.0)
    p.add_argument("--pci-trial-sim-ms", type=float, default=8000.0)
    p.add_argument("--stim-amplitude-pci", type=float, default=0.00030)
    p.add_argument("--stim-duration-ms-pci", type=float, default=10.0)
    p.add_argument("--stim-region-pci", type=int, nargs="+", default=[18])
    p.add_argument("--stim-onset-seed", type=int, default=0)
    return p.parse_args()


def _select_scenarios(keys: tuple[str, ...], scenarios_arg: list[str] | None) -> dict[str, dict[str, Any]]:
    selected = tuple(scenarios_arg) if scenarios_arg else keys
    out: dict[str, dict[str, Any]] = {}
    for s in selected:
        if s not in SCENARIOS:
            raise KeyError(f"unknown scenario: {s}")
        out[s] = SCENARIOS[s]
    return out


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_parameter_plan_csv(
    path: Path,
    *,
    branch_name: str,
    scenarios: dict[str, dict[str, Any]],
    b_values: list[float],
    condition_b_gradients: dict[str, dict[str, float]],
    subject_count: int,
    n_trials: int,
) -> None:
    rows: list[dict[str, Any]] = []
    if branch_name == "shared_b":
        for b_val in b_values:
            for scenario_key, scenario_cfg in scenarios.items():
                rows.append({
                    "branch": branch_name,
                    "b_tag": _b_tag(float(b_val)),
                    "scenario": scenario_key,
                    "noise_alpha": float(scenario_cfg["noise_alpha"]),
                    "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
                    "b_mode": "shared",
                    "b_e": float(b_val),
                    "b_CNT": float(b_val),
                    "b_EMCS": float(b_val),
                    "b_MCS": float(b_val),
                    "b_UWS": float(b_val),
                    "b_COMA": float(b_val),
                    "subjects": int(subject_count),
                    "trials_per_subject": int(n_trials),
                    "trial_simulations": int(subject_count) * int(n_trials),
                })
    else:
        for gradient_name, gradient in condition_b_gradients.items():
            for scenario_key, scenario_cfg in scenarios.items():
                rows.append({
                    "branch": branch_name,
                    "b_tag": _condition_b_tag(str(gradient_name)),
                    "scenario": scenario_key,
                    "noise_alpha": float(scenario_cfg["noise_alpha"]),
                    "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
                    "b_mode": "condition_specific",
                    "b_e": "",
                    "b_CNT": float(gradient["CNT"]),
                    "b_EMCS": float(gradient["EMCS"]),
                    "b_MCS": float(gradient["MCS"]),
                    "b_UWS": float(gradient["UWS"]),
                    "b_COMA": float(gradient["COMA"]),
                    "subjects": int(subject_count),
                    "trials_per_subject": int(n_trials),
                    "trial_simulations": int(subject_count) * int(n_trials),
                })
    write_rows_csv(path, rows)


def run_pool(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not jobs:
        return results

    print(f"[03] dispatching {len(jobs)} PCI trial jobs on {workers} workers")
    with ProcessPoolExecutor(max_workers=int(workers), initializer=worker_initializer) as ex:
        futs = [ex.submit(run_pci_trial_job, **job) for job in jobs]
        n = len(futs)
        for i, fut in enumerate(as_completed(futs), start=1):
            out = fut.result()
            results.append(out)
            if i == 1 or i % max(1, n // 100) == 0 or i == n:
                print(f"[03] progress {i}/{n}")
    return results


def main() -> None:
    args = parse_args()

    b_values = [float(x) for x in args.b_values]
    selected_gradient_names = (
        list(args.condition_b_gradients)
        if args.condition_b_gradients
        else list(CONDITION_B_GRADIENTS.keys())
    )
    condition_b_gradients = {
        name: CONDITION_B_GRADIENTS[name]
        for name in selected_gradient_names
    }
    subject_jobs = get_subject_jobs(args.dataset_root)
    rate_monitor_period_ms = float(RATE_MONITOR_PERIOD_MS_OLD)

    rng = np.random.default_rng(int(args.stim_onset_seed))
    stim_onsets: dict[int, int] = {
        trial_seed: int(
            rng.integers(
                int(args.pci_transient_ms + args.t_analysis_ms_pci),
                int(args.pci_trial_sim_ms - args.t_analysis_ms_pci),
            )
        )
        for trial_seed in range(int(args.n_trials_pci))
    }

    output_root = args.output_root
    branches: list[dict[str, Any]] = []
    if args.sweep_mode in {"both", "shared_b"}:
        scenarios = _select_scenarios(SHARED_B_PCI_SCENARIO_KEYS, args.scenarios)
        branches.append({
            "name": "shared_b",
            "root": output_root / "shared_b",
            "mode": "shared_b",
            "scenarios": scenarios,
            "b_values": b_values,
            "condition_b_gradients": {},
            "parameter_combinations": len(scenarios) * len(b_values),
        })
    if args.sweep_mode in {"both", "condition_b"}:
        scenarios = _select_scenarios(CONDITION_B_PCI_SCENARIO_KEYS, args.scenarios)
        branches.append({
            "name": "condition_b",
            "root": output_root / "condition_b",
            "mode": "condition_b",
            "scenarios": scenarios,
            "b_values": [],
            "condition_b_gradients": condition_b_gradients,
            "parameter_combinations": len(scenarios) * len(condition_b_gradients),
        })

    for branch in branches:
        branch_root = Path(branch["root"])
        sims_pci_root = branch_root / "sims_pci"
        logs_root = branch_root / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        scenarios = dict(branch["scenarios"])
        write_parameter_plan_csv(
            branch_root / "parameter_plan_pci.csv",
            branch_name=str(branch["name"]),
            scenarios=scenarios,
            b_values=list(branch["b_values"]),
            condition_b_gradients=dict(branch["condition_b_gradients"]),
            subject_count=len(subject_jobs),
            n_trials=int(args.n_trials_pci),
        )
        manifest = {
            "script": "03_pci_trial_sims_hybrid.py",
            "branch_name": str(branch["name"]),
            "dataset_root": str(args.dataset_root),
            "output_root": str(branch_root),
            "sims_pci_root": str(sims_pci_root),
            "scenarios": scenarios,
            "subjects_total": len(subject_jobs),
            "base_parameter_model_new": BASE_PARAMETER_MODEL_NEW,
            "b_values": [float(x) for x in branch["b_values"]],
            "condition_b_gradients": dict(branch["condition_b_gradients"]),
            "rate_monitor_period_ms_old": rate_monitor_period_ms,
            "n_trials_pci": int(args.n_trials_pci),
            "pci_transient_ms": float(args.pci_transient_ms),
            "t_analysis_ms_pci": float(args.t_analysis_ms_pci),
            "pci_trial_sim_ms": float(args.pci_trial_sim_ms),
            "stim_amplitude_pci": float(args.stim_amplitude_pci),
            "stim_duration_ms_pci": float(args.stim_duration_ms_pci),
            "stim_region_pci": list(int(x) for x in args.stim_region_pci),
            "stim_onset_seed": int(args.stim_onset_seed),
            "stim_onsets_ms_by_trial_seed": {str(k): float(v) for k, v in stim_onsets.items()},
            "total_parameter_combinations": int(branch["parameter_combinations"]),
            "total_trial_simulations_theoretical": int(branch["parameter_combinations"]) * len(subject_jobs) * int(args.n_trials_pci),
        }
        save_json(branch_root / "run_manifest_03_pci.json", manifest)

        jobs: list[dict[str, Any]] = []
        if branch["mode"] == "shared_b":
            for b_val in branch["b_values"]:
                btag = _b_tag(float(b_val))
                for scenario_key, scenario_cfg in scenarios.items():
                    for sj in subject_jobs:
                        out_dir = sims_pci_root / btag / scenario_key / sj.cohort / sj.subject_id
                        for trial_seed in range(int(args.n_trials_pci)):
                            npz_path = out_dir / f"trial_{trial_seed:03d}.npz"
                            if npz_path.exists() and not args.overwrite:
                                continue
                            base_model = deepcopy(BASE_PARAMETER_MODEL_NEW)
                            base_model["b_e"] = float(b_val)
                            jobs.append({
                                "scenario_key": scenario_key,
                                "noise_alpha": float(scenario_cfg["noise_alpha"]),
                                "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
                                "cohort": sj.cohort,
                                "subject_id": sj.subject_id,
                                "trial_seed": int(trial_seed),
                                "dataset_root": str(args.dataset_root),
                                "output_dir": str(out_dir),
                                "transient_ms": float(args.pci_transient_ms),
                                "t_analysis_ms": float(args.t_analysis_ms_pci),
                                "rate_monitor_period_ms": float(rate_monitor_period_ms),
                                "base_parameter_model": base_model,
                                "stim_amplitude": float(args.stim_amplitude_pci),
                                "stim_duration_ms": float(args.stim_duration_ms_pci),
                                "stim_region": [int(x) for x in args.stim_region_pci],
                                "stim_onset_ms": float(stim_onsets[trial_seed]),
                                "total_sim_ms": float(args.pci_trial_sim_ms),
                            })
        else:
            for gradient_name, gradient in branch["condition_b_gradients"].items():
                btag = _condition_b_tag(str(gradient_name))
                for scenario_key, scenario_cfg in scenarios.items():
                    for sj in subject_jobs:
                        if sj.condition not in gradient:
                            raise KeyError(
                                f"condition {sj.condition!r} missing from b-gradient {gradient_name!r}"
                            )
                        out_dir = sims_pci_root / btag / scenario_key / sj.cohort / sj.subject_id
                        b_val = float(gradient[sj.condition])
                        for trial_seed in range(int(args.n_trials_pci)):
                            npz_path = out_dir / f"trial_{trial_seed:03d}.npz"
                            if npz_path.exists() and not args.overwrite:
                                continue
                            base_model = deepcopy(BASE_PARAMETER_MODEL_NEW)
                            base_model["b_e"] = b_val
                            jobs.append({
                                "scenario_key": scenario_key,
                                "noise_alpha": float(scenario_cfg["noise_alpha"]),
                                "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
                                "cohort": sj.cohort,
                                "subject_id": sj.subject_id,
                                "trial_seed": int(trial_seed),
                                "dataset_root": str(args.dataset_root),
                                "output_dir": str(out_dir),
                                "transient_ms": float(args.pci_transient_ms),
                                "t_analysis_ms": float(args.t_analysis_ms_pci),
                                "rate_monitor_period_ms": float(rate_monitor_period_ms),
                                "base_parameter_model": base_model,
                                "stim_amplitude": float(args.stim_amplitude_pci),
                                "stim_duration_ms": float(args.stim_duration_ms_pci),
                                "stim_region": [int(x) for x in args.stim_region_pci],
                                "stim_onset_ms": float(stim_onsets[trial_seed]),
                                "total_sim_ms": float(args.pci_trial_sim_ms),
                            })

        print(f"[03] branch={branch['name']} PCI trial jobs queued: {len(jobs)}")

        rows: list[dict[str, Any]] = []
        if args.dry_run:
            print(f"[03] branch={branch['name']} dry-run only; no PCI simulations launched.")
        elif jobs:
            rows = run_pool(jobs, args.workers)
            write_rows_csv(logs_root / "pci_jobs_completed.csv", rows)

        summary = {
            "branch_name": str(branch["name"]),
            "dry_run": bool(args.dry_run),
            "pci_jobs_completed": len(rows),
            "pci_jobs_queued": len(jobs),
        }
        save_json(logs_root / "run_summary_pci.json", summary)

    print(f"[03] done. outputs -> {output_root}")


if __name__ == "__main__":
    main()
