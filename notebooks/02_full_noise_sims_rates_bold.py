#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from brain_act_hybrid_common import (
    BASE_PARAMETER_MODEL_NEW,
    DATASET_ROOT,
    PROJECT_ROOT,
    RATE_MONITOR_PERIOD_MS_OLD,
    SCENARIOS,
    get_subject_jobs,
    save_json,
)

from tvbtoolkit.core.paths import doc_liege_results
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import (
    run_simulation_only_job,
    worker_initializer,
)


ALPHA_SWEEP_PCT = tuple(range(5, 51, 5))
DEFAULT_SCENARIO_KEYS = (
    "private_alpha0",
    *(f"global_alpha_{a:03d}" for a in ALPHA_SWEEP_PCT),
    *(f"sc_alpha_{a:03d}" for a in ALPHA_SWEEP_PCT),
)
DEFAULT_B_VALUES = (5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
EXCLUDED_CONDITIONS: set[str] = set()
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


def _compute_bold_target_points(*, bold_target_points: int | None, bold_target_minutes: float | None, bold_tr_s: float) -> int:
    if bold_target_points is not None:
        if int(bold_target_points) <= 0:
            raise ValueError("--bold-target-points must be > 0")
        return int(bold_target_points)
    minutes = 4.0 if bold_target_minutes is None else float(bold_target_minutes)
    if minutes <= 0.0:
        raise ValueError("--bold-target-minutes must be > 0")
    return max(1, int(round((minutes * 60.0) / float(bold_tr_s))))


def _normalise_b_values(raw: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for v in raw:
        x = float(v)
        if x <= 0.0:
            raise ValueError("all --b-values must be > 0")
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _condition_b_tag(name: str) -> str:
    return f"condb_{name}"



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run spontaneous whole-brain simulations (firing rates + BOLD) with parameter sweeps over "
            "noise scenario and adaptation b_e using updated model params (T=20, updated P_e/P_i)."
        )
    )
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--output-root", type=Path, default=doc_liege_results("notebooks_outputs", "ba_sim_hybrid"))
    p.add_argument("--workers", type=int, default=max(1, int(round((os.cpu_count() or 8) * 0.8))))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Build manifests and report queued jobs without launching simulations.")
    p.add_argument("--scenario", action="append", dest="scenarios", default=None)
    p.add_argument("--b-values", type=float, nargs="+", default=list(DEFAULT_B_VALUES))
    p.add_argument("--sweep-mode", choices=("both", "shared_b", "condition_b"), default="both",
                   help="Run shared-b grid, condition-specific b gradients, or both.")
    p.add_argument("--condition-b-gradient", action="append", dest="condition_b_gradients", default=None,
                   choices=tuple(CONDITION_B_GRADIENTS.keys()),
                   help="Condition-specific b gradient to run. Can be passed multiple times; "
                        "defaults to all gradients.")

    p.add_argument("--seed-spontaneous", type=int, default=0)
    p.add_argument("--transient-ms", type=float, default=4000.0)
    p.add_argument("--bold-target-minutes", type=float, default=4.0)
    p.add_argument("--bold-target-points", type=int, default=None)
    p.add_argument("--bold-tr-s", type=float, default=2.4)
    return p.parse_args()



def choose_scenarios(scenarios_arg: list[str] | None) -> dict[str, dict[str, Any]]:
    if not scenarios_arg:
        return {k: SCENARIOS[k] for k in DEFAULT_SCENARIO_KEYS}
    out: dict[str, dict[str, Any]] = {}
    for s in scenarios_arg:
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


def write_manifest(
    *,
    output_root: Path,
    args: argparse.Namespace,
    scenarios: dict[str, dict[str, Any]],
    b_values: list[float],
    condition_b_gradients: dict[str, dict[str, float]],
    subject_jobs_total: int,
    rate_monitor_period_ms: float,
    bold_target_points: int,
    spontaneous_sim_ms: float,
    branch_name: str,
    branch_sims_root: Path,
    total_parameter_combinations: int,
    subject_jobs_all_total: int,
) -> None:
    manifest = {
        "script": "02_full_noise_sims_rates_bold.py",
        "branch_name": branch_name,
        "dataset_root": str(args.dataset_root),
        "output_root": str(output_root),
        "sims_root": str(branch_sims_root),
        "scenarios": scenarios,
        "default_scenarios_when_unspecified": list(DEFAULT_SCENARIO_KEYS),
        "alpha_zero_policy": (
            "alpha=0 is simulated once as private_alpha0; global/sc-informed "
            "shared-noise sweeps start at alpha=0.05 to avoid duplicate no-shared-noise runs."
        ),
        "b_values": [float(x) for x in b_values],
        "condition_b_gradients": condition_b_gradients,
        "subjects_total": int(subject_jobs_total),
        "subjects_total_before_exclusions": int(subject_jobs_all_total),
        "excluded_conditions": sorted(EXCLUDED_CONDITIONS),
        "base_parameter_model_new": BASE_PARAMETER_MODEL_NEW,
        "rate_monitor_period_ms_old": float(rate_monitor_period_ms),
        "transient_ms": float(args.transient_ms),
        "bold_target_minutes_requested": float(args.bold_target_minutes) if args.bold_target_minutes is not None else None,
        "bold_target_points": int(bold_target_points),
        "bold_minutes_effective": float(bold_target_points) * float(args.bold_tr_s) / 60.0,
        "bold_tr_s": float(args.bold_tr_s),
        "spontaneous_sim_ms": float(spontaneous_sim_ms),
        "seed_spontaneous": int(args.seed_spontaneous),
        "total_noise_scenarios": len(scenarios),
        "total_parameter_combinations": int(total_parameter_combinations),
        "total_simulations_theoretical": int(total_parameter_combinations) * int(subject_jobs_total),
    }
    save_json(output_root / "run_manifest_02_spontaneous.json", manifest)


def write_parameter_plan_csv(
    path: Path,
    *,
    branch_name: str,
    scenarios: dict[str, dict[str, Any]],
    b_values: list[float],
    condition_b_gradients: dict[str, dict[str, float]],
    subject_count: int,
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
                    "simulations": int(subject_count),
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
                    "simulations": int(subject_count),
                })
    write_rows_csv(path, rows)



def run_pool(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not jobs:
        return results

    print(f"[02] dispatching {len(jobs)} spontaneous jobs on {workers} workers")
    with ProcessPoolExecutor(max_workers=int(workers), initializer=worker_initializer) as ex:
        futs = [ex.submit(run_simulation_only_job, **job) for job in jobs]
        n = len(futs)
        for i, fut in enumerate(as_completed(futs), start=1):
            out = fut.result()
            results.append(out)
            if i == 1 or i % max(1, n // 100) == 0 or i == n:
                print(f"[02] progress {i}/{n}")
    return results



def main() -> None:
    args = parse_args()

    scenarios = choose_scenarios(args.scenarios)
    b_values = _normalise_b_values(list(args.b_values))
    selected_gradient_names = (
        list(args.condition_b_gradients)
        if args.condition_b_gradients
        else list(CONDITION_B_GRADIENTS.keys())
    )
    condition_b_gradients = {
        name: CONDITION_B_GRADIENTS[name]
        for name in selected_gradient_names
    }
    subject_jobs_all = get_subject_jobs(args.dataset_root)
    subject_jobs = [sj for sj in subject_jobs_all if sj.condition not in EXCLUDED_CONDITIONS]

    bold_period_ms = float(args.bold_tr_s) * 1000.0
    bold_target_points = _compute_bold_target_points(
        bold_target_points=args.bold_target_points,
        bold_target_minutes=args.bold_target_minutes,
        bold_tr_s=float(args.bold_tr_s),
    )
    spontaneous_sim_ms = float(args.transient_ms) + float(bold_target_points) * bold_period_ms
    rate_monitor_period_ms = float(RATE_MONITOR_PERIOD_MS_OLD)

    output_root = args.output_root

    branches: list[dict[str, Any]] = []
    if args.sweep_mode in {"both", "shared_b"}:
        branches.append({
            "name": "shared_b",
            "root": output_root / "shared_b",
            "mode": "shared_b",
            "b_values": b_values,
            "condition_b_gradients": {},
            "parameter_combinations": len(scenarios) * len(b_values),
        })
    if args.sweep_mode in {"both", "condition_b"}:
        branches.append({
            "name": "condition_b",
            "root": output_root / "condition_b",
            "mode": "condition_b",
            "b_values": [],
            "condition_b_gradients": condition_b_gradients,
            "parameter_combinations": len(scenarios) * len(condition_b_gradients),
        })

    for branch in branches:
        branch_root = Path(branch["root"])
        sims_root = branch_root / "sims"
        logs_root = branch_root / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        write_manifest(
            output_root=branch_root,
            args=args,
            scenarios=scenarios,
            b_values=list(branch["b_values"]),
            condition_b_gradients=dict(branch["condition_b_gradients"]),
            subject_jobs_total=len(subject_jobs),
            rate_monitor_period_ms=rate_monitor_period_ms,
            bold_target_points=bold_target_points,
            spontaneous_sim_ms=spontaneous_sim_ms,
            branch_name=str(branch["name"]),
            branch_sims_root=sims_root,
            total_parameter_combinations=int(branch["parameter_combinations"]),
            subject_jobs_all_total=len(subject_jobs_all),
        )
        write_parameter_plan_csv(
            branch_root / "parameter_plan.csv",
            branch_name=str(branch["name"]),
            scenarios=scenarios,
            b_values=list(branch["b_values"]),
            condition_b_gradients=dict(branch["condition_b_gradients"]),
            subject_count=len(subject_jobs),
        )

        jobs: list[dict[str, Any]] = []
        if branch["mode"] == "shared_b":
            for b_val in branch["b_values"]:
                btag = _b_tag(float(b_val))
                for scenario_key, scenario_cfg in scenarios.items():
                    for sj in subject_jobs:
                        out_dir = sims_root / btag / scenario_key / sj.cohort / sj.subject_id
                        npz_path = out_dir / f"seed_{int(args.seed_spontaneous):03d}.npz"
                        if npz_path.exists() and not args.overwrite:
                            continue
                        base_model = deepcopy(BASE_PARAMETER_MODEL_NEW)
                        base_model["b_e"] = float(b_val)
                        jobs.append(
                            {
                                "scenario_key": scenario_key,
                                "noise_alpha": float(scenario_cfg["noise_alpha"]),
                                "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
                                "cohort": sj.cohort,
                                "subject_id": sj.subject_id,
                                "seed": int(args.seed_spontaneous),
                                "dataset_root": str(args.dataset_root),
                                "output_dir": str(out_dir),
                                "simulation_length_ms": float(spontaneous_sim_ms),
                                "rate_monitor_period_ms": float(rate_monitor_period_ms),
                                "transient_ms": float(args.transient_ms),
                                "base_parameter_model": base_model,
                                "enable_bold": True,
                                "bold_period_ms": float(bold_period_ms),
                            }
                        )
        else:
            for gradient_name, gradient in branch["condition_b_gradients"].items():
                btag = _condition_b_tag(str(gradient_name))
                for scenario_key, scenario_cfg in scenarios.items():
                    for sj in subject_jobs:
                        if sj.condition not in gradient:
                            raise KeyError(
                                f"condition {sj.condition!r} missing from b-gradient {gradient_name!r}"
                            )
                        b_val = float(gradient[sj.condition])
                        out_dir = sims_root / btag / scenario_key / sj.cohort / sj.subject_id
                        npz_path = out_dir / f"seed_{int(args.seed_spontaneous):03d}.npz"
                        if npz_path.exists() and not args.overwrite:
                            continue
                        base_model = deepcopy(BASE_PARAMETER_MODEL_NEW)
                        base_model["b_e"] = b_val
                        jobs.append(
                            {
                                "scenario_key": scenario_key,
                                "noise_alpha": float(scenario_cfg["noise_alpha"]),
                                "shared_noise_mode": str(scenario_cfg["shared_noise_mode"]),
                                "cohort": sj.cohort,
                                "subject_id": sj.subject_id,
                                "seed": int(args.seed_spontaneous),
                                "dataset_root": str(args.dataset_root),
                                "output_dir": str(out_dir),
                                "simulation_length_ms": float(spontaneous_sim_ms),
                                "rate_monitor_period_ms": float(rate_monitor_period_ms),
                                "transient_ms": float(args.transient_ms),
                                "base_parameter_model": base_model,
                                "enable_bold": True,
                                "bold_period_ms": float(bold_period_ms),
                            }
                        )

        print(f"[02] branch={branch['name']} spontaneous jobs queued: {len(jobs)}")

        rows: list[dict[str, Any]] = []
        if args.dry_run:
            print(f"[02] branch={branch['name']} dry-run only; no simulations launched.")
        elif jobs:
            rows = run_pool(jobs, args.workers)
            write_rows_csv(logs_root / "spontaneous_jobs_completed.csv", rows)

        summary = {
            "branch_name": str(branch["name"]),
            "dry_run": bool(args.dry_run),
            "spontaneous_jobs_completed": len(rows),
            "spontaneous_jobs_queued": len(jobs),
        }
        save_json(logs_root / "run_summary_spontaneous.json", summary)

    print(f"[02] done. outputs -> {output_root}")


if __name__ == "__main__":
    main()
