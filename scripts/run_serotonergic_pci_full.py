#!/usr/bin/env python3
"""Full-dataset serotonergic PCI run.

This is the production version of ``run_serotonergic_pci_pilot.py``. It keeps
the same model and PCI protocol, but defaults to all available subjects and 50
perturbation trials per subject. By default baseline PCI can be read from an
existing condition-b PCI cache; pass ``--simulate-baseline`` to simulate
occupancy 0.0 alongside all positive-dose 5-HT2A split-leak conditions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd

import run_serotonergic_pci_pilot as pilot
from run_serotonergic_pci_pilot import worker_initializer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=pilot.DATASET_ROOT)
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=pilot.doc_liege_results("doc_simulation_run", "ba_sim_hybrid", "condition_b", "sims_pci"),
        help="Existing condition-b PCI trial cache used for occupancy 0.0 when --simulate-baseline is not set.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=pilot._REPO_ROOT / "results" / "serotonergic_pci_full_50trials",
    )
    p.add_argument("--cohorts", nargs="+", default=["coma", "uws", "mcs", "emcs", "control"])
    p.add_argument(
        "--max-subjects-per-cohort",
        type=int,
        default=None,
        help="Optional cap per cohort. Omit for the full available dataset.",
    )
    p.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Explicit subject as cohort:subject_id. Can be passed multiple times.",
    )
    p.add_argument("--scenario", default="private_alpha0")
    p.add_argument("--trial-seeds", type=int, nargs="+", default=list(range(50)))
    p.add_argument(
        "--occupancies",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.766],
        help=(
            "5-HT2A occupancy levels. 0.0 is baseline and is read from --baseline-root; "
            "positive values are simulated."
        ),
    )
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
    p.add_argument(
        "--simulate-baseline",
        action="store_true",
        help="Simulate occupancy 0.0 into output-root/sims_pci/occ_000 instead of reading baseline from --baseline-root.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate-only", action="store_true", help="Skip simulations and recompute tables/figures from existing files.")
    p.add_argument("--skip-aggregate", action="store_true", help="Run simulations only; do not compute PCI tables at the end.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _select_subjects(args: argparse.Namespace):
    if args.subject:
        return pilot._select_subjects(
            args.dataset_root,
            args.cohorts,
            subjects_per_cohort=10**9,
            explicit_subjects=args.subject,
        )

    jobs = pilot.get_subject_jobs(args.dataset_root)
    selected = []
    for cohort in args.cohorts:
        cohort_jobs = [j for j in jobs if j.cohort == cohort]
        if args.max_subjects_per_cohort is not None:
            cohort_jobs = cohort_jobs[: max(0, int(args.max_subjects_per_cohort))]
        selected.extend(cohort_jobs)
    return selected


def _run_manifest(args: argparse.Namespace, subjects: list[Any], scenario_cfg: dict[str, Any], stim_onsets: dict[int, float]) -> dict[str, Any]:
    occupancies = [float(o) for o in args.occupancies]
    positive_occupancies = [o for o in occupancies if o > 0.0]
    return {
        "script": "scripts/run_serotonergic_pci_full.py",
        "dataset_root": str(args.dataset_root),
        "baseline_root": str(args.baseline_root),
        "output_root": str(args.output_root),
        "scenario": args.scenario,
        "scenario_cfg": scenario_cfg,
        "subjects": [s.__dict__ for s in subjects],
        "n_subjects": int(len(subjects)),
        "cohorts": list(args.cohorts),
        "trial_seeds": [int(s) for s in args.trial_seeds],
        "n_trials": int(len(args.trial_seeds)),
        "stim_onsets_ms_by_trial_seed": {str(k): float(v) for k, v in stim_onsets.items()},
        "occupancies": occupancies,
        "positive_occupancies": positive_occupancies,
        "simulate_baseline": bool(args.simulate_baseline),
        "n_positive_doses": int(len(positive_occupancies)),
        "e_l_e_drug": float(args.e_l_e_drug),
        "e_l_i_drug": float(args.e_l_i_drug),
        "workers": int(args.workers),
        "overwrite": bool(args.overwrite),
    }


def _build_trial_jobs(
    args: argparse.Namespace,
    subjects: list[Any],
    scenario_cfg: dict[str, Any],
    receptor_map,
    stim_onsets: dict[int, float],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    sim_occupancies = [
        float(o)
        for o in args.occupancies
        if float(o) > 0.0 or (float(o) <= 0.0 and bool(args.simulate_baseline))
    ]
    for occ in sim_occupancies:
        for sj in subjects:
            out_dir = args.output_root / "sims_pci" / pilot._occ_tag(occ) / args.scenario / sj.cohort / sj.subject_id
            for trial_seed in [int(s) for s in args.trial_seeds]:
                save_path = out_dir / f"trial_{trial_seed:03d}.npz"
                if save_path.exists() and not args.overwrite:
                    continue
                jobs.append(
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
    return jobs


def _aggregate(args: argparse.Namespace, subjects: list[Any]) -> pd.DataFrame:
    metric_rows: list[dict[str, Any]] = []
    trial_seeds = [int(s) for s in args.trial_seeds]
    for sj in subjects:
        for occ in [float(o) for o in args.occupancies]:
            if occ <= 0.0 and bool(args.simulate_baseline):
                base = args.output_root / "sims_pci" / pilot._occ_tag(occ) / args.scenario / sj.cohort / sj.subject_id
                paths = [base / f"trial_{seed:03d}.npz" for seed in trial_seeds]
            else:
                root = args.baseline_root if occ <= 0.0 else args.output_root
                paths = pilot._condition_paths(root, occ, args.scenario, sj.cohort, sj.subject_id, trial_seeds)
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing {len(missing)} trial files for {sj.cohort}/{sj.subject_id} "
                    f"occ={occ}; first missing: {missing[0]}"
                )
            pci_mean, pci_per_trial = pilot._compute_pci_for_condition(paths)
            metric_rows.append(
                {
                    "cohort": sj.cohort,
                    "condition": sj.condition,
                    "subject_id": sj.subject_id,
                    "scenario": args.scenario,
                    "occupancy": float(occ),
                    "n_trials": int(len(paths)),
                    "pci_mean": float(pci_mean),
                    "pci_std": float(pd.Series(pci_per_trial).std(ddof=0)),
                    "pci_per_trial": json.dumps([float(x) for x in pci_per_trial]),
                    "trial_paths": json.dumps([str(p) for p in paths]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    tables_dir = args.output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables_dir / "serotonergic_pci_subject_metrics.csv", index=False)
    pilot._plot(metrics, args.output_root)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_root / "tables").mkdir(parents=True, exist_ok=True)

    if args.scenario not in pilot.SCENARIOS:
        raise KeyError(f"Unknown scenario {args.scenario!r}.")

    scenario_cfg = pilot.SCENARIOS[args.scenario]
    subjects = _select_subjects(args)
    receptor_map = pilot.get_5ht2a_aal90()
    stim_onsets = pilot._stim_onsets(
        [int(s) for s in args.trial_seeds],
        transient_ms=float(args.transient_ms),
        t_analysis_ms=float(args.t_analysis_ms),
        trial_sim_ms=float(args.trial_sim_ms),
        seed=int(args.stim_onset_seed),
    )

    manifest = _run_manifest(args, subjects, scenario_cfg, stim_onsets)
    pilot._write_json(args.output_root / "logs" / "run_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["n_subjects", "n_trials", "occupancies", "n_positive_doses", "workers"]}, indent=2))

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    completed_rows: list[dict[str, Any]] = []
    if not args.aggregate_only:
        trial_jobs = _build_trial_jobs(args, subjects, scenario_cfg, receptor_map, stim_onsets)
        n_simulated_occupancies = len(
            [
                o
                for o in args.occupancies
                if float(o) > 0.0 or (float(o) <= 0.0 and bool(args.simulate_baseline))
            ]
        )
        expected_total = len(subjects) * n_simulated_occupancies * len(args.trial_seeds)
        print(
            "[sero-pci-full] "
            f"queued {len(trial_jobs)} missing serotonergic trial simulations "
            f"({expected_total} total simulated occupancy trials) on {int(args.workers)} workers",
            flush=True,
        )
        if trial_jobs:
            with ProcessPoolExecutor(max_workers=int(args.workers), initializer=worker_initializer) as ex:
                futures = [ex.submit(pilot._run_trial_job, job) for job in trial_jobs]
                total = len(futures)
                for i, fut in enumerate(as_completed(futures), start=1):
                    row = fut.result()
                    completed_rows.append(row)
                    print(
                        "[sero-pci-full] "
                        f"{i}/{total} done occ={row['occupancy']:.3f} "
                        f"{row['condition']}/{row['subject_id']} trial={row['trial_seed']} "
                        f"runtime={row['runtime_s']:.1f}s",
                        flush=True,
                    )

    stamp = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%dT%H%M%S")
    _write_csv(args.output_root / "logs" / f"completed_trials_{stamp}.csv", completed_rows)

    if not args.skip_aggregate:
        metrics = _aggregate(args, subjects)
        print(f"[sero-pci-full] wrote metrics for {len(metrics)} subject/occupancy rows to {args.output_root}")
    else:
        print(f"[sero-pci-full] simulations complete; aggregation skipped for {args.output_root}")


if __name__ == "__main__":
    main()
