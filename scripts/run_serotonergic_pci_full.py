#!/usr/bin/env python3
"""Full-dataset serotonergic PCI run.

This is the production version of ``run_serotonergic_pci_pilot.py``. It runs
all available subjects with 100 perturbation trials per subject. Every dose,
including occupancy zero, uses the same split-gK/gNa model form. Trials are
epoched around their own recorded onset, aligned at the common midpoint, and
reduced to one Casali PCI from the trial-averaged response.
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

import numpy as np
import pandas as pd

import run_serotonergic_pci_pilot as pilot
from run_serotonergic_pci_pilot import worker_initializer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=pilot.DATASET_ROOT)
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=pilot.doc_liege_results("doc_simulation_run", "ba_sim_hybrid", "condition_b", "sims_pci"),
        help="Retained only as provenance; the corrected production run simulates occupancy 0.0 fresh.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=pilot._REPO_ROOT / "results" / "serotonergic_pci_full_100trials_corrected",
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
    p.add_argument("--trial-seeds", type=int, nargs="+", default=list(range(100)))
    p.add_argument(
        "--occupancies",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.766],
        help=(
            "5-HT2A occupancy levels. Every level, including 0.0, is simulated "
            "with the same model form."
        ),
    )
    p.add_argument("--transient-ms", type=float, default=4000.0)
    p.add_argument("--t-analysis-ms", type=float, default=300.0)
    p.add_argument("--trial-sim-ms", type=float, default=8000.0)
    p.add_argument("--stim-amplitude", type=float, default=0.00030)
    p.add_argument("--stim-duration-ms", type=float, default=10.0)
    target_group = p.add_mutually_exclusive_group()
    target_group.add_argument(
        "--stim-region",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit zero-based indices in the converted dataset ordering. "
            "Prefer --stim-region-label so atlas reordering cannot silently change the target."
        ),
    )
    target_group.add_argument(
        "--stim-region-label",
        nargs="+",
        default=None,
        help=(
            "AAL90 target labels resolved against the converted dataset. "
            f"Production default: {pilot.DEFAULT_STIM_REGION_LABEL}."
        ),
    )
    p.add_argument("--stim-onset-seed", type=int, default=0)
    p.add_argument(
        "--receptor-tracer",
        choices=["cimbi", "savli", "talbot"],
        default="cimbi",
    )
    p.add_argument(
        "--receptor-csv",
        type=Path,
        default=pilot.DEFAULT_RECEPTOR_CSV,
    )
    p.add_argument(
        "--pci-binarise-method",
        choices=["casali", "tvbsim"],
        default="casali",
        help=(
            "PCI significance route. 'casali' computes one PCI from the "
            "trial-averaged, bootstrap-thresholded response."
        ),
    )
    p.add_argument("--pci-bootstrap-replicates", type=int, default=500)
    p.add_argument("--pci-alpha", type=float, default=0.01)
    p.add_argument("--pci-bootstrap-seed", type=int, default=0)
    p.add_argument("--e-l-e-drug", type=float, default=-61.2)
    p.add_argument("--e-l-i-drug", type=float, default=-64.4)
    p.add_argument(
        "--b-e-override",
        type=float,
        default=None,
        help=(
            "Use one excitatory-adaptation value (pA) for every diagnosis. "
            "Omit for the diagnosis-configured gradient."
        ),
    )
    p.set_defaults(split_model_all_occupancies=True)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument(
        "--simulate-baseline",
        action="store_true",
        default=True,
        help="Simulate occupancy 0.0 into output-root/sims_pci/occ_000 instead of reading baseline from --baseline-root.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate-only", action="store_true", help="Skip simulations and recompute tables/figures from existing files.")
    p.add_argument("--skip-aggregate", action="store_true", help="Run simulations only; do not compute PCI tables at the end.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.stim_target_was_label = args.stim_region is None
    if args.stim_region is None and args.stim_region_label is None:
        args.stim_region_label = [pilot.DEFAULT_STIM_REGION_LABEL]
    return args


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


def _validate_full_protocol(args: argparse.Namespace) -> None:
    """Fail before simulation if the approved production protocol is changed."""
    pilot._validate_protocol_args(args)
    expected_seeds = list(range(100))
    if args.trial_seeds != expected_seeds:
        raise ValueError(
            "The production serotonergic PCI run requires exactly trial seeds "
            "0..99 (100 trials). Use the pilot runner for smaller diagnostics."
        )
    expected_occupancies = [0.0, 0.25, 0.5, 0.766]
    if len(args.occupancies) != len(expected_occupancies) or not np.allclose(
        np.asarray(args.occupancies, dtype=float),
        np.asarray(expected_occupancies, dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The production dose schedule is exactly 0, 0.25, 0.5, and 0.766."
        )
    if args.cohorts != ["coma", "uws", "mcs", "emcs", "control"]:
        raise ValueError("The production run requires all five cohorts.")
    if args.subject is not None or args.max_subjects_per_cohort is not None:
        raise ValueError(
            "The production runner requires the complete cohort. Use the pilot "
            "runner for subject subsets."
        )
    if str(args.scenario) != "private_alpha0":
        raise ValueError("The approved production scenario is 'private_alpha0'.")
    if not bool(args.stim_target_was_label):
        raise ValueError(
            "The production stimulation target must be supplied by anatomical "
            "label, not by a positional atlas index."
        )
    if [str(label) for label in args.stim_region_label] != [
        pilot.DEFAULT_STIM_REGION_LABEL
    ]:
        raise ValueError(
            "The approved production target is exactly "
            f"{pilot.DEFAULT_STIM_REGION_LABEL!r}."
        )
    if str(args.receptor_tracer) != "cimbi":
        raise ValueError("The approved production receptor tracer is 'cimbi'.")
    if str(args.pci_binarise_method) != "casali":
        raise ValueError(
            "The production analysis requires one Casali PCI from the "
            "time-locked trial-averaged response."
        )
    if not bool(args.simulate_baseline):
        raise ValueError(
            "Occupancy zero must be simulated fresh with the same split-gK/gNa "
            "model form as the positive doses."
        )
    if not bool(args.split_model_all_occupancies):
        raise ValueError("The split-gK/gNa model must be used at every occupancy.")
    if args.b_e_override is not None:
        raise ValueError(
            "The production run requires the diagnosis-configured b_e gradient; "
            "--b-e-override is pilot-only."
        )
    if not np.isclose(float(args.e_l_e_drug), -61.2, rtol=0.0, atol=1e-12):
        raise ValueError("The production excitatory drug endpoint is -61.2 mV.")
    if not np.isclose(float(args.e_l_i_drug), -64.4, rtol=0.0, atol=1e-12):
        raise ValueError("The production inhibitory drug endpoint is -64.4 mV.")
    if args.aggregate_only and args.overwrite:
        raise ValueError(
            "--aggregate-only cannot be combined with --overwrite because "
            "aggregation must validate the existing manifest and trial files."
        )


def _run_manifest(args: argparse.Namespace, subjects: list[Any], scenario_cfg: dict[str, Any], stim_onsets: dict[int, float]) -> dict[str, Any]:
    occupancies = [float(o) for o in args.occupancies]
    positive_occupancies = [o for o in occupancies if o > 0.0]
    manifest = {
        "script": "scripts/run_serotonergic_pci_full.py",
        "protocol_version": pilot.PROTOCOL_VERSION,
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
        "stim_onset_schedule": "unique integer-millisecond onsets sampled without replacement",
        "occupancies": occupancies,
        "positive_occupancies": positive_occupancies,
        "simulate_baseline": bool(args.simulate_baseline),
        "n_positive_doses": int(len(positive_occupancies)),
        "n_expected_trial_files": int(
            len(subjects) * len(occupancies) * len(args.trial_seeds)
        ),
        "transient_ms": float(args.transient_ms),
        "t_analysis_ms": float(args.t_analysis_ms),
        "trial_sim_ms": float(args.trial_sim_ms),
        "integration_dt_ms": 0.1,
        "rate_monitor_period_ms": float(pilot.RATE_MONITOR_PERIOD_MS_OLD),
        "conduction_speed": 4.0,
        "coupling_strength": 0.25,
        "stim_amplitude": float(args.stim_amplitude),
        "stim_duration_ms": float(args.stim_duration_ms),
        "stim_variables": [0],
        "stim_onset_alignment": (
            "per-trial nearest temporal-average sample, then [-window,+window) "
            "epoching to a common midpoint"
        ),
        "pci_binarise_method": str(args.pci_binarise_method),
        "pci_estimator": (
            "one Casali PCI from the baseline-normalized, time-locked "
            "trial-averaged response"
        ),
        "pci_bootstrap_replicates": int(args.pci_bootstrap_replicates),
        "pci_alpha": float(args.pci_alpha),
        "pci_bootstrap_seed": int(args.pci_bootstrap_seed),
        "atlas_ordering": str(args.atlas_ordering),
        "atlas_source": str(args.atlas_source),
        "atlas_labels_sha256": str(args.atlas_labels_sha256),
        "receptor_map_alignment": "AAL region-label join",
        "receptor_tracer": str(args.receptor_tracer),
        "receptor_csv": str(args.receptor_csv),
        "receptor_csv_sha256": str(args.receptor_csv_sha256),
        "receptor_map_sha256": str(args.receptor_map_sha256),
        "stim_region_indices_zero_based": [int(index) for index in args.stim_region],
        "stim_region_labels": [str(label) for label in args.stim_region_label],
        "stim_target_provenance": (
            "Resolved by AAL label; default follows the original notebook's "
            "documented Supp_Motor_Area_L intent, not the legacy positional index."
        ),
        "model_form": (
            "split_gK_gNa_all_occupancies"
            if bool(args.split_model_all_occupancies)
            else "legacy_switch_at_positive_occupancy"
        ),
        "e_l_e_drug": float(args.e_l_e_drug),
        "e_l_i_drug": float(args.e_l_i_drug),
        "b_e_override": (
            None if args.b_e_override is None else float(args.b_e_override)
        ),
        "diagnosis_configured_b_e_pA": {
            str(condition): float(value)
            for condition, value in pilot.CONDITION_B_GRADIENT.items()
        },
        "workers": int(args.workers),
        "overwrite": bool(args.overwrite),
    }
    manifest["protocol_fingerprint"] = pilot._protocol_fingerprint(manifest)
    return manifest


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
                    pilot._validate_existing_trial(
                        save_path,
                        protocol_fingerprint=args.protocol_fingerprint,
                        trial_seed=trial_seed,
                        occupancy=occ,
                        stim_region_labels=args.stim_region_label,
                        receptor_map_sha256=args.receptor_map_sha256,
                        cohort=sj.cohort,
                        condition=sj.condition,
                        subject_id=sj.subject_id,
                        scenario=args.scenario,
                        expected_stim_onset_ms=stim_onsets[trial_seed],
                        atlas_labels_sha256=args.atlas_labels_sha256,
                        receptor_tracer=args.receptor_tracer,
                        receptor_csv_sha256=args.receptor_csv_sha256,
                        expected_t_analysis_ms=args.t_analysis_ms,
                    )
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


def _aggregate(
    args: argparse.Namespace,
    subjects: list[Any],
    stim_onsets: dict[int, float],
) -> pd.DataFrame:
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
            for path, trial_seed in zip(paths, trial_seeds, strict=True):
                pilot._validate_existing_trial(
                    path,
                    protocol_fingerprint=args.protocol_fingerprint,
                    trial_seed=trial_seed,
                    occupancy=occ,
                    stim_region_labels=args.stim_region_label,
                    receptor_map_sha256=args.receptor_map_sha256,
                    cohort=sj.cohort,
                    condition=sj.condition,
                    subject_id=sj.subject_id,
                    scenario=args.scenario,
                    expected_stim_onset_ms=stim_onsets[trial_seed],
                    atlas_labels_sha256=args.atlas_labels_sha256,
                    receptor_tracer=args.receptor_tracer,
                    receptor_csv_sha256=args.receptor_csv_sha256,
                    expected_t_analysis_ms=args.t_analysis_ms,
                )
            pci_mean, pci_per_trial = pilot._compute_pci_for_condition(
                paths,
                binarise_method=args.pci_binarise_method,
                n_bootstrap=args.pci_bootstrap_replicates,
                alpha=args.pci_alpha,
                bootstrap_seed=args.pci_bootstrap_seed,
            )
            metric_rows.append(
                {
                    "cohort": sj.cohort,
                    "condition": sj.condition,
                    "subject_id": sj.subject_id,
                    "scenario": args.scenario,
                    "occupancy": float(occ),
                    "n_trials": int(len(paths)),
                    "pci_estimator": "casali_trial_average",
                    "pci_mean": float(pci_mean),
                    # One condition-level PCI is computed from the averaged
                    # evoked response; this is not a per-trial uncertainty.
                    "pci_std": float("nan"),
                    "n_returned_pci_values": int(len(pci_per_trial)),
                    "pci_values": json.dumps(
                        [float(value) for value in pci_per_trial]
                    ),
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
    _validate_full_protocol(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_root / "tables").mkdir(parents=True, exist_ok=True)

    if args.scenario not in pilot.SCENARIOS:
        raise KeyError(f"Unknown scenario {args.scenario!r}.")

    scenario_cfg = pilot.SCENARIOS[args.scenario]
    subjects = _select_subjects(args)
    if len(subjects) != 189:
        raise RuntimeError(
            "The corrected full-cohort protocol expects exactly 189 subjects; "
            f"the converted dataset resolved {len(subjects)}."
        )
    atlas = pilot._resolve_stim_regions(args)
    if [str(label) for label in args.stim_region_label] != [
        pilot.DEFAULT_STIM_REGION_LABEL
    ]:
        raise AssertionError("Resolved stimulation label changed unexpectedly.")
    args.atlas_ordering = str(atlas.ordering)
    args.atlas_source = str(atlas.source)
    args.atlas_labels_sha256 = pilot._sha256_array(
        np.asarray(atlas.labels, dtype="U128")
    )
    receptor_map = pilot.get_5ht2a_aal90(
        tracer=str(args.receptor_tracer),
        csv_path=args.receptor_csv,
        target_labels=atlas.labels,
    )
    args.receptor_csv_sha256 = pilot._sha256_file(args.receptor_csv)
    args.receptor_map_sha256 = pilot._sha256_array(
        np.asarray(receptor_map, dtype=np.float64)
    )
    stim_onsets = pilot._stim_onsets(
        [int(s) for s in args.trial_seeds],
        transient_ms=float(args.transient_ms),
        t_analysis_ms=float(args.t_analysis_ms),
        trial_sim_ms=float(args.trial_sim_ms),
        seed=int(args.stim_onset_seed),
    )

    manifest = _run_manifest(args, subjects, scenario_cfg, stim_onsets)
    args.protocol_fingerprint = str(manifest["protocol_fingerprint"])
    pilot._write_or_validate_manifest(
        args.output_root / "logs" / "run_manifest.json",
        manifest,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in [
                    "n_subjects",
                    "n_trials",
                    "occupancies",
                    "n_expected_trial_files",
                    "stim_region_labels",
                    "stim_region_indices_zero_based",
                    "receptor_map_alignment",
                    "pci_estimator",
                    "workers",
                ]
            },
            indent=2,
        )
    )

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
        metrics = _aggregate(args, subjects, stim_onsets)
        print(f"[sero-pci-full] wrote metrics for {len(metrics)} subject/occupancy rows to {args.output_root}")
    else:
        print(f"[sero-pci-full] simulations complete; aggregation skipped for {args.output_root}")


if __name__ == "__main__":
    main()
