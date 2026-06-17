"""Demo: subject-specific Brain-Act AAL90 whole-brain runs in TVBToolkit."""

from __future__ import annotations

from pathlib import Path

from tvbtoolkit import (
    BrainActSubjectConfig,
    list_subjects,
    plot_brain_state_occupancy,
    plot_cohort_subject_metrics,
    run_cohort_batch,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "brain_act" / "converted"
    output_root = project_root / "examples" / "outputs" / "brain_act_subject_batch_demo"

    cfg = BrainActSubjectConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        seeds=(0, 1),
        simulation_length_ms=5000.0,
        monitor_mode="temporal_average",
        temporal_average_period_ms=1.0,
        zerlaut_order=2,  # second-order: matches TVBSim default
    )

    # Demo subset: one subject per cohort
    cohorts = ["control", "mcs", "uws"]
    results_by_cohort = {}
    for cohort in cohorts:
        sid = list_subjects(dataset_root=dataset_root, cohort=cohort)[0]
        results_by_cohort[cohort] = run_cohort_batch(
            cohort=cohort,
            subjects=[sid],
            cfg=cfg,
            n_jobs=1,
            use_processes=False,
            show_progress=True,
        )

    plot_cohort_subject_metrics(
        results_by_cohort,
        save_path=output_root / "figures" / "cohort_subject_metrics.png",
    )
    plot_brain_state_occupancy(
        results_by_cohort,
        save_path=output_root / "figures" / "cohort_brain_state_occupancy.png",
    )

    print("Saved outputs under", output_root)


if __name__ == "__main__":
    main()
