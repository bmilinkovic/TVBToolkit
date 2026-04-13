"""End-to-end Brain-Act structural loading, QC plotting, and minimal simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tvbtoolkit import (
    OutputConfig,
    WholeBrainConfig,
    list_subjects,
    load_subject_structural,
    run_whole_brain_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to converted Brain-Act dataset directory (contains index.json).",
    )
    parser.add_argument(
        "--output-root",
        default="examples/outputs/brain_act_demo",
        help="Directory for QC plots and quick simulation outputs.",
    )
    parser.add_argument(
        "--sim-ms",
        type=float,
        default=400.0,
        help="Simulation length in milliseconds for quick smoke runs.",
    )
    args = parser.parse_args()

    out = OutputConfig(root=Path(args.output_root))
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    cohorts = ["control", "mcs", "uws"]

    qc = {}
    for cohort in cohorts:
        sid = list_subjects(dataset_root, cohort=cohort)[0]
        c, l, atlas, meta = load_subject_structural(
            subject_id=sid,
            cohort=cohort,
            dataset_root=dataset_root,
            validate=True,
            normalize="max",
        )
        qc[cohort] = {"subject_id": sid, "C": c, "L": l, "atlas": atlas, "meta": meta}
        print(f"[{cohort}] loaded {sid} with shape {c.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for cohort, color in zip(cohorts, ["#355070", "#6d597a", "#b56576"]):
        c = qc[cohort]["C"]
        degree = np.sum(c > 0, axis=0)
        axes[0].plot(np.sort(degree), label=f"{cohort} ({qc[cohort]['subject_id']})", color=color)
    axes[0].set_title("Degree Distribution")
    axes[0].set_xlabel("Region rank")
    axes[0].set_ylabel("Degree")
    axes[0].legend(frameon=False, fontsize=8)

    for cohort, color in zip(cohorts, ["#355070", "#6d597a", "#b56576"]):
        c = qc[cohort]["C"]
        vals = c[c > 0]
        axes[1].hist(vals, bins=30, alpha=0.45, label=cohort, color=color)
    axes[1].set_title("Weight Histogram")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False, fontsize=8)

    speed_mm_per_ms = 4.0
    delays = []
    labels = []
    for cohort in cohorts:
        l = qc[cohort]["L"]
        d = l[l > 0] / speed_mm_per_ms
        delays.append(d)
        labels.append(cohort)
    axes[2].boxplot(delays, labels=labels, showfliers=False)
    axes[2].set_title("Conduction Delay Summary")
    axes[2].set_ylabel("Delay (ms)")

    fig.tight_layout()
    qc_path = out.figures_dir / "brain_act_qc_summary.png"
    fig.savefig(qc_path, dpi=150)
    plt.close(fig)
    print(f"Saved QC plot to {qc_path}")

    # Minimal simulation for one representative subject per cohort
    for cohort in cohorts:
        c = qc[cohort]["C"]
        l = qc[cohort]["L"]
        sid = qc[cohort]["subject_id"]
        cfg = WholeBrainConfig(
            model_family="adex_zerlaut",
            zerlaut_order=1,
            simulation_length_ms=args.sim_ms,
            dt_ms=0.25,
            monitor_mode="temporal_average",
            temporal_average_period_ms=1.0,
            coupling_strength=0.3,
            conduction_speed=4.0,
            weights=c,
            tract_lengths=l,
        )
        sim = run_whole_brain_simulation(cfg, seed=0)
        save_path = out.simulations_dir / f"{cohort}_{sid}_quick_timeseries.npz"
        np.savez_compressed(save_path, time_ms=sim.time_ms, excitatory=sim.raw, inhibitory=sim.raw_inh)
        print(f"Saved simulation output to {save_path}")


if __name__ == "__main__":
    main()

