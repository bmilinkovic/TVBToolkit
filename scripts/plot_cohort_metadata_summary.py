#!/usr/bin/env python3
"""Summarise cohort composition and structural zero-mask burden."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COHORTS = ("control", "emcs", "mcs", "uws", "coma")
LABELS = {"control": "CNT", "emcs": "EMCS", "mcs": "MCS", "uws": "UWS", "coma": "COMA"}
COLORS = {
    "control": "#5B8E77",
    "emcs": "#E9B45F",
    "mcs": "#C65D2E",
    "uws": "#8B6D8F",
    "coma": "#3E506F",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _despine(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3)
    ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.7, zorder=0)


def _stacked_percent(
    ax: mpl.axes.Axes,
    counts: dict[str, Counter],
    categories: list[tuple[str, str, str]],
) -> None:
    x = np.arange(len(COHORTS))
    bottom = np.zeros(len(COHORTS), dtype=float)
    for key, label, color in categories:
        values = np.array(
            [100.0 * counts[c][key] / max(1, sum(counts[c].values())) for c in COHORTS]
        )
        bars = ax.bar(x, values, bottom=bottom, width=0.68, color=color, label=label, zorder=2)
        for bar, value, base in zip(bars, values, bottom):
            if value >= 9.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if color not in {"#D9D9D9", "#B9D8CA"} else "#222222",
                    fontweight="medium",
                )
        bottom += values
    ax.set_ylim(0, 100)
    ax.set_xticks(x, [LABELS[c] for c in COHORTS])
    ax.set_ylabel("Subjects (%)")
    _despine(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    subjects = index["subjects"]
    metadata = {(s["cohort"], s["subject_id"]): s for s in subjects}
    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        with np.load(root / f"subjects_{cohort}.npz") as data:
            ids = np.asarray(data["subject_ids"]).astype(str)
            matrices = np.asarray(data["connectivity"], dtype=float)
        n_nodes = matrices.shape[1]
        upper = np.triu_indices(n_nodes, k=1)
        possible_edges = len(upper[0])
        for sid, matrix in zip(ids, matrices):
            meta = metadata[(cohort, sid)]
            masked_edges = int(np.count_nonzero(matrix[upper] <= 0.0))
            rows.append(
                {
                    "subject_id": sid,
                    "cohort": cohort,
                    "diagnosis": LABELS[cohort],
                    "sedation": meta["sedation"],
                    "stage": meta["stage"],
                    "zero_masked_edges": masked_edges,
                    "possible_undirected_edges": possible_edges,
                    "zero_masked_connections_percent": 100.0 * masked_edges / possible_edges,
                }
            )

    cohort_counts = Counter(str(r["cohort"]) for r in rows)
    sedation = {
        c: Counter(str(r["sedation"]) for r in rows if r["cohort"] == c) for c in COHORTS
    }
    stage = {c: Counter(str(r["stage"]) for r in rows if r["cohort"] == c) for c in COHORTS}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    ax = axes[0, 0]
    x = np.arange(len(COHORTS))
    values = np.array([cohort_counts[c] for c in COHORTS])
    bars = ax.bar(x, values, width=0.68, color=[COLORS[c] for c in COHORTS], zorder=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, str(value), ha="center", va="bottom")
    ax.set_xticks(x, [LABELS[c] for c in COHORTS])
    ax.set_ylabel("Number of subjects")
    ax.set_ylim(0, max(values) * 1.16)
    ax.set_title("A   Cohort composition", loc="left", fontweight="normal")
    _despine(ax)

    ax = axes[0, 1]
    _stacked_percent(
        ax,
        sedation,
        [("non_sedated", "Not sedated", "#B9D8CA"), ("sedated", "Sedated", "#5A4963")],
    )
    ax.set_title("B   Sedation status", loc="left", fontweight="normal")
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.12), ncol=2)

    ax = axes[1, 0]
    _stacked_percent(
        ax,
        stage,
        [("control", "Control", "#B9D8CA"), ("chronic", "Chronic", "#5278A5"), ("acute", "Acute", "#D56B3F")],
    )
    ax.set_title("C   Clinical stage", loc="left", fontweight="normal")
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.12), ncol=3)

    ax = axes[1, 1]
    rng = np.random.default_rng(17)
    damage_by_cohort = [
        np.array([float(r["zero_masked_connections_percent"]) for r in rows if r["cohort"] == c])
        for c in COHORTS
    ]
    violins = ax.violinplot(
        damage_by_cohort,
        positions=x,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, cohort in zip(violins["bodies"], COHORTS):
        body.set_facecolor(COLORS[cohort])
        body.set_edgecolor("none")
        body.set_alpha(0.24)
    for i, (cohort, values_i) in enumerate(zip(COHORTS, damage_by_cohort)):
        jitter = rng.uniform(-0.16, 0.16, len(values_i))
        ax.scatter(
            np.full(len(values_i), i) + jitter,
            values_i,
            s=14,
            color=COLORS[cohort],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            zorder=3,
        )
        median = float(np.median(values_i))
        q1, q3 = np.percentile(values_i, [25, 75])
        ax.vlines(i, q1, q3, color="#171717", lw=2.2, zorder=4)
        ax.scatter(i, median, marker="_", s=120, color="#171717", linewidth=1.5, zorder=5)
    ax.set_xticks(x, [LABELS[c] for c in COHORTS])
    ax.set_ylabel("Zero-masked connections (%)")
    ax.set_title("D   Structural disconnection proxy", loc="left", fontweight="normal")
    _despine(ax)

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.09, top=0.90, wspace=0.30, hspace=0.68)
    stem = output / "cohort_metadata_and_damage_summary"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=400 if suffix == "png" else None,
                    facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    fieldnames = list(rows[0].keys())
    with (output / "cohort_subject_metadata_and_damage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output / "cohort_summary_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cohort", "n", "n_sedated", "n_non_sedated", "n_acute", "n_chronic",
            "damage_median_percent", "damage_q1_percent", "damage_q3_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort, values_i in zip(COHORTS, damage_by_cohort):
            q1, median, q3 = np.percentile(values_i, [25, 50, 75])
            writer.writerow(
                {
                    "cohort": LABELS[cohort],
                    "n": cohort_counts[cohort],
                    "n_sedated": sedation[cohort]["sedated"],
                    "n_non_sedated": sedation[cohort]["non_sedated"],
                    "n_acute": stage[cohort]["acute"],
                    "n_chronic": stage[cohort]["chronic"],
                    "damage_median_percent": f"{median:.6f}",
                    "damage_q1_percent": f"{q1:.6f}",
                    "damage_q3_percent": f"{q3:.6f}",
                }
            )
    print(f"Wrote cohort summary figure and CSV tables to {output}")


if __name__ == "__main__":
    _style()
    main()
