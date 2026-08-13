#!/usr/bin/env python3
"""Summarize brain regions that are structurally affected across subjects."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tvbtoolkit.datasets.brain_act import load_aal90_atlas, load_subject_structural


CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS"]
COND_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
    "EMCS": "#E8B56D",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ranking-csv",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/restartability/tables/doc_subject_restartability_rankings.csv"),
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/ex_data/cnrs/data_doc_liege/raw/doc_data/converted_structural"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/restartability/shared_damage_regions"),
    )
    p.add_argument(
        "--affected-threshold-pct",
        type=float,
        default=5.0,
        help="A region is called affected when at least this percent of its links are zero.",
    )
    p.add_argument("--top-regions", type=int, default=30)
    return p.parse_args()


def simple_subject_label(condition: str, subject_id: str) -> str:
    tail = str(subject_id)[1:]
    number = str(int(tail)) if tail.isdigit() else str(subject_id)
    return f"{condition} {number}"


def region_link_damage_percent(connectivity: np.ndarray) -> np.ndarray:
    matrix = np.asarray(connectivity, dtype=float).copy()
    np.fill_diagonal(matrix, np.nan)
    return np.nanmean(matrix <= 0.0, axis=1) * 100.0


def build_subject_region_table(rankings: pd.DataFrame, dataset_root: Path, affected_threshold_pct: float) -> pd.DataFrame:
    atlas = load_aal90_atlas(dataset_root)
    region_labels = [str(x) for x in atlas.labels]
    rows = []
    for row in rankings.itertuples(index=False):
        connectivity, _, _, _ = load_subject_structural(
            subject_id=str(row.subject_id),
            cohort=str(row.cohort),
            dataset_root=dataset_root,
        )
        damage_pct = region_link_damage_percent(connectivity)
        for region_index, (region_label, value) in enumerate(zip(region_labels, damage_pct, strict=True), start=1):
            rows.append(
                {
                    "cohort": row.cohort,
                    "condition": row.condition,
                    "subject_id": row.subject_id,
                    "subject_label": simple_subject_label(str(row.condition), str(row.subject_id)),
                    "rank_by_restartability_doc": row.rank_by_restartability_doc,
                    "restartability_score": row.restartability_score,
                    "region_index": region_index,
                    "region_label": region_label,
                    "damaged_links_from_region_pct": float(value),
                    "region_affected": bool(value >= affected_threshold_pct),
                    "affected_threshold_pct": float(affected_threshold_pct),
                }
            )
    return pd.DataFrame(rows)


def shared_region_summary(subject_region: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("All DoC", subject_region)]
    for condition in CONDITION_ORDER:
        groups.append((condition, subject_region[subject_region["condition"].eq(condition)]))
    top35_subjects = (
        subject_region[["condition", "subject_id", "rank_by_restartability_doc"]]
        .drop_duplicates()
        .nsmallest(35, "rank_by_restartability_doc")[["condition", "subject_id"]]
    )
    top35_keys = set(zip(top35_subjects["condition"], top35_subjects["subject_id"], strict=True))
    groups.append(
        (
            "Top 35 restartability",
            subject_region[
                [key in top35_keys for key in zip(subject_region["condition"], subject_region["subject_id"], strict=True)]
            ],
        )
    )

    for group_name, group_df in groups:
        n_subjects = int(group_df[["condition", "subject_id"]].drop_duplicates().shape[0])
        if n_subjects == 0:
            continue
        summary = (
            group_df.groupby(["region_index", "region_label"], observed=True)
            .agg(
                n_subjects=("subject_id", "nunique"),
                n_subjects_region_affected=("region_affected", "sum"),
                mean_damaged_links_from_region_pct=("damaged_links_from_region_pct", "mean"),
                median_damaged_links_from_region_pct=("damaged_links_from_region_pct", "median"),
            )
            .reset_index()
        )
        summary["group"] = group_name
        summary["pct_subjects_region_affected"] = 100.0 * summary["n_subjects_region_affected"] / n_subjects
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(["group", "pct_subjects_region_affected"], ascending=[True, False])


def plot_shared_regions(summary: pd.DataFrame, out_path: Path, group: str, top_regions: int) -> None:
    group_df = summary[summary["group"].eq(group)].nlargest(top_regions, "pct_subjects_region_affected")
    group_df = group_df.sort_values("pct_subjects_region_affected")
    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.24 * len(group_df) + 1.4)), constrained_layout=True)
    ax.barh(group_df["region_label"], group_df["pct_subjects_region_affected"], color="#6C7A89", alpha=0.9)
    for y, value, mean_damage in zip(
        np.arange(len(group_df)),
        group_df["pct_subjects_region_affected"],
        group_df["mean_damaged_links_from_region_pct"],
        strict=True,
    ):
        ax.text(value + 1.0, y, f"{value:.0f}% subjects", va="center", fontsize=7.2, color="#333333")
    ax.set_xlim(0, min(105, max(20, float(group_df["pct_subjects_region_affected"].max()) + 12)))
    ax.set_xlabel("Subjects where this region is affected (%)")
    ax.set_ylabel("AAL90 region")
    ax.set_title(f"Most shared affected regions: {group}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_subject_region_heatmap(subject_region: pd.DataFrame, summary: pd.DataFrame, out_path: Path, group: str, top_regions: int) -> None:
    region_order = (
        summary[summary["group"].eq(group)]
        .nlargest(top_regions, "pct_subjects_region_affected")
        .sort_values("pct_subjects_region_affected", ascending=False)["region_label"]
        .tolist()
    )
    subjects = (
        subject_region[["condition", "subject_id", "subject_label", "rank_by_restartability_doc"]]
        .drop_duplicates()
        .sort_values("rank_by_restartability_doc")
    )
    plot_df = subject_region[subject_region["region_label"].isin(region_order)].copy()
    matrix = plot_df.pivot_table(
        index="subject_label",
        columns="region_label",
        values="damaged_links_from_region_pct",
        observed=True,
    ).reindex(index=subjects["subject_label"], columns=region_order)
    fig, ax = plt.subplots(figsize=(11.5, max(7.0, 0.105 * matrix.shape[0] + 2.0)), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(float), aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index, fontsize=5.6)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
    ax.set_xlabel("Most commonly affected regions")
    ax.set_ylabel("Subject, sorted by restartability rank")
    ax.set_title("Which affected regions are shared across subjects?")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("Damaged links from region (%)")
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mpl.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rankings = pd.read_csv(args.ranking_csv)
    subject_region = build_subject_region_table(rankings, args.dataset_root, args.affected_threshold_pct)
    summary = shared_region_summary(subject_region)

    subject_region.to_csv(tables_dir / "subject_region_damage_percent.csv", index=False)
    summary.to_csv(tables_dir / "shared_affected_regions_summary.csv", index=False)

    plot_shared_regions(summary, figures_dir / "shared_affected_regions_all_doc.png", "All DoC", args.top_regions)
    plot_shared_regions(summary, figures_dir / "shared_affected_regions_top35_restartability.png", "Top 35 restartability", args.top_regions)
    plot_subject_region_heatmap(subject_region, summary, figures_dir / "subject_by_shared_region_damage_heatmap.png", "All DoC", args.top_regions)

    print(f"Wrote subject-region table to {tables_dir / 'subject_region_damage_percent.csv'}")
    print(f"Wrote shared-region summary to {tables_dir / 'shared_affected_regions_summary.csv'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
