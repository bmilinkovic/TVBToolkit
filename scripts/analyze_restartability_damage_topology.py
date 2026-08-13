#!/usr/bin/env python3
"""Explore whether damage topology differs between high and low restartability subjects."""

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


COND_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
    "EMCS": "#E8B56D",
    "CNT": "#5B8A72",
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
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/restartability/damage_topology"),
    )
    p.add_argument("--top-n", type=int, default=18)
    p.add_argument("--edge-top-n", type=int, default=40)
    return p.parse_args()


def subject_label(condition: str, subject_id: str) -> str:
    tail = str(subject_id)[1:]
    number = str(int(tail)) if tail.isdigit() else str(subject_id)
    return f"{condition} {number}"


def zero_edge_mask(connectivity: np.ndarray) -> np.ndarray:
    c = np.asarray(connectivity, dtype=float).copy()
    np.fill_diagonal(c, np.nan)
    return c <= 0.0


def load_damage_masks(subjects: pd.DataFrame, dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    atlas = load_aal90_atlas(dataset_root)
    labels = [str(x) for x in atlas.labels]
    masks = []
    node_damage = []
    for row in subjects.itertuples(index=False):
        connectivity, _, _, _ = load_subject_structural(
            subject_id=str(row.subject_id),
            cohort=str(row.cohort),
            dataset_root=dataset_root,
        )
        mask = zero_edge_mask(connectivity)
        masks.append(mask)
        node_damage.append(np.nanmean(mask, axis=1) * 100.0)
    return np.asarray(masks, dtype=float), np.asarray(node_damage, dtype=float), labels


def select_groups(rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rankings = rankings.sort_values("rank_by_restartability_doc").reset_index(drop=True)
    top = rankings.head(top_n).copy()
    bottom = rankings.tail(top_n).copy()
    top["restartability_group"] = "Top restartability"
    bottom["restartability_group"] = "Bottom restartability"
    return pd.concat([top, bottom], ignore_index=True)


def regional_table(grouped: pd.DataFrame, node_damage: np.ndarray, region_labels: list[str]) -> pd.DataFrame:
    rows = []
    for group in ["Top restartability", "Bottom restartability"]:
        idx = np.where(grouped["restartability_group"].to_numpy() == group)[0]
        values = node_damage[idx]
        means = np.nanmean(values, axis=0)
        consistency = np.nanmean(values > 0.0, axis=0) * 100.0
        for region_index, label in enumerate(region_labels):
            rows.append(
                {
                    "restartability_group": group,
                    "region_index": region_index + 1,
                    "region_label": label,
                    "mean_incident_zero_edges_pct": float(means[region_index]),
                    "subjects_with_any_incident_damage_pct": float(consistency[region_index]),
                }
            )
    table = pd.DataFrame(rows)
    wide = table.pivot(index=["region_index", "region_label"], columns="restartability_group", values="mean_incident_zero_edges_pct")
    wide = wide.reset_index()
    wide["bottom_minus_top_mean_incident_zero_edges_pct"] = wide["Bottom restartability"] - wide["Top restartability"]
    wide["top_minus_bottom_mean_incident_zero_edges_pct"] = wide["Top restartability"] - wide["Bottom restartability"]
    consistency_wide = table.pivot(
        index=["region_index", "region_label"],
        columns="restartability_group",
        values="subjects_with_any_incident_damage_pct",
    ).reset_index()
    consistency_wide["bottom_minus_top_subject_consistency_pct"] = (
        consistency_wide["Bottom restartability"] - consistency_wide["Top restartability"]
    )
    consistency_wide["top_minus_bottom_subject_consistency_pct"] = (
        consistency_wide["Top restartability"] - consistency_wide["Bottom restartability"]
    )
    return wide.merge(
        consistency_wide[
            [
                "region_index",
                "region_label",
                "Top restartability",
                "Bottom restartability",
                "bottom_minus_top_subject_consistency_pct",
                "top_minus_bottom_subject_consistency_pct",
            ]
        ],
        on=["region_index", "region_label"],
        suffixes=("_mean_damage_pct", "_subject_consistency_pct"),
    ).sort_values("top_minus_bottom_mean_incident_zero_edges_pct", ascending=False)


def edge_table(grouped: pd.DataFrame, masks: np.ndarray, region_labels: list[str], edge_top_n: int) -> pd.DataFrame:
    group_values = {}
    for group in ["Top restartability", "Bottom restartability"]:
        idx = np.where(grouped["restartability_group"].to_numpy() == group)[0]
        group_values[group] = np.nanmean(masks[idx], axis=0) * 100.0
    rows = []
    iu = np.triu_indices(len(region_labels), k=1)
    for i, j in zip(*iu, strict=True):
        top = float(group_values["Top restartability"][i, j])
        bottom = float(group_values["Bottom restartability"][i, j])
        rows.append(
            {
                "region_i": i + 1,
                "region_j": j + 1,
                "region_i_label": region_labels[i],
                "region_j_label": region_labels[j],
                "top_damaged_subjects_pct": top,
                "bottom_damaged_subjects_pct": bottom,
                "bottom_minus_top_damaged_subjects_pct": bottom - top,
                "top_minus_bottom_damaged_subjects_pct": top - bottom,
            }
        )
    out = pd.DataFrame(rows).sort_values("top_minus_bottom_damaged_subjects_pct", ascending=False)
    return out.head(edge_top_n).reset_index(drop=True)


def plot_regional_difference(regional: pd.DataFrame, out_path: Path, n_regions: int = 25) -> None:
    plot_df = regional.reindex(regional["top_minus_bottom_mean_incident_zero_edges_pct"].abs().sort_values(ascending=False).index).head(n_regions)
    plot_df = plot_df.sort_values("top_minus_bottom_mean_incident_zero_edges_pct")
    colors = ["#E8B56D" if x > 0 else "#3B4A6B" for x in plot_df["top_minus_bottom_mean_incident_zero_edges_pct"]]
    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.24 * len(plot_df) + 1.4)), constrained_layout=True)
    ax.barh(plot_df["region_label"], plot_df["top_minus_bottom_mean_incident_zero_edges_pct"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Top minus bottom mean incident zero-edge damage (%)")
    ax.set_ylabel("AAL90 region")
    ax.set_title("Regional damage topology: top versus bottom restartability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_subject_region_heatmap(grouped: pd.DataFrame, node_damage: np.ndarray, regional: pd.DataFrame, out_path: Path, n_regions: int = 25) -> None:
    selected = regional.reindex(regional["top_minus_bottom_mean_incident_zero_edges_pct"].abs().sort_values(ascending=False).index).head(n_regions)
    region_indices = (selected["region_index"].to_numpy(int) - 1).tolist()
    order = grouped.sort_values(["restartability_group", "rank_by_restartability_doc"], ascending=[False, True]).index.to_numpy()
    matrix = node_damage[order][:, region_indices]
    ylabels = [subject_label(str(grouped.loc[i, "condition"]), str(grouped.loc[i, "subject_id"])) for i in order]
    xlabels = selected["region_label"].tolist()
    fig, ax = plt.subplots(figsize=(10.2, max(5.2, 0.20 * len(order) + 1.8)), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_title("Subject-by-region structural damage among top and bottom restartability groups")
    ax.set_xlabel("AAL90 region")
    ax.set_ylabel("Subject")
    boundary = int(np.sum(grouped.loc[order, "restartability_group"].eq("Top restartability")))
    ax.axhline(boundary - 0.5, color="white", linewidth=1.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Incident zero-edge damage (%)")
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
    grouped = select_groups(rankings, args.top_n)
    masks, node_damage, region_labels = load_damage_masks(grouped, args.dataset_root)

    grouped.to_csv(tables_dir / "top_bottom_subjects.csv", index=False)
    regional = regional_table(grouped, node_damage, region_labels)
    regional.to_csv(tables_dir / "regional_damage_top_bottom.csv", index=False)
    edges = edge_table(grouped, masks, region_labels, args.edge_top_n)
    edges.to_csv(tables_dir / "edge_damage_top_enriched.csv", index=False)

    plot_regional_difference(regional, figures_dir / "regional_damage_bottom_minus_top.png")
    plot_subject_region_heatmap(grouped, node_damage, regional, figures_dir / "subject_region_damage_heatmap.png")

    print(f"Wrote regional topology table to {tables_dir / 'regional_damage_top_bottom.csv'}")
    print(f"Wrote edge topology table to {tables_dir / 'edge_damage_top_enriched.csv'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
