#!/usr/bin/env python3
"""Exploratory restartability extensions using structural damage location."""

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
}
RESPONDER_COLORS = {
    "Strong positive responders": "#2F6F5E",
    "Other DoC subjects": "#8A8F98",
}
SYSTEM_ORDER = ["Frontal", "Parietal", "Temporal", "Occipital", "Limbic", "Subcortical"]


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
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/restartability/extensions"),
    )
    p.add_argument(
        "--affected-threshold-pct",
        type=float,
        default=5.0,
        help="A region is called affected when at least this percent of its links are missing.",
    )
    p.add_argument("--top-regions", type=int, default=25)
    return p.parse_args()


def simple_subject_label(condition: str, subject_id: str) -> str:
    tail = str(subject_id)[1:]
    number = str(int(tail)) if tail.isdigit() else str(subject_id)
    return f"{condition} {number}"


def region_system(label: str) -> str:
    label = str(label)
    if any(key in label for key in ["Caudate", "Putamen", "Pallidum", "Thalamus"]):
        return "Subcortical"
    if any(key in label for key in ["Hippocampus", "ParaHippocampal", "Amygdala", "Cingulum", "Insula"]):
        return "Limbic"
    if any(key in label for key in ["Occipital", "Calcarine", "Cuneus", "Lingual", "Fusiform"]):
        return "Occipital"
    if any(key in label for key in ["Temporal", "Heschl"]):
        return "Temporal"
    if any(key in label for key in ["Parietal", "Precuneus", "Angular", "SupraMarginal", "Postcentral"]):
        return "Parietal"
    if any(key in label for key in ["Frontal", "Precentral", "Supp_Motor", "Rolandic", "Olfactory", "Rectus", "Paracentral"]):
        return "Frontal"
    return "Other"


def directional_responder_labels(rankings: pd.DataFrame) -> pd.DataFrame:
    out = rankings.copy()
    positive = out["positive_dose_slope"] > 0.0
    if not bool(positive.any()):
        raise ValueError("No subjects have a positive dose-response slope.")
    threshold = float(out.loc[positive, "positive_dose_slope"].quantile(0.75))
    strong_positive = positive & (out["positive_dose_slope"] >= threshold)
    out["positive_dose_response"] = positive
    out["strong_positive_responder"] = strong_positive
    out["responder_group"] = np.where(
        strong_positive,
        "Strong positive responders",
        "Other DoC subjects",
    )
    out["positive_slope_threshold"] = 0.0
    out["strong_positive_slope_threshold"] = threshold
    out["subject_label"] = [
        simple_subject_label(condition, subject_id)
        for condition, subject_id in zip(out["condition"], out["subject_id"], strict=True)
    ]
    return out


def region_missing_link_percent(connectivity: np.ndarray) -> np.ndarray:
    matrix = np.asarray(connectivity, dtype=float).copy()
    np.fill_diagonal(matrix, np.nan)
    return np.nanmean(matrix <= 0.0, axis=1) * 100.0


def upper_edges_for_mask(mask: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mask, k=1)
    return np.asarray(mask[iu], dtype=bool)


def build_tables(rankings: pd.DataFrame, dataset_root: Path, affected_threshold_pct: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = load_aal90_atlas(dataset_root)
    region_labels = [str(x) for x in atlas.labels]
    systems = [region_system(x) for x in region_labels]

    subject_region_rows = []
    network_rows = []
    for row in rankings.itertuples(index=False):
        connectivity, _, _, _ = load_subject_structural(
            subject_id=str(row.subject_id),
            cohort=str(row.cohort),
            dataset_root=dataset_root,
        )
        missing = np.asarray(connectivity <= 0.0, dtype=bool)
        np.fill_diagonal(missing, False)
        missing_pct = region_missing_link_percent(connectivity)

        for region_index, (label, system, value) in enumerate(zip(region_labels, systems, missing_pct, strict=True), start=1):
            subject_region_rows.append(
                {
                    "cohort": row.cohort,
                    "condition": row.condition,
                    "subject_id": row.subject_id,
                    "subject_label": row.subject_label,
                    "responder_group": row.responder_group,
                    "positive_dose_slope": row.positive_dose_slope,
                    "max_delta_pci": row.max_delta_pci,
                    "restartability_score": row.restartability_score,
                    "region_index": region_index,
                    "region_label": label,
                    "system": system,
                    "missing_links_from_region_pct": float(value),
                    "region_affected": bool(value >= affected_threshold_pct),
                    "affected_threshold_pct": float(affected_threshold_pct),
                }
            )

        network_rows.extend(candidate_network_rows(row, missing, systems))

    return pd.DataFrame(subject_region_rows), pd.DataFrame(network_rows), pd.DataFrame(
        {"region_index": np.arange(1, len(region_labels) + 1), "region_label": region_labels, "system": systems}
    )


def pair_mask(systems: list[str], left: set[str], right: set[str] | None = None) -> np.ndarray:
    arr = np.asarray(systems, dtype=object)
    if right is None:
        idx = np.isin(arr, list(left))
        return np.outer(idx, idx)
    left_idx = np.isin(arr, list(left))
    right_idx = np.isin(arr, list(right))
    return np.outer(left_idx, right_idx) | np.outer(right_idx, left_idx)


def candidate_network_rows(row, missing: np.ndarray, systems: list[str]) -> list[dict[str, object]]:
    network_defs = {
        "Thalamo-cortical links": pair_mask(systems, {"Subcortical"}, {"Frontal", "Parietal", "Temporal", "Occipital", "Limbic"}),
        "Fronto-parietal links": pair_mask(systems, {"Frontal"}, {"Parietal"}),
        "Sensory-association links": pair_mask(systems, {"Occipital", "Temporal", "Parietal"}),
        "Limbic-cortical links": pair_mask(systems, {"Limbic"}, {"Frontal", "Parietal", "Temporal", "Occipital"}),
    }
    rows = []
    for network, mask in network_defs.items():
        mask = np.asarray(mask, dtype=bool)
        np.fill_diagonal(mask, False)
        candidate = upper_edges_for_mask(mask)
        damaged = upper_edges_for_mask(missing & mask)
        n_links = int(np.count_nonzero(candidate))
        n_missing = int(np.count_nonzero(damaged))
        rows.append(
            {
                "cohort": row.cohort,
                "condition": row.condition,
                "subject_id": row.subject_id,
                "subject_label": row.subject_label,
                "responder_group": row.responder_group,
                "positive_dose_slope": row.positive_dose_slope,
                "max_delta_pci": row.max_delta_pci,
                "restartability_score": row.restartability_score,
                "network": network,
                "n_candidate_links": n_links,
                "n_missing_links": n_missing,
                "missing_links_pct": float(100.0 * n_missing / max(n_links, 1)),
                "preserved_links_pct": float(100.0 * (n_links - n_missing) / max(n_links, 1)),
            }
        )
    return rows


def shared_region_summary(subject_region: pd.DataFrame) -> pd.DataFrame:
    return (
        subject_region.groupby(["responder_group", "region_index", "region_label", "system"], observed=True)
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_subjects_region_affected=("region_affected", "sum"),
            mean_missing_links_from_region_pct=("missing_links_from_region_pct", "mean"),
        )
        .reset_index()
        .assign(
            pct_subjects_region_affected=lambda d: 100.0 * d["n_subjects_region_affected"] / d["n_subjects"],
        )
        .sort_values(["responder_group", "pct_subjects_region_affected"], ascending=[True, False])
    )


def system_summary(subject_region: pd.DataFrame) -> pd.DataFrame:
    subject_system = (
        subject_region.groupby(["condition", "subject_id", "subject_label", "responder_group", "system"], observed=True)
        .agg(mean_missing_links_pct=("missing_links_from_region_pct", "mean"))
        .reset_index()
    )
    summary = (
        subject_system.groupby(["responder_group", "system"], observed=True)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_missing_links_pct=("mean_missing_links_pct", "mean"),
            median_missing_links_pct=("mean_missing_links_pct", "median"),
        )
        .reset_index()
    )
    return subject_system, summary


def network_summary(networks: pd.DataFrame) -> pd.DataFrame:
    return (
        networks.groupby(["responder_group", "network"], observed=True)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_preserved_links_pct=("preserved_links_pct", "mean"),
            median_preserved_links_pct=("preserved_links_pct", "median"),
            mean_missing_links_pct=("missing_links_pct", "mean"),
        )
        .reset_index()
    )


def plot_region_comparison(summary: pd.DataFrame, out_path: Path, top_regions: int) -> None:
    wide = summary.pivot_table(
        index=["region_label", "system"],
        columns="responder_group",
        values="pct_subjects_region_affected",
        observed=True,
    ).fillna(0.0)
    wide["Strong-positive-minus-other"] = wide["Strong positive responders"] - wide["Other DoC subjects"]
    wide = wide.reindex(wide["Strong-positive-minus-other"].abs().sort_values(ascending=False).index).head(top_regions)
    wide = wide.sort_values("Strong-positive-minus-other")
    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.25 * len(wide) + 1.6)), constrained_layout=True)
    colors = ["#2F6F5E" if x > 0 else "#8A8F98" for x in wide["Strong-positive-minus-other"]]
    labels = [idx[0] for idx in wide.index]
    ax.barh(labels, wide["Strong-positive-minus-other"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Strong positive responders minus other subjects: affected-region frequency (percentage points)")
    ax.set_ylabel("AAL90 region")
    ax.set_title("Regions more often affected in strong positive responders or in other subjects")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_top_responder_regions(summary: pd.DataFrame, out_path: Path, top_regions: int) -> None:
    top = summary[summary["responder_group"].eq("Strong positive responders")].nlargest(top_regions, "pct_subjects_region_affected")
    top = top.sort_values("pct_subjects_region_affected")
    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.24 * len(top) + 1.4)), constrained_layout=True)
    ax.barh(top["region_label"], top["pct_subjects_region_affected"], color="#2F6F5E", alpha=0.9)
    for y, value in enumerate(top["pct_subjects_region_affected"]):
        ax.text(value + 1.0, y, f"{value:.0f}%", va="center", fontsize=7.3, color="#333333")
    ax.set_xlim(0.0, min(105.0, float(top["pct_subjects_region_affected"].max()) + 12.0))
    ax.set_xlabel("Strong positive responders where this region is affected (%)")
    ax.set_ylabel("AAL90 region")
    ax.set_title("Shared affected regions among strong positive responders")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_system_summary(subject_system: pd.DataFrame, out_path: Path) -> None:
    systems = [s for s in SYSTEM_ORDER if s in set(subject_system["system"])]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    positions = np.arange(len(systems), dtype=float)
    width = 0.34
    for offset, group in [(-width / 2, "Strong positive responders"), (width / 2, "Other DoC subjects")]:
        data = [subject_system.loc[subject_system["responder_group"].eq(group) & subject_system["system"].eq(system), "mean_missing_links_pct"].to_numpy(float) for system in systems]
        means = [float(np.mean(x)) if len(x) else np.nan for x in data]
        sems = [float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0 for x in data]
        ax.bar(positions + offset, means, width=width, yerr=sems, capsize=2.5, color=RESPONDER_COLORS[group], alpha=0.88, label=group)
    ax.set_xticks(positions, systems, rotation=25, ha="right")
    ax.set_ylabel("Mean missing links from regions in system (%)")
    ax.set_title("Damage burden summarized by broad brain system")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_network_sparing(networks: pd.DataFrame, out_path: Path) -> None:
    network_names = list(dict.fromkeys(networks["network"]))
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    positions = np.arange(len(network_names), dtype=float)
    width = 0.34
    for offset, group in [(-width / 2, "Strong positive responders"), (width / 2, "Other DoC subjects")]:
        data = [networks.loc[networks["responder_group"].eq(group) & networks["network"].eq(network), "preserved_links_pct"].to_numpy(float) for network in network_names]
        means = [float(np.mean(x)) if len(x) else np.nan for x in data]
        sems = [float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0 for x in data]
        ax.bar(positions + offset, means, width=width, yerr=sems, capsize=2.5, color=RESPONDER_COLORS[group], alpha=0.88, label=group)
    ax.set_xticks(positions, network_names, rotation=25, ha="right")
    ax.set_ylabel("Preserved links (%)")
    ax.set_title("Candidate network sparing")
    ax.set_ylim(0.0, 105.0)
    ax.legend(frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mpl.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rankings = directional_responder_labels(pd.read_csv(args.ranking_csv))
    subject_region, networks, atlas_systems = build_tables(rankings, args.dataset_root, args.affected_threshold_pct)
    region_summary = shared_region_summary(subject_region)
    subject_system, system_sum = system_summary(subject_region)
    network_sum = network_summary(networks)

    rankings.to_csv(tables_dir / "subjects_with_strong_positive_responder_label.csv", index=False)
    atlas_systems.to_csv(tables_dir / "aal90_region_system_map.csv", index=False)
    subject_region.to_csv(tables_dir / "subject_region_damage.csv", index=False)
    region_summary.to_csv(tables_dir / "responder_region_shared_damage_summary.csv", index=False)
    subject_system.to_csv(tables_dir / "subject_system_damage.csv", index=False)
    system_sum.to_csv(tables_dir / "responder_system_damage_summary.csv", index=False)
    networks.to_csv(tables_dir / "subject_candidate_network_sparing.csv", index=False)
    network_sum.to_csv(tables_dir / "responder_candidate_network_sparing_summary.csv", index=False)

    plot_top_responder_regions(region_summary, figures_dir / "strong_positive_shared_affected_regions.png", args.top_regions)
    plot_region_comparison(region_summary, figures_dir / "strong_positive_vs_other_region_frequency.png", args.top_regions)
    plot_system_summary(subject_system, figures_dir / "strong_positive_vs_other_system_damage.png")
    plot_network_sparing(networks, figures_dir / "strong_positive_vs_other_network_sparing.png")

    threshold = float(rankings["strong_positive_slope_threshold"].iloc[0])
    n_positive = int(rankings["positive_dose_response"].sum())
    n_strong = int(rankings["responder_group"].eq("Strong positive responders").sum())
    print("Directional responder definition: positive slope first, then top quartile among positive slopes.")
    print(f"Positive dose-response subjects: {n_positive}/{rankings.shape[0]}")
    print(f"Strong positive responder threshold: positive_dose_slope >= {threshold:.6f}")
    print(f"Strong positive responders: {n_strong}/{rankings.shape[0]}")
    print(f"Wrote tables to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
