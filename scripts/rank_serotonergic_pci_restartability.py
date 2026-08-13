#!/usr/bin/env python3
"""Rank subjects by serotonergic PCI restartability and optional structural damage."""

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


CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
COHORT_TO_CONDITION = {
    "coma": "COMA",
    "uws": "UWS",
    "mcs": "MCS",
    "emcs": "EMCS",
    "control": "CNT",
}
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
        "--pci-csv",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/tables/serotonergic_pci_subject_metrics_with_rescue.csv"),
    )
    p.add_argument(
        "--damage-csv",
        type=Path,
        default=None,
        help="Optional structural damage CSV with cohort/subject_id and pct_zero_edges columns.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional converted structural dataset root. Used to compute pct_zero_edges when --damage-csv is absent.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/restartability"),
    )
    p.add_argument("--top-n", type=int, default=30)
    return p.parse_args()


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.mean())) / sd


def subject_response_table(pci_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    max_occ = float(pci_df["occupancy"].max())
    cnt_baseline = float(pci_df.loc[pci_df["condition"].eq("CNT") & pci_df["occupancy"].eq(0.0), "pci_mean"].mean())
    for (cohort, condition, subject_id), g in pci_df.groupby(["cohort", "condition", "subject_id"], observed=True):
        g = g.sort_values("occupancy")
        baseline = float(g.loc[g["occupancy"].eq(0.0), "pci_mean"].iloc[0])
        max_row = g.loc[g["occupancy"].eq(max_occ)].iloc[0]
        positive = g[g["occupancy"].gt(0.0)]
        x = positive["occupancy"].to_numpy(float)
        y = positive["pci_rescue"].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) >= 2 else np.nan
        auc = float(np.trapz(y, x) / (x.max() - x.min())) if x.size >= 2 and x.max() > x.min() else np.nan
        wake_gap = cnt_baseline - baseline
        wake_gap_closure = float(max_row["pci_rescue"] / wake_gap) if wake_gap > 1e-12 else np.nan
        rows.append(
            {
                "cohort": cohort,
                "condition": condition,
                "subject_id": subject_id,
                "subject_key": f"{cohort}:{subject_id}",
                "baseline_pci": baseline,
                "max_occupancy": max_occ,
                "max_dose_pci": float(max_row["pci_mean"]),
                "max_delta_pci": float(max_row["pci_rescue"]),
                "positive_dose_slope": slope,
                "positive_dose_auc_delta": auc,
                "control_baseline_mean": cnt_baseline,
                "wake_gap_to_control": wake_gap,
                "wake_gap_closure_fraction": wake_gap_closure,
            }
        )
    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(out["condition"], categories=CONDITION_ORDER, ordered=True)
    doc_mask = ~out["condition"].eq("CNT")
    out["response_z_slope_doc"] = np.nan
    out["response_z_max_delta_doc"] = np.nan
    out["response_z_wake_gap_closure_doc"] = np.nan
    out.loc[doc_mask, "response_z_slope_doc"] = zscore(out.loc[doc_mask, "positive_dose_slope"])
    out.loc[doc_mask, "response_z_max_delta_doc"] = zscore(out.loc[doc_mask, "max_delta_pci"])
    out.loc[doc_mask, "response_z_wake_gap_closure_doc"] = zscore(out.loc[doc_mask, "wake_gap_closure_fraction"].fillna(0.0))
    out["restartability_score"] = out[["response_z_slope_doc", "response_z_max_delta_doc", "response_z_wake_gap_closure_doc"]].mean(axis=1)
    out["rank_by_slope_all"] = out["positive_dose_slope"].rank(ascending=False, method="min").astype(int)
    out["rank_by_max_delta_all"] = out["max_delta_pci"].rank(ascending=False, method="min").astype(int)
    out["rank_by_restartability_doc"] = np.nan
    out.loc[doc_mask, "rank_by_restartability_doc"] = out.loc[doc_mask, "restartability_score"].rank(ascending=False, method="min")
    return out.sort_values(["rank_by_restartability_doc", "rank_by_slope_all", "condition", "subject_id"]).reset_index(drop=True)


def compute_damage_from_dataset(dataset_root: Path) -> pd.DataFrame:
    from tvbtoolkit.datasets.brain_act import list_subjects, load_subject_structural

    rows = []
    for cohort, condition in COHORT_TO_CONDITION.items():
        for subject_id in list_subjects(dataset_root=dataset_root, cohort=cohort):
            connectivity, _, _, meta = load_subject_structural(subject_id=subject_id, dataset_root=dataset_root, cohort=cohort)
            iu = np.triu_indices_from(connectivity, k=1)
            edges = np.asarray(connectivity[iu], dtype=float)
            n_zero = int(np.count_nonzero(edges <= 0.0))
            rows.append(
                {
                    "cohort": meta.cohort,
                    "condition": condition,
                    "subject_id": subject_id,
                    "n_zero_edges": n_zero,
                    "n_total_edges": int(edges.size),
                    "pct_zero_edges": float(100.0 * n_zero / max(edges.size, 1)),
                }
            )
    return pd.DataFrame(rows)


def load_damage(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.damage_csv is not None:
        damage = pd.read_csv(args.damage_csv)
    elif args.dataset_root is not None:
        damage = compute_damage_from_dataset(args.dataset_root)
    else:
        return None

    damage = damage.copy()
    if "condition" not in damage and "cohort" in damage:
        damage["condition"] = damage["cohort"].map(COHORT_TO_CONDITION)
    required = {"cohort", "subject_id", "pct_zero_edges"}
    missing = required.difference(damage.columns)
    if missing:
        raise ValueError(f"Damage table is missing required columns: {sorted(missing)}")
    return damage


def merge_damage(response: pd.DataFrame, damage: pd.DataFrame | None) -> pd.DataFrame:
    if damage is None:
        return response
    keep_cols = [c for c in ["cohort", "condition", "subject_id", "n_zero_edges", "n_total_edges", "pct_zero_edges"] if c in damage.columns]
    damage_small = damage[keep_cols].copy()
    return response.merge(damage_small, on=["cohort", "subject_id"], how="left", suffixes=("", "_damage"))


def plot_top_ranked(df: pd.DataFrame, out_path: Path, top_n: int) -> None:
    doc = df[~df["condition"].eq("CNT")].dropna(subset=["rank_by_restartability_doc"]).copy()
    top = doc.nsmallest(top_n, "rank_by_restartability_doc").sort_values("restartability_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(4.6, 0.22 * len(top) + 1.6)), constrained_layout=True)
    colors = [COND_COLORS[str(c)] for c in top["condition"]]
    labels = [
        f"{condition} {int(str(subject_id)[1:]) if str(subject_id)[1:].isdigit() else subject_id}"
        for condition, subject_id in zip(top["condition"].astype(str), top["subject_id"].astype(str), strict=True)
    ]
    bars = ax.barh(labels, top["restartability_score"], color=colors, alpha=0.88)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Restartability score (DOC z-composite)")
    ax.set_ylabel("Subject")
    ax.set_title(f"Top {len(top)} simulated PCI restartability candidates")
    max_score = float(top["restartability_score"].max()) if len(top) else 0.0
    for bar, damage in zip(bars, top.get("pct_zero_edges", pd.Series(np.nan, index=top.index)), strict=True):
        score = float(bar.get_width())
        label = "damage n/a" if not np.isfinite(damage) else f"{float(damage):.1f}% damage"
        if score > max_score * 0.22:
            x = score - max_score * 0.025
            ha = "right"
            color = "white"
        else:
            x = score + max_score * 0.025
            ha = "left"
            color = "#333333"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2.0,
            label,
            va="center",
            ha=ha,
            fontsize=7.2,
            color=color,
            fontweight="bold" if color == "white" else "normal",
        )
    ax.set_xlim(0.0, max_score * 1.10 if max_score > 0 else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_damage_response(df: pd.DataFrame, out_path: Path) -> None:
    if "pct_zero_edges" not in df:
        return
    doc = df[~df["condition"].eq("CNT")].dropna(subset=["pct_zero_edges", "positive_dose_slope"]).copy()
    if doc.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True, sharey=True)
    for ax, title, xlim in [
        (axes[0], "Full range", None),
        (axes[1], "Zoom: <=30% zero edges", (0.0, 30.0)),
    ]:
        for condition in CONDITION_ORDER:
            if condition == "CNT":
                continue
            g = doc[doc["condition"].eq(condition)]
            ax.scatter(
                g["pct_zero_edges"],
                g["positive_dose_slope"],
                s=38,
                color=COND_COLORS[condition],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                label=condition,
            )
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xlabel("Structural damage: zero SC edges (%)")
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="#D8D8D8", linewidth=0.6, alpha=0.65)
    axes[0].set_ylabel(r"Subject $\Delta$PCI dose slope")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Structural damage versus serotonergic PCI response", fontsize=12)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mpl.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    out_dir = args.output_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pci = pd.read_csv(args.pci_csv)
    response = subject_response_table(pci)
    damage = load_damage(args)
    if damage is not None:
        damage.to_csv(tables_dir / "structural_damage_by_subject.csv", index=False)
    ranked = merge_damage(response, damage)
    ranked.to_csv(tables_dir / "subject_restartability_rankings.csv", index=False)
    ranked[~ranked["condition"].eq("CNT")].to_csv(tables_dir / "doc_subject_restartability_rankings.csv", index=False)

    summary_cols = {
        "n_subjects": ("subject_id", "nunique"),
        "baseline_pci_mean": ("baseline_pci", "mean"),
        "slope_mean": ("positive_dose_slope", "mean"),
        "slope_sem": ("positive_dose_slope", sem),
        "max_delta_mean": ("max_delta_pci", "mean"),
        "max_delta_sem": ("max_delta_pci", sem),
        "wake_gap_closure_mean": ("wake_gap_closure_fraction", "mean"),
    }
    if "pct_zero_edges" in ranked:
        summary_cols["pct_zero_edges_mean"] = ("pct_zero_edges", "mean")
        summary_cols["pct_zero_edges_sem"] = ("pct_zero_edges", sem)
    condition_summary = ranked.groupby("condition", observed=True).agg(**summary_cols).reset_index()
    condition_summary.to_csv(tables_dir / "condition_restartability_summary.csv", index=False)

    plot_top_ranked(ranked, figures_dir / "top_restartability_subjects.png", top_n=args.top_n)
    plot_damage_response(ranked, figures_dir / "damage_vs_pci_response.png")

    print(f"Wrote rankings to {tables_dir / 'subject_restartability_rankings.csv'}")
    print(f"Wrote DOC rankings to {tables_dir / 'doc_subject_restartability_rankings.csv'}")
    if damage is None:
        print("No structural damage table was provided, so the damage-response figure was not created.")


if __name__ == "__main__":
    main()
