#!/usr/bin/env python3
"""Region-wise structural markers of PCI dose-response direction.

This is descriptive/statistical, not causal. It asks which regions' structural
damage or preservation is associated with the PCI slope across serotonergic
occupancy levels.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


RESPONDER_LESS_DAMAGE = "#2D5AA7"
RESPONDER_MORE_DAMAGE = "#B8325A"
TEXT = "#1F2933"
GRID = "#D9DEE7"


REGION_NAME_OVERRIDES = {
    "Frontal_Inf_Oper_L": "Left inferior frontal gyrus, opercular part",
    "Frontal_Inf_Oper_R": "Right inferior frontal gyrus, opercular part",
    "Frontal_Inf_Orb_L": "Left inferior frontal gyrus, orbital part",
    "Frontal_Inf_Orb_R": "Right inferior frontal gyrus, orbital part",
    "Frontal_Inf_Tri_L": "Left inferior frontal gyrus, triangular part",
    "Frontal_Inf_Tri_R": "Right inferior frontal gyrus, triangular part",
    "Frontal_Med_Orb_L": "Left medial orbitofrontal cortex",
    "Frontal_Med_Orb_R": "Right medial orbitofrontal cortex",
    "Frontal_Mid_L": "Left middle frontal gyrus",
    "Frontal_Mid_R": "Right middle frontal gyrus",
    "Frontal_Mid_Orb_L": "Left middle orbitofrontal cortex",
    "Frontal_Mid_Orb_R": "Right middle orbitofrontal cortex",
    "Frontal_Sup_L": "Left superior frontal gyrus",
    "Frontal_Sup_Orb_L": "Left superior orbitofrontal cortex",
    "Frontal_Sup_Orb_R": "Right superior orbitofrontal cortex",
    "Heschl_L": "Left Heschl's gyrus",
    "Heschl_R": "Right Heschl's gyrus",
    "Occipital_Mid_R": "Right middle occipital gyrus",
    "Occipital_Sup_R": "Right superior occipital gyrus",
    "Paracentral_Lobule_R": "Right paracentral lobule",
    "Parietal_Inf_L": "Left inferior parietal lobule",
    "Parietal_Inf_R": "Right inferior parietal lobule",
    "Parietal_Sup_L": "Left superior parietal lobule",
    "Parietal_Sup_R": "Right superior parietal lobule",
    "Rolandic_Oper_L": "Left rolandic operculum",
    "Rolandic_Oper_R": "Right rolandic operculum",
    "Temporal_Inf_R": "Right inferior temporal gyrus",
    "Temporal_Pole_Mid_R": "Right temporal pole, middle temporal gyrus",
}


def significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value) or q_value >= 0.05:
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    return "*"


def display_region_name(region_label: str) -> str:
    label = str(region_label)
    if label in REGION_NAME_OVERRIDES:
        return REGION_NAME_OVERRIDES[label]

    side = ""
    core = label
    if label.endswith("_L"):
        side = "Left "
        core = label[:-2]
    elif label.endswith("_R"):
        side = "Right "
        core = label[:-2]

    replacements = {
        "Ant": "anterior",
        "Calcarine": "calcarine cortex",
        "Cingulum": "cingulate",
        "Inf": "inferior",
        "Mid": "middle",
        "Orb": "orbital",
        "Oper": "opercular",
        "ParaHippocampal": "parahippocampal gyrus",
        "Post": "posterior",
        "Precentral": "precentral gyrus",
        "Postcentral": "postcentral gyrus",
        "Rectus": "gyrus rectus",
        "Sup": "superior",
        "Supp": "supplementary",
        "Temporal_Pole": "temporal pole",
        "Tri": "triangular",
    }
    parts = [replacements.get(part, part.lower()) for part in core.split("_")]
    pretty = " ".join(parts).replace("  ", " ").strip()
    return f"{side}{pretty}".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subject-region-csv",
        type=Path,
        default=Path(
            "notebooks/outputs/serotonergic_pci_full_50trials/restartability/extensions/tables/subject_region_damage.csv"
        ),
    )
    p.add_argument(
        "--responder-csv",
        type=Path,
        default=Path(
            "notebooks/outputs/serotonergic_pci_full_50trials/restartability/extensions/tables/subjects_with_strong_positive_responder_label.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/restartability/region_direction"),
    )
    p.add_argument("--top-regions", type=int, default=20)
    return p.parse_args()


def fit_region_models(subject_region: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, df in subject_region.groupby("region_label", observed=True):
        work = df[
            [
                "condition",
                "subject_id",
                "region_index",
                "region_label",
                "system",
                "missing_links_from_region_pct",
                "positive_dose_slope",
            ]
        ].drop_duplicates(["condition", "subject_id", "region_label"])
        work = work.assign(preserved_links_from_region_pct=100.0 - work["missing_links_from_region_pct"])
        try:
            damage_model = smf.ols(
                "positive_dose_slope ~ missing_links_from_region_pct + C(condition)",
                data=work,
            ).fit()
            preserved_model = smf.ols(
                "positive_dose_slope ~ preserved_links_from_region_pct + C(condition)",
                data=work,
            ).fit()
            damage_beta = float(damage_model.params["missing_links_from_region_pct"])
            damage_p = float(damage_model.pvalues["missing_links_from_region_pct"])
            preserved_beta = float(preserved_model.params["preserved_links_from_region_pct"])
            preserved_p = float(preserved_model.pvalues["preserved_links_from_region_pct"])
        except Exception:
            damage_beta = np.nan
            damage_p = np.nan
            preserved_beta = np.nan
            preserved_p = np.nan

        positive = work[work["positive_dose_slope"] > 0.0]
        negative = work[work["positive_dose_slope"] < 0.0]
        rows.append(
            {
                "region_index": int(work["region_index"].iloc[0]),
                "region_label": region,
                "system": str(work["system"].iloc[0]),
                "n_subjects": int(work["subject_id"].nunique()),
                "n_positive_slope": int(positive["subject_id"].nunique()),
                "n_negative_slope": int(negative["subject_id"].nunique()),
                "mean_missing_pct_positive_slope": float(positive["missing_links_from_region_pct"].mean()),
                "mean_missing_pct_negative_slope": float(negative["missing_links_from_region_pct"].mean()),
                "positive_minus_negative_missing_pct": float(
                    positive["missing_links_from_region_pct"].mean()
                    - negative["missing_links_from_region_pct"].mean()
                ),
                "damage_beta_slope_per_1pct_missing": damage_beta,
                "damage_beta_slope_per_10pct_missing": damage_beta * 10.0 if np.isfinite(damage_beta) else np.nan,
                "damage_p": damage_p,
                "preserved_beta_slope_per_1pct_preserved": preserved_beta,
                "preserved_beta_slope_per_10pct_preserved": preserved_beta * 10.0 if np.isfinite(preserved_beta) else np.nan,
                "preserved_p": preserved_p,
            }
        )
    out = pd.DataFrame(rows)
    for p_col, q_col in [("damage_p", "damage_q"), ("preserved_p", "preserved_q")]:
        mask = out[p_col].notna()
        out.loc[mask, q_col] = multipletests(out.loc[mask, p_col], method="fdr_bh")[1]
    return out.sort_values("preserved_beta_slope_per_10pct_preserved", ascending=False)


def summarize_groups(subject_region: pd.DataFrame, responders: pd.DataFrame) -> pd.DataFrame:
    labels = responders[["condition", "subject_id", "positive_dose_response", "strong_positive_responder"]]
    merged = subject_region.merge(labels, on=["condition", "subject_id"], how="left")
    rows = []
    for region, df in merged.groupby("region_label", observed=True):
        strong = df[df["strong_positive_responder"].fillna(False)]
        positive = df[df["positive_dose_response"].fillna(False)]
        negative = df[df["positive_dose_slope"] < 0.0]
        mcs_positive = df[df["condition"].eq("MCS") & df["positive_dose_response"].fillna(False)]
        mcs_negative = df[df["condition"].eq("MCS") & (df["positive_dose_slope"] < 0.0)]
        emcs_strong = df[df["condition"].eq("EMCS") & df["strong_positive_responder"].fillna(False)]
        emcs_other = df[df["condition"].eq("EMCS") & ~df["strong_positive_responder"].fillna(False)]

        strong_vs_negative_p = rank_sum_p(strong["missing_links_from_region_pct"], negative["missing_links_from_region_pct"])
        positive_vs_negative_p = rank_sum_p(positive["missing_links_from_region_pct"], negative["missing_links_from_region_pct"])
        mcs_positive_vs_negative_p = rank_sum_p(mcs_positive["missing_links_from_region_pct"], mcs_negative["missing_links_from_region_pct"])
        emcs_strong_vs_other_p = rank_sum_p(emcs_strong["missing_links_from_region_pct"], emcs_other["missing_links_from_region_pct"])
        rows.append(
            {
                "region_index": int(df["region_index"].iloc[0]),
                "region_label": region,
                "system": str(df["system"].iloc[0]),
                "n_strong_positive": int(strong["subject_id"].nunique()),
                "n_any_positive": int(positive["subject_id"].nunique()),
                "n_negative": int(negative["subject_id"].nunique()),
                "n_mcs_positive": int(mcs_positive["subject_id"].nunique()),
                "n_mcs_negative": int(mcs_negative["subject_id"].nunique()),
                "n_emcs_strong": int(emcs_strong["subject_id"].nunique()),
                "n_emcs_other": int(emcs_other["subject_id"].nunique()),
                "mean_missing_pct_strong_positive": float(strong["missing_links_from_region_pct"].mean()),
                "mean_missing_pct_any_positive": float(positive["missing_links_from_region_pct"].mean()),
                "mean_missing_pct_negative": float(negative["missing_links_from_region_pct"].mean()),
                "mean_missing_pct_mcs_positive": float(mcs_positive["missing_links_from_region_pct"].mean()),
                "mean_missing_pct_mcs_negative": float(mcs_negative["missing_links_from_region_pct"].mean()),
                "mean_missing_pct_emcs_strong": float(emcs_strong["missing_links_from_region_pct"].mean()),
                "mean_missing_pct_emcs_other": float(emcs_other["missing_links_from_region_pct"].mean()),
                "strong_positive_minus_negative_missing_pct": float(
                    strong["missing_links_from_region_pct"].mean()
                    - negative["missing_links_from_region_pct"].mean()
                ),
                "any_positive_minus_negative_missing_pct": float(
                    positive["missing_links_from_region_pct"].mean()
                    - negative["missing_links_from_region_pct"].mean()
                ),
                "mcs_positive_minus_mcs_negative_missing_pct": float(
                    mcs_positive["missing_links_from_region_pct"].mean()
                    - mcs_negative["missing_links_from_region_pct"].mean()
                ),
                "emcs_strong_minus_emcs_other_missing_pct": float(
                    emcs_strong["missing_links_from_region_pct"].mean()
                    - emcs_other["missing_links_from_region_pct"].mean()
                ),
                "strong_positive_vs_negative_p": strong_vs_negative_p,
                "any_positive_vs_negative_p": positive_vs_negative_p,
                "mcs_positive_vs_mcs_negative_p": mcs_positive_vs_negative_p,
                "emcs_strong_vs_emcs_other_p": emcs_strong_vs_other_p,
            }
        )
    out = pd.DataFrame(rows)
    for p_col in [
        "strong_positive_vs_negative_p",
        "any_positive_vs_negative_p",
        "mcs_positive_vs_mcs_negative_p",
        "emcs_strong_vs_emcs_other_p",
    ]:
        q_col = p_col.removesuffix("_p") + "_q"
        mask = out[p_col].notna()
        out.loc[mask, q_col] = multipletests(out.loc[mask, p_col], method="fdr_bh")[1]
    return out


def rank_sum_p(left: pd.Series, right: pd.Series) -> float:
    left_values = pd.to_numeric(left, errors="coerce").dropna().to_numpy(float)
    right_values = pd.to_numeric(right, errors="coerce").dropna().to_numpy(float)
    if len(left_values) < 2 or len(right_values) < 2:
        return np.nan
    return float(mannwhitneyu(left_values, right_values, alternative="two-sided").pvalue)


def plot_damage_associations(models: pd.DataFrame, out_path: Path, top_regions: int) -> None:
    selected = pd.concat(
        [
            models.nlargest(top_regions // 2, "damage_beta_slope_per_10pct_missing"),
            models.nsmallest(top_regions // 2, "damage_beta_slope_per_10pct_missing"),
        ],
        ignore_index=True,
    ).drop_duplicates("region_label")
    selected = selected.sort_values("damage_beta_slope_per_10pct_missing")

    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.27 * len(selected) + 1.5)), constrained_layout=True)
    values = selected["damage_beta_slope_per_10pct_missing"]
    colors = [RESPONDER_MORE_DAMAGE if value > 0 else RESPONDER_LESS_DAMAGE for value in values]
    ax.barh([display_region_name(x) for x in selected["region_label"]], values, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Change in PCI dose-response slope per 10% more missing regional links")
    ax.set_ylabel("AAL90 region")
    ax.set_title("Region damage associated with PCI dose-response slope")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_group_damage(
    group_summary: pd.DataFrame,
    out_path: Path,
    top_regions: int,
    column: str,
    q_column: str,
    title: str,
    xlabel: str,
) -> None:
    selected = group_summary.reindex(
        group_summary[column].abs().sort_values(ascending=False).index
    ).head(top_regions)
    selected = selected.sort_values(column)

    fig, ax = plt.subplots(figsize=(7.2, max(4.7, 0.24 * len(selected) + 1.25)), constrained_layout=True)
    values = selected[column]
    colors = [RESPONDER_LESS_DAMAGE if value < 0 else RESPONDER_MORE_DAMAGE for value in values]
    labels = [display_region_name(x) for x in selected["region_label"]]
    ax.barh(labels, values, color=colors, alpha=0.94, height=0.68)
    x_range = max(float(np.nanmax(np.abs(values))) * 0.16, 1.0)
    for y, (_, row) in enumerate(selected.iterrows()):
        stars = significance_stars(float(row[q_column]))
        if not stars:
            continue
        value = float(row[column])
        x = value + x_range if value >= 0 else value - x_range
        ha = "left" if value >= 0 else "right"
        ax.text(x, y, stars, va="center", ha=ha, fontsize=8.0, fontweight="bold", color=TEXT)

    n_significant = int((group_summary[q_column] < 0.05).sum())
    subtitle = "Asterisks mark FDR-corrected q < 0.05" if n_significant else "No regions survived FDR correction"
    ax.axvline(0.0, color=TEXT, linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AAL90 region")
    ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, color=TEXT, pad=10)
    ax.tick_params(axis="both", labelsize=7.0, colors=TEXT, width=0.6)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
    ax.legend(
        handles=[
            Patch(color=RESPONDER_LESS_DAMAGE, label="Less damage in strong positive responders"),
            Patch(color=RESPONDER_MORE_DAMAGE, label="More damage in strong positive responders"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=6.8,
        handlelength=1.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.8)
    fig.savefig(out_path, dpi=400)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_region = pd.read_csv(args.subject_region_csv)
    responders = pd.read_csv(args.responder_csv)
    models = fit_region_models(subject_region)
    group_summary = summarize_groups(subject_region, responders)

    models.to_csv(tables_dir / "region_pci_slope_association_models.csv", index=False)
    group_summary.to_csv(tables_dir / "region_damage_positive_vs_negative_response.csv", index=False)
    plot_group_damage(
        group_summary,
        figures_dir / "region_damage_strong_positive_vs_negative_response_starred.png",
        args.top_regions,
        "strong_positive_minus_negative_missing_pct",
        "strong_positive_vs_negative_q",
        "Structural damage by PCI dose-response direction\nstrong positive responders (n=6) vs subjects whose PCI decreased (n=132)",
        "Missing links in strong positive responders minus PCI-decrease subjects (%)",
    )

    print(f"Wrote tables to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")
    print("Regions where missing links track a more positive PCI slope after condition adjustment:")
    cols = ["region_label", "system", "damage_beta_slope_per_10pct_missing", "damage_q"]
    print(models.nlargest(8, "damage_beta_slope_per_10pct_missing")[cols].to_string(index=False))
    print("Regions where that association is weakest:")
    print(models.nsmallest(8, "damage_beta_slope_per_10pct_missing")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
