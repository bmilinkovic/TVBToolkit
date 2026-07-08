#!/usr/bin/env python3
"""Create a publication-style figure for the serotonergic PCI pilot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
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
        "--input",
        type=Path,
        default=Path("results/serotonergic_pci_3per_condition_10trials/tables/serotonergic_pci_subject_metrics_with_rescue.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serotonergic_pci_3per_condition_10trials/figures/publishable"),
    )
    p.add_argument("--prefix", default="serotonergic_pci_rescue_publishable")
    return p.parse_args()


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def mean_sem_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (condition, occupancy), g in df.groupby(["condition", "occupancy"], observed=True):
        pci = g["pci_mean"].to_numpy(float)
        rescue = g["pci_rescue"].to_numpy(float)
        rows.append(
            {
                "condition": str(condition),
                "occupancy": float(occupancy),
                "n_subjects": int(g["subject_id"].nunique()),
                "pci_mean": float(np.mean(pci)),
                "pci_sem": sem(pci),
                "pci_rescue_mean": float(np.mean(rescue)),
                "pci_rescue_sem": sem(rescue),
            }
        )
    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(out["condition"], categories=CONDITION_ORDER, ordered=True)
    return out.sort_values(["condition", "occupancy"]).reset_index(drop=True)


def draw_mean_sem(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, color: str, label: str | None = None) -> None:
    ax.plot(x, y, color=color, linewidth=2.2, marker="o", markersize=4.8, label=label, zorder=4)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.16, linewidth=0, zorder=2)


def strip_points(ax: mpl.axes.Axes, x: float, values: np.ndarray, color: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.055, 0.055, size=len(values))
    ax.scatter(
        np.full(len(values), x) + jitter,
        values,
        s=30,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.92,
        zorder=5,
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def build_figure(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.09, top=0.93, wspace=0.32, hspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    ax_baseline, ax_pci, ax_rescue, ax_max = axes

    x_occ = np.array(sorted(df["occupancy"].unique()), dtype=float)
    x_pos = np.arange(len(CONDITION_ORDER), dtype=float)

    baseline = df[df["occupancy"].eq(0.0)].copy()
    for xi, condition in zip(x_pos, CONDITION_ORDER, strict=True):
        vals = baseline.loc[baseline["condition"].eq(condition), "pci_mean"].to_numpy(float)
        color = COND_COLORS[condition]
        strip_points(ax_baseline, float(xi), vals, color, seed=int(xi) + 1)
        mean = float(np.mean(vals))
        err = sem(vals)
        ax_baseline.errorbar(xi, mean, yerr=err, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=6)
        ax_baseline.scatter([xi], [mean], s=58, marker="_", color="black", linewidth=1.8, zorder=7)

    ax_baseline.set_xticks(x_pos)
    ax_baseline.set_xticklabels(CONDITION_ORDER, rotation=30, ha="right")
    ax_baseline.set_ylabel("PCI")
    ax_baseline.set_title("Baseline perturbational complexity")
    ax_baseline.set_ylim(0.50, 0.96)
    ax_baseline.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    panel_label(ax_baseline, "A")

    for condition in CONDITION_ORDER:
        color = COND_COLORS[condition]
        g = df[df["condition"].eq(condition)]
        for _, sg in g.groupby("subject_id", observed=True):
            ax_pci.plot(sg["occupancy"], sg["pci_mean"], color=color, alpha=0.22, linewidth=1.0, zorder=1)
        sm = summary[summary["condition"].eq(condition)]
        draw_mean_sem(
            ax_pci,
            sm["occupancy"].to_numpy(float),
            sm["pci_mean"].to_numpy(float),
            sm["pci_sem"].to_numpy(float),
            color,
            condition,
        )

    ax_pci.set_xlabel(r"5-HT$_{2A}$ occupancy")
    ax_pci.set_ylabel("PCI")
    ax_pci.set_title("Dose-response of PCI")
    ax_pci.set_xlim(-0.03, 0.81)
    ax_pci.set_ylim(0.50, 0.96)
    ax_pci.set_xticks(x_occ)
    ax_pci.grid(color="#D8D8D8", linewidth=0.6, alpha=0.65)
    ax_pci.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    panel_label(ax_pci, "B")

    for condition in CONDITION_ORDER:
        color = COND_COLORS[condition]
        g = df[df["condition"].eq(condition)]
        for _, sg in g.groupby("subject_id", observed=True):
            sg_nz = sg[sg["occupancy"].gt(0.0)]
            ax_rescue.plot(sg_nz["occupancy"], sg_nz["pci_rescue"], color=color, alpha=0.22, linewidth=1.0, zorder=1)
        sm = summary[(summary["condition"].eq(condition)) & (summary["occupancy"].gt(0.0))]
        draw_mean_sem(
            ax_rescue,
            sm["occupancy"].to_numpy(float),
            sm["pci_rescue_mean"].to_numpy(float),
            sm["pci_rescue_sem"].to_numpy(float),
            color,
        )

    ax_rescue.axhline(0.0, color="#222222", linewidth=0.9)
    ax_rescue.set_xlabel(r"5-HT$_{2A}$ occupancy")
    ax_rescue.set_ylabel(r"$\Delta$PCI from baseline")
    ax_rescue.set_title("PCI rescue curve")
    ax_rescue.set_xlim(0.21, 0.81)
    ax_rescue.set_ylim(-0.09, 0.105)
    ax_rescue.set_xticks(x_occ[x_occ > 0.0])
    ax_rescue.grid(color="#D8D8D8", linewidth=0.6, alpha=0.65)
    panel_label(ax_rescue, "C")

    max_occ = float(np.max(x_occ))
    max_df = df[df["occupancy"].eq(max_occ)].copy()
    for xi, condition in zip(x_pos, CONDITION_ORDER, strict=True):
        color = COND_COLORS[condition]
        vals = max_df.loc[max_df["condition"].eq(condition), "pci_rescue"].to_numpy(float)
        strip_points(ax_max, float(xi), vals, color, seed=int(xi) + 101)
        mean = float(np.mean(vals))
        err = sem(vals)
        ax_max.errorbar(xi, mean, yerr=err, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=6)
        ax_max.scatter([xi], [mean], s=58, marker="_", color="black", linewidth=1.8, zorder=7)

    ax_max.axhline(0.0, color="#222222", linewidth=0.9)
    ax_max.set_xticks(x_pos)
    ax_max.set_xticklabels(CONDITION_ORDER, rotation=30, ha="right")
    ax_max.set_ylabel(r"$\Delta$PCI at 0.766 occupancy")
    ax_max.set_title("Max-dose subject effects")
    ax_max.set_ylim(-0.09, 0.105)
    ax_max.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    panel_label(ax_max, "D")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.7)

    fig.suptitle(r"Simulated 5-HT$_{2A}$ modulation of perturbational complexity", fontsize=11, y=0.985)
    fig.text(
        0.08,
        0.018,
        "Points are subjects (n=3 per condition); lines/shaded bands show mean +/- SEM across subjects. Each subject PCI is estimated from 10 perturbation trials.",
        fontsize=6.8,
        color="#333333",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 600), ("pdf", 600), ("svg", 600)]:
        fig.savefig(out_dir / f"{prefix}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    df = df.sort_values(["condition", "subject_id", "occupancy"]).reset_index(drop=True)

    summary = mean_sem_table(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / f"{args.prefix}_summary.csv", index=False)
    build_figure(df, summary, args.output_dir, args.prefix)
    print(f"Wrote {args.output_dir / (args.prefix + '.pdf')}")
    print(f"Wrote {args.output_dir / (args.prefix + '_summary.csv')}")


if __name__ == "__main__":
    main()
