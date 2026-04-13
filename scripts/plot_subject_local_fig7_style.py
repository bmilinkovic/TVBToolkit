#!/usr/bin/env python3
"""Create Fig7-style plot for subject-local k=5 analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


COHORTS = ("control", "emcs", "mcs", "uws")
PALETTE = {
    "control": "#2E86AB",
    "emcs": "#4DAF4A",
    "mcs": "#E67E22",
    "uws": "#C0392B",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F9FC",
            "axes.edgecolor": "#2A2A2A",
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.dpi": 220,
        }
    )


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _bh_fdr(pvals: list[float], alpha: float) -> list[tuple[float, bool]]:
    p = np.array([1.0 if (x is None or not np.isfinite(x)) else float(x) for x in pvals], dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    p_adj_rank = np.empty(m, dtype=float)
    cmin = 1.0
    for i in range(m - 1, -1, -1):
        cmin = min(cmin, ranked[i] * m / float(i + 1))
        p_adj_rank[i] = cmin
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = np.clip(p_adj_rank, 0.0, 1.0)
    sig = p_adj < float(alpha)
    return [(float(pa), bool(ss)) for pa, ss in zip(p_adj, sig)]


def _sig_stars(p_adj: float, alpha: float = 0.05) -> str:
    if (not np.isfinite(p_adj)) or p_adj >= alpha:
        return ""
    if p_adj < 1e-4:
        return "****"
    if p_adj < 1e-3:
        return "***"
    if p_adj < 1e-2:
        return "**"
    return "*"


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.input_table).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    fig_dir = out_root / "figs"
    tab_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    needed = {"subject_id", "cohort", "state_rank", "occupancy", args.x_col}
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {in_path}: {sorted(missing)}")

    df = df[df["cohort"].isin(COHORTS)].copy()
    df["state_rank"] = df["state_rank"].astype(int)
    n_states = int(df["state_rank"].max())
    if n_states <= 0:
        raise RuntimeError("No states found in input table.")

    _set_style()
    fig = plt.figure(figsize=(3.7 * (n_states + 1), 5.3), dpi=300)
    gs = fig.add_gridspec(1, n_states + 1, wspace=0.28)

    rng = np.random.default_rng(42)
    x_positions = np.arange(len(COHORTS), dtype=float)
    p_rows: list[dict[str, Any]] = []

    ctrl_idx = int(COHORTS.index("control"))
    tgt_idx = [i for i in range(len(COHORTS)) if i != ctrl_idx]

    for rank in range(1, n_states + 1):
        ax = fig.add_subplot(gs[0, rank - 1])
        dt = df[df["state_rank"] == rank]
        arrays = [dt.loc[dt["cohort"] == c, "occupancy"].to_numpy(dtype=float) for c in COHORTS]

        vp = ax.violinplot(
            arrays,
            positions=x_positions,
            widths=0.62,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body, cohort in zip(vp["bodies"], COHORTS):
            body.set_facecolor(PALETTE[cohort])
            body.set_alpha(0.50)
            body.set_edgecolor(PALETTE[cohort])
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in vp:
                vp[key].set_colors("black")
                vp[key].set_linewidths(1.0)

        for xi, arr, cohort in zip(x_positions, arrays, COHORTS):
            if arr.size == 0:
                continue
            jx = xi + rng.uniform(-0.085, 0.085, size=arr.size)
            ax.scatter(jx, arr, s=18, color=PALETTE[cohort], alpha=0.72, edgecolor="none", zorder=3)

        raw_p = []
        pairs = []
        for j in tgt_idx:
            if arrays[ctrl_idx].size == 0 or arrays[j].size == 0:
                raw_p.append(np.nan)
            else:
                _, p = ttest_ind(arrays[ctrl_idx], arrays[j], equal_var=False, nan_policy="omit")
                raw_p.append(float(p))
            pairs.append((ctrl_idx, j))
        adj = _bh_fdr(raw_p, alpha=float(args.alpha_fdr))

        ymax = float(max([arr.max() if arr.size else 0.0 for arr in arrays] + [0.0]))
        y_star = min(1.18, ymax + 0.08)
        for (i, j), (pa, sig) in zip(pairs, adj):
            label = _sig_stars(pa, alpha=float(args.alpha_fdr))
            if label:
                ax.text(
                    x_positions[j],
                    y_star,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="bold",
                    color="black",
                )
            p_rows.append(
                {
                    "state_rank": int(rank),
                    "comparison": f"{COHORTS[i]} vs {COHORTS[j]}",
                    "p_raw": float(raw_p[tgt_idx.index(j)]) if np.isfinite(raw_p[tgt_idx.index(j)]) else np.nan,
                    "p_adj_bh": float(pa),
                    "significant_0.05": bool(sig),
                }
            )

        ax.set_xticks(x_positions, [c.upper() for c in COHORTS], rotation=25, ha="right")
        ax.set_ylim(0.0, 1.24)
        ax.set_title(f"Pattern {rank}")
        ax.set_ylabel("Occupancy rate" if rank == 1 else "")
        ax.grid(alpha=0.20)

    ax = fig.add_subplot(gs[0, n_states])
    x_all: list[float] = []
    y_all: list[float] = []
    point_rows: list[dict[str, Any]] = []

    for cohort in COHORTS:
        dt = df[df["cohort"] == cohort]
        x = dt[args.x_col].to_numpy(dtype=float)
        y = dt["occupancy"].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size == 0:
            continue
        x_all.extend(x.tolist())
        y_all.extend(y.tolist())
        ax.scatter(x, y, color=PALETTE[cohort], alpha=0.42, s=18, label=cohort.upper(), edgecolors="none")
        if x.size >= 2 and float(np.ptp(x)) > 0.0:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            yy = p[0] * xx + p[1]
            valid = yy >= 0.0
            if np.any(valid):
                ax.plot(xx[valid], yy[valid], color=PALETTE[cohort], linewidth=2.2, alpha=0.9)

        for row in dt.loc[good, ["subject_id", "cohort", "state_rank", args.x_col, "occupancy"]].itertuples(index=False):
            point_rows.append(
                {
                    "subject_id": row[0],
                    "cohort": row[1],
                    "state_rank": int(row[2]),
                    "sfc": float(row[3]),
                    "occupancy": float(row[4]),
                }
            )

    xa = np.asarray(x_all, dtype=float)
    ya = np.asarray(y_all, dtype=float)
    if ya.size:
        ymax = float(np.max(ya))
        ax.set_ylim(0.0, min(1.0, ymax + 0.05 * max(ymax, 1e-6)))
    else:
        ax.set_ylim(0.0, 1.0)
    if xa.size:
        x_margin = 0.05 * max(float(np.ptp(xa)), 1e-6)
        ax.set_xlim(float(np.min(xa)) - x_margin, float(np.max(xa)) + x_margin)
    ax.set_xlabel(args.x_label)
    ax.set_ylabel("")
    ax.set_title("SFC vs occupancy")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    _save_figure(fig, fig_dir, args.figure_stem)
    pd.DataFrame(p_rows).to_csv(tab_dir / f"{args.figure_stem}_pairwise_pvalues.tsv", sep="\t", index=False)
    pd.DataFrame(point_rows).to_csv(tab_dir / f"{args.figure_stem}_subject_points.csv", index=False)

    print(f"Saved figure: {fig_dir / (args.figure_stem + '.pdf')}")
    print(f"Saved p-values: {tab_dir / (args.figure_stem + '_pairwise_pvalues.tsv')}")
    print(f"Saved points: {tab_dir / (args.figure_stem + '_subject_points.csv')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-table",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/tables/rank_aligned_state_metrics_long.csv",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/subject_local_fig7_style",
    )
    p.add_argument("--x-col", type=str, default="sfc", choices=["sfc", "sfc_full"])
    p.add_argument("--x-label", type=str, default="SC-FC coupling (subject-specific SC)")
    p.add_argument("--alpha-fdr", type=float, default=0.05)
    p.add_argument("--figure-stem", type=str, default="fig7_subject_local_k5_distributions_and_sfc")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
