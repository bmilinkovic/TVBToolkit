#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "src").exists() and (root.parent / "src").exists():
        root = root.parent
    return root


PROJECT_ROOT = resolve_project_root()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TVB_USER_HOME", str(PROJECT_ROOT / ".tvb-temp"))

from tvbtoolkit.datasets.brain_act import list_subjects, load_subject_structural
from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results


DATASET_ROOT = doc_liege_raw("doc_data", "converted_structural")
DEFAULT_OUTPUT_DIR = doc_liege_results("structural_zero_edges")

COHORT_ORDER = ["control", "emcs", "mcs", "uws", "coma"]
COHORT_LABELS = {
    "control": "Control",
    "emcs": "EMCS",
    "mcs": "MCS",
    "uws": "UWS",
    "coma": "Coma",
}
COHORT_COLORS = {
    "control": "#5B8A72",
    "emcs": "#E8B56D",
    "mcs": "#C5622F",
    "uws": "#8B6B8B",
    "coma": "#3B4A6B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the percentage of zero structural edges per BrainAct participant, "
            "grouped by condition."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.0,
        help="Count an edge as damaged when its weight is <= this threshold (default: 0.0).",
    )
    parser.add_argument(
        "--include-diagonal",
        action="store_true",
        help="Include the matrix diagonal in the edge count. By default only unique off-diagonal edges are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for x-jitter of participant points.",
    )
    return parser.parse_args()


def extract_edges(connectivity: np.ndarray, include_diagonal: bool) -> np.ndarray:
    k = 0 if include_diagonal else 1
    tri_upper = np.triu_indices_from(connectivity, k=k)
    return np.asarray(connectivity[tri_upper], dtype=float)


def compute_zero_edge_percentage(
    connectivity: np.ndarray,
    *,
    zero_threshold: float,
    include_diagonal: bool,
) -> tuple[float, int, int]:
    edges = extract_edges(connectivity, include_diagonal=include_diagonal)
    n_total = int(edges.size)
    n_zero = int(np.count_nonzero(edges <= zero_threshold))
    pct_zero = 100.0 * n_zero / n_total if n_total else float("nan")
    return pct_zero, n_zero, n_total


def build_dataframe(
    *,
    dataset_root: Path,
    zero_threshold: float,
    include_diagonal: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in COHORT_ORDER:
        subject_ids = list_subjects(dataset_root=dataset_root, cohort=cohort)
        for subject_id in subject_ids:
            connectivity, _, _, meta = load_subject_structural(
                subject_id=subject_id,
                dataset_root=dataset_root,
                cohort=cohort,
            )
            pct_zero, n_zero, n_total = compute_zero_edge_percentage(
                connectivity,
                zero_threshold=zero_threshold,
                include_diagonal=include_diagonal,
            )
            rows.append(
                {
                    "cohort": cohort,
                    "condition": COHORT_LABELS[cohort],
                    "subject_id": subject_id,
                    "stage": meta.stage,
                    "sedation": meta.sedation,
                    "n_zero_edges": n_zero,
                    "n_total_edges": n_total,
                    "pct_zero_edges": pct_zero,
                }
            )
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(
        df["condition"],
        categories=[COHORT_LABELS[c] for c in COHORT_ORDER],
        ordered=True,
    )
    return df.sort_values(["condition", "subject_id"]).reset_index(drop=True)


def plot_dataframe(df: pd.DataFrame, output_path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(10.8, 5.6), constrained_layout=True)

    data = [
        df.loc[df["cohort"] == cohort, "pct_zero_edges"].to_numpy(dtype=float)
        for cohort in COHORT_ORDER
    ]
    positions = np.arange(1, len(COHORT_ORDER) + 1)

    violin = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, cohort in zip(violin["bodies"], COHORT_ORDER, strict=True):
        body.set_facecolor(COHORT_COLORS[cohort])
        body.set_edgecolor(COHORT_COLORS[cohort])
        body.set_alpha(0.28)

    for xpos, cohort in zip(positions, COHORT_ORDER, strict=True):
        vals = df.loc[df["cohort"] == cohort, "pct_zero_edges"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.14, 0.14, size=vals.size)
        ax.scatter(
            np.full(vals.shape, xpos, dtype=float) + jitter,
            vals,
            s=24,
            alpha=0.82,
            color=COHORT_COLORS[cohort],
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        if vals.size:
            median = float(np.median(vals))
            ax.hlines(
                median,
                xpos - 0.19,
                xpos + 0.19,
                colors="black",
                linewidth=1.1,
                zorder=4,
            )

    ax.set_xticks(positions, [COHORT_LABELS[c] for c in COHORT_ORDER])
    ax.set_xlabel("Condition", fontsize=16, fontweight="bold")
    ax.set_ylabel("Zero structural edges (%)", fontsize=16, fontweight="bold")
    ax.set_title("Structural connectome edge damage across DOC conditions", fontsize=17, fontweight="bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(
        dataset_root=args.dataset_root.resolve(),
        zero_threshold=args.zero_threshold,
        include_diagonal=args.include_diagonal,
    )

    csv_path = output_dir / "brain_act_structural_zero_edges_by_subject.csv"
    png_path = output_dir / "brain_act_structural_zero_edges_by_condition.png"
    pdf_path = output_dir / "brain_act_structural_zero_edges_by_condition.pdf"
    summary_path = output_dir / "brain_act_structural_zero_edges_summary.csv"

    df.to_csv(csv_path, index=False)
    plot_dataframe(df, png_path, seed=args.seed)
    plot_dataframe(df, pdf_path, seed=args.seed)

    summary = (
        df.groupby(["cohort", "condition"], observed=True)["pct_zero_edges"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved subject-level data to: {csv_path}")
    print(f"Saved figure to: {png_path}")
    print(f"Saved figure to: {pdf_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
