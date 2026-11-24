"""Exploratory, simulation-specific PCI cutoff for behavioural MCS vs UWS.

This is NOT the empirical Casarotto PCI*=0.31, which was calibrated against
subjective reports using maximum empirical TMS-EEG PCI across sites. Here the
only available labels are behavioural MCS/UWS. The cutoff is selected using
zero-occupancy subject means, never the 100 trials as independent people.
Repeated subject-stratified cross-validation estimates out-of-sample group
discrimination; applying the full-data cutoff to its training cohort is
descriptive only. No simulation or PCI mask is recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = ("CNT", "EMCS", "MCS", "UWS")
COLORS = {"CNT": "#9a83c7", "EMCS": "#b578c4", "MCS": "#9554a8", "UWS": "#593878"}
EMPIRICAL_MCS_SENSITIVITY = 36 / 38


def read_baseline(path: Path) -> dict[str, list[dict]]:
    by_group = {group: [] for group in GROUPS}
    seen = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["occupancy"]) != 0.0:
                continue
            group, subject = row["group"], row["subject"]
            if group not in by_group or (group, subject) in seen:
                raise ValueError(f"Invalid or duplicate baseline row: {group}/{subject}")
            seen.add((group, subject))
            row["pci"] = float(row["mean_pci_lz"])
            if not np.isfinite(row["pci"]):
                raise ValueError(f"Nonfinite PCI: {group}/{subject}")
            by_group[group].append(row)
    if any(not rows for rows in by_group.values()):
        raise ValueError("Expected all four cohorts at zero occupancy")
    return by_group


def read_clinical_annotations(path: Path | None) -> dict[str, str]:
    """Optional user-supplied pseudonymised annotations; never infer them."""
    if path is None:
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"subject", "evidence"}.issubset(rows[0]):
        raise ValueError("Clinical CSV needs subject and evidence columns")
    annotations = {row["subject"]: row["evidence"] for row in rows}
    if len(annotations) != len(rows):
        raise ValueError("Duplicate subjects in clinical annotations")
    return annotations


def classify(mcs: np.ndarray, uws: np.ndarray, threshold: float) -> dict:
    """Positive means PCI >= threshold, with MCS as the positive label."""
    tp, fn = int(np.sum(mcs >= threshold)), int(np.sum(mcs < threshold))
    fp, tn = int(np.sum(uws >= threshold)), int(np.sum(uws < threshold))
    sensitivity, specificity = tp / len(mcs), tn / len(uws)
    return dict(threshold=float(threshold), mcs_above=tp, mcs_below=fn,
                uws_above=fp, uws_below=tn,
                mcs_sensitivity=sensitivity, uws_specificity=specificity,
                balanced_accuracy=(sensitivity + specificity) / 2,
                youden_j=sensitivity + specificity - 1)


def choose_youden(mcs: np.ndarray, uws: np.ndarray) -> dict:
    """Choose highest J, then higher specificity, then lower cutoff on ties."""
    candidates = np.unique(np.r_[mcs, uws, np.nextafter(max(mcs.max(), uws.max()), np.inf)])
    scores = [classify(mcs, uws, t) for t in candidates]
    return max(scores, key=lambda r: (r["youden_j"], r["uws_specificity"],
                                      -r["threshold"]))


def closest_empirical_sensitivity(mcs: np.ndarray, uws: np.ndarray) -> dict:
    """Highest threshold retaining at least 36/38 MCS sensitivity."""
    candidates = np.unique(mcs)
    eligible = [classify(mcs, uws, t) for t in candidates
                if np.mean(mcs >= t) >= EMPIRICAL_MCS_SENSITIVITY]
    return max(eligible, key=lambda r: r["threshold"])


def auc(mcs: np.ndarray, uws: np.ndarray) -> float:
    comparison = mcs[:, None] - uws[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def cross_validate(mcs: np.ndarray, uws: np.ndarray,
                   repeats: int, seed: int) -> dict:
    """Refit on four folds, evaluate on one; split by subject, not trial."""
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        folds_m = np.array_split(rng.permutation(len(mcs)), 5)
        folds_u = np.array_split(rng.permutation(len(uws)), 5)
        for fold in range(5):
            train_m = np.concatenate([part for i, part in enumerate(folds_m) if i != fold])
            train_u = np.concatenate([part for i, part in enumerate(folds_u) if i != fold])
            threshold = choose_youden(mcs[train_m], uws[train_u])["threshold"]
            result = classify(mcs[folds_m[fold]], uws[folds_u[fold]], threshold)
            rows.append(dict(repeat=repeat, fold=fold, **result))
    return dict(folds=rows,
                pooled_mcs_sensitivity=sum(row["mcs_above"] for row in rows) /
                sum(row["mcs_above"] + row["mcs_below"] for row in rows),
                pooled_uws_specificity=sum(row["uws_below"] for row in rows) /
                sum(row["uws_above"] + row["uws_below"] for row in rows),
                mean_fold_balanced_accuracy=float(np.mean(
                    [row["balanced_accuracy"] for row in rows])))


def bootstrap_threshold(mcs: np.ndarray, uws: np.ndarray,
                        draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    thresholds = np.array([
        choose_youden(rng.choice(mcs, len(mcs), replace=True),
                      rng.choice(uws, len(uws), replace=True))["threshold"]
        for _ in range(draws)])
    return dict(draws=draws, percentile_2_5=float(np.percentile(thresholds, 2.5)),
                median=float(np.median(thresholds)),
                percentile_97_5=float(np.percentile(thresholds, 97.5)))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_groups(by_group: dict, threshold: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    values = [[row["pci"] for row in by_group[group]] for group in GROUPS]
    violins = ax.violinplot(values, positions=np.arange(4),
                            showmeans=False, showmedians=True, showextrema=False)
    for body, group in zip(violins["bodies"], GROUPS):
        body.set_facecolor(COLORS[group])
        body.set_edgecolor("none")
        body.set_alpha(0.40)
    violins["cmedians"].set_color("#28212d")
    violins["cmedians"].set_linewidth(1.5)
    rng = np.random.default_rng(27042)
    for i, (group, group_values) in enumerate(zip(GROUPS, values)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(group_values)), group_values,
                   s=15, color=COLORS[group], alpha=0.55, linewidths=0)
    ax.axhline(threshold, color="#25212a", linestyle="--", lw=1.7,
               label=f"Exploratory model cutoff: {threshold:.3f}")
    ax.set(xticks=np.arange(4), xticklabels=GROUPS, ylabel="PCI",
           xlim=(-0.48, 3.48))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "baseline_cutoff.pdf", bbox_inches="tight")
    fig.savefig(output / "baseline_cutoff.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=27042)
    parser.add_argument("--clinical-annotations", type=Path,
                        help="Optional local CSV with subject,evidence; never upload patient details to public GitHub")
    args = parser.parse_args()
    if args.cv_repeats < 1 or args.bootstrap_draws < 1:
        parser.error("cv-repeats and bootstrap-draws must be positive")
    by_group = read_baseline(args.input_csv)
    annotations = read_clinical_annotations(args.clinical_annotations)
    mcs = np.array([row["pci"] for row in by_group["MCS"]])
    uws = np.array([row["pci"] for row in by_group["UWS"]])
    selected = choose_youden(mcs, uws)
    benchmark = closest_empirical_sensitivity(mcs, uws)
    cv = cross_validate(mcs, uws, args.cv_repeats, args.seed)
    boot = bootstrap_threshold(mcs, uws, args.bootstrap_draws, args.seed + 1)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(definition="Exploratory discrimination of behavioural MCS vs UWS, not a consciousness threshold",
                   source_csv_sha256=hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
                   n_by_group={group: len(rows) for group, rows in by_group.items()},
                   baseline_occupancy=0.0, score="per-subject mean of 100 trial PCI-LZ values",
                   positive_label="MCS", decision_rule="PCI >= cutoff",
                   selected_youden=selected, auc_mcs_vs_uws=auc(mcs, uws),
                   cross_validation={k: v for k, v in cv.items() if k != "folds"},
                   bootstrap_threshold=boot,
                   sensitivity_matched_to_empirical_mcs_36_of_38=benchmark,
                   reference_cutoffs={str(t): classify(mcs, uws, t) for t in (.28, .31, .44)},
                   group_counts_at_selected={group: sum(row["pci"] >= selected["threshold"]
                                                        for row in rows)
                                             for group, rows in by_group.items()})
    if annotations:
        known = {row["subject"] for rows in by_group.values() for row in rows}
        if not set(annotations).issubset(known):
            raise ValueError("Clinical CSV contains subjects absent from baseline scores")
        summary["clinical_annotation_overlap"] = dict(
            annotation_csv_sha256=hashlib.sha256(args.clinical_annotations.read_bytes()).hexdigest(),
            annotation_count=len(annotations),
            annotated_uws_above_model_cutoff=[row["subject"] for row in by_group["UWS"]
                                              if row["subject"] in annotations and
                                              row["pci"] >= selected["threshold"]],
            caution="Heterogeneous evidence of potential covert consciousness, not ground-truth labels")
    (args.output / "cutoff_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output / "cross_validation_folds.csv", cv["folds"])
    subjects = [dict(group=group, subject=row["subject"],
                     pci=row["pci"], b_pa=float(row["b_pa"]),
                     high_rate_trials=int(row["high_rate_trials"]),
                     above_model_cutoff=row["pci"] >= selected["threshold"],
                     above_0p28=row["pci"] >= .28,
                     above_0p31=row["pci"] >= .31,
                     clinical_evidence=annotations.get(row["subject"], ""))
                for group, rows in by_group.items() for row in rows]
    write_csv(args.output / "subject_classification.csv", subjects)
    plot_groups(by_group, selected["threshold"], args.output)
    print(f"Exploratory cutoff={selected['threshold']:.4f}; "
          f"MCS above={selected['mcs_above']}/{len(mcs)}; "
          f"UWS above={selected['uws_above']}/{len(uws)}; "
          f"CV balanced accuracy={cv['mean_fold_balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
