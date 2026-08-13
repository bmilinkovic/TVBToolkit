"""Audit PCI provenance and test baseline diagnostic stratification.

This analysis deliberately keeps the two locally available corrected-protocol
PCI tables separate.  The first was calculated with the earlier baseline
trial-bootstrap source threshold.  The second was produced after the
pre/post-exchange threshold was introduced, but its matching HPC manifest is
not available locally and it is therefore labelled as an unverified candidate.

The empirical comparison tests ordinal recovery rather than numerical equality:
model PCI is calculated from 90 regional firing-rate signals, so the empirical
TMS--EEG cutoffs of 0.31 and 0.44 are not transferable to this scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations, pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

CONDITION_ORDER = ["UWS", "MCS", "EMCS", "CNT"]
DISPLAY_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
COLORS = {
    "COMA": "#344765",
    "UWS": "#876985",
    "MCS": "#C75D26",
    "EMCS": "#E8B56D",
    "CNT": "#5B8A72",
}
EMPIRICAL_2013 = pd.DataFrame(
    [
        {
            "condition": "UWS",
            "mean": 0.24,
            "sd": 0.04,
            "minimum": 0.19,
            "maximum": 0.31,
            "n_measurements": 15,
        },
        {
            "condition": "MCS",
            "mean": 0.39,
            "sd": 0.05,
            "minimum": 0.32,
            "maximum": 0.49,
            "n_measurements": 15,
        },
        {
            "condition": "EMCS",
            "mean": 0.43,
            "sd": 0.05,
            "minimum": 0.37,
            "maximum": 0.52,
            "n_measurements": 14,
        },
        {
            "condition": "CNT",
            "mean": 0.55,
            "sd": 0.05,
            "minimum": 0.44,
            "maximum": 0.67,
            "n_measurements": 110,
        },
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-csv",
        type=Path,
        default=Path(
            "/Users/borjan/Downloads/serotonergic_pci_subject_metrics (1).csv"
        ),
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path(
            "/Users/borjan/Downloads/serotonergic_pci_subject_metrics (2).csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pci_implementation_and_stratification_validation"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in available else "Helvetica"
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    sns.set_theme(style="ticks", context="paper", font=font)


def load_baseline(path: Path, estimator: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"condition", "subject_id", "occupancy", "n_trials", "pci_mean"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    n_subjects = data[["condition", "subject_id"]].drop_duplicates().shape[0]
    if data.shape[0] != 756 or n_subjects != 189:
        raise ValueError(f"{path} is not the expected 189 x 4 result table.")
    if set(data["occupancy"].round(3)) != {0.0, 0.25, 0.5, 0.766}:
        raise ValueError(f"{path} does not contain the four expected occupancies.")
    if not data["n_trials"].eq(100).all():
        raise ValueError(f"{path} does not report 100 trials in every cell.")
    baseline = data.loc[data["occupancy"].eq(0.0)].copy()
    if baseline.shape[0] != 189:
        raise ValueError(f"{path} does not contain 189 baseline rows.")
    baseline["estimator"] = estimator
    baseline["input_file"] = str(path.resolve())
    baseline["input_sha256"] = sha256(path)
    return baseline


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(replicates, values.size), replace=True).mean(
        axis=1
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    baseline: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (estimator, condition), group in baseline.groupby(
        ["estimator", "condition"], observed=True
    ):
        values = group["pci_mean"].to_numpy(float)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            replicates=replicates,
            rng=rng,
        )
        rows.append(
            {
                "estimator": estimator,
                "condition": condition,
                "n_subjects": values.size,
                "mean_pci": float(values.mean()),
                "sd_pci": float(values.std(ddof=1)),
                "median_pci": float(np.median(values)),
                "q25_pci": float(np.quantile(values, 0.25)),
                "q75_pci": float(np.quantile(values, 0.75)),
                "mean_ci95_low": ci_low,
                "mean_ci95_high": ci_high,
                "zero_pci_n": int(np.count_nonzero(values == 0.0)),
                "zero_pci_fraction": float(np.mean(values == 0.0)),
            }
        )
    return pd.DataFrame(rows)


def ordered_rank_statistic(values: np.ndarray, order_scores: np.ndarray) -> float:
    ranks = stats.rankdata(values, method="average")
    return float(np.corrcoef(ranks, order_scores)[0, 1])


def ordered_permutation_test(
    data: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    ordered = data.loc[data["condition"].isin(CONDITION_ORDER)].copy()
    scores = (
        ordered["condition"]
        .map({condition: index for index, condition in enumerate(CONDITION_ORDER)})
        .to_numpy(float)
    )
    values = ordered["pci_mean"].to_numpy(float)
    observed = ordered_rank_statistic(values, scores)
    ranks = stats.rankdata(values, method="average")
    centered_ranks = ranks - ranks.mean()
    centered_scores = scores - scores.mean()
    denominator = np.sqrt(
        np.square(centered_ranks).sum() * np.square(centered_scores).sum()
    )
    exceedances = 0
    chunk = 1_000
    for start in range(0, replicates, chunk):
        n_chunk = min(chunk, replicates - start)
        permuted = np.vstack([rng.permutation(centered_scores) for _ in range(n_chunk)])
        statistics = permuted @ centered_ranks / denominator
        exceedances += int(np.count_nonzero(statistics >= observed - 1e-15))
    p_value = (exceedances + 1.0) / (replicates + 1.0)
    return observed, float(p_value)


def omnibus_and_trend(
    baseline: pd.DataFrame,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator, data in baseline.groupby("estimator", observed=True):
        groups = [
            data.loc[data["condition"].eq(condition), "pci_mean"].to_numpy(float)
            for condition in CONDITION_ORDER
        ]
        welch = anova_oneway(groups, use_var="unequal")
        kruskal = stats.kruskal(*groups)
        ordered_rho, ordered_p = ordered_permutation_test(
            data,
            replicates=permutations,
            rng=rng,
        )
        rows.extend(
            [
                {
                    "estimator": estimator,
                    "test": "Welch ANOVA across UWS/MCS/EMCS/CNT",
                    "alternative": "any group difference",
                    "statistic": float(welch.statistic),
                    "df1": float(welch.df_num),
                    "df2": float(welch.df_denom),
                    "p_value": float(welch.pvalue),
                    "n_permutations": np.nan,
                },
                {
                    "estimator": estimator,
                    "test": "Kruskal-Wallis across UWS/MCS/EMCS/CNT",
                    "alternative": "any distributional difference",
                    "statistic": float(kruskal.statistic),
                    "df1": 3.0,
                    "df2": np.nan,
                    "p_value": float(kruskal.pvalue),
                    "n_permutations": np.nan,
                },
                {
                    "estimator": estimator,
                    "test": "Ordered rank permutation trend",
                    "alternative": "UWS < MCS < EMCS < CNT",
                    "statistic": ordered_rho,
                    "df1": np.nan,
                    "df2": np.nan,
                    "p_value": ordered_p,
                    "n_permutations": permutations,
                },
            ]
        )
    return pd.DataFrame(rows)


def pairwise_tests(baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator, data in baseline.groupby("estimator", observed=True):
        for lower, higher in combinations(CONDITION_ORDER, 2):
            lower_values = data.loc[data["condition"].eq(lower), "pci_mean"].to_numpy(
                float
            )
            higher_values = data.loc[data["condition"].eq(higher), "pci_mean"].to_numpy(
                float
            )
            welch = stats.ttest_ind(
                higher_values,
                lower_values,
                equal_var=False,
                alternative="greater",
            )
            mann_whitney = stats.mannwhitneyu(
                higher_values,
                lower_values,
                alternative="greater",
                method="auto",
            )
            rows.append(
                {
                    "estimator": estimator,
                    "lower_expected_group": lower,
                    "higher_expected_group": higher,
                    "mean_difference_higher_minus_lower": float(
                        higher_values.mean() - lower_values.mean()
                    ),
                    "probability_superiority": float(
                        mann_whitney.statistic
                        / (higher_values.size * lower_values.size)
                    ),
                    "welch_t": float(welch.statistic),
                    "welch_p_one_sided": float(welch.pvalue),
                    "mann_whitney_u": float(mann_whitney.statistic),
                    "mann_whitney_p_one_sided": float(mann_whitney.pvalue),
                    "n_lower": lower_values.size,
                    "n_higher": higher_values.size,
                    "adjacent_empirical_step": bool(
                        CONDITION_ORDER.index(higher) - CONDITION_ORDER.index(lower)
                        == 1
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output["welch_p_holm_within_estimator"] = np.nan
    output["mann_whitney_p_holm_within_estimator"] = np.nan
    for indices in output.groupby("estimator", observed=True).groups.values():
        output.loc[indices, "welch_p_holm_within_estimator"] = multipletests(
            output.loc[indices, "welch_p_one_sided"], method="holm"
        )[1]
        output.loc[indices, "mann_whitney_p_holm_within_estimator"] = multipletests(
            output.loc[indices, "mann_whitney_p_one_sided"], method="holm"
        )[1]
    output["direction_matches_empirical"] = (
        output["mean_difference_higher_minus_lower"] > 0
    )
    return output


def stratification_score(
    summary: pd.DataFrame, pairwise_results: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator, data in summary.groupby("estimator", observed=True):
        means = data.loc[data["condition"].isin(CONDITION_ORDER)].set_index(
            "condition"
        )["mean_pci"]
        adjacent = pairwise_results.loc[
            pairwise_results["estimator"].eq(estimator)
            & pairwise_results["adjacent_empirical_step"]
        ]
        correct = adjacent["direction_matches_empirical"]
        significant = correct & adjacent["mann_whitney_p_holm_within_estimator"].lt(
            0.05
        )
        rows.append(
            {
                "estimator": estimator,
                "observed_mean_order_low_to_high": " < ".join(
                    means.sort_values().index.tolist()
                ),
                "empirical_expected_order": "UWS < MCS < EMCS < CNT",
                "strict_empirical_mean_order_recovered": bool(
                    all(
                        means.loc[a] < means.loc[b]
                        for a, b in pairwise(CONDITION_ORDER)
                    )
                ),
                "adjacent_steps_correct_n_of_3": int(correct.sum()),
                "adjacent_steps_correct_and_holm_significant_n_of_3": int(
                    significant.sum()
                ),
                "all_pci_zero_n": int(data["zero_pci_n"].sum()),
            }
        )
    return pd.DataFrame(rows)


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_validation(
    baseline: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int,
) -> None:
    configure_style()
    width = 183.0 / 25.4
    fig = plt.figure(figsize=(width, 5.25))
    grid = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.32)
    empirical_axis = fig.add_subplot(grid[0, 0])
    primary_axis = fig.add_subplot(grid[0, 1])
    probability_axis = fig.add_subplot(grid[1, 0])
    candidate_axis = fig.add_subplot(grid[1, 1])

    x_positions = np.arange(len(CONDITION_ORDER))
    empirical = EMPIRICAL_2013.set_index("condition").loc[CONDITION_ORDER]
    for x, condition in zip(x_positions, CONDITION_ORDER, strict=True):
        empirical_axis.errorbar(
            x,
            empirical.loc[condition, "mean"],
            yerr=np.array(
                [
                    [
                        empirical.loc[condition, "mean"]
                        - empirical.loc[condition, "minimum"]
                    ],
                    [
                        empirical.loc[condition, "maximum"]
                        - empirical.loc[condition, "mean"]
                    ],
                ]
            ),
            fmt="none",
            ecolor=COLORS[condition],
            elinewidth=1.2,
            capsize=3,
            capthick=1.0,
            zorder=1,
        )
        empirical_axis.scatter(
            x,
            empirical.loc[condition, "mean"],
            s=28,
            color=COLORS[condition],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
    empirical_axis.plot(x_positions, empirical["mean"], color="#777777", lw=0.8)
    empirical_axis.set_xticks(x_positions, CONDITION_ORDER)
    empirical_axis.set_ylabel("Empirical PCI")
    empirical_axis.set_title("Casali et al. 2013: mean and reported range", loc="left")
    empirical_axis.text(
        0.02,
        0.96,
        "Expected order: UWS < MCS < EMCS < wake",
        transform=empirical_axis.transAxes,
        va="top",
        fontsize=6.2,
    )

    rng = np.random.default_rng(seed)

    def draw_distribution(axis: mpl.axes.Axes, estimator: str, title: str) -> None:
        subset = baseline.loc[
            baseline["estimator"].eq(estimator)
            & baseline["condition"].isin(CONDITION_ORDER)
        ]
        sns.boxplot(
            data=subset,
            x="condition",
            y="pci_mean",
            order=CONDITION_ORDER,
            hue="condition",
            palette=COLORS,
            dodge=False,
            width=0.5,
            fliersize=0,
            linewidth=0.7,
            saturation=0.85,
            ax=axis,
            legend=False,
        )
        for index, condition in enumerate(CONDITION_ORDER):
            values = subset.loc[subset["condition"].eq(condition), "pci_mean"].to_numpy(
                float
            )
            jitter = rng.uniform(-0.16, 0.16, values.size)
            axis.scatter(
                index + jitter,
                values,
                s=8,
                color=COLORS[condition],
                alpha=0.5,
                edgecolors="none",
                rasterized=True,
            )
        means = (
            summary.loc[
                summary["estimator"].eq(estimator)
                & summary["condition"].isin(CONDITION_ORDER)
            ]
            .set_index("condition")
            .loc[CONDITION_ORDER]
        )
        axis.plot(
            np.arange(4),
            means["mean_pci"],
            color="#222222",
            lw=1.0,
            marker="o",
            ms=2.7,
            zorder=5,
        )
        axis.set_xlabel("")
        axis.set_ylabel("Baseline model PCI")
        axis.set_title(title, loc="left")

    primary_name = "baseline-bootstrap, alpha=0.01"
    candidate_name = "pre/post candidate, alpha=0.05"
    draw_distribution(
        primary_axis,
        primary_name,
        "Corrected simulation; analysed PCI estimator",
    )
    draw_distribution(
        candidate_axis,
        candidate_name,
        "Later significance estimator (provenance incomplete)",
    )

    adjacent = pairwise.loc[
        pairwise["estimator"].eq(primary_name) & pairwise["adjacent_empirical_step"]
    ].copy()
    labels = [
        f"{row.lower_expected_group} → {row.higher_expected_group}"
        for row in adjacent.itertuples(index=False)
    ]
    y = np.arange(len(adjacent))
    colors = [
        COLORS[row.higher_expected_group] for row in adjacent.itertuples(index=False)
    ]
    probability_axis.axvline(0.5, color="#555555", lw=0.8, ls="--")
    probability_axis.scatter(
        adjacent["probability_superiority"],
        y,
        color=colors,
        s=30,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for position, row in enumerate(adjacent.itertuples(index=False)):
        probability_axis.text(
            min(row.probability_superiority + 0.025, 0.96),
            position,
            significance_label(row.mann_whitney_p_holm_within_estimator),
            va="center",
            fontsize=6.5,
        )
    probability_axis.set_yticks(y, labels)
    probability_axis.invert_yaxis()
    probability_axis.set_xlim(0.25, 1.0)
    probability_axis.set_xlabel("P(higher-state subject has higher PCI)")
    probability_axis.set_title("Recovery of adjacent empirical ordering", loc="left")

    for label, axis in zip("abcd", fig.axes, strict=True):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.5)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "pci_baseline_stratification_validation"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    primary = "baseline-bootstrap, alpha=0.01"
    candidate = "pre/post candidate, alpha=0.05"
    primary_summary = summary.loc[summary["estimator"].eq(primary)].set_index(
        "condition"
    )
    candidate_summary = summary.loc[summary["estimator"].eq(candidate)].set_index(
        "condition"
    )
    primary_score = scores.loc[scores["estimator"].eq(primary)].iloc[0]
    candidate_score = scores.loc[scores["estimator"].eq(candidate)].iloc[0]
    primary_trend = omnibus.loc[
        omnibus["estimator"].eq(primary)
        & omnibus["test"].eq("Ordered rank permutation trend")
    ].iloc[0]
    primary_pairs = pairwise.loc[
        pairwise["estimator"].eq(primary) & pairwise["adjacent_empirical_step"]
    ].set_index(["lower_expected_group", "higher_expected_group"])

    report = f"""# PCI implementation and baseline-stratification validation

## Scope

This audit distinguishes simulation protocol, source-significance estimation and
Lempel--Ziv compression. It compares ordinal diagnostic stratification, not the
absolute empirical PCI scale.

## Implementation conclusion

The corrected simulation workflow performs the essential conceptual PCI steps:

1. apply the same direct perturbation repeatedly;
2. cut each trial around its own stimulation onset;
3. align trials to a common onset and average them;
4. identify statistically active regions and times;
5. sort regions by total post-stimulation activity;
6. compute two-dimensional Lempel--Ziv complexity; and
7. normalize by binary source entropy.

The simulation is not a literal reproduction of empirical TMS--EEG. Its input is
90 AAL regional firing rates, not artifact-cleaned scalp EEG reconstructed onto a
dense cortical source space. The pulse is a 10-ms model input to left SMA, not a
measured TMS electric field. These are necessary model abstractions and prohibit
using empirical numerical PCI cutoffs.

| Stage | Current implementation | Empirical Casali protocol | Assessment |
|---|---|---|---|
| Repetitions | 100 independent simulations with matched seeds | About 150 TMS trials | Conceptually aligned; trial convergence still needs to be shown |
| Timing | Each epoch is cut around its recorded onset, then aligned | Trials are time-locked to TMS | Aligned |
| Perturbation | 10-ms direct input to AAL left SMA | Brief TMS pulse at a viable cortical target | Necessary model abstraction, but fixed targeting needs sensitivity analysis |
| Signals | 90 regional excitatory firing-rate traces at 7.8125-ms sampling | 60-channel EEG reconstructed to distributed cortical currents | Major observation-model difference; absolute PCI values are not interchangeable |
| Window | 8--300 ms after onset | First 300 ms after TMS, after artifact handling | Closely aligned in duration; model omits the first sampled pulse bin |
| Significance | Current default: pre/post whole-block exchange with a per-time maximum over regions | Nonparametric source-level bootstrap; later empirical descriptions use maxima from resampled baseline activity | Related family-wise inference, but not demonstrated to be the same null |
| Compression | Sort active regions, two-dimensional Lempel--Ziv complexity, entropy normalization | Same conceptual operations | Aligned |
| Low-information guard | Source entropy must exceed 0.08 | Entropy floor 0.08 in the empirical software lineage | Aligned in intent |

The relevant code is in `scripts/run_serotonergic_pci_pilot.py`: anatomical target
resolution (lines 906--937), randomized onsets (940--957), model pulse
(1068--1075), temporal-average monitor (1077--1092), onset-centred epoching
(1212--1316), and one PCI on the aligned trials (1319--1353). Receptor values are
joined to the structural atlas by label in
`src/tvbtoolkit/workflows/pharmacology.py` (20--88). The compression and entropy
normalization are in `src/tvbtoolkit/complexity/pci_casali.py` (103--180).

The principal unresolved difference is the source-significance null. The analysed
dataset used a maximum-statistic baseline trial bootstrap at alpha=0.01. The current
production default uses complete within-trial pre/post exchange, 1,000 permutations,
alpha=0.05 and an entropy floor of 0.08. Casali et al. 2013 describe a nonparametric
bootstrap at source level, and later empirical descriptions specify the 99th
percentile of maximum bootstrap-resampled baseline activity. The pre/post-exchange
method is defensible as a separate randomization test, but it is not established as
software-identical to the 2013 PCI implementation and should not currently be called
the canonical 2013 algorithm.

There is also a documentation inconsistency: the significance function signature
defaults to `alpha=0.05`, while its parameter docstring still says
`default=0.01`. More importantly, aliases named `casali` and `canonical` currently
resolve to `pre_post_swap`. Those names express an interpretation that this
literature audit does not establish.

## Empirical benchmark

Casali et al. 2013 reported the ordering UWS < MCS < EMCS < awake control:

| State | Mean PCI | SD | Range | Measurements |
|---|---:|---:|---:|---:|
| UWS | 0.24 | 0.04 | 0.19--0.31 | 15 |
| MCS | 0.39 | 0.05 | 0.32--0.49 | 15 |
| EMCS | 0.43 | 0.05 | 0.37--0.52 | 14 |
| Awake | 0.55 | 0.05 | 0.44--0.67 | 110 |

The paper did not contain an acute COMA group: the UWS, MCS and EMCS patients had
emerged from coma. COMA is therefore displayed descriptively but excluded from the
ordered replication test.

Primary sources: [Casali et al. 2013](https://old.postlab.psych.wisc.edu/files/7713/8014/7368/Sci_Transl_Med-2013.pdf)
and the larger independent validation by
[Casarotto et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5132045/).

## Model stratification: analysed corrected dataset

Baseline group means were UWS={primary_summary.loc["UWS", "mean_pci"]:.3f},
MCS={primary_summary.loc["MCS", "mean_pci"]:.3f},
EMCS={primary_summary.loc["EMCS", "mean_pci"]:.3f} and
CNT={primary_summary.loc["CNT", "mean_pci"]:.3f}.

Observed order: **{primary_score["observed_mean_order_low_to_high"]}**.
The strict empirical ordering was therefore
**{"recovered" if primary_score["strict_empirical_mean_order_recovered"] else "not recovered"}**.
{int(primary_score["adjacent_steps_correct_n_of_3"])}/3 adjacent directions were
correct.

The overall ordered rank association was rho={primary_trend["statistic"]:.3f},
permutation P={primary_trend["p_value"]:.3g}; however, this global trend is largely
driven by the clear separation of controls and should not be interpreted as complete
recovery of the clinical hierarchy.

Adjacent probability-of-superiority results were:

- UWS -> MCS: {primary_pairs.loc[("UWS", "MCS"), "probability_superiority"]:.3f},
  Holm P={primary_pairs.loc[("UWS", "MCS"), "mann_whitney_p_holm_within_estimator"]:.3g}.
- MCS -> EMCS: {primary_pairs.loc[("MCS", "EMCS"), "probability_superiority"]:.3f},
  Holm P={primary_pairs.loc[("MCS", "EMCS"), "mann_whitney_p_holm_within_estimator"]:.3g}.
- EMCS -> CNT: {primary_pairs.loc[("EMCS", "CNT"), "probability_superiority"]:.3f},
  Holm P={primary_pairs.loc[("EMCS", "CNT"), "mann_whitney_p_holm_within_estimator"]:.3g}.

Thus the model recovers UWS < MCS and patient groups < controls, but it does not
recover MCS < EMCS. EMCS has a lower mean than MCS and overlaps UWS substantially.
This is partial, not full, reproduction of Casali's stratification.

## Later pre/post candidate

The later table has baseline means UWS={candidate_summary.loc["UWS", "mean_pci"]:.3f},
MCS={candidate_summary.loc["MCS", "mean_pci"]:.3f},
EMCS={candidate_summary.loc["EMCS", "mean_pci"]:.3f} and
CNT={candidate_summary.loc["CNT", "mean_pci"]:.3f}, with
{int(candidate_summary["zero_pci_n"].sum())} baseline zero values.
Its observed order is **{candidate_score["observed_mean_order_low_to_high"]}** and it
does not recover the empirical hierarchy. Controls collapsing to approximately zero
is a validation failure for this estimator/configuration, not evidence that controls
have low perturbational complexity.

## Likely explanations for incomplete stratification

1. Diagnosis is imposed partly through group-shared adaptation values, but each
   structural connectome can alter the response non-monotonically.
2. All subjects are stimulated at one fixed left-SMA parcel. Casali used multiple
   viable targets and cautioned against stimulating structurally damaged cortex.
3. AAL90 firing rates are a much coarser observation space than empirical cortical
   source reconstruction; PCI depends on the number and pattern of binary sources.
4. The 128-Hz temporal-average monitor supplies only about 37 response samples from
   8--300 ms, far fewer than empirical source-space TMS--EEG.
5. The source-significance null is not yet validated for simulated firing rates.
6. The EMCS group contains only 18 heterogeneous connectomes, and its baseline model
   state is not guaranteed to be monotonically ordered by the hard-coded adaptation
   parameter alone.

## Required next checks

1. Recompute both baseline-bootstrap and pre/post-exchange PCI from the same cached
   aligned trials while changing one option at a time.
2. Save active fraction, entropy, threshold and PCI-before-entropy-floor for every
   subject to identify why controls collapse.
3. Complete 20/40/60/80/100-trial convergence for both PCI-LZ and PCI-ST.
4. Test alternative undamaged stimulation targets or subject-specific viable targets.
5. Test whether increased temporal sampling and a denser observation/source space
   restore EMCS > MCS.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    primary = load_baseline(
        args.primary_csv,
        "baseline-bootstrap, alpha=0.01",
    )
    candidate = load_baseline(
        args.candidate_csv,
        "pre/post candidate, alpha=0.05",
    )
    baseline = pd.concat([primary, candidate], ignore_index=True)
    summary = summarize(
        baseline,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    omnibus = omnibus_and_trend(
        baseline,
        permutations=args.permutation_replicates,
        rng=rng,
    )
    pairwise = pairwise_tests(baseline)
    scores = stratification_score(summary, pairwise)

    baseline.to_csv(table_dir / "baseline_subject_values.csv", index=False)
    summary.to_csv(table_dir / "baseline_group_summary.csv", index=False)
    omnibus.to_csv(table_dir / "baseline_omnibus_and_trend_tests.csv", index=False)
    pairwise.to_csv(table_dir / "baseline_pairwise_order_tests.csv", index=False)
    scores.to_csv(table_dir / "baseline_stratification_scorecard.csv", index=False)
    EMPIRICAL_2013.to_csv(
        table_dir / "casali_2013_reported_stratification.csv",
        index=False,
    )

    plot_validation(
        baseline,
        summary,
        pairwise,
        args.output_dir,
        seed=args.seed,
    )
    write_report(args.output_dir, summary, omnibus, pairwise, scores)

    manifest = {
        "analysis": "PCI implementation and baseline diagnostic stratification validation",
        "primary_input": str(args.primary_csv.resolve()),
        "primary_sha256": sha256(args.primary_csv),
        "candidate_input": str(args.candidate_csv.resolve()),
        "candidate_sha256": sha256(args.candidate_csv),
        "baseline_only": True,
        "ordered_replication_groups": CONDITION_ORDER,
        "coma_excluded_from_ordered_replication": True,
        "reason_coma_excluded": (
            "Casali 2013 tested UWS, MCS, EMCS and LIS patients who had emerged "
            "from coma; it did not define an acute COMA PCI reference group."
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "empirical_numeric_cutoffs_transferred_to_model": False,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote PCI validation analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
