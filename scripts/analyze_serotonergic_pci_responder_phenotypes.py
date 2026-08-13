"""Relate serotonergic PCI responses to cohort metadata and structural anatomy.

This is an exploratory in-silico responder-phenotyping analysis. The resulting
associations are not clinical treatment-response predictors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from PIL import Image
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests

CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]
PATIENT_CONDITIONS = ["COMA", "UWS", "MCS", "EMCS"]
CONDITION_COLORS = {
    "COMA": "#3B4A6B",
    "UWS": "#8B6B8B",
    "MCS": "#C5622F",
    "EMCS": "#E8B56D",
    "CNT": "#5B8A72",
}
COHORT_TO_CONDITION = {
    "coma": "COMA",
    "uws": "UWS",
    "mcs": "MCS",
    "emcs": "EMCS",
    "control": "CNT",
}
NATURE_DOUBLE_COLUMN_IN = 183.0 / 25.4
STIM_REGION_ZERO_BASED = 9
STIM_REGION_EXPECTED_LABEL = "Supp_Motor_Area_L"

GLOBAL_FEATURE_LABELS = {
    "sc_density": "Whole-brain edge density",
    "sc_mean_positive_weight_log": "Mean positive edge weight",
    "sc_total_weight_log": "Total structural weight",
    "sc_strength_cv": "Node-strength variability",
    "sc_spectral_radius_log": "Spectral radius",
    "stim_region_strength_log": "Left SMA weighted strength",
    "stim_region_degree": "Left SMA preserved links",
    "mean_connected_tract_length": "Mean connected tract length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/serotonergic_pci_personalized_analysis_original_new"),
    )
    parser.add_argument(
        "--analysis-prefix",
        default="serotonergic_pci_personalized_original_new",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path(
            "/Volumes/ex_data/cnrs/data_doc_liege/raw/doc_data/"
            "converted_structural/source_subject_map.csv"
        ),
    )
    parser.add_argument(
        "--damage-csv",
        type=Path,
        default=Path(
            "/Volumes/ex_data/cnrs/data_doc_liege/results/notebooks_outputs/"
            "brain_act_mask_qc_check/brain_act_damage_mask_per_subject.tsv"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/Volumes/ex_data/cnrs/data_doc_liege/raw/doc_data/converted_structural"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serotonergic_pci_responder_phenotyping_original_new"),
    )
    parser.add_argument("--control-baseline-quantile", type=float, default=0.05)
    parser.add_argument("--low-baseline-quantile", type=float, default=0.25)
    parser.add_argument("--jitter-seed", type=int, default=20260730)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    standard_deviation = float(numeric.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - float(numeric.mean())) / standard_deviation


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    z_value = stats.norm.ppf(0.975)
    denominator = 1 + z_value**2 / total
    centre = (proportion + z_value**2 / (2 * total)) / denominator
    half_width = (
        z_value
        * np.sqrt(proportion * (1 - proportion) / total + z_value**2 / (4 * total**2))
        / denominator
    )
    return float(centre - half_width), float(centre + half_width)


def load_structural_features(
    dataset_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    atlas_path = dataset_root / "atlas.npz"
    with np.load(atlas_path, allow_pickle=False) as atlas:
        region_labels = atlas["labels"].astype(str).tolist()
    if region_labels[STIM_REGION_ZERO_BASED] != STIM_REGION_EXPECTED_LABEL:
        raise ValueError(
            "Structural atlas does not place Supp_Motor_Area_L at zero-based "
            f"index {STIM_REGION_ZERO_BASED}: "
            f"{region_labels[STIM_REGION_ZERO_BASED]!r}"
        )

    global_rows: list[dict[str, object]] = []
    regional_rows: list[dict[str, object]] = []
    for cohort, condition in COHORT_TO_CONDITION.items():
        path = dataset_root / f"subjects_{cohort}.npz"
        with np.load(path, allow_pickle=False) as archive:
            subject_ids = archive["subject_ids"].astype(str)
            connectivities = archive["connectivity"]
            tract_lengths = archive["tract_lengths"]
            for subject_id, connectivity, lengths in zip(
                subject_ids,
                connectivities,
                tract_lengths,
                strict=True,
            ):
                matrix = np.asarray(connectivity, dtype=float)
                tract_matrix = np.asarray(lengths, dtype=float)
                upper = np.triu_indices_from(matrix, k=1)
                edges = matrix[upper]
                connected = edges > 0
                positive_edges = edges[connected]
                node_strength = matrix.sum(axis=1)
                node_degree = (matrix > 0).sum(axis=1)
                connected_lengths = tract_matrix[upper][connected]
                global_rows.append(
                    {
                        "cohort": cohort,
                        "condition": condition,
                        "subject_id": subject_id,
                        "sc_density": float(connected.mean()),
                        "sc_mean_positive_weight": float(positive_edges.mean()),
                        "sc_total_weight": float(edges.sum()),
                        "sc_strength_cv": float(
                            node_strength.std(ddof=0) / node_strength.mean()
                        ),
                        "sc_spectral_radius": float(np.linalg.eigvalsh(matrix)[-1]),
                        "stim_region_strength": float(
                            node_strength[STIM_REGION_ZERO_BASED]
                        ),
                        "stim_region_degree": int(node_degree[STIM_REGION_ZERO_BASED]),
                        "mean_connected_tract_length": float(connected_lengths.mean()),
                    }
                )
                for region_index, (region_label, degree, strength) in enumerate(
                    zip(
                        region_labels,
                        node_degree,
                        node_strength,
                        strict=True,
                    ),
                    start=1,
                ):
                    regional_rows.append(
                        {
                            "cohort": cohort,
                            "condition": condition,
                            "subject_id": subject_id,
                            "region_index": region_index,
                            "region_label": region_label,
                            "preserved_link_count": int(degree),
                            "weighted_strength": float(strength),
                        }
                    )

    global_features = pd.DataFrame(global_rows)
    for column in [
        "sc_mean_positive_weight",
        "sc_total_weight",
        "sc_spectral_radius",
        "stim_region_strength",
    ]:
        global_features[f"{column}_log"] = np.log1p(global_features[column])
    return global_features, pd.DataFrame(regional_rows), region_labels


def load_and_join(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    subject_path = (
        args.analysis_dir / f"{args.analysis_prefix}_subject_responder_table.csv"
    )
    long_path = args.analysis_dir / f"{args.analysis_prefix}_analysis_input.csv"
    subjects = pd.read_csv(subject_path)
    trajectories = pd.read_csv(long_path)
    metadata = pd.read_csv(args.metadata_csv)
    damage = pd.read_csv(args.damage_csv, sep="\t")

    metadata = metadata.copy()
    metadata["condition"] = metadata["cohort"].map(COHORT_TO_CONDITION)
    damage = damage.copy()
    damage["condition"] = damage["cohort"].map(COHORT_TO_CONDITION)
    global_features, regional_features, _ = load_structural_features(args.dataset_root)

    joined = (
        subjects.merge(
            metadata[
                [
                    "condition",
                    "subject_id",
                    "cohort",
                    "stage",
                    "sedation",
                    "source_sc_file",
                    "source_tl_file",
                    "source_subject_index",
                ]
            ],
            on=["condition", "subject_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            damage.drop(columns=["cohort"]),
            on=["condition", "subject_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            global_features.drop(columns=["cohort"]),
            on=["condition", "subject_id"],
            how="left",
            validate="one_to_one",
        )
    )
    required_complete = [
        "stage",
        "sedation",
        "sc_zero_fraction_upper",
        "stim_region_degree",
    ]
    if len(joined) != 189 or joined[required_complete].isna().any().any():
        raise ValueError(
            "The metadata/structural join did not produce 189 complete subjects."
        )

    control_baseline = joined.loc[
        joined["condition"].eq("CNT"),
        "baseline_pci",
    ].to_numpy(float)
    patients = joined.loc[~joined["condition"].eq("CNT")]
    control_lower_bound = float(
        np.quantile(control_baseline, args.control_baseline_quantile)
    )
    low_baseline_cutoff = float(
        np.quantile(patients["baseline_pci"], args.low_baseline_quantile)
    )
    joined["is_patient"] = ~joined["condition"].eq("CNT")
    joined["responder_int"] = (
        joined["control_referenced_response"].astype(bool).astype(int)
    )
    joined["baseline_below_control_range"] = (
        joined["baseline_pci"] < control_lower_bound
    )
    joined["max_dose_reaches_control_range"] = (
        joined["max_dose_pci"] >= control_lower_bound
    )
    joined["crosses_control_lower_bound"] = (
        joined["baseline_below_control_range"]
        & joined["max_dose_reaches_control_range"]
    )
    joined["low_baseline_doc"] = joined["is_patient"] & joined["baseline_pci"].le(
        low_baseline_cutoff
    )
    joined["low_baseline_to_control_range"] = (
        joined["low_baseline_doc"] & joined["max_dose_reaches_control_range"]
    )
    joined["wake_gap_to_control_lower_bound"] = (
        control_lower_bound - joined["baseline_pci"]
    )
    joined["control_gap_closure_fraction"] = np.where(
        joined["wake_gap_to_control_lower_bound"] > 0,
        joined["max_dose_delta"] / joined["wake_gap_to_control_lower_bound"],
        np.nan,
    )
    joined["stage"] = pd.Categorical(
        joined["stage"],
        categories=["chronic", "acute", "control"],
    )
    joined["sedation"] = pd.Categorical(
        joined["sedation"],
        categories=["non_sedated", "sedated"],
    )

    trajectories = trajectories.merge(
        joined[
            [
                "condition",
                "subject_id",
                "low_baseline_to_control_range",
            ]
        ],
        on=["condition", "subject_id"],
        how="left",
        validate="many_to_one",
    )
    regional_features = regional_features.merge(
        joined[
            [
                "condition",
                "subject_id",
                "baseline_pci",
                "max_dose_pci",
                "max_dose_delta",
                "linear_slope_per_occupancy",
                "stage",
                "sedation",
            ]
        ],
        on=["condition", "subject_id"],
        how="left",
        validate="many_to_one",
    )
    thresholds = {
        "control_baseline_quantile": args.control_baseline_quantile,
        "control_lower_pci_bound": control_lower_bound,
        "low_doc_baseline_quantile": args.low_baseline_quantile,
        "low_doc_baseline_cutoff": low_baseline_cutoff,
    }
    return joined, trajectories, regional_features, thresholds


def metadata_coverage(joined: pd.DataFrame) -> pd.DataFrame:
    variables = [
        ("condition", "categorical", "available"),
        ("stage", "categorical", "available"),
        ("sedation", "categorical", "available"),
        ("age", "continuous", "not found in attached sources"),
        ("sex", "categorical", "not found in attached sources"),
        ("aetiology", "categorical", "not found in attached sources"),
        ("time_since_injury", "continuous", "not found in attached sources"),
        ("CRS-R/GCS", "continuous", "not found in attached sources"),
        ("clinical_outcome", "categorical", "not found in attached sources"),
        ("structural_disconnection", "continuous", "available"),
        ("regional_structural_degree", "continuous", "available"),
    ]
    rows = []
    for variable, kind, status in variables:
        if variable in joined:
            complete = int(joined[variable].notna().sum())
        elif status == "available":
            complete = len(joined)
        else:
            complete = 0
        rows.append(
            {
                "variable": variable,
                "type": kind,
                "status": status,
                "n_complete": complete,
                "n_total": len(joined),
                "complete_fraction": complete / len(joined),
            }
        )
    return pd.DataFrame(rows)


def subgroup_summary(joined: pd.DataFrame) -> pd.DataFrame:
    patients = joined.loc[joined["is_patient"]].copy()
    rows: list[dict[str, object]] = []
    groupings = [
        ("condition", ["condition"]),
        ("stage", ["stage"]),
        ("sedation", ["sedation"]),
        ("condition_stage_sedation", ["condition", "stage", "sedation"]),
    ]
    for grouping, columns in groupings:
        for keys, group in patients.groupby(columns, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            responders = int(group["responder_int"].sum())
            low_group = group.loc[group["low_baseline_doc"]]
            low_entrants = int(low_group["max_dose_reaches_control_range"].sum())
            response_ci = wilson_interval(responders, len(group))
            low_ci = wilson_interval(low_entrants, len(low_group))
            row: dict[str, object] = {
                "grouping": grouping,
                "n_subjects": len(group),
                "baseline_pci_mean": float(group["baseline_pci"].mean()),
                "max_dose_pci_mean": float(group["max_dose_pci"].mean()),
                "max_dose_delta_mean": float(group["max_dose_delta"].mean()),
                "control_referenced_responders_n": responders,
                "control_referenced_responders_fraction": responders / len(group),
                "control_referenced_ci95_low": response_ci[0],
                "control_referenced_ci95_high": response_ci[1],
                "low_baseline_n": len(low_group),
                "low_baseline_to_control_range_n": low_entrants,
                "low_baseline_to_control_range_fraction": (
                    low_entrants / len(low_group) if len(low_group) else np.nan
                ),
                "low_baseline_to_control_ci95_low": low_ci[0],
                "low_baseline_to_control_ci95_high": low_ci[1],
                "stim_region_degree_mean": float(group["stim_region_degree"].mean()),
            }
            row.update(dict(zip(columns, keys, strict=True)))
            rows.append(row)
    return pd.DataFrame(rows)


def _population(joined: pd.DataFrame, name: str) -> pd.DataFrame:
    patients = joined.loc[joined["is_patient"]]
    if name == "DOC":
        return patients.copy()
    if name == "MCS_UWS":
        return patients.loc[patients["condition"].isin(["MCS", "UWS"])].copy()
    if name in PATIENT_CONDITIONS:
        return patients.loc[patients["condition"].eq(name)].copy()
    raise ValueError(name)


def exact_metadata_tests(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictor_specs = [
        ("stage", "acute", "chronic"),
        ("sedation", "sedated", "non_sedated"),
    ]
    for population_name in ["DOC", "MCS_UWS", "MCS", "UWS"]:
        population = _population(joined, population_name)
        for predictor, exposure, reference in predictor_specs:
            for outcome in ["responder_int"]:
                subset = population.loc[
                    population[predictor].isin([exposure, reference])
                ]
                table = pd.crosstab(subset[predictor], subset[outcome])
                if not {0, 1}.issubset(table.columns):
                    continue
                counts = np.array(
                    [
                        [
                            int(table.loc[exposure, 1]),
                            int(table.loc[exposure, 0]),
                        ],
                        [
                            int(table.loc[reference, 1]),
                            int(table.loc[reference, 0]),
                        ],
                    ]
                )
                odds_ratio, p_value = stats.fisher_exact(
                    counts,
                    alternative="two-sided",
                )
                table_model = Table2x2(counts, shift_zeros=True)
                ci_low, ci_high = table_model.oddsratio_confint()
                rows.append(
                    {
                        "population": population_name,
                        "outcome": "control_referenced_response",
                        "predictor": predictor,
                        "exposure": exposure,
                        "reference": reference,
                        "n_exposure": int(counts[0].sum()),
                        "n_reference": int(counts[1].sum()),
                        "events_exposure": int(counts[0, 0]),
                        "events_reference": int(counts[1, 0]),
                        "odds_ratio": float(odds_ratio),
                        "odds_ratio_ci95_low": float(ci_low),
                        "odds_ratio_ci95_high": float(ci_high),
                        "p_value": float(p_value),
                    }
                )

    for population_name in ["DOC", "MCS_UWS"]:
        population = _population(joined, population_name)
        population = population.loc[population["low_baseline_doc"]].copy()
        population["reached_int"] = population["max_dose_reaches_control_range"].astype(
            int
        )
        for predictor, exposure, reference in predictor_specs:
            subset = population.loc[population[predictor].isin([exposure, reference])]
            table = pd.crosstab(subset[predictor], subset["reached_int"])
            if not {0, 1}.issubset(table.columns):
                continue
            counts = np.array(
                [
                    [
                        int(table.loc[exposure, 1]),
                        int(table.loc[exposure, 0]),
                    ],
                    [
                        int(table.loc[reference, 1]),
                        int(table.loc[reference, 0]),
                    ],
                ]
            )
            odds_ratio, p_value = stats.fisher_exact(counts)
            table_model = Table2x2(counts, shift_zeros=True)
            ci_low, ci_high = table_model.oddsratio_confint()
            rows.append(
                {
                    "population": f"LOW_BASELINE_{population_name}",
                    "outcome": "maximum_dose_reaches_control_range",
                    "predictor": predictor,
                    "exposure": exposure,
                    "reference": reference,
                    "n_exposure": int(counts[0].sum()),
                    "n_reference": int(counts[1].sum()),
                    "events_exposure": int(counts[0, 0]),
                    "events_reference": int(counts[1, 0]),
                    "odds_ratio": float(odds_ratio),
                    "odds_ratio_ci95_low": float(ci_low),
                    "odds_ratio_ci95_high": float(ci_high),
                    "p_value": float(p_value),
                }
            )

    output = pd.DataFrame(rows)
    output["p_holm_all_exact_tests"] = multipletests(
        output["p_value"],
        method="holm",
    )[1]
    return output


def continuous_metadata_tests(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictor_specs = [
        ("stage", "acute", "chronic"),
        ("sedation", "sedated", "non_sedated"),
    ]
    for population_name in ["DOC", "MCS_UWS", "MCS", "UWS"]:
        population = _population(joined, population_name)
        for outcome in [
            "max_dose_delta",
            "linear_slope_per_occupancy",
        ]:
            for predictor, exposure, reference in predictor_specs:
                exposed = population.loc[
                    population[predictor].eq(exposure),
                    outcome,
                ].to_numpy(float)
                referenced = population.loc[
                    population[predictor].eq(reference),
                    outcome,
                ].to_numpy(float)
                if len(exposed) < 2 or len(referenced) < 2:
                    continue
                statistic, p_value = stats.mannwhitneyu(
                    exposed,
                    referenced,
                    alternative="two-sided",
                )
                rank_biserial = 2 * statistic / (len(exposed) * len(referenced)) - 1
                rows.append(
                    {
                        "population": population_name,
                        "outcome": outcome,
                        "predictor": predictor,
                        "exposure": exposure,
                        "reference": reference,
                        "n_exposure": len(exposed),
                        "n_reference": len(referenced),
                        "median_exposure": float(np.median(exposed)),
                        "median_reference": float(np.median(referenced)),
                        "mean_exposure": float(np.mean(exposed)),
                        "mean_reference": float(np.mean(referenced)),
                        "mann_whitney_u": float(statistic),
                        "rank_biserial_exposure_minus_reference": float(rank_biserial),
                        "p_value": float(p_value),
                    }
                )
    output = pd.DataFrame(rows)
    output["p_holm_all_continuous_tests"] = multipletests(
        output["p_value"],
        method="holm",
    )[1]
    return output


def fit_core_models(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population_name in ["DOC", "MCS_UWS"]:
        data = _population(joined, population_name)
        data["stage"] = data["stage"].cat.remove_unused_categories()
        data["sedation"] = data["sedation"].cat.remove_unused_categories()
        data["baseline_pci_z"] = zscore(data["baseline_pci"])
        data["sc_zero_fraction_z"] = zscore(data["sc_zero_fraction_upper"])
        covariates = (
            "baseline_pci_z + "
            "C(condition, Treatment(reference='UWS')) + "
            "C(stage, Treatment(reference='chronic')) + "
            "C(sedation, Treatment(reference='non_sedated')) + "
            "sc_zero_fraction_z"
        )
        for outcome in [
            "max_dose_pci",
            "linear_slope_per_occupancy",
        ]:
            model = smf.ols(
                f"{outcome} ~ {covariates}",
                data=data,
            ).fit(cov_type="HC3")
            confidence = model.conf_int()
            for term in model.params.index:
                rows.append(
                    {
                        "population": population_name,
                        "model": "robust_ols_hc3",
                        "outcome": outcome,
                        "term": term,
                        "estimate": float(model.params[term]),
                        "ci95_low": float(confidence.loc[term, 0]),
                        "ci95_high": float(confidence.loc[term, 1]),
                        "p_value": float(model.pvalues[term]),
                        "n_subjects": len(data),
                        "r_squared": float(model.rsquared),
                    }
                )

        logistic = smf.glm(
            f"responder_int ~ {covariates}",
            data=data,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC3")
        confidence = logistic.conf_int()
        for term in logistic.params.index:
            rows.append(
                {
                    "population": population_name,
                    "model": "binomial_glm_hc3",
                    "outcome": "control_referenced_response",
                    "term": term,
                    "estimate": float(np.exp(logistic.params[term])),
                    "ci95_low": float(np.exp(confidence.loc[term, 0])),
                    "ci95_high": float(np.exp(confidence.loc[term, 1])),
                    "p_value": float(logistic.pvalues[term]),
                    "n_subjects": len(data),
                    "r_squared": np.nan,
                }
            )
    return pd.DataFrame(rows)


def fit_global_structural_models(joined: pd.DataFrame) -> pd.DataFrame:
    features = list(GLOBAL_FEATURE_LABELS)
    rows = []
    for population_name in ["DOC", "MCS_UWS"]:
        population = _population(joined, population_name)
        population["baseline_pci_z"] = zscore(population["baseline_pci"])
        for outcome in [
            "max_dose_pci",
            "linear_slope_per_occupancy",
        ]:
            for feature in features:
                data = population.copy()
                data["feature_z"] = zscore(data[feature])
                formula = (
                    f"{outcome} ~ baseline_pci_z + "
                    "C(condition, Treatment(reference='UWS')) + "
                    "C(stage, Treatment(reference='chronic')) + "
                    "C(sedation, Treatment(reference='non_sedated')) + "
                    "feature_z"
                )
                model = smf.ols(formula, data=data).fit(cov_type="HC3")
                ci_low, ci_high = model.conf_int().loc["feature_z"]
                rows.append(
                    {
                        "population": population_name,
                        "outcome": outcome,
                        "feature": feature,
                        "feature_label": GLOBAL_FEATURE_LABELS[feature],
                        "standardized_beta": float(model.params["feature_z"]),
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "p_value": float(model.pvalues["feature_z"]),
                        "n_subjects": len(data),
                        "model_r_squared": float(model.rsquared),
                    }
                )
    output = pd.DataFrame(rows)
    output["q_fdr_within_population_outcome"] = np.nan
    for index in output.groupby(
        ["population", "outcome"],
    ).groups.values():
        output.loc[index, "q_fdr_within_population_outcome"] = multipletests(
            output.loc[index, "p_value"],
            method="fdr_bh",
        )[1]
    return output.sort_values(["population", "outcome", "p_value"]).reset_index(
        drop=True
    )


def fit_regional_models(regional_features: pd.DataFrame) -> pd.DataFrame:
    population = regional_features.loc[
        regional_features["condition"].isin(["MCS", "UWS"])
    ].copy()
    population["baseline_pci_z"] = zscore(population["baseline_pci"])
    rows = []
    for (region_index, region_label), region in population.groupby(
        ["region_index", "region_label"],
        observed=True,
    ):
        region = region.copy()
        if region["preserved_link_count"].std(ddof=1) == 0:
            continue
        region["feature_z"] = zscore(region["preserved_link_count"])
        for outcome in [
            "max_dose_pci",
            "linear_slope_per_occupancy",
        ]:
            formula = (
                f"{outcome} ~ baseline_pci_z + "
                "C(condition, Treatment(reference='UWS')) + "
                "C(stage, Treatment(reference='chronic')) + "
                "C(sedation, Treatment(reference='non_sedated')) + "
                "feature_z"
            )
            model = smf.ols(formula, data=region).fit(cov_type="HC3")
            ci_low, ci_high = model.conf_int().loc["feature_z"]
            rows.append(
                {
                    "population": "MCS_UWS",
                    "outcome": outcome,
                    "region_index": int(region_index),
                    "region_label": str(region_label),
                    "standardized_beta": float(model.params["feature_z"]),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "p_value": float(model.pvalues["feature_z"]),
                    "n_subjects": int(
                        region[["condition", "subject_id"]].drop_duplicates().shape[0]
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output["q_fdr_across_90_regions"] = np.nan
    for outcome, index in output.groupby("outcome").groups.items():
        output.loc[index, "q_fdr_across_90_regions"] = multipletests(
            output.loc[index, "p_value"],
            method="fdr_bh",
        )[1]
    return output.sort_values(["outcome", "p_value"]).reset_index(drop=True)


def threshold_sensitivity(joined: pd.DataFrame) -> pd.DataFrame:
    patients = joined.loc[joined["is_patient"]]
    control = joined.loc[joined["condition"].eq("CNT"), "baseline_pci"]
    rows = []
    for control_quantile in [0.025, 0.05, 0.10]:
        control_cutoff = float(np.quantile(control, control_quantile))
        for low_quantile in [0.10, 0.20, 0.25, 0.33]:
            low_cutoff = float(np.quantile(patients["baseline_pci"], low_quantile))
            eligible = patients.loc[patients["baseline_pci"].le(low_cutoff)]
            entrants = eligible.loc[eligible["max_dose_pci"].ge(control_cutoff)]
            rows.append(
                {
                    "control_baseline_quantile": control_quantile,
                    "control_lower_pci_cutoff": control_cutoff,
                    "low_doc_baseline_quantile": low_quantile,
                    "low_doc_baseline_cutoff": low_cutoff,
                    "eligible_low_baseline_n": len(eligible),
                    "entrants_n": len(entrants),
                    "entrants_fraction": len(entrants) / len(eligible),
                    "coma_n": int(entrants["condition"].eq("COMA").sum()),
                    "uws_n": int(entrants["condition"].eq("UWS").sum()),
                    "mcs_n": int(entrants["condition"].eq("MCS").sum()),
                    "emcs_n": int(entrants["condition"].eq("EMCS").sum()),
                }
            )
    return pd.DataFrame(rows)


def target_degree_sensitivity(joined: pd.DataFrame) -> pd.DataFrame:
    data = _population(joined, "MCS_UWS")
    rows = []
    for minimum_degree in [0, 20, 40, 50, 60, 70, 80]:
        subset = data.loc[data["stim_region_degree"].ge(minimum_degree)].copy()
        subset["baseline_pci_z"] = zscore(subset["baseline_pci"])
        subset["feature_z"] = zscore(subset["stim_region_degree"])
        for outcome in [
            "max_dose_pci",
            "linear_slope_per_occupancy",
        ]:
            model = smf.ols(
                f"{outcome} ~ baseline_pci_z + "
                "C(condition, Treatment(reference='UWS')) + "
                "C(stage, Treatment(reference='chronic')) + "
                "C(sedation, Treatment(reference='non_sedated')) + "
                "feature_z",
                data=subset,
            ).fit(cov_type="HC3")
            ci_low, ci_high = model.conf_int().loc["feature_z"]
            rows.append(
                {
                    "minimum_left_sma_degree_included": minimum_degree,
                    "outcome": outcome,
                    "n_subjects": len(subset),
                    "standardized_beta": float(model.params["feature_z"]),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "p_value": float(model.pvalues["feature_z"]),
                }
            )
    return pd.DataFrame(rows)


def nature_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font="Arial")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def style_axis(axis: mpl.axes.Axes, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.grid(False)
    if grid_axis:
        axis.grid(
            axis=grid_axis,
            color="#D9D9D9",
            linewidth=0.4,
            alpha=0.72,
        )


def panel_label(axis: mpl.axes.Axes, label: str, title: str) -> None:
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    axis.set_title(title, loc="left", pad=4.0, fontweight="normal")


def save_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg"]:
        figure.savefig(output_dir / f"{stem}.{extension}")
    png_path = output_dir / f"{stem}.png"
    figure.savefig(png_path, dpi=600)
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
    tiff_path = output_dir / f"{stem}.tiff"
    figure.savefig(
        tiff_path,
        dpi=300,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(tiff_path) as image:
        image.convert("RGB").save(
            tiff_path,
            dpi=(300, 300),
            compression="tiff_lzw",
        )


def plot_main_figure(
    joined: pd.DataFrame,
    exact_tests: pd.DataFrame,
    global_models: pd.DataFrame,
    thresholds: dict[str, float],
    output_dir: Path,
    jitter_seed: int,
) -> None:
    rng = np.random.default_rng(jitter_seed)
    figure = plt.figure(
        figsize=(NATURE_DOUBLE_COLUMN_IN, 3.85),
        layout="constrained",
    )
    grid = figure.add_gridspec(1, 3, width_ratios=[1.05, 0.85, 1.0])
    baseline_axis = figure.add_subplot(grid[0, 0])
    metadata_axis = figure.add_subplot(grid[0, 1])
    target_axis = figure.add_subplot(grid[0, 2])
    patients = joined.loc[joined["is_patient"]].copy()

    for condition in PATIENT_CONDITIONS:
        group = patients.loc[patients["condition"].eq(condition)]
        regular = group.loc[~group["low_baseline_to_control_range"]]
        entrants = group.loc[group["low_baseline_to_control_range"]]
        baseline_axis.scatter(
            regular["baseline_pci"],
            regular["max_dose_pci"],
            s=8,
            color=CONDITION_COLORS[condition],
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            label=condition,
        )
        baseline_axis.scatter(
            entrants["baseline_pci"],
            entrants["max_dose_pci"],
            s=18,
            color=CONDITION_COLORS[condition],
            edgecolors="#111111",
            linewidths=0.45,
            zorder=4,
        )
    low_limit = min(
        float(patients["baseline_pci"].min()),
        float(patients["max_dose_pci"].min()),
    )
    high_limit = max(
        float(patients["baseline_pci"].max()),
        float(patients["max_dose_pci"].max()),
    )
    baseline_axis.plot(
        [low_limit, high_limit],
        [low_limit, high_limit],
        color="#B5B5B5",
        linewidth=0.6,
        zorder=0,
    )
    baseline_axis.axhline(
        thresholds["control_lower_pci_bound"],
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=0.65,
    )
    baseline_axis.axvline(
        thresholds["low_doc_baseline_cutoff"],
        color="#9A9A9A",
        linestyle=(0, (1.5, 2)),
        linewidth=0.55,
    )
    top_entrants = patients.loc[patients["low_baseline_to_control_range"]].nlargest(
        5, "max_dose_delta"
    )
    for row in top_entrants.itertuples(index=False):
        baseline_axis.annotate(
            f"{row.condition}:{row.subject_id}",
            (row.baseline_pci, row.max_dose_pci),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=4.7,
            color="#222222",
        )
    baseline_axis.set_xlim(low_limit - 0.02, high_limit + 0.02)
    baseline_axis.set_ylim(low_limit - 0.02, high_limit + 0.02)
    baseline_axis.set_xlabel("Baseline PCI")
    baseline_axis.set_ylabel("PCI at 0.766 occupancy")
    baseline_axis.legend(
        frameon=False,
        loc="lower right",
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    panel_label(
        baseline_axis,
        "a",
        "Low-baseline entrants to the control range",
    )
    style_axis(baseline_axis, grid_axis="both")

    metadata_groups = [
        ("Acute", patients["stage"].eq("acute")),
        ("Chronic", patients["stage"].eq("chronic")),
        ("Sedated", patients["sedation"].eq("sedated")),
        ("Not sedated", patients["sedation"].eq("non_sedated")),
    ]
    positions = np.arange(len(metadata_groups))
    for position, (label, mask) in zip(
        positions,
        metadata_groups,
        strict=True,
    ):
        group = patients.loc[mask]
        successes = int(group["responder_int"].sum())
        proportion = successes / len(group)
        ci_low, ci_high = wilson_interval(successes, len(group))
        metadata_axis.errorbar(
            position,
            proportion,
            yerr=[[proportion - ci_low], [ci_high - proportion]],
            fmt="o",
            markersize=3.8,
            markerfacecolor="#4F555A",
            markeredgecolor="white",
            markeredgewidth=0.4,
            color="#4F555A",
            elinewidth=0.8,
            capsize=2,
        )
        metadata_axis.text(
            position,
            min(ci_high + 0.035, 0.95),
            f"{successes}/{len(group)}",
            ha="center",
            va="bottom",
            fontsize=5.0,
        )
    metadata_axis.set_xticks(
        positions,
        [label for label, _ in metadata_groups],
        rotation=35,
        ha="right",
    )
    metadata_axis.set_ylim(0, 0.8)
    metadata_axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    metadata_axis.set_ylabel("Control-referenced responders")
    exact_primary = exact_tests.loc[
        exact_tests["population"].eq("DOC")
        & exact_tests["outcome"].eq("control_referenced_response")
    ]
    stage_p = float(
        exact_primary.loc[
            exact_primary["predictor"].eq("stage"),
            "p_value",
        ].iloc[0]
    )
    sedation_p = float(
        exact_primary.loc[
            exact_primary["predictor"].eq("sedation"),
            "p_value",
        ].iloc[0]
    )
    metadata_axis.text(
        0.5,
        0.02,
        f"Stage P={stage_p:.3f}",
        transform=metadata_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.0,
    )
    metadata_axis.text(
        0.5,
        0.09,
        f"Sedation P={sedation_p:.3f}",
        transform=metadata_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.0,
    )
    panel_label(metadata_axis, "b", "Response by clinical metadata")
    style_axis(metadata_axis, grid_axis="y")

    target_data = patients.loc[patients["condition"].isin(["MCS", "UWS"])].copy()
    for condition in ["UWS", "MCS"]:
        group = target_data.loc[target_data["condition"].eq(condition)]
        target_axis.scatter(
            group["stim_region_degree"] + rng.normal(0, 0.35, len(group)),
            group["max_dose_pci"],
            s=9,
            color=CONDITION_COLORS[condition],
            alpha=0.58,
            linewidths=0,
            rasterized=True,
            label=condition,
        )
    slope, intercept = np.polyfit(
        target_data["stim_region_degree"],
        target_data["max_dose_pci"],
        1,
    )
    x_line = np.linspace(
        target_data["stim_region_degree"].min(),
        target_data["stim_region_degree"].max(),
        100,
    )
    target_axis.plot(
        x_line,
        intercept + slope * x_line,
        color="#222222",
        linewidth=0.8,
    )
    target_result = global_models.loc[
        global_models["population"].eq("MCS_UWS")
        & global_models["outcome"].eq("max_dose_pci")
        & global_models["feature"].eq("stim_region_degree")
    ].iloc[0]
    target_axis.text(
        0.03,
        0.97,
        (
            f"Adjusted \N{GREEK SMALL LETTER BETA}="
            f"{target_result['standardized_beta']:.3f}\n"
            f"FDR q={target_result['q_fdr_within_population_outcome']:.2g}"
        ),
        transform=target_axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
    )
    target_axis.set_xlabel("Preserved links from left SMA")
    target_axis.set_ylabel("PCI at 0.766 occupancy")
    target_axis.legend(frameon=False, loc="lower right")
    panel_label(target_axis, "c", "Connectivity of the stimulated region")
    style_axis(target_axis, grid_axis="both")

    save_figure(
        figure,
        output_dir,
        "serotonergic_pci_responder_phenotyping_main",
    )
    plt.close(figure)


def plot_extended_figure(
    joined: pd.DataFrame,
    trajectories: pd.DataFrame,
    regional_models: pd.DataFrame,
    thresholds: dict[str, float],
    output_dir: Path,
) -> None:
    figure = plt.figure(
        figsize=(NATURE_DOUBLE_COLUMN_IN, 3.7),
        layout="constrained",
    )
    grid = figure.add_gridspec(1, 2, width_ratios=[0.95, 1.25])
    region_axis = figure.add_subplot(grid[0, 0])
    trajectory_axis = figure.add_subplot(grid[0, 1])

    regional = regional_models.loc[
        regional_models["outcome"].eq("max_dose_pci")
    ].nsmallest(12, "p_value")
    regional = regional.sort_values("standardized_beta")
    y_positions = np.arange(len(regional))
    significant = regional["q_fdr_across_90_regions"].lt(0.05)
    colors = np.where(significant, "#176D75", "#999999")
    for position, (_, row) in zip(
        y_positions,
        regional.iterrows(),
        strict=True,
    ):
        region_axis.errorbar(
            row["standardized_beta"],
            position,
            xerr=[
                [row["standardized_beta"] - row["ci95_low"]],
                [row["ci95_high"] - row["standardized_beta"]],
            ],
            fmt="o",
            markersize=3.3,
            markerfacecolor=colors[position] if significant.iloc[position] else "white",
            markeredgecolor=colors[position],
            markeredgewidth=0.6,
            ecolor=colors[position],
            elinewidth=0.75,
            capsize=1.7,
        )
    region_axis.axvline(0, color="#222222", linewidth=0.6)
    region_axis.set_yticks(y_positions, regional["region_label"])
    region_axis.set_xlabel(
        "Adjusted PCI difference per 1 s.d.\nmore preserved regional links"
    )
    panel_label(
        region_axis,
        "a",
        "Regional structural associations",
    )
    style_axis(region_axis, grid_axis="x")

    selected = trajectories.loc[
        trajectories["low_baseline_to_control_range"].fillna(False)
    ].copy()
    for (condition, subject_id), group in selected.groupby(
        ["condition", "subject_id"],
        observed=True,
    ):
        group = group.sort_values("occupancy")
        trajectory_axis.plot(
            group["occupancy"],
            group["pci_mean"],
            color=CONDITION_COLORS[str(condition)],
            linewidth=0.65,
            alpha=0.65,
        )
    trajectory_axis.axhline(
        thresholds["control_lower_pci_bound"],
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=0.7,
        label="Control lower bound",
    )
    occupancies = np.sort(selected["occupancy"].unique())
    trajectory_axis.set_xticks(
        occupancies,
        ["0", "0.25", "0.50", "0.766"],
    )
    trajectory_axis.set_xlabel(r"5-HT$_{2A}$ occupancy")
    trajectory_axis.set_ylabel("PCI")
    trajectory_axis.legend(frameon=False, loc="lower right")
    panel_label(
        trajectory_axis,
        "b",
        f"Low-baseline control-range entrants (n={selected['subject_id'].nunique()})",
    )
    style_axis(trajectory_axis, grid_axis="y")

    save_figure(
        figure,
        output_dir,
        "serotonergic_pci_responder_phenotyping_extended",
    )
    plt.close(figure)


def write_readme(
    path: Path,
    joined: pd.DataFrame,
    exact_tests: pd.DataFrame,
    global_models: pd.DataFrame,
    regional_models: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    patients = joined.loc[joined["is_patient"]]
    entrants = patients.loc[patients["low_baseline_to_control_range"]]
    target = global_models.loc[
        global_models["population"].eq("MCS_UWS")
        & global_models["outcome"].eq("max_dose_pci")
        & global_models["feature"].eq("stim_region_degree")
    ].iloc[0]
    significant_regions = regional_models.loc[
        regional_models["outcome"].eq("max_dose_pci")
        & regional_models["q_fdr_across_90_regions"].lt(0.05)
    ]
    primary_tests = exact_tests.loc[
        exact_tests["population"].eq("DOC")
        & exact_tests["outcome"].eq("control_referenced_response")
    ].set_index("predictor")
    text = f"""# Serotonergic PCI responder phenotyping

## Definitions

- The established control-referenced responder definition is unchanged: maximum-dose
  delta PCI exceeds the 95th percentile of the control delta-PCI distribution.
- The waking-control lower bound is the {thresholds["control_baseline_quantile"]:.1%}
  quantile of baseline control PCI ({thresholds["control_lower_pci_bound"]:.6f}).
- "Low baseline" is the bottom {thresholds["low_doc_baseline_quantile"]:.0%} of
  patient baseline PCI (PCI <= {thresholds["low_doc_baseline_cutoff"]:.6f}).
- A low-baseline control-range entrant starts below that patient cutoff and reaches
  or exceeds the waking-control lower bound at 0.766 occupancy. This is an
  in-silico dynamical classification, not evidence of clinical awakening.

## Main findings

- {len(entrants)}/{int(patients["low_baseline_doc"].sum())} low-baseline patients
  entered the control PCI range: {int(entrants["condition"].eq("COMA").sum())}
  COMA, {int(entrants["condition"].eq("UWS").sum())} UWS,
  {int(entrants["condition"].eq("MCS").sum())} MCS and
  {int(entrants["condition"].eq("EMCS").sum())} EMCS.
- Acute versus chronic status was not associated with the established responder
  classification (two-sided Fisher exact P={primary_tests.loc["stage", "p_value"]:.3f}).
- Sedated versus non-sedated status was not associated with the established
  responder classification (two-sided Fisher exact
  P={primary_tests.loc["sedation", "p_value"]:.3f}).
- Within MCS and UWS, preserved non-zero connections from the stimulated left
  supplementary motor area were associated with higher maximum-dose PCI after
  adjustment for baseline PCI, diagnosis, stage and sedation
  (standardized beta={target["standardized_beta"]:.3f},
  95% CI {target["ci95_low"]:.3f} to {target["ci95_high"]:.3f},
  FDR q={target["q_fdr_within_population_outcome"]:.3g}).
- Regions surviving FDR correction across 90 AAL regions were:
  {", ".join(significant_regions["region_label"].tolist())}.

## Important limitations

- Age, sex, aetiology, time since injury, CRS-R/GCS and clinical outcome were not
  found in the attached sources and therefore were not tested.
- Stage and sedation are observational source-cohort labels. Sedation was not an
  explicit pharmacological term in the simulations, so any association would be
  indirect through the subject connectome or cohort composition.
- The left-SMA association is strongest when severely disconnected subjects are
  included and weakens when analysis is restricted to the highly preserved range;
  see `target_degree_sensitivity.csv`.
- These are exploratory associations in simulated responses and do not identify
  patients likely to awaken or respond clinically to psilocybin.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    joined, trajectories, regional_features, thresholds = load_and_join(args)
    coverage = metadata_coverage(joined)
    summaries = subgroup_summary(joined)
    exact_tests = exact_metadata_tests(joined)
    continuous_tests = continuous_metadata_tests(joined)
    core_models = fit_core_models(joined)
    global_models = fit_global_structural_models(joined)
    regional_models = fit_regional_models(regional_features)
    threshold_table = threshold_sensitivity(joined)
    target_sensitivity = target_degree_sensitivity(joined)

    joined.sort_values(
        ["low_baseline_to_control_range", "max_dose_delta"],
        ascending=[False, False],
    ).to_csv(tables_dir / "subject_responder_phenotypes.csv", index=False)
    coverage.to_csv(tables_dir / "metadata_coverage.csv", index=False)
    summaries.to_csv(tables_dir / "metadata_subgroup_summary.csv", index=False)
    exact_tests.to_csv(tables_dir / "metadata_exact_tests.csv", index=False)
    continuous_tests.to_csv(
        tables_dir / "metadata_continuous_tests.csv",
        index=False,
    )
    core_models.to_csv(tables_dir / "adjusted_core_models.csv", index=False)
    global_models.to_csv(
        tables_dir / "adjusted_global_structural_models.csv",
        index=False,
    )
    regional_models.to_csv(
        tables_dir / "adjusted_regional_degree_models.csv",
        index=False,
    )
    threshold_table.to_csv(
        tables_dir / "control_range_threshold_sensitivity.csv",
        index=False,
    )
    target_sensitivity.to_csv(
        tables_dir / "target_degree_sensitivity.csv",
        index=False,
    )
    regional_features.to_csv(
        tables_dir / "subject_regional_structural_features.csv",
        index=False,
    )

    nature_style()
    plot_main_figure(
        joined,
        exact_tests,
        global_models,
        thresholds,
        figures_dir,
        args.jitter_seed,
    )
    plot_extended_figure(
        joined,
        trajectories,
        regional_models,
        thresholds,
        figures_dir,
    )
    write_readme(
        args.output_dir / "README.md",
        joined,
        exact_tests,
        global_models,
        regional_models,
        thresholds,
    )
    manifest = {
        "analysis": "exploratory in-silico responder phenotyping",
        "n_subjects": len(joined),
        "n_patients": int(joined["is_patient"].sum()),
        "stimulated_region": STIM_REGION_EXPECTED_LABEL,
        "stimulated_region_index_zero_based": STIM_REGION_ZERO_BASED,
        "thresholds": thresholds,
        "sources": {
            "subject_responder_table": str(
                (
                    args.analysis_dir
                    / f"{args.analysis_prefix}_subject_responder_table.csv"
                ).resolve()
            ),
            "subject_responder_table_sha256": sha256(
                args.analysis_dir
                / f"{args.analysis_prefix}_subject_responder_table.csv"
            ),
            "metadata_csv": str(args.metadata_csv),
            "metadata_csv_sha256": sha256(args.metadata_csv),
            "damage_csv": str(args.damage_csv),
            "damage_csv_sha256": sha256(args.damage_csv),
            "dataset_root": str(args.dataset_root),
        },
        "multiplicity": {
            "metadata_exact_tests": "Holm across all exact tests",
            "metadata_continuous_tests": "Holm across all rank tests",
            "global_structural_models": "Benjamini-Hochberg within population and outcome",
            "regional_models": "Benjamini-Hochberg across 90 regions within outcome",
        },
        "figure_format": {
            "font": "Arial",
            "width_mm": 183,
            "vector": ["pdf", "svg"],
            "raster": {"png_dpi": 600, "tiff_dpi": 300},
            "colour_space": "RGB",
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote responder-phenotyping analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
