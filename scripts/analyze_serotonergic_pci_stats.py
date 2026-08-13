#!/usr/bin/env python3
"""Subject-level statistics for the serotonergic PCI analysis."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


CONDITION_ORDER = ["COMA", "UWS", "MCS", "EMCS", "CNT"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/tables/serotonergic_pci_subject_metrics_with_rescue.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/outputs/serotonergic_pci_full_50trials/tables/stats"),
    )
    return p.parse_args()


def fdr_bh(p_values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    p_finite = p[finite]
    if p_finite.size == 0:
        return q
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(p_finite)
    out[order] = adjusted
    q[finite] = out
    return q


def welch_anova(groups: list[np.ndarray]) -> dict[str, float]:
    clean = [np.asarray(g, dtype=float) for g in groups]
    clean = [g[np.isfinite(g)] for g in clean if np.isfinite(g).sum() > 1]
    k = len(clean)
    if k < 2:
        return {"f": np.nan, "df_num": np.nan, "df_den": np.nan, "p": np.nan}

    n = np.array([len(g) for g in clean], dtype=float)
    means = np.array([np.mean(g) for g in clean], dtype=float)
    variances = np.array([np.var(g, ddof=1) for g in clean], dtype=float)
    if np.any(variances <= 0.0):
        return {"f": np.nan, "df_num": np.nan, "df_den": np.nan, "p": np.nan}

    weights = n / variances
    weight_sum = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / weight_sum)
    correction_terms = (1.0 / (n - 1.0)) * (1.0 - weights / weight_sum) ** 2
    correction = float(np.sum(correction_terms))
    df_num = float(k - 1)
    df_den = float((k**2 - 1.0) / (3.0 * correction))
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / df_num)
    denominator = float(1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * correction)
    f_value = numerator / denominator
    p_value = float(stats.f.sf(f_value, df_num, df_den))
    return {"f": f_value, "df_num": df_num, "df_den": df_den, "p": p_value}


def mean_ci(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = sd / float(np.sqrt(n)) if n > 0 else np.nan
    if n > 1:
        margin = float(stats.t.ppf(0.975, n - 1) * sem)
    else:
        margin = 0.0
    return mean, sd, sem, mean - margin, mean + margin


def one_sample_table(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        values = df.loc[df["condition"].eq(condition), value_col].to_numpy(float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        mean, sd, sem, ci_low, ci_high = mean_ci(values)
        if n > 1 and sd > 0:
            test = stats.ttest_1samp(values, 0.0)
            p = float(test.pvalue)
            t = float(test.statistic)
            cohen_d = mean / sd
        else:
            p = np.nan
            t = np.nan
            cohen_d = np.nan
        rows.append(
            {
                "effect": label,
                "condition": condition,
                "n_subjects": n,
                "mean": mean,
                "sd": sd,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "t": t,
                "p": p,
                "cohen_dz": cohen_d,
            }
        )
    out = pd.DataFrame(rows)
    out["q_fdr"] = fdr_bh(out["p"])
    return out


def pairwise_table(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
    rows = []
    for a, b in combinations(CONDITION_ORDER, 2):
        va = df.loc[df["condition"].eq(a), value_col].to_numpy(float)
        vb = df.loc[df["condition"].eq(b), value_col].to_numpy(float)
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        test = stats.ttest_ind(va, vb, equal_var=False)
        rows.append(
            {
                "effect": label,
                "contrast": f"{a} - {b}",
                "condition_a": a,
                "condition_b": b,
                "n_a": int(va.size),
                "n_b": int(vb.size),
                "mean_a": float(np.mean(va)),
                "mean_b": float(np.mean(vb)),
                "mean_diff": float(np.mean(va) - np.mean(vb)),
                "t_welch": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    out["q_fdr"] = fdr_bh(out["p"])
    return out


def planned_emcs_contrasts(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
    rows = []
    emcs = df.loc[df["condition"].eq("EMCS"), value_col].to_numpy(float)
    emcs = emcs[np.isfinite(emcs)]
    for condition in [c for c in CONDITION_ORDER if c != "EMCS"]:
        other = df.loc[df["condition"].eq(condition), value_col].to_numpy(float)
        other = other[np.isfinite(other)]
        test = stats.ttest_ind(emcs, other, equal_var=False)
        rows.append(
            {
                "effect": label,
                "contrast": f"EMCS - {condition}",
                "condition_a": "EMCS",
                "condition_b": condition,
                "n_a": int(emcs.size),
                "n_b": int(other.size),
                "mean_a": float(np.mean(emcs)),
                "mean_b": float(np.mean(other)),
                "mean_diff": float(np.mean(emcs) - np.mean(other)),
                "t_welch": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    out["q_fdr"] = fdr_bh(out["p"])
    return out


def subject_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive = df[df["occupancy"].gt(0.0)].copy()
    for (condition, subject_id), g in positive.groupby(["condition", "subject_id"], observed=True):
        x = g["occupancy"].to_numpy(float)
        y = g["pci_rescue"].to_numpy(float)
        if len(np.unique(x)) < 2:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        rows.append(
            {
                "condition": condition,
                "subject_id": subject_id,
                "slope_delta_pci_per_occupancy": float(slope),
                "intercept": float(intercept),
            }
        )
    return pd.DataFrame(rows)


def responder_table(max_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        values = max_df.loc[max_df["condition"].eq(condition), "pci_rescue"].to_numpy(float)
        rows.append(
            {
                "condition": condition,
                "n_subjects": int(values.size),
                "n_delta_pci_gt_0": int(np.sum(values > 0.0)),
                "prop_delta_pci_gt_0": float(np.mean(values > 0.0)),
                "n_delta_pci_gt_0_05": int(np.sum(values > 0.05)),
                "prop_delta_pci_gt_0_05": float(np.mean(values > 0.05)),
            }
        )
    return pd.DataFrame(rows)


def fit_mixed_occupancy_model(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = df.copy()
    model_df["subject_key"] = model_df["cohort"].astype(str) + ":" + model_df["subject_id"].astype(str)
    model_df["condition"] = pd.Categorical(model_df["condition"], categories=["EMCS", "COMA", "UWS", "MCS", "CNT"])

    full = smf.mixedlm(
        "pci_rescue ~ occupancy * condition",
        model_df,
        groups=model_df["subject_key"],
        re_formula="~occupancy",
    ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    reduced = smf.mixedlm(
        "pci_rescue ~ occupancy + condition",
        model_df,
        groups=model_df["subject_key"],
        re_formula="~occupancy",
    ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)

    lr = float(2.0 * (full.llf - reduced.llf))
    df_diff = int(full.df_modelwc - reduced.df_modelwc)
    comparison = pd.DataFrame(
        [
            {
                "analysis": label,
                "model": "full: pci_rescue ~ occupancy * condition + (occupancy | subject)",
                "n_observations": int(model_df.shape[0]),
                "n_subjects": int(model_df["subject_key"].nunique()),
                "full_converged": bool(full.converged),
                "reduced_converged": bool(reduced.converged),
                "full_log_likelihood": float(full.llf),
                "reduced_log_likelihood": float(reduced.llf),
                "likelihood_ratio_chi2": lr,
                "df": df_diff,
                "p": float(stats.chi2.sf(lr, df_diff)),
            }
        ]
    )

    fixed_rows = []
    ci = full.conf_int()
    for term in full.params.index:
        fixed_rows.append(
            {
                "analysis": label,
                "term": str(term),
                "estimate": float(full.params[term]),
                "se": float(full.bse[term]) if term in full.bse.index else np.nan,
                "z": float(full.tvalues[term]) if term in full.tvalues.index else np.nan,
                "p": float(full.pvalues[term]) if term in full.pvalues.index else np.nan,
                "ci95_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                "ci95_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
            }
        )
    fixed_effects = pd.DataFrame(fixed_rows)

    contrast_rows = []
    for condition in ["COMA", "UWS", "MCS", "CNT"]:
        term = f"occupancy:condition[T.{condition}]"
        estimate_other_minus_emcs = float(full.params[term])
        contrast_rows.append(
            {
                "analysis": label,
                "contrast": f"EMCS slope - {condition} slope",
                "condition_a": "EMCS",
                "condition_b": condition,
                "mean_slope_diff": -estimate_other_minus_emcs,
                "se": float(full.bse[term]),
                "z": float(-full.tvalues[term]),
                "p": float(full.pvalues[term]),
                "ci95_low": float(-ci.loc[term, 1]),
                "ci95_high": float(-ci.loc[term, 0]),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["q_fdr"] = fdr_bh(contrasts["p"])
    return comparison, fixed_effects, contrasts


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_occ = float(df["occupancy"].max())
    max_df = df[df["occupancy"].eq(max_occ)].copy()
    slopes = subject_slopes(df)

    summary = (
        df.groupby(["condition", "occupancy"], observed=True)
        .agg(
            n_subjects=("subject_id", "nunique"),
            pci_mean=("pci_mean", "mean"),
            pci_sd=("pci_mean", "std"),
            pci_rescue_mean=("pci_rescue", "mean"),
            pci_rescue_sd=("pci_rescue", "std"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "dose_summary_by_condition.csv", index=False)
    slopes.to_csv(args.output_dir / "subject_dose_slopes.csv", index=False)

    max_one = one_sample_table(max_df, "pci_rescue", f"max-dose delta PCI at occupancy {max_occ:g}")
    slope_one = one_sample_table(slopes, "slope_delta_pci_per_occupancy", "linear dose-response slope")
    max_pair = pairwise_table(max_df, "pci_rescue", f"max-dose delta PCI at occupancy {max_occ:g}")
    slope_pair = pairwise_table(slopes, "slope_delta_pci_per_occupancy", "linear dose-response slope")
    slope_emcs = planned_emcs_contrasts(slopes, "slope_delta_pci_per_occupancy", "linear dose-response slope")
    max_emcs = planned_emcs_contrasts(max_df, "pci_rescue", f"max-dose delta PCI at occupancy {max_occ:g}")
    responders = responder_table(max_df)
    slope_groups = [
        slopes.loc[slopes["condition"].eq(condition), "slope_delta_pci_per_occupancy"].to_numpy(float)
        for condition in CONDITION_ORDER
    ]
    slope_omnibus = pd.DataFrame([{"effect": "linear dose-response slope", **welch_anova(slope_groups)}])
    mixed_positive = fit_mixed_occupancy_model(df[df["occupancy"].gt(0.0)].copy(), "positive_doses_primary")
    mixed_all = fit_mixed_occupancy_model(df.copy(), "all_doses_sensitivity")

    max_one.to_csv(args.output_dir / "max_dose_one_sample_tests.csv", index=False)
    slope_one.to_csv(args.output_dir / "dose_slope_one_sample_tests.csv", index=False)
    max_pair.to_csv(args.output_dir / "max_dose_pairwise_welch_tests.csv", index=False)
    slope_pair.to_csv(args.output_dir / "dose_slope_pairwise_welch_tests.csv", index=False)
    slope_emcs.to_csv(args.output_dir / "dose_slope_planned_emcs_contrasts.csv", index=False)
    max_emcs.to_csv(args.output_dir / "max_dose_planned_emcs_contrasts.csv", index=False)
    slope_omnibus.to_csv(args.output_dir / "dose_slope_welch_anova.csv", index=False)
    mixed_positive[0].to_csv(args.output_dir / "mixed_model_positive_doses_lrt.csv", index=False)
    mixed_positive[1].to_csv(args.output_dir / "mixed_model_positive_doses_fixed_effects.csv", index=False)
    mixed_positive[2].to_csv(args.output_dir / "mixed_model_positive_doses_emcs_slope_contrasts.csv", index=False)
    mixed_all[0].to_csv(args.output_dir / "mixed_model_all_doses_lrt.csv", index=False)
    mixed_all[1].to_csv(args.output_dir / "mixed_model_all_doses_fixed_effects.csv", index=False)
    mixed_all[2].to_csv(args.output_dir / "mixed_model_all_doses_emcs_slope_contrasts.csv", index=False)
    responders.to_csv(args.output_dir / "max_dose_responder_counts.csv", index=False)

    print(f"Wrote statistics tables to {args.output_dir}")
    print("Omnibus Welch ANOVA on subject-level positive-dose slopes:")
    print(slope_omnibus.to_string(index=False))
    print("\nPrimary mixed model likelihood-ratio test:")
    print(mixed_positive[0].to_string(index=False))
    print("\nPrimary mixed model planned EMCS slope contrasts:")
    print(mixed_positive[2][["contrast", "mean_slope_diff", "z", "p", "q_fdr"]].to_string(index=False))
    print("\nPlanned EMCS-vs-other slope contrasts:")
    print(slope_emcs[["contrast", "mean_a", "mean_b", "mean_diff", "t_welch", "p", "q_fdr"]].to_string(index=False))


if __name__ == "__main__":
    main()
