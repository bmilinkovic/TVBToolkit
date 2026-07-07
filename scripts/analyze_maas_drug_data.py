"""Initial QC and functional-connectivity summaries for Maas drug fMRI data.

The source files are MATLAB v7.3/HDF5 cell arrays under the external
``data_drugs_maastricht/raw/drugs_data`` tree.
This script decodes the cells, joins the companion metadata workbook, and writes
small CSV summaries under ``results/maas_drug_data_initial``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from tvbtoolkit.core.paths import drugs_raw, drugs_results

DATA_DIR = drugs_raw("drugs_data")
OUT_DIR = drugs_results("maas_drug_data_initial")


@dataclass(frozen=True)
class Scan:
    dataset: str
    subject: str
    condition: str
    run: str
    ts: np.ndarray


def _decode_ref(file: h5py.File, ref):
    obj = file[ref]
    arr = obj[()]
    if arr.dtype == np.uint16:
        return "".join(map(chr, arr.flatten(order="F"))).strip("\x00")
    return np.asarray(arr, dtype=float)


def load_psil_between(path: Path) -> list[Scan]:
    scans: list[Scan] = []
    with h5py.File(path, "r") as file:
        cell = file["roi_data"]
        subjects = [_decode_ref(file, cell[0, col]) for col in range(1, cell.shape[1])]
        groups = [_decode_ref(file, cell[1, col]) for col in range(1, cell.shape[1])]
        for col, (subject, group) in enumerate(zip(subjects, groups), start=1):
            scans.append(
                Scan(
                    dataset="psil_between",
                    subject=str(int(subject)),
                    condition=str(group),
                    run="run_1",
                    ts=_decode_ref(file, cell[2, col]),
                )
            )
    return scans


def load_psil_within(path: Path) -> list[Scan]:
    scans: list[Scan] = []
    with h5py.File(path, "r") as file:
        cell = file["roi_data"]
        subjects = [_decode_ref(file, cell[0, col]) for col in range(1, cell.shape[1])]
        for row, condition in ((1, "pla"), (2, "psil")):
            for col, subject in enumerate(subjects, start=1):
                scans.append(
                    Scan(
                        dataset="psil_within",
                        subject=str(int(subject)),
                        condition=condition,
                        run="run_1",
                        ts=_decode_ref(file, cell[row, col]),
                    )
                )
    return scans


def load_thc_within(path: Path) -> list[Scan]:
    scans: list[Scan] = []
    with h5py.File(path, "r") as file:
        cell = file["roi_data"]
        subjects = [_decode_ref(file, cell[0, col]) for col in range(1, cell.shape[1])]
        for row in range(1, cell.shape[0]):
            label = _decode_ref(file, cell[row, 0])
            condition, run = label.split("_", 1)
            for col, subject in enumerate(subjects, start=1):
                scans.append(
                    Scan(
                        dataset="thc_within",
                        subject=str(int(subject)),
                        condition=condition,
                        run=f"run_{run}",
                        ts=_decode_ref(file, cell[row, col]),
                    )
                )
    return scans


def scan_metrics(scan: Scan) -> dict[str, float | str | int]:
    ts = np.asarray(scan.ts, dtype=float)
    if ts.shape[0] != 116 and ts.shape[1] == 116:
        ts = ts.T
    finite_var = np.nanstd(ts, axis=1) > 0
    fc = np.corrcoef(ts[finite_var])
    upper = fc[np.triu_indices_from(fc, k=1)]
    global_signal = ts.mean(axis=0)
    return {
        "dataset": scan.dataset,
        "subject": scan.subject,
        "condition": scan.condition,
        "run": scan.run,
        "n_regions": int(ts.shape[0]),
        "n_timepoints": int(ts.shape[1]),
        "n_constant_regions": int((~finite_var).sum()),
        "nan_count": int(np.isnan(ts).sum()),
        "ts_mean": float(np.nanmean(ts)),
        "ts_std": float(np.nanstd(ts)),
        "mean_abs_roi_corr": float(np.nanmean(np.abs(upper))),
        "mean_roi_corr": float(np.nanmean(upper)),
        "global_signal_std": float(np.nanstd(global_signal)),
    }


def duplicate_checks(scans: list[Scan]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    by_subject: dict[tuple[str, str], list[Scan]] = {}
    for scan in scans:
        by_subject.setdefault((scan.dataset, scan.subject), []).append(scan)

    for (dataset, subject), subject_scans in by_subject.items():
        for i, left in enumerate(subject_scans):
            for right in subject_scans[i + 1 :]:
                left_ts = np.asarray(left.ts, dtype=float)
                right_ts = np.asarray(right.ts, dtype=float)
                common_regions = min(left_ts.shape[0], right_ts.shape[0])
                common_timepoints = min(left_ts.shape[1], right_ts.shape[1])
                diff = left_ts[:common_regions, :common_timepoints] - right_ts[:common_regions, :common_timepoints]
                rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "left": f"{left.condition}_{left.run}",
                        "right": f"{right.condition}_{right.run}",
                        "common_regions": int(common_regions),
                        "common_timepoints": int(common_timepoints),
                        "max_abs_diff": float(np.nanmax(np.abs(diff))),
                    }
                )
    return pd.DataFrame(rows)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(metadata_path, sheet_name=None)

    psil_between = sheets["psil_between "].copy()
    psil_between["dataset"] = "psil_between"
    psil_between["subject"] = psil_between["sub"].astype(int).astype(str)
    psil_between["condition"] = psil_between["drug"].astype(str)
    psil_between["run"] = "run_1"
    psil_between = psil_between.rename(columns={"emax_psilocin_pk_ngml": "emax_pk_ng"})

    psil_within = sheets["psil_within"].copy()
    psil_within["dataset"] = "psil_within"
    psil_within["subject"] = psil_within["participant"].astype(int).astype(str)
    psil_within["condition"] = psil_within["drug"].astype(str)
    psil_within["run"] = "run_1"

    thc_within = sheets["thc_within"].copy()
    thc_within["dataset"] = "thc_within"
    thc_within["subject"] = thc_within["sub"].astype(int).astype(str)
    thc_within["condition"] = thc_within["drug"].astype(str)
    thc_within["run"] = "run_" + thc_within["run"].astype(int).astype(str)

    cols = [
        "dataset",
        "subject",
        "condition",
        "run",
        "scan_complete",
        "weight_kg",
        "emax_pk_ng",
        "scrub_perc",
        "fd_mean",
    ]
    return pd.concat([psil_between[cols], psil_within[cols], thc_within[cols]], ignore_index=True)


def paired_summary(metrics: pd.DataFrame, dataset: str, drug: str, value: str) -> dict[str, float | str | int]:
    df = metrics[metrics["dataset"].eq(dataset)].copy()
    agg = df.groupby(["subject", "condition"], as_index=False)[value].mean()
    wide = agg.pivot(index="subject", columns="condition", values=value).dropna(subset=["pla", drug])
    delta = wide[drug] - wide["pla"]
    test = stats.ttest_rel(wide[drug], wide["pla"])
    return {
        "dataset": dataset,
        "comparison": f"{drug}-pla",
        "metric": value,
        "n_pairs": int(len(wide)),
        "placebo_mean": float(wide["pla"].mean()),
        "drug_mean": float(wide[drug].mean()),
        "delta_mean": float(delta.mean()),
        "delta_sd": float(delta.std(ddof=1)),
        "t": float(test.statistic),
        "p": float(test.pvalue),
    }


def between_summary(metrics: pd.DataFrame, value: str) -> dict[str, float | str | int]:
    df = metrics[metrics["dataset"].eq("psil_between")].copy()
    pla = df[df["condition"].eq("pla")][value]
    psil = df[df["condition"].eq("psil")][value]
    test = stats.ttest_ind(psil, pla, equal_var=False)
    return {
        "dataset": "psil_between",
        "comparison": "psil-pla",
        "metric": value,
        "n_placebo": int(pla.size),
        "n_drug": int(psil.size),
        "placebo_mean": float(pla.mean()),
        "drug_mean": float(psil.mean()),
        "delta_mean": float(psil.mean() - pla.mean()),
        "t": float(test.statistic),
        "p": float(test.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scans = [
        *load_psil_between(DATA_DIR / "maas_psilbetween_ts_aal116_noGSR.mat"),
        *load_psil_within(DATA_DIR / "maas_psilwithin_ts_aal116_noGSR.mat"),
        *load_thc_within(DATA_DIR / "maas_thcwithin_ts_aal116_noGSR.mat"),
    ]
    metrics = pd.DataFrame(scan_metrics(scan) for scan in scans)
    duplicates = duplicate_checks(scans)
    metadata = load_metadata(DATA_DIR / "metadata.xlsx")
    joined = metrics.merge(
        metadata,
        how="left",
        on=["dataset", "subject", "condition", "run"],
        validate="one_to_one",
    )

    inventory = (
        joined.groupby(["dataset", "condition", "run"], dropna=False)
        .agg(
            n_scans=("subject", "count"),
            n_complete=("scan_complete", "sum"),
            n_timepoints_min=("n_timepoints", "min"),
            n_timepoints_max=("n_timepoints", "max"),
            fd_mean=("fd_mean", "mean"),
            scrub_perc_mean=("scrub_perc", "mean"),
            mean_abs_roi_corr=("mean_abs_roi_corr", "mean"),
            global_signal_std=("global_signal_std", "mean"),
        )
        .reset_index()
    )

    comparisons = pd.DataFrame(
        [
            paired_summary(joined, "psil_within", "psil", "mean_abs_roi_corr"),
            paired_summary(joined, "psil_within", "psil", "global_signal_std"),
            paired_summary(joined, "thc_within", "thc", "mean_abs_roi_corr"),
            paired_summary(joined, "thc_within", "thc", "global_signal_std"),
            between_summary(joined, "mean_abs_roi_corr"),
            between_summary(joined, "global_signal_std"),
        ]
    )

    joined.to_csv(args.out_dir / "scan_level_metrics.csv", index=False)
    inventory.to_csv(args.out_dir / "inventory_qc_summary.csv", index=False)
    comparisons.to_csv(args.out_dir / "first_pass_fc_comparisons.csv", index=False)
    duplicates.to_csv(args.out_dir / "duplicate_condition_checks.csv", index=False)
    print(f"Wrote {args.out_dir}")
    print(inventory.to_string(index=False))
    print()
    print(comparisons.to_string(index=False))
    print()
    print(
        duplicates.groupby("dataset")["max_abs_diff"]
        .agg(n_pairs="count", n_zero_diff=lambda s: int((s == 0).sum()), max_abs_diff_max="max")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
