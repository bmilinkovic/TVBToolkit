"""Assemble a virtual full cohort: old CNT/EMCS/UWS + new MCS b=32 PCI.

No raw recordings or binary masks are copied. The merged tables carry a
provenance manifest pointing to the two underlying PCI/bank directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ("CNT", "EMCS", "MCS", "UWS")
COLORS = {"CNT": "#dc92bb", "EMCS": "#bc72c0",
          "MCS": "#8d4bad", "UWS": "#4a226c"}
OCC = (0.0, 0.25, 0.5, 0.766)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-bank", type=Path, required=True)
    parser.add_argument("--original-pci", type=Path, required=True,
                        help="Original first-order hybrid PCI directory")
    parser.add_argument("--mcs-bank", type=Path, required=True)
    parser.add_argument("--mcs-pci", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_bank_manifest = args.original_bank / "manifest.json"
    new_bank_manifest = args.mcs_bank / "manifest.json"
    old_manifest = json.loads(old_bank_manifest.read_text())
    new_manifest = json.loads(new_bank_manifest.read_text())
    old_completion = json.loads((args.original_pci / "completion.json").read_text())
    new_completion = json.loads((args.mcs_pci / "completion.json").read_text())
    if old_completion != dict(method="hybrid", cases=716, trials=71600,
                              bank_manifest_sha256=sha256(old_bank_manifest)):
        raise ValueError("Original hybrid PCI is incomplete or from another bank")
    if new_completion != dict(method="hybrid", cases=300, trials=30000,
                              bank_manifest_sha256=sha256(new_bank_manifest)):
        raise ValueError("MCS b=32 hybrid PCI is incomplete or from another bank")
    if new_manifest["reference_manifest_sha256"] != sha256(old_bank_manifest):
        raise ValueError("The MCS simulation bank was not prepared against this original bank")
    old_method = json.loads((args.original_pci / "methods.json").read_text())
    new_method = json.loads((args.mcs_pci / "methods.json").read_text())
    if (old_method.get("method") != "hybrid" or new_method.get("method") != "hybrid"
            or old_method.get("algorithm_sha256") != new_method.get("algorithm_sha256")
            or old_method.get("bootstrap_draws") != new_method.get("bootstrap_draws")
            or old_method.get("alpha_nominal") != new_method.get("alpha_nominal")):
        raise ValueError("Original and new PCI algorithms or thresholds differ")
    changed = {"version", "b_pa_by_cohort"}
    if ({key: value for key, value in old_manifest["protocol"].items() if key not in changed}
            != {key: value for key, value in new_manifest["protocol"].items() if key not in changed}):
        raise ValueError("Simulation protocols differ beyond cohort b and version")
    if (new_manifest["protocol"]["b_pa_by_cohort"]["mcs"] != 32.0 or
            old_manifest["protocol"]["b_pa_by_cohort"]["mcs"] != 35.0):
        raise ValueError("Wrong MCS b values")

    old_subject = pd.read_csv(args.original_pci / "subject_summary.csv")
    new_subject = pd.read_csv(args.mcs_pci / "subject_summary.csv")
    old_trials = pd.read_csv(args.original_pci / "trial_scores.csv")
    new_trials = pd.read_csv(args.mcs_pci / "trial_scores.csv")
    if (len(old_subject), len(new_subject), len(old_trials), len(new_trials)) != (
            716, 300, 71600, 30000):
        raise ValueError("Input table row counts do not match completed banks")
    if (set(old_subject.columns) != set(new_subject.columns) or
            set(old_trials.columns) != set(new_trials.columns)):
        raise ValueError("Original and new PCI table schemas differ")
    if set(new_subject.group) != {"MCS"} or set(new_trials.group) != {"MCS"}:
        raise ValueError("New PCI contains non-MCS subjects")
    if (new_subject.groupby("subject").size().ne(4).any() or
            new_trials.groupby(["subject", "occupancy"]).size().ne(100).any()):
        raise ValueError("New PCI does not cover every MCS subject/occupancy/trial")
    keys = ["cohort", "subject", "occupancy"]
    original_mcs = old_subject.loc[old_subject.group.eq("MCS")]
    original_keys = original_mcs[keys].sort_values(keys).reset_index(drop=True)
    new_keys = new_subject[keys].sort_values(keys).reset_index(drop=True)
    if not original_keys.equals(new_keys):
        raise ValueError("MCS subject/occupancy keys do not match the original bank")

    # Check that the same random seeds and pulse onsets were used trial-by-trial.
    trial_keys = [*keys, "trial"]
    pair = old_trials.loc[old_trials.group.eq("MCS"), trial_keys + ["seed", "onset_ms"]].merge(
        new_trials[trial_keys + ["seed", "onset_ms"]], on=trial_keys,
        validate="one_to_one", suffixes=("_old", "_new"))
    if (len(pair) != 30000 or
            not np.array_equal(pair.seed_old.to_numpy(), pair.seed_new.to_numpy()) or
            not np.array_equal(pair.onset_ms_old.to_numpy(), pair.onset_ms_new.to_numpy())):
        raise ValueError("MCS trial seeds or pulse onsets are not matched")

    # Restore original case indices so downstream joins retain one unique index.
    case_map = original_mcs[keys + ["case_index"]].rename(columns={"case_index": "original_index"})
    def restore_case_index(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.merge(case_map, on=keys, validate="many_to_one")
        result["case_index"] = result.pop("original_index")
        return result

    new_subject = restore_case_index(new_subject)
    new_trials = restore_case_index(new_trials)
    combined_subject = pd.concat([old_subject.loc[old_subject.group.ne("MCS")], new_subject],
                                 ignore_index=True).sort_values("case_index")
    combined_trials = pd.concat([old_trials.loc[old_trials.group.ne("MCS")], new_trials],
                                ignore_index=True).sort_values(["case_index", "trial"])
    if (len(combined_subject), len(combined_trials)) != (716, 71600):
        raise ValueError("Combined cohort is incomplete")
    if combined_subject.case_index.nunique() != 716:
        raise ValueError("Combined case indices are not unique")
    args.output.mkdir(parents=True, exist_ok=True)
    combined_subject.to_csv(args.output / "subject_summary.csv", index=False)
    combined_trials.to_csv(args.output / "trial_scores.csv", index=False)
    provenance = dict(method="hybrid", model_order=1, cases=716, trials=71600,
                      b_pa_by_cohort={"control": 5.0, "emcs": 15.0,
                                      "mcs": 32.0, "uws": 40.0},
                      original_bank=str(args.original_bank),
                      original_pci=str(args.original_pci),
                      mcs_b32_bank=str(args.mcs_bank), mcs_b32_pci=str(args.mcs_pci),
                      original_bank_manifest_sha256=sha256(old_bank_manifest),
                      mcs_bank_manifest_sha256=sha256(new_bank_manifest),
                      original_pci_subject_summary_sha256=sha256(args.original_pci / "subject_summary.csv"),
                      mcs_pci_subject_summary_sha256=sha256(args.mcs_pci / "subject_summary.csv"),
                      masks={"CNT_EMCS_UWS": str(args.original_pci / "masks"),
                             "MCS_b32": str(args.mcs_pci / "masks")})
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for group in GROUPS:
        data = combined_subject.loc[combined_subject.group.eq(group)]
        curves = data.pivot(index="subject", columns="occupancy", values="mean_pci_lz")
        curves = curves.reindex(columns=OCC)
        ax.plot(OCC, curves.mean(axis=0), color=COLORS[group], lw=2.9,
                label=f"{group} (n={len(curves)})")
        for values in curves.to_numpy():
            ax.plot(OCC, values, color=COLORS[group], alpha=.07, lw=.6)
    ax.set(xlabel="5-HT2A occupancy", ylabel="PCI", xticks=OCC)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "pci_by_occupancy_combined.pdf")
    fig.savefig(figures / "pci_by_occupancy_combined.png", dpi=230)
    plt.close(fig)
    print("Combined 179 subjects, 716 cases and 71,600 trial scores; only MCS was re-simulated.")


if __name__ == "__main__":
    main()
