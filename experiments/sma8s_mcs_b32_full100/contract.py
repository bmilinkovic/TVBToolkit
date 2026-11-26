"""MCS-only b=32 pA sensitivity bank for the frozen first-order SMA protocol.

All other parameters, subjects, occupancies, seed rules and pulse timing are
inherited from sma8s_full100. This is deliberately a separate bank.
"""

import json
from pathlib import Path

from experiments.sma8s_full100 import common as original

ROOT = original.ROOT
TRIALS = original.TRIALS
OCCUPANCIES = original.OCCUPANCIES
atomic_json = original.atomic_json
sha256_file = original.sha256_file
validate_dataset = original.validate_dataset


def protocol() -> dict:
    result = dict(original.PROTOCOL)
    result["b_pa_by_cohort"] = dict(original.PROTOCOL["b_pa_by_cohort"])
    result["b_pa_by_cohort"]["mcs"] = 32.0
    result["version"] = "sma8s_el62_full100_mcs_b32_order1_v1"
    return result


def verify_reference(reference_bank: Path, candidate: dict) -> str:
    """Refuse to mix banks if the old simulation inputs/code are not identical."""
    reference_path = reference_bank / "manifest.json"
    reference = json.loads(reference_path.read_text())
    if reference["protocol"] != original.PROTOCOL or reference["scope"] != "all_noncoma":
        raise ValueError("Reference bank is not the original first-order full100 protocol")
    for key in ("dataset_index_sha256", "receptor_csv_sha256", "master_seed"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Reference bank differs in {key}")
    for relative, digest in reference["model_source_sha256"].items():
        if candidate["model_source_sha256"].get(relative) != digest:
            raise ValueError(f"Reference bank model source differs: {relative}")
    old_mcs = [case for case in reference["cases"] if case["cohort"] == "mcs"]
    if len(old_mcs) != 300 or reference["subjects"]["mcs"] != candidate["subjects"]["mcs"]:
        raise ValueError("Reference bank MCS subject set differs")
    for old, new in zip(old_mcs, candidate["cases"]):
        if (old["subject"], old["occupancy"]) != (new["subject"], new["occupancy"]):
            raise ValueError("Reference bank MCS case ordering differs")
    return sha256_file(reference_path)


def manifest_for(dataset: Path, reference_bank: Path) -> dict:
    result = original.manifest_for(dataset, "all_noncoma")
    result["protocol"] = protocol()
    result["scope"] = "mcs_only"
    result["subjects"] = {"mcs": result["subjects"]["mcs"]}
    result["cases"] = [dict(case, b_pa=32.0) for case in result["cases"]
                       if case["cohort"] == "mcs"]
    # Freeze the unchanged engine and contract used to produce each trial.
    for relative in ("experiments/sma8s_full100/simulate.py",
                     "experiments/sma8s_full100/common.py"):
        result["model_source_sha256"][relative] = sha256_file(ROOT / relative)
    assert len(result["subjects"]["mcs"]) == 75
    assert len(result["cases"]) == 75 * len(OCCUPANCIES)
    result["reference_manifest_sha256"] = verify_reference(reference_bank, result)
    return result
