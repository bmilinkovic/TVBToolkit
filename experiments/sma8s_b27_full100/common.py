"""Separate MCS/UWS b=27/42 pA sensitivity contract for the SMA bank.

The published first- and second-order simulation implementations remain
unchanged. This contract alters only MCS and UWS b relative to their
respective frozen 5/15/35/40-pA protocols.
"""

from __future__ import annotations

from pathlib import Path

from experiments.sma8s_full100 import common as first
from experiments.sma8s_full100_second_order import common as second

ROOT = first.ROOT
TRIALS = first.TRIALS
COHORTS = {"control": ("CNT", 5.0), "emcs": ("EMCS", 15.0),
           "mcs": ("MCS", 27.0), "uws": ("UWS", 42.0)}
OCCUPANCIES = first.OCCUPANCIES
atomic_json = first.atomic_json
sha256_file = first.sha256_file
validate_dataset = first.validate_dataset


def protocol_for(order: int) -> dict:
    """Return a copy of the original protocol with only MCS/UWS b changed."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    original = first.PROTOCOL if order == 1 else second.PROTOCOL
    protocol = dict(original)
    protocol["b_pa_by_cohort"] = dict(original["b_pa_by_cohort"])
    protocol["b_pa_by_cohort"]["mcs"] = 27.0
    protocol["b_pa_by_cohort"]["uws"] = 42.0
    protocol["version"] = f"sma8s_el62_full100_b27_u42_order{order}_v1"
    return protocol


def manifest_for(dataset: Path, scope: str, order: int) -> dict:
    """Preserve subjects, occupancies and seeds; record the changed b explicitly."""
    original = first.manifest_for if order == 1 else second.manifest_for
    manifest = original(dataset, scope)
    manifest["protocol"] = protocol_for(order)
    for case in manifest["cases"]:
        case["b_pa"] = COHORTS[case["cohort"]][1]
    sources = manifest["model_source_sha256"]
    engine = (ROOT / "experiments/sma8s_full100/simulate.py" if order == 1
              else ROOT / "experiments/sma8s_full100_second_order/simulate.py")
    for path in (engine,
                 ROOT / "experiments/sma8s_full100/common.py",
                 ROOT / "experiments/sma8s_full100_second_order/common.py"):
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
    return manifest
