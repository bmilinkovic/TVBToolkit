"""Frozen second-order variant of the SMA full-cohort recording contract.

All subject, seed, onset, connectome and stimulation choices are inherited
from the first-order contract. The model order and covariance audit differ.
"""

from __future__ import annotations

from experiments.sma8s_full100 import common as first

ROOT = first.ROOT
TRIALS = first.TRIALS
COHORTS = first.COHORTS
OCCUPANCIES = first.OCCUPANCIES
atomic_json = first.atomic_json
recording_path = first.recording_path
sha256_file = first.sha256_file
trial_seed = first.trial_seed
trial_onset_ms = first.trial_onset_ms
validate_dataset = first.validate_dataset

PROTOCOL = dict(first.PROTOCOL)
PROTOCOL.update(
    version="sma8s_el62_full100_second_order_scaled_v1",
    model_order=2,
    derivative_adapter="scaled_rhs: Jacobian /1000, Hessian /1000000; experimental attenuation, not a proven unit conversion",
    covariance_policy="direct integration; no clipping or PSD projection; audit minimum eigenvalue at each raw step",
)
PROTOCOL.pop("second_order")


def manifest_for(dataset, scope):
    manifest = first.manifest_for(dataset, scope)
    manifest["protocol"] = PROTOCOL
    adapter = ROOT / "experiments/gnak_order_comparison/model_adapter.py"
    manifest["model_source_sha256"][str(adapter.relative_to(ROOT))] = sha256_file(adapter)
    base_contract = ROOT / "experiments/sma8s_full100/common.py"
    manifest["model_source_sha256"][str(base_contract.relative_to(ROOT))] = sha256_file(base_contract)
    return manifest
