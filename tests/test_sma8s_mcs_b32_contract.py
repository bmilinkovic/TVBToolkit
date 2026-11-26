"""Guard the MCS-only sensitivity bank against protocol drift."""

import json
from pathlib import Path

import pytest

from experiments.sma8s_full100 import common as original
from experiments.sma8s_mcs_b32_full100 import contract


DATASET = (Path(__file__).resolve().parents[1] /
           "data/doc_data/converted_structural_invnodevol_control_max")


@pytest.mark.skipif(not (DATASET / "index.json").exists(), reason="SC dataset absent")
def test_mcs_only_manifest_matches_reference_except_b(tmp_path):
    reference = tmp_path / "original"
    reference.mkdir()
    old = original.manifest_for(DATASET, "all_noncoma")
    (reference / "manifest.json").write_text(json.dumps(old))
    new = contract.manifest_for(DATASET, reference)
    assert len(new["cases"]) == 300
    assert len(new["subjects"]["mcs"]) == 75
    assert {case["cohort"] for case in new["cases"]} == {"mcs"}
    assert {case["b_pa"] for case in new["cases"]} == {32.0}
    assert new["master_seed"] == old["master_seed"]
    assert new["dataset_index_sha256"] == old["dataset_index_sha256"]
    assert new["receptor_csv_sha256"] == old["receptor_csv_sha256"]
    for key in old["protocol"]:
        if key not in ("version", "b_pa_by_cohort"):
            assert new["protocol"][key] == old["protocol"][key]
    assert new["protocol"]["b_pa_by_cohort"] == {
        "control": 5.0, "emcs": 15.0, "mcs": 32.0, "uws": 40.0}

    old["model_source_sha256"]["src/tvbtoolkit/whole_brain/simulation.py"] = "wrong"
    (reference / "manifest.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="model source differs"):
        contract.manifest_for(DATASET, reference)
