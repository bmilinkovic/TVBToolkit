"""Small deterministic checks for the spatial-null simulation contract."""

from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest

from experiments.sma8s_receptor_topography.common import (
    ROOT, build_maps, make_manifest,
)
from tvbtoolkit.workflows.pharmacology import leak_to_conductances


def test_maps_keep_original_values_and_uniform_mean():
    with np.load(ROOT / "data/doc_data/converted_structural_invnodevol_control_max/atlas.npz") as atlas:
        labels = atlas["labels"].astype(str)
    names, maps, rows = build_maps(labels, 2, 2)
    assert names == ["original", "uniform", "spin_000", "spin_001",
                     "permutation_000", "permutation_001"]
    assert maps.shape == (6, 90)
    np.testing.assert_allclose(maps[1], maps[0].mean())
    for values in maps[2:]:
        np.testing.assert_array_equal(np.sort(values), np.sort(maps[0]))
    for row in rows[2:]:
        assert sorted(row["permutation"]) == list(range(90))
    names_again, maps_again, _ = build_maps(labels, 2, 2)
    assert names_again == names
    np.testing.assert_array_equal(maps_again, maps)


def test_permutation_only_maps():
    with np.load(ROOT / "data/doc_data/converted_structural_invnodevol_control_max/atlas.npz") as atlas:
        labels = atlas["labels"].astype(str)
    names, maps, _ = build_maps(labels, 0, 3)
    assert names == ["original", "uniform", "permutation_000",
                     "permutation_001", "permutation_002"]
    for values in maps[2:]:
        np.testing.assert_array_equal(np.sort(values), np.sort(maps[0]))


def test_manifest_contains_only_mcs_uws_and_one_shared_baseline():
    dataset = ROOT / "data/doc_data/converted_structural_invnodevol_control_max"
    manifest, _, _ = make_manifest(dataset, 1, 1)
    assert len(manifest["cases"]) == 126 * 5
    assert {case["cohort"] for case in manifest["cases"]} == {"mcs", "uws"}
    assert sum(case["map_id"] == "baseline" for case in manifest["cases"]) == 126
    assert sum(case["map_id"] == "original" for case in manifest["cases"]) == 126


def test_gk_override_keeps_original_tvb_trait_shape_and_uniform_effect():
    pytest.importorskip("tvb")
    from experiments.sma8s_receptor_topography.simulate import apply_receptor_map
    from experiments.sma8s_receptor_topography.common import PROTOCOL

    model = SimpleNamespace()
    for pop in ("e", "i"):
        baseline_gk, gna = leak_to_conductances(
            50.0, -90.0, PROTOCOL[f"E_L_{pop}_mv"], g_L=10.0)
        setattr(model, f"g_K_{pop}", np.full((90, 1), baseline_gk))
        setattr(model, f"g_Na_{pop}", np.asarray([gna]))
    sim = SimpleNamespace(model=model)
    apply_receptor_map(sim, np.full(90, .5), .766)
    for pop in ("e", "i"):
        baseline_gk, gna = leak_to_conductances(
            50.0, -90.0, PROTOCOL[f"E_L_{pop}_mv"], g_L=10.0)
        drug_gk, _ = leak_to_conductances(
            50.0, -90.0, PROTOCOL[f"drug_E_L_{pop}_mv"], g_Na=gna)
        expected = baseline_gk - .766 * .5 * (baseline_gk - drug_gk)
        assert getattr(model, f"g_K_{pop}").shape == (90, 1)
        np.testing.assert_allclose(getattr(model, f"g_K_{pop}"), expected)


def test_null_summary_pairs_each_map_with_one_subject_baseline(tmp_path: Path):
    import pandas as pd
    from experiments.sma8s_receptor_topography.analyze import _null_results

    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest, _, _ = make_manifest(
        ROOT / "data/doc_data/converted_structural_invnodevol_control_max", 1, 1)
    effect = {"baseline": 0., "original": .10, "uniform": .02,
              "spin_000": .04, "permutation_000": .01}
    rows = []
    for case in manifest["cases"]:
        rows.append(dict(**case, mean_pci_lz=.25 + effect[case["map_id"]]
                         * (1 if case["group"] == "MCS" else .5),
                         high_rate_post_transient_trials=0))
    _null_results(pd.DataFrame(rows), manifest, tmp_path)
    deltas = pd.read_csv(tmp_path / "delta_by_subject_and_map.csv")
    assert len(deltas) == 126 * 4
    np.testing.assert_allclose(
        deltas.loc[deltas.map_id.eq("original") & deltas.group.eq("MCS"),
                   "delta_pci_lz"], .10)
    tests = pd.read_csv(tmp_path / "null_tests.csv")
    assert len(tests) == 6
    assert (tmp_path / "figures/original_vs_spatial_nulls.pdf").exists()


def test_permutation_only_null_summary(tmp_path: Path):
    import pandas as pd
    from experiments.sma8s_receptor_topography.analyze import _null_results

    manifest, _, _ = make_manifest(
        ROOT / "data/doc_data/converted_structural_invnodevol_control_max", 0, 1)
    effect = {"baseline": 0., "original": .1, "uniform": .02,
              "permutation_000": .04}
    rows = [dict(**case, mean_pci_lz=.25 + effect[case["map_id"]],
                 high_rate_post_transient_trials=0)
            for case in manifest["cases"]]
    _null_results(pd.DataFrame(rows), manifest, tmp_path)
    tests = pd.read_csv(tmp_path / "null_tests.csv")
    assert len(tests) == 3
    assert set(tests["kind"]) == {"permutation"}


def test_one_permutation_paired_figure(tmp_path: Path):
    import pandas as pd
    from experiments.sma8s_receptor_topography.analyze import _single_permutation_figure

    manifest, _, _ = make_manifest(
        ROOT / "data/doc_data/converted_structural_invnodevol_control_max", 0, 1)
    effect = {"baseline": 0., "original": .1, "uniform": .02,
              "permutation_000": .04}
    rows = [dict(**case, mean_pci_lz=.25 + effect[case["map_id"]])
            for case in manifest["cases"]]
    _single_permutation_figure(pd.DataFrame(rows), tmp_path)
    assert (tmp_path / "figures/one_permutation_paired_comparison.pdf").is_file()


def test_recording_cleanup_requires_complete_derived_outputs(tmp_path: Path):
    import pandas as pd
    from experiments.sma8s_receptor_topography.analyze import _verify_and_delete_recordings

    bank, output = tmp_path / "bank", tmp_path / "pci"
    recordings = bank / "recordings"
    recordings.mkdir(parents=True)
    (recordings / "raw.bin").write_bytes(b"trial")
    (output / "cases").mkdir(parents=True)
    (output / "masks").mkdir()
    (output / "figures").mkdir()
    for index in range(2):
        (output / "cases" / f"case_{index:05d}.json").write_text(
            json.dumps({"case_index": index, "trials": [{}] * 100}))
        np.savez_compressed(output / "masks" / f"case_{index:05d}.npz",
                            binary=np.zeros((100, 90, 300), dtype=bool))
    pd.DataFrame({"mean_pci_lz": [.2, .3]}).to_csv(
        output / "subject_summary.csv", index=False)
    pd.DataFrame({"delta_pci_lz": [.1]}).to_csv(
        output / "delta_by_subject_and_map.csv", index=False)
    with (output / "trial_scores.csv").open("w") as stream:
        stream.write("trial\n")
        stream.writelines(f"{i}\n" for i in range(200))
    for name in ("null_tests.csv", "null_map_group_summary.csv",
                 "null_map_gradient_summary.csv"):
        (output / name).write_text("metric\n")
    (output / "completion.json").write_text(json.dumps({"bank_manifest_sha256": "abc"}))
    manifest = {"cases": [{"map_id": "baseline"}, {"map_id": "original"}],
                "n_spin": 0, "n_permutation": 2}
    with pytest.raises(ValueError, match="Missing downstream results"):
        _verify_and_delete_recordings(bank, output, manifest, "abc")
    assert (recordings / "raw.bin").is_file()
    (output / "figures/original_vs_spatial_nulls.pdf").write_bytes(b"figure")
    _verify_and_delete_recordings(bank, output, manifest, "abc")
    assert not recordings.exists()
    assert (bank / "recordings_deleted_after_pci.json").is_file()
