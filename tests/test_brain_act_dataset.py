from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from tvbtoolkit.datasets.brain_act import (
    convert_brain_act_dataset,
    list_subjects,
    load_aal90_atlas,
    load_subject_structural,
    validate_structural_matrices,
)


def _write_lookup(path: Path, n_regions: int) -> None:
    lines = ["# synthetic lookup", "0 ??? Unknown 0 0 0 0"]
    for i in range(1, n_regions + 1):
        lines.append(f"{i} R{i:03d} Region_{i:03d} 255 0 0 255")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_subject(sc_path: Path, tl_path: Path, n_regions: int, scale: float) -> None:
    rng = np.random.default_rng(int(scale * 10))
    m = rng.uniform(0.0, scale, size=(n_regions, n_regions))
    c = 0.5 * (m + m.T)
    np.fill_diagonal(c, 0.0)
    l = rng.uniform(5.0, 200.0, size=(n_regions, n_regions))
    l = 0.5 * (l + l.T)
    np.fill_diagonal(l, 0.0)
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    tl_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        sc_path,
        {
            "structural_connectome": c,
            "subject_id": "sub-01",
            "condition": "CNT",
        },
    )
    np.savetxt(tl_path, l)


def _build_minimal_source_tree(tmp_path: Path, n_regions: int = 8) -> Path:
    data_root = tmp_path / "brain-act" / "data"
    (data_root / "atlases").mkdir(parents=True)
    _write_lookup(data_root / "atlases" / "custom_lookuptable_AAL.txt", n_regions=n_regions)

    for cohort in ("CNT", "MCS", "UWS"):
        _write_subject(
            data_root / "organized" / "structural_connectomes" / cohort / "sub-01_structural_connectome.mat",
            data_root / "organized" / "tract_lengths" / cohort / "sub-01_tract_lengths.txt",
            n_regions=n_regions,
            scale={"CNT": 1.0, "MCS": 2.0, "UWS": 3.0}[cohort],
        )
    return data_root


def test_convert_and_list_subjects(tmp_path: Path):
    source = _build_minimal_source_tree(tmp_path)
    out = tmp_path / "converted"
    index_path = convert_brain_act_dataset(source, out)
    assert index_path.exists()

    all_subjects = list_subjects(out)
    assert set(all_subjects.keys()) == {"control", "mcs", "uws"}
    assert all_subjects["control"] == ["sub-01"]
    assert list_subjects(out, cohort="CNT") == ["sub-01"]
    assert list_subjects(out, cohort="control") == ["sub-01"]

    atlas = load_aal90_atlas(out)
    assert atlas.n_regions == 8
    assert atlas.labels[0] == "Region_001"


def test_load_subject_structural_requires_cohort_when_ambiguous(tmp_path: Path):
    source = _build_minimal_source_tree(tmp_path)
    out = tmp_path / "converted"
    convert_brain_act_dataset(source, out)

    with pytest.raises(ValueError):
        load_subject_structural("sub-01", out)

    c, l, atlas, meta = load_subject_structural(
        "sub-01",
        out,
        cohort="control",
        validate=True,
        normalize="max",
        percentile=50.0,
    )
    assert c.shape == (8, 8)
    assert l.shape == (8, 8)
    assert atlas.n_regions == 8
    assert meta.cohort == "control"
    assert np.allclose(np.diag(c), 0.0)
    assert np.allclose(np.diag(l), 0.0)


def test_validate_structural_matrices_cleaning():
    c = np.array(
        [
            [np.nan, 2.0, 0.0],
            [1.0, 0.0, np.inf],
            [0.0, 3.0, 0.0],
        ]
    )
    l = np.array(
        [
            [0.0, 10.0, 20.0],
            [11.0, 0.0, 30.0],
            [19.0, 30.0, 0.0],
        ]
    )
    c_out, l_out, report = validate_structural_matrices(
        c,
        l,
        nonfinite="zero",
        enforce_symmetry=True,
        threshold=0.5,
        normalize=None,
    )
    assert c_out.shape == (3, 3)
    assert l_out.shape == (3, 3)
    assert np.isfinite(c_out).all()
    assert np.isfinite(l_out).all()
    assert np.allclose(c_out, c_out.T)
    assert np.allclose(l_out, l_out.T)
    assert report["tract_length_sanity"]["is_plausible"] is True

