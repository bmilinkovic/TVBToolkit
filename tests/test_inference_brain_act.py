from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from tvbtoolkit.inference.brain_act import (
    aal90_interleaved_to_symmetric_index,
    load_brain_act_bold_mat,
)


def test_aal90_reorder_index_is_a_permutation() -> None:
    index = aal90_interleaved_to_symmetric_index()
    assert index.shape == (90,)
    np.testing.assert_array_equal(np.sort(index), np.arange(90))
    np.testing.assert_array_equal(index[:4], [0, 2, 4, 6])
    np.testing.assert_array_equal(index[-4:], [7, 5, 3, 1])


def test_load_brain_act_mat_orients_reorders_and_normalizes(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    sc = rng.uniform(size=(90, 90))
    sc = 0.5 * (sc + sc.T)
    np.fill_diagonal(sc, 0.0)
    bold = np.tile(np.arange(90, dtype=float)[:, None], (1, 80))
    bold += rng.normal(scale=0.01, size=bold.shape)
    lengths = np.ones((90, 90), dtype=float)
    np.fill_diagonal(lengths, 0.0)
    source = tmp_path / "subject.mat"
    scipy.io.savemat(
        source,
        {"SC": sc, "BOLD": bold, "subject_id": "sub-01", "cohort": "control"},
    )

    record = load_brain_act_bold_mat(source, tract_lengths=lengths)

    assert record.bold.shape == (80, 90)
    assert record.structural_connectivity.shape == (90, 90)
    assert np.max(record.structural_connectivity) == pytest.approx(1.0)
    assert record.tract_lengths is not None
    assert record.subject_id == "sub-01"
    expected_first_row = bold[:, 0][aal90_interleaved_to_symmetric_index()]
    np.testing.assert_allclose(record.bold[0], expected_first_row)


def test_load_brain_act_mat_rejects_nonfinite_bold(tmp_path: Path) -> None:
    sc = np.ones((3, 3), dtype=float) - np.eye(3)
    bold = np.ones((3, 10), dtype=float)
    bold[0, 0] = np.nan
    source = tmp_path / "bad.mat"
    scipy.io.savemat(source, {"SC": sc, "BOLD": bold})

    with pytest.raises(ValueError, match="NaN or Inf"):
        load_brain_act_bold_mat(source, roi_order="as_stored")
