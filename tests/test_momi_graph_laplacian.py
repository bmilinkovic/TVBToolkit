from __future__ import annotations

import numpy as np
import pytest

from experiments.momi_laplacian_pci.graph_laplacian import (
    momi_laplacian,
    validate_tract_lengths,
)


def test_momi_laplacian_has_expected_sign_scale_and_spectrum() -> None:
    adjacency = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    transformed, diagnostics = momi_laplacian(adjacency)
    unscaled = np.diag(adjacency.sum(axis=1)) - adjacency

    np.testing.assert_allclose(transformed, unscaled / np.linalg.norm(unscaled))
    np.testing.assert_allclose(transformed.sum(axis=1), 0.0, atol=1e-15)
    assert np.all(np.diag(transformed) > 0.0)
    assert np.all(transformed[np.triu_indices(3, 1)] <= 0.0)
    assert np.linalg.eigvalsh(transformed)[0] == pytest.approx(0.0, abs=1e-14)
    assert np.linalg.norm(transformed) == pytest.approx(1.0)
    assert diagnostics.nonzero_undirected_edges == 2
    assert diagnostics.disconnected_edge_fraction == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 3)),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    ],
)
def test_momi_laplacian_rejects_invalid_adjacency(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        momi_laplacian(bad)


def test_tract_lengths_are_validated_but_not_folded_into_operator() -> None:
    adjacency = np.array([[0.0, 2.0], [2.0, 0.0]])
    lengths = np.array([[0.0, 40.0], [40.0, 0.0]])
    transformed, _ = momi_laplacian(adjacency)
    diagnostics = validate_tract_lengths(lengths, adjacency)

    np.testing.assert_allclose(transformed, np.array([[0.5, -0.5], [-0.5, 0.5]]))
    assert diagnostics["tract_length_median_mm"] == 40.0

    with pytest.raises(ValueError, match="lack tract lengths"):
        validate_tract_lengths(np.zeros((2, 2)), adjacency)
