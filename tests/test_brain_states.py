from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.analysis.brain_states import (
    cluster_brain_states,
    phase_patterns,
    summarize_brain_states,
)


def test_phase_patterns_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 12))
    patterns, sync, iu, ju = phase_patterns(x, trim_edge_samples=5)

    assert patterns.ndim == 2
    assert sync.ndim == 1
    assert patterns.shape[0] == sync.shape[0]
    assert patterns.shape[1] == iu.size == ju.size


def test_phase_patterns_legacy_shapes() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(320, 10))
    patterns, sync, iu, ju = phase_patterns(
        x,
        trim_edge_samples=0,
        pipeline="brain_act_legacy",
        tr_seconds=2.4,
        bandpass_hz=(0.01, 0.20),
        filter_order=3,
    )

    assert patterns.ndim == 2
    assert sync.ndim == 1
    assert patterns.shape[0] == sync.shape[0]
    assert patterns.shape[1] == iu.size == ju.size


def test_cluster_brain_states_returns_labels_and_centers() -> None:
    rng = np.random.default_rng(1)
    patterns = rng.normal(size=(200, 30))
    labels, centers = cluster_brain_states(patterns, n_states=4, random_seed=1, n_init=5)

    assert labels.shape == (200,)
    assert centers.shape[0] == 4
    assert centers.shape[1] == 30
    assert np.min(labels) >= 0
    assert np.max(labels) <= 3


def test_cluster_brain_states_sklearn_backend() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(12)
    patterns = rng.normal(size=(180, 22))
    labels, centers = cluster_brain_states(
        patterns,
        n_states=5,
        random_seed=12,
        n_init=8,
        max_iter=120,
        backend="sklearn",
    )

    assert labels.shape == (180,)
    assert centers.shape == (5, 22)


def test_summarize_brain_states_outputs_probabilities() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 10))
    out = summarize_brain_states(x, n_states=5, trim_edge_samples=5, random_seed=2, n_init=5)

    assert out.occupancy.ndim == 1
    assert np.isclose(out.occupancy.sum(), 1.0, atol=1e-6)
    assert out.transition_matrix.shape[0] == out.transition_matrix.shape[1]


def test_summarize_brain_states_legacy_pipeline() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(21)
    x = rng.normal(size=(450, 14))
    out = summarize_brain_states(
        x,
        n_states=5,
        trim_edge_samples=0,
        random_seed=21,
        n_init=10,
        max_iter=150,
        pipeline="brain_act_legacy",
        clustering_backend="sklearn",
        tr_seconds=2.4,
        bandpass_hz=(0.01, 0.20),
        filter_order=3,
    )

    assert out.labels.shape[0] == out.global_synchrony.shape[0]
    assert out.centers.shape[0] == 5
    assert np.isclose(out.occupancy.sum(), 1.0, atol=1e-6)
