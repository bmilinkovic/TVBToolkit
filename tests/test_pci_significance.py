from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.complexity.pci_casali import (
    CasaliSignificanceResult,
    binarise_signals_casali,
)


def _slow_pre_post_reference(
    signal: np.ndarray,
    swap_matrix: np.ndarray,
    *,
    alpha: float,
    two_sided: bool = True,
) -> dict[str, np.ndarray | float]:
    """Transparent reference implementation for small test tensors."""
    data = np.asarray(signal, dtype=float)
    swaps = np.asarray(swap_matrix, dtype=bool)
    n_trials, n_sources, n_bins = data.shape
    t_stim = n_bins // 2

    baseline_mean = data[:, :, :t_stim].mean(axis=(0, 2))
    baseline_sd = data[:, :, :t_stim].std(axis=(0, 2), ddof=1)
    centred = data - baseline_mean[np.newaxis, :, np.newaxis]
    averaged_response = centred.mean(axis=0) / baseline_sd[:, np.newaxis]
    observed_statistic = averaged_response * np.sqrt(n_trials)
    observed_post = observed_statistic[:, t_stim:]
    observed_for_test = (
        np.abs(observed_post) if two_sided else observed_post
    )

    pre = data[:, :, :t_stim]
    post = data[:, :, t_stim:]
    null_maxima = np.empty((swaps.shape[0], t_stim), dtype=float)
    for permutation_index, swap in enumerate(swaps):
        permuted_pre = pre.copy()
        permuted_post = post.copy()
        permuted_pre[swap] = post[swap]
        permuted_post[swap] = pre[swap]

        permuted_mean = permuted_pre.mean(axis=(0, 2))
        permuted_sd = permuted_pre.std(axis=(0, 2), ddof=1)
        permuted_statistic = (
            permuted_post.mean(axis=0)
            - permuted_mean[:, np.newaxis]
        ) / (permuted_sd[:, np.newaxis] / np.sqrt(n_trials))
        values = (
            np.abs(permuted_statistic)
            if two_sided
            else permuted_statistic
        )
        null_maxima[permutation_index] = values.max(axis=0)

    corrected_post = np.empty((n_sources, t_stim), dtype=float)
    for time_index in range(t_stim):
        comparison_scale = max(
            1.0,
            float(np.max(np.abs(null_maxima[:, time_index]))),
            float(np.max(np.abs(observed_for_test[:, time_index]))),
        )
        tie_tolerance = 512.0 * np.finfo(float).eps * comparison_scale
        corrected_post[:, time_index] = (
            1
            + np.count_nonzero(
                null_maxima[:, time_index, np.newaxis]
                >= (
                    observed_for_test[np.newaxis, :, time_index]
                    - tie_tolerance
                ),
                axis=0,
            )
        ) / float(swaps.shape[0] + 1)

    threshold_by_time = np.quantile(
        null_maxima,
        1.0 - alpha,
        axis=0,
        method="higher",
    )
    corrected = np.ones((n_sources, n_bins), dtype=float)
    corrected[:, t_stim:] = corrected_post
    binary = np.zeros((n_sources, n_bins), dtype=np.uint8)
    binary[:, t_stim:] = corrected_post <= alpha
    return {
        "averaged_response": averaged_response,
        "observed_statistic": observed_statistic,
        "null_maxima": null_maxima,
        "corrected_p_values": corrected,
        "threshold_by_time": threshold_by_time,
        "threshold": float(np.max(threshold_by_time)),
        "binary": binary,
    }


def test_pre_post_swap_matches_slow_reference_exactly() -> None:
    rng = np.random.default_rng(12)
    signal = rng.normal(size=(5, 4, 12))
    signal[:, 0, 7:10] += 1.25
    swap_matrix = rng.integers(0, 2, size=(31, 5), dtype=np.uint8)
    alpha = 0.2

    expected = _slow_pre_post_reference(
        signal,
        swap_matrix,
        alpha=alpha,
    )
    result = binarise_signals_casali(
        signal,
        t_stim=6,
        alpha=alpha,
        swap_matrix=swap_matrix,
        chunk_size=3,
        return_details=True,
    )

    assert isinstance(result, CasaliSignificanceResult)
    assert result.significance_method == "pre_post_swap"
    assert result.fwer_scope == "sources_per_response_time"
    np.testing.assert_allclose(
        result.averaged_response,
        expected["averaged_response"],
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.observed_statistic,
        expected["observed_statistic"],
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.null_maxima,
        expected["null_maxima"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.corrected_p_values,
        expected["corrected_p_values"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        result.threshold_by_time,
        expected["threshold_by_time"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.threshold == pytest.approx(expected["threshold"])
    np.testing.assert_array_equal(result.binary, expected["binary"])
    np.testing.assert_array_equal(result.binary[:, :6], 0)


def test_pre_post_swap_is_reproducible_and_chunk_invariant() -> None:
    signal = np.random.default_rng(21).normal(size=(9, 5, 16))

    np.random.seed(1)
    first = binarise_signals_casali(
        signal,
        t_stim=8,
        n_bootstrap=97,
        alpha=0.05,
        seed=44,
        chunk_size=1,
        return_details=True,
    )
    np.random.seed(999_999)
    second = binarise_signals_casali(
        signal,
        t_stim=8,
        n_bootstrap=97,
        alpha=0.05,
        seed=44,
        chunk_size=13,
        return_details=True,
    )

    assert first.rng_bit_generator == "PCG64"
    assert first.swap_matrix_sha256 == second.swap_matrix_sha256
    np.testing.assert_allclose(
        first.null_maxima,
        second.null_maxima,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_array_equal(first.binary, second.binary)
    np.testing.assert_array_equal(
        first.corrected_p_values,
        second.corrected_p_values,
    )


def test_pre_post_swap_is_source_offset_and_scale_invariant() -> None:
    rng = np.random.default_rng(31)
    signal = rng.normal(size=(7, 4, 14))
    signal[:, 2, 9:12] -= 1.5
    swap_matrix = rng.integers(0, 2, size=(101, 7), dtype=np.uint8)
    scales = np.asarray([0.25, 2.0, 7.0, 0.75])
    offsets = np.asarray([100.0, -5.0, 0.25, 17.0])
    transformed = (
        signal * scales[np.newaxis, :, np.newaxis]
        + offsets[np.newaxis, :, np.newaxis]
    )

    original = binarise_signals_casali(
        signal,
        t_stim=7,
        alpha=0.05,
        swap_matrix=swap_matrix,
        return_details=True,
    )
    rescaled = binarise_signals_casali(
        transformed,
        t_stim=7,
        alpha=0.05,
        swap_matrix=swap_matrix,
        return_details=True,
    )

    np.testing.assert_allclose(
        original.observed_statistic,
        rescaled.observed_statistic,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.null_maxima,
        rescaled.null_maxima,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_array_equal(
        original.corrected_p_values,
        rescaled.corrected_p_values,
    )
    np.testing.assert_array_equal(original.binary, rescaled.binary)


def test_corrected_p_values_use_plus_one_monte_carlo_rule() -> None:
    rng = np.random.default_rng(41)
    signal = rng.normal(size=(4, 3, 10))
    swap_matrix = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    result = binarise_signals_casali(
        signal,
        t_stim=5,
        alpha=0.2,
        swap_matrix=swap_matrix,
        return_details=True,
    )

    post_p = result.corrected_p_values[:, 5:]
    assert np.min(post_p) >= 1.0 / 5.0
    np.testing.assert_allclose(post_p * 5.0, np.round(post_p * 5.0))
    np.testing.assert_array_equal(
        result.binary[:, 5:],
        (post_p <= 0.2).astype(np.uint8),
    )


def test_pre_post_swap_requires_equal_duration_blocks() -> None:
    signal = np.random.default_rng(51).normal(size=(5, 3, 12))
    with pytest.raises(
        ValueError,
        match="equal prestimulus and poststimulus",
    ):
        binarise_signals_casali(signal[:, :, :-1], t_stim=6)


def test_pre_post_swap_rejects_nonfinite_and_zero_sd_sources() -> None:
    signal = np.random.default_rng(61).normal(size=(5, 3, 12))
    nonfinite = signal.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        binarise_signals_casali(nonfinite, t_stim=6)

    constant_baseline = signal.copy()
    constant_baseline[:, 1, :6] = 3.0
    with pytest.raises(ValueError, match="Prestimulus SD"):
        binarise_signals_casali(constant_baseline, t_stim=6)


def test_prescribed_swap_matrix_is_validated() -> None:
    signal = np.random.default_rng(71).normal(size=(5, 3, 12))
    with pytest.raises(ValueError, match="swap_matrix must have shape"):
        binarise_signals_casali(
            signal,
            t_stim=6,
            swap_matrix=np.zeros((4, 4), dtype=bool),
        )
    with pytest.raises(ValueError, match="only 0/1"):
        binarise_signals_casali(
            signal,
            t_stim=6,
            swap_matrix=np.full((4, 5), 2),
        )
    with pytest.raises(ValueError, match="only valid"):
        binarise_signals_casali(
            signal,
            t_stim=6,
            significance_method="temporal_shuffle",
            swap_matrix=np.zeros((4, 5), dtype=bool),
        )


@pytest.mark.parametrize("method", ["temporal_shuffle", "trial_bootstrap"])
def test_legacy_nulls_remain_explicit_sensitivity_methods(method: str) -> None:
    signal = np.random.default_rng(81).normal(size=(6, 4, 12))
    result = binarise_signals_casali(
        signal,
        t_stim=6,
        n_bootstrap=31,
        alpha=0.1,
        seed=9,
        significance_method=method,
        return_details=True,
    )

    assert result.significance_method == method
    assert result.fwer_scope == "sources_and_prestimulus_time_global"
    assert result.null_maxima.shape == (31,)
    assert result.swap_matrix_sha256 is None
