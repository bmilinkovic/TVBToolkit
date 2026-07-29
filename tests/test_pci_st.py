import numpy as np
import pytest

from tvbtoolkit.complexity.pci_st import PCIStResult, pci_st, pci_st_from_trials


COMMON = {
    "k": 1.2,
    "min_snr": 1.1,
    "max_var_percent": 99.0,
    "n_steps": 100,
    "max_threshold_fraction": 1.0,
}


def _one_component_fixture():
    times = np.arange(-5.0, 5.0, 1.0)
    evoked = np.array([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 4.0, 0.0, -4.0, 0.0]])
    kwargs = {
        **COMMON,
        "baseline_window_ms": (-5.0, 0.0),
        "response_window_ms": (0.0, 5.0),
    }
    return evoked, times, kwargs


def _two_component_fixture():
    times = np.arange(-8.0, 8.0, 1.0)
    evoked = np.array(
        [
            [
                0.0,
                0.5,
                0.0,
                -0.5,
                0.0,
                0.5,
                0.0,
                -0.5,
                4.0,
                -4.0,
                4.0,
                -4.0,
                4.0,
                -4.0,
                4.0,
                -4.0,
            ],
            [
                0.25,
                0.0,
                -0.25,
                0.0,
                0.25,
                0.0,
                -0.25,
                0.0,
                2.0,
                2.0,
                -2.0,
                -2.0,
                2.0,
                2.0,
                -2.0,
                -2.0,
            ],
        ]
    )
    kwargs = {
        **COMMON,
        "baseline_window_ms": (-8.0, 0.0),
        "response_window_ms": (0.0, 8.0),
    }
    return evoked, times, kwargs


def _snr_rejected_fixture():
    times = np.arange(-6.0, 6.0, 1.0)
    evoked = np.array(
        [
            [
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
            ],
            [
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                0.25,
            ],
        ]
    )
    kwargs = {
        **COMMON,
        "baseline_window_ms": (-6.0, 0.0),
        "response_window_ms": (0.0, 6.0),
    }
    return evoked, times, kwargs


def test_official_reference_fixture_one_component():
    evoked, times, kwargs = _one_component_fixture()
    result = pci_st(evoked, times, return_details=True, **kwargs)

    assert isinstance(result, PCIStResult)
    assert result.pci_st == pytest.approx(3.2, abs=1e-12)
    np.testing.assert_allclose(result.component_contributions, [3.2], atol=1e-12)
    np.testing.assert_allclose(result.retained_snrs, [4.0], atol=1e-12)
    np.testing.assert_allclose(
        result.singular_values,
        [5.656854249492381],
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.optimal_threshold_indices, [15])
    np.testing.assert_allclose(
        result.optimal_thresholds,
        [2.0606060606060606],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.optimal_nst_response, [0.64], atol=1e-12)
    np.testing.assert_allclose(result.optimal_nst_baseline, [0.0], atol=1e-12)


def test_official_reference_fixture_two_components():
    evoked, times, kwargs = _two_component_fixture()
    result = pci_st(evoked, times, return_details=True, **kwargs)

    assert isinstance(result, PCIStResult)
    assert result.pci_st == pytest.approx(10.0, abs=1e-12)
    assert result.n_components == 2
    np.testing.assert_allclose(
        result.component_contributions,
        [7.0, 3.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.retained_snrs,
        [11.313708498984761, 11.313708498984761],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.singular_values,
        [11.313708498984761, 5.656854249492381],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.explained_variance_percent,
        [80.0, 20.0],
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.optimal_threshold_indices, [7, 7])
    np.testing.assert_allclose(
        result.optimal_thresholds,
        [1.0303030303030303, 0.5151515151515151],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.optimal_nst_response,
        [0.875, 0.375],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.optimal_nst_baseline, [0.0, 0.0], atol=1e-12)


def test_official_reference_fixture_rejects_low_snr_components():
    evoked, times, kwargs = _snr_rejected_fixture()
    result = pci_st(evoked, times, return_details=True, **kwargs)

    assert isinstance(result, PCIStResult)
    assert result.pci_st == 0.0
    assert result.n_components == 0
    assert result.component_contributions.size == 0
    assert result.retained_snrs.size == 0
    np.testing.assert_allclose(result.snr_before_filter, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(
        result.singular_values,
        [1.224744871391589, 0.6123724356957946],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.explained_variance_percent,
        [80.0, 20.0],
        atol=1e-12,
    )


def test_scale_channel_permutation_and_sign_invariance():
    evoked, times, kwargs = _two_component_fixture()
    reference = pci_st(evoked, times, **kwargs)
    transformed = np.array([[-1.0], [1.0]]) * evoked[[1, 0], :] * 7.25

    assert pci_st(transformed, times, **kwargs) == pytest.approx(
        reference,
        abs=1e-12,
    )
    assert pci_st(evoked * 1e-12, times, **kwargs) == pytest.approx(
        reference,
        abs=1e-12,
    )


def test_windows_are_half_open_like_official_python_reference():
    evoked, times, kwargs = _one_component_fixture()
    extended_times = np.append(times, 5.0)
    extended_evoked = np.column_stack([evoked, np.array([1e9])])

    result = pci_st(
        extended_evoked,
        extended_times,
        return_details=True,
        **kwargs,
    )

    assert isinstance(result, PCIStResult)
    assert result.pci_st == pytest.approx(3.2, abs=1e-12)
    assert result.n_baseline_samples == 5
    assert result.n_response_samples == 5
    assert 5.0 not in result.response_sample_times_ms


def test_zero_response_returns_zero_without_nan_diagnostics():
    times = np.arange(-5.0, 5.0)
    evoked = np.zeros((3, times.size))
    result = pci_st(
        evoked,
        times,
        baseline_window_ms=(-5.0, 0.0),
        response_window_ms=(0.0, 5.0),
        return_details=True,
    )

    assert isinstance(result, PCIStResult)
    assert result.pci_st == 0.0
    assert result.n_components == 0
    assert np.isfinite(result.explained_variance_percent).all()


def test_trial_convenience_function_averages_before_pcist():
    evoked, times, kwargs = _two_component_fixture()
    rng = np.random.default_rng(123)
    perturbation = rng.normal(scale=0.75, size=evoked.shape)
    trials = np.stack([evoked + perturbation, evoked - perturbation])

    expected = pci_st(evoked, times, return_details=True, **kwargs)
    result = pci_st_from_trials(
        trials,
        times,
        baseline_center_trials=False,
        return_details=True,
        **kwargs,
    )
    reversed_result = pci_st_from_trials(
        trials[::-1],
        times,
        baseline_center_trials=False,
        **kwargs,
    )

    assert isinstance(expected, PCIStResult)
    assert isinstance(result, PCIStResult)
    assert result.pci_st == pytest.approx(expected.pci_st, abs=1e-12)
    assert reversed_result == pytest.approx(expected.pci_st, abs=1e-12)
    assert result.n_trials_averaged == 2
    assert not result.trial_baseline_centered


def test_trial_baseline_centering_matches_explicit_preprocessing():
    evoked, times, kwargs = _two_component_fixture()
    trials = np.stack([evoked + 10.0, evoked - 3.0])
    baseline_mask = (times >= -8.0) & (times < 0.0)
    centered = trials - trials[:, :, baseline_mask].mean(axis=2, keepdims=True)
    expected = pci_st(centered.mean(axis=0), times, **kwargs)

    result = pci_st_from_trials(trials, times, **kwargs)

    assert result == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("evoked", "times", "match"),
    [
        (np.zeros(10), np.arange(10.0), "shape"),
        (np.zeros((2, 10)), np.arange(9.0), "length"),
        (
            np.zeros((2, 10)),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0, 9.0]),
            "strictly increasing",
        ),
        (
            np.column_stack([np.zeros((2, 9)), np.full(2, np.nan)]),
            np.arange(10.0),
            "finite",
        ),
    ],
)
def test_input_validation(evoked, times, match):
    with pytest.raises(ValueError, match=match):
        pci_st(
            evoked,
            times,
            baseline_window_ms=(0.0, 4.0),
            response_window_ms=(5.0, 9.0),
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("k", 0.9),
        ("k", 1.0),
        ("min_snr", -1.0),
        ("max_var_percent", 0.0),
        ("max_var_percent", 101.0),
        ("n_steps", 1),
        ("max_threshold_fraction", 0.0),
        ("embedding_dimension", 0),
        ("embedding_delay_samples", 0),
    ],
)
def test_parameter_validation(parameter, value):
    evoked, times, kwargs = _one_component_fixture()
    kwargs[parameter] = value
    with pytest.raises(ValueError):
        pci_st(evoked, times, **kwargs)


def test_trial_shape_is_explicit_and_not_inferred():
    times = np.arange(-5.0, 5.0)
    with pytest.raises(ValueError, match="trials, channels, time"):
        pci_st_from_trials(
            np.zeros((3, times.size)),
            times,
            baseline_window_ms=(-5.0, 0.0),
            response_window_ms=(0.0, 5.0),
        )


def test_irregular_sampling_is_rejected():
    times = np.arange(-10.0, 10.0)
    times[12:] += 0.2
    with pytest.raises(ValueError, match="uniformly sampled"):
        pci_st(
            np.ones((2, times.size)),
            times,
            baseline_window_ms=(-10.0, -2.0),
            response_window_ms=(0.0, 9.0),
        )


def test_truncated_requested_window_is_rejected():
    times = np.arange(-10.0, 10.0)
    with pytest.raises(ValueError, match="does not cover.*baseline_window_ms"):
        pci_st(
            np.ones((2, times.size)),
            times,
            baseline_window_ms=(-20.0, -2.0),
            response_window_ms=(0.0, 9.0),
        )


def test_sampling_and_effective_ranges_are_reported():
    evoked, times, kwargs = _one_component_fixture()
    result = pci_st(evoked, times, return_details=True, **kwargs)

    assert isinstance(result, PCIStResult)
    assert result.sampling_interval_ms == pytest.approx(
        float(np.median(np.diff(times)))
    )
    assert result.sampling_rate_hz == pytest.approx(
        1000.0 / result.sampling_interval_ms
    )
    assert result.effective_baseline_sample_range_ms == pytest.approx(
        (
            float(result.baseline_sample_times_ms[0]),
            float(result.baseline_sample_times_ms[-1]),
        )
    )
    assert result.effective_response_sample_range_ms == pytest.approx(
        (
            float(result.response_sample_times_ms[0]),
            float(result.response_sample_times_ms[-1]),
        )
    )


def test_trial_wrapper_rejects_double_baseline_correction():
    evoked, times, kwargs = _one_component_fixture()
    with pytest.raises(ValueError, match="either baseline_center_trials"):
        pci_st_from_trials(
            evoked[np.newaxis, :, :],
            times,
            baseline_center_trials=True,
            baseline_correct=True,
            **kwargs,
        )
