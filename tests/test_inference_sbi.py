from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tvbtoolkit.inference.parameters import AdExParameterSpec, AdExPrior
from tvbtoolkit.inference.sbi import (
    SimulationDataset,
    sample_vbi_posterior,
    simulate_prior,
    train_vbi_posterior,
)


def test_simulate_prior_and_dataset_roundtrip(tmp_path: Path) -> None:
    prior = AdExPrior.default()

    def simulator(theta: np.ndarray, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.array([theta[0], theta[1] + rng.normal(scale=1e-6)], dtype=np.float32)

    dataset = simulate_prior(
        prior,
        simulator,
        num_simulations=5,
        feature_names=("adaptation_proxy", "coupling_proxy"),
        seed=7,
    )
    path = dataset.save(tmp_path / "simulations.npz")
    loaded = SimulationDataset.load(path)

    np.testing.assert_array_equal(loaded.theta, dataset.theta)
    np.testing.assert_array_equal(loaded.features, dataset.features)
    np.testing.assert_array_equal(loaded.seeds, dataset.seeds)
    assert loaded.parameter_names == prior.names
    assert loaded.feature_names == ("adaptation_proxy", "coupling_proxy")


def test_simulate_prior_fails_loudly_by_default() -> None:
    prior = AdExPrior.default()

    def bad_simulator(theta: np.ndarray, *, seed: int) -> np.ndarray:
        del theta, seed
        return np.array([np.nan])

    with pytest.raises(RuntimeError, match="NaN or Inf"):
        simulate_prior(
            prior,
            bad_simulator,
            num_simulations=1,
            feature_names=("bad",),
        )


def test_maintained_vbi_backend_trains_and_samples_a_synthetic_posterior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    vbi = pytest.importorskip("vbi")
    if tuple(int(part) for part in vbi.__version__.split(".")[:2]) < (0, 4):
        pytest.skip("Maintained VBI 0.4.x is not installed.")
    monkeypatch.chdir(tmp_path)

    prior = AdExPrior(
        (
            AdExParameterSpec(
                "global_coupling",
                0.05,
                0.50,
                "dimensionless",
                "config",
                target="coupling_strength",
            ),
            AdExParameterSpec(
                "conduction_speed",
                1.0,
                20.0,
                "mm/ms",
                "config",
                target="conduction_speed",
            ),
        )
    )
    rng = np.random.default_rng(31)
    theta = prior.sample(200, seed=31)
    features = np.column_stack(
        (
            theta[:, 0] + rng.normal(0.0, 0.01, theta.shape[0]),
            np.log(theta[:, 1]) + rng.normal(0.0, 0.03, theta.shape[0]),
        )
    ).astype(np.float32)
    dataset = SimulationDataset(
        theta=theta,
        features=features,
        parameter_names=prior.names,
        feature_names=("coupling_proxy", "speed_proxy"),
        seeds=np.arange(theta.shape[0]),
    )

    torch.manual_seed(31)
    posterior = train_vbi_posterior(dataset, prior, method="SNPE", num_threads=1)
    truth = np.array([0.22, 8.0], dtype=np.float32)
    observation = np.array([truth[0], np.log(truth[1])], dtype=np.float32)
    samples = sample_vbi_posterior(posterior, observation, num_samples=200)

    assert samples.shape == (200, 2)
    assert np.isfinite(samples).all()
    assert np.all(samples >= prior.low)
    assert np.all(samples <= prior.high)
    assert np.mean(samples[:, 0]) == pytest.approx(truth[0], abs=0.08)
    assert np.mean(samples[:, 1]) == pytest.approx(truth[1], abs=2.5)
