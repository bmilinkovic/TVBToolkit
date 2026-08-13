"""Simulation storage and the maintained VBI 0.4.x inference backend."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tvbtoolkit.inference.parameters import AdExPrior, make_sbi_prior


@dataclass(frozen=True)
class SimulationDataset:
    """Finite successful simulations ready for VBI training."""

    theta: np.ndarray
    features: np.ndarray
    parameter_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    seeds: np.ndarray
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta)
        features = np.asarray(self.features)
        if theta.ndim != 2 or features.ndim != 2:
            raise ValueError("theta and features must both be 2D.")
        if theta.shape[0] != features.shape[0]:
            raise ValueError("theta/features simulation counts differ.")
        if theta.shape[1] != len(self.parameter_names):
            raise ValueError("theta width does not match parameter_names.")
        if features.shape[1] != len(self.feature_names):
            raise ValueError("feature width does not match feature_names.")
        if np.asarray(self.seeds).shape != (theta.shape[0],):
            raise ValueError("seeds must have one entry per successful simulation.")
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(features)):
            raise ValueError("SimulationDataset contains NaN or Inf.")

    def save(self, path: str | Path) -> Path:
        """Atomically save a compressed, non-pickle dataset."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp.npz")
        metadata = json.dumps(
            {
                "parameter_names": list(self.parameter_names),
                "feature_names": list(self.feature_names),
                "failures": list(self.failures),
                "format_version": 1,
            },
            sort_keys=True,
        )
        np.savez_compressed(
            temporary,
            theta=np.asarray(self.theta, dtype=np.float32),
            features=np.asarray(self.features, dtype=np.float32),
            seeds=np.asarray(self.seeds, dtype=np.int64),
            metadata=np.asarray(metadata),
        )
        os.replace(temporary, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> SimulationDataset:
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"]).item()))
            return cls(
                theta=np.asarray(data["theta"], dtype=np.float32),
                features=np.asarray(data["features"], dtype=np.float32),
                parameter_names=tuple(metadata["parameter_names"]),
                feature_names=tuple(metadata["feature_names"]),
                seeds=np.asarray(data["seeds"], dtype=np.int64),
                failures=tuple(metadata.get("failures", [])),
            )


def simulate_prior(
    prior: AdExPrior,
    simulator: Callable[..., np.ndarray],
    *,
    num_simulations: int,
    feature_names: tuple[str, ...],
    seed: int = 0,
    max_failures: int = 0,
) -> SimulationDataset:
    """Run prior simulations sequentially with deterministic per-run seeds."""
    theta_all = prior.sample(num_simulations, seed=seed)
    seed_rng = np.random.default_rng(seed + 1)
    run_seeds = seed_rng.integers(0, np.iinfo(np.int32).max, size=num_simulations)
    theta_ok: list[np.ndarray] = []
    features_ok: list[np.ndarray] = []
    seeds_ok: list[int] = []
    failures: list[str] = []

    for index, (theta, run_seed) in enumerate(zip(theta_all, run_seeds)):
        try:
            feature = np.asarray(simulator(theta, seed=int(run_seed)), dtype=np.float32)
            if feature.shape != (len(feature_names),):
                raise ValueError(
                    f"Expected feature shape {(len(feature_names),)}, got {feature.shape}."
                )
            if not np.all(np.isfinite(feature)):
                raise ValueError("Simulator returned NaN or Inf.")
        except Exception as exc:
            failures.append(f"simulation={index}, seed={int(run_seed)}: {type(exc).__name__}: {exc}")
            if len(failures) > int(max_failures):
                raise RuntimeError(failures[-1]) from exc
            continue
        theta_ok.append(theta)
        features_ok.append(feature)
        seeds_ok.append(int(run_seed))

    if not theta_ok:
        raise RuntimeError("No simulations completed successfully.")
    return SimulationDataset(
        theta=np.vstack(theta_ok),
        features=np.vstack(features_ok),
        parameter_names=prior.names,
        feature_names=feature_names,
        seeds=np.asarray(seeds_ok, dtype=np.int64),
        failures=tuple(failures),
    )


def train_vbi_posterior(
    dataset: SimulationDataset,
    prior: AdExPrior,
    *,
    method: str = "SNPE",
    density_estimator: str = "maf",
    device: str = "cpu",
    num_threads: int = 1,
):
    """Train SNPE/SNLE/SNRE through the maintained VBI 0.4.x wrapper."""
    if tuple(dataset.parameter_names) != prior.names:
        raise ValueError("Dataset parameter order does not match the prior.")
    try:
        import torch
        import vbi
        from packaging.version import Version
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Install the inference dependencies with "
            "`python -m pip install -e '.[inference]'`."
        ) from exc
    if Version(vbi.__version__) < Version("0.4.3"):
        raise RuntimeError(
            f"TVBToolkit requires maintained VBI >=0.4.3; found {vbi.__version__}."
        )
    if method not in {"SNPE", "SNLE", "SNRE"}:
        raise ValueError("method must be one of: SNPE, SNLE, SNRE.")

    inference = vbi.Inference()
    return inference.train(
        torch.as_tensor(dataset.theta, dtype=torch.float32),
        torch.as_tensor(dataset.features, dtype=torch.float32),
        make_sbi_prior(prior),
        num_threads=int(num_threads),
        method=method,
        device=device,
        density_estimator=density_estimator,
    )


def sample_vbi_posterior(
    posterior,
    observation_features: np.ndarray,
    *,
    num_samples: int = 10_000,
) -> np.ndarray:
    """Draw posterior samples through VBI and return a NumPy array."""
    try:
        import vbi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VBI is not installed; use the `inference` extra.") from exc
    samples = vbi.Inference.sample_posterior(
        np.asarray(observation_features, dtype=np.float32),
        int(num_samples),
        posterior,
    )
    return samples.detach().cpu().numpy()
