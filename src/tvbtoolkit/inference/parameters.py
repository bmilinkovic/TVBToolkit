"""Prior definitions and parameter mapping for AdEx whole-brain inference."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from tvbtoolkit.core.config import WholeBrainConfig

ParameterLocation = Literal["config", "parameter_model"]


@dataclass(frozen=True)
class AdExParameterSpec:
    """One scalar uniform-prior parameter and its AdEx configuration target.

    ``model_keys`` may contain more than one key when a biological parameter is
    represented by multiple inputs in the Zerlaut parameter dictionary.
    """

    name: str
    low: float
    high: float
    unit: str
    location: ParameterLocation
    target: str | None = None
    model_keys: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter name must not be empty.")
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError(f"Bounds for {self.name!r} must be finite.")
        if self.low >= self.high:
            raise ValueError(f"Expected low < high for {self.name!r}.")
        if self.location == "config" and not self.target:
            raise ValueError(f"Config parameter {self.name!r} requires target.")
        if self.location == "parameter_model" and not self.model_keys:
            raise ValueError(f"Model parameter {self.name!r} requires model_keys.")


@dataclass(frozen=True)
class AdExPrior:
    """Ordered independent uniform prior for whole-brain AdEx parameters."""

    parameters: tuple[AdExParameterSpec, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("At least one prior parameter is required.")
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError(f"Prior parameter names must be unique, got {names}.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.parameters)

    @property
    def low(self) -> np.ndarray:
        return np.asarray([spec.low for spec in self.parameters], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.asarray([spec.high for spec in self.parameters], dtype=np.float32)

    @property
    def ndim(self) -> int:
        return len(self.parameters)

    def sample(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """Draw reproducible NumPy samples with shape ``(n, n_parameters)``."""
        if int(n) <= 0:
            raise ValueError("n must be positive.")
        rng = np.random.default_rng(seed)
        return rng.uniform(self.low, self.high, size=(int(n), self.ndim)).astype(np.float32)

    def as_dict(self, theta: np.ndarray) -> dict[str, float]:
        """Convert one ordered parameter vector to a name/value mapping."""
        values = np.asarray(theta, dtype=float).reshape(-1)
        if values.size != self.ndim:
            raise ValueError(
                f"theta has {values.size} values, but prior expects {self.ndim}: {self.names}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("theta contains non-finite values.")
        if np.any(values < self.low) or np.any(values > self.high):
            raise ValueError("theta contains values outside the declared prior bounds.")
        return {name: float(value) for name, value in zip(self.names, values)}

    def apply(self, base: WholeBrainConfig, theta: np.ndarray) -> WholeBrainConfig:
        """Return a deep-copied config with ``theta`` mapped to AdEx targets."""
        values = self.as_dict(theta)
        cfg = replace(base)
        cfg.parameter_overrides = deepcopy(base.parameter_overrides)
        model_overrides = deepcopy(cfg.parameter_overrides.get("parameter_model", {}))

        for spec in self.parameters:
            value = values[spec.name]
            if spec.location == "config":
                if not hasattr(cfg, str(spec.target)):
                    raise AttributeError(
                        f"WholeBrainConfig has no target {spec.target!r} for {spec.name!r}."
                    )
                setattr(cfg, str(spec.target), value)
            else:
                for key in spec.model_keys:
                    model_overrides[key] = value

        if model_overrides:
            cfg.parameter_overrides["parameter_model"] = model_overrides
        return cfg

    @classmethod
    def default(cls, *, include_external_drive: bool = False) -> AdExPrior:
        """Return a biologically interpretable starting prior.

        E.g. ``adaptation_b_e``: the Zerlaut/AdEx spike-triggered excitatory
        adaptation increment ``b_e``.
        """
        parameters: list[AdExParameterSpec] = [
            AdExParameterSpec(
                name="adaptation_b_e",
                low=0.0,
                high=120.0,
                unit="pA",
                location="parameter_model",
                model_keys=("b_e",),
                description="Spike-triggered excitatory adaptation (the requested beta-like parameter).",
            ),
            AdExParameterSpec(
                name="global_coupling",
                low=0.05,
                high=0.50,
                unit="dimensionless",
                location="config",
                target="coupling_strength",
                description="Linear long-range coupling coefficient.",
            ),
            AdExParameterSpec(
                name="conduction_speed",
                low=1.0,
                high=20.0,
                unit="mm/ms (numerically m/s)",
                location="config",
                target="conduction_speed",
                description="Long-range propagation speed; requires non-zero tract lengths.",
            ),
            AdExParameterSpec(
                name="noise_amplitude",
                low=5.0e-5,
                high=2.0e-4,
                unit="model input units",
                location="parameter_model",
                model_keys=("weight_noise",),
                description="Amplitude of the Zerlaut OU noise drive.",
            ),
        ]
        if include_external_drive:
            parameters.append(
                AdExParameterSpec(
                    name="external_drive",
                    low=1.0e-4,
                    high=6.0e-4,
                    unit="kHz",
                    location="parameter_model",
                    model_keys=("external_input_ex_ex", "external_input_in_ex"),
                    description="External excitatory drive delivered to E and I populations.",
                )
            )
        return cls(tuple(parameters))


def make_sbi_prior(prior: AdExPrior):
    """Build the ``sbi`` BoxUniform distribution consumed by VBI."""
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError as exc:  # pragma: no cover - exercised without inference extra
        raise ImportError(
            "SBI support is optional. Install TVBToolkit with "
            "`python -m pip install -e '.[inference]'`."
        ) from exc

    return BoxUniform(
        low=torch.as_tensor(prior.low, dtype=torch.float32),
        high=torch.as_tensor(prior.high, dtype=torch.float32),
    )
