"""TVB AdEx simulator adapter for VBI/SBI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.inference.features import BOLDFeatureExtractor
from tvbtoolkit.inference.parameters import AdExPrior
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation


def extract_bold_monitor(
    full_monitor_output: Any,
    *,
    expected_period_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the TVB monitor closest to the requested BOLD period."""
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for item in full_monitor_output or []:
        if item is None:
            continue
        time, data = item
        time_arr = np.asarray(time, dtype=float).reshape(-1)
        data_arr = np.asarray(data)
        if time_arr.size == 0 or data_arr.shape[0] != time_arr.size:
            continue
        if data_arr.ndim == 4:
            signal = np.asarray(data_arr[:, 0, :, 0], dtype=float)
        elif data_arr.ndim == 2:
            signal = np.asarray(data_arr, dtype=float)
        else:
            signal = np.asarray(data_arr, dtype=float).reshape(time_arr.size, -1)
        period = (
            float(np.median(np.diff(time_arr)))
            if time_arr.size > 1
            else float("inf")
        )
        candidates.append((period, time_arr, signal))
    if not candidates:
        raise RuntimeError("TVB returned no usable monitor output.")

    finite = [item for item in candidates if np.isfinite(item[0])]
    if not finite:
        raise RuntimeError("BOLD monitor returned fewer than two samples.")
    chosen = min(finite, key=lambda item: abs(item[0] - float(expected_period_ms)))
    tolerance = max(1e-6, 0.05 * float(expected_period_ms))
    if abs(chosen[0] - float(expected_period_ms)) > tolerance:
        periods = [item[0] for item in finite]
        raise RuntimeError(
            f"No monitor has expected BOLD period {expected_period_ms} ms; got {periods}."
        )
    return chosen[1], chosen[2]


@dataclass
class AdExBOLDSimulator:
    """Callable mapping an ordered parameter vector to a BOLD feature vector."""

    base_config: WholeBrainConfig
    prior: AdExPrior
    feature_extractor: BOLDFeatureExtractor
    transient_ms: float = 20_000.0

    def __post_init__(self) -> None:
        if self.base_config.weights is None:
            raise ValueError("AdEx inference requires an explicit structural weight matrix.")
        if self.base_config.tract_lengths is None:
            if "conduction_speed" in self.prior.names:
                raise ValueError(
                    "Conduction speed is in the prior, but tract_lengths is missing. "
                    "Speed would be non-identifiable."
                )
            raise ValueError("AdEx inference requires explicit tract_lengths.")
        lengths = np.asarray(self.base_config.tract_lengths, dtype=float)
        if "conduction_speed" in self.prior.names and not np.any(lengths > 0):
            raise ValueError(
                "Conduction speed is in the prior, but all tract lengths are zero."
            )
        if self.transient_ms < 0:
            raise ValueError("transient_ms must be non-negative.")
        if self.base_config.simulation_length_ms <= self.transient_ms:
            raise ValueError("simulation_length_ms must exceed transient_ms.")
        if not self.feature_extractor.is_fitted:
            raise ValueError("Fit the BOLDFeatureExtractor to the observation first.")

    def __call__(self, theta: np.ndarray, *, seed: int = 0) -> np.ndarray:
        cfg = self.prior.apply(self.base_config, theta)
        cfg.include_bold_monitor = True
        cfg.bold_monitor_period_ms = 1000.0 * self.feature_extractor.config.tr_seconds
        cfg.bold_monitor_variables = (0,)

        result = run_whole_brain_simulation(cfg, seed=int(seed))
        time_ms, bold = extract_bold_monitor(
            result.full_monitor_output,
            expected_period_ms=cfg.bold_monitor_period_ms,
        )
        keep = time_ms >= float(self.transient_ms)
        if not np.any(keep):
            raise RuntimeError(
                f"No BOLD samples remain after transient_ms={self.transient_ms}."
            )
        return self.feature_extractor.transform(bold[keep])
