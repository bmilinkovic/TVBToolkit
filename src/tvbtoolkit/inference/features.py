"""Comparable empirical/simulated BOLD features for VBI inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from tvbtoolkit.analysis.brain_states import (
    cluster_brain_states,
    phase_patterns,
    sfc_sort_centroids,
)
from tvbtoolkit.bold import BOLDParams, preprocess_bold_signal


@dataclass(frozen=True)
class BOLDFeatureConfig:
    """Settings shared by empirical and simulated BOLD feature extraction."""

    tr_seconds: float = 2.4
    bandpass_hz: tuple[float, float] = (0.01, 0.10)
    filter_order: int = 2
    include_fc: bool = True
    include_fcd: bool = True
    fcd_window_seconds: float = 30.0
    connectivity_statistics: tuple[str, ...] = ("mean", "std", "skew", "kurtosis")
    connectivity_quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    include_states: bool = True
    n_states: int = 5
    state_bandpass_hz: tuple[float, float] = (0.01, 0.20)
    state_filter_order: int = 3
    state_random_seed: int = 1
    state_n_init: int = 20

    def __post_init__(self) -> None:
        if self.tr_seconds <= 0:
            raise ValueError("tr_seconds must be positive.")
        if self.fcd_window_seconds <= 0:
            raise ValueError("fcd_window_seconds must be positive.")
        if self.n_states < 1:
            raise ValueError("n_states must be at least one.")
        allowed = {"sum", "max", "min", "mean", "std", "skew", "kurtosis"}
        unknown = set(self.connectivity_statistics) - allowed
        if unknown:
            raise ValueError(f"Unsupported connectivity statistics: {sorted(unknown)}.")


class BOLDFeatureExtractor:
    """Extract VBI FC/FCD statistics plus template-aligned brain-state features.

    State centroids are fitted once to the observation and then held fixed.
    Every simulation is assigned to those same centroids, avoiding the label
    permutation error caused by independently clustering every time series.
    """

    def __init__(self, config: BOLDFeatureConfig | None = None) -> None:
        self.config = config if config is not None else BOLDFeatureConfig()
        self.state_centers_: np.ndarray | None = None
        self.n_regions_: int | None = None
        self.feature_names_: tuple[str, ...] | None = None

    @property
    def is_fitted(self) -> bool:
        return not self.config.include_states or self.state_centers_ is not None

    def fit(
        self,
        observation_bold: np.ndarray,
        *,
        structural_connectivity: np.ndarray | None = None,
    ) -> BOLDFeatureExtractor:
        """Fit and optionally SC-order the observation-derived state template."""
        x = self._validate_bold(observation_bold)
        self.n_regions_ = int(x.shape[1])

        if self.config.include_states:
            patterns, _, _, _ = self._phase_patterns(x)
            if patterns.shape[0] < self.config.n_states:
                raise ValueError(
                    f"Need at least {self.config.n_states} valid volumes to fit states; "
                    f"got {patterns.shape[0]}."
                )
            labels, centers = cluster_brain_states(
                patterns,
                n_states=self.config.n_states,
                random_seed=self.config.state_random_seed,
                n_init=self.config.state_n_init,
                backend="sklearn",
            )
            if structural_connectivity is not None:
                sc = np.asarray(structural_connectivity, dtype=float)
                if sc.shape != (self.n_regions_, self.n_regions_):
                    raise ValueError(
                        f"SC shape {sc.shape} does not match BOLD regions {self.n_regions_}."
                    )
                centers, _, _, _ = sfc_sort_centroids(centers, labels, sc)
            self.state_centers_ = np.asarray(centers, dtype=float)

        values, names = self._transform_impl(x)
        self.feature_names_ = tuple(names)
        if not np.all(np.isfinite(values)):
            raise ValueError("Observation produced non-finite inference features.")
        return self

    def fit_transform(
        self,
        observation_bold: np.ndarray,
        *,
        structural_connectivity: np.ndarray | None = None,
    ) -> np.ndarray:
        self.fit(observation_bold, structural_connectivity=structural_connectivity)
        return self.transform(observation_bold)

    def transform(self, bold: np.ndarray) -> np.ndarray:
        """Return one finite, ordered feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Call fit(observation_bold) before transform().")
        x = self._validate_bold(bold)
        if self.n_regions_ is not None and x.shape[1] != self.n_regions_:
            raise ValueError(
                f"BOLD has {x.shape[1]} regions; fitted extractor expects {self.n_regions_}."
            )
        values, names = self._transform_impl(x)
        if self.feature_names_ is None:
            self.feature_names_ = tuple(names)
        elif tuple(names) != self.feature_names_:
            raise RuntimeError("Feature labels changed between observation and simulation.")
        if not np.all(np.isfinite(values)):
            bad = [name for name, value in zip(names, values) if not np.isfinite(value)]
            raise ValueError(f"Non-finite BOLD features: {bad}.")
        return values.astype(np.float32, copy=False)

    def _validate_bold(self, bold: np.ndarray) -> np.ndarray:
        x = np.asarray(bold, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"Expected BOLD shape (time, regions), got {x.shape}.")
        if x.shape[0] < 4 or x.shape[1] < 2:
            raise ValueError(f"BOLD is too small for inference: {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("BOLD contains NaN or Inf.")
        if np.any(np.std(x, axis=0) <= 1e-12):
            raise ValueError("At least one BOLD region is constant.")
        return x

    def _preprocess_connectivity(self, x: np.ndarray) -> np.ndarray:
        low, high = self.config.bandpass_hz
        return preprocess_bold_signal(
            x,
            params=BOLDParams(
                TR=self.config.tr_seconds,
                n_order=self.config.filter_order,
                low_f_num=low,
                high_f_num=high,
            ),
            apply_zscore=True,
            apply_bandpass=True,
            n_regions_hint=x.shape[1],
        )

    def _phase_patterns(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return phase_patterns(
            x,
            trim_edge_samples=0,
            pipeline="brain_act_legacy",
            tr_seconds=self.config.tr_seconds,
            bandpass_hz=self.config.state_bandpass_hz,
            filter_order=self.config.state_filter_order,
        )

    @staticmethod
    def _state_summary(labels: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
        counts = np.bincount(labels, minlength=n_states).astype(float)
        occupancy = counts / max(float(labels.size), 1.0)
        transitions = np.zeros((n_states, n_states), dtype=float)
        if labels.size > 1:
            np.add.at(transitions, (labels[:-1], labels[1:]), 1.0)
        row_sum = transitions.sum(axis=1, keepdims=True)
        transition_prob = np.divide(
            transitions,
            row_sum,
            out=np.zeros_like(transitions),
            where=row_sum > 0,
        )
        return occupancy, np.diag(transition_prob)

    def _transform_impl(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        values: list[float] = []
        names: list[str] = []

        if self.config.include_fc or self.config.include_fcd:
            connectivity_signal = self._preprocess_connectivity(x).T
            try:
                from vbi.feature_extraction.features import fc_stat, fcd_stat
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError(
                    "VBI 0.4.x is required for FC/FCD inference features. "
                    "Install `python -m pip install -e '.[inference]'`."
                ) from exc

            common = {
                "eigenvalues": False,
                "pca_num_components": 0,
                "quantiles": list(self.config.connectivity_quantiles),
                "features": list(self.config.connectivity_statistics),
                "verbose": False,
            }
            if self.config.include_fc:
                fc_values, fc_names = fc_stat(connectivity_signal, k=1, **common)
                values.extend(np.asarray(fc_values, dtype=float).tolist())
                names.extend([f"vbi_{name}" for name in fc_names])
            if self.config.include_fcd:
                fcd_values, fcd_names = fcd_stat(
                    connectivity_signal,
                    TR=self.config.tr_seconds,
                    win_len=self.config.fcd_window_seconds,
                    k=max(
                        1,
                        round(self.config.fcd_window_seconds / self.config.tr_seconds),
                    ),
                    **common,
                )
                values.extend(np.asarray(fcd_values, dtype=float).tolist())
                names.extend([f"vbi_{name}" for name in fcd_names])

        if self.config.include_states:
            if self.state_centers_ is None:
                raise RuntimeError("State template is not fitted.")
            patterns, synchrony, _, _ = self._phase_patterns(x)
            labels = np.argmin(cdist(patterns, self.state_centers_, metric="sqeuclidean"), axis=1)
            occupancy, persistence = self._state_summary(
                labels.astype(int), self.state_centers_.shape[0]
            )
            values.extend(occupancy.tolist())
            names.extend([f"state_occupancy_{i}" for i in range(occupancy.size)])
            values.extend(persistence.tolist())
            names.extend([f"state_self_transition_{i}" for i in range(persistence.size)])
            values.extend([float(np.mean(synchrony)), float(np.std(synchrony))])
            names.extend(["phase_synchrony_mean", "phase_synchrony_std"])

        return np.asarray(values, dtype=float), names
