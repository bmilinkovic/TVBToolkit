from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.inference.features import BOLDFeatureConfig, BOLDFeatureExtractor


def _synthetic_bold(seed: int, *, n_time: int = 140, n_regions: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(n_time) * 2.4
    latent = np.column_stack(
        [
            np.sin(2 * np.pi * 0.035 * time),
            np.sin(2 * np.pi * 0.070 * time + 0.4),
            rng.normal(size=n_time),
        ]
    )
    mixing = rng.normal(size=(3, n_regions))
    return latent @ mixing + 0.2 * rng.normal(size=(n_time, n_regions))


def test_bold_features_are_finite_stable_and_template_aligned() -> None:
    observation = _synthetic_bold(1)
    simulation = _synthetic_bold(2)
    sc = np.corrcoef(observation, rowvar=False)
    np.fill_diagonal(sc, 0.0)
    extractor = BOLDFeatureExtractor(
        BOLDFeatureConfig(
            n_states=3,
            state_n_init=3,
            fcd_window_seconds=20.0,
        )
    )

    observed_features = extractor.fit_transform(
        observation, structural_connectivity=np.abs(sc)
    )
    simulated_features = extractor.transform(simulation)

    assert observed_features.ndim == 1
    assert simulated_features.shape == observed_features.shape
    assert extractor.feature_names_ is not None
    assert len(extractor.feature_names_) == observed_features.size
    assert np.isfinite(observed_features).all()
    assert np.isfinite(simulated_features).all()
    occupancy_idx = [
        i for i, name in enumerate(extractor.feature_names_) if name.startswith("state_occupancy_")
    ]
    assert np.sum(observed_features[occupancy_idx]) == pytest.approx(1.0)
    assert np.sum(simulated_features[occupancy_idx]) == pytest.approx(1.0)


def test_bold_feature_transform_requires_fitted_state_template() -> None:
    extractor = BOLDFeatureExtractor(
        BOLDFeatureConfig(include_fc=False, include_fcd=False, include_states=True)
    )
    with pytest.raises(RuntimeError, match="Call fit"):
        extractor.transform(_synthetic_bold(3))


def test_bold_features_reject_region_count_mismatch() -> None:
    extractor = BOLDFeatureExtractor(
        BOLDFeatureConfig(
            include_fc=True,
            include_fcd=False,
            include_states=False,
        )
    )
    extractor.fit(_synthetic_bold(4, n_regions=6))
    with pytest.raises(ValueError, match="expects 6"):
        extractor.transform(_synthetic_bold(5, n_regions=7))
