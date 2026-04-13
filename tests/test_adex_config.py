from tvbtoolkit.core.config import SingleRegionConfig


def test_adex_config_defaults():
    cfg = SingleRegionConfig()
    assert cfg.n_total > 0
    assert 0.0 < cfg.inhibitory_fraction < 1.0
    assert cfg.duration_ms > 0.0

