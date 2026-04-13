import numpy as np

from tvbtoolkit.complexity.measures import ace, lzc_multichannel, lzc_single_channel, pci_casali_like, sce


def test_complexity_outputs_finite():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 12))
    vals = [
        lzc_multichannel(x),
        lzc_single_channel(x),
        ace(x),
        sce(x),
        pci_casali_like(x, stimulation_index=400, t_analysis_ms=100.0, dt_ms=1.0),
    ]
    for v in vals:
        assert np.isfinite(v)
