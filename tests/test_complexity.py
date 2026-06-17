import numpy as np

from tvbtoolkit.complexity.measures import (
    ace,
    lzc_multichannel,
    lzc_single_channel,
    pci_casali_like,
    pci_casali_like_multi_trial,
    sce,
)


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


def test_multi_trial_pci_preserves_precut_source_time_orientation():
    rng = np.random.default_rng(2)
    dt_ms = 7.8125
    t_analysis_ms = 300.0
    nbins = int(round(t_analysis_ms / dt_ms))
    n_sources = 90
    n_trials = 3

    # Pre-cut windows as saved/loaded by 06_pci_analysis_pub.py:
    # (sources, 2*nbins).  Because sources > time bins here, the old heuristic
    # incorrectly treated sources as time and transposed the window.
    precut = rng.normal(size=(n_trials, n_sources, 2 * nbins))

    # Equivalent full time-series in canonical (time, sources) orientation.
    full = np.zeros((n_trials, 3 * nbins, n_sources), dtype=float)
    full[:, nbins : 3 * nbins, :] = np.transpose(precut, (0, 2, 1))

    np.random.seed(123)
    pci_precut, trials_precut = pci_casali_like_multi_trial(
        precut,
        stimulation_index=nbins,
        t_analysis_ms=t_analysis_ms,
        dt_ms=dt_ms,
    )

    np.random.seed(123)
    pci_full, trials_full = pci_casali_like_multi_trial(
        full,
        stimulation_index=2 * nbins,
        t_analysis_ms=t_analysis_ms,
        dt_ms=dt_ms,
    )

    assert np.isfinite(pci_precut)
    assert np.allclose(pci_precut, pci_full)
    assert np.allclose(trials_precut, trials_full)
