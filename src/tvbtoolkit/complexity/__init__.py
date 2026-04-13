"""Public complexity APIs for TVBToolkit.

This package exposes:

- ``pci_casali_like``: Casali-style PCI (single trial) — normalized 2D
  Lempel-Ziv complexity on baseline-thresholded spatiotemporal responses.
- ``pci_casali_like_multi_trial``: Exact parity with TVBSim's
  ``parallelized_PCI`` / ``_calculate_PCI_seed_subset`` workflow using
  multiple stimulation trials (TVBSim default: n_trials=5, t_analysis=300 ms,
  nshuffles=10, percentile=100).  Trials are jointly binarized using pooled
  pre-stimulus baseline statistics before per-trial LZc computation.
- ``pci_ratio_proxy``: legacy ``LZ(post) / LZ(pre)`` proxy retained for
  backward compatibility only.  **Not** Casali PCI.
"""

from .measures import (
    ace,
    lzc_multichannel,
    lzc_single_channel,
    pci_casali_like,
    pci_casali_like_multi_trial,
    pci_ratio_proxy,
    sce,
)

__all__ = [
    "ace",
    "lzc_multichannel",
    "lzc_single_channel",
    "pci_casali_like",
    "pci_casali_like_multi_trial",
    "pci_ratio_proxy",
    "sce",
]
