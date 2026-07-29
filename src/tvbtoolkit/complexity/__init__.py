"""Public complexity APIs for TVBToolkit.

This package exposes:

- ``pci_casali_like``: Casali-style PCI (single trial) — normalized 2D
  Lempel-Ziv complexity on baseline-thresholded spatiotemporal responses.
- ``pci_casali_like_multi_trial``: multi-trial perturbational complexity.
  Its ``casali`` route computes one Lempel-Ziv PCI from the statistically
  thresholded trial average using a within-trial pre/post permutation by
  default; its explicit ``tvbsim`` route retains legacy TVBSim parity.
- ``pci_st`` / ``pci_st_from_trials``: state-transition PCI following
  Comolatti et al. (2019). PCI-ST is a distinct estimator and is not bounded
  to the interval 0--1.
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
from .pci_st import PCIStResult, pci_st, pci_st_from_trials

__all__ = [
    "ace",
    "lzc_multichannel",
    "lzc_single_channel",
    "pci_casali_like",
    "pci_casali_like_multi_trial",
    "pci_ratio_proxy",
    "PCIStResult",
    "pci_st",
    "pci_st_from_trials",
    "sce",
]
