"""Complexity metrics demo."""

import numpy as np

from tvbtoolkit.complexity.measures import ace, lzc_multichannel, pci_casali_like, sce


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 32))
    print("LZc:", lzc_multichannel(x))
    print("ACE:", ace(x))
    print("SCE:", sce(x))
    print(
        "PCI (Casali-like):",
        pci_casali_like(x, stimulation_index=750, t_analysis_ms=200.0, dt_ms=1.0),
    )


if __name__ == "__main__":
    main()
