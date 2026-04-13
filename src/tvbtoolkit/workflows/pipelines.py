"""High-level workflow helpers combining simulation + complexity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tvbtoolkit.complexity.measures import ace, lzc_multichannel, sce
from tvbtoolkit.core.config import SingleRegionConfig, WholeBrainConfig
from tvbtoolkit.single_region.simulation import run_single_region_simulation
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation


@dataclass
class ComplexitySummary:
    """Standardized output from workflow complexity computation."""

    lzc: float
    ace: float
    sce: float


def run_whole_brain_with_complexity(
    cfg: WholeBrainConfig,
    seed: int = 0,
) -> tuple:
    """Run whole-brain simulation and compute complexity on returned activity."""
    res = run_whole_brain_simulation(cfg, seed=seed)
    x = np.asarray(res.raw)
    summary = ComplexitySummary(lzc=lzc_multichannel(x), ace=ace(x), sce=sce(x))
    return res, summary


def run_single_region_with_complexity(
    cfg: SingleRegionConfig,
    seed: int = 0,
) -> tuple:
    """Run AdEx single-region simulation and compute complexity on population rates."""
    res = run_single_region_simulation(cfg, seed_value=seed)
    x = np.column_stack([res.exc_rate_hz, res.inh_rate_hz])
    summary = ComplexitySummary(lzc=lzc_multichannel(x), ace=ace(x), sce=sce(x))
    return res, summary

