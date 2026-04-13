"""Core configuration models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class WholeBrainConfig:
    """Configuration for a TVB whole-brain run.

    By default this uses the AdEx mean-field Zerlaut family to keep parity with
    legacy TVBSim behavior.
    """

    simulation_length_ms: float = 2000.0
    dt_ms: float = 0.1
    conduction_speed: float = 4.0
    monitor_period_ms: float = 1.0
    coupling_strength: float = 0.015
    model_family: Literal["adex_zerlaut", "generic2d"] = "adex_zerlaut"
    zerlaut_matteo: bool = False
    zerlaut_gk_gna: bool = False
    zerlaut_order: Literal[1, 2] = 1
    stochastic_integrator: bool = True
    # Monitor selection:
    # - None: keep legacy monitor settings from parameter schema
    # - "raw": use Raw monitor
    # - "temporal_average": use TemporalAverage monitor
    monitor_mode: Literal["raw", "temporal_average"] | None = None
    temporal_average_period_ms: float = 1.0
    monitor_variables: tuple[int, ...] = (0, 1)

    # Optional TVB Bold monitor (can run alongside raw/temporal-average monitor).
    include_bold_monitor: bool = False
    bold_monitor_period_ms: float = 2000.0
    bold_monitor_variables: tuple[int, ...] = (0,)
    connectivity_zip: str | Path | None = None
    weights: np.ndarray | None = None
    tract_lengths: np.ndarray | None = None
    parameter_overrides: dict = field(default_factory=dict)


@dataclass
class SingleRegionConfig:
    """Configuration for a two-population AdEx Brian2 run."""

    duration_ms: float = 1000.0
    dt_ms: float = 0.1
    n_total: int = 4000
    inhibitory_fraction: float = 0.2
    p_connect: float = 0.05
    p_external: float = 0.05
    external_rate_e_hz: float = 3.0
    external_rate_i_hz: float = 3.0

    # Shared membrane parameters
    c_m_pf: float = 200.0
    g_l_ns: float = 10.0
    v_reset_mv: float = -65.0
    v_thresh_mv: float = -50.0
    v_cut_mv: float = -30.0
    e_e_mv: float = 0.0
    e_i_mv: float = -80.0

    # Excitatory population (RS-like)
    e_l_e_mv: float = -63.0
    delta_t_e_mv: float = 2.0
    a_e_ns: float = 0.0
    b_e_pa: float = 5.0
    tau_w_e_ms: float = 500.0

    # Inhibitory population (FS-like)
    e_l_i_mv: float = -65.0
    delta_t_i_mv: float = 0.5
    a_i_ns: float = 0.0
    b_i_pa: float = 0.0
    tau_w_i_ms: float = 500.0

    # Synapse parameters
    tau_e_ms: float = 5.0
    tau_i_ms: float = 5.0
    q_e_ns: float = 1.5
    q_i_ns: float = 5.0
    refractory_ms: float = 5.0
    record_spikes: bool = False


@dataclass
class OutputConfig:
    """Standardized output roots."""

    root: Path
    simulations_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.simulations_dir = self.root / "simulations"
        self.figures_dir = self.root / "figures"
        self.metrics_dir = self.root / "metrics"
        self.simulations_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
