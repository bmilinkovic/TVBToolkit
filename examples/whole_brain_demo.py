"""Whole-brain TVB demo."""

from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation


def main():
    cfg = WholeBrainConfig(simulation_length_ms=500.0, dt_ms=0.5)
    res = run_whole_brain_simulation(cfg, seed=1)
    print("Time samples:", res.time_ms.shape[0])
    print("Raw shape:", res.raw.shape)


if __name__ == "__main__":
    main()

