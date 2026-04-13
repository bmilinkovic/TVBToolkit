"""AdEx single-region demo."""

from tvbtoolkit.core.config import SingleRegionConfig
from tvbtoolkit.single_region.simulation import run_single_region_simulation


def main():
    cfg = SingleRegionConfig(
        duration_ms=400.0,
        n_total=2000,
        external_rate_e_hz=4.0,
        external_rate_i_hz=4.0,
        b_e_pa=5.0,
        b_i_pa=0.0,
    )
    out = run_single_region_simulation(cfg, seed_value=1)
    print("Exc mean rate shape:", out.exc_rate_hz.shape)
    print("Inh mean rate shape:", out.inh_rate_hz.shape)


if __name__ == "__main__":
    main()

