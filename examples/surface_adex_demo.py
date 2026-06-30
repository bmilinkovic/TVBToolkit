"""Run a short surface-based AdEx/Zerlaut simulation.

Example:
    python examples/surface_adex_demo.py \
        --connectivity data/connectivity/connectivity_68.zip \
        --surface /path/to/cortex_16384.zip \
        --region-mapping /path/to/regionMapping_16k_68.txt \
        --local-connectivity /path/to/local_connectivity_16384.mat
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TVB_USER_HOME", str(PROJECT_ROOT / ".tvb-temp"))

from tvbtoolkit.surface import SurfaceConfig, run_surface_adex_simulation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectivity", required=True, help="TVB connectivity zip.")
    parser.add_argument("--surface", required=True, help="TVB cortical surface zip.")
    parser.add_argument("--region-mapping", required=True, help="TVB region mapping txt.")
    parser.add_argument(
        "--local-connectivity",
        default=None,
        help="Optional precomputed TVB local connectivity .mat. If omitted, TVB computes it with gdist.",
    )
    parser.add_argument("--length-ms", type=float, default=100.0)
    parser.add_argument("--period-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="outputs/surface_adex_demo.npz")
    args = parser.parse_args()

    cfg = SurfaceConfig(
        connectivity_zip=args.connectivity,
        surface_file=args.surface,
        region_mapping_file=args.region_mapping,
        local_connectivity_file=args.local_connectivity,
        simulation_length_ms=float(args.length_ms),
        temporal_average_period_ms=float(args.period_ms),
        monitor_mode="spatial_average",
    )
    result = run_surface_adex_simulation(cfg, seed=int(args.seed))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        time_ms=result.time_ms,
        region_average=result.region_average,
        region_average_inh=(
            result.region_average_inh if result.region_average_inh is not None else np.array([])
        ),
        region_labels=result.region_labels,
        region_mapping=result.region_mapping,
    )
    print(f"Saved {out} | region_average={result.region_average.shape}")


if __name__ == "__main__":
    main()
