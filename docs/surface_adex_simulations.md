# Surface-Based AdEx Simulations

`tvbtoolkit.surface` runs TVB surface simulations with the same AdEx/Zerlaut
mean-field regime used by the whole-brain workflows.

## Required Assets

A surface run needs aligned TVB assets:

- a long-range `Connectivity` zip
- a cortical surface zip
- a region mapping text file whose region indices refer to that connectivity
- optionally, a precomputed local-connectivity `.mat`

If the local-connectivity file is omitted, TVB computes local geodesic
connectivity from the mesh. That path requires the `gdist` Python package,
which is now listed in `pyproject.toml`.

## Default Regime

`SurfaceConfig` defaults to:

- Zerlaut/AdEx second-order mean field (`zerlaut_order=2`)
- stochastic Heun integration
- `dt_ms=0.1`
- long-range coupling strength `0.3`
- local coupling strength `1.0`
- `SpatialAverage` monitoring, returning region-level time series

This keeps the surface workflow close to the current publication/parity
whole-brain configuration while avoiding accidental storage of large vertex
time series.

## Minimal Usage

```python
from tvbtoolkit.surface import SurfaceConfig, run_surface_adex_simulation

cfg = SurfaceConfig(
    connectivity_zip="data/connectivity/connectivity_68.zip",
    surface_file="/path/to/cortex_16384.zip",
    region_mapping_file="/path/to/regionMapping_16k_68.txt",
    local_connectivity_file="/path/to/local_connectivity_16384.mat",
    simulation_length_ms=1000.0,
)

result = run_surface_adex_simulation(cfg, seed=1)
print(result.region_average.shape)
```

To inspect node-level output for short debug runs, set
`monitor_mode="temporal_average"` or `"raw"`. Region-average signals are still
computed from the node output using the surface region mapping.

## Region-Wise Parameters

Region-wise biophysical parameters such as `E_L_e`, `E_L_i`, `g_K_e`, `g_K_i`,
`g_Na_e`, and `g_Na_i` are expanded through `cortex.region_mapping`. This means
existing receptor-gradient or pharmacological overrides can remain region-wise
as long as their length matches the connectivity region count.
