"""Surface-based AdEx/Zerlaut simulation routines built on TVB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tvb.simulator import lab

from tvbtoolkit.core.config import SurfaceConfig
from tvbtoolkit.surface.io import load_surface_cortex
from tvbtoolkit.surface.mapping import (
    average_nodes_to_regions,
    full_region_mapping,
    prepare_surface_parameter_value,
)
from tvbtoolkit.whole_brain.legacy_engine.parameter.parameter_M_Berlin_new import Parameters
from tvbtoolkit.whole_brain.simulation import (
    _apply_parameter_overrides,
    _build_connectivity,
    _build_coupling,
    _build_integrator,
    _build_stimulation,
    _configure_shared_noise,
    _configure_zerlaut_model_parameters,
    _select_zerlaut_model,
)


@dataclass
class SurfaceResult:
    """Container for surface-based simulation outputs."""

    time_ms: np.ndarray
    region_average: np.ndarray
    region_labels: np.ndarray
    region_average_inh: np.ndarray | None = None
    vertex: np.ndarray | None = None
    vertex_inh: np.ndarray | None = None
    region_mapping: np.ndarray | None = None
    full_monitor_output: Any | None = None


def _build_surface_monitors(cfg: SurfaceConfig):
    variables = np.asarray(list(cfg.monitor_variables), dtype=int)
    if cfg.monitor_mode == "spatial_average":
        return (
            lab.monitors.SpatialAverage(
                variables_of_interest=variables,
                period=float(cfg.temporal_average_period_ms),
            ),
        )
    if cfg.monitor_mode == "temporal_average":
        return (
            lab.monitors.TemporalAverage(
                variables_of_interest=variables,
                period=float(cfg.temporal_average_period_ms),
            ),
        )
    if cfg.monitor_mode == "raw":
        raw_cls = getattr(lab.monitors, "RawVoi", None) or lab.monitors.Raw
        return (raw_cls(variables_of_interest=variables),)
    raise ValueError(
        "Unsupported surface monitor_mode. Use 'spatial_average', "
        f"'temporal_average', or 'raw'. Got {cfg.monitor_mode!r}."
    )


def _prepare_legacy_parameters(cfg: SurfaceConfig) -> Parameters:
    parameters = Parameters()
    pm = parameters.parameter_model
    pm["matteo"] = cfg.zerlaut_matteo
    pm["gK_gNa"] = cfg.zerlaut_gk_gna
    pm["order"] = cfg.zerlaut_order

    parameters.parameter_connection_between_region["speed"] = cfg.conduction_speed
    parameters.parameter_coupling["coupling_parameter"]["a"] = cfg.coupling_strength
    parameters.parameter_integrator["dt"] = cfg.dt_ms
    parameters.parameter_integrator["stochastic"] = cfg.stochastic_integrator
    if "noise_parameter" in parameters.parameter_integrator:
        parameters.parameter_integrator["noise_parameter"]["dt"] = cfg.dt_ms

    _apply_parameter_overrides(parameters, cfg.parameter_overrides)
    # Keep model-family selection controlled by explicit config, even when
    # legacy parameter overrides include these keys.
    pm["matteo"] = cfg.zerlaut_matteo
    pm["gK_gNa"] = cfg.zerlaut_gk_gna
    pm["order"] = cfg.zerlaut_order
    return parameters


def _surface_parameter_preparer(cortex, n_regions: int):
    mapping = full_region_mapping(cortex)
    n_vertices = int(cortex.number_of_vertices)

    def _prepare(key: str, value, _: int) -> np.ndarray:
        return prepare_surface_parameter_value(
            key,
            value,
            mapping,
            n_regions,
            n_vertices=n_vertices,
        )

    return _prepare


def _extract_voi(data: np.ndarray, index: int) -> np.ndarray | None:
    if data.ndim != 4 or data.shape[1] <= index:
        return None
    return np.asarray(data[:, index, :, 0], dtype=float)


def run_surface_adex_simulation(cfg: SurfaceConfig, seed: int = 0) -> SurfaceResult:
    """Run a TVB surface simulation with the current AdEx/Zerlaut parameter regime.

    This routine reuses the whole-brain AdEx/Zerlaut model selection,
    connectivity, coupling, integrator, stimulation, and parameter override
    machinery. Region-wise biophysical parameters are expanded to surface nodes
    through ``cortex.region_mapping``.
    """
    if cfg.model_family != "adex_zerlaut":
        raise ValueError("Surface simulations currently support only model_family='adex_zerlaut'.")

    np.random.seed(seed)
    parameters = _prepare_legacy_parameters(cfg)
    model = _select_zerlaut_model(parameters.parameter_model)
    connection = _build_connectivity(parameters, cfg)
    cortex = load_surface_cortex(cfg, connection)

    n_regions = int(np.asarray(connection.weights).shape[0])
    mapping = full_region_mapping(cortex)
    _configure_zerlaut_model_parameters(
        model,
        parameters,
        int(mapping.size),
        parameter_value_prepare=_surface_parameter_preparer(cortex, n_regions),
    )
    _configure_shared_noise(
        model,
        parameters.parameter_model,
        connection,
        region_mapping=mapping,
    )

    coupling = _build_coupling(parameters)
    integrator = _build_integrator(parameters, cfg, model, connection, seed)
    monitors = _build_surface_monitors(cfg)
    stimulation = _build_stimulation(parameters.parameter_stimulus, connection, model)

    simulator = lab.simulator.Simulator(
        model=model,
        connectivity=connection,
        coupling=coupling,
        integrator=integrator,
        monitors=monitors,
        stimulus=stimulation,
        surface=cortex,
    )
    simulator.configure()
    output = simulator.run(simulation_length=cfg.simulation_length_ms)
    if not output or output[0] is None:
        raise RuntimeError("TVB surface simulation returned no output.")

    t, data = output[0]
    data = np.asarray(data)
    exc = _extract_voi(data, 0)
    inh = _extract_voi(data, 1)
    if exc is None:
        raise RuntimeError(f"Unexpected TVB monitor output shape: {data.shape}.")

    if cfg.monitor_mode == "spatial_average":
        region_average = exc
        region_average_inh = inh
        vertex = None
        vertex_inh = None
    else:
        vertex = exc
        vertex_inh = inh
        region_average = average_nodes_to_regions(vertex, mapping, n_regions)
        region_average_inh = (
            average_nodes_to_regions(vertex_inh, mapping, n_regions)
            if vertex_inh is not None
            else None
        )

    if getattr(connection, "region_labels", None) is not None and len(connection.region_labels):
        region_labels = np.asarray(connection.region_labels)
    else:
        region_labels = np.array([f"R{i}" for i in range(region_average.shape[1])], dtype="U16")

    return SurfaceResult(
        time_ms=np.asarray(t).reshape(-1),
        region_average=region_average,
        region_average_inh=region_average_inh,
        vertex=vertex,
        vertex_inh=vertex_inh,
        region_labels=region_labels,
        region_mapping=mapping,
        full_monitor_output=output,
    )
