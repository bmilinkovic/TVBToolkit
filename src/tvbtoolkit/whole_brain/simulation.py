"""Whole-brain simulation engine with strict AdEx Zerlaut parity support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tvb.simulator import lab

from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.whole_brain.legacy_engine.parameter.parameter_M_Berlin_new import Parameters
from tvbtoolkit.whole_brain.legacy_engine.src import (
    Zerlaut,
    Zerlaut_gK_gNa,
    Zerlaut_matteo,
    Zerlaut_matteo_gK_gNa,
)


@dataclass
class WholeBrainResult:
    """Container for whole-brain simulation outputs."""

    time_ms: np.ndarray
    raw: np.ndarray
    region_labels: np.ndarray
    raw_inh: np.ndarray | None = None
    full_monitor_output: Any | None = None


def _apply_parameter_overrides(parameters: Parameters, overrides: dict[str, Any]) -> None:
    """Apply flat or nested overrides into legacy parameter dictionaries."""
    if not overrides:
        return

    param_group_map = {k: v for k, v in vars(parameters).items() if k.startswith("parameter")}
    param_dicts = list(param_group_map.values())
    nested = []
    for d in param_dicts:
        for val in d.values():
            if isinstance(val, dict):
                nested.append(val)
    all_dicts = param_dicts + nested

    for key, value in overrides.items():
        applied = False
        # Direct override of top-level parameter groups (e.g. parameter_model)
        if key in param_group_map:
            if not isinstance(value, dict):
                raise TypeError(f"Override for '{key}' must be a dict.")
            param_group_map[key].update(value)
            continue

        if isinstance(value, dict):
            for d in all_dicts:
                if key in d and isinstance(d[key], dict):
                    d[key].update(value)
                    applied = True
            if applied:
                continue

        for d in all_dicts:
            if key in d:
                d[key] = value
                applied = True
        if not applied:
            raise KeyError(f"Unknown override key '{key}' for legacy parameter schema.")


def _select_zerlaut_model(pm: dict):
    """Match TVBSim model-selection logic for Zerlaut variants."""
    matteo = pm["matteo"]
    gk = pm["gK_gNa"]
    order = pm["order"]

    if matteo:
        if gk:
            model_module = Zerlaut_matteo_gK_gNa
        else:
            model_module = Zerlaut_matteo
    else:
        if gk:
            model_module = Zerlaut_gK_gNa
        else:
            model_module = Zerlaut

    if order == 1:
        return model_module.Zerlaut_adaptation_first_order(
            variables_of_interest="E I W_e W_i noise".split()
        )

    if order == 2:
        if hasattr(model_module, "Zerlaut_adaptation_second_order"):
            return model_module.Zerlaut_adaptation_second_order(
                variables_of_interest="E I C_ee C_ei C_ii W_e W_i noise".split()
            )
        # Legacy matteo_gK_gNa naming mismatch fallback.
        if hasattr(model_module, "Zerlaut_adaptation_second_order_gK_gNa"):
            return model_module.Zerlaut_adaptation_second_order_gK_gNa(
                variables_of_interest="E I C_ee C_ei C_ii W_e W_i noise".split()
            )

    raise ValueError(f"Unsupported Zerlaut order={order}")


def _build_connectivity(parameters: Parameters, cfg: WholeBrainConfig):
    pc = parameters.parameter_connection_between_region

    if cfg.connectivity_zip is not None:
        conn_path = Path(cfg.connectivity_zip).expanduser().resolve()
        if not conn_path.exists():
            raise FileNotFoundError(f"Configured connectivity_zip not found: {conn_path}")
        connection = lab.connectivity.Connectivity().from_file(str(conn_path))
    elif cfg.weights is not None and cfg.tract_lengths is not None:
        n_regions = cfg.weights.shape[0]
        connection = lab.connectivity.Connectivity(
            number_of_regions=n_regions,
            tract_lengths=np.asarray(cfg.tract_lengths),
            weights=np.asarray(cfg.weights),
            region_labels=np.asarray([f"R{i:03d}" for i in range(n_regions)], dtype="U128"),
            centres=np.zeros((n_regions, 3), dtype=float),
        )
    elif pc["default"]:
        # Prefer explicit DK-68 packaged atlas to keep strict legacy parity.
        root = Path(__file__).resolve().parents[3]
        conn_path = root / "data" / "connectivity" / "connectivity_68.zip"
        if not conn_path.exists():
            raise FileNotFoundError(
                "Default connectivity requested but packaged DK-68 atlas is missing at "
                f"{conn_path}. Set WholeBrainConfig.connectivity_zip explicitly."
            )
        connection = lab.connectivity.Connectivity().from_file(str(conn_path))
    elif pc["from_file"]:
        conn_name = pc.get("conn_name", "connectivity_68.zip")
        raw_path = Path(pc.get("path", ""))
        root = Path(__file__).resolve().parents[3]
        candidates = [
            raw_path / conn_name,
            root / "data" / "connectivity" / conn_name,
            root.parent / "TVBSim" / "tvbsim" / "TVB" / "tvb_model_reference" / "data" / "connectivity" / conn_name,
        ]
        conn_path = next((p for p in candidates if p.exists()), None)
        if conn_path is None:
            raise FileNotFoundError(
                f"Connectivity file '{conn_name}' not found. "
                f"Tried: {[str(p) for p in candidates]}"
            )
        connection = lab.connectivity.Connectivity().from_file(str(conn_path))
    elif pc.get("from_h5", False):
        connection = lab.connectivity.Connectivity().from_file(pc["path"])
    elif pc.get("from_folder", False):
        tract_lengths = np.loadtxt(Path(pc["path"]) / "tract_lengths.txt")
        weights = np.loadtxt(Path(pc["path"]) / "weights.txt")
        connection = lab.connectivity.Connectivity(
            tract_lengths=tract_lengths,
            weights=weights,
            region_labels=np.array([], dtype=np.dtype("<U128")),
            centres=np.array([]),
            cortical=None,
        )
    else:
        root = Path(__file__).resolve().parents[3]
        conn_path = root / "data" / "connectivity" / "connectivity_68.zip"
        if not conn_path.exists():
            raise FileNotFoundError(
                "Connectivity source not configured and default DK-68 atlas was not found at "
                f"{conn_path}."
            )
        connection = lab.connectivity.Connectivity().from_file(str(conn_path))

    if pc.get("nullify_diagonals", False):
        connection.weights[np.diag_indices(len(connection.weights))] = 0.0

    if pc.get("normalised", False):
        connection.weights = connection.weights / (np.sum(connection.weights, axis=0) + 1e-12)

    if pc.get("disconnect_regions", []):
        disconnect = pc["disconnect_regions"]
        connection.weights[disconnect] = 0.0
        connection.weights[:, disconnect] = 0.0

    connection.speed = np.array(pc["speed"])
    return connection


def _configure_shared_noise(model, parameter_model: dict[str, Any], connection) -> None:
    """Configure private/shared OU-noise mixing for Zerlaut-family models."""
    noise_alpha = float(parameter_model.get("noise_alpha", 0.0))
    if hasattr(model, "noise_alpha"):
        model.noise_alpha = np.array([noise_alpha], dtype=float)

    n_regions = int(np.asarray(connection.weights).shape[0])
    shared_noise_mode = str(parameter_model.get("shared_noise_mode", "none")).lower()
    if shared_noise_mode in ("none", "private"):
        shared_noise_matrix = np.eye(n_regions, dtype=float)
    elif shared_noise_mode == "global":
        shared_noise_matrix = np.full((n_regions, n_regions), 1.0 / max(n_regions, 1), dtype=float)
    elif shared_noise_mode in ("connectivity", "sc", "weights"):
        weights_nonneg = np.array(connection.weights, dtype=float)
        weights_nonneg = np.maximum(weights_nonneg, 0.0)
        np.fill_diagonal(weights_nonneg, 0.0)
        row_sum = weights_nonneg.sum(axis=1, keepdims=True)
        shared_noise_matrix = np.zeros_like(weights_nonneg)
        non_zero_rows = row_sum[:, 0] > 0.0
        shared_noise_matrix[non_zero_rows] = weights_nonneg[non_zero_rows] / row_sum[non_zero_rows]
        zero_rows = np.where(~non_zero_rows)[0]
        shared_noise_matrix[zero_rows, zero_rows] = 1.0
    else:
        raise ValueError(
            "Unsupported shared_noise_mode. Use one of: 'none', 'global', 'connectivity'. "
            f"Got '{shared_noise_mode}'."
        )

    model._shared_noise_mode = shared_noise_mode
    model._shared_noise_matrix = shared_noise_matrix


def _build_monitors(parameter_monitor: dict):
    monitors = []

    if parameter_monitor.get("Raw", False):
        raw_cls = getattr(lab.monitors, "RawVoi", None)
        if raw_cls is None:
            raw_cls = lab.monitors.Raw
        kwargs = {}
        if "parameter_Raw" in parameter_monitor and "variables_of_interest" in parameter_monitor["parameter_Raw"]:
            kwargs["variables_of_interest"] = np.array(parameter_monitor["parameter_Raw"]["variables_of_interest"])
        monitors.append(raw_cls(**kwargs))

    if parameter_monitor.get("TemporalAverage", False):
        p = parameter_monitor["parameter_TemporalAverage"]
        monitors.append(
            lab.monitors.TemporalAverage(
                variables_of_interest=np.array(p["variables_of_interest"]),
                period=p["period"],
            )
        )

    if parameter_monitor.get("Bold", False):
        p = parameter_monitor["parameter_Bold"]
        monitors.append(
            lab.monitors.Bold(
                variables_of_interest=np.array(p["variables_of_interest"]),
                period=p["period"],
            )
        )

    if parameter_monitor.get("Ca", False):
        p = parameter_monitor["parameter_Ca"]
        monitors.append(
            lab.monitors.Ca(
                variables_of_interest=np.array(p["variables_of_interest"]),
                tau_rise=p["tau_rise"],
                tau_decay=p["tau_decay"],
            )
        )

    return monitors


def _prepare_region_parameter_value(key: str, value: Any, n_regions: int) -> np.ndarray:
    """Validate/shape region-wise model parameters for TVB broadcast semantics."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr

    if arr.ndim == 1:
        if arr.size == 1:
            return np.asarray(float(arr[0]), dtype=float)
        if arr.size != n_regions:
            raise ValueError(
                f"Region-wise parameter '{key}' has length {arr.size}, but connectivity has "
                f"{n_regions} regions."
            )
        return arr.reshape(n_regions, 1)

    if arr.ndim == 2:
        if arr.shape == (n_regions, 1):
            return arr
        if arr.shape == (1, n_regions):
            return arr.reshape(n_regions, 1)
        raise ValueError(
            f"Region-wise parameter '{key}' has invalid shape {arr.shape}; expected scalar, "
            f"({n_regions},), ({n_regions}, 1), or (1, {n_regions})."
        )

    raise ValueError(
        f"Region-wise parameter '{key}' has invalid ndim={arr.ndim}; expected scalar/vector."
    )


def _configure_monitor_mode(parameter_monitor: dict, cfg: WholeBrainConfig) -> None:
    """Apply monitor mode selection from config onto legacy monitor dictionary."""
    if cfg.monitor_mode is None:
        return

    vars_oi = list(cfg.monitor_variables)
    if cfg.monitor_mode == "raw":
        parameter_monitor["Raw"] = True
        parameter_monitor["TemporalAverage"] = False
        parameter_monitor.setdefault("parameter_Raw", {})
        parameter_monitor["parameter_Raw"]["variables_of_interest"] = vars_oi
        return

    if cfg.monitor_mode == "temporal_average":
        parameter_monitor["Raw"] = False
        parameter_monitor["TemporalAverage"] = True
        parameter_monitor.setdefault("parameter_TemporalAverage", {})
        parameter_monitor["parameter_TemporalAverage"]["variables_of_interest"] = vars_oi
        parameter_monitor["parameter_TemporalAverage"]["period"] = float(cfg.temporal_average_period_ms)
        return

    raise ValueError(f"Unsupported monitor_mode={cfg.monitor_mode!r}")


def _build_stimulation(parameter_stimulation: dict, connection, model):
    if parameter_stimulation.get("stimval", 0.0) == 0.0:
        return None

    eqn_t = lab.equations.PulseTrain()
    eqn_t.parameters["onset"] = np.array(parameter_stimulation["stimtime"])
    eqn_t.parameters["tau"] = np.array(parameter_stimulation["stimdur"])
    eqn_t.parameters["T"] = np.array(parameter_stimulation["stimperiod"])

    weights = np.zeros(len(connection.weights))
    stimregion = parameter_stimulation.get("stimregion", None)
    if stimregion is None:
        raise ValueError("stimregion must be provided when stimval is non-zero")
    weights[list(stimregion)] = parameter_stimulation["stimval"]

    model.stvar = parameter_stimulation.get("stimvariables", [0])

    return lab.patterns.StimuliRegion(
        temporal=eqn_t,
        connectivity=connection,
        weight=weights,
    )


def run_whole_brain_simulation(cfg: WholeBrainConfig, seed: int = 0) -> WholeBrainResult:
    """Run a whole-brain simulation.

    Default behavior (`model_family='adex_zerlaut'`) reproduces the same model
    family/parameter logic as legacy TVBSim.
    """
    np.random.seed(seed)
    connection_obj = None

    if cfg.model_family == "generic2d":
        if cfg.connectivity_zip:
            conn = lab.connectivity.Connectivity().from_file(str(cfg.connectivity_zip))
        else:
            conn = lab.connectivity.Connectivity().from_file()
        connection_obj = conn
        model = lab.models.Generic2dOscillator()
        coupling = lab.coupling.Linear(a=np.array([cfg.coupling_strength]))
        integrator = lab.integrators.HeunDeterministic(dt=cfg.dt_ms)
        if cfg.monitor_mode == "temporal_average":
            monitors = (
                lab.monitors.TemporalAverage(
                    variables_of_interest=np.array(list(cfg.monitor_variables)),
                    period=float(cfg.temporal_average_period_ms),
                ),
            )
        else:
            monitors = (lab.monitors.Raw(),)
        sim = lab.simulator.Simulator(
            model=model,
            connectivity=conn,
            coupling=coupling,
            integrator=integrator,
            monitors=monitors,
        ).configure()
        output = sim.run(simulation_length=cfg.simulation_length_ms)
    else:
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
        _configure_monitor_mode(parameters.parameter_monitor, cfg)

        model = _select_zerlaut_model(parameters.parameter_model)

        connection = _build_connectivity(parameters, cfg)
        connection_obj = connection
        n_regions = int(np.asarray(connection.weights).shape[0])

        # Match legacy model-parameter assignment behavior.
        to_skip = ["initial_condition", "matteo", "order", "gK_gNa", "noise_alpha", "shared_noise_mode"]
        regionwise_keys = {
            "g_K_e",
            "g_K_i",
            "g_Na_e",
            "g_Na_i",
            "E_L_e",
            "E_L_i",
        }
        for key, value in parameters.parameter_model.items():
            if key in to_skip:
                continue
            arr = np.array(value)
            if key in regionwise_keys:
                arr = _prepare_region_parameter_value(key, value, n_regions)
            setattr(model, key, arr)
        for key, val in parameters.parameter_model["initial_condition"].items():
            model.state_variable_range[key] = val

        _configure_shared_noise(model, parameters.parameter_model, connection)

        pc = parameters.parameter_coupling
        coupling = getattr(lab.coupling, pc["type"])(
            **{k: np.array(v) for k, v in pc["coupling_parameter"].items()}
        )

        pi = parameters.parameter_integrator
        if not pi["stochastic"]:
            if pi["type"] == "Heun":
                integrator = lab.integrators.HeunDeterministic(dt=np.array(pi["dt"]))
            elif pi["type"] == "Euler":
                integrator = lab.integrators.EulerDeterministic(dt=np.array(pi["dt"]))
            else:
                raise ValueError(f"Unsupported deterministic integrator: {pi['type']}")
        else:
            if pi["noise_type"] != "Additive":
                raise ValueError("Only Additive stochastic noise is currently supported for parity mode.")
            # Re-shape legacy nsig safely if model dimensionality changed
            # (e.g. switching Zerlaut order 2 <-> 1 in notebooks).
            nsig_raw = np.asarray(pi["noise_parameter"]["nsig"], dtype=float)
            nvar = max(1, int(getattr(model, "nvar", max(1, int(nsig_raw.size)))))
            # `connection.number_of_regions` can remain 0 in some construction
            # paths before full TVB configuration; derive robustly from weights.
            weights = np.asarray(connection.weights)
            weights_regions = int(weights.shape[0]) if weights.ndim >= 2 else int(weights.size)
            conn_regions = int(getattr(connection, "number_of_regions", 0) or 0)
            n_regions = max(1, conn_regions, weights_regions)
            n_modes = max(1, int(getattr(model, "number_of_modes", 1)))

            def _default_nsig_value() -> float:
                if nsig_raw.size > 0:
                    return float(np.ravel(nsig_raw)[0])
                return 1e-5

            base = np.full((nvar, n_regions), _default_nsig_value(), dtype=float)

            if nsig_raw.ndim == 0:
                base[:, :] = float(nsig_raw)
            elif nsig_raw.ndim == 1:
                vec = np.ravel(nsig_raw)
                if vec.size == 0:
                    pass
                elif vec.size == 1:
                    base[:, :] = float(vec[0])
                else:
                    if vec.size < nvar:
                        vec = np.pad(vec, (0, nvar - vec.size), mode="edge")
                    elif vec.size > nvar:
                        # Legacy parity: keep trailing entries so that the
                        # dedicated noise-state coefficient (last index) is
                        # preserved when switching 8-var -> 5-var models.
                        vec = vec[-nvar:]
                    base = np.repeat(vec[:, None], n_regions, axis=1)
            elif nsig_raw.ndim >= 2:
                arr = np.asarray(nsig_raw, dtype=float)
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)

                # Ensure non-empty 2D noise table.
                if arr.shape[0] == 0 or arr.shape[1] == 0:
                    arr = np.full((max(1, arr.shape[0]), max(1, arr.shape[1])), _default_nsig_value(), dtype=float)

                if arr.shape[0] < nvar:
                    arr = np.vstack([arr, np.repeat(arr[-1:, :], nvar - arr.shape[0], axis=0)])
                elif arr.shape[0] > nvar:
                    arr = arr[-nvar:, :]
                arr = arr[:nvar, :]

                if arr.shape[1] < n_regions:
                    arr = np.hstack([arr, np.repeat(arr[:, -1:], n_regions - arr.shape[1], axis=1)])
                arr = arr[:, :n_regions]
                base = arr

            # TVB stochastic integrators are safest with explicit (nvar, n_regions, n_modes)
            nsig = np.repeat(base[:, :, None], n_modes, axis=2)

            noise = lab.noise.Additive(
                nsig=nsig,
                ntau=pi["noise_parameter"]["ntau"],
            )
            noise.random_stream.seed(seed)
            if pi["type"] == "Heun":
                integrator = lab.integrators.HeunStochastic(noise=noise, dt=pi["dt"])
            elif pi["type"] == "Euler":
                integrator = lab.integrators.EulerStochastic(noise=noise, dt=pi["dt"])
            else:
                raise ValueError(f"Unsupported stochastic integrator: {pi['type']}")

        monitors = _build_monitors(parameters.parameter_monitor)
        if not monitors:
            raise ValueError("No monitors configured. Set monitor_mode or monitor parameters.")
        stimulation = _build_stimulation(parameters.parameter_stimulus, connection, model)

        simulator = lab.simulator.Simulator(
            model=model,
            connectivity=connection,
            coupling=coupling,
            integrator=integrator,
            monitors=monitors,
            stimulus=stimulation,
        )
        simulator.configure()
        output = simulator.run(simulation_length=cfg.simulation_length_ms)

    if not output or output[0] is None:
        raise RuntimeError("TVB simulation returned no output.")

    t, data = output[0]
    data = np.asarray(data)
    raw = data[:, 0, :, 0]
    raw_inh = data[:, 1, :, 0] if data.ndim == 4 and data.shape[1] > 1 else None

    if connection_obj is not None and getattr(connection_obj, "region_labels", None) is not None and len(connection_obj.region_labels):
        region_labels = np.asarray(connection_obj.region_labels)
    else:
        region_labels = np.array([f"R{i}" for i in range(raw.shape[1])], dtype="U16")
    return WholeBrainResult(
        time_ms=np.asarray(t).reshape(-1),
        raw=raw,
        raw_inh=raw_inh,
        region_labels=region_labels,
        full_monitor_output=output,
    )
