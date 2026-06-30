"""Surface-to-region mapping helpers."""

from __future__ import annotations

import numpy as np


def cortical_region_mapping(cortex) -> np.ndarray:
    """Return the vertex-to-region mapping stored in a TVB cortex object."""
    return np.asarray(cortex.region_mapping_data.array_data, dtype=int).reshape(-1)


def full_region_mapping(cortex) -> np.ndarray:
    """Return TVB's full surface mapping, including unmapped non-cortical nodes."""
    return np.asarray(cortex.region_mapping, dtype=int).reshape(-1)


def validate_region_mapping(region_mapping: np.ndarray, n_regions: int) -> np.ndarray:
    """Validate a node-to-region mapping against a connectivity region count."""
    mapping = np.asarray(region_mapping, dtype=int).reshape(-1)
    if mapping.size == 0:
        raise ValueError("Surface region mapping is empty.")
    if np.any(mapping < 0) or np.any(mapping >= int(n_regions)):
        raise ValueError(
            "Surface region mapping contains indices outside the connectivity "
            f"range 0..{int(n_regions) - 1}."
        )
    return mapping


def prepare_surface_parameter_value(
    key: str,
    value,
    region_mapping: np.ndarray,
    n_regions: int,
    *,
    n_vertices: int | None = None,
) -> np.ndarray:
    """Shape a scalar, region-wise, or node-wise parameter for TVB surface runs.

    Region-wise vectors are expanded through ``region_mapping``. Node-wise
    vectors are accepted as-is. Vertex-wise vectors are accepted only when the
    surface has no extra non-cortical nodes in TVB's full mapping.
    """
    mapping = validate_region_mapping(region_mapping, n_regions)
    arr = np.asarray(value, dtype=float)
    n_nodes = int(mapping.size)

    if arr.ndim == 0:
        return arr

    flat = arr.reshape(-1) if arr.ndim <= 2 else None
    if flat is None:
        raise ValueError(
            f"Surface parameter '{key}' has invalid ndim={arr.ndim}; expected scalar/vector."
        )

    if flat.size == 1:
        return np.asarray(float(flat[0]), dtype=float)
    if flat.size == int(n_regions):
        return flat[mapping].reshape(n_nodes, 1)
    if flat.size == n_nodes:
        return flat.reshape(n_nodes, 1)
    if n_vertices is not None and flat.size == int(n_vertices):
        if int(n_vertices) != n_nodes:
            raise ValueError(
                f"Surface parameter '{key}' has one value per cortical vertex "
                f"({n_vertices}), but TVB's full surface mapping has {n_nodes} "
                "nodes including non-cortical regions. Provide region-wise or "
                "full node-wise values instead."
            )
        return flat.reshape(n_nodes, 1)

    raise ValueError(
        f"Surface parameter '{key}' has length {flat.size}; expected scalar, "
        f"{n_regions} region values, or {n_nodes} node values."
    )


def average_nodes_to_regions(
    node_timeseries: np.ndarray,
    region_mapping: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    """Average a ``(time, node)`` array back to ``(time, region)``."""
    x = np.asarray(node_timeseries, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected node_timeseries with shape (time, node), got {x.shape}.")
    mapping = validate_region_mapping(region_mapping, n_regions)
    if x.shape[1] != mapping.size:
        raise ValueError(
            f"node_timeseries has {x.shape[1]} nodes, but mapping has {mapping.size} entries."
        )

    out = np.full((x.shape[0], int(n_regions)), np.nan, dtype=float)
    counts = np.bincount(mapping, minlength=int(n_regions)).astype(float)
    for region in np.where(counts > 0)[0]:
        out[:, region] = x[:, mapping == region].mean(axis=1)
    return out
