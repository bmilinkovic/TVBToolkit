"""TVB surface asset loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.local_connectivity import LocalConnectivity

from tvbtoolkit.core.config import SurfaceConfig
from tvbtoolkit.surface.mapping import cortical_region_mapping, validate_region_mapping


def _path_or_none(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def load_surface_cortex(cfg: SurfaceConfig, connectivity) -> Cortex:
    """Load and validate a TVB ``Cortex`` for surface simulation.

    The cortex is assembled from a cortical surface, region mapping, and local
    connectivity source. If no local-connectivity file or matrix is supplied,
    TVB will compute one during ``Cortex.configure()``, which requires the
    optional ``gdist`` package.
    """
    cortex = Cortex.from_file(
        source_file=_path_or_none(cfg.surface_file),
        region_mapping_file=_path_or_none(cfg.region_mapping_file),
        local_connectivity_file=_path_or_none(cfg.local_connectivity_file),
    )

    if cortex.region_mapping_data is None:
        raise ValueError("Surface configuration did not produce a RegionMapping.")
    if cortex.region_mapping_data.surface is None:
        raise ValueError("Surface configuration did not produce a cortical surface.")

    cortex.region_mapping_data.connectivity = connectivity
    surface = cortex.region_mapping_data.surface

    if cfg.local_connectivity_matrix is not None:
        matrix = scipy.sparse.csr_matrix(cfg.local_connectivity_matrix)
        cortex.local_connectivity = LocalConnectivity(
            surface=surface,
            matrix=matrix,
            cutoff=float(cfg.local_connectivity_cutoff_mm),
        )
    elif cortex.local_connectivity is None:
        cortex.local_connectivity = LocalConnectivity(
            surface=surface,
            cutoff=float(cfg.local_connectivity_cutoff_mm),
        )
    else:
        cortex.local_connectivity.surface = surface
        cortex.local_connectivity.cutoff = float(cfg.local_connectivity_cutoff_mm)

    cortex.coupling_strength = np.array([float(cfg.local_coupling_strength)], dtype=float)

    n_regions = int(np.asarray(connectivity.weights).shape[0])
    validate_region_mapping(cortical_region_mapping(cortex), n_regions)
    return cortex
