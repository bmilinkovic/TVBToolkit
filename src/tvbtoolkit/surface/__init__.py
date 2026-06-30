"""Surface-based TVB simulation APIs."""

from tvbtoolkit.core.config import SurfaceConfig
from tvbtoolkit.surface.mapping import (
    average_nodes_to_regions,
    cortical_region_mapping,
    full_region_mapping,
    prepare_surface_parameter_value,
    validate_region_mapping,
)

try:
    from tvbtoolkit.surface.io import load_surface_cortex
    from tvbtoolkit.surface.simulation import SurfaceResult, run_surface_adex_simulation
except Exception as _surface_import_error:
    load_surface_cortex = None
    SurfaceResult = None
    run_surface_adex_simulation = None

__all__ = [
    "SurfaceConfig",
    "SurfaceResult",
    "run_surface_adex_simulation",
    "load_surface_cortex",
    "average_nodes_to_regions",
    "cortical_region_mapping",
    "full_region_mapping",
    "prepare_surface_parameter_value",
    "validate_region_mapping",
]
