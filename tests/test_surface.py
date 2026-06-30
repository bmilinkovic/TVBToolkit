import numpy as np
import pytest

from tvbtoolkit.core.config import SurfaceConfig
from tvbtoolkit.surface.mapping import (
    average_nodes_to_regions,
    prepare_surface_parameter_value,
    validate_region_mapping,
)


def test_surface_config_defaults_to_region_average_adex():
    cfg = SurfaceConfig()

    assert cfg.model_family == "adex_zerlaut"
    assert cfg.zerlaut_order == 2
    assert cfg.monitor_mode == "spatial_average"
    assert cfg.coupling_strength == 0.3


def test_regionwise_parameter_expands_to_surface_nodes():
    mapping = np.array([0, 1, 0, 2])
    values = np.array([10.0, 20.0, 30.0])

    out = prepare_surface_parameter_value("E_L_e", values, mapping, n_regions=3)

    assert out.shape == (4, 1)
    assert np.allclose(out[:, 0], [10.0, 20.0, 10.0, 30.0])


def test_nodewise_parameter_is_preserved():
    mapping = np.array([0, 1, 0, 2])
    values = np.array([10.0, 11.0, 12.0, 13.0])

    out = prepare_surface_parameter_value("g_K_e", values, mapping, n_regions=3)

    assert out.shape == (4, 1)
    assert np.allclose(out[:, 0], values)


def test_average_nodes_to_regions():
    mapping = np.array([0, 1, 0, 2])
    x = np.array(
        [
            [1.0, 10.0, 3.0, 30.0],
            [5.0, 20.0, 7.0, 40.0],
        ]
    )

    out = average_nodes_to_regions(x, mapping, n_regions=3)

    assert out.shape == (2, 3)
    assert np.allclose(out, [[2.0, 10.0, 30.0], [6.0, 20.0, 40.0]])


def test_invalid_mapping_raises():
    with pytest.raises(ValueError, match="outside the connectivity"):
        validate_region_mapping(np.array([0, 3]), n_regions=3)
