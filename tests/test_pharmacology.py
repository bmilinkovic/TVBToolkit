from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90


def test_get_5ht2a_aal90_loads_static_atlas() -> None:
    receptor_map = get_5ht2a_aal90()

    assert receptor_map.shape == (90,)
    assert np.isfinite(receptor_map).all()
    assert receptor_map.min() >= 0.0
    assert receptor_map.max() <= 1.0


def test_get_5ht2a_aal90_rejects_unknown_tracer() -> None:
    with pytest.raises(ValueError, match="tracer must be one of"):
        get_5ht2a_aal90("unknown")
