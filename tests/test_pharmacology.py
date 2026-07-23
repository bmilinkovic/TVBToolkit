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


def test_get_5ht2a_aal90_aligns_values_by_region_label() -> None:
    source_order = get_5ht2a_aal90()

    # Use the real table labels while requesting their reverse order.  This
    # catches accidental positional assignment without hard-coding PET values.
    import csv
    from pathlib import Path

    table_path = Path(__file__).resolve().parents[1] / "data" / "receptors" / "hansen_receptors_aal90.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source_labels = np.asarray([row[""] for row in rows], dtype=object)
    requested = source_labels[::-1]

    aligned = get_5ht2a_aal90(target_labels=requested)
    np.testing.assert_allclose(aligned, source_order[::-1])


def test_get_5ht2a_aal90_rejects_unmatched_target_labels() -> None:
    labels = np.asarray([f"not_a_region_{i}" for i in range(90)], dtype=object)
    with pytest.raises(ValueError, match="Could not align"):
        get_5ht2a_aal90(target_labels=labels)


def test_get_5ht2a_aal90_rejects_duplicate_target_labels() -> None:
    labels = np.asarray(["Precentral_L"] * 90, dtype=object)
    with pytest.raises(ValueError, match="must be unique"):
        get_5ht2a_aal90(target_labels=labels)
