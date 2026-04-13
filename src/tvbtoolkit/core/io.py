"""Lightweight I/O helpers for simulation and metric artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz(path: str | Path, **arrays: Any) -> Path:
    """Save arrays to compressed `.npz` and ensure parent directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a `.npz` archive into a plain dictionary."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}

