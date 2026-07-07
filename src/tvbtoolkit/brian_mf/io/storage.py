"""Storage helpers for brian_MF parameter databases and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from tvbtoolkit.core.paths import legacy_results


@dataclass(frozen=True)
class ParameterSetRecord:
    """A stored parameter set with metadata."""

    name: str
    params: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    file: Path


def _db_dir(path: str | Path | None) -> Path:
    return Path(path) if path is not None else legacy_results("brian_mf", "param_db")


def save_param_set(name: str, params: dict[str, Any], metadata: dict[str, Any], path: str | Path | None = None) -> Path:
    """Save a fitted parameter set in JSON format.

    Metadata is expected to include at least:
    `cell_type`, `species`, `temperature`, `recording_condition`, `source`,
    `toolbox_version`, and `date`.
    """

    out_dir = _db_dir(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.json"
    payload = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metadata": metadata,
    }
    out_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_file


def load_param_set(name_or_id: str, path: str | Path | None = None) -> ParameterSetRecord:
    """Load a stored parameter set by name (without `.json`) or filename."""

    in_dir = _db_dir(path)
    candidate = in_dir / name_or_id
    if candidate.suffix != ".json":
        candidate = in_dir / f"{name_or_id}.json"
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return ParameterSetRecord(
        name=payload["name"],
        params=payload["params"],
        metadata=payload["metadata"],
        created_at=payload["created_at"],
        file=candidate,
    )


def list_param_sets(filters: dict[str, Any] | None = None, path: str | Path | None = None) -> list[ParameterSetRecord]:
    """List parameter sets, optionally filtering by metadata key/value pairs."""

    in_dir = _db_dir(path)
    if not in_dir.exists():
        return []
    records: list[ParameterSetRecord] = []
    for file in sorted(in_dir.glob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        rec = ParameterSetRecord(
            name=payload["name"],
            params=payload["params"],
            metadata=payload["metadata"],
            created_at=payload["created_at"],
            file=file,
        )
        if filters:
            keep = True
            for key, value in filters.items():
                if rec.metadata.get(key) != value:
                    keep = False
                    break
            if not keep:
                continue
        records.append(rec)
    return records
