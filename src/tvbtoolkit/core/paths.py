"""Shared filesystem roots for CNRS/TVBToolkit data and outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

DatasetKey = Literal[
    "data_drugs_maastricht",
    "data_doc_liege",
    "data_stimulation_liege",
    "legacy_unsorted",
]

DATA_DRUGS_MAASTRICHT: DatasetKey = "data_drugs_maastricht"
DATA_DOC_LIEGE: DatasetKey = "data_doc_liege"
DATA_STIMULATION_LIEGE: DatasetKey = "data_stimulation_liege"
LEGACY_UNSORTED: DatasetKey = "legacy_unsorted"

_DEFAULT_CNRS_ROOT = Path("/Volumes/ex_data/cnrs")


def cnrs_root() -> Path:
    """Return the external CNRS data root.

    Override with ``CNRS_DATA_ROOT`` when running on a cluster or another
    workstation. The local default matches the lab external-drive layout.
    """

    return Path(os.environ.get("CNRS_DATA_ROOT", _DEFAULT_CNRS_ROOT)).expanduser()


def project_root() -> Path:
    """Return the TVBToolkit repository root."""

    return Path(__file__).resolve().parents[3]


def dataset_root(dataset: DatasetKey, *, create: bool = False) -> Path:
    root = cnrs_root() / dataset
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_path(
    dataset: DatasetKey,
    purpose: Literal["raw", "results", "legacy", "tmp"],
    *parts: str | Path,
    create: bool = False,
) -> Path:
    path = dataset_root(dataset) / purpose
    for part in parts:
        path = path / part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def drugs_raw(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(DATA_DRUGS_MAASTRICHT, "raw", *parts, create=create)


def drugs_results(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(DATA_DRUGS_MAASTRICHT, "results", *parts, create=create)


def doc_liege_raw(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(DATA_DOC_LIEGE, "raw", *parts, create=create)


def doc_liege_results(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(DATA_DOC_LIEGE, "results", *parts, create=create)


def stimulation_raw(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(DATA_STIMULATION_LIEGE, "raw", *parts, create=create)


def stimulation_results(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(DATA_STIMULATION_LIEGE, "results", *parts, create=create)


def legacy_results(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(LEGACY_UNSORTED, "results", *parts, create=create)


def legacy_tmp(*parts: str | Path, create: bool = False) -> Path:
    return dataset_path(LEGACY_UNSORTED, "tmp", *parts, create=create)
