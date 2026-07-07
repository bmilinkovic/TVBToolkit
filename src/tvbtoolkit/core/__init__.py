"""Core models and I/O."""

from tvbtoolkit.core.config import OutputConfig, SingleRegionConfig, WholeBrainConfig
from tvbtoolkit.core.io import load_npz, save_npz
from tvbtoolkit.core.paths import (
    DATA_DOC_LIEGE,
    DATA_DRUGS_MAASTRICHT,
    DATA_STIMULATION_LIEGE,
    LEGACY_UNSORTED,
    cnrs_root,
    dataset_path,
    dataset_root,
    doc_liege_raw,
    doc_liege_results,
    drugs_raw,
    drugs_results,
    legacy_results,
    legacy_tmp,
    project_root,
    stimulation_raw,
    stimulation_results,
)
from tvbtoolkit.core.system import SystemSpecs, detect_system_specs, recommend_parallel_workers

__all__ = [
    "WholeBrainConfig",
    "SingleRegionConfig",
    "OutputConfig",
    "save_npz",
    "load_npz",
    "SystemSpecs",
    "detect_system_specs",
    "recommend_parallel_workers",
    "DATA_DOC_LIEGE",
    "DATA_DRUGS_MAASTRICHT",
    "DATA_STIMULATION_LIEGE",
    "LEGACY_UNSORTED",
    "cnrs_root",
    "project_root",
    "dataset_root",
    "dataset_path",
    "doc_liege_raw",
    "doc_liege_results",
    "drugs_raw",
    "drugs_results",
    "stimulation_raw",
    "stimulation_results",
    "legacy_results",
    "legacy_tmp",
]
