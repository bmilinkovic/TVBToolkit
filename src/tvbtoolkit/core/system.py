"""System introspection and runtime-parallel tuning helpers."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SystemSpecs:
    """Minimal system specs used for reproducibility and scheduling."""

    os_name: str
    os_version: str
    machine: str
    cpu_brand: str
    physical_cores: int
    logical_cores: int
    memory_gb: float
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def _read_sysctl(name: str) -> str | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        return out
    except Exception:
        return None


def detect_system_specs() -> SystemSpecs:
    """Collect host specs with graceful fallbacks."""
    cpu_brand = _read_sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    physical = _read_sysctl("hw.physicalcpu")
    logical = _read_sysctl("hw.logicalcpu")
    mem_bytes = _read_sysctl("hw.memsize")

    logical_cores = int(logical) if logical and logical.isdigit() else (os.cpu_count() or 1)
    physical_cores = int(physical) if physical and physical.isdigit() else max(1, logical_cores)
    memory_gb = float(mem_bytes) / (1024**3) if mem_bytes and mem_bytes.isdigit() else 0.0

    return SystemSpecs(
        os_name=platform.system(),
        os_version=platform.version(),
        machine=platform.machine(),
        cpu_brand=cpu_brand,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        memory_gb=round(memory_gb, 1),
        notes="TVB/Brian2 execution path is CPU-based; integrated GPU is not used by default.",
    )


def recommend_parallel_workers(task: str = "whole_brain_tvb") -> int:
    """Recommend a conservative worker count for laptop stability.

    Args:
        task: One of `whole_brain_tvb`, `single_region_adex`, `complexity_only`.
    """
    specs = detect_system_specs()
    p = max(1, specs.physical_cores)

    if task == "whole_brain_tvb":
        # Heavy memory footprint per process: use about half physical cores.
        return max(1, min(p // 2, p - 1))
    if task == "single_region_adex":
        return max(1, min((2 * p) // 3, p - 1))
    if task == "complexity_only":
        return max(1, min(p - 1, specs.logical_cores - 1))
    return max(1, p // 2)

