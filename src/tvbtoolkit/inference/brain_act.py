"""Load paired Brain-Act/DoC SC and BOLD records for parameter inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.io

ROIOrder = Literal["as_stored", "aal90_fc_to_symmetric"]


def aal90_interleaved_to_symmetric_index() -> np.ndarray:
    """Return the audited Brain-Act BOLD interleaved-to-symmetric permutation."""
    return np.concatenate((np.arange(0, 90, 2), np.arange(1, 90, 2)[::-1])).astype(int)


@dataclass(frozen=True)
class BrainActBOLDRecord:
    """One paired empirical BOLD/SC observation."""

    bold: np.ndarray
    structural_connectivity: np.ndarray
    tract_lengths: np.ndarray | None
    tr_seconds: float
    subject_id: str
    cohort: str
    stage: str
    sedation: str
    source: str
    roi_order: str

    @property
    def n_regions(self) -> int:
        return int(self.bold.shape[1])


def _matlab_scalar(value: Any, default: str = "") -> str:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace").strip()
    return str(item).strip()


def _orient_bold(bold: np.ndarray, n_regions: int) -> np.ndarray:
    x = np.asarray(bold, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"BOLD must be 2D, got {x.shape}.")
    if x.shape[1] == n_regions:
        return x
    if x.shape[0] == n_regions:
        return x.T
    raise ValueError(
        f"Neither BOLD axis matches SC region count {n_regions}; got {x.shape}."
    )


def _validate_square_matrix(matrix: np.ndarray, *, name: str, n_regions: int) -> np.ndarray:
    out = np.asarray(matrix, dtype=float).copy()
    if out.shape != (n_regions, n_regions):
        raise ValueError(f"{name} must have shape {(n_regions, n_regions)}, got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or Inf.")
    if np.any(out < 0):
        raise ValueError(f"{name} contains negative values.")
    if not np.allclose(out, out.T, atol=1e-8, rtol=1e-6):
        raise ValueError(f"{name} is not symmetric.")
    np.fill_diagonal(out, 0.0)
    return out


def load_brain_act_bold_mat(
    path: str | Path,
    *,
    tr_seconds: float = 2.4,
    roi_order: ROIOrder = "aal90_fc_to_symmetric",
    tract_lengths: np.ndarray | str | Path | None = None,
    normalize_sc: bool = True,
) -> BrainActBOLDRecord:
    """Load one exported Brain-Act ``.mat`` containing ``SC`` and ``BOLD``.

    The audited DoC files store BOLD in interleaved AAL90 order and SC in the
    symmetric order. The default therefore reorders only the BOLD columns.
    """
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    data = scipy.io.loadmat(source, squeeze_me=True, struct_as_record=False)
    if "SC" not in data or "BOLD" not in data:
        raise KeyError(f"{source} must contain variables 'SC' and 'BOLD'.")

    sc_raw = np.asarray(data["SC"], dtype=float)
    if sc_raw.ndim != 2 or sc_raw.shape[0] != sc_raw.shape[1]:
        raise ValueError(f"SC must be square, got {sc_raw.shape}.")
    n_regions = int(sc_raw.shape[0])
    sc = _validate_square_matrix(sc_raw, name="SC", n_regions=n_regions)
    if normalize_sc:
        maximum = float(np.max(sc))
        if maximum > 0:
            sc /= maximum

    bold = _orient_bold(data["BOLD"], n_regions=n_regions)
    if not np.all(np.isfinite(bold)):
        raise ValueError("BOLD contains NaN or Inf; exclude or clean the subject explicitly.")

    if roi_order == "aal90_fc_to_symmetric":
        if n_regions != 90:
            raise ValueError("The audited AAL90 permutation requires exactly 90 regions.")
        bold = bold[:, aal90_interleaved_to_symmetric_index()]
    elif roi_order != "as_stored":
        raise ValueError(f"Unsupported roi_order={roi_order!r}.")

    lengths_out: np.ndarray | None = None
    if tract_lengths is not None:
        lengths_value = (
            np.loadtxt(Path(tract_lengths).expanduser().resolve(), dtype=float)
            if isinstance(tract_lengths, (str, Path))
            else np.asarray(tract_lengths, dtype=float)
        )
        lengths_out = _validate_square_matrix(
            lengths_value, name="tract_lengths", n_regions=n_regions
        )

    return BrainActBOLDRecord(
        bold=np.asarray(bold, dtype=float),
        structural_connectivity=sc,
        tract_lengths=lengths_out,
        tr_seconds=float(tr_seconds),
        subject_id=_matlab_scalar(data.get("subject_id"), source.stem),
        cohort=_matlab_scalar(data.get("cohort"), _matlab_scalar(data.get("condition"))),
        stage=_matlab_scalar(data.get("stage")),
        sedation=_matlab_scalar(data.get("sedation")),
        source=str(source),
        roi_order=roi_order,
    )
