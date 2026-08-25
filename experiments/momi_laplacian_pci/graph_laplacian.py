"""Graph operators used by the Momi-style TMS propagation experiment.

Momi, Wang, and Griffiths (2023) use the conventional graph Laplacian
``L = D - A``, where ``D`` contains row-wise weighted degrees, with Frobenius
normalization. Their released dynamical model applies ``-L`` to activity:
``A @ x_delayed - D @ x_current``. Keeping the mathematical Laplacian and the
coupling sign separate avoids an otherwise easy-to-miss sign ambiguity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LaplacianDiagnostics:
    """Numerical and structural provenance for one transformed connectome."""

    n_regions: int
    nonzero_undirected_edges: int
    disconnected_edge_fraction: float
    adjacency_sum: float
    adjacency_frobenius_norm: float
    laplacian_frobenius_norm_before_scaling: float
    degree_min: float
    degree_median: float
    degree_max: float
    eigenvalue_min: float
    eigenvalue_max: float
    max_abs_row_sum: float
    isolated_nodes: int
    convention: str = "combinatorial_D_minus_A"
    scaling: str = "frobenius_norm"

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _validate_adjacency(adjacency: np.ndarray, *, atol: float) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square two-dimensional matrix")
    if matrix.shape[0] < 2:
        raise ValueError("adjacency must contain at least two nodes")
    if not np.isfinite(matrix).all():
        raise ValueError("adjacency contains NaN or infinite values")
    if np.min(matrix) < -atol:
        raise ValueError("adjacency must be non-negative before transformation")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=atol):
        raise ValueError("Momi-style Laplacian requires an undirected matrix")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=atol):
        raise ValueError("adjacency diagonal must be zero before transformation")
    matrix = matrix.copy()
    matrix[np.abs(matrix) <= atol] = 0.0
    return matrix


def momi_laplacian(
    adjacency: np.ndarray,
    *,
    atol: float = 1e-10,
) -> tuple[np.ndarray, LaplacianDiagnostics]:
    """Return the normalized combinatorial Laplacian ``(D-A) / ||D-A||F``."""

    matrix = _validate_adjacency(adjacency, atol=atol)
    degree = np.sum(matrix, axis=1)
    laplacian = np.diag(degree) - matrix
    norm = float(np.linalg.norm(laplacian, ord="fro"))
    if not np.isfinite(norm) or norm <= atol:
        raise ValueError("cannot normalize a connectome with no weighted edges")
    transformed = laplacian / norm

    row_residual = float(np.max(np.abs(np.sum(transformed, axis=1))))
    eigenvalues = np.linalg.eigvalsh(transformed)
    if row_residual > max(100.0 * atol, 1e-12):
        raise RuntimeError("normalized Laplacian rows do not sum to zero")
    if float(eigenvalues[0]) < -max(100.0 * atol, 1e-12):
        raise RuntimeError("graph Laplacian has an unexpected negative eigenvalue")

    n_regions = matrix.shape[0]
    upper = np.triu(matrix, k=1)
    possible_edges = n_regions * (n_regions - 1) // 2
    nonzero_edges = int(np.count_nonzero(upper))
    diagnostics = LaplacianDiagnostics(
        n_regions=n_regions,
        nonzero_undirected_edges=nonzero_edges,
        disconnected_edge_fraction=float(1.0 - nonzero_edges / possible_edges),
        adjacency_sum=float(np.sum(matrix)),
        adjacency_frobenius_norm=float(np.linalg.norm(matrix, ord="fro")),
        laplacian_frobenius_norm_before_scaling=norm,
        degree_min=float(np.min(degree)),
        degree_median=float(np.median(degree)),
        degree_max=float(np.max(degree)),
        eigenvalue_min=float(eigenvalues[0]),
        eigenvalue_max=float(eigenvalues[-1]),
        max_abs_row_sum=row_residual,
        isolated_nodes=int(np.count_nonzero(degree <= atol)),
    )
    return transformed, diagnostics


def validate_tract_lengths(
    tract_lengths: np.ndarray,
    adjacency: np.ndarray,
    *,
    atol: float = 1e-10,
) -> dict[str, float | int]:
    """Validate delay lengths without using them to reweight the Laplacian."""

    lengths = np.asarray(tract_lengths, dtype=np.float64)
    weights = np.asarray(adjacency, dtype=np.float64)
    if lengths.shape != weights.shape:
        raise ValueError("tract lengths and adjacency must have identical shapes")
    if not np.isfinite(lengths).all() or np.min(lengths) < -atol:
        raise ValueError("tract lengths must be finite and non-negative")
    if not np.allclose(lengths, lengths.T, rtol=0.0, atol=atol):
        raise ValueError("tract-length matrix must be symmetric")
    edge_mask = weights > atol
    missing = edge_mask & (lengths <= atol)
    if np.any(missing):
        raise ValueError(
            f"{int(np.count_nonzero(missing))} weighted directed edges lack tract lengths"
        )
    values = lengths[edge_mask]
    return {
        "weighted_directed_edges": int(np.count_nonzero(edge_mask)),
        "zero_weight_nonzero_length_entries": int(
            np.count_nonzero((~edge_mask) & (lengths > atol))
        ),
        "tract_length_min_mm": float(np.min(values)),
        "tract_length_median_mm": float(np.median(values)),
        "tract_length_max_mm": float(np.max(values)),
    }
