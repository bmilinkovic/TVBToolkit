"""Helpers for reproducing Luppi et al. (2022) style PhiID analyses.

This module focuses on the Figure 1 to 4 analysis pattern from:

- Luppi et al. (2022), Nature Neuroscience

The goal here is not to hard-code the paper's atlas-specific resources, but to
provide the reusable computations needed once subject- or cohort-level synergy
and redundancy matrices have been generated.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def compute_fc_matrix(timeseries: np.ndarray) -> np.ndarray:
    """Return Pearson FC from ``(time, regions)`` or ``(regions, time)`` data."""
    x = np.asarray(timeseries, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {x.shape}.")
    if x.shape[0] == x.shape[1]:
        raise ValueError("Ambiguous square timeseries array; expected time x region data.")
    if x.shape[0] < x.shape[1]:
        x = x.T
    return np.corrcoef(x, rowvar=False)


def upper_triangle_values(matrix: np.ndarray, *, k: int = 1) -> np.ndarray:
    """Return upper-triangle values from a square matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {arr.shape}.")
    idx = np.triu_indices(arr.shape[0], k=k)
    return arr[idx]


def matrix_spearman_similarity(a: np.ndarray, b: np.ndarray, *, k: int = 1) -> float:
    """Compare two symmetric matrices using Spearman correlation of upper triangles."""
    xa = upper_triangle_values(a, k=k)
    xb = upper_triangle_values(b, k=k)
    mask = np.isfinite(xa) & np.isfinite(xb)
    if not np.any(mask):
        raise ValueError("No finite entries available for similarity computation.")
    rho = spearmanr(xa[mask], xb[mask]).statistic
    return float(rho)


def threshold_top_density(
    matrix: np.ndarray,
    density: float,
    *,
    include_diagonal: bool = False,
) -> np.ndarray:
    """Keep only the strongest positive edges at a requested density."""
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {arr.shape}.")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1.")

    n = arr.shape[0]
    tri_k = 0 if include_diagonal else 1
    tri = np.triu_indices(n, k=tri_k)
    values = arr[tri]
    positive = np.where(np.isfinite(values) & (values > 0))[0]
    n_keep = int(np.floor(density * positive.size))

    out = np.zeros_like(arr)
    if n_keep <= 0:
        return out

    keep_local = positive[np.argsort(values[positive])[-n_keep:]]
    rows = tri[0][keep_local]
    cols = tri[1][keep_local]
    out[rows, cols] = arr[rows, cols]
    out[cols, rows] = arr[rows, cols]
    if include_diagonal:
        np.fill_diagonal(out, np.diag(arr))
    return out


def nodal_strength(matrix: np.ndarray, *, exclude_diagonal: bool = True) -> np.ndarray:
    """Return nodal strength as the row-wise sum of edge weights."""
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {arr.shape}.")
    if exclude_diagonal:
        np.fill_diagonal(arr, 0.0)
    return np.sum(arr, axis=1)


def redundancy_synergy_rank_gradient(
    synergy_matrix: np.ndarray,
    redundancy_matrix: np.ndarray,
) -> np.ndarray:
    """Return the nodal rank gradient used by Luppi et al. (synergy minus redundancy)."""
    synergy_rank = rankdata(nodal_strength(synergy_matrix), method="average")
    redundancy_rank = rankdata(nodal_strength(redundancy_matrix), method="average")
    return np.asarray(synergy_rank - redundancy_rank, dtype=float)


def edge_rank_gradient(
    synergy_matrix: np.ndarray,
    redundancy_matrix: np.ndarray,
) -> np.ndarray:
    """Return the edge-wise rank gradient matrix (synergy rank minus redundancy rank)."""
    syn = np.asarray(synergy_matrix, dtype=float)
    red = np.asarray(redundancy_matrix, dtype=float)
    if syn.shape != red.shape or syn.ndim != 2 or syn.shape[0] != syn.shape[1]:
        raise ValueError("synergy_matrix and redundancy_matrix must be square and have the same shape.")

    n = syn.shape[0]
    tri = np.triu_indices(n, k=1)
    syn_rank = rankdata(syn[tri], method="average")
    red_rank = rankdata(red[tri], method="average")
    grad = np.zeros_like(syn, dtype=float)
    grad[tri] = syn_rank - red_rank
    grad[(tri[1], tri[0])] = grad[tri]
    return grad


def weighted_global_efficiency(matrix: np.ndarray) -> float:
    """Compute weighted global efficiency using inverse positive weights as distances."""
    weights = np.asarray(matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {weights.shape}.")

    n = weights.shape[0]
    if n < 2:
        return 0.0

    dist = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    mask = np.isfinite(weights) & (weights > 0)
    dist[mask] = 1.0 / weights[mask]

    # Floyd-Warshall is fine here because our target size is AAL90.
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dist
    np.fill_diagonal(inv, 0.0)
    inv[~np.isfinite(inv)] = 0.0
    return float(np.sum(inv) / (n * (n - 1)))


def weighted_modularity(matrix: np.ndarray) -> float:
    """Estimate weighted modularity with a best-effort backend.

    If ``bct`` is available, we use it. Otherwise we fall back to a
    NetworkX-based greedy community estimate. The fallback is useful for planning
    and exploratory work, but it is not numerically identical to the exact
    Brain Connectivity Toolbox path used in the paper.
    """
    weights = np.asarray(matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {weights.shape}.")

    try:
        import bct  # type: ignore

        ci, q = bct.community_louvain(weights)
        _ = ci
        return float(q)
    except Exception:
        pass

    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities, modularity
    except Exception as exc:
        raise RuntimeError(
            "weighted_modularity requires either 'bct' or 'networkx' to be installed."
        ) from exc

    graph = nx.from_numpy_array(np.where(np.isfinite(weights) & (weights > 0), weights, 0.0))
    communities = list(greedy_modularity_communities(graph, weight="weight"))
    if not communities:
        return 0.0
    return float(modularity(graph, communities, weight="weight"))


def summarize_within_between(
    matrix: np.ndarray,
    labels: Sequence[str],
) -> dict[str, float]:
    """Summarize within- vs between-label mean connectivity."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {arr.shape}.")
    if len(labels) != arr.shape[0]:
        raise ValueError("labels length must match matrix size.")

    groups = np.asarray([str(x) for x in labels], dtype=object)
    same = groups[:, None] == groups[None, :]
    offdiag = ~np.eye(arr.shape[0], dtype=bool)

    within_vals = arr[same & offdiag]
    between_vals = arr[(~same) & offdiag]
    return {
        "within_mean": float(np.nanmean(within_vals)),
        "between_mean": float(np.nanmean(between_vals)),
        "within_minus_between": float(np.nanmean(within_vals) - np.nanmean(between_vals)),
    }


def build_annotation_template(roi_labels: Sequence[str]) -> pd.DataFrame:
    """Build a fillable AAL90 annotation template for network/cytoarchitectonic labels."""
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(roi_labels, start=1):
        label_text = str(label)
        hemi = "L" if label_text.endswith("_L") else "R" if label_text.endswith("_R") else ""
        rows.append(
            {
                "roi_index": idx,
                "roi_label": label_text,
                "hemisphere": hemi,
                "aal_base_label": label_text.rsplit("_", 1)[0] if "_" in label_text else label_text,
                "rsn_7": "",
                "rsn_source": "",
                "cyto_class": "",
                "cyto_source": "",
                "include_in_cortical_only": int(not label_text.startswith(("Thalamus", "Caudate", "Putamen", "Pallidum", "Amygdala", "Hippocampus"))),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def save_annotation_template(
    roi_labels: Sequence[str],
    path: str | Path,
) -> Path:
    """Save a fillable annotation template CSV and return its path."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    build_annotation_template(roi_labels).to_csv(out, index=False)
    return out


__all__ = [
    "build_annotation_template",
    "compute_fc_matrix",
    "edge_rank_gradient",
    "matrix_spearman_similarity",
    "nodal_strength",
    "redundancy_synergy_rank_gradient",
    "save_annotation_template",
    "summarize_within_between",
    "threshold_top_density",
    "upper_triangle_values",
    "weighted_global_efficiency",
    "weighted_modularity",
]
