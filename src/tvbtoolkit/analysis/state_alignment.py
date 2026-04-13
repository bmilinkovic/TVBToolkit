"""Cross-subject brain-state alignment utilities.

This module aligns subject-specific state centroids to shared template states,
so occupancy can be compared across subjects without assuming local state-index
identity (e.g., local state 1 in subject A equals local state 1 in subject B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation with finite/variance safeguards."""
    xa = np.asarray(a, dtype=float).reshape(-1)
    xb = np.asarray(b, dtype=float).reshape(-1)
    if xa.size != xb.size or xa.size == 0:
        return float("nan")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(xb)):
        return float("nan")
    if float(np.std(xa)) <= 0.0 or float(np.std(xb)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def centroid_similarity_matrix(centers_a: np.ndarray, centers_b: np.ndarray) -> np.ndarray:
    """Pairwise centroid similarity matrix (Pearson r).

    Parameters
    ----------
    centers_a : np.ndarray
        Matrix of shape ``(k_a, features)``.
    centers_b : np.ndarray
        Matrix of shape ``(k_b, features)``.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape ``(k_a, k_b)``.
    """
    a = np.asarray(centers_a, dtype=float)
    b = np.asarray(centers_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both centroid inputs must be 2D arrays.")
    if a.shape[1] != b.shape[1]:
        raise ValueError("Centroid feature dimensions must match.")

    sim = np.empty((a.shape[0], b.shape[0]), dtype=float)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            sim[i, j] = safe_pearson(a[i], b[j])
    return np.nan_to_num(sim, nan=-1.0, posinf=-1.0, neginf=-1.0)


@dataclass(frozen=True)
class AlignmentResult:
    """State-alignment result for one subject/seed."""

    assignment_local_to_template: np.ndarray
    similarity_matrix: np.ndarray
    matched_similarity: np.ndarray
    occupancy_aligned: np.ndarray


def align_states_to_templates(
    local_centers: np.ndarray,
    local_occupancy: np.ndarray,
    template_centers: np.ndarray,
) -> AlignmentResult:
    """Align local states to shared templates via Hungarian assignment.

    Parameters
    ----------
    local_centers : np.ndarray
        Local state centroids ``(k_local, features)``.
    local_occupancy : np.ndarray
        Local occupancy vector ``(k_local,)``.
    template_centers : np.ndarray
        Template centroids ``(k_template, features)``.

    Returns
    -------
    AlignmentResult
        Alignment outputs including occupancy mapped to template indices.
    """
    centers = np.asarray(local_centers, dtype=float)
    occ = np.asarray(local_occupancy, dtype=float).reshape(-1)
    templates = np.asarray(template_centers, dtype=float)

    if centers.ndim != 2 or templates.ndim != 2:
        raise ValueError("Centroids must be 2D arrays.")
    if centers.shape[0] != occ.size:
        raise ValueError("local_occupancy length must match number of local states.")

    sim = centroid_similarity_matrix(centers, templates)
    rows, cols = linear_sum_assignment(-sim)

    assignment = np.full(centers.shape[0], -1, dtype=int)
    assignment[rows] = cols
    matched = sim[rows, cols]

    occ_aligned = np.zeros(templates.shape[0], dtype=float)
    occ_aligned[cols] = occ[rows]

    return AlignmentResult(
        assignment_local_to_template=assignment,
        similarity_matrix=sim,
        matched_similarity=np.asarray(matched, dtype=float),
        occupancy_aligned=occ_aligned,
    )


def fit_state_templates(
    all_centers: np.ndarray,
    n_states: int,
    *,
    random_seed: int = 0,
    n_init: int = 32,
    max_iter: int = 200,
) -> np.ndarray:
    """Fit shared template states from pooled local centroids.

    Parameters
    ----------
    all_centers : np.ndarray
        Pooled centroid matrix ``(n_samples, features)``.
    n_states : int
        Number of template states.
    random_seed : int, default=0
        RNG seed.
    n_init : int, default=32
        Number of random restarts for k-means.
    max_iter : int, default=200
        Max iterations per restart.

    Returns
    -------
    np.ndarray
        Template centroids of shape ``(n_states, features)``.
    """
    x = np.asarray(all_centers, dtype=float)
    if x.ndim != 2:
        raise ValueError("all_centers must be a 2D array.")
    if x.shape[0] < int(n_states):
        raise ValueError("Need at least n_states rows to fit templates.")

    rng = np.random.default_rng(random_seed)
    best_centers = None
    best_inertia = np.inf

    for _ in range(max(1, int(n_init))):
        seed = int(rng.integers(0, 2**31 - 1))
        centers, labels = kmeans2(x, k=int(n_states), minit="points", iter=int(max_iter), seed=seed)
        inertia = float(np.sum((x - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = np.asarray(centers, dtype=float)

    return np.asarray(best_centers, dtype=float)
