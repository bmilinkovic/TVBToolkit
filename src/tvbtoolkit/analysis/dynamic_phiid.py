"""Dynamic local-PhiID helpers for STS/RTR state analysis."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import IncrementalPCA
    from sklearn.metrics import silhouette_score
except Exception:  # pragma: no cover
    MiniBatchKMeans = None
    IncrementalPCA = None
    silhouette_score = None


_LOCAL_STS_RTR_RE = re.compile(
    r"^(?P<subject>.+)__local_sts_rtr_(?P<redundancy>[A-Za-z0-9_]+)$"
)


def parse_local_phiid_name(path: str | Path) -> dict[str, str] | None:
    """Parse a local dynamic PhiID output filename."""
    stem = Path(path).stem
    match = _LOCAL_STS_RTR_RE.match(stem)
    if match is None:
        return None
    return match.groupdict()


def load_local_phiid_subject(path: str | Path) -> dict[str, Any]:
    """Load one subject's local STS/RTR edge time series."""
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    parsed = parse_local_phiid_name(path)
    if parsed is None:
        raise ValueError(f"Unrecognized local PhiID filename: {path}")

    sts_edges = np.asarray(mat["sts_edges"], dtype=np.float32)
    rtr_edges = np.asarray(mat["rtr_edges"], dtype=np.float32)
    edge_i = np.asarray(mat["edge_i"], dtype=np.int32).reshape(-1) - 1
    edge_j = np.asarray(mat["edge_j"], dtype=np.int32).reshape(-1) - 1
    skipped_edges = np.asarray(mat.get("skipped_edges", np.zeros(edge_i.shape[0])), dtype=bool).reshape(-1)
    subject_meta = mat.get("subject_meta")

    return {
        "subject_stub": parsed["subject"],
        "redundancy": parsed["redundancy"],
        "sts_edges": sts_edges,
        "rtr_edges": rtr_edges,
        "edge_i": edge_i,
        "edge_j": edge_j,
        "skipped_edges": skipped_edges,
        "subject_meta": subject_meta,
        "n_timepoints_local": int(sts_edges.shape[0]),
        "n_edges": int(sts_edges.shape[1]),
    }


def load_local_phiid_index(
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Index local STS/RTR subject files and merge manifest metadata when available."""
    out = Path(output_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(out.glob("*.mat")):
        parsed = parse_local_phiid_name(path)
        if parsed is None:
            continue
        row = {
            "path": str(path),
            "filename": path.name,
            "subject_stub": parsed["subject"],
            "redundancy": parsed["redundancy"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    manifest: pd.DataFrame | None = None
    if manifest_path is not None:
        manifest = pd.read_csv(Path(manifest_path).expanduser().resolve())
    if manifest is not None and not manifest.empty and "subject_stub" in manifest.columns:
        df = df.merge(manifest, how="left", on="subject_stub")
    return df.sort_values(["cohort", "stage", "sedation", "subject_stub"]).reset_index(drop=True)


def edge_vector_to_matrix(
    edge_values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    *,
    n_regions: int = 90,
    fill_diagonal: float = 0.0,
) -> np.ndarray:
    """Reconstruct a symmetric matrix from an upper-triangle edge vector."""
    vec = np.asarray(edge_values, dtype=float).reshape(-1)
    mat = np.full((n_regions, n_regions), fill_diagonal, dtype=float)
    mat[edge_i, edge_j] = vec
    mat[edge_j, edge_i] = vec
    np.fill_diagonal(mat, fill_diagonal)
    return mat


def _smooth_time_axis(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if window <= 1:
        return arr
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    pad = int(window) // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(arr)
    for col in range(arr.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")[: arr.shape[0]]
    return out


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mu = np.mean(arr, axis=0, keepdims=True)
    sigma = np.std(arr, axis=0, keepdims=True)
    sigma[sigma <= 0.0] = 1.0
    return (arr - mu) / sigma


def build_subject_dynamic_features(
    sts_edges: np.ndarray,
    rtr_edges: np.ndarray,
    *,
    mode: str = "balance_and_magnitude",
    smooth_window: int = 1,
    zscore_within_subject: bool = True,
) -> np.ndarray:
    """Build a time-by-feature matrix for dynamic STS/RTR state analysis."""
    sts = np.asarray(sts_edges, dtype=np.float32)
    rtr = np.asarray(rtr_edges, dtype=np.float32)
    if sts.shape != rtr.shape:
        raise ValueError(f"STS and RTR edge arrays must match, got {sts.shape} vs {rtr.shape}.")

    sts = _smooth_time_axis(sts, window=smooth_window)
    rtr = _smooth_time_axis(rtr, window=smooth_window)

    if zscore_within_subject:
        sts = _zscore_columns(sts)
        rtr = _zscore_columns(rtr)

    if mode == "sts":
        return sts
    if mode == "rtr":
        return rtr
    if mode == "concat":
        return np.hstack([sts, rtr]).astype(np.float32, copy=False)

    balance = sts - rtr
    if mode == "balance":
        return balance.astype(np.float32, copy=False)
    if mode == "balance_and_magnitude":
        magnitude = sts + rtr
        return np.hstack([balance, magnitude]).astype(np.float32, copy=False)

    raise ValueError("mode must be one of: 'sts', 'rtr', 'concat', 'balance', 'balance_and_magnitude'.")


def iter_subject_feature_blocks(
    index_df: pd.DataFrame,
    *,
    mode: str = "balance_and_magnitude",
    smooth_window: int = 1,
    zscore_within_subject: bool = True,
) -> Generator[tuple[pd.Series, np.ndarray, dict[str, Any]], None, None]:
    """Yield one feature block per subject from indexed local PhiID outputs."""
    for row in index_df.to_dict(orient="records"):
        row_s = pd.Series(row)
        loaded = load_local_phiid_subject(row_s["path"])
        features = build_subject_dynamic_features(
            loaded["sts_edges"],
            loaded["rtr_edges"],
            mode=mode,
            smooth_window=smooth_window,
            zscore_within_subject=zscore_within_subject,
        )
        yield row_s, features, loaded


def fit_incremental_pca(
    index_df: pd.DataFrame,
    *,
    mode: str = "balance_and_magnitude",
    n_components: int = 32,
    smooth_window: int = 1,
    zscore_within_subject: bool = True,
) -> Any:
    """Fit an IncrementalPCA model over pooled dynamic subject blocks."""
    if IncrementalPCA is None:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for IncrementalPCA.")
    ipca = IncrementalPCA(n_components=int(n_components))
    for _, features, _ in iter_subject_feature_blocks(
        index_df,
        mode=mode,
        smooth_window=smooth_window,
        zscore_within_subject=zscore_within_subject,
    ):
        ipca.partial_fit(features)
    return ipca


def pool_reduced_dynamic_features(
    index_df: pd.DataFrame,
    *,
    pca_model: Any,
    mode: str = "balance_and_magnitude",
    smooth_window: int = 1,
    zscore_within_subject: bool = True,
    max_timepoints_per_subject: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Transform and pool reduced dynamic features across subjects."""
    reduced_blocks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    edge_meta: dict[str, Any] | None = None

    for row_s, features, loaded in iter_subject_feature_blocks(
        index_df,
        mode=mode,
        smooth_window=smooth_window,
        zscore_within_subject=zscore_within_subject,
    ):
        if max_timepoints_per_subject is not None and features.shape[0] > int(max_timepoints_per_subject):
            idx = np.linspace(0, features.shape[0] - 1, int(max_timepoints_per_subject)).round().astype(int)
            features_use = features[idx]
            time_idx = idx
        else:
            features_use = features
            time_idx = np.arange(features.shape[0], dtype=int)

        reduced = np.asarray(pca_model.transform(features_use), dtype=np.float32)
        reduced_blocks.append(reduced)
        for local_row, t in enumerate(time_idx):
            rows.append(
                {
                    "subject_stub": row_s["subject_stub"],
                    "cohort": row_s.get("cohort", ""),
                    "stage": row_s.get("stage", ""),
                    "sedation": row_s.get("sedation", ""),
                    "time_index_local": int(t),
                    "reduced_row_index": int(local_row),
                }
            )
        if edge_meta is None:
            edge_meta = {
                "edge_i": loaded["edge_i"],
                "edge_j": loaded["edge_j"],
                "n_edges": int(loaded["n_edges"]),
                "n_timepoints_local": int(loaded["n_timepoints_local"]),
                "mode": mode,
            }

    if edge_meta is None:
        raise ValueError("No local PhiID subject blocks were available.")
    return np.vstack(reduced_blocks), pd.DataFrame(rows), edge_meta


def score_kmeans_range(
    reduced_features: np.ndarray,
    *,
    k_values: Sequence[int],
    random_state: int = 0,
    batch_size: int = 2048,
    silhouette_sample: int = 5000,
) -> pd.DataFrame:
    """Evaluate a range of k values with MiniBatchKMeans."""
    if MiniBatchKMeans is None:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for MiniBatchKMeans.")
    rows: list[dict[str, Any]] = []
    x = np.asarray(reduced_features, dtype=np.float32)
    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=int(random_state),
            batch_size=int(batch_size),
            n_init="auto",
        )
        labels = model.fit_predict(x)
        sil = np.nan
        if silhouette_score is not None and x.shape[0] > int(k):
            if x.shape[0] > silhouette_sample:
                idx = np.linspace(0, x.shape[0] - 1, silhouette_sample).round().astype(int)
                sil = float(silhouette_score(x[idx], labels[idx]))
            else:
                sil = float(silhouette_score(x, labels))
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": sil,
            }
        )
    return pd.DataFrame(rows)


def fit_dynamic_state_model(
    reduced_features: np.ndarray,
    *,
    k: int,
    random_state: int = 0,
    batch_size: int = 2048,
) -> tuple[Any, np.ndarray]:
    """Fit a MiniBatchKMeans model on reduced dynamic features."""
    if MiniBatchKMeans is None:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for MiniBatchKMeans.")
    x = np.asarray(reduced_features, dtype=np.float32)
    model = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(random_state),
        batch_size=int(batch_size),
        n_init="auto",
    )
    labels = np.asarray(model.fit_predict(x), dtype=int)
    return model, labels


def split_reconstructed_features(
    reconstructed: np.ndarray,
    *,
    mode: str,
    n_edges: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Split reconstructed feature vectors into STS/RTR or balance/magnitude blocks."""
    x = np.asarray(reconstructed, dtype=np.float32)
    if mode == "sts":
        return x, None, None, None
    if mode == "rtr":
        return None, x, None, None
    if mode == "concat":
        return x[:, :n_edges], x[:, n_edges:], None, None
    if mode == "balance":
        return None, None, x, None
    if mode == "balance_and_magnitude":
        balance = x[:, :n_edges]
        magnitude = x[:, n_edges:]
        sts = 0.5 * (magnitude + balance)
        rtr = 0.5 * (magnitude - balance)
        return sts, rtr, balance, magnitude
    raise ValueError(f"Unsupported mode: {mode}")


def reconstruct_state_centroids(
    pca_model: Any,
    cluster_model: Any,
    *,
    mode: str,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    n_regions: int = 90,
) -> dict[str, Any]:
    """Reconstruct state centroids back to edge and matrix space."""
    reduced_centers = np.asarray(cluster_model.cluster_centers_, dtype=np.float32)
    full = np.asarray(pca_model.inverse_transform(reduced_centers), dtype=np.float32)
    n_edges = int(edge_i.shape[0])
    sts_edges, rtr_edges, balance_edges, magnitude_edges = split_reconstructed_features(
        full,
        mode=mode,
        n_edges=n_edges,
    )

    result: dict[str, Any] = {
        "reduced_centers": reduced_centers,
        "full_centers": full,
        "edge_i": edge_i,
        "edge_j": edge_j,
    }
    if sts_edges is not None:
        result["sts_edges"] = sts_edges
        result["sts_matrices"] = np.stack(
            [edge_vector_to_matrix(v, edge_i, edge_j, n_regions=n_regions) for v in sts_edges],
            axis=0,
        )
    if rtr_edges is not None:
        result["rtr_edges"] = rtr_edges
        result["rtr_matrices"] = np.stack(
            [edge_vector_to_matrix(v, edge_i, edge_j, n_regions=n_regions) for v in rtr_edges],
            axis=0,
        )
    if balance_edges is not None:
        result["balance_edges"] = balance_edges
        result["balance_matrices"] = np.stack(
            [edge_vector_to_matrix(v, edge_i, edge_j, n_regions=n_regions) for v in balance_edges],
            axis=0,
        )
    if magnitude_edges is not None:
        result["magnitude_edges"] = magnitude_edges
        result["magnitude_matrices"] = np.stack(
            [edge_vector_to_matrix(v, edge_i, edge_j, n_regions=n_regions) for v in magnitude_edges],
            axis=0,
        )
    return result


def connectivity_vectors_from_records(
    records: Sequence[Any],
    *,
    subject_stub_field: str = "subject_id",
) -> pd.DataFrame:
    """Build subject-level FC/SC upper-triangle vectors from subject records."""
    rows: list[dict[str, Any]] = []
    tri = np.triu_indices(90, k=1)
    for rec in records:
        ts = np.asarray(rec.timeseries, dtype=float)
        if ts.shape[1] != 90 and ts.shape[0] == 90:
            ts = ts.T
        fc = np.corrcoef(ts, rowvar=False)
        sc = np.asarray(rec.sc_matrix, dtype=float)
        rows.append(
            {
                "subject_stub": str(getattr(rec, subject_stub_field)),
                "fc_edges": fc[tri].astype(np.float32),
                "sc_edges": sc[tri].astype(np.float32),
            }
        )
    return pd.DataFrame(rows)


def centroid_connectivity_similarity(
    centroid_edges: np.ndarray,
    connectome_edges: np.ndarray,
) -> float:
    """Compare one centroid edge vector to a connectome edge vector with Spearman rho."""
    x = np.asarray(centroid_edges, dtype=float).reshape(-1)
    y = np.asarray(connectome_edges, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def compare_centroids_to_connectomes(
    centroid_bundle: dict[str, Any],
    connectome_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare reconstructed STS/RTR centroids to subject FC and SC edge vectors."""
    rows: list[dict[str, Any]] = []
    for _, subj in connectome_df.iterrows():
        for family in ("sts_edges", "rtr_edges"):
            if family not in centroid_bundle:
                continue
            centroids = np.asarray(centroid_bundle[family], dtype=float)
            for state_idx, centroid in enumerate(centroids, start=1):
                rows.append(
                    {
                        "subject_stub": str(subj["subject_stub"]),
                        "family": family.replace("_edges", ""),
                        "state": int(state_idx),
                        "rho_fc": centroid_connectivity_similarity(centroid, subj["fc_edges"]),
                        "rho_sc": centroid_connectivity_similarity(centroid, subj["sc_edges"]),
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "build_subject_dynamic_features",
    "centroid_connectivity_similarity",
    "compare_centroids_to_connectomes",
    "connectivity_vectors_from_records",
    "edge_vector_to_matrix",
    "fit_dynamic_state_model",
    "fit_incremental_pca",
    "iter_subject_feature_blocks",
    "load_local_phiid_index",
    "load_local_phiid_subject",
    "parse_local_phiid_name",
    "pool_reduced_dynamic_features",
    "reconstruct_state_centroids",
    "score_kmeans_range",
    "split_reconstructed_features",
]
