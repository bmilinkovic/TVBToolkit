#!/usr/bin/env python3
"""Run Luppi-style gradient statistics adapted to the AAL90 DoC PhiID outputs.

This script implements three complementary analyses:

1. Within-cohort one-sample permutation tests of macro-system mean gradients
   against zero, analogous to Luppi et al.'s RSN/class tests.
2. Between-cohort two-sample permutation tests of subject-level macro-system
   mean gradients.
3. Cortical map-to-map correlations between the cohort-average nodal gradient
   and cohort-average nodal FC / SC strength, assessed with spin permutations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import nibabel as nib
from nilearn import datasets, surface
from neuromaps import nulls
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.analysis import (  # noqa: E402
    PUBLICATION_COHORT_ORDER,
    load_phiid_index,
    load_phiid_matrix,
)
from tvbtoolkit.analysis.luppi2022 import compute_fc_matrix  # noqa: E402
from brain_states_new_doc_bold_audited import (  # noqa: E402
    _maybe_apply_roi_reordering,
    build_roi_order_reference,
    load_new_doc_subjects,
    resolve_roi_order_names,
    validate_final_roi_order_or_raise,
)


COHORT_ORDER = list(PUBLICATION_COHORT_ORDER)
COHORT_DISPLAY = {
    "control": "CNTL",
    "emcs": "EMCS",
    "mcs": "MCS",
    "uws": "UWS",
    "coma": "COMA",
}
METHOD_ORDER = ["mmi", "ccs"]
METHOD_LABELS = {"mmi": "MMI", "ccs": "CCS"}
REFMAP_ORDER = ["fc_strength", "sc_strength"]
REFMAP_LABELS = {"fc_strength": "FC strength", "sc_strength": "SC strength"}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _canon_label(text: str) -> str:
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


def _gradient_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "gradient_stats",
        ["#1446A0", "#5CA3FF", "#F7F7F7", "#F46D6B", "#B00020"],
        N=256,
    )


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    ok = np.isfinite(arr)
    if not np.any(ok):
        return out
    vals = arr[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = ranked.size
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    back = np.empty_like(q)
    back[order] = q
    out[ok] = back
    return out


def _hedges_g_one_sample(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan")
    sd = arr.std(ddof=1)
    if sd <= 0:
        return float("nan")
    d = arr.mean() / sd
    j = 1.0 - 3.0 / (4.0 * n - 5.0)
    return float(j * d)


def _hedges_g_two_sample(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if min(n1, n2) < 2:
        return float("nan")
    v1 = a.var(ddof=1)
    v2 = b.var(ddof=1)
    pooled_num = (n1 - 1) * v1 + (n2 - 1) * v2
    pooled_den = n1 + n2 - 2
    if pooled_den <= 0:
        return float("nan")
    sp = np.sqrt(pooled_num / pooled_den)
    if sp <= 0:
        return float("nan")
    d = (b.mean() - a.mean()) / sp
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(j * d)


def _perm_test_one_sample(x: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, arr.size), replace=True)
    perm = (signs * arr[None, :]).mean(axis=1)
    p = (1.0 + np.sum(np.abs(perm) >= abs(obs))) / (n_perm + 1.0)
    return obs, float(p)


def _perm_test_two_sample(x: np.ndarray, y: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if min(a.size, b.size) == 0:
        return float("nan"), float("nan")
    obs = float(b.mean() - a.mean())
    combined = np.concatenate([a, b])
    n1 = a.size
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        idx = rng.permutation(combined.size)
        perm_a = combined[idx[:n1]]
        perm_b = combined[idx[n1:]]
        perm[i] = perm_b.mean() - perm_a.mean()
    p = (1.0 + np.sum(np.abs(perm) >= abs(obs))) / (n_perm + 1.0)
    return obs, float(p)


def _macro_system_from_base_label(base: str) -> str:
    if base in {"Precentral", "Rolandic_Oper", "Supp_Motor_Area", "Postcentral", "Paracentral_Lobule"}:
        return "Sensorimotor"
    if base in {"Frontal_Sup", "Frontal_Mid", "Frontal_Inf_Oper", "Frontal_Inf_Tri", "Frontal_Sup_Medial"}:
        return "Frontal association"
    if base in {"Frontal_Sup_Orb", "Frontal_Mid_Orb", "Frontal_Inf_Orb", "Frontal_Med_Orb", "Rectus", "Olfactory"}:
        return "Orbitofrontal"
    if base == "Insula":
        return "Insula"
    if base in {"Cingulum_Ant", "Cingulum_Mid", "Cingulum_Post", "Hippocampus", "ParaHippocampal", "Amygdala"}:
        return "Limbic"
    if base in {"Calcarine", "Cuneus", "Lingual", "Occipital_Sup", "Occipital_Mid", "Occipital_Inf", "Fusiform"}:
        return "Occipital-ventral"
    if base in {"Parietal_Sup", "Parietal_Inf", "SupraMarginal", "Angular", "Precuneus"}:
        return "Parietal"
    if base in {"Heschl", "Temporal_Sup", "Temporal_Pole_Sup", "Temporal_Mid", "Temporal_Pole_Mid", "Temporal_Inf"}:
        return "Temporal"
    if base in {"Caudate", "Putamen", "Pallidum", "Thalamus"}:
        return "Subcortical"
    raise KeyError(f"Unhandled AAL base label: {base}")


def _build_macro_mapping() -> pd.DataFrame:
    ann = pd.read_csv(_REPO_ROOT / "data" / "reference" / "aal90_luppi2022_annotations_template.csv")
    ann["macro_system"] = ann["aal_base_label"].astype(str).map(_macro_system_from_base_label)
    ann["macro_system_order"] = ann["macro_system"].map(
        {
            "Sensorimotor": 0,
            "Frontal association": 1,
            "Orbitofrontal": 2,
            "Insula": 3,
            "Limbic": 4,
            "Parietal": 5,
            "Temporal": 6,
            "Occipital-ventral": 7,
            "Subcortical": 8,
        }
    )
    return ann


def _load_subject_gradient_table(
    *,
    method: str,
    phiid_root: Path,
    manifest_path: Path,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    index_df = load_phiid_index(phiid_root / method, manifest_path=manifest_path)
    grad_df = index_df.loc[index_df["atom"] == "sr_gradient"].copy()
    rows: list[dict[str, Any]] = []
    map_idx = mapping.set_index("roi_index")
    for _, row in grad_df.iterrows():
        vec = np.asarray(load_phiid_matrix(row["path"], atom="sr_gradient"), dtype=float).reshape(-1)
        if vec.size != len(mapping):
            raise ValueError(f"Expected gradient length {len(mapping)}, got {vec.size} for {row['path']}.")
        for roi_index, value in enumerate(vec, start=1):
            info = map_idx.loc[roi_index]
            rows.append(
                {
                    "method": method,
                    "subject_id": row["subject_id"],
                    "subject_stub": row["subject_stub"],
                    "cohort": row["cohort"],
                    "stage": row["stage"],
                    "sedation": row["sedation"],
                    "roi_index": roi_index,
                    "roi_label": info["roi_label"],
                    "macro_system": info["macro_system"],
                    "include_in_cortical_only": int(info["include_in_cortical_only"]),
                    "gradient_value": float(value),
                }
            )
    return pd.DataFrame(rows)


def _summarize_subject_macro_means(subject_gradients: pd.DataFrame) -> pd.DataFrame:
    return (
        subject_gradients.groupby(
            ["method", "subject_id", "subject_stub", "cohort", "stage", "sedation", "macro_system"],
            as_index=False,
        )["gradient_value"]
        .mean()
        .rename(columns={"gradient_value": "macro_gradient_mean"})
    )


def _one_sample_macro_tests(df: pd.DataFrame, *, n_perm: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, ((method, cohort, macro), sub) in enumerate(
        df.groupby(["method", "cohort", "macro_system"], sort=False)
    ):
        vals = sub["macro_gradient_mean"].to_numpy(dtype=float)
        mean_val, p = _perm_test_one_sample(vals, n_perm=n_perm, seed=seed + i)
        rows.append(
            {
                "method": method,
                "cohort": cohort,
                "macro_system": macro,
                "n_subjects": int(np.isfinite(vals).sum()),
                "mean_gradient": mean_val,
                "median_gradient": float(np.nanmedian(vals)),
                "perm_p": p,
                "hedges_g": _hedges_g_one_sample(vals),
            }
        )
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = np.nan
    for (method, cohort), idx in out.groupby(["method", "cohort"]).groups.items():
        out.loc[list(idx), "perm_q_fdr"] = _bh_fdr(out.loc[list(idx), "perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out


def _between_cohort_macro_tests(df: pd.DataFrame, *, n_perm: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pair_order = list(itertools.combinations(COHORT_ORDER, 2))
    for method in METHOD_ORDER:
        sub_method = df.loc[df["method"] == method].copy()
        for macro in sorted(sub_method["macro_system"].unique()):
            sub_macro = sub_method.loc[sub_method["macro_system"] == macro]
            for j, (c0, c1) in enumerate(pair_order):
                a = sub_macro.loc[sub_macro["cohort"] == c0, "macro_gradient_mean"].to_numpy(dtype=float)
                b = sub_macro.loc[sub_macro["cohort"] == c1, "macro_gradient_mean"].to_numpy(dtype=float)
                diff, p = _perm_test_two_sample(a, b, n_perm=n_perm, seed=seed + 1000 * METHOD_ORDER.index(method) + 100 * j)
                rows.append(
                    {
                        "method": method,
                        "macro_system": macro,
                        "cohort_a": c0,
                        "cohort_b": c1,
                        "contrast": f"{COHORT_DISPLAY[c1]} - {COHORT_DISPLAY[c0]}",
                        "n_a": int(np.isfinite(a).sum()),
                        "n_b": int(np.isfinite(b).sum()),
                        "mean_a": float(np.nanmean(a)),
                        "mean_b": float(np.nanmean(b)),
                        "mean_diff_b_minus_a": diff,
                        "perm_p": p,
                        "hedges_g": _hedges_g_two_sample(a, b),
                    }
                )
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        out.loc[list(idx), "perm_q_fdr"] = _bh_fdr(out.loc[list(idx), "perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out


def _atlas_label_lookup() -> tuple[nib.Nifti1Image, dict[str, int]]:
    atlas = datasets.fetch_atlas_aal(verbose=0)
    atlas_img = nib.load(atlas.maps)
    lookup: dict[str, int] = {}
    for idx, label in zip(atlas.indices, atlas.labels, strict=True):
        lookup[_canon_label(label)] = int(idx)
    return atlas_img, lookup


def _load_fc_labels(data_root: Path) -> list[str]:
    roi_ref = build_roi_order_reference(data_root)
    labels, _ = resolve_roi_order_names(roi_ref, mode="aal90_fc")
    return list(labels)


def _project_roi_mask_to_surface(
    atlas_img: nib.Nifti1Image,
    atlas_lookup: dict[str, int],
    label: str,
    hemi: str,
    fsavg: dict[str, str],
) -> np.ndarray:
    atlas_idx = atlas_lookup[_canon_label(label)]
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    mask = np.zeros(atlas_img.shape, dtype=np.float32)
    mask[atlas_data == atlas_idx] = 1.0
    mask_img = nib.Nifti1Image(mask, affine=atlas_img.affine, header=atlas_img.header)
    surf = surface.vol_to_surf(
        mask_img,
        fsavg[f"pial_{hemi}"],
        interpolation="nearest_most_frequent",
    )
    return np.asarray(surf, dtype=float) > 0.5


def _build_cortical_spin_geometry(data_root: Path, mapping: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    roi_labels_fc = _load_fc_labels(data_root)
    atlas_img, atlas_lookup = _atlas_label_lookup()
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    sphere_left = surface.load_surf_mesh(fsavg["sphere_left"]).coordinates
    sphere_right = surface.load_surf_mesh(fsavg["sphere_right"]).coordinates

    rows: list[dict[str, Any]] = []
    coords: list[np.ndarray] = []
    hemiid: list[int] = []
    mapping_by_label = mapping.set_index("roi_label")
    for roi_index, label in enumerate(roi_labels_fc, start=1):
        info = mapping_by_label.loc[label]
        if int(info["include_in_cortical_only"]) != 1:
            continue
        hemi = "left" if str(label).endswith("_L") else "right"
        mask = _project_roi_mask_to_surface(atlas_img, atlas_lookup, label, hemi, fsavg)
        if hemi == "left":
            surf_coords = sphere_left[mask]
            hemi_code = 0
        else:
            surf_coords = sphere_right[mask]
            hemi_code = 1
        if surf_coords.size == 0:
            raise RuntimeError(f"No surface vertices found for cortical ROI {label}.")
        centroid = surf_coords.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        coords.append(centroid)
        hemiid.append(hemi_code)
        rows.append(
            {
                "roi_index": roi_index,
                "roi_label": label,
                "macro_system": info["macro_system"],
                "hemisphere": hemi.upper(),
            }
        )
    return pd.DataFrame(rows), np.asarray(coords, dtype=float), np.asarray(hemiid, dtype=int)


def _load_subject_records_reordered(data_root: Path) -> list[Any]:
    records, _ = load_new_doc_subjects(data_root)
    records_use, _, _ = _maybe_apply_roi_reordering(records, mode="aal90_fc")
    ref = build_roi_order_reference(data_root)
    validate_final_roi_order_or_raise(ref, "aal90_fc")
    return records_use


def _cohort_reference_maps(records: list[Any], cortical_labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        sub = [rec for rec in records if str(rec.cohort) == cohort]
        mats_fc = []
        for rec in sub:
            ts = np.asarray(rec.timeseries, dtype=float)
            finite_rows = np.isfinite(ts).all(axis=1)
            ts = ts[finite_rows]
            fc_sub = np.asarray(compute_fc_matrix(ts), dtype=float)
            fc_sub[~np.isfinite(fc_sub)] = 0.0
            np.fill_diagonal(fc_sub, 0.0)
            mats_fc.append(fc_sub)
        mats_sc = [np.asarray(rec.sc_matrix, dtype=float) for rec in sub]
        fc = np.mean(np.stack(mats_fc, axis=0), axis=0)
        sc = np.mean(np.stack(mats_sc, axis=0), axis=0)
        fc_strength = np.sum(np.abs(fc), axis=1) - np.abs(np.diag(fc))
        sc_strength = np.sum(sc, axis=1) - np.diag(sc)
        rows.append(
            {
                "cohort": cohort,
                "fc_strength": fc_strength,
                "sc_strength": sc_strength,
            }
        )
    return pd.DataFrame(rows)


def _cohort_gradient_vectors(results_root: Path, cortical_meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roi_order = cortical_meta["roi_index"].to_numpy(dtype=int)
    for method in METHOD_ORDER:
        df = pd.read_csv(results_root / method / "tables" / "cohort_nodal_gradients.csv")
        for cohort in COHORT_ORDER:
            sub = df.loc[df["cohort"] == cohort].sort_values("roi_index")
            full_vec = sub["gradient_value"].to_numpy(dtype=float)
            vec = full_vec[roi_order - 1]
            rows.append({"method": method, "cohort": cohort, "gradient": vec})
    return pd.DataFrame(rows)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _spin_correlation_tests(
    *,
    gradient_vectors: pd.DataFrame,
    reference_maps: pd.DataFrame,
    cortical_meta: pd.DataFrame,
    spins: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ref_map = reference_maps.set_index("cohort")
    for method in METHOD_ORDER:
        for cohort in COHORT_ORDER:
            grad = np.asarray(
                gradient_vectors.loc[
                    (gradient_vectors["method"] == method) & (gradient_vectors["cohort"] == cohort),
                    "gradient",
                ].iloc[0],
                dtype=float,
            )
            for ref_name in REFMAP_ORDER:
                ref_full = np.asarray(ref_map.loc[cohort, ref_name], dtype=float)
                ref_vec = ref_full[cortical_meta["roi_index"].to_numpy(dtype=int) - 1]
                rho = _safe_spearman(grad, ref_vec)
                perm_rho = np.empty(spins.shape[1], dtype=float)
                for i in range(spins.shape[1]):
                    perm_rho[i] = _safe_spearman(grad[spins[:, i]], ref_vec)
                finite = np.isfinite(perm_rho)
                if not np.isfinite(rho) or not np.any(finite):
                    p_spin = float("nan")
                else:
                    p_spin = (1.0 + np.sum(np.abs(perm_rho[finite]) >= abs(rho))) / (int(finite.sum()) + 1.0)
                rows.append(
                    {
                        "method": method,
                        "cohort": cohort,
                        "reference_map": ref_name,
                        "n_cortical_rois": int(grad.size),
                        "spearman_rho": rho,
                        "p_spin": float(p_spin),
                    }
                )
    out = pd.DataFrame(rows)
    out["q_spin_fdr"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        out.loc[list(idx), "q_spin_fdr"] = _bh_fdr(out.loc[list(idx), "p_spin"].to_numpy(dtype=float))
    out["significant_fdr"] = out["q_spin_fdr"] < 0.05
    return out


def _plot_within_system_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    macro_order = (
        df[["macro_system"]]
        .drop_duplicates()
        .sort_values("macro_system")
        ["macro_system"]
        .tolist()
    )
    arrs = []
    sigs = []
    for method in METHOD_ORDER:
        sub = df.loc[df["method"] == method]
        mat = np.full((len(macro_order), len(COHORT_ORDER)), np.nan)
        sig = np.zeros_like(mat, dtype=bool)
        for i, macro in enumerate(macro_order):
            for j, cohort in enumerate(COHORT_ORDER):
                row = sub.loc[(sub["macro_system"] == macro) & (sub["cohort"] == cohort)]
                if row.empty:
                    continue
                mat[i, j] = float(row["mean_gradient"].iloc[0])
                sig[i, j] = bool(row["significant_fdr"].iloc[0])
        arrs.append(mat)
        sigs.append(sig)
    vmax = float(np.nanmax(np.abs(np.stack(arrs, axis=0))))
    cmap = _gradient_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), constrained_layout=True)
    for ax, method, mat, sig in zip(axes, METHOD_ORDER, arrs, sigs, strict=True):
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(METHOD_LABELS[method], fontsize=14)
        ax.set_xticks(range(len(COHORT_ORDER)))
        ax.set_xticklabels([COHORT_DISPLAY[c] for c in COHORT_ORDER], rotation=0)
        ax.set_yticks(range(len(macro_order)))
        ax.set_yticklabels(macro_order)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    label = f"{mat[i, j]:.1f}"
                    if sig[i, j]:
                        label += " *"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color="#1f2430")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
    cbar.set_label("Mean macro-system gradient")
    fig.suptitle("Within-cohort macro-system gradient tests vs zero", fontsize=18, y=1.02)
    _save_figure(fig, out_dir, "within_system_gradient_vs_zero_heatmap")


def _plot_between_cohort_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    macro_order = (
        df[["macro_system"]]
        .drop_duplicates()
        .sort_values("macro_system")
        ["macro_system"]
        .tolist()
    )
    contrast_order = [f"{COHORT_DISPLAY[b]}-{COHORT_DISPLAY[a]}" for a, b in itertools.combinations(COHORT_ORDER, 2)]
    arrs = []
    sigs = []
    for method in METHOD_ORDER:
        sub = df.loc[df["method"] == method].copy()
        sub["contrast_key"] = sub["contrast"].str.replace(" - ", "-", regex=False)
        mat = np.full((len(macro_order), len(contrast_order)), np.nan)
        sig = np.zeros_like(mat, dtype=bool)
        for i, macro in enumerate(macro_order):
            for j, contrast in enumerate(contrast_order):
                row = sub.loc[(sub["macro_system"] == macro) & (sub["contrast_key"] == contrast)]
                if row.empty:
                    continue
                mat[i, j] = float(row["mean_diff_b_minus_a"].iloc[0])
                sig[i, j] = bool(row["significant_fdr"].iloc[0])
        arrs.append(mat)
        sigs.append(sig)
    vmax = float(np.nanmax(np.abs(np.stack(arrs, axis=0))))
    cmap = _gradient_cmap()
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.8), constrained_layout=True)
    for ax, method, mat, sig in zip(axes, METHOD_ORDER, arrs, sigs, strict=True):
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(METHOD_LABELS[method], fontsize=14)
        ax.set_xticks(range(len(contrast_order)))
        ax.set_xticklabels(contrast_order, rotation=45, ha="right")
        ax.set_yticks(range(len(macro_order)))
        ax.set_yticklabels(macro_order)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]) and sig[i, j]:
                    ax.text(j, i, "•", ha="center", va="center", fontsize=16, color="#1f2430")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    cbar.set_label("Mean gradient difference (later cohort minus earlier cohort)")
    fig.suptitle("Between-cohort macro-system gradient permutation contrasts", fontsize=18, y=1.02)
    _save_figure(fig, out_dir, "between_cohort_gradient_contrasts_heatmap")


def _plot_spin_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    arr = np.full((len(METHOD_ORDER) * len(REFMAP_ORDER), len(COHORT_ORDER)), np.nan)
    sig = np.zeros_like(arr, dtype=bool)
    row_labels: list[str] = []
    for i, method in enumerate(METHOD_ORDER):
        for k, ref_name in enumerate(REFMAP_ORDER):
            row_labels.append(f"{METHOD_LABELS[method]} | {REFMAP_LABELS[ref_name]}")
            sub = df.loc[(df["method"] == method) & (df["reference_map"] == ref_name)]
            for j, cohort in enumerate(COHORT_ORDER):
                row = sub.loc[sub["cohort"] == cohort]
                if row.empty:
                    continue
                arr[i * len(REFMAP_ORDER) + k, j] = float(row["spearman_rho"].iloc[0])
                sig[i * len(REFMAP_ORDER) + k, j] = bool(row["significant_fdr"].iloc[0])
    vmax = float(np.nanmax(np.abs(arr)))
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    im = ax.imshow(arr, cmap=_gradient_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels([COHORT_DISPLAY[c] for c in COHORT_ORDER])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                label = f"{arr[i, j]:.2f}"
                if sig[i, j]:
                    label += " *"
                ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color="#1f2430")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman rho (spin-tested)")
    ax.set_title("Cortical nodal gradient correlation with FC / SC strength", fontsize=16)
    _save_figure(fig, out_dir, "gradient_fc_sc_spin_correlation_heatmap")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default="data/doc_patients_new_data")
    p.add_argument("--manifest", type=str, default="results/phiid_empirical_bold/inputs/manifest.csv")
    p.add_argument("--phiid-root", type=str, default="results/phiid_empirical_bold/phiid")
    p.add_argument("--results-root", type=str, default="results/phiid_empirical_bold/downstream_luppi2022")
    p.add_argument("--output-root", type=str, default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/gradient_stats")
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-spin", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    phiid_root = Path(args.phiid_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    table_root = output_root / "tables"
    fig_root = output_root / "figures"
    log_root = output_root / "logs"
    for d in [table_root, fig_root, log_root]:
        d.mkdir(parents=True, exist_ok=True)

    _set_style()
    mapping = _build_macro_mapping()
    mapping.to_csv(table_root / "aal90_macro_system_mapping.csv", index=False)

    subject_gradients = pd.concat(
        [
            _load_subject_gradient_table(method=method, phiid_root=phiid_root, manifest_path=manifest_path, mapping=mapping)
            for method in METHOD_ORDER
        ],
        ignore_index=True,
    )
    subject_gradients.to_csv(table_root / "subject_gradient_values.csv", index=False)

    subject_macro = _summarize_subject_macro_means(subject_gradients)
    subject_macro.to_csv(table_root / "subject_macro_gradient_means.csv", index=False)

    within_tests = _one_sample_macro_tests(subject_macro, n_perm=args.n_perm, seed=args.seed)
    within_tests.to_csv(table_root / "within_system_one_sample_permutation_tests.csv", index=False)

    between_tests = _between_cohort_macro_tests(subject_macro, n_perm=args.n_perm, seed=args.seed + 101)
    between_tests.to_csv(table_root / "between_cohort_two_sample_permutation_tests.csv", index=False)

    cortical_meta, coords, hemiid = _build_cortical_spin_geometry(data_root, mapping)
    cortical_meta.to_csv(table_root / "cortical_spin_geometry.csv", index=False)
    spins = nulls.spins.gen_spinsamples(
        coords,
        hemiid,
        n_rotate=args.n_spin,
        method="vasa",
        seed=args.seed,
        check_duplicates=True,
    )
    np.save(output_root / "spin_indices.npy", spins)

    records = _load_subject_records_reordered(data_root)
    ref_maps = _cohort_reference_maps(records, cortical_meta["roi_label"].tolist())
    gradient_vectors = _cohort_gradient_vectors(results_root, cortical_meta)
    spin_tests = _spin_correlation_tests(
        gradient_vectors=gradient_vectors,
        reference_maps=ref_maps,
        cortical_meta=cortical_meta,
        spins=spins,
    )
    spin_tests.to_csv(table_root / "gradient_fc_sc_spin_correlations.csv", index=False)

    _plot_within_system_heatmap(within_tests, fig_root)
    _plot_between_cohort_heatmap(between_tests, fig_root)
    _plot_spin_correlation_heatmap(spin_tests, fig_root)

    summary = {
        "data_root": str(data_root),
        "phiid_root": str(phiid_root),
        "results_root": str(results_root),
        "n_perm": int(args.n_perm),
        "n_spin": int(args.n_spin),
        "n_subjects": int(subject_macro["subject_id"].nunique()),
        "macro_systems": sorted(mapping["macro_system"].unique().tolist()),
        "n_cortical_rois_for_spin": int(cortical_meta.shape[0]),
        "outputs": {
            "tables": str(table_root),
            "figures": str(fig_root),
        },
    }
    (log_root / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
