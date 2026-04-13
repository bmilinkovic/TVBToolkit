#!/usr/bin/env python3
"""Standalone FC/SC alignment verification for new DoC raw .mat files.

Checks performed:
1) Subject alignment per FC/SC file pair (count + optional subject-name arrays).
2) Region-count agreement (AAL90 expected).
3) Exact interleaved->symmetric reorder mapping from lookup files.
4) Final FC/SC ROI-order equality after requested reorder mode.

Outputs:
- alignment_report.json
- subject_pair_counts.csv
- roi_reorder_mapping_interleaved_to_symmetric.csv
- roi_order_final_fc.csv
- roi_order_final_sc.csv
- roi_order_coupling_qc.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required to load MATLAB files") from exc

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


FILE_SPECS = [
    {
        "cohort": "control",
        "fc_path": "CNT_send/FC/DoC_CNT.mat",
        "fc_var": "DoC_CNT",
        "sc_path": "CNT_send/SC/CNT_SC.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "emcs",
        "fc_path": "FC_send/DoC_acute_EMCS_matched.mat",
        "fc_var": "DoC_acute_EMCS",
        "sc_path": "SC_send/norm/anon/DoC_acute_EMCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "emcs",
        "fc_path": "FC_send/DoC_chronic_EMCS_matched.mat",
        "fc_var": "DoC_chronic_EMCS",
        "sc_path": "SC_send/norm/anon/DoC_EMCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "emcs",
        "fc_path": "FC_send/DoC_chronic_sedated_EMCS_matched.mat",
        "fc_var": "DoC_chronic_sedated_EMCS",
        "sc_path": "SC_send/norm/anon/DoC_EMCS_sedated_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "fc_path": "FC_send/DoC_acute_MCS_matched.mat",
        "fc_var": "DoC_acute_MCS",
        "sc_path": "SC_send/norm/anon/DoC_acute_MCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "fc_path": "FC_send/DoC_acute_sedated_MCS_matched.mat",
        "fc_var": "DoC_acute_sedated_MCS",
        "sc_path": "SC_send/norm/anon/DoC_acute_sedated_MCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "fc_path": "FC_send/DoC_chronic_MCS_matched.mat",
        "fc_var": "DoC_chronic_MCS",
        "sc_path": "SC_send/norm/anon/DoC_MCS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "mcs",
        "fc_path": "FC_send/DoC_chronic_sedated_MCS_matched.mat",
        "fc_var": "DoC_chronic_sedated_MCS",
        "sc_path": "SC_send/norm/anon/DoC_MCS_sedated_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "fc_path": "FC_send/DoC_acute_UWS_matched.mat",
        "fc_var": "DoC_acute_UWS",
        "sc_path": "SC_send/norm/anon/DoC_acute_UWS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "fc_path": "FC_send/DoC_acute_sedated_UWS_matched.mat",
        "fc_var": "DoC_acute_sedated_UWS",
        "sc_path": "SC_send/norm/anon/DoC_acute_sedated_UWS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "fc_path": "FC_send/DoC_chronic_UWS_matched.mat",
        "fc_var": "DoC_chronic_UWS",
        "sc_path": "SC_send/norm/anon/DoC_UWS_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "uws",
        "fc_path": "FC_send/DoC_chronic_sedated_UWS_matched.mat",
        "fc_var": "DoC_chronic_sedated_UWS",
        "sc_path": "SC_send/norm/anon/DoC_UWS_sedated_SC_matched.mat",
        "sc_var": "SC",
    },
]

OPTIONAL_COMA_SPECS = [
    {
        "cohort": "coma",
        "fc_path": "FC_send/DoC_acute_coma_matched.mat",
        "fc_var": "DoC_acute_coma",
        "sc_path": "SC_send/norm/anon/DoC_acute_COMA_SC_matched.mat",
        "sc_var": "SC",
    },
    {
        "cohort": "coma",
        "fc_path": "FC_send/DoC_acute_sedated_COMA_matched.mat",
        "fc_var": "DoC_acute_sedated_COMA",
        "sc_path": "SC_send/norm/anon/DoC_acute_sedated_COMA_SC_matched.mat",
        "sc_var": "SC",
    },
]


def _load_mat_mapping(path: Path) -> dict[str, np.ndarray]:
    try:
        raw = sio.loadmat(path)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except Exception:
        if h5py is None:
            raise
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                out[key] = obj[()]
    return out


def _select_3d_numeric(mapping: dict[str, np.ndarray], variable_hint: str | None = None) -> tuple[str, np.ndarray]:
    if variable_hint and variable_hint in mapping:
        arr = np.asarray(mapping[variable_hint])
        if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
            return variable_hint, np.asarray(arr, dtype=float)
    candidates: list[tuple[str, np.ndarray]] = []
    for key, val in mapping.items():
        arr = np.asarray(val)
        if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
            candidates.append((key, np.asarray(arr, dtype=float)))
    if not candidates:
        raise RuntimeError(f"No 3D numeric arrays found. Keys: {list(mapping.keys())}")
    candidates.sort(key=lambda kv: (0 if kv[0].lower().startswith("doc") else 1, kv[0]))
    return candidates[0]


def _to_subject_roi_time(arr3d: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr3d, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D FC array, got {arr.shape}")
    shape = arr.shape
    roi_axes = [i for i, s in enumerate(shape) if s == 90]
    if not roi_axes:
        raise ValueError(f"Cannot infer ROI axis in FC shape {shape}")
    roi_axis = roi_axes[0]
    time_axes = [i for i, s in enumerate(shape) if s in (295, 297) and i != roi_axis]
    if time_axes:
        time_axis = time_axes[0]
    else:
        other = [i for i in range(3) if i != roi_axis]
        time_axis = max(other, key=lambda i: shape[i])
    subject_axis = [i for i in range(3) if i not in (roi_axis, time_axis)][0]
    return np.moveaxis(arr, (subject_axis, roi_axis, time_axis), (0, 1, 2))


def _to_subject_roi_roi(arr3d: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr3d, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D SC array, got {arr.shape}")
    shape = arr.shape
    non_90_axes = [i for i, s in enumerate(shape) if s != 90]
    subject_axis = non_90_axes[0] if len(non_90_axes) == 1 else 2
    out = np.moveaxis(arr, subject_axis, 0)
    if out.shape[1:] != (90, 90):
        raise ValueError(f"SC reshape failed: {shape} -> {out.shape}")
    return out


def _decode_subject_names(value: np.ndarray | None) -> list[str]:
    if value is None:
        return []
    arr = np.asarray(value)
    names: list[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            txt = item.decode("utf-8", errors="ignore").strip()
        elif isinstance(item, str):
            txt = item.strip()
        elif isinstance(item, np.ndarray) and item.dtype.kind in {"U", "S"}:
            flat = item.reshape(-1)
            txt = "".join(
                [
                    ch.decode("utf-8", errors="ignore") if isinstance(ch, bytes) else str(ch)
                    for ch in flat
                    if str(ch).strip() not in {"", "0"}
                ]
            ).strip()
        else:
            txt = str(item).strip()
        if txt and txt != "[]":
            names.append(txt)
    return names


def _compute_static_fc(timeseries: np.ndarray) -> np.ndarray:
    x = np.asarray(timeseries, dtype=float)
    fc = np.corrcoef(x, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    fc = 0.5 * (fc + fc.T)
    np.fill_diagonal(fc, 1.0)
    return fc


def _upper_triangle_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    iu = np.triu_indices(arr.shape[0], k=1)
    return np.asarray(arr[iu], dtype=float)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=float).reshape(-1)
    xb = np.asarray(b, dtype=float).reshape(-1)
    if xa.size != xb.size or xa.size == 0:
        return float("nan")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(xb)):
        return float("nan")
    sa = float(np.std(xa))
    sb = float(np.std(xb))
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def _aal90_symmetry_reorder_index() -> np.ndarray:
    left = np.arange(0, 90, 2)
    right = np.arange(1, 90, 2)[::-1]
    idx = np.concatenate([left, right])
    if idx.size != 90:
        raise RuntimeError("Invalid AAL90 reorder index")
    return idx.astype(int)


def _parse_roi_mni_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(parts[1].strip())
        else:
            ws = line.split()
            if len(ws) >= 2:
                names.append(ws[1].strip())
    if len(names) != 90:
        raise RuntimeError(f"Expected 90 names in {path}, found {len(names)}")
    return names


def _parse_symmetric_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(parts[1].strip())
        else:
            ws = line.split()
            if len(ws) >= 2:
                names.append(ws[1].strip())
    if len(names) != 90:
        raise RuntimeError(f"Expected 90 names in {path}, found {len(names)}")
    return names


def _resolve_order_names(interleaved: list[str], symmetric: list[str], idx: np.ndarray, mode: str) -> tuple[list[str], list[str]]:
    if mode not in {"none", "aal90_fc", "aal90_sc", "aal90_both"}:
        raise ValueError(f"Unsupported mode: {mode}")
    fc = [interleaved[i] for i in idx] if mode in {"aal90_fc", "aal90_both"} else list(interleaved)
    sc = [symmetric[i] for i in idx] if mode in {"aal90_sc", "aal90_both"} else list(symmetric)
    return fc, sc


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = list(FILE_SPECS)
    for spec in OPTIONAL_COMA_SPECS:
        if (data_root / spec["fc_path"]).exists() and (data_root / spec["sc_path"]).exists():
            specs.append(spec)

    inter_path = data_root / "ROI_MNI_V4_90.txt"
    sym_path = data_root / "symmetric_lookuptable_clean.txt"
    inter_names = _parse_roi_mni_names(inter_path)
    sym_names = _parse_symmetric_names(sym_path)
    idx = _aal90_symmetry_reorder_index()

    # Verify mapping from names matches hard-coded permutation.
    inter_name_to_idx = {n: i for i, n in enumerate(inter_names)}
    idx_from_names = np.asarray([inter_name_to_idx[n] for n in sym_names], dtype=int)
    if not np.array_equal(idx_from_names, idx):
        raise RuntimeError("Lookup-name-derived mapping does not match hard-coded AAL90 symmetric permutation.")

    rows: list[dict[str, Any]] = []
    reorder_qc_rows: list[dict[str, Any]] = []

    for spec in specs:
        fc_path = data_root / str(spec["fc_path"])
        sc_path = data_root / str(spec["sc_path"])
        if not fc_path.exists() or not sc_path.exists():
            continue

        fc_map = _load_mat_mapping(fc_path)
        sc_map = _load_mat_mapping(sc_path)
        _, fc_arr = _select_3d_numeric(fc_map, variable_hint=str(spec.get("fc_var", "")))
        _, sc_arr = _select_3d_numeric(sc_map, variable_hint=str(spec.get("sc_var", "")))

        fc_srt = _to_subject_roi_time(fc_arr)  # (S, 90, T)
        sc_srr = _to_subject_roi_roi(sc_arr)   # (S, 90, 90)

        n_fc = int(fc_srt.shape[0])
        n_sc = int(sc_srr.shape[0])
        if n_fc != n_sc:
            raise RuntimeError(f"Subject count mismatch: {fc_path.name} ({n_fc}) vs {sc_path.name} ({n_sc})")

        fc_names = _decode_subject_names(fc_map.get("subj_names"))
        sc_names = _decode_subject_names(sc_map.get("subj_names"))
        if fc_names and len(fc_names) != n_fc:
            fc_names = []
        if sc_names and len(sc_names) != n_sc:
            sc_names = []

        for i in range(n_fc):
            if args.max_subjects_per_group is not None and i >= int(args.max_subjects_per_group):
                break
            ts = np.asarray(fc_srt[i].T, dtype=float)  # (T,90)
            sc = np.asarray(sc_srr[i], dtype=float)
            if ts.ndim != 2 or ts.shape[1] != 90:
                raise RuntimeError(f"Unexpected FC shape at {fc_path.name}[{i}]: {ts.shape}")
            if sc.shape != (90, 90):
                raise RuntimeError(f"Unexpected SC shape at {sc_path.name}[{i}]: {sc.shape}")

            if fc_names and sc_names and str(fc_names[i]) != str(sc_names[i]):
                raise RuntimeError(
                    "Subject name mismatch at same index: "
                    f"fc={fc_names[i]} sc={sc_names[i]} file_pair=({fc_path.name},{sc_path.name}) index={i}"
                )

            rows.append(
                {
                    "cohort": str(spec["cohort"]),
                    "source_fc_file": str(spec["fc_path"]),
                    "source_sc_file": str(spec["sc_path"]),
                    "source_subject_index": int(i),
                    "n_timepoints": int(ts.shape[0]),
                    "n_regions": int(ts.shape[1]),
                    "has_fc_subject_names": bool(fc_names),
                    "has_sc_subject_names": bool(sc_names),
                    "fc_subject_name": fc_names[i] if fc_names else "",
                    "sc_subject_name": sc_names[i] if sc_names else "",
                }
            )

            fc_id = _compute_static_fc(ts)
            sc_id = sc
            fc_vec_id = _upper_triangle_vector(fc_id)
            sc_vec_id = _upper_triangle_vector(sc_id)
            r_identity = _safe_pearson(fc_vec_id, sc_vec_id)
            fc_re = fc_id[np.ix_(idx, idx)]
            sc_re = sc_id[np.ix_(idx, idx)]
            r_reorder_fc = _safe_pearson(_upper_triangle_vector(fc_re), sc_vec_id)
            r_reorder_sc = _safe_pearson(fc_vec_id, _upper_triangle_vector(sc_re))
            r_reorder_both = _safe_pearson(_upper_triangle_vector(fc_re), _upper_triangle_vector(sc_re))
            reorder_qc_rows.append(
                {
                    "cohort": str(spec["cohort"]),
                    "source_fc_file": str(spec["fc_path"]),
                    "source_sc_file": str(spec["sc_path"]),
                    "source_subject_index": int(i),
                    "coupling_identity": float(r_identity),
                    "coupling_reorder_fc": float(r_reorder_fc),
                    "coupling_reorder_sc": float(r_reorder_sc),
                    "coupling_reorder_both": float(r_reorder_both),
                }
            )

    load_qc_df = pd.DataFrame(rows)
    if load_qc_df.empty:
        raise RuntimeError("No FC/SC subjects loaded from provided data-root.")

    dup = int(load_qc_df.duplicated(["source_fc_file", "source_sc_file", "source_subject_index"]).sum())
    if dup > 0:
        raise RuntimeError(f"Duplicate source pair rows detected: {dup}")

    bad_regions = load_qc_df[load_qc_df["n_regions"].astype(int) != 90]
    if not bad_regions.empty:
        raise RuntimeError("Region-count mismatch found (n_regions != 90).")

    pairs_df = (
        load_qc_df.groupby(["source_fc_file", "source_sc_file"], as_index=False)["source_subject_index"]
        .count()
        .rename(columns={"source_subject_index": "n_subjects"})
        .sort_values(["source_fc_file", "source_sc_file"])
    )

    reorder_qc_df = pd.DataFrame(reorder_qc_rows)
    if args.mode == "auto":
        mean_delta_fc = float((reorder_qc_df["coupling_reorder_fc"] - reorder_qc_df["coupling_identity"]).mean())
        mean_delta_sc = float((reorder_qc_df["coupling_reorder_sc"] - reorder_qc_df["coupling_identity"]).mean())
        mode_use = "aal90_fc" if mean_delta_fc >= mean_delta_sc else "aal90_sc"
    else:
        mode_use = args.mode

    fc_final, sc_final = _resolve_order_names(inter_names, sym_names, idx, mode_use)
    mismatch = [
        {"position_1based": i + 1, "fc_roi": fc_final[i], "sc_roi": sc_final[i]}
        for i in range(90)
        if fc_final[i] != sc_final[i]
    ]

    load_qc_df.to_csv(out_dir / "subject_loading_qc.csv", index=False)
    pairs_df.to_csv(out_dir / "subject_pair_counts.csv", index=False)
    reorder_qc_df.to_csv(out_dir / "roi_order_coupling_qc.csv", index=False)
    pd.DataFrame(
        {
            "symmetric_position_1based": np.arange(1, 91, dtype=int),
            "interleaved_index_0based": idx,
            "interleaved_position_1based": idx + 1,
            "interleaved_roi": [inter_names[i] for i in idx],
            "symmetric_roi": sym_names,
        }
    ).to_csv(out_dir / "roi_reorder_mapping_interleaved_to_symmetric.csv", index=False)
    pd.DataFrame({"position_1based": np.arange(1, 91, dtype=int), "roi_name": fc_final}).to_csv(
        out_dir / "roi_order_final_fc.csv", index=False
    )
    pd.DataFrame({"position_1based": np.arange(1, 91, dtype=int), "roi_name": sc_final}).to_csv(
        out_dir / "roi_order_final_sc.csv", index=False
    )

    report = {
        "data_root": str(data_root),
        "functional_order_source": str(inter_path),
        "structural_order_source": str(sym_path),
        "functional_input_order": "aal90_interleaved_lr",
        "structural_input_order": "aal90_symmetric_left_then_right_reverse",
        "mode_requested": args.mode,
        "mode_applied": mode_use,
        "n_subjects_total": int(load_qc_df.shape[0]),
        "n_unique_file_pairs": int(pairs_df.shape[0]),
        "subject_pair_counts": pairs_df.to_dict(orient="records"),
        "coupling_means": {
            "identity": float(reorder_qc_df["coupling_identity"].mean()),
            "reorder_fc": float(reorder_qc_df["coupling_reorder_fc"].mean()),
            "reorder_sc": float(reorder_qc_df["coupling_reorder_sc"].mean()),
            "reorder_both": float(reorder_qc_df["coupling_reorder_both"].mean()),
        },
        "final_roi_order_equal": len(mismatch) == 0,
        "n_final_roi_mismatches": len(mismatch),
        "final_roi_mismatch_examples": mismatch[:10],
        "final_order_first10": fc_final[:10],
        "final_order_last10": fc_final[-10:],
    }
    (out_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))

    print(
        "[verify] summary:",
        {
            "mode_requested": args.mode,
            "mode_applied": mode_use,
            "n_subjects_total": int(load_qc_df.shape[0]),
            "n_pairs": int(pairs_df.shape[0]),
            "final_roi_order_equal": bool(report["final_roi_order_equal"]),
            "n_final_roi_mismatches": int(report["n_final_roi_mismatches"]),
        },
    )
    print("[verify] outputs:", out_dir)

    if mismatch:
        first = mismatch[0]
        raise SystemExit(
            "FC/SC final ROI-order mismatch. "
            f"First mismatch at position {first['position_1based']}: fc={first['fc_roi']} sc={first['sc_roi']}."
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default="data/doc_patients_new_data")
    p.add_argument(
        "--mode",
        type=str,
        default="aal90_fc",
        choices=["auto", "none", "aal90_fc", "aal90_sc", "aal90_both"],
        help="For this dataset the expected mode is 'aal90_fc'.",
    )
    p.add_argument("--max-subjects-per-group", type=int, default=None)
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/doc_patients_new_bold_brain_states_audited/alignment_verification",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
