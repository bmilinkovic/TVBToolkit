#!/usr/bin/env python3
"""Render MMI/CCS cohort nodal gradients with parcel-wise significance contours."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting, surface

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from tvbtoolkit.core.paths import doc_liege_raw, doc_liege_results  # noqa: E402
from brain_states_new_doc_bold_audited import build_roi_order_reference, resolve_roi_order_names  # noqa: E402


COHORT_ORDER = ["control", "emcs", "mcs", "uws", "coma"]
METHOD_ORDER = ["mmi", "ccs"]
METHOD_LABELS = {"mmi": "MMI", "ccs": "CCS"}


def _canon_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(text).strip()).lower()


def _gradient_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "phiid_surface_gradient",
        ["#1446A0", "#3C8DFF", "#F7F7F7", "#F46D6B", "#B00020"],
        N=256,
    )


def _load_fc_reordered_labels(data_root: Path) -> list[str]:
    roi_ref = build_roi_order_reference(data_root)
    labels, _ = resolve_roi_order_names(roi_ref, mode="aal90_fc")
    return list(labels)


def _atlas_label_lookup() -> tuple[nib.Nifti1Image, dict[str, int]]:
    atlas = datasets.fetch_atlas_aal(verbose=0)
    atlas_img = nib.load(atlas.maps)
    label_to_index: dict[str, int] = {}
    for idx, label in zip(atlas.indices, atlas.labels, strict=True):
        label_to_index[_canon_label(label)] = int(idx)
    return atlas_img, label_to_index


def _gradient_volume(
    cohort_df: pd.DataFrame,
    roi_labels_fc: list[str],
    atlas_img: nib.Nifti1Image,
    atlas_lookup: dict[str, int],
) -> nib.Nifti1Image:
    data = np.zeros(atlas_img.shape, dtype=np.float32)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    values = cohort_df.sort_values("roi_index")["gradient_value"].to_numpy(dtype=float)
    if len(values) != len(roi_labels_fc):
        raise ValueError(f"Expected {len(roi_labels_fc)} ROI values, got {len(values)}.")
    for label, value in zip(roi_labels_fc, values, strict=True):
        atlas_idx = atlas_lookup.get(_canon_label(label))
        if atlas_idx is None:
            raise KeyError(f"Could not map ROI label '{label}' into the Nilearn AAL atlas.")
        data[atlas_data == atlas_idx] = float(value)
    return nib.Nifti1Image(data, affine=atlas_img.affine, header=atlas_img.header)


def _significance_volume(
    sig_labels: list[str],
    atlas_img: nib.Nifti1Image,
    atlas_lookup: dict[str, int],
) -> nib.Nifti1Image:
    data = np.zeros(atlas_img.shape, dtype=np.int16)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    for label in sig_labels:
        atlas_idx = atlas_lookup.get(_canon_label(label))
        if atlas_idx is not None:
            data[atlas_data == atlas_idx] = 1
    return nib.Nifti1Image(data, affine=atlas_img.affine, header=atlas_img.header)


def _project_hemispheres(volume_img: nib.Nifti1Image, fsavg: dict[str, str], *, interpolation: str = "linear") -> tuple[np.ndarray, np.ndarray]:
    left = surface.vol_to_surf(volume_img, fsavg["pial_left"], interpolation=interpolation)
    right = surface.vol_to_surf(volume_img, fsavg["pial_right"], interpolation=interpolation)
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def _plot_one_cell(
    fig: plt.Figure,
    parent_spec: any,
    *,
    left_data: np.ndarray,
    right_data: np.ndarray,
    sig_left: np.ndarray | None,
    sig_right: np.ndarray | None,
    fsavg: dict[str, str],
    cmap: mpl.colors.Colormap,
    vabs: float,
) -> tuple[plt.Axes, plt.Axes]:
    sub = parent_spec.subgridspec(1, 2, wspace=0.02)
    ax_l = fig.add_subplot(sub[0, 0], projection="3d")
    ax_r = fig.add_subplot(sub[0, 1], projection="3d")
    plotting.plot_surf_stat_map(
        fsavg["infl_left"],
        left_data,
        hemi="left",
        view="lateral",
        bg_map=fsavg["sulc_left"],
        cmap=cmap,
        colorbar=False,
        symmetric_cbar=False,
        vmin=-vabs,
        vmax=vabs,
        axes=ax_l,
        figure=fig,
        darkness=None,
    )
    plotting.plot_surf_stat_map(
        fsavg["infl_right"],
        right_data,
        hemi="right",
        view="lateral",
        bg_map=fsavg["sulc_right"],
        cmap=cmap,
        colorbar=False,
        symmetric_cbar=False,
        vmin=-vabs,
        vmax=vabs,
        axes=ax_r,
        figure=fig,
        darkness=None,
    )
    if sig_left is not None and np.nanmax(sig_left) > 0:
        plotting.plot_surf_contours(
            fsavg["infl_left"],
            roi_map=(sig_left > 0.5).astype(int),
            levels=[1],
            colors=["#111111"],
            axes=ax_l,
            figure=fig,
        )
    if sig_right is not None and np.nanmax(sig_right) > 0:
        plotting.plot_surf_contours(
            fsavg["infl_right"],
            roi_map=(sig_right > 0.5).astype(int),
            levels=[1],
            colors=["#111111"],
            axes=ax_r,
            figure=fig,
        )
    pos_l = ax_l.get_position()
    pos_r = ax_r.get_position()
    expand = 1.03
    ax_l.set_position([pos_l.x0 + 0.001, pos_l.y0 - 0.001, pos_l.width * expand, pos_l.height * expand])
    ax_r.set_position([pos_r.x0 - 0.006, pos_r.y0 - 0.001, pos_r.width * expand, pos_r.height * expand])
    return ax_l, ax_r


def render_surface_panel(
    *,
    data_root: Path,
    results_root: Path,
    sig_table: Path,
    output_dir: Path,
) -> Path:
    roi_labels_fc = _load_fc_reordered_labels(data_root)
    gradient_tables = {
        method: pd.read_csv(results_root / method / "tables" / "cohort_nodal_gradients.csv")
        for method in METHOD_ORDER
    }
    sig_df = pd.read_csv(sig_table)
    sig_df = sig_df.loc[sig_df["significant_fdr"]].copy()
    atlas_img, atlas_lookup = _atlas_label_lookup()
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    cmap = _gradient_cmap()

    row_limits: dict[str, float] = {}
    projected: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    projected_sig: dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None]] = {}
    sig_counts: dict[tuple[str, str], int] = {}

    for method, df in gradient_tables.items():
        row_limits[method] = float(np.nanmax(np.abs(df["gradient_value"].to_numpy(dtype=float))))
        for cohort in COHORT_ORDER:
            cohort_df = df.loc[df["cohort"] == cohort].copy()
            vol = _gradient_volume(cohort_df, roi_labels_fc, atlas_img, atlas_lookup)
            projected[(method, cohort)] = _project_hemispheres(vol, fsavg)

            sig_labels = (
                sig_df.loc[(sig_df["method"] == method) & (sig_df["cohort"] == cohort), "roi_label"]
                .astype(str)
                .tolist()
            )
            sig_counts[(method, cohort)] = len(sig_labels)
            if sig_labels:
                sig_vol = _significance_volume(sig_labels, atlas_img, atlas_lookup)
                projected_sig[(method, cohort)] = _project_hemispheres(sig_vol, fsavg, interpolation="nearest_most_frequent")
            else:
                projected_sig[(method, cohort)] = (None, None)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 14.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(22.8, 7.6))
    outer = GridSpec(
        nrows=2,
        ncols=6,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 0.10],
        hspace=-0.18,
        wspace=0.012,
        left=0.04,
        right=0.972,
        top=0.885,
        bottom=0.09,
    )

    cbar_positions = []
    for row_idx, method in enumerate(METHOD_ORDER):
        vabs = row_limits[method]
        row_axes = []
        for col_idx, cohort in enumerate(COHORT_ORDER):
            left_data, right_data = projected[(method, cohort)]
            sig_left, sig_right = projected_sig[(method, cohort)]
            pair = _plot_one_cell(
                fig,
                outer[row_idx, col_idx],
                left_data=left_data,
                right_data=right_data,
                sig_left=sig_left,
                sig_right=sig_right,
                fsavg=fsavg,
                cmap=cmap,
                vabs=vabs,
            )
            row_axes.extend(pair)
            pos_l = pair[0].get_position()
            pos_r = pair[1].get_position()
            x_mid = 0.5 * (pos_l.x0 + pos_r.x1)
            if row_idx == 0:
                fig.text(x_mid, max(pos_l.y1, pos_r.y1) + 0.012, cohort.upper(), ha="center", va="bottom", fontsize=16)
            fig.text(x_mid, min(pos_l.y0, pos_r.y0) - 0.010, f"n={sig_counts[(method, cohort)]}", ha="center", va="top", fontsize=10, color="#111111")

        cax = fig.add_subplot(outer[row_idx, 5])
        row_y0 = min(ax.get_position().y0 for ax in row_axes)
        row_y1 = max(ax.get_position().y1 for ax in row_axes)
        cpos = cax.get_position()
        cax.set_position([cpos.x0 + 0.006, row_y0, cpos.width * 0.58, row_y1 - row_y0])
        cbar_positions.append(cax.get_position())
        sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=-vabs, vmax=vabs), cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=13, width=0.8, length=3.4)
        cbar.set_label("")
        fig.text(0.018, 0.69 if row_idx == 0 else 0.29, METHOD_LABELS[method], rotation=90, va="center", ha="center", fontsize=22, color="#1F2430")

    cbar_top = cbar_positions[0]
    cbar_bottom = cbar_positions[1]
    x_arrow = max(cbar_top.x1, cbar_bottom.x1) + 0.048
    y_top = max(cbar_top.y1, cbar_bottom.y1) - 0.020
    y_bottom = min(cbar_top.y0, cbar_bottom.y0) + 0.020
    arrow_width = 0.016
    head_height = 0.030
    shaft_y0 = y_bottom + head_height
    shaft_y1 = y_top - head_height
    arrow_ax = fig.add_axes([x_arrow - arrow_width / 2.0, y_bottom, arrow_width, y_top - y_bottom])
    gradient = np.linspace(-1.0, 1.0, 512)[:, None]
    arrow_ax.imshow(gradient, cmap=cmap, aspect="auto", origin="lower", extent=[0.2, 0.8, shaft_y0, shaft_y1])
    arrow_ax.add_patch(Polygon([[0.5, y_top], [0.18, shaft_y1], [0.82, shaft_y1]], closed=True, facecolor="#B00020", edgecolor="none", transform=arrow_ax.transData))
    arrow_ax.add_patch(Polygon([[0.5, y_bottom], [0.18, shaft_y0], [0.82, shaft_y0]], closed=True, facecolor="#1446A0", edgecolor="none", transform=arrow_ax.transData))
    arrow_ax.set_xlim(0.0, 1.0)
    arrow_ax.set_ylim(y_bottom, y_top)
    arrow_ax.axis("off")
    fig.text(x_arrow, y_top + 0.010, "Synergy", color="#B00020", fontsize=18, fontweight="bold", va="bottom", ha="center")
    fig.text(x_arrow, y_bottom - 0.010, "Redundancy", color="#1446A0", fontsize=18, fontweight="bold", va="top", ha="center")

    fig.suptitle("Regional Synergy-Redundancy Gradient of Participation across DoC conditions", fontsize=24, y=0.972)
    fig.text(0.5, 0.03, "Black outlines: cortical parcels with one-sample sign-flip q<0.05 (global FDR across all panels).", ha="center", va="center", fontsize=11, color="#222222")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "cohort_nodal_gradient_surface_mmi_ccs_2x5_lateral_stats"
    fig.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_dir / f"{stem}.png"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(doc_liege_raw("doc_data")))
    p.add_argument("--results-root", type=str, default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022")))
    p.add_argument(
        "--sig-table",
        type=str,
        default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022", "mmi_ccs_comparison", "pairwise_stats", "tables", "surface_gradient_roi_one_sample_tests.csv")),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(doc_liege_results("phiid_empirical_bold", "downstream_luppi2022", "mmi_ccs_comparison", "figures", "gradient_surface")),
    )
    args = p.parse_args()
    render_surface_panel(
        data_root=Path(args.data_root).expanduser().resolve(),
        results_root=Path(args.results_root).expanduser().resolve(),
        sig_table=Path(args.sig_table).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
