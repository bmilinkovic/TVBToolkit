#!/usr/bin/env python3
"""Render the coarse AAL90 macro-system grouping used for gradient statistics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import nibabel as nib
from nilearn import datasets, plotting, surface
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from brain_states_new_doc_bold_audited import build_roi_order_reference, resolve_roi_order_names  # noqa: E402


SYSTEM_ORDER = [
    "Sensorimotor",
    "Frontal association",
    "Orbitofrontal",
    "Insula",
    "Limbic",
    "Parietal",
    "Temporal",
    "Occipital-ventral",
    "Subcortical",
]

SYSTEM_COLORS = {
    "Sensorimotor": "#D1495B",
    "Frontal association": "#2E4057",
    "Orbitofrontal": "#F4A259",
    "Insula": "#8E6C8A",
    "Limbic": "#5B8E7D",
    "Parietal": "#4C78A8",
    "Temporal": "#B279A2",
    "Occipital-ventral": "#59A14F",
    "Subcortical": "#9C755F",
}


def _canon_label(text: str) -> str:
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


def _load_fc_labels(data_root: Path) -> list[str]:
    roi_ref = build_roi_order_reference(data_root)
    labels, _ = resolve_roi_order_names(roi_ref, mode="aal90_fc")
    return list(labels)


def _atlas_label_lookup() -> tuple[nib.Nifti1Image, dict[str, int]]:
    atlas = datasets.fetch_atlas_aal(verbose=0)
    atlas_img = nib.load(atlas.maps)
    lookup: dict[str, int] = {}
    for idx, label in zip(atlas.indices, atlas.labels, strict=True):
        lookup[_canon_label(label)] = int(idx)
    return atlas_img, lookup


def _macro_volume(
    mapping: pd.DataFrame,
    roi_labels_fc: list[str],
    atlas_img: nib.Nifti1Image,
    atlas_lookup: dict[str, int],
) -> nib.Nifti1Image:
    data = np.zeros(atlas_img.shape, dtype=np.float32)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    by_label = mapping.set_index("roi_label")
    system_to_idx = {name: i + 1 for i, name in enumerate(SYSTEM_ORDER)}
    for label in roi_labels_fc:
        row = by_label.loc[label]
        atlas_idx = atlas_lookup.get(_canon_label(label))
        if atlas_idx is None:
            raise KeyError(f"Could not map ROI label '{label}' into the AAL atlas.")
        data[atlas_data == atlas_idx] = float(system_to_idx[row["macro_system"]])
    return nib.Nifti1Image(data, affine=atlas_img.affine, header=atlas_img.header)


def _project(volume_img: nib.Nifti1Image, fsavg: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    left = surface.vol_to_surf(volume_img, fsavg["pial_left"], interpolation="nearest_most_frequent")
    right = surface.vol_to_surf(volume_img, fsavg["pial_right"], interpolation="nearest_most_frequent")
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def render_macro_system_surface(
    *,
    data_root: Path,
    mapping_csv: Path,
    output_dir: Path,
) -> Path:
    mapping = pd.read_csv(mapping_csv)
    roi_labels_fc = _load_fc_labels(data_root)
    atlas_img, atlas_lookup = _atlas_label_lookup()
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    vol = _macro_volume(mapping, roi_labels_fc, atlas_img, atlas_lookup)
    left_data, right_data = _project(vol, fsavg)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    cmap = ListedColormap([SYSTEM_COLORS[name] for name in SYSTEM_ORDER])
    fig = plt.figure(figsize=(16.8, 9.2))
    outer = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.95],
        left=0.03,
        right=0.985,
        top=0.91,
        bottom=0.07,
        hspace=0.00,
        wspace=0.03,
    )

    ax_l_lat = fig.add_subplot(outer[0, 0], projection="3d")
    ax_r_lat = fig.add_subplot(outer[0, 1], projection="3d")
    ax_l_med = fig.add_subplot(outer[1, 0], projection="3d")
    ax_r_med = fig.add_subplot(outer[1, 1], projection="3d")
    plotting.plot_surf_roi(
        fsavg["infl_left"],
        left_data,
        hemi="left",
        view="lateral",
        bg_map=fsavg["sulc_left"],
        cmap=cmap,
        colorbar=False,
        axes=ax_l_lat,
        figure=fig,
        darkness=None,
    )
    plotting.plot_surf_roi(
        fsavg["infl_right"],
        right_data,
        hemi="right",
        view="lateral",
        bg_map=fsavg["sulc_right"],
        cmap=cmap,
        colorbar=False,
        axes=ax_r_lat,
        figure=fig,
        darkness=None,
    )
    plotting.plot_surf_roi(
        fsavg["infl_left"],
        left_data,
        hemi="left",
        view="medial",
        bg_map=fsavg["sulc_left"],
        cmap=cmap,
        colorbar=False,
        axes=ax_l_med,
        figure=fig,
        darkness=None,
    )
    plotting.plot_surf_roi(
        fsavg["infl_right"],
        right_data,
        hemi="right",
        view="medial",
        bg_map=fsavg["sulc_right"],
        cmap=cmap,
        colorbar=False,
        axes=ax_r_med,
        figure=fig,
        darkness=None,
    )

    for ax, dx, dy in [
        (ax_l_lat, 0.005, -0.008),
        (ax_r_lat, -0.015, -0.008),
        (ax_l_med, 0.005, 0.000),
        (ax_r_med, -0.015, 0.000),
    ]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width * 1.06, pos.height * 1.06])

    fig.text(0.32, 0.86, "Lateral", ha="center", va="bottom", fontsize=16, color="#1F2430")
    fig.text(0.32, 0.445, "Medial", ha="center", va="bottom", fontsize=16, color="#1F2430")

    legend_ax = fig.add_subplot(outer[:, 2])
    legend_ax.axis("off")
    handles = [Patch(facecolor=SYSTEM_COLORS[name], edgecolor="none", label=name) for name in SYSTEM_ORDER]
    legend = legend_ax.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=14,
        labelspacing=1.2,
        handlelength=1.3,
        handleheight=1.1,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(legend)
    legend_ax.text(
        0.0,
        0.03,
        "Subcortical parcels are included in the mapping\nbut are not visible on the cortical surface render.",
        transform=legend_ax.transAxes,
        fontsize=10.5,
        color="#4D5560",
        va="bottom",
        ha="left",
    )

    fig.suptitle("Coarse AAL90 Macro-System Mapping Used for Gradient Statistics", fontsize=24, y=0.965)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "aal90_macro_system_surface_atlas"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_name(stem.name + "_transparent.png"), dpi=320, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default="data/doc_patients_new_data")
    p.add_argument(
        "--mapping-csv",
        type=str,
        default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/gradient_stats/tables/aal90_macro_system_mapping.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/gradient_stats/figures",
    )
    args = p.parse_args()
    render_macro_system_surface(
        data_root=Path(args.data_root).expanduser().resolve(),
        mapping_csv=Path(args.mapping_csv).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
