#!/usr/bin/env python3
"""Render Luppi-style macro-system gradient violin plots with significance marks."""

from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import ListedColormap
import nibabel as nib
from nilearn import datasets, plotting
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from brain_states_new_doc_bold_audited import build_roi_order_reference, resolve_roi_order_names  # noqa: E402


COHORT_ORDER = ["control", "emcs", "mcs", "uws", "coma"]
COHORT_DISPLAY = {
    "control": "CNTL",
    "emcs": "EMCS",
    "mcs": "MCS",
    "uws": "UWS",
    "coma": "COMA",
}
METHOD_ORDER = ["mmi", "ccs"]
METHOD_DISPLAY = {"mmi": "MMI", "ccs": "CCS"}
MACRO_ORDER = [
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
MACRO_DISPLAY = {
    "Sensorimotor": "SM",
    "Frontal association": "F-A",
    "Orbitofrontal": "OFC",
    "Insula": "INS",
    "Limbic": "LIM",
    "Parietal": "PAR",
    "Temporal": "TMP",
    "Occipital-ventral": "OCC",
    "Subcortical": "SUB",
}
MACRO_COLORS = {
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


def _atlas_label_lookup() -> tuple[nib.Nifti1Image, dict[str, int], dict[int, np.ndarray]]:
    atlas = datasets.fetch_atlas_aal(verbose=0)
    atlas_img = nib.load(atlas.maps)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    lookup: dict[str, int] = {}
    coord_lookup: dict[int, np.ndarray] = {}
    for idx, label in zip(atlas.indices, atlas.labels, strict=True):
        idx_i = int(idx)
        lookup[_canon_label(label)] = idx_i
        ijk = np.argwhere(atlas_data == idx_i)
        if ijk.size:
            coord_lookup[idx_i] = nib.affines.apply_affine(atlas_img.affine, ijk.mean(axis=0))
    return atlas_img, lookup, coord_lookup


def _macro_system_icon_arrays(data_root: Path, mapping_csv: Path) -> dict[str, np.ndarray]:
    mapping = pd.read_csv(mapping_csv)
    roi_labels_fc = _load_fc_labels(data_root)
    _, atlas_lookup, coord_lookup = _atlas_label_lookup()
    by_label = mapping.set_index("roi_label")
    out: dict[str, np.ndarray] = {}

    for macro in MACRO_ORDER:
        coords: list[np.ndarray] = []
        for label in roi_labels_fc:
            if by_label.loc[label, "macro_system"] != macro:
                continue
            atlas_idx = atlas_lookup.get(_canon_label(label))
            if atlas_idx is not None and atlas_idx in coord_lookup:
                coords.append(coord_lookup[atlas_idx])

        fig = plt.figure(figsize=(1.35, 0.88))
        ax_l = fig.add_subplot(111)
        disp_l = plotting.plot_glass_brain(
            None,
            display_mode="l",
            axes=ax_l,
            figure=fig,
            black_bg=False,
            plot_abs=False,
            colorbar=False,
            alpha=0.22,
        )
        if coords:
            coord_arr = np.vstack(coords)
            disp_l.add_markers(coord_arr, marker_color=MACRO_COLORS[macro], marker_size=32)
        ax_l.set_position([0.00, 0.03, 1.00, 0.94])
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=260, transparent=True, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        buf.seek(0)
        out[macro] = plt.imread(buf)
    return out


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.0,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)


def _star(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def render_violin_grid(
    *,
    subject_macro_csv: Path,
    within_tests_csv: Path,
    output_dir: Path,
    data_root: Path,
    mapping_csv: Path,
) -> Path:
    subject_df = pd.read_csv(subject_macro_csv)
    stats_df = pd.read_csv(within_tests_csv)
    _set_style()
    icon_arrays = _macro_system_icon_arrays(data_root, mapping_csv)

    all_vals = subject_df["macro_gradient_mean"].to_numpy(dtype=float)
    y_abs = float(np.nanmax(np.abs(all_vals)))
    y_min = -1.55 * y_abs
    y_max = 1.20 * y_abs

    fig, axes = plt.subplots(2, 5, figsize=(24.2, 10.2), sharey=True)
    rng = np.random.default_rng(7)

    for r, method in enumerate(METHOD_ORDER):
        for c, cohort in enumerate(COHORT_ORDER):
            ax = axes[r, c]
            sub = subject_df.loc[(subject_df["method"] == method) & (subject_df["cohort"] == cohort)].copy()
            if sub.empty:
                ax.axis("off")
                continue

            data = []
            positions = np.arange(1, len(MACRO_ORDER) + 1)
            for macro in MACRO_ORDER:
                vals = sub.loc[sub["macro_system"] == macro, "macro_gradient_mean"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                data.append(vals)

            for x, macro in zip(positions, MACRO_ORDER, strict=True):
                row = stats_df.loc[
                    (stats_df["method"] == method)
                    & (stats_df["cohort"] == cohort)
                    & (stats_df["macro_system"] == macro)
                ]
                if row.empty or not bool(row["significant_fdr"].iloc[0]):
                    continue
                mean_grad = float(row["mean_gradient"].iloc[0])
                shade = "#F46D6B" if mean_grad > 0 else "#5CA3FF"
                ax.axvspan(x - 0.47, x + 0.47, color=shade, alpha=0.08, zorder=0)

            viol = ax.violinplot(
                data,
                positions=positions,
                widths=0.84,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, macro in zip(viol["bodies"], MACRO_ORDER, strict=True):
                body.set_facecolor(MACRO_COLORS[macro])
                body.set_edgecolor(MACRO_COLORS[macro])
                body.set_alpha(0.28)
                body.set_linewidth(1.3)

            for x, macro, vals in zip(positions, MACRO_ORDER, data, strict=True):
                if vals.size == 0:
                    continue
                jitter = rng.normal(0.0, 0.07, size=vals.size)
                ax.scatter(
                    np.full(vals.size, x) + jitter,
                    vals,
                    s=34,
                    color=MACRO_COLORS[macro],
                    alpha=0.42,
                    linewidths=0,
                    zorder=3,
                )
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                mean = float(np.mean(vals))
                ax.vlines(x, q1, q3, color=MACRO_COLORS[macro], lw=2.2, zorder=4)
                ax.hlines(med, x - 0.17, x + 0.17, color=MACRO_COLORS[macro], lw=2.4, zorder=4)
                ax.scatter([x], [mean], s=62, facecolor="white", edgecolor=MACRO_COLORS[macro], linewidth=1.5, zorder=5)

                row = stats_df.loc[
                    (stats_df["method"] == method)
                    & (stats_df["cohort"] == cohort)
                    & (stats_df["macro_system"] == macro)
                ]
                if not row.empty:
                    stars = _star(float(row["perm_q_fdr"].iloc[0]))
                    if stars:
                        y = min(y_max - 0.06 * (y_max - y_min), np.nanmax(vals) + 0.07 * (y_max - y_min))
                        ax.text(x, y, stars, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1F2430")

            ax.axhline(0.0, color="#98A2B3", lw=1.1, ls="--", zorder=1)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(positions)
            ax.set_xticklabels([MACRO_DISPLAY[m] for m in MACRO_ORDER], rotation=33, ha="right")
            ax.tick_params(axis="x", pad=26, labelsize=12)
            ax.tick_params(axis="y", labelsize=11)
            for x, macro in zip(positions, MACRO_ORDER, strict=True):
                ab = AnnotationBbox(
                    OffsetImage(icon_arrays[macro], zoom=0.14),
                    (x, 0.083),
                    xycoords=("data", "axes fraction"),
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                    annotation_clip=False,
                    zorder=2,
                )
                ax.add_artist(ab)
            if c == 0:
                ax.set_ylabel("Syn-Red Rank", fontsize=15, labelpad=10)
                ax.text(-0.31, 0.50, METHOD_DISPLAY[method], transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=18)
            if r == 0:
                ax.set_title(COHORT_DISPLAY[cohort], pad=12)

    handles = [Patch(facecolor=MACRO_COLORS[m], edgecolor="none", label=f"{MACRO_DISPLAY[m]} = {m}") for m in MACRO_ORDER]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.01),
        columnspacing=1.4,
        handlelength=1.2,
    )
    arrow_width = 0.016
    head_height = 0.022
    x_arrow = 1.018
    for r in range(2):
        row_axes = axes[r, :]
        top = max(ax.get_position().y1 for ax in row_axes) - 0.004
        bottom = min(ax.get_position().y0 for ax in row_axes) + 0.060
        shaft_y0 = bottom + head_height
        shaft_y1 = top - head_height
        arrow_ax = fig.add_axes([x_arrow - arrow_width / 2.0, bottom, arrow_width, top - bottom])
        arrow_ax.add_patch(
            Rectangle(
                (0.24, shaft_y0),
                0.52,
                shaft_y1 - shaft_y0,
                facecolor="#B0B7C3",
                edgecolor="none",
                transform=arrow_ax.transData,
            )
        )
        arrow_ax.add_patch(
            Polygon(
                [[0.5, top], [0.18, shaft_y1], [0.82, shaft_y1]],
                closed=True,
                facecolor="#8C94A3",
                edgecolor="none",
                transform=arrow_ax.transData,
            )
        )
        arrow_ax.add_patch(
            Polygon(
                [[0.5, bottom], [0.18, shaft_y0], [0.82, shaft_y0]],
                closed=True,
                facecolor="#8C94A3",
                edgecolor="none",
                transform=arrow_ax.transData,
            )
        )
        arrow_ax.set_xlim(0.0, 1.0)
        arrow_ax.set_ylim(bottom, top)
        arrow_ax.axis("off")
        fig.text(x_arrow, top + 0.010, "Syn", color="#B00020", fontsize=15, fontweight="bold", va="bottom", ha="center")
        fig.text(x_arrow, bottom - 0.012, "Red", color="#1446A0", fontsize=15, fontweight="bold", va="top", ha="center")

    suptitle = fig.suptitle(
        "Macro-Regional Synergy or Redundancy Participation Gradient across DoC conditions",
        fontsize=22,
        y=0.98,
    )
    fig.subplots_adjust(left=0.085, right=0.992, top=0.86, bottom=0.17, wspace=0.14, hspace=0.34)

    stem = "macro_system_gradient_violin_grid_mmi_ccs"
    _save(fig, output_dir, stem)

    suptitle.set_visible(False)
    stem_no_title = "macro_system_gradient_violin_grid_mmi_ccs_no_title"
    _save(fig, output_dir, stem_no_title)
    plt.close(fig)
    return output_dir / f"{stem}.png"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subject-macro-csv",
        type=str,
        default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/gradient_stats/tables/subject_macro_gradient_means.csv",
    )
    p.add_argument(
        "--within-tests-csv",
        type=str,
        default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/gradient_stats/tables/within_system_one_sample_permutation_tests.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/gradient_stats/figures",
    )
    p.add_argument("--data-root", type=str, default="data/doc_patients_new_data")
    p.add_argument(
        "--mapping-csv",
        type=str,
        default="results/phiid_empirical_bold/downstream_luppi2022/mmi_ccs_comparison/gradient_stats/tables/aal90_macro_system_mapping.csv",
    )
    args = p.parse_args()
    render_violin_grid(
        subject_macro_csv=Path(args.subject_macro_csv).expanduser().resolve(),
        within_tests_csv=Path(args.within_tests_csv).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        data_root=Path(args.data_root).expanduser().resolve(),
        mapping_csv=Path(args.mapping_csv).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
