#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn.cluster import KMeans, MiniBatchKMeans

# ── Publication aesthetics ────────────────────────────────────────────────────
# Nature guidelines: sans-serif, 5–7 pt text, no top/right spines,
# embedded fonts (pdf.fonttype=42), max column widths 89 mm (single) /
# 183 mm (double).
_NATURE_RC: dict[str, Any] = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":             6,
    "axes.labelsize":        6,
    "axes.titlesize":        6.5,
    "axes.titleweight":      "bold",
    "xtick.labelsize":       5.5,
    "ytick.labelsize":       5.5,
    "xtick.major.size":      2.5,
    "ytick.major.size":      2.5,
    "xtick.minor.size":      1.5,
    "ytick.minor.size":      1.5,
    "xtick.major.width":     0.5,
    "ytick.major.width":     0.5,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "axes.linewidth":        0.5,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "lines.linewidth":       0.75,
    "patch.linewidth":       0.5,
    "legend.fontsize":       6,
    "legend.frameon":        False,
    "legend.handlelength":   1.0,
    "legend.handletextpad":  0.4,
    "legend.labelspacing":   0.3,
    "legend.borderpad":      0.3,
    "legend.columnspacing":  0.8,
    "legend.title_fontsize": 6.5,
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "pdf.fonttype":          42,   # embed fonts as TrueType (required by Nature)
    "ps.fonttype":           42,
    "savefig.dpi":           300,
    "figure.dpi":            150,
}

# Compact two-line column header labels for the b-sweep figure panels.
_SCENARIO_SHORT: dict[str, str] = {
    "private_alpha0":    "Private\n(α = 0.00)",
    "global_alpha_low":  "Global shared\n(α = 0.15)",
    "global_alpha_med":  "Global shared\n(α = 0.40)",
    "global_alpha_high": "Global shared\n(α = 0.70)",
    "sc_alpha_med":      "SC-shaped\n(α = 0.40)",
}
for _alpha_i in range(5, 51, 5):
    _alpha = _alpha_i / 100.0
    _SCENARIO_SHORT[f"global_alpha_{_alpha_i:03d}"] = f"Global\n(α = {_alpha:.2f})"
    _SCENARIO_SHORT[f"sc_alpha_{_alpha_i:03d}"] = f"SC-shaped\n(α = {_alpha:.2f})"

from brain_act_hybrid_common import (
    COHORT_TO_CONDITION,
    CONDITION_ORDER,
    COND_COLORS,
    DATASET_ROOT,
    PROJECT_ROOT,
    SCENARIOS,
    parse_sim_npz_path,
    save_json,
)

from tvbtoolkit.analysis.brain_states import phase_patterns
from tvbtoolkit.datasets.brain_act import load_subject_structural


EXCLUDED_CONDITIONS = {"COMA"}
ANALYSIS_CONDITION_ORDER = [c for c in CONDITION_ORDER if c not in EXCLUDED_CONDITIONS]


@dataclass(frozen=True)
class DomainConfig:
    name: str
    x_key: str
    t_key: str
    pipeline: str
    trim_edge_samples: int
    pre_subsample_timeseries: bool
    max_rows_per_job: int | None
    bandpass_hz: tuple[float, float]
    filter_order: int
    crop_middle_ms: float | None = None  # None = use full timeseries
    # Optional per-domain k-means hyper-parameter overrides.  When set, these
    # take precedence over the corresponding CLI arguments.  This lets rate
    # keep the v2-optimal settings while BOLD matches the empirical fig7 run.
    kmeans_seed_override: int | None = None
    kmeans_n_init_override: int | None = None
    kmeans_max_iter_override: int | None = None


DOMAIN_CONFIGS = [
    DomainConfig(
        name="rate",
        x_key="rate",
        t_key="time_rate_ms",
        pipeline="firing_rate",
        trim_edge_samples=9,
        # Do NOT pre-subsample before pattern extraction: the rate monitor runs
        # at ~128 Hz (dt ≈ 7.8 ms, Nyquist ≈ 64 Hz).  Downsampling to a handful
        # of rows would collapse the sample rate far below the bandpass lower
        # edge (2 Hz) and crash scipy's butter().  Instead we let extract_patterns
        # filter the full cropped timeseries (~7 680 samples) and subsample the
        # resulting phase patterns afterwards via max_rows_per_job.
        pre_subsample_timeseries=False,
        # Rate domain: 1 200 patterns/subject — the largest round value that
        # still keeps the pooled matrix comfortably below the 4 GB Lloyd gate
        # (1 200 · 179 · 4 005 · 4 B ≈ 3.44 GB).  Maximum rows before gate
        # is 1 498; 1 200 leaves headroom for Lloyd's workspace allocation.
        # Rationale:
        #   (1) Stays on deterministic Lloyd KMeans, matching v1's algorithm.
        #   (2) 24× more training data than v1 (50) → more stable centroids
        #       and better local-minimum coverage with n_init=10 restarts.
        #   (3) At 50 ms spacing per pattern vs the ~13 ms autocorrelation
        #       timescale of the 2–40 Hz bandpass, samples are still ~4×
        #       decorrelated — k-means' independence assumption is satisfied.
        # If this drifts from the v1 figures, fall back to 50 rows.
        max_rows_per_job=1_200,
        # Upper limit must be strictly below Nyquist (≈ 64 Hz).  40 Hz captures
        # delta → low-gamma dynamics relevant for rate-based brain states.
        bandpass_hz=(2.0, 40.0),
        filter_order=4,
        # Crop to middle 60 s: avoids start/end transient edge effects and
        # keeps RAM/compute manageable (~7 680 timepoints instead of ~30 720
        # for the full 4-minute post-transient window).
        crop_middle_ms=60_000.0,
        # No k-means overrides → rate uses the CLI defaults, which match the
        # empirical fig7 pipeline (seed=11, n_init=40, max_iter=260).
    ),
    DomainConfig(
        name="bold",
        x_key="bold",
        t_key="time_bold_ms",
        pipeline="brain_act_legacy",
        # trim_edge_samples=9 matches the empirical fig7 pipeline
        # (reproduce_legacy_figures_new_doc.py run_metadata.json).
        trim_edge_samples=9,
        pre_subsample_timeseries=False,
        # BOLD: no subsampling (TR = 2.4 s is already near-Nyquist for the
        # 0.01–0.20 Hz bandpass; every pattern is effectively independent).
        max_rows_per_job=None,
        bandpass_hz=(0.01, 0.20),
        filter_order=3,
        crop_middle_ms=None,  # BOLD: use full timeseries (transient already discounted)
    ),
]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Publication-ready pooled brain-state analysis with scenario-wise shared centroids. "
            "Rate: 1 200 patterns/subject + Lloyd k-means (seed=11, n_init=40, max_iter=260, "
            "matching the empirical fig7 pipeline). "
            "1 200 is the largest round value keeping the pooled matrix under the 4 GB "
            "Lloyd gate (3.44 GB), giving 24× more training data than v1 while still "
            "preserving decorrelated-sample independence (50 ms spacing vs 13 ms autocorr. timescale). "
            "BOLD: NO subsampling + k-means (seed=11, n_init=40, max_iter=260) + trim_edge=9, "
            "matching the empirical fig7 pipeline. "
            "Per-domain k-means hyper-parameter overrides are declared in DomainConfig and "
            "take precedence over CLI defaults. "
            "Pass --diagnose-centroids to additionally refit under an alternative subsampling "
            "regime and compare centroids via Hungarian assignment."
        )
    )
    p.add_argument("--sim-root", type=Path, default=PROJECT_ROOT / "notebooks" / "outputs" / "ba_sim_hybrid" / "shared_b" / "sims")
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "notebooks" / "outputs" / "04_brain_states_analysis_pub_v5")
    p.add_argument("--n-states", type=int, default=5)
    # Defaults match the empirical fig7 pipeline:
    #   kmeans-seed=11, kmeans-n-init=40, kmeans-max-iter=260.
    p.add_argument("--kmeans-seed", type=int, default=11)
    p.add_argument("--kmeans-n-init", type=int, default=40)
    p.add_argument("--kmeans-max-iter", type=int, default=260)
    p.add_argument("--scenario", action="append", dest="scenarios", default=None)
    # 02_ inserts a {b_tag} level (e.g. b005, b035) between sims/ and scenario/.
    # Pass --b-tag b035 to select one value, or leave blank to use all b_tags found.
    p.add_argument("--b-tag", type=str, default=None,
                   help="b_e tag subfolder produced by 02_ (e.g. b035). "
                        "If omitted all b_tag subdirs are processed separately.")
    p.add_argument("--diagnose-centroids", action="store_true",
                   help="Additionally fit the v1 aggressive-subsample regime "
                        "(BOLD=12, rate=50 rows/subject) and compare centroids "
                        "to the production (Nyquist) regime via Hungarian "
                        "assignment.  Writes CSVs and heatmap PDFs to "
                        "{output-dir}/diagnostics/.")
    p.add_argument("--centroid-order", choices=("both", "reference_sc", "native"), default="both",
                   help="How to order shared k-means centroids in output rows. "
                        "'reference_sc' sorts once by the scenario/domain mean SC; "
                        "'native' leaves sklearn/MiniBatch centroid IDs unchanged; "
                        "'both' writes separate output folders for both options.")
    return p.parse_args()


# ── Centroid-comparison diagnostic ───────────────────────────────────────────
# v1 (aggressive subsampling, reproduces literature gradient) vs production.
_V1_AGGRESSIVE_ROWS: dict[str, int] = {"rate": 50, "bold": 12}



def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return float("nan")
    xa = aa[mask]
    xb = bb[mask]
    if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])



def subsample_rows(x: np.ndarray, max_rows: int | None) -> np.ndarray:
    if max_rows is None:
        return x
    k = int(max_rows)
    if k <= 0 or x.shape[0] <= k:
        return x
    idx = np.linspace(0, x.shape[0] - 1, k, dtype=int)
    return x[idx]



def subsample_timeseries_rows(x: np.ndarray, t: np.ndarray, max_rows: int | None, trim_edge_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows is None:
        return x, t
    k = int(max_rows)
    if k <= 0:
        return x, t
    keep = int(k + 2 * max(0, int(trim_edge_samples)))
    if x.shape[0] <= keep:
        return x, t
    idx = np.linspace(0, x.shape[0] - 1, keep, dtype=int)
    return x[idx], t[idx]


def crop_middle(
    x: np.ndarray,
    t: np.ndarray,
    crop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the central ``crop_ms`` milliseconds of the timeseries.

    If the timeseries is shorter than ``crop_ms``, the full timeseries is
    returned unchanged.  The crop is centred on the midpoint of the time axis
    so start/end edge effects (filter ringing, model warm-up) are avoided.
    """
    duration_ms = float(t[-1]) - float(t[0])
    if duration_ms <= crop_ms:
        return x, t
    mid_ms = float(t[0]) + duration_ms / 2.0
    half = crop_ms / 2.0
    mask = (t >= mid_ms - half) & (t <= mid_ms + half)
    return x[mask], t[mask]



def apply_damage_parity(connectivity: np.ndarray, tract_lengths: np.ndarray, cohort: str) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(connectivity, dtype=float).copy()
    l = np.asarray(tract_lengths, dtype=float).copy()
    np.fill_diagonal(c, 0.0)
    np.fill_diagonal(l, 0.0)
    if cohort.lower() in {"mcs", "uws", "emcs", "coma"}:
        mismatch = (c == 0.0) & (l != 0.0)
        if np.any(mismatch):
            l[mismatch] = 0.0
    cmax = float(np.max(c))
    if cmax > 0.0:
        c /= cmax
    return c, l



def load_sc_cache(dataset_root: Path, pairs: set[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for cohort, subject_id in sorted(pairs):
        c, l, _atlas, _meta = load_subject_structural(
            subject_id=subject_id,
            cohort=cohort,
            dataset_root=dataset_root,
            validate=True,
            enforce_symmetry=True,
            zero_diagonal=True,
            nonfinite="raise",
        )
        c_fix, _ = apply_damage_parity(c, l, cohort)
        out[(cohort, subject_id)] = c_fix
    return out



def extract_patterns(
    x: np.ndarray,
    t_ms: np.ndarray,
    cfg: DomainConfig,
) -> np.ndarray:
    dt_ms = float(np.median(np.diff(t_ms))) if t_ms.size > 1 else 1.0
    tr_s = max(dt_ms / 1000.0, 1e-6)
    pats, *_ = phase_patterns(
        x,
        trim_edge_samples=int(cfg.trim_edge_samples),
        pipeline=str(cfg.pipeline),
        tr_seconds=float(tr_s),
        dt_ms=float(dt_ms),
        transient_ms=0.0,
        bandpass_hz=cfg.bandpass_hz,
        filter_order=int(cfg.filter_order),
    )
    return np.asarray(pats, dtype=float)



def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)



def _scatter_panel(
    ax: "plt.Axes",
    records: list[dict[str, Any]],
    scenario: str,
    domain: str,
    b_tag: str,
) -> None:
    """Draw one SC–FC vs occupancy scatter panel for a given b_tag/scenario/domain.

    Each condition is rendered as a scatter cloud (coloured dots) plus a linear
    regression line in the same colour.  No per-panel legend is drawn; the
    caller adds a single shared legend to the figure.
    """
    rec_s = [
        r for r in records
        if r["b_tag"] == b_tag and r["scenario"] == scenario and r["domain"] == domain
    ]
    for cond in ANALYSIS_CONDITION_ORDER:
        rec_c = [r for r in rec_s if r["condition"] == cond]
        if not rec_c:
            continue
        xs = np.array([float(r["sfc_sub"]) for r in rec_c], dtype=float)
        ys = np.array([float(r["occupancy_pct"]) for r in rec_c], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if xs.size == 0:
            continue
        color = COND_COLORS[cond]
        # Scatter: small markers, no edge, semi-transparent
        ax.scatter(xs, ys, s=4, alpha=0.40, color=color,
                   linewidths=0, zorder=2, label=cond)
        # Regression line: opaque, slightly thicker than default
        if xs.size >= 3 and np.std(xs) > 1e-12:
            fit = linregress(xs, ys)
            xline = np.linspace(xs.min(), xs.max(), 120)
            ax.plot(xline, fit.slope * xline + fit.intercept,
                    color=color, lw=0.9, zorder=3)
    ax.set_xlabel("SC–FC coupling (r)")


def plot_b_sweep(
    all_records: list[dict[str, Any]],
    b_tags: list[str],
    scenarios: list[str],
    domain: str,
    domain_label: str,
    out_path: Path,
) -> None:
    """Publication-ready b-sweep figure (Nature double-column style).

    Layout: rows = b_e values, cols = noise scenarios.
    Each panel is a SC–FC vs occupancy scatter coloured by clinical condition,
    with per-condition linear regression lines.
    A single shared legend is placed to the right of the grid.
    """
    n_rows = len(b_tags)
    n_cols = len(scenarios)

    # Nature double-column = 183 mm ≈ 7.2 in.  Reserve ~1.1 in on the right
    # for the legend.  Panel height: 1.55 in leaves room for row labels + title.
    panel_w = 2.0          # inches per panel column
    panel_h = 1.55         # inches per panel row
    legend_w = 1.1         # inches reserved for condition legend
    fig_w = panel_w * n_cols + legend_w
    fig_h = panel_h * n_rows + 0.45   # + top margin for suptitle

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            sharey=False, sharex=False,
            squeeze=False,
        )

        for ri, b_tag in enumerate(b_tags):
            for ci, scenario in enumerate(scenarios):
                ax = axes[ri, ci]
                _scatter_panel(ax, all_records, scenario, domain, b_tag)

                # ── Column headers (top row only) ──────────────────────────
                if ri == 0:
                    short = _SCENARIO_SHORT.get(
                        scenario,
                        SCENARIOS.get(scenario, {}).get("label", scenario),
                    )
                    ax.set_title(short, fontsize=6.5, fontweight="bold",
                                 pad=3, linespacing=1.3)

                # ── Y-axis: label only on left column ─────────────────────
                if ci == 0:
                    ax.set_ylabel("Occupancy (%)")
                else:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)

        # ── Row labels: b_e value, rotated, left of each row ──────────────
        # We use tight_layout first so get_position() returns final coords.
        fig.tight_layout(
            rect=[0.0, 0.0, (fig_w - legend_w) / fig_w, 0.93],
            h_pad=0.8, w_pad=0.5,
        )
        for ri, b_tag in enumerate(b_tags):
            ax0 = axes[ri, 0]
            pos = ax0.get_position()          # in figure-fraction coords
            y_centre = pos.y0 + pos.height / 2.0
            x_left = pos.x0 - 0.055          # just left of the left spine
            fig.text(
                x_left, y_centre,
                f"$b_e$ = {b_tag}",
                va="center", ha="center",
                fontsize=6, fontweight="bold",
                rotation=90,
            )

        # ── Shared condition legend, right of grid ─────────────────────────
        # Collect one handle per condition from the first populated axes.
        cond_handles: dict[str, Any] = {}
        for ax_row in axes:
            for ax in ax_row:
                for h, lbl in zip(*ax.get_legend_handles_labels()):
                    if lbl not in cond_handles:
                        cond_handles[lbl] = h
            if cond_handles:
                break

        if cond_handles:
            fig.legend(
                list(cond_handles.values()),
                list(cond_handles.keys()),
                loc="center left",
                bbox_to_anchor=((fig_w - legend_w + 0.05) / fig_w, 0.5),
                title="Condition",
                title_fontsize=6.5,
                fontsize=6,
                frameon=False,
                handlelength=0.8,
                handleheight=0.75,
                handletextpad=0.4,
                labelspacing=0.4,
                borderpad=0.3,
                markerscale=1.8,
            )

        fig.suptitle(
            f"Brain states — {domain_label}   |   SC–FC coupling vs. state occupancy",
            fontsize=7, fontweight="bold", y=0.98,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[04]  saved → {out_path}")


# ── Centroid-comparison diagnostic helpers ───────────────────────────────────

def _fit_centers_from_blocks(
    per_subject_blocks: list[np.ndarray],
    max_rows_per_subject: int | None,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    """Fit k-means under a given per-subject subsampling regime.

    Returns the (k, n_features) centroid matrix.  Raises if not enough
    rows remain after subsampling.
    """
    blocks: list[np.ndarray] = []
    for pats in per_subject_blocks:
        sub = subsample_rows(np.asarray(pats, dtype=np.float32), max_rows_per_subject)
        if sub.shape[0] >= n_clusters:
            blocks.append(sub)
    if not blocks:
        raise RuntimeError("No usable blocks after subsampling for diagnostic fit.")
    pooled = np.concatenate(blocks, axis=0)
    # Use MiniBatchKMeans only if >4 GB (matches main pipeline gate).
    if pooled.nbytes > 4 * (1024 ** 3):
        km = MiniBatchKMeans(
            n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
            random_state=seed, batch_size=min(pooled.shape[0], 4096),
        )
    else:
        km = KMeans(
            n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
            random_state=seed, algorithm="lloyd",
        )
    km.fit(pooled)
    return np.asarray(km.cluster_centers_, dtype=float)


def _centroid_correlation_matrix(centers_a: np.ndarray, centers_b: np.ndarray) -> np.ndarray:
    """Pearson correlation between every (A_i, B_j) centroid pair."""
    ka, kb = centers_a.shape[0], centers_b.shape[0]
    corr = np.zeros((ka, kb), dtype=float)
    for i in range(ka):
        for j in range(kb):
            corr[i, j] = safe_pearson(centers_a[i], centers_b[j])
    return corr


def _hungarian_match(corr_matrix: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Find the one-to-one assignment of rows→cols maximising total correlation.

    Returns:
        assignment: list such that row i is matched with col assignment[i].
        matched_corrs: corr_matrix[i, assignment[i]] for each i.
    """
    from scipy.optimize import linear_sum_assignment
    # Minimise -corr ⇔ maximise corr.  Replace NaNs with -inf so they can never be picked.
    cost = -np.where(np.isfinite(corr_matrix), corr_matrix, -1e9)
    row_ind, col_ind = linear_sum_assignment(cost)
    # linear_sum_assignment returns row_ind sorted; make assignment indexable by i.
    assignment_arr = np.full(corr_matrix.shape[0], -1, dtype=int)
    assignment_arr[row_ind] = col_ind
    matched = np.array(
        [corr_matrix[i, assignment_arr[i]] if assignment_arr[i] >= 0 else np.nan
         for i in range(corr_matrix.shape[0])],
        dtype=float,
    )
    return assignment_arr.tolist(), matched


def _plot_centroid_diag_grid(
    entries: list[dict[str, Any]],
    domain: str,
    domain_label: str,
    out_path: Path,
) -> None:
    """Heatmap grid: rows = b_tags, cols = scenarios.

    Each panel shows the k×k Pearson correlation between v1-aggressive
    centroids (rows) and production (Nyquist) centroids (columns), with
    the Hungarian-optimal assignment highlighted by black rectangles.
    """
    if not entries:
        return

    b_tags = sorted({e["b_tag"] for e in entries})
    scenarios = [s for s in SCENARIOS if any(e["scenario"] == s for e in entries)]
    n_rows = len(b_tags)
    n_cols = len(scenarios)

    panel = 1.6
    fig_w = panel * n_cols + 1.0
    fig_h = panel * n_rows + 0.8

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
        for ri, b_tag in enumerate(b_tags):
            for ci, scenario in enumerate(scenarios):
                ax = axes[ri, ci]
                match = [e for e in entries
                         if e["b_tag"] == b_tag and e["scenario"] == scenario]
                if not match:
                    ax.axis("off")
                    continue
                e = match[0]
                corr = np.asarray(e["corr_matrix"], dtype=float)
                im = ax.imshow(
                    corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                    aspect="equal", origin="upper",
                )
                # Highlight Hungarian matches
                for i, j in enumerate(e["assignment"]):
                    if j >= 0:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="black", linewidth=0.9,
                        ))
                ax.set_xticks(range(corr.shape[1]))
                ax.set_yticks(range(corr.shape[0]))
                ax.set_xticklabels([str(k + 1) for k in range(corr.shape[1])])
                ax.set_yticklabels([str(k + 1) for k in range(corr.shape[0])])
                if ri == 0:
                    short = _SCENARIO_SHORT.get(
                        scenario,
                        SCENARIOS.get(scenario, {}).get("label", scenario),
                    )
                    ax.set_title(short, fontsize=6.5, fontweight="bold",
                                 pad=3, linespacing=1.3)
                if ci == 0:
                    ax.set_ylabel(f"$b_e$={b_tag}\nv1 state", fontsize=6)
                else:
                    ax.set_ylabel("")
                if ri == n_rows - 1:
                    ax.set_xlabel("Production state", fontsize=6)

        fig.suptitle(
            f"Centroid-matching diagnostic — {domain_label}   |   "
            f"v1 aggressive subsample vs. production (Nyquist)",
            fontsize=7, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.96], h_pad=0.6, w_pad=0.4)

        # Shared colorbar
        cbar_ax = fig.add_axes([0.955, 0.15, 0.012, 0.70])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label("Pearson r (centroid ↔ centroid)", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[04]  diagnostic saved → {out_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_modes = (
        ["reference_sc", "native"]
        if args.centroid_order == "both"
        else [str(args.centroid_order)]
    )
    order_mode_dirs = {
        "reference_sc": "reference_sc_order",
        "native": "native_centroid_order",
    }

    # Build scenario list: start from CLI args or all known scenarios, then
    # restrict to those that actually exist on disk across any sim_root.
    candidate_scenarios = [s for s in (args.scenarios or list(SCENARIOS.keys())) if s in SCENARIOS]

    # Resolve b_tag level: 02_ saves as sims/{b_tag}/{scenario}/...
    # Detect whether the sim_root has a b_tag layer by checking if immediate
    # subdirs are b_tag folders (e.g. b005, b035) rather than scenario names.
    known_scenarios = set(SCENARIOS.keys())
    immediate_subdirs = [d.name for d in sorted(args.sim_root.iterdir()) if d.is_dir()] if args.sim_root.exists() else []
    has_b_tag_level = bool(immediate_subdirs) and not any(d in known_scenarios for d in immediate_subdirs)

    if has_b_tag_level:
        if args.b_tag:
            sim_roots = [args.sim_root / args.b_tag]
        else:
            sim_roots = sorted([args.sim_root / d for d in immediate_subdirs])
        print(f"[04] Detected b_tag layer. Processing: {[r.name for r in sim_roots]}")
    else:
        sim_roots = [args.sim_root]

    # Filter candidate_scenarios to those that actually exist on disk
    scenarios = [
        s for s in candidate_scenarios
        if any((sr / s).is_dir() for sr in sim_roots)
    ]
    missing = [s for s in candidate_scenarios if s not in scenarios]
    if missing:
        print(f"[04] Skipping scenarios not found on disk: {missing}")
    print(f"[04] Scenarios to process: {scenarios}")

    # ── Pre-load SC cache (shared across all b_tags) ──────────────────────────
    all_npz: list[Path] = []
    for sr in sim_roots:
        found = sorted(sr.glob("*/*/*/seed_*.npz"))
        found = [p for p in found if p.parts[-4] in set(scenarios)]
        all_npz.extend(found)

    if not all_npz:
        print(f"[04] ERROR: No NPZ files found.")
        return

    cohort_subject_pairs: set[tuple[str, str]] = set()
    for p in all_npz:
        parts = p.parts
        # parts[-4]=scenario, [-3]=cohort, [-2]=subject_id, [-1]=seed_*.npz
        cohort = parts[-3]
        cond = COHORT_TO_CONDITION.get(cohort)
        if cond is None or cond in EXCLUDED_CONDITIONS:
            continue
        cohort_subject_pairs.add((cohort, parts[-2]))
    sc_cache = load_sc_cache(args.dataset_root, cohort_subject_pairs)

    # ── Run one completely independent analysis per b_tag ─────────────────────
    all_records_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in order_modes}
    processed_b_tags_by_mode: dict[str, list[str]] = {m: [] for m in order_modes}
    diag_entries: dict[str, list[dict[str, Any]]] = {"rate": [], "bold": []}

    for sr in sim_roots:
        b_tag = sr.name
        print(f"\n{'='*60}")
        print(f"[04] b_tag={b_tag}  →  order modes: {order_modes}")
        print(f"{'='*60}")

        sr_scenarios = [s for s in scenarios if (sr / s).is_dir()]
        if not sr_scenarios:
            print(f"[04]  No scenario dirs found under {sr}, skipping.")
            continue

        state_rows_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in order_modes}

        for cfg in DOMAIN_CONFIGS:
            print(f"[04]  domain={cfg.name}")
            for scenario in sr_scenarios:
                npz_paths = sorted((sr / scenario).glob("*/*/seed_*.npz"))
                if not npz_paths:
                    print(f"[04]    scenario={scenario}  (no files)")
                    continue
                print(f"[04]    scenario={scenario}  ({len(npz_paths)} files)")

                pooled_blocks: list[np.ndarray] = []
                pooled_meta: list[dict[str, Any]] = []
                subject_sc_mats: list[np.ndarray] = []

                for p in npz_paths:
                    d = np.load(p, allow_pickle=True)
                    if cfg.x_key not in d or cfg.t_key not in d:
                        continue

                    scenario_k, cohort, subject_id, seed = parse_sim_npz_path(p, sr)
                    cond = COHORT_TO_CONDITION.get(cohort)
                    if cond is None or cond in EXCLUDED_CONDITIONS:
                        continue

                    x = np.asarray(d[cfg.x_key], dtype=float)
                    t = np.asarray(d[cfg.t_key], dtype=float)
                    if x.ndim != 2 or x.shape[0] < max(10, int(args.n_states)):
                        continue

                    # Crop to middle window for rate (avoids edge effects).
                    # BOLD uses full timeseries (crop_middle_ms=None).
                    if cfg.crop_middle_ms is not None:
                        x, t = crop_middle(x, t, cfg.crop_middle_ms)
                        if x.shape[0] < max(10, int(args.n_states)):
                            continue

                    if cfg.pre_subsample_timeseries:
                        x_fit, t_fit = subsample_timeseries_rows(
                            x, t,
                            max_rows=cfg.max_rows_per_job,
                            trim_edge_samples=cfg.trim_edge_samples,
                        )
                    else:
                        x_fit, t_fit = x, t

                    try:
                        pats = extract_patterns(x_fit, t_fit, cfg)
                    except Exception:
                        continue
                    if pats.shape[0] < max(2, int(args.n_states)):
                        continue

                    # Per-subject pattern count depends on the domain:
                    #   BOLD (max_rows_per_job=None) → keep every pattern (TR 2.4 s
                    #       is already near-Nyquist for 0.01–0.20 Hz).
                    #   Rate (max_rows_per_job=4 500) → subsample to the Nyquist-
                    #       equivalent decorrelated count (2·B·T = 2·38·60).
                    # subsample_rows() is a no-op when max_rows_per_job is None.
                    block = subsample_rows(pats, cfg.max_rows_per_job).astype(
                        np.float32, copy=False
                    )
                    if block.shape[0] < max(2, int(args.n_states)):
                        continue

                    pooled_blocks.append(block)
                    pooled_meta.append({
                        "path": p,
                        "scenario": scenario_k,
                        "cohort": cohort,
                        "subject_id": subject_id,
                        "condition": cond,
                        "seed": seed,
                    })
                    subject_sc_mats.append(sc_cache[(cohort, subject_id)])

                if not pooled_blocks:
                    continue

                pooled = np.concatenate(pooled_blocks, axis=0)
                n_clusters = min(int(args.n_states), int(pooled.shape[0]))
                if n_clusters < 2:
                    continue

                # Per-domain k-means hyper-parameters: DomainConfig overrides
                # win; otherwise fall back to CLI defaults.
                _km_seed = (cfg.kmeans_seed_override
                            if cfg.kmeans_seed_override is not None
                            else int(args.kmeans_seed))
                _km_ninit = (cfg.kmeans_n_init_override
                             if cfg.kmeans_n_init_override is not None
                             else int(args.kmeans_n_init))
                _km_maxit = (cfg.kmeans_max_iter_override
                             if cfg.kmeans_max_iter_override is not None
                             else int(args.kmeans_max_iter))

                # Choose clustering algorithm based on pooled-matrix memory footprint.
                # BOLD: ~100 TRs × N subjects × 4 005 cols ≈ < 1 GB → standard KMeans.
                # Rate: ~7 660 samples × N subjects × 4 005 cols can exceed 4 GB →
                #   fall back to MiniBatchKMeans to avoid OOM while still training on
                #   the full (unsubsampled) dataset.
                _MB = pooled.nbytes / (1024 ** 2)
                _GB = _MB / 1024.0
                if pooled.nbytes > 4 * (1024 ** 3):  # > 4 GB
                    _batch = min(pooled.shape[0], 4096)
                    print(
                        f"[04]      MiniBatchKMeans on {pooled.shape} "
                        f"({_GB:.1f} GB, batch={_batch}, "
                        f"seed={_km_seed}, n_init={_km_ninit}, max_iter={_km_maxit}) …"
                    )
                    kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        n_init=int(_km_ninit),
                        max_iter=int(_km_maxit),
                        random_state=int(_km_seed),
                        batch_size=_batch,
                    )
                else:
                    print(
                        f"[04]      KMeans on {pooled.shape} ({_MB:.0f} MB, "
                        f"seed={_km_seed}, n_init={_km_ninit}, max_iter={_km_maxit}) …"
                    )
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        n_init=int(_km_ninit),
                        max_iter=int(_km_maxit),
                        random_state=int(_km_seed),
                        algorithm="lloyd",
                    )
                _labels = kmeans.fit_predict(pooled)
                centers = np.asarray(kmeans.cluster_centers_, dtype=float)

                # ── Optional centroid-matching diagnostic ─────────────────
                # Refit under the v1 aggressive-subsample regime and compare
                # centroids to the production fit via Hungarian assignment.
                if args.diagnose_centroids:
                    v1_rows = _V1_AGGRESSIVE_ROWS.get(cfg.name)
                    if v1_rows is not None and pooled_blocks:
                        try:
                            centers_v1 = _fit_centers_from_blocks(
                                per_subject_blocks=pooled_blocks,
                                max_rows_per_subject=int(v1_rows),
                                n_clusters=n_clusters,
                                n_init=int(_km_ninit),
                                max_iter=int(_km_maxit),
                                seed=int(_km_seed),
                            )
                            corr = _centroid_correlation_matrix(centers_v1, centers)
                            assignment, matched = _hungarian_match(corr)
                            diag_entries[cfg.name].append({
                                "b_tag":            b_tag,
                                "scenario":         scenario,
                                "v1_rows":          int(v1_rows),
                                "corr_matrix":      corr.tolist(),
                                "assignment":       list(assignment),
                                "matched_corrs":    matched.tolist(),
                                "mean_matched_corr": float(np.nanmean(matched)),
                            })
                            print(
                                f"[04]      centroid diag "
                                f"(v1 @ {v1_rows} rows/sub vs prod)  "
                                f"mean matched r = {np.nanmean(matched):.3f}"
                            )
                        except Exception as exc:
                            print(f"[04]      centroid diag FAILED: {exc}")

                sc_ref_mat = np.mean(np.stack(subject_sc_mats, axis=0), axis=0)
                iu = np.triu_indices(sc_ref_mat.shape[0], k=1)
                sc_ref_vec = sc_ref_mat[iu]
                sfc_ref = np.array([safe_pearson(c, sc_ref_vec) for c in centers], dtype=float)
                centroid_orders = {
                    "reference_sc": np.argsort(np.nan_to_num(sfc_ref, nan=np.inf)),
                    "native": np.arange(n_clusters, dtype=int),
                }

                for meta in pooled_meta:
                    p = meta["path"]
                    d_full = np.load(p, allow_pickle=True)
                    x_full = np.asarray(d_full[cfg.x_key], dtype=float)
                    t_full = np.asarray(d_full[cfg.t_key], dtype=float)
                    if x_full.ndim != 2 or x_full.shape[0] < max(10, int(args.n_states)):
                        continue
                    try:
                        pats_full = extract_patterns(x_full, t_full, cfg)
                    except Exception:
                        continue
                    if pats_full.shape[0] < 1:
                        continue

                    sc_sub = sc_cache[(meta["cohort"], meta["subject_id"])]
                    sc_sub_vec = sc_sub[np.triu_indices(sc_sub.shape[0], k=1)]

                    for mode in order_modes:
                        order = centroid_orders[mode]
                        centers_ord = centers[order]
                        sfc_ref_ord = sfc_ref[order]

                        diffs = pats_full[:, None, :].astype(float) - centers_ord[None, :, :]
                        full_lab = np.argmin(np.sum(diffs**2, axis=2), axis=1)
                        occ = (
                            np.bincount(full_lab, minlength=n_clusters).astype(float)
                            / float(max(1, full_lab.size))
                        )
                        sfc_sub = np.array(
                            [safe_pearson(c, sc_sub_vec) for c in centers_ord],
                            dtype=float,
                        )

                        for k in range(n_clusters):
                            state_rows_by_mode[mode].append({
                                "centroid_order_mode": mode,
                                "b_tag":          b_tag,
                                "domain":         cfg.name,
                                "scenario":       meta["scenario"],
                                "scenario_label": SCENARIOS[meta["scenario"]]["label"],
                                "cohort":         meta["cohort"],
                                "condition":      meta["condition"],
                                "subject_id":     meta["subject_id"],
                                "seed":           int(meta["seed"]),
                                "state_rank":     int(k + 1),
                                "centroid_id":    int(order[k]) + 1,
                                "occupancy_pct":  float(occ[k] * 100.0),
                                "sfc_ref":        float(sfc_ref_ord[k]),
                                "sfc_sub":        float(sfc_sub[k]),
                            })

        # Save per-b_tag CSV and accumulate for sweep figures
        for mode in order_modes:
            out_dir = args.output_dir / order_mode_dirs[mode] / b_tag
            out_dir.mkdir(parents=True, exist_ok=True)
            state_rows = state_rows_by_mode[mode]
            write_csv(out_dir / "brain_states_subject_state_rows.csv", state_rows)
            all_records_by_mode[mode].extend(state_rows)
            if state_rows:
                processed_b_tags_by_mode[mode].append(b_tag)
            print(f"[04]  outputs [{mode}] → {out_dir}  ({len(state_rows)} state rows)")

    # ── Combined b-sweep figures (one per domain) ─────────────────────────────
    for mode in order_modes:
        all_records = all_records_by_mode[mode]
        processed_b_tags = processed_b_tags_by_mode[mode]
        if all_records and len(processed_b_tags) >= 1:
            sweep_fig_dir = args.output_dir / order_mode_dirs[mode] / "figs"
            # Scenarios present in all_records (preserve display order from SCENARIOS)
            present_scenarios = [s for s in SCENARIOS if any(r["scenario"] == s for r in all_records)]
            plot_b_sweep(
                all_records,
                processed_b_tags,
                present_scenarios,
                domain="rate",
                domain_label="Firing Rates",
                out_path=sweep_fig_dir / "fig_rate_b_sweep.pdf",
            )
            plot_b_sweep(
                all_records,
                processed_b_tags,
                present_scenarios,
                domain="bold",
                domain_label="BOLD",
                out_path=sweep_fig_dir / "fig_bold_b_sweep.pdf",
            )

    # ── Centroid-matching diagnostic outputs (optional) ───────────────────────
    if args.diagnose_centroids and any(diag_entries.values()):
        diag_dir = args.output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        for domain, entries in diag_entries.items():
            if not entries:
                continue
            # Flatten to CSV: one row per (b_tag, scenario, v1_state) match
            rows: list[dict[str, Any]] = []
            for e in entries:
                for i, (j, r) in enumerate(zip(e["assignment"], e["matched_corrs"])):
                    rows.append({
                        "domain":            domain,
                        "b_tag":             e["b_tag"],
                        "scenario":          e["scenario"],
                        "v1_rows_per_sub":   int(e["v1_rows"]),
                        "v1_state_index":    int(i) + 1,
                        "prod_state_index":  (int(j) + 1) if j >= 0 else -1,
                        "matched_pearson_r": float(r),
                    })
            write_csv(diag_dir / f"centroid_match_{domain}.csv", rows)
            _plot_centroid_diag_grid(
                entries, domain,
                domain_label={"rate": "Firing Rates", "bold": "BOLD"}[domain],
                out_path=diag_dir / f"fig_centroid_match_{domain}.pdf",
            )
        print(f"[04]  centroid-match diagnostics → {diag_dir}")

    save_json(
        args.output_dir / "run_manifest.json",
        {
            "script": "04_brain_states_analysis_pub.py",
            "sim_root": str(args.sim_root),
            "dataset_root": str(args.dataset_root),
            "b_tags": [sr.name for sr in sim_roots],
            "scenarios": scenarios,
            "excluded_conditions": sorted(EXCLUDED_CONDITIONS),
            "n_states": int(args.n_states),
            "kmeans_seed": int(args.kmeans_seed),
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
            "centroid_order_requested": str(args.centroid_order),
            "centroid_order_modes": order_modes,
            "centroid_order_output_dirs": order_mode_dirs,
            "centroid_order_notes": {
                "reference_sc": (
                    "Centroids are sorted once by Pearson coupling to the "
                    "scenario/domain mean SC; sfc_sub is still recomputed "
                    "against each subject's own SC for plotting."
                ),
                "native": (
                    "Centroids retain native k-means centroid IDs. The "
                    "SCFC-vs-occupancy scatter/regression still uses "
                    "subject-specific sfc_sub and is invariant to row order."
                ),
            },
            "diagnose_centroids": bool(args.diagnose_centroids),
            "domain_configs": [cfg.__dict__ for cfg in DOMAIN_CONFIGS],
        },
    )

    print(f"\n[04] done. outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
