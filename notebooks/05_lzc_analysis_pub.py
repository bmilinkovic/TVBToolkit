#!/usr/bin/env python3
"""Publication-ready LZc (Lempel-Ziv complexity) analysis.

Processes spontaneous simulation outputs from 02_ (ba_sim_hybrid), which are
organised as:  sims/{b_tag}/{scenario}/{cohort}/{subject_id}/seed_*.npz

Key conventions (matching 04_brain_states_analysis_pub.py)
----------------------------------------------------------
* b_tag layer auto-detected (b005, b035, b125, b220, …)
* Rate timeseries cropped to the **middle 60 s** before computing LZc
* Rate bandpass: 2–40 Hz  (Nyquist ≈ 64 Hz @ 128 Hz monitor; 80 Hz cap
  exceeded Nyquist and caused silent filter failures)
* BOLD bandpass: 0.01–0.20 Hz (unchanged)
* Sedation status loaded from structural metadata for every subject
  (meta.sedation ∈ {'non_sedated', 'sedated'})
* Seeds averaged per subject before statistics (one point per subject)
* Statistics: Kruskal-Wallis omnibus  +  pairwise Mann-Whitney + Holm
  + Cliff's δ; run for (i) all, (ii) non-sedated only,
  (iii) sedated patients only, (iv) sedated vs non-sedated within cohort
* Nature-quality figures: two per domain
    fig_lzc_{domain}_non_sedated_sweep.pdf  — rows=b_tags, cols=scenarios
    fig_lzc_{domain}_sedation_split.pdf     — sedated vs non-sedated facet
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, iirfilter
from scipy.stats import kruskal, mannwhitneyu, zscore

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

from tvbtoolkit.complexity.measures import lzc_multichannel
from tvbtoolkit.datasets.brain_act import load_subject_structural


# ── Publication aesthetics (Nature guidelines) ───────────────────────────────
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
    "legend.title_fontsize": 6.5,
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "savefig.dpi":           300,
}

# Compact two-line scenario column headers (mirrors 04_)
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

EXCLUDED_CONDITIONS = {"COMA"}
ANALYSIS_CONDITION_ORDER = [c for c in CONDITION_ORDER if c not in EXCLUDED_CONDITIONS]

# Patient conditions (no control) used in sedation split figure
_PATIENT_CONDITIONS = [c for c in ANALYSIS_CONDITION_ORDER if c != "CNT"]

RATE_CROP_MS   = 60_000.0   # crop to middle 60 s for rate
RATE_BAND_HZ   = (2.0, 40.0)
BOLD_BAND_HZ   = (0.01, 0.20)
RATE_FILT_ORD  = 4
BOLD_FILT_ORD  = 3


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publication-ready LZc analysis from ba_sim_hybrid sims."
    )
    p.add_argument("--sim-root",   type=Path,
                   default=PROJECT_ROOT / "notebooks" / "outputs" / "ba_sim_hybrid" / "shared_b" / "sims")
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "notebooks" / "outputs" / "05_lzc_analysis_pub")
    p.add_argument("--scenario",   action="append", dest="scenarios", default=None)
    p.add_argument("--b-tag",      type=str, default=None,
                   help="b_e tag (e.g. b035); omit to process all b_tags found.")
    return p.parse_args()


# ── Signal processing ─────────────────────────────────────────────────────────

def crop_middle(x: np.ndarray, t: np.ndarray, crop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the central crop_ms milliseconds of the timeseries."""
    duration_ms = float(t[-1]) - float(t[0])
    if duration_ms <= crop_ms:
        return x, t
    mid_ms = float(t[0]) + duration_ms / 2.0
    half   = crop_ms / 2.0
    mask   = (t >= mid_ms - half) & (t <= mid_ms + half)
    return x[mask], t[mask]


def bandpass_filter(x: np.ndarray, dt_ms: float,
                    band_hz: tuple[float, float], order: int) -> np.ndarray:
    nyq = 0.5 / (dt_ms / 1000.0)
    lo  = max(min(float(band_hz[0]) / nyq, 0.99), 1e-6)
    hi  = max(min(float(band_hz[1]) / nyq, 0.999), lo + 1e-6)
    b, a = iirfilter(int(order), (lo, hi), btype="bandpass", ftype="butter", output="ba")
    try:
        return filtfilt(b, a, x, axis=0)
    except ValueError:
        return filtfilt(b, a, x, axis=0, method="gust")


def compute_lzc_rate(rate: np.ndarray, t_ms: np.ndarray) -> float:
    """Crop middle 60 s → z-score → bandpass (2–40 Hz) → LZc."""
    xc, tc = crop_middle(rate, t_ms, RATE_CROP_MS)
    if xc.shape[0] < 64:
        return float("nan")
    dt_ms = float(np.median(np.diff(tc))) if tc.size > 1 else 7.8125
    xz = zscore(xc, axis=0, ddof=1)
    xz = np.nan_to_num(xz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xf = bandpass_filter(xz, dt_ms, RATE_BAND_HZ, RATE_FILT_ORD)
    return float(lzc_multichannel(xf))


def compute_lzc_bold(bold: np.ndarray, t_ms: np.ndarray) -> float:
    """Bandpass (0.01–0.20 Hz) → LZc on full BOLD timeseries."""
    if bold.shape[0] < 30:
        return float("nan")
    dt_ms = float(np.median(np.diff(t_ms))) if t_ms.size > 1 else 2400.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xf = bandpass_filter(bold, dt_ms, BOLD_BAND_HZ, BOLD_FILT_ORD)
    return float(lzc_multichannel(xf))


# ── Metadata / sedation cache ─────────────────────────────────────────────────

def build_sedation_cache(
    dataset_root: Path,
    pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Return {(cohort, subject_id): sedation_str} for every pair."""
    cache: dict[tuple[str, str], str] = {}
    for cohort, sid in sorted(pairs):
        try:
            _, _, _, meta = load_subject_structural(
                subject_id=sid, cohort=cohort, dataset_root=dataset_root,
                validate=False, enforce_symmetry=False,
                zero_diagonal=False, nonfinite="ignore",
            )
            cache[(cohort, sid)] = str(meta.sedation) if meta.sedation else "non_sedated"
        except Exception:
            cache[(cohort, sid)] = "non_sedated"   # safe fallback
    return cache


# ── Statistics helpers ────────────────────────────────────────────────────────

def _holm_correct(pvals: list[float]) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    if arr.size == 0:
        return arr
    m     = arr.size
    order = np.argsort(arr)
    out   = np.empty(m, dtype=float)
    prev  = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * arr[idx], 1.0)
        adj = max(adj, prev)
        out[idx] = adj
        prev = out[idx]
    return out


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    xa, ya = np.asarray(x).ravel(), np.asarray(y).ravel()
    if xa.size == 0 or ya.size == 0:
        return float("nan")
    gt = int(np.sum(xa[:, None] > ya[None, :]))
    lt = int(np.sum(xa[:, None] < ya[None, :]))
    return float((gt - lt) / (xa.size * ya.size))


def _pvalue_label(p: float) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def run_omnibus_and_pairwise(
    groups: dict[str, np.ndarray],   # condition → values (already seed-averaged)
    subset_label: str,
) -> tuple[list[dict], list[dict]]:
    """Kruskal-Wallis + pairwise Mann-Whitney with Holm correction."""
    omni_rows:  list[dict] = []
    pair_rows:  list[dict] = []

    valid = {c: v for c, v in groups.items() if v.size >= 2}
    if len(valid) < 2:
        return omni_rows, pair_rows

    # Omnibus
    try:
        H, p_omni = kruskal(*valid.values())
    except Exception:
        H, p_omni = float("nan"), float("nan")
    omni_rows.append({
        "subset":      subset_label,
        "conditions":  "|".join(valid.keys()),
        "n_groups":    len(valid),
        "H_statistic": round(float(H), 4),
        "pvalue":      round(float(p_omni), 6),
        "n_per_group": "|".join(str(v.size) for v in valid.values()),
    })

    # Pairwise
    conds = list(valid.keys())
    raw_p, records = [], []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            ca, cb = conds[i], conds[j]
            xa, xb = valid[ca], valid[cb]
            try:
                U, p = mannwhitneyu(xa, xb, alternative="two-sided")
            except Exception:
                U, p = float("nan"), float("nan")
            d = _cliffs_delta(xa, xb)
            records.append({
                "subset":       subset_label,
                "cond_a":       ca,
                "cond_b":       cb,
                "n_a":          xa.size,
                "n_b":          xb.size,
                "U_statistic":  round(float(U), 2),
                "pvalue_raw":   round(float(p), 6),
                "pvalue_holm":  None,          # filled below
                "cliffs_delta": round(float(d), 4),
            })
            raw_p.append(float(p))

    if records:
        adj = _holm_correct(raw_p)
        for rec, ap in zip(records, adj):
            rec["pvalue_holm"] = round(float(ap), 6)
        pair_rows.extend(records)

    return omni_rows, pair_rows


def run_sedation_stats(
    sed_groups: dict[str, dict[str, np.ndarray]],  # cohort → {sedation → values}
) -> list[dict]:
    """Mann-Whitney sedated vs non-sedated within each patient cohort."""
    rows: list[dict] = []
    for cohort, by_sed in sorted(sed_groups.items()):
        xn = by_sed.get("non_sedated", np.array([]))
        xs = by_sed.get("sedated",     np.array([]))
        if xn.size < 2 or xs.size < 2:
            continue
        try:
            U, p = mannwhitneyu(xn, xs, alternative="two-sided")
        except Exception:
            U, p = float("nan"), float("nan")
        d = _cliffs_delta(xn, xs)
        rows.append({
            "cohort":        cohort,
            "condition":     COHORT_TO_CONDITION.get(cohort, cohort),
            "n_non_sedated": xn.size,
            "n_sedated":     xs.size,
            "U_statistic":   round(float(U), 2),
            "pvalue_raw":    round(float(p), 6),
            "cliffs_delta":  round(float(d), 4),
        })
    return rows


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: float = 0.55) -> str:
    """Mix hex_color towards white by amount ∈ [0, 1]."""
    c = np.array(mc.to_rgb(hex_color))
    return mc.to_hex(1.0 - amount * (1.0 - c))


def _strip_mean_se(
    ax: "plt.Axes",
    xi: float,
    vals: np.ndarray,
    color: str,
    *,
    jitter_w: float = 0.16,
    seed: int = 0,
    alpha_dot: float = 0.45,
) -> None:
    """Jittered strip + mean line + SE whisker at position xi."""
    if vals.size == 0:
        return
    rng = np.random.default_rng(seed)
    jit = rng.uniform(-jitter_w, jitter_w, size=vals.size)
    ax.scatter(xi + jit, vals, s=5, alpha=alpha_dot,
               color=color, linewidths=0, zorder=3)
    mean = float(np.mean(vals))
    se   = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    ax.hlines(mean, xi - jitter_w * 0.85, xi + jitter_w * 0.85,
              colors=color, lw=1.8, zorder=5)
    ax.errorbar(xi, mean, yerr=se, fmt="none",
                color=color, capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)


def _sig_bracket(
    ax: "plt.Axes",
    x1: float, x2: float,
    y_bot: float,
    label: str,
    *,
    col: str = "0.25",
    lw: float = 0.55,
    dy_frac: float = 0.045,
) -> float:
    """Draw bracket at y_bot; return y_top so callers can stack brackets."""
    if not label:
        return y_bot
    ylo, yhi = ax.get_ylim()
    dy = (yhi - ylo) * dy_frac
    ax.plot([x1, x1, x2, x2], [y_bot, y_bot + dy, y_bot + dy, y_bot],
            lw=lw, color=col, clip_on=False)
    ax.text((x1 + x2) / 2, y_bot + dy * 1.05, label,
            ha="center", va="bottom", fontsize=5, color=col)
    return y_bot + dy * 2.2


def _panel_lzc(
    ax: "plt.Axes",
    records: list[dict],     # already filtered to b_tag + scenario + domain
    *,
    subset: str = "non_sedated",   # "non_sedated" | "all"
    pair_stats: list[dict] | None = None,
    show_xtick_labels: bool = True,
) -> None:
    """
    Draw one LZc panel: jittered strip + mean±SE per condition.

    pair_stats: pairwise rows for this panel (used to draw sig brackets).
    subset: if "non_sedated" only those rows are used.
    """
    # Seed-average per subject
    from collections import defaultdict
    by_subj: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        if subset == "non_sedated" and r["sedation"] != "non_sedated":
            continue
        key = (r["condition"], r["cohort"], r["subject_id"])
        v   = float(r["lzc"])
        if np.isfinite(v):
            by_subj[key].append(v)

    subj_vals: dict[str, list[float]] = defaultdict(list)
    for (cond, _cohort, _sid), vs in by_subj.items():
        subj_vals[cond].append(float(np.mean(vs)))

    xi_map: dict[str, int] = {c: i for i, c in enumerate(ANALYSIS_CONDITION_ORDER)}
    for cond in ANALYSIS_CONDITION_ORDER:
        vals = np.array(subj_vals.get(cond, []), dtype=float)
        xi   = xi_map[cond]
        _strip_mean_se(ax, xi, vals, COND_COLORS[cond])

    ax.set_xlim(-0.6, len(ANALYSIS_CONDITION_ORDER) - 0.4)
    ax.set_xticks(range(len(ANALYSIS_CONDITION_ORDER)))
    if show_xtick_labels:
        ax.set_xticklabels(ANALYSIS_CONDITION_ORDER, rotation=30, ha="right")
    else:
        ax.set_xticklabels([])

    # Significance brackets: CNT vs each patient condition
    if pair_stats:
        # Build lookup: frozenset({cond_a, cond_b}) → pvalue_holm
        lookup = {frozenset({r["cond_a"], r["cond_b"]}): float(r["pvalue_holm"])
                  for r in pair_stats if r["subset"] == subset and r["pvalue_holm"] is not None}

        cnt_xi = xi_map.get("CNT")
        if cnt_xi is not None:
            # y_bot: just above the max data in the panel
            ax.autoscale_view()
            ylo, yhi = ax.get_ylim()
            y_bot = yhi + (yhi - ylo) * 0.04
            ax.set_ylim(ylo, yhi * 1.35)   # headroom for brackets

            for other_cond in _PATIENT_CONDITIONS:
                p_adj = lookup.get(frozenset({"CNT", other_cond}), 1.0)
                lbl   = _pvalue_label(p_adj)
                if lbl:
                    other_xi = xi_map.get(other_cond)
                    if other_xi is not None:
                        y_bot = _sig_bracket(ax, other_xi, cnt_xi, y_bot, lbl)


def _panel_sedation(
    ax: "plt.Axes",
    records: list[dict],
    *,
    sed_stats: list[dict] | None = None,
    show_xtick_labels: bool = True,
) -> None:
    """
    Draw one sedation-split panel: non-sedated (solid) vs sedated (lighter)
    within each patient condition, side-by-side.
    """
    from collections import defaultdict
    by_subj: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r["condition"], r["cohort"], r["subject_id"], r["sedation"])
        v   = float(r["lzc"])
        if np.isfinite(v):
            by_subj[key].append(v)

    vals_by_cond_sed: dict[tuple, list[float]] = defaultdict(list)
    for (cond, _c, _s, sed), vs in by_subj.items():
        vals_by_cond_sed[(cond, sed)].append(float(np.mean(vs)))

    OFFSET = 0.18
    xtick_pos, xtick_lbl = [], []
    for xi, cond in enumerate(_PATIENT_CONDITIONS):
        color = COND_COLORS[cond]
        light = _lighten(color, 0.55)
        for offset, sed, col in [(-OFFSET, "non_sedated", color),
                                  (+OFFSET, "sedated",     light)]:
            vals = np.array(vals_by_cond_sed.get((cond, sed), []), dtype=float)
            _strip_mean_se(ax, xi + offset, vals, col, jitter_w=0.10)
        xtick_pos.append(xi)
        xtick_lbl.append(cond)

    ax.set_xlim(-0.6, len(_PATIENT_CONDITIONS) - 0.4)
    ax.set_xticks(xtick_pos)
    if show_xtick_labels:
        ax.set_xticklabels(xtick_lbl, rotation=30, ha="right")
    else:
        ax.set_xticklabels([])

    # Significance brackets: sedated vs non-sedated within each condition
    if sed_stats:
        lookup = {r["condition"]: float(r["pvalue_raw"]) for r in sed_stats}
        ax.autoscale_view()
        ylo, yhi = ax.get_ylim()
        y_bot = yhi + (yhi - ylo) * 0.04
        ax.set_ylim(ylo, yhi * 1.3)
        for xi, cond in enumerate(_PATIENT_CONDITIONS):
            p = lookup.get(cond, 1.0)
            lbl = _pvalue_label(p)
            if lbl:
                y_bot = _sig_bracket(ax, xi - OFFSET, xi + OFFSET, y_bot, lbl)


# ── Figure constructors ───────────────────────────────────────────────────────

def plot_lzc_sweep(
    all_records: list[dict],
    b_tags:      list[str],
    scenarios:   list[str],
    domain:      str,
    domain_label: str,
    pair_stats_lookup: dict[tuple, list[dict]],
    out_path: Path,
    *,
    subset: str = "non_sedated",
) -> None:
    """Rows = b_tags, cols = scenarios. Non-sedated strip+mean±SE + significance."""
    n_rows, n_cols = len(b_tags), len(scenarios)
    panel_w, panel_h = 2.05, 1.75
    legend_w = 1.05
    fig_w = panel_w * n_cols + legend_w
    fig_h = panel_h * n_rows + 0.45

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_w, fig_h),
            sharey=False, sharex=False, squeeze=False,
        )

        for ri, b_tag in enumerate(b_tags):
            for ci, scenario in enumerate(scenarios):
                ax  = axes[ri, ci]
                rec = [r for r in all_records
                       if r["b_tag"] == b_tag
                       and r["scenario"] == scenario
                       and r["domain"] == domain]
                ps  = pair_stats_lookup.get((b_tag, scenario, domain), None)
                _panel_lzc(ax, rec, subset=subset, pair_stats=ps,
                           show_xtick_labels=(ri == n_rows - 1))

                if ri == 0:
                    short = _SCENARIO_SHORT.get(
                        scenario, SCENARIOS.get(scenario, {}).get("label", scenario))
                    ax.set_title(short, fontsize=6.5, fontweight="bold",
                                 pad=3, linespacing=1.3)
                if ci == 0:
                    ax.set_ylabel("LZc")
                else:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)

        # Row labels (b_e value) — placed after tight_layout
        fig.tight_layout(
            rect=[0.0, 0.0, (fig_w - legend_w) / fig_w, 0.93],
            h_pad=0.8, w_pad=0.5,
        )
        for ri, b_tag in enumerate(b_tags):
            pos = axes[ri, 0].get_position()
            fig.text(
                pos.x0 - 0.055, pos.y0 + pos.height / 2,
                f"$b_e$ = {b_tag}", va="center", ha="center",
                fontsize=6, fontweight="bold", rotation=90,
            )

        # Shared condition legend (right of grid)
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
                list(cond_handles.values()), list(cond_handles.keys()),
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
                markerscale=1.8,
            )

        subset_str = "non-sedated" if subset == "non_sedated" else "all subjects"
        fig.suptitle(
            f"LZc — {domain_label}   |   {subset_str}   |   * p<0.05  ** p<0.01  *** p<0.001 (Holm)",
            fontsize=7, fontweight="bold", y=0.98,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[05]  saved → {out_path}")


def plot_sedation_split(
    all_records: list[dict],
    b_tags:      list[str],
    scenarios:   list[str],
    domain:      str,
    domain_label: str,
    sed_stats_lookup: dict[tuple, list[dict]],
    out_path: Path,
) -> None:
    """Rows = b_tags, cols = scenarios. Non-sed (solid) vs sed (lighter) per condition."""
    n_rows, n_cols = len(b_tags), len(scenarios)
    panel_w, panel_h = 2.05, 1.75
    legend_w = 1.25
    fig_w = panel_w * n_cols + legend_w
    fig_h = panel_h * n_rows + 0.45

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_w, fig_h),
            sharey=False, sharex=False, squeeze=False,
        )

        for ri, b_tag in enumerate(b_tags):
            for ci, scenario in enumerate(scenarios):
                ax  = axes[ri, ci]
                rec = [r for r in all_records
                       if r["b_tag"] == b_tag
                       and r["scenario"] == scenario
                       and r["domain"] == domain]
                ss  = sed_stats_lookup.get((b_tag, scenario, domain), None)
                _panel_sedation(ax, rec, sed_stats=ss,
                                show_xtick_labels=(ri == n_rows - 1))

                if ri == 0:
                    short = _SCENARIO_SHORT.get(
                        scenario, SCENARIOS.get(scenario, {}).get("label", scenario))
                    ax.set_title(short, fontsize=6.5, fontweight="bold",
                                 pad=3, linespacing=1.3)
                if ci == 0:
                    ax.set_ylabel("LZc")
                else:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)

        fig.tight_layout(
            rect=[0.0, 0.0, (fig_w - legend_w) / fig_w, 0.93],
            h_pad=0.8, w_pad=0.5,
        )
        for ri, b_tag in enumerate(b_tags):
            pos = axes[ri, 0].get_position()
            fig.text(
                pos.x0 - 0.055, pos.y0 + pos.height / 2,
                f"$b_e$ = {b_tag}", va="center", ha="center",
                fontsize=6, fontweight="bold", rotation=90,
            )

        # Legend: solid = non-sedated, lighter patch = sedated
        from matplotlib.patches import Patch
        example_color = COND_COLORS.get("MCS", "#C5622F")
        legend_elements = [
            Patch(facecolor=example_color,               label="Non-sedated"),
            Patch(facecolor=_lighten(example_color, 0.55), label="Sedated"),
        ]
        fig.legend(
            legend_elements,
            ["Non-sedated", "Sedated"],
            loc="center left",
            bbox_to_anchor=((fig_w - legend_w + 0.05) / fig_w, 0.55),
            title="Sedation",
            title_fontsize=6.5,
            fontsize=6, frameon=False,
            handlelength=1.0, handleheight=0.85,
            handletextpad=0.4, labelspacing=0.4,
        )
        # Also add condition colour patches below
        cond_patches = [
            Patch(facecolor=COND_COLORS[c], label=c)
            for c in _PATIENT_CONDITIONS
        ]
        fig.legend(
            cond_patches, _PATIENT_CONDITIONS,
            loc="center left",
            bbox_to_anchor=((fig_w - legend_w + 0.05) / fig_w, 0.30),
            title="Condition",
            title_fontsize=6.5,
            fontsize=6, frameon=False,
            handlelength=1.0, handleheight=0.85,
            handletextpad=0.4, labelspacing=0.4,
        )

        fig.suptitle(
            f"LZc — {domain_label}   |   sedation effect (patient cohorts)",
            fontsize=7, fontweight="bold", y=0.98,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[05]  saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    known_scenarios  = set(SCENARIOS.keys())
    candidate_scens  = [s for s in (args.scenarios or list(SCENARIOS.keys()))
                        if s in SCENARIOS]

    # ── Detect b_tag level ────────────────────────────────────────────────────
    immediate_subdirs = (
        [d.name for d in sorted(args.sim_root.iterdir()) if d.is_dir()]
        if args.sim_root.exists() else []
    )
    has_b_tag_level = (bool(immediate_subdirs) and
                       not any(d in known_scenarios for d in immediate_subdirs))

    if has_b_tag_level:
        sim_roots = ([args.sim_root / args.b_tag] if args.b_tag
                     else sorted([args.sim_root / d for d in immediate_subdirs]))
        print(f"[05] Detected b_tag layer. Processing: {[r.name for r in sim_roots]}")
    else:
        sim_roots = [args.sim_root]

    # Filter scenarios to those on disk
    scenarios = [s for s in candidate_scens
                 if any((sr / s).is_dir() for sr in sim_roots)]
    missing   = set(candidate_scens) - set(scenarios)
    if missing:
        print(f"[05] Skipping scenarios not on disk: {missing}")
    print(f"[05] Scenarios: {scenarios}")

    # ── Pre-build sedation cache (shared across all b_tags) ──────────────────
    all_npz: list[Path] = []
    for sr in sim_roots:
        found = sorted(sr.glob("*/*/*/seed_*.npz"))
        found = [p for p in found if p.parts[-4] in set(scenarios)]
        all_npz.extend(found)

    if not all_npz:
        print("[05] ERROR: No NPZ files found. Check --sim-root and scenario names.")
        return

    cohort_subject_pairs: set[tuple[str, str]] = set()
    for p in all_npz:
        cohort = p.parts[-3]
        cond = COHORT_TO_CONDITION.get(cohort)
        if cond is None or cond in EXCLUDED_CONDITIONS:
            continue
        cohort_subject_pairs.add((cohort, p.parts[-2]))

    print(f"[05] Building sedation cache for {len(cohort_subject_pairs)} subjects …")
    sedation_cache = build_sedation_cache(args.dataset_root, cohort_subject_pairs)

    # ── Process each b_tag independently ─────────────────────────────────────
    all_records:         list[dict]  = []
    all_omni_rows:       list[dict]  = []
    all_pair_rows:       list[dict]  = []
    all_sed_stat_rows:   list[dict]  = []
    processed_b_tags:    list[str]   = []

    # Lookups for figures (populated after per-b_tag loop)
    pair_stats_lookup: dict[tuple, list[dict]] = {}
    sed_stats_lookup:  dict[tuple, list[dict]] = {}

    for sr in sim_roots:
        b_tag   = sr.name
        out_dir = args.output_dir / b_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n[05] b_tag={b_tag}  →  {out_dir}\n{'='*60}")

        sr_scenarios = [s for s in scenarios if (sr / s).is_dir()]
        if not sr_scenarios:
            print(f"[05]  No scenario dirs under {sr}, skipping.")
            continue

        b_rows: list[dict] = []

        for scenario in sr_scenarios:
            npz_paths = sorted((sr / scenario).glob("*/*/seed_*.npz"))
            print(f"[05]  scenario={scenario}  ({len(npz_paths)} files)")

            for p in npz_paths:
                d = np.load(p, allow_pickle=True)
                scenario_k, cohort, subject_id, seed = parse_sim_npz_path(p, sr)
                cond = COHORT_TO_CONDITION.get(cohort)
                if cond is None or cond in EXCLUDED_CONDITIONS:
                    continue
                sedation = sedation_cache.get((cohort, subject_id), "non_sedated")

                # Rate LZc
                lzc_rate = float("nan")
                if "rate" in d and "time_rate_ms" in d:
                    rate = np.asarray(d["rate"], dtype=float)
                    t_r  = np.asarray(d["time_rate_ms"], dtype=float)
                    if rate.ndim == 2 and rate.shape[0] >= 64:
                        try:
                            lzc_rate = compute_lzc_rate(rate, t_r)
                        except Exception:
                            pass

                # BOLD LZc
                lzc_bold = float("nan")
                if "bold" in d and "time_bold_ms" in d:
                    bold = np.asarray(d["bold"], dtype=float)
                    t_b  = np.asarray(d["time_bold_ms"], dtype=float)
                    if bold.ndim == 2 and bold.shape[0] >= 30:
                        try:
                            lzc_bold = compute_lzc_bold(bold, t_b)
                        except Exception:
                            pass

                base = dict(
                    b_tag=b_tag, scenario=scenario_k,
                    scenario_label=SCENARIOS[scenario_k]["label"],
                    cohort=cohort, condition=cond, subject_id=subject_id,
                    seed=int(seed), sedation=sedation,
                )
                b_rows.append({**base, "domain": "rate", "lzc": float(lzc_rate)})
                b_rows.append({**base, "domain": "bold", "lzc": float(lzc_bold)})

        # Per-b_tag CSV
        write_csv(out_dir / "lzc_subject_rows.csv", b_rows)
        all_records.extend(b_rows)
        if b_rows:
            processed_b_tags.append(b_tag)

        # Per-b_tag statistics
        from collections import defaultdict
        b_omni, b_pair, b_sed = [], [], []

        for domain in ("rate", "bold"):
            for scenario in sr_scenarios:
                rec = [r for r in b_rows
                       if r["scenario"] == scenario and r["domain"] == domain]

                # Seed-average per subject
                by_subj: dict[tuple, list] = defaultdict(list)
                for r in rec:
                    v = float(r["lzc"])
                    if np.isfinite(v):
                        by_subj[(r["condition"], r["cohort"],
                                 r["subject_id"], r["sedation"])].append(v)
                subj_avg = {k: float(np.mean(vs)) for k, vs in by_subj.items()}

                def _group(subset_fn):
                    groups: dict[str, list] = defaultdict(list)
                    for (cond, _c, _s, sed), v in subj_avg.items():
                        if subset_fn(sed):
                            groups[cond].append(v)
                    return {c: np.array(vs) for c, vs in groups.items()}

                for subset_label, fn in [
                    ("all",             lambda s: True),
                    ("non_sedated",     lambda s: s == "non_sedated"),
                    ("sedated_patients", lambda s: s == "sedated"),
                ]:
                    grp = _group(fn)
                    omni, pair = run_omnibus_and_pairwise(grp, subset_label)
                    for row in omni + pair:
                        row.update({"b_tag": b_tag, "scenario": scenario, "domain": domain})
                    b_omni.extend(omni); b_pair.extend(pair)

                # Sedation within cohort
                sed_by_cohort: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
                for (cond, cohort, sid, sed), v in subj_avg.items():
                    if cond != "CNT":   # control has no sedated subjects
                        sed_by_cohort[cohort][sed].append(v)
                sed_rows = run_sedation_stats(
                    {c: {s: np.array(vs) for s, vs in d.items()}
                     for c, d in sed_by_cohort.items()})
                for row in sed_rows:
                    row.update({"b_tag": b_tag, "scenario": scenario, "domain": domain})
                b_sed.extend(sed_rows)

                # Store for figure lookup
                pair_stats_lookup[(b_tag, scenario, domain)] = [
                    r for r in b_pair
                    if r["scenario"] == scenario and r["domain"] == domain
                ]
                sed_stats_lookup[(b_tag, scenario, domain)] = [
                    r for r in b_sed
                    if r["scenario"] == scenario and r["domain"] == domain
                ]

        write_csv(out_dir / "stats" / "lzc_omnibus.csv",        b_omni)
        write_csv(out_dir / "stats" / "lzc_pairwise.csv",       b_pair)
        write_csv(out_dir / "stats" / "lzc_sedation_within.csv", b_sed)
        all_omni_rows.extend(b_omni)
        all_pair_rows.extend(b_pair)
        all_sed_stat_rows.extend(b_sed)
        print(f"[05]  {len(b_rows)//2} subject×domain rows  |  "
              f"{len(b_omni)} omnibus  {len(b_pair)} pairwise  {len(b_sed)} sedation stat rows")

    # ── Combined stats CSVs ───────────────────────────────────────────────────
    figs_dir = args.output_dir / "figs"
    write_csv(args.output_dir / "stats_all_lzc_omnibus.csv",        all_omni_rows)
    write_csv(args.output_dir / "stats_all_lzc_pairwise.csv",       all_pair_rows)
    write_csv(args.output_dir / "stats_all_lzc_sedation_within.csv", all_sed_stat_rows)

    # ── Combined figures ──────────────────────────────────────────────────────
    if all_records and processed_b_tags:
        present_scenarios = [s for s in SCENARIOS
                             if any(r["scenario"] == s for r in all_records)]
        for domain, domain_label in [("rate", "Firing Rates"), ("bold", "BOLD")]:
            # Main: non-sedated comparison
            plot_lzc_sweep(
                all_records, processed_b_tags, present_scenarios,
                domain=domain, domain_label=domain_label,
                pair_stats_lookup=pair_stats_lookup,
                out_path=figs_dir / f"fig_lzc_{domain}_non_sedated_sweep.pdf",
                subset="non_sedated",
            )
            # Sedation split (patient cohorts)
            plot_sedation_split(
                all_records, processed_b_tags, present_scenarios,
                domain=domain, domain_label=domain_label,
                sed_stats_lookup=sed_stats_lookup,
                out_path=figs_dir / f"fig_lzc_{domain}_sedation_split.pdf",
            )

    save_json(
        args.output_dir / "run_manifest.json",
        {
            "script":          "05_lzc_analysis_pub.py",
            "sim_root":        str(args.sim_root),
            "b_tags":          processed_b_tags,
            "scenarios":       scenarios,
            "excluded_conditions": sorted(EXCLUDED_CONDITIONS),
            "lzc_shuffle_seed": 0,
            "rate_band_hz":    list(RATE_BAND_HZ),
            "bold_band_hz":    list(BOLD_BAND_HZ),
            "rate_crop_ms":    RATE_CROP_MS,
            "rate_filter_order": RATE_FILT_ORD,
            "bold_filter_order": BOLD_FILT_ORD,
        },
    )

    print(f"\n[05] done. outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
