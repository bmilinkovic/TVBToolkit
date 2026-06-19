#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from brain_act_hybrid_common import (
    COHORT_TO_CONDITION,
    CONDITION_ORDER,
    COND_COLORS,
    PROJECT_ROOT,
    SCENARIOS,
    save_json,
)

from tvbtoolkit.complexity.measures import pci_casali_like_multi_trial


EXCLUDED_CONDITIONS = {"COMA"}
ANALYSIS_CONDITION_ORDER = [c for c in CONDITION_ORDER if c not in EXCLUDED_CONDITIONS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Publication-ready PCI analysis from real trial outputs (sims_pci) using "
            "multi-trial Casali PCI with TVBSim-style parameters."
        )
    )
    p.add_argument("--sim-pci-root", type=Path, default=PROJECT_ROOT / "notebooks" / "outputs" / "ba_sim_hybrid" / "shared_b" / "sims_pci")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "notebooks" / "outputs" / "06_pci_analysis_pub")
    p.add_argument("--scenario", action="append", dest="scenarios", default=None)
    p.add_argument("--b-tag", type=str, default=None,
                   help="Optional b-tag level to process, e.g. b035 or condb_doc_gradient.")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--min-trials", type=int, default=100)
    p.add_argument("--t-analysis-ms", type=float, default=300.0)
    p.add_argument("--dt-ms-default", type=float, default=7.8125)
    p.add_argument("--nshuffles", type=int, default=10)
    p.add_argument("--percentile", type=float, default=100.0)
    p.add_argument("--shuffle-seed", type=int, default=0,
                   help="Base seed for deterministic PCI binarization shuffles.")
    return p.parse_args()



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



def load_trial_windows(trial_paths: list[Path], t_analysis_ms: float, dt_ms_default: float) -> tuple[list[np.ndarray] | None, int | None, float | None]:
    windows: list[np.ndarray] = []
    nbins_ref: int | None = None
    dt_ref: float | None = None

    for path in trial_paths:
        d = np.load(path, allow_pickle=True)
        if "rate" not in d or "time_ms" not in d:
            return None, None, None

        x = np.asarray(d["rate"], dtype=float)
        t = np.asarray(d["time_ms"], dtype=float)
        if x.ndim != 2 or t.ndim != 1 or x.shape[0] != t.shape[0] or x.shape[0] < 10:
            return None, None, None

        stim_ms = float(d["stim_onset_ms"][0]) if "stim_onset_ms" in d else float(t_analysis_ms)
        t_ana = float(d["t_analysis_ms"][0]) if "t_analysis_ms" in d else float(t_analysis_ms)

        dt_ms = float(np.median(np.diff(t))) if t.size > 1 else float(dt_ms_default)
        nbins = int(round(t_ana / max(dt_ms, 1e-9)))
        if nbins < 2:
            return None, None, None

        if nbins_ref is None:
            nbins_ref = nbins
            dt_ref = dt_ms
        elif nbins != nbins_ref:
            return None, None, None

        stim_idx = int(round((stim_ms - t[0]) / max(dt_ms, 1e-9)))
        i0 = stim_idx - nbins
        i1 = stim_idx + nbins
        if i0 < 0 or i1 > x.shape[0]:
            return None, None, None

        win = x[i0:i1, :]
        windows.append(win.T)  # (n_regions, 2*nbins)

    return windows, nbins_ref, dt_ref



def _stable_seed(*parts: object, base_seed: int = 0) -> int:
    text = "|".join(str(p) for p in parts)
    return (int(base_seed) + zlib.crc32(text.encode("utf-8"))) % (2**32)


def plot_pci(rows: list[dict[str, Any]], scenarios: list[str], out_path: Path, *, b_tag: str | None = None) -> None:
    n_cols = len(scenarios)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        r_s = [r for r in rows if r["scenario"] == scenario and (b_tag is None or r["b_tag"] == b_tag)]
        for xi, cond in enumerate(ANALYSIS_CONDITION_ORDER):
            vals = np.array([float(r["pci_mean"]) for r in r_s if r["condition"] == cond], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(max(vals.size, 1))) if vals.size > 1 else 0.0
            ax.bar(xi, mean, color=COND_COLORS[cond], alpha=0.75, edgecolor="black", linewidth=0.5)
            ax.errorbar(xi, mean, yerr=se, fmt="none", ecolor="black", capsize=3, lw=1.0)

            jitter = np.random.default_rng(0).uniform(-0.13, 0.13, size=vals.size)
            ax.scatter(np.full(vals.size, xi) + jitter, vals, s=8, color="black", alpha=0.6)

        ax.set_title(SCENARIOS.get(scenario, {}).get("label", scenario), fontsize=8)
        ax.set_xticks(range(len(ANALYSIS_CONDITION_ORDER)))
        ax.set_xticklabels(ANALYSIS_CONDITION_ORDER, rotation=35, ha="right", fontsize=7)
        ax.grid(alpha=0.2, axis="y")

    axes[0].set_ylabel("PCI")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    known_scenarios = set(SCENARIOS.keys())
    immediate_subdirs = (
        [d.name for d in sorted(args.sim_pci_root.iterdir()) if d.is_dir()]
        if args.sim_pci_root.exists() else []
    )
    has_b_tag_level = bool(immediate_subdirs) and not any(d in known_scenarios for d in immediate_subdirs)

    if has_b_tag_level:
        sim_roots = ([args.sim_pci_root / args.b_tag] if args.b_tag
                     else sorted([args.sim_pci_root / d for d in immediate_subdirs]))
    else:
        sim_roots = [args.sim_pci_root]

    candidate_scenarios = [s for s in (args.scenarios or list(SCENARIOS.keys())) if s in SCENARIOS]
    scenarios = [
        s for s in candidate_scenarios
        if any((sr / s).is_dir() for sr in sim_roots)
    ]

    rows: list[dict[str, Any]] = []

    for sr in sim_roots:
        b_tag = sr.name if has_b_tag_level else "single"
        for scenario in scenarios:
            scenario_dir = sr / scenario
            if not scenario_dir.exists():
                continue

            for cohort_dir in sorted([p for p in scenario_dir.iterdir() if p.is_dir()]):
                cohort = cohort_dir.name
                cond = COHORT_TO_CONDITION.get(cohort)
                if cond is None or cond in EXCLUDED_CONDITIONS:
                    continue

                for subj_dir in sorted([p for p in cohort_dir.iterdir() if p.is_dir()]):
                    subject_id = subj_dir.name
                    trial_paths = sorted(subj_dir.glob("trial_*.npz"))[: int(args.n_trials)]
                    if len(trial_paths) < int(args.min_trials):
                        continue

                    windows, nbins, dt_ms = load_trial_windows(
                        trial_paths=trial_paths,
                        t_analysis_ms=float(args.t_analysis_ms),
                        dt_ms_default=float(args.dt_ms_default),
                    )
                    if windows is None or nbins is None or dt_ms is None:
                        continue

                    try:
                        np.random.seed(_stable_seed(b_tag, scenario, cohort, subject_id, base_seed=int(args.shuffle_seed)))
                        pci_mean, pci_trials = pci_casali_like_multi_trial(
                            windows,
                            stimulation_index=int(nbins),
                            t_analysis_ms=float(args.t_analysis_ms),
                            dt_ms=float(dt_ms),
                            nshuffles=int(args.nshuffles),
                            percentile=float(args.percentile),
                        )
                    except Exception:
                        continue

                    rows.append(
                        {
                            "b_tag": b_tag,
                            "scenario": scenario,
                            "scenario_label": SCENARIOS[scenario]["label"],
                            "cohort": cohort,
                            "condition": cond,
                            "subject_id": subject_id,
                            "n_trials_used": int(len(windows)),
                            "dt_ms": float(dt_ms),
                            "pci_mean": float(pci_mean),
                            "pci_trials_mean": float(np.nanmean(np.asarray(pci_trials, dtype=float))),
                            "pci_trials_std": float(np.nanstd(np.asarray(pci_trials, dtype=float))),
                        }
                    )

    table_path = args.output_dir / "pci_subject_rows.csv"
    write_csv(table_path, rows)

    fig_dir = args.output_dir / "figs"
    b_tags = sorted({r["b_tag"] for r in rows})
    fig_paths: list[str] = []
    for b_tag in b_tags:
        fig_path = fig_dir / f"fig01_pci_by_condition_scenario_{b_tag}.pdf"
        plot_pci(rows, scenarios, fig_path, b_tag=b_tag)
        fig_paths.append(str(fig_path))

    save_json(
        args.output_dir / "run_manifest.json",
        {
            "script": "06_pci_analysis_pub.py",
            "sim_pci_root": str(args.sim_pci_root),
            "b_tags": b_tags,
            "scenarios": scenarios,
            "excluded_conditions": sorted(EXCLUDED_CONDITIONS),
            "n_trials": int(args.n_trials),
            "min_trials": int(args.min_trials),
            "t_analysis_ms": float(args.t_analysis_ms),
            "nshuffles": int(args.nshuffles),
            "percentile": float(args.percentile),
            "shuffle_seed": int(args.shuffle_seed),
            "outputs": {
                "table_csv": str(table_path),
                "fig_pdfs": fig_paths,
            },
        },
    )

    print(f"[06] done. outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
