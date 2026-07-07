#!/usr/bin/env python3
"""Three-way PCI comparison: empirical stored vs the two binarisation routes.

For every ``Droutine_<PRE|POST>_PCI.mat`` we compare three PCI values:

1. **stored**  — the empirical PCI saved by the original (Casali) pipeline.
2. **casali**  — recomputed by us, binarising the continuous source signal ``J``
   via the paper-faithful bootstrap route.
3. **tvbsim**  — recomputed by us, binarising the same ``J`` via the existing
   shuffle route.

This lets you gauge, per subject, how well each of your routes aligns with the
empirical values before wiring a route through the simulation pipeline.

Output (under ``results/stim_data/route_compare/``):
- ``route_compare_table.csv``
- ``route_compare_summary.json``
- ``route_compare.png``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.datasets.stim_pci import (  # noqa: E402
    compute_route_pci,
    discover_pci_files,
)
from tvbtoolkit.core.paths import stimulation_raw, stimulation_results  # noqa: E402

ROUTE_COLOR = {"stored": "#444444", "casali": "#0e9aa7", "tvbsim": "#d1495b"}
PCI_CUTOFF = 0.31


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def build_table(
    eeg_root: Path,
    primary_only: bool,
    n_bootstrap: int,
    single_trial_strategy: str | None,
) -> pd.DataFrame:
    records = discover_pci_files(eeg_root)
    if primary_only:
        records = [r for r in records if r.is_primary]
    rows = []
    for i, rec in enumerate(records, 1):
        print(f"  [{i:2d}/{len(records)}] {rec.subject:6s} {rec.condition:4s} "
              f"S{rec.session} ({rec.variant})", flush=True)
        row = compute_route_pci(
            rec,
            n_bootstrap=n_bootstrap,
            single_trial_strategy=single_trial_strategy,
        )
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["subject", "condition", "session"]).reset_index(drop=True)


def make_figure(
    df: pd.DataFrame,
    out_path: Path,
    *,
    single_trial_strategy: str | None = None,
) -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })
    prim = df[df["is_primary"]] if "is_primary" in df else df
    subj_means = prim.groupby("subject")[
        ["stored_pci", "casali_pci", "tvbsim_pci"]
    ].mean()
    subjects = list(subj_means.index)

    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.24)
    axA = fig.add_subplot(gs[0, :])
    axB, axC = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- A: grouped bars per subject (stored / casali / tvbsim) ---------------
    x = np.arange(len(subjects))
    w = 0.26
    for j, route in enumerate(["stored", "casali", "tvbsim"]):
        vals = subj_means[f"{route}_pci"]
        axA.bar(x + (j - 1) * w, vals, w, label=route,
                color=ROUTE_COLOR[route], edgecolor="black", linewidth=0.4,
                alpha=1.0 if np.isfinite(vals).any() else 0.25)
    axA.axhline(PCI_CUTOFF, ls="--", color="0.4", lw=1, zorder=1)
    axA.set_xticks(x)
    axA.set_xticklabels(subjects)
    axA.set_ylabel("PCI (subject mean)")
    axA.set_title("A  PCI per subject — empirical vs both routes")
    axA.legend(title="route", framealpha=0.9)

    # ---- B: alignment scatter (stored vs each route, all files) ---------------
    all_vals = _finite(df[["stored_pci", "casali_pci", "tvbsim_pci"]].to_numpy().ravel())
    lo = float(all_vals.min()) - 0.03
    hi = float(all_vals.max()) + 0.03
    axB.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1, zorder=1)
    for route in ["casali", "tvbsim"]:
        d = df[np.isfinite(df[f"{route}_pci"])]
        if len(d):
            axB.scatter(d["stored_pci"], d[f"{route}_pci"], s=42, color=ROUTE_COLOR[route],
                        edgecolor="black", linewidth=0.4, alpha=0.85, label=route, zorder=3)
    axB.set(xlim=(lo, hi), ylim=(lo, hi),
            xlabel="stored (empirical) PCI", ylabel="recomputed PCI")
    axB.set_title("B  Alignment vs empirical (per file)")
    # mean absolute deviation from stored, per route
    mad = {}
    for r in ["casali", "tvbsim"]:
        err = (df[f"{r}_pci"] - df["stored_pci"]).abs()
        mad[r] = float(err.mean()) if np.isfinite(err).any() else float("nan")
    casali_mad = f"{mad['casali']:.3f}" if np.isfinite(mad["casali"]) else "unavailable"
    axB.text(0.04, 0.96, f"mean |Δ vs stored|\n casali = {casali_mad}",
             transform=axB.transAxes, va="top", ha="left", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axB.text(0.04, 0.80, f" tvbsim = {mad['tvbsim']:.3f}",
             transform=axB.transAxes, va="top", ha="left", fontsize=9)
    axB.legend(framealpha=0.9, loc="lower right")

    # ---- C: signed error distribution per route -------------------------------
    err_ca = _finite(df["casali_pci"] - df["stored_pci"])
    err_tv = _finite(df["tvbsim_pci"] - df["stored_pci"])
    err_data = []
    err_labels = []
    for route, err in [("casali", err_ca), ("tvbsim", err_tv)]:
        if err.size:
            err_data.append(err)
            err_labels.append(route)
    if err_data:
        parts = axC.boxplot(err_data, labels=err_labels, patch_artist=True,
                            widths=0.5, showmeans=True)
        for patch, route in zip(parts["boxes"], err_labels):
            patch.set_facecolor(ROUTE_COLOR[route]); patch.set_alpha(0.65)
    else:
        axC.text(0.5, 0.5, "No finite route errors", transform=axC.transAxes,
                 ha="center", va="center")
    axC.axhline(0, ls="--", color="0.4", lw=1)
    axC.set_ylabel("PCI error (route − stored)")
    axC.set_title("C  Signed deviation from empirical")

    title = "PCI route comparison — empirical vs casali vs tvbsim binarisation"
    if single_trial_strategy is not None:
        title += f" ({single_trial_strategy} sensitivity)"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-root", type=Path,
                    default=stimulation_raw("stim_data", "tdcs-eeg"))
    ap.add_argument("--out-dir", type=Path,
                    default=stimulation_results("stim_data", "route_compare"))
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument(
        "--single-trial-strategy",
        choices=["baseline_resample"],
        default=None,
        help=(
            "Optional non-canonical Casali fallback for averaged J traces. "
            "Use only for sensitivity plots when trial-level baseline+post data "
            "are unavailable."
        ),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing three-way PCI (empirical / casali / tvbsim) ...")
    df = build_table(
        args.eeg_root,
        args.primary_only,
        args.n_bootstrap,
        args.single_trial_strategy,
    )
    df.to_csv(args.out_dir / "route_compare_table.csv", index=False)

    prim = df[df["is_primary"]]
    summary = {
        "n_files": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "single_trial_strategy": args.single_trial_strategy,
        "mean_abs_dev_from_stored": {
            r: (
                float((df[f"{r}_pci"] - df["stored_pci"]).abs().mean())
                if np.isfinite(df[f"{r}_pci"]).any()
                else None
            )
            for r in ["casali", "tvbsim"]
        },
        "casali_status_counts": (
            df["casali_status"].value_counts(dropna=False).to_dict()
            if "casali_status" in df else {}
        ),
        "subject_means": prim.groupby("subject")[
            ["stored_pci", "casali_pci", "tvbsim_pci"]
        ].mean().round(4).to_dict(orient="index"),
    }
    (args.out_dir / "route_compare_summary.json").write_text(json.dumps(summary, indent=2))
    make_figure(
        df,
        args.out_dir / "route_compare.png",
        single_trial_strategy=args.single_trial_strategy,
    )

    md = summary["mean_abs_dev_from_stored"]
    print(f"\n  files: {summary['n_files']}  subjects: {summary['n_subjects']}")
    ca_txt = f"{md['casali']:.3f}" if md["casali"] is not None else "unavailable"
    tv_txt = f"{md['tvbsim']:.3f}" if md["tvbsim"] is not None else "unavailable"
    print(f"  mean |Δ vs empirical|:  casali = {ca_txt}   tvbsim = {tv_txt}")
    if summary["casali_status_counts"]:
        print(f"  casali statuses       : {summary['casali_status_counts']}")
    if args.single_trial_strategy is not None:
        print(f"  sensitivity mode      : {args.single_trial_strategy} (non-canonical)")
    print("\n  per-subject mean PCI (stored | casali | tvbsim):")
    for s, v in summary["subject_means"].items():
        print(f"    {s:6s}  {v['stored_pci']:.3f} | {v['casali_pci']:.3f} | {v['tvbsim_pci']:.3f}")
    print(f"\n  wrote: {args.out_dir / 'route_compare.png'}")
    print(f"  wrote: {args.out_dir / 'route_compare_table.csv'}")


if __name__ == "__main__":
    main()
