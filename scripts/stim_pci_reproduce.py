#!/usr/bin/env python3
"""Reproduce and validate source-space PCI for the tDCS/TMS-EEG dataset.

Deliverable (validated anchor for downstream complexity/entropy/emergence work):

1. Discover every ``Droutine_<PRE|POST>_PCI.mat`` under ``data/stim_data/tdcs-eeg``.
2. Recompute the PCI scalar from the stored ``binJ`` with the toolkit's own
   Casali engine and check it against the value the original pipeline saved.
3. Emit a tidy per-subject / per-session table (PCI pre- vs post-tDCS) and a
   composite validation + pre/post figure.

Outputs (under ``results/stim_data/pci_reproduction/``):
- ``pci_reproduction_table.csv``    : one row per file (stored vs reproduced).
- ``pci_reproduction_summary.json`` : reproduction accuracy + pre/post summary.
- ``pci_reproduction.png``          : composite figure.

Usage
-----
    python scripts/stim_pci_reproduce.py
    python scripts/stim_pci_reproduce.py --primary-only
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
    reproduce_all,
    reproduction_to_dict,
)

PCI_CUTOFF = 0.31  # Casali et al. (2013) consciousness-compatible threshold.
COND_ORDER = ["pre", "post"]
COND_LABEL = {"pre": "pre-tDCS", "post": "post-tDCS"}
COND_COLOR = {"pre": "#d1495b", "post": "#0e9aa7"}  # red / teal (paper palette)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def build_table(eeg_root: Path, primary_only: bool) -> pd.DataFrame:
    reps = reproduce_all(eeg_root, primary_only=primary_only, progress=True)
    if not reps:
        raise SystemExit(f"No PCI files found under {eeg_root}")
    df = pd.DataFrame(reproduction_to_dict(r) for r in reps)
    df["condition"] = pd.Categorical(df["condition"], categories=COND_ORDER, ordered=True)
    return df.sort_values(["subject", "condition", "session"]).reset_index(drop=True)


def _subject_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean reproduced PCI per subject x condition over (primary) sessions."""
    prim = df[df["is_primary"]]
    return (
        prim.groupby(["subject", "condition"], observed=True)["pci_repro"]
        .mean()
        .unstack("condition")
        .reindex(columns=COND_ORDER)
    )


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    _set_style()
    subjects = sorted(df["subject"].unique())
    cmap = plt.get_cmap("tab10")
    subj_color = {s: cmap(i % 10) for i, s in enumerate(subjects)}

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- A: reproduction validation (stored vs recomputed PCI) ----------------
    for s in subjects:
        d = df[df["subject"] == s]
        axA.scatter(d["stored_pci"], d["pci_repro"], s=46, color=subj_color[s],
                    edgecolor="black", linewidth=0.4, label=s, zorder=3)
    lo = float(min(df["stored_pci"].min(), df["pci_repro"].min())) - 0.02
    hi = float(max(df["stored_pci"].max(), df["pci_repro"].max())) + 0.02
    axA.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1, zorder=1, label="identity")
    axA.set(xlim=(lo, hi), ylim=(lo, hi),
            xlabel="stored PCI (original pipeline)", ylabel="reproduced PCI (toolkit)")
    axA.set_title("A  Reproduction validation")
    max_rel = float(df["rel_err"].max()) * 100
    r2 = float(np.corrcoef(df["stored_pci"], df["pci_repro"])[0, 1] ** 2)
    axA.text(0.04, 0.96, f"n = {len(df)} files\nR² = {r2:.4f}\nmax rel. err = {max_rel:.2f}%",
             transform=axA.transAxes, va="top", ha="left", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axA.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)

    # ---- B: paired pre -> post PCI per subject --------------------------------
    means = _subject_means(df)
    x = {"pre": 0, "post": 1}
    for s, row in means.iterrows():
        pre, post = row.get("pre"), row.get("post")
        axB.plot([x["pre"], x["post"]], [pre, post], "-", color=subj_color[s],
                 lw=1.6, marker="o", ms=7, mec="black", mew=0.4, label=s, zorder=3)
    axB.axhline(PCI_CUTOFF, ls="--", color="0.35", lw=1.2, zorder=1)
    axB.text(1.02, PCI_CUTOFF, f"  cutoff {PCI_CUTOFF}", va="center", fontsize=8, color="0.35")
    axB.set_xticks([0, 1])
    axB.set_xticklabels([COND_LABEL["pre"], COND_LABEL["post"]])
    axB.set_xlim(-0.35, 1.35)
    axB.set_ylabel("PCI (subject mean over sessions)")
    axB.set_title("B  PCI pre- vs post-tDCS")
    axB.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)

    # ---- C: distribution of all (primary) session PCI by condition ------------
    prim = df[df["is_primary"]]
    for i, cond in enumerate(COND_ORDER):
        vals = prim.loc[prim["condition"] == cond, "pci_repro"].to_numpy()
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.18
        axC.scatter(np.full(len(vals), i) + jitter, vals, s=40,
                    color=COND_COLOR[cond], edgecolor="black", linewidth=0.4, zorder=3)
        if len(vals):
            axC.hlines(vals.mean(), i - 0.22, i + 0.22, color="black", lw=2, zorder=4)
    axC.axhline(PCI_CUTOFF, ls="--", color="0.35", lw=1.2, zorder=1)
    axC.set_xticks([0, 1])
    axC.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    axC.set_xlim(-0.5, 1.5)
    axC.set_ylabel("PCI (per session)")
    axC.set_title("C  Per-session PCI by condition")

    # ---- D: reproduction error per file ---------------------------------------
    # Stored vs reproduced PCI side-by-side, one row per file (sorted by stored).
    d = df.sort_values("stored_pci").reset_index(drop=True)
    y = np.arange(len(d))
    axD.hlines(y, d["stored_pci"], d["pci_repro"], color="0.6", lw=1.0, zorder=1)
    axD.scatter(d["stored_pci"], y, s=44, marker="o", facecolor="white",
                edgecolor="black", linewidth=1.1, zorder=3, label="stored (original pipeline)")
    axD.scatter(d["pci_repro"], y, s=30, marker="D",
                color=[COND_COLOR[c] for c in d["condition"]],
                edgecolor="black", linewidth=0.4, zorder=4, label="reproduced (toolkit)")
    axD.axvline(PCI_CUTOFF, ls="--", color="0.35", lw=1.0, zorder=1)
    axD.set_yticks(y)
    axD.set_yticklabels([f"{r.subject} {r.condition} S{r.session}"
                         for r in d.itertuples()], fontsize=6.5)
    axD.set_ylim(-0.7, len(d) - 0.3)
    axD.set_xlabel("PCI value")
    axD.set_title("D  Stored vs reproduced PCI (per file)")
    axD.grid(axis="y", alpha=0)
    axD.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    he = (df["repro_H"] - df["stored_H"]).abs().max()
    ne = (df["repro_norm"] - df["stored_norm"]).abs().max()
    mre = df["rel_err"].max() * 100
    axD.text(0.03, 0.97,
             f"max rel. err = {mre:.2f}%\nmax |ΔH| = {he:.0e}\nmax |ΔNorm| = {ne:.0e}",
             transform=axD.transAxes, va="top", ha="left", fontsize=7.5,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle("tDCS/TMS-EEG source-space PCI — reproduction & pre/post anchor",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarise(df: pd.DataFrame) -> dict:
    means = _subject_means(df)
    delta = (means["post"] - means["pre"]).dropna()
    return {
        "n_files": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "subjects": sorted(df["subject"].unique()),
        "reproduction": {
            "pci_max_rel_err": float(df["rel_err"].max()),
            "pci_mean_rel_err": float(df["rel_err"].mean()),
            "H_max_abs_err": float((df["repro_H"] - df["stored_H"]).abs().max()),
            "norm_max_abs_err": float((df["repro_norm"] - df["stored_norm"]).abs().max()),
        },
        "pre_post": {
            "subject_pre_mean": means["pre"].dropna().round(4).to_dict(),
            "subject_post_mean": means["post"].dropna().round(4).to_dict(),
            "subject_delta_post_minus_pre": delta.round(4).to_dict(),
            "n_subjects_with_both": int(delta.shape[0]),
            "mean_delta": float(delta.mean()) if len(delta) else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-root", type=Path,
                    default=_REPO_ROOT / "data/stim_data/tdcs-eeg")
    ap.add_argument("--out-dir", type=Path,
                    default=_REPO_ROOT / "results/stim_data/pci_reproduction")
    ap.add_argument("--primary-only", action="store_true",
                    help="Exclude reanalysis/test lineages (Second_Analysis, Test NotMNI).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Discovering + reproducing PCI under {args.eeg_root} ...")
    df = build_table(args.eeg_root, primary_only=args.primary_only)

    table_path = args.out_dir / "pci_reproduction_table.csv"
    df.to_csv(table_path, index=False)

    summary = summarise(df)
    (args.out_dir / "pci_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    fig_path = args.out_dir / "pci_reproduction.png"
    make_figure(df, fig_path)

    print(f"\n  files reproduced : {summary['n_files']}")
    print(f"  subjects         : {summary['n_subjects']} ({', '.join(summary['subjects'])})")
    rep = summary["reproduction"]
    print(f"  PCI max rel err  : {rep['pci_max_rel_err'] * 100:.2f}%  "
          f"(mean {rep['pci_mean_rel_err'] * 100:.2f}%)")
    print(f"  H / Norm match   : |ΔH|<={rep['H_max_abs_err']:.1e}, "
          f"|ΔNorm|<={rep['norm_max_abs_err']:.1e}")
    print(f"\n  wrote: {table_path}")
    print(f"  wrote: {fig_path}")
    print(f"  wrote: {args.out_dir / 'pci_reproduction_summary.json'}")


if __name__ == "__main__":
    main()
