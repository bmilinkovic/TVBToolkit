#!/usr/bin/env python3
"""Compare stored PCI, TVBSim PCI, and reconstructed vertex-level Casali PCI.

This script reconstructs source-level trials from the D30 products:

    AllTf[:, :, trial] @ sensor_trial

where sensor trials are read from ``Droutine_<COND>.dat``. It then computes a
Casali-style bootstrap PCI on the reconstructed vertex-time data.

The reconstruction is expensive. Start with ``--n-bootstrap 20`` for a quick
audit, then increase once the route looks stable.
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
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/mpl").resolve()))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.datasets.stim_pci import (  # noqa: E402
    compute_reconstructed_vertex_casali,
    compute_route_pci,
    discover_pci_files,
)
from tvbtoolkit.core.paths import stimulation_raw, stimulation_results  # noqa: E402

ROUTE_COLOR = {
    "stored": "#444444",
    "tvbsim": "#d1495b",
    "reconstructed": "#0e9aa7",
}


def build_table(
    eeg_root: Path,
    *,
    primary_only: bool,
    n_bootstrap: int,
    max_files: int | None,
    work_dir: Path,
) -> pd.DataFrame:
    records = discover_pci_files(eeg_root)
    if primary_only:
        records = [r for r in records if r.is_primary]
    if max_files is not None:
        records = records[:max_files]

    rows = []
    for i, rec in enumerate(records, 1):
        print(
            f"  [{i:2d}/{len(records)}] {rec.subject:6s} {rec.condition:4s} "
            f"S{rec.session} ({rec.variant})",
            flush=True,
        )
        route = compute_route_pci(rec)
        recon = compute_reconstructed_vertex_casali(
            rec,
            n_bootstrap=n_bootstrap,
            work_dir=work_dir,
        )
        rows.append(
            {
                **recon,
                "tvbsim_pci": route["tvbsim_pci"] if route is not None else np.nan,
                "tvbsim_active_frac": (
                    route["tvbsim_active_frac"] if route is not None else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["subject", "condition", "session"])


def make_figure(df: pd.DataFrame, out_path: Path, *, n_bootstrap: int) -> None:
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
    prim = df[df["is_primary"]] if "is_primary" in df else df
    means = prim.groupby("subject")[
        ["stored_pci", "tvbsim_pci", "reconstructed_casali_pci_sorted"]
    ].mean()
    subjects = list(means.index)

    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.26)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    x = np.arange(len(subjects))
    width = 0.26
    bars = [
        ("stored", "stored_pci", "stored"),
        ("reconstructed", "reconstructed_casali_pci_sorted", "reconstructed Casali"),
        ("tvbsim", "tvbsim_pci", "TVBSim"),
    ]
    for j, (key, col, label) in enumerate(bars):
        axA.bar(
            x + (j - 1) * width,
            means[col],
            width,
            label=label,
            color=ROUTE_COLOR[key],
            edgecolor="black",
            linewidth=0.4,
        )
    axA.set_xticks(x)
    axA.set_xticklabels(subjects)
    axA.set_ylabel("PCI (subject mean)")
    axA.set_title("A  Stored vs reconstructed vertex Casali vs TVBSim")
    axA.legend(framealpha=0.9)

    lo = float(
        df[["stored_pci", "tvbsim_pci", "reconstructed_casali_pci_sorted"]]
        .min()
        .min()
    ) - 0.03
    hi = float(
        df[["stored_pci", "tvbsim_pci", "reconstructed_casali_pci_sorted"]]
        .max()
        .max()
    ) + 0.03
    axB.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1)
    axB.scatter(
        df["stored_pci"],
        df["reconstructed_casali_pci_sorted"],
        s=42,
        color=ROUTE_COLOR["reconstructed"],
        edgecolor="black",
        linewidth=0.4,
        label="reconstructed Casali",
    )
    axB.scatter(
        df["stored_pci"],
        df["tvbsim_pci"],
        s=42,
        color=ROUTE_COLOR["tvbsim"],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
        label="TVBSim",
    )
    axB.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="stored PCI", ylabel="computed PCI")
    axB.set_title("B  Alignment with stored PCI")
    axB.legend(framealpha=0.9)

    err_recon = df["reconstructed_casali_pci_sorted"] - df["stored_pci"]
    err_tv = df["tvbsim_pci"] - df["stored_pci"]
    parts = axC.boxplot(
        [err_recon, err_tv],
        labels=["reconstructed\nCasali", "TVBSim"],
        patch_artist=True,
        widths=0.5,
        showmeans=True,
    )
    for patch, key in zip(parts["boxes"], ["reconstructed", "tvbsim"]):
        patch.set_facecolor(ROUTE_COLOR[key])
        patch.set_alpha(0.65)
    axC.axhline(0, ls="--", color="0.4", lw=1)
    axC.set_ylabel("PCI error (computed - stored)")
    axC.set_title("C  Signed error")

    mad_recon = float(err_recon.abs().mean())
    mad_tv = float(err_tv.abs().mean())
    fig.suptitle(
        "Vertex-level PCI reconstructed from D30 sensor trials "
        f"(n_bootstrap={n_bootstrap}; mean |err| Casali={mad_recon:.3f}, TVBSim={mad_tv:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-root", type=Path, default=stimulation_raw("stim_data", "tdcs-eeg"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=stimulation_results("stim_data", "reconstructed_vertex_compare"),
    )
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=20)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--work-dir", type=Path, default=Path("/private/tmp"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_table(
        args.eeg_root,
        primary_only=args.primary_only,
        n_bootstrap=args.n_bootstrap,
        max_files=args.max_files,
        work_dir=args.work_dir,
    )
    table_path = args.out_dir / "reconstructed_vertex_compare_table.csv"
    df.to_csv(table_path, index=False)

    summary = {
        "n_files": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_bootstrap": int(args.n_bootstrap),
        "mean_abs_dev_from_stored": {
            "reconstructed_casali_sorted": float(
                (df["reconstructed_casali_pci_sorted"] - df["stored_pci"]).abs().mean()
            ),
            "tvbsim": float((df["tvbsim_pci"] - df["stored_pci"]).abs().mean()),
        },
        "subject_means": df.groupby("subject")[
            ["stored_pci", "reconstructed_casali_pci_sorted", "tvbsim_pci"]
        ].mean().round(4).to_dict(orient="index"),
    }
    (args.out_dir / "reconstructed_vertex_compare_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    make_figure(df, args.out_dir / "reconstructed_vertex_compare.png", n_bootstrap=args.n_bootstrap)

    md = summary["mean_abs_dev_from_stored"]
    print(f"\n  files: {summary['n_files']}  subjects: {summary['n_subjects']}")
    print(
        "  mean |Δ vs stored|: "
        f"reconstructed Casali = {md['reconstructed_casali_sorted']:.3f}   "
        f"TVBSim = {md['tvbsim']:.3f}"
    )
    print(f"\n  wrote: {table_path}")
    print(f"  wrote: {args.out_dir / 'reconstructed_vertex_compare.png'}")


if __name__ == "__main__":
    main()
