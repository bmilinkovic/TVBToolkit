#!/usr/bin/env python3
"""Compare the two PCI binarisation routes on a controlled signal.

Motivation
----------
Empirical ``binJ`` matrices in the tDCS/TMS-EEG dataset were produced with the
canonical Casali (2013) binarisation (bootstrap max-statistic, two-sided,
trial-averaged). The toolkit's simulation PCI (``pci_casali_like``) historically
used a different shuffle-based route (``binarise_signals``). This script shows,
on a signal with a **known** sparse evoked response (a stand-in for a
simulation), how the two routes differ in:

- recovered active fraction vs the injected ground truth,
- detection precision/recall,
- the resulting PCI value.

Both routes are now selectable via ``binarise(method=...)`` /
``pci_casali_like(binarise_method=...)``.

Output (under ``results/stim_data/binarisation_compare/``):
- ``binarisation_compare.png``      : composite figure.
- ``binarisation_compare.json``     : numeric summary.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tvbtoolkit.complexity.pci_casali import (  # noqa: E402
    binarise_signals,
    binarise_signals_casali,
    lz_complexity_2d,
    pci_norm_factor,
    sort_binJ,
)
from tvbtoolkit.core.paths import stimulation_results  # noqa: E402

ROUTE_COLOR = {"ground truth": "#444444", "tvbsim": "#d1495b", "casali": "#0e9aa7"}


def make_synthetic(
    *,
    n_trials: int = 30,
    n_sources: int = 300,
    n_bins: int = 200,
    onset: int = 100,
    baseline_offset: float = 10.0,
    noise_sd: float = 1.0,
    evoked_frac: float = 0.06,
    evoked_amp: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A positive-baseline signal with a sparse, known post-stimulus response.

    Returns
    -------
    signal : (n_trials, n_sources, n_bins)
    truth  : (n_sources, n_bins-onset) boolean ground-truth active mask (post).
    """
    rng = np.random.default_rng(seed)
    signal = baseline_offset + noise_sd * rng.standard_normal((n_trials, n_sources, n_bins))

    n_evoked = int(round(evoked_frac * n_sources))
    evoked_sources = rng.choice(n_sources, size=n_evoked, replace=False)
    truth = np.zeros((n_sources, n_bins - onset), dtype=bool)

    for src in evoked_sources:
        # Each evoked source is active in a short, random post-stimulus window.
        w0 = rng.integers(2, 40)
        w1 = w0 + rng.integers(8, 30)
        w1 = min(w1, n_bins - onset)
        amp = evoked_amp * (0.6 + 0.8 * rng.random())
        signal[:, src, onset + w0 : onset + w1] += amp
        truth[src, w0:w1] = True

    return signal, truth


def _pci_from_binJ(binJ: np.ndarray) -> float:
    binJs = sort_binJ(binJ.astype(np.uint8))
    if not np.any(binJs):
        return 0.0
    return float(lz_complexity_2d(binJs) / max(pci_norm_factor(binJs), np.finfo(float).eps))


def _scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = pred.astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "active_fraction": float(pred.mean()),
        "precision": precision,
        "recall": recall,
        "pci": _pci_from_binJ(pred),
    }


def run(out_dir: Path) -> dict:
    onset = 100
    signal, truth = make_synthetic(onset=onset)
    n_trials, n_sources, n_bins = signal.shape

    # --- tvbsim route (per-trial) -> take the trial-averaged post binJ ----------
    tv = binarise_signals(signal, t_stim=onset, nshuffles=10, percentile=100.0)
    # collapse per-trial decisions to a single matrix (majority across trials)
    tv_post = tv[:, :, onset:].mean(axis=0) >= 0.5  # (n_sources, n_post)

    # --- casali route (single trial-averaged matrix) ---------------------------
    ca = binarise_signals_casali(signal, t_stim=onset, n_bootstrap=500, alpha=0.01)
    ca_post = ca[:, onset:].astype(bool)

    truth_frac = float(truth.mean())
    res = {
        "signal_shape": list(signal.shape),
        "onset": onset,
        "ground_truth_active_fraction": truth_frac,
        "ground_truth_pci": _pci_from_binJ(truth),
        "tvbsim": _scores(tv_post, truth),
        "casali": _scores(ca_post, truth),
    }

    _figure(signal, truth, tv_post, ca_post, onset, res, out_dir / "binarisation_compare.png")
    (out_dir / "binarisation_compare.json").write_text(json.dumps(res, indent=2))
    return res


def _figure(signal, truth, tv_post, ca_post, onset, res, path: Path) -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
    })
    avg = signal.mean(axis=0)  # (n_sources, n_bins)
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.34, wspace=0.28)

    axT = fig.add_subplot(gs[0, 0])
    im = axT.imshow(avg, aspect="auto", cmap="magma", interpolation="nearest")
    axT.axvline(onset, color="cyan", lw=1.2, ls="--")
    axT.set(title="A  Trial-averaged signal", xlabel="time (bins)", ylabel="source")
    fig.colorbar(im, ax=axT, fraction=0.046, pad=0.04)

    panels = [
        (gs[0, 1], "B  Ground-truth active (post)", truth, ROUTE_COLOR["ground truth"]),
        (gs[0, 2], "C  Casali route binJ (post)", ca_post, ROUTE_COLOR["casali"]),
        (gs[1, 0], "D  TVBSim route binJ (post)", tv_post, ROUTE_COLOR["tvbsim"]),
    ]
    for spec, title, mat, color in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        ax.set(title=title, xlabel="time (bins, post)", ylabel="source")
        ax.text(0.97, 0.97, f"active={mat.mean()*100:.1f}%", transform=ax.transAxes,
                va="top", ha="right", fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9))

    # Summary bars: active fraction and PCI
    axS = fig.add_subplot(gs[1, 1])
    labels = ["ground\ntruth", "casali", "tvbsim"]
    fracs = [res["ground_truth_active_fraction"], res["casali"]["active_fraction"],
             res["tvbsim"]["active_fraction"]]
    cols = [ROUTE_COLOR["ground truth"], ROUTE_COLOR["casali"], ROUTE_COLOR["tvbsim"]]
    axS.bar(labels, np.array(fracs) * 100, color=cols, edgecolor="black", linewidth=0.4)
    axS.set(title="E  Active fraction", ylabel="% active bins (post)")
    for i, f in enumerate(fracs):
        axS.text(i, f * 100, f"{f*100:.1f}", ha="center", va="bottom", fontsize=9)

    axP = fig.add_subplot(gs[1, 2])
    pvals = [res["ground_truth_pci"], res["casali"]["pci"], res["tvbsim"]["pci"]]
    axP.bar(labels, pvals, color=cols, edgecolor="black", linewidth=0.4)
    axP.set(title="F  Resulting PCI", ylabel="PCI")
    for i, p in enumerate(pvals):
        axP.text(i, p, f"{p:.3f}", ha="center", va="bottom", fontsize=9)
    pr = res["casali"]; tr = res["tvbsim"]
    axP.text(0.5, -0.32,
             f"precision/recall  —  casali: {pr['precision']:.2f}/{pr['recall']:.2f}   "
             f"tvbsim: {tr['precision']:.2f}/{tr['recall']:.2f}",
             transform=axP.transAxes, ha="center", va="top", fontsize=8.5)

    fig.suptitle("PCI binarisation routes on a known sparse evoked signal "
                 "(simulation stand-in)", fontsize=13, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=stimulation_results("stim_data", "binarisation_compare"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    res = run(args.out_dir)

    gt = res["ground_truth_active_fraction"] * 100
    print(f"  ground-truth active : {gt:.1f}%   PCI={res['ground_truth_pci']:.3f}")
    for route in ("casali", "tvbsim"):
        r = res[route]
        print(f"  {route:7s} active     : {r['active_fraction']*100:5.1f}%   "
              f"PCI={r['pci']:.3f}   precision={r['precision']:.2f} recall={r['recall']:.2f}")
    print(f"\n  wrote: {args.out_dir / 'binarisation_compare.png'}")
    print(f"  wrote: {args.out_dir / 'binarisation_compare.json'}")


if __name__ == "__main__":
    main()
