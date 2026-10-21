"""Trial-AVERAGED Dalla Porta notebook-style PCI on the 5×50 AAL90 bank.

The notebook's binJproc is translated literally except that N0 is explicitly
300 samples, avoiding its off-by-one `idx[-1]` mistake. The original notebook
draws bootstrap trial identities ONCE, before its 1000 time-resampling draws;
we retain that behavior here. Region-wise pooled baseline mean/SD, absolute
two-sided response, spatial maximum, pooled surrogate-time 99th percentile,
descending active-row sort, and LZ76/entropy normalization are retained.

The output interval is [0,300) ms to match the current Toolkit PCI protocol;
the supplied notebook's illustrative [0,500] ms interval is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tvb-della-porta-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "src")]
from study import BANK, TRIALS, paths, write_json
from tvbtoolkit.complexity.pci_casali import (lz_complexity_2d, pci_norm_factor,
                                               sort_binJ, source_entropy)

COLORS = {"CNT": "#3275a9", "EMCS": "#49a78c", "MCS": "#c58921", "UWS": "#bd5b71"}


def notebook_binJproc(epochs: np.ndarray, *, seed: int, nboot: int = 1000,
                       alpha: float = .01) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Faithful bootstrap test, input (trial, region, time), time=pre300+post300.

    Trial IDs are sampled once, as in the supplied PCI notebook. For every
    bootstrap and pseudo-trial, baseline time indices are redrawn. All regions
    use the same time indices, preserving within-sample spatial covariance,
    although temporal autocorrelation is not preserved by this procedure.
    """
    if epochs.ndim != 3 or epochs.shape[1:] != (90, 600) or epochs.shape[0] < 2:
        raise ValueError(f"Expected trials × 90 × 600: {epochs.shape}")
    trials, regions, _ = epochs.shape
    base = epochs[:, :, :300]
    mean = base.mean(axis=(0, 2))
    sd = base.transpose(2, 1, 0).std(axis=(0, 2), ddof=1)
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("Invalid pooled baseline SD")
    observed = epochs.mean(axis=0) - mean[:, None]
    rng = np.random.RandomState(seed)
    trial_ids = rng.randint(0, trials, trials)  # intentionally outside loop
    maxima = np.empty((nboot, 300), dtype=np.float64)
    for boot in range(nboot):
        surrogate = np.zeros((regions, 300), dtype=np.float64)
        for trial_id in trial_ids:
            sampled_times = rng.randint(0, 300, 300)
            surrogate += base[trial_id, :, sampled_times].T
        z = (surrogate/trials - mean[:, None]) / sd[:, None]
        maxima[boot] = np.max(np.abs(z), axis=0)
    sorted_null = np.sort(maxima.ravel())
    index = int(np.round((1-alpha)*sorted_null.size))-1
    zcut = float(sorted_null[index])
    binary = np.abs(observed[:, 300:]) > sd[:, None]*zcut
    return binary, zcut, observed[:, 300:], sd


def pci_from_mask(mask: np.ndarray) -> dict:
    """Same 2-D LZ76 structure and L*H/log2(L) normalization as source code.

    Toolkit uses byte-substring search in place of ``bitarray.search``; the
    emitted 0/1 sequence, parsing, and normalization are otherwise identical.
    Constant binary matrices have undefined entropy normalization: an all-zero
    map is reported as PCI=0 with status ``no_activation``; an all-one map is
    reported as NaN with status ``undefined_full_activation``.
    """
    ordered = sort_binJ(mask.astype(np.uint8))
    entropy = float(source_entropy(ordered))
    norm = float(pci_norm_factor(ordered))
    active = int(mask.sum())
    if norm == 0:
        return dict(pci=0.0 if active == 0 else np.nan, lz_count=0,
                    entropy=entropy, active_fraction=float(mask.mean()),
                    responsive_regions=int(mask.any(axis=1).sum()),
                    status="no_activation" if active == 0 else "undefined_full_activation")
    count = int(lz_complexity_2d(ordered))
    return dict(pci=float(count/norm), lz_count=count, entropy=entropy,
                active_fraction=float(mask.mean()),
                responsive_regions=int(mask.any(axis=1).sum()), status="ok")


def load_case(bank: Path, case_index: int) -> tuple[np.ndarray, list[dict], np.ndarray, int]:
    epochs, metadata, labels, source = [], [], None, None
    for trial in range(TRIALS):
        recording, meta_path = paths(bank, case_index, trial)
        with np.load(recording) as d:
            t = d["time_ms"]
            pre = (t >= -300) & (t < 0)
            post = (t >= 0) & (t < 300)
            if d["E_hz"].shape != (1000, 90) or pre.sum() != 300 or post.sum() != 300:
                raise ValueError(f"Incomplete or misaligned recording: {recording}")
            current_labels, current_source = d["labels"].astype(str), int(d["source"])
            if labels is None:
                labels, source = current_labels, current_source
            elif not np.array_equal(labels, current_labels) or source != current_source:
                raise ValueError(f"Atlas/source changed: {recording}")
            e = d["E_hz"].astype(np.float64)
            epochs.append(np.concatenate((e[pre].T, e[post].T), axis=1))
        metadata.append(json.loads(meta_path.read_text()))
    return np.stack(epochs), metadata, labels, source


def plot_masks(rows: list[dict], out: Path, bank: Path) -> None:
    """Five source-sorted binary maps per cohort and leak, one figure each."""
    for leak in (-63.0, -62.0):
        for group in COLORS:
            selected = [r for r in rows if r["leak_mv"] == leak and
                        r["group"] == group and r["occupancy"] == 0]
            if len(selected) != 5:
                raise ValueError(f"Missing zero-occupancy cases: {leak}, {group}")
            fig, axes = plt.subplots(5, 1, figsize=(10.5, 10), sharex=True,
                                     layout="constrained")
            for ax, row in zip(axes, selected):
                with np.load(out / "masks" / f"case_{row['case_index']:03d}.npz") as d:
                    mask = d["binary"].astype(bool)
                    source = int(d["source"])
                    rank = d["rank"]
                sorted_mask = mask[rank]
                ax.imshow(sorted_mask, aspect="auto", interpolation="nearest",
                          origin="upper", cmap="Greys", vmin=0, vmax=1,
                          extent=[0, 300, 89.5, -.5])
                source_row = int(np.flatnonzero(rank == source)[0])
                ax.axhline(source_row, color="#df3d39", lw=.9, ls="--")
                ax.set(ylabel=f"{row['subject']}\nAAL90 rank",
                       xlim=(0, 300), ylim=(89.5, -.5))
                ax.text(298, 3, f"PCI {row['pci']:.3f} | {row['responsive_regions']} sources",
                        ha="right", va="top", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.75))
            axes[-1].set_xlabel("Time after pulse onset (ms)")
            fig.suptitle(f"{group} | E_L={leak:.0f} mV | zero occupancy | source row dashed red")
            base = out / "figures" / f"zero_occ_masks_{group.lower()}_el{int(-leak)}"
            base.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(base.with_suffix(".pdf"), dpi=300)
            fig.savefig(base.with_suffix(".png"), dpi=220)
            plt.close(fig)


def plot_pci(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True,
                             layout="constrained")
    for ax, leak in zip(axes, (-63.0, -62.0)):
        for group, color in COLORS.items():
            data = [r for r in rows if r["leak_mv"] == leak and r["group"] == group]
            for subject in sorted({r["subject"] for r in data}):
                one = sorted((r for r in data if r["subject"] == subject),
                             key=lambda r: r["occupancy"])
                ax.plot([r["occupancy"] for r in one], [r["pci"] for r in one],
                        color=color, lw=.85, alpha=.26)
            means = [np.nanmean([r["pci"] for r in data if r["occupancy"] == occ])
                     for occ in (0., .25, .5, .766)]
            ax.plot((0., .25, .5, .766), means, "o-", color=color, lw=2.4,
                    label=group)
        ax.set(title=f"E_L={leak:.0f} mV", xlabel="5-HT2A occupancy",
               xticks=(0., .25, .5, .766), xlim=(-.02, .79))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("PCI-LZ76 (trial-averaged response)")
    axes[1].legend(frameon=False, ncol=2)
    path = out / "figures/pci_by_occupancy"
    fig.savefig(path.with_suffix(".pdf"), dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--nboot", type=int, default=1000)
    args = p.parse_args()
    bank = args.bank
    out = args.output or bank / "pci_notebook_style"
    manifest = json.loads((bank / "manifest.json").read_text())
    expected = len(manifest["cases"])*TRIALS
    missing = [(ci, ti) for ci in range(len(manifest["cases"])) for ti in range(TRIALS)
               if not all(path.exists() for path in paths(bank, ci, ti))]
    if missing:
        raise ValueError(f"Bank incomplete: {len(missing)}/{expected} trial pairs missing; first {missing[0]}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, case in enumerate(manifest["cases"]):
        epochs, meta, labels, source = load_case(bank, ci)
        seed = int(np.random.SeedSequence([meta[0]["seed"], 7611]).generate_state(1)[0])
        mask, cutoff, response, sd = notebook_binJproc(
            epochs, seed=seed, nboot=args.nboot)
        score = pci_from_mask(mask)
        rank = mask.sum(axis=1).argsort()[::-1]
        mask_path = out / "masks" / f"case_{ci:03d}.npz"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask_path, binary=mask, response_hz=response,
                            baseline_sd_hz=sd, rank=rank, labels=labels,
                            source=source, time_ms=np.arange(300), z_cutoff=cutoff)
        row = dict(case_index=ci, **case, **score, z_cutoff=cutoff,
                   trials=TRIALS, nboot=args.nboot, alpha=.01,
                   high_rate_trials=sum(m["high_rate_post_transient_steps"]>0 for m in meta),
                   covariance_invalid_trials=sum(m["covariance_invalid_1ms_samples"]>0 for m in meta),
                   source_index=source)
        rows.append(row)
        if (ci+1) % 10 == 0 or ci+1 == len(manifest["cases"]):
            print(f"{ci+1}/{len(manifest['cases'])} cases analysed", flush=True)
    with (out / "pci_by_subject_occupancy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_masks(rows, out, bank)
    plot_pci(rows, out)
    write_json(out / "methods.json", dict(method="Dalla Porta PCI notebook binJproc + LZ76 equivalent byte-search port",
        pre_ms=[-300, 0], post_ms=[0, 300], trial_average_before_binarization=True,
        bootstrap_trial_ids_reused_across_draws=True, bootstrap_draws=args.nboot,
        alpha=.01, spatial_maximum=True, two_sided=True,
        entropy_floor=None, high_rate_trials_retained=True,
        zero_entropy_policy="zero map -> PCI 0; full map -> NaN",
        source_notebook="della-porta-stim/PCI.ipynb", cases=len(rows)))
    print(f"Saved {len(rows)} subject×occupancy×leak PCI rows and figures to {out}")


if __name__ == "__main__":
    main()
