"""Compare the supplied Gaglioti binJproc with our paper-prose reconstruction.

Input is the previously saved first-order AdEx trial bank. The author routine
defines only binarisation; the same Toolkit LZ calculation is applied to all
masks so score differences isolate this thresholding step. Nbegin=0, as the
supplied function requires for shape consistency. No model is rerun.
"""

import json
import os
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tvb-publication-mpl")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from gaglioti_adex_analysis import complexity

BANK = ROOT / "results/gaglioti_adex_drive0315_full_tracked_20261009"
OUT = BANK / "analysis/gaglioti_supplied_code_comparison"


def supplied_binJproc_on_windows(data, n0=300, nboot=500, seed=19009, chunk=16):
    """The supplied code's statistic and sampling, with two alpha cutoffs.

    `data` has shape trial × area × time, with the 300 baseline samples then
    300 post-onset samples. Drawing one trial per pseudo-trial and independent
    baseline times for that pseudo-trial preserves spatial covariance because
    its time draw is shared across all areas. Chunking pseudo-trials only saves
    temporary memory; the statistical sampling scheme is unchanged.
    """
    if data.ndim != 3 or data.shape[2] != 2 * n0:
        raise ValueError("Expected trial × area × (300 baseline + 300 post)")
    ntrials, narea, _ = data.shape
    J = data.transpose(2, 1, 0)
    baseline = J[:n0]
    base = np.mean(baseline, axis=(0, 2))
    norm = np.std(baseline, axis=(0, 2), ddof=1)
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("Nonpositive or invalid pooled baseline SD")
    observed_hz = J.mean(axis=2).T - base[:, None]
    rng = np.random.RandomState(seed)
    maxima = np.empty((nboot, n0), dtype=np.float64)
    for boot in range(nboot):
        et = np.zeros((narea, n0), dtype=np.float64)
        trial_ids = rng.randint(0, ntrials, ntrials)
        for start in range(0, ntrials, chunk):
            ids = trial_ids[start : start + chunk]
            times = rng.randint(0, n0, (len(ids), n0))
            draws = baseline[times, :, ids[:, None]]
            if draws.shape != (len(ids), n0, narea):
                raise AssertionError("Advanced indexing changed spatial axes")
            et += draws.sum(axis=0).T
        z = (et / ntrials - base[:, None]) / norm[:, None]
        maxima[boot] = np.max(np.abs(z), axis=0)
    ordered = np.sort(maxima.ravel())
    masks = {}
    thresholds_z = {}
    for alpha in (0.01, 0.05):
        # Exactly the supplied round( (1-alpha)*Nboot*N0 ) - 1 index.
        index = int(np.round((1 - alpha) * nboot * n0)) - 1
        zcrit = float(ordered[index])
        threshold_hz = norm * zcrit
        masks[alpha] = np.abs(observed_hz[:, n0:]) > threshold_hz[:, None]
        thresholds_z[alpha] = zcrit
    return masks, thresholds_z, norm, observed_hz, maxima


def load_windows(gi, arm, n):
    values = []
    labels, source = None, None
    for trial in range(n):
        with np.load(BANK / f"g{gi}_trial{trial:03d}" / f"{arm}.npz") as d:
            time = d["time_ms"]
            pre = (time >= -400) & (time < -100)
            post = (time >= 0) & (time < 300)
            if int(pre.sum()) != 300 or int(post.sum()) != 300:
                raise ValueError("A required 300-sample window is missing")
            if labels is None:
                labels = d["labels"].astype(str)
                source = int(d["source_index"])
            values.append(
                np.concatenate((d["state"][pre, 0], d["state"][post, 0]), axis=0).T
                * 1000
            )
    return np.asarray(values), labels, source


def plot_saved_masks():
    """Plot three threshold maps with one common row order in each case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for g in (0.5333333333333333, 0.6111111111111112):
        for arm in ("sham", "sma", "parietal"):
            with np.load(OUT / f"g{g:.3f}_{arm}.npz") as d:
                masks = [
                    d["old_mask"],
                    d["exact_alpha005_mask"],
                    d["exact_alpha001_mask"],
                ]
                source = int(d["source_index"])
            order = np.argsort(-sum(m.sum(axis=1) for m in masks), kind="stable")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
            for ax, mask, title in zip(
                axes,
                masks,
                (
                    "Our reconstruction; α=.05",
                    "Supplied code; α=.05",
                    "Supplied code; α=.01",
                ),
            ):
                ax.imshow(
                    mask[order],
                    aspect="auto",
                    origin="upper",
                    cmap="binary",
                    vmin=0,
                    vmax=1,
                    extent=[0, 300, 89.5, -0.5],
                    interpolation="nearest",
                )
                if arm != "sham":
                    row = int(np.flatnonzero(order == source)[0])
                    ax.axhline(row, color="limegreen", lw=1)
                ax.set(
                    title=f"{title}\n{int(mask.sum())} marked samples",
                    xlabel="Time after pulse onset (ms)",
                    ylabel="AAL90 rows; common descending order",
                )
            fig.suptitle(f"G={g:.3f}, {arm}; same 200 trials, same LZ input window")
            fig.savefig(OUT / f"g{g:.3f}_{arm}_threshold_maps.png", dpi=180)
            fig.savefig(OUT / f"g{g:.3f}_{arm}_threshold_maps.pdf")
            plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BANK / "manifest.json").read_text())
    rows = []
    for gi, g in enumerate(manifest["G"]):
        for arm in ("sham", "sma", "parietal"):
            print(f"G={g:.3f} {arm}: loading saved trials", flush=True)
            data, labels, source = load_windows(gi, arm, manifest["trials"])
            masks, thresholds, pooled_sd, response, maxima = (
                supplied_binJproc_on_windows(data)
            )
            old = np.load(BANK / "analysis" / f"g{g:.3f}_{arm}_current.npz")
            previous = old["paper_prose_spatial_mask"].astype(bool)
            within_sd = np.sqrt(
                np.mean(
                    (data[:, :, :300] - data[:, :, :300].mean(axis=2, keepdims=True))
                    ** 2,
                    axis=(0, 2),
                )
            )
            scale_ratio = pooled_sd / within_sd
            for alpha in (0.01, 0.05):
                mask = masks[alpha]
                remote = np.arange(len(labels)) != source
                rows.append(
                    dict(
                        G=g,
                        condition=arm,
                        n_trials=len(data),
                        method="supplied_binJproc",
                        alpha=alpha,
                        Nbegin=0,
                        Nboot=500,
                        n_baseline=300,
                        threshold_z=thresholds[alpha],
                        threshold_hz_median=float(
                            np.median(pooled_sd * thresholds[alpha])
                        ),
                        pooled_sd_over_within_sd_median=float(np.median(scale_ratio)),
                        source_bins=int(mask[source].sum()) if arm != "sham" else None,
                        remote_bins=int(mask[remote].sum()) if arm != "sham" else None,
                        remote_regions=int(mask[remote].any(axis=1).sum())
                        if arm != "sham"
                        else None,
                        significant_time_columns=int(mask.any(axis=0).sum()),
                        overlap_bins_with_previous=int((mask & previous).sum()),
                        **complexity(mask),
                    )
                )
            rows.append(
                dict(
                    G=g,
                    condition=arm,
                    n_trials=len(data),
                    method="our_paper_prose_spatial",
                    alpha=0.05,
                    significant_time_columns=int(previous.any(axis=0).sum()),
                    **complexity(previous),
                )
            )
            np.savez_compressed(
                OUT / f"g{g:.3f}_{arm}.npz",
                labels=labels,
                source_index=source,
                old_mask=previous,
                exact_alpha001_mask=masks[0.01],
                exact_alpha005_mask=masks[0.05],
                average_baseline_corrected_hz=response,
                author_baseline_sd_hz=pooled_sd,
                within_trial_baseline_sd_hz=within_sd,
                bootstrap_spatial_maxima_z=maxima,
            )
            (OUT / "scores.json").write_text(json.dumps(rows, indent=2))
            print(
                f"  alpha .01: {int(masks[0.01].sum())} bins; alpha .05: {int(masks[0.05].sum())} bins",
                flush=True,
            )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for gi, g in enumerate(manifest["G"]):
        for ai, arm in enumerate(("sham", "sma", "parietal")):
            ax = axes[gi, ai]
            subset = [r for r in rows if r["G"] == g and r["condition"] == arm]
            vals = [
                next(
                    r["pci_lz"]
                    for r in subset
                    if r["method"] == "our_paper_prose_spatial"
                ),
                next(
                    r["pci_lz"]
                    for r in subset
                    if r["method"] == "supplied_binJproc" and r["alpha"] == 0.05
                ),
                next(
                    r["pci_lz"]
                    for r in subset
                    if r["method"] == "supplied_binJproc" and r["alpha"] == 0.01
                ),
            ]
            ax.bar(np.arange(3), vals, color=("#c08454", "#4b81a8", "#364b74"))
            ax.set_xticks(
                np.arange(3),
                (
                    "Our reconstruction\nα=.05",
                    "Supplied code\nα=.05",
                    "Supplied code\nα=.01",
                ),
            )
            ax.set(title=f"G={g:.3f} | {arm}", ylabel="Toolkit PCI-LZ on binary map")
            for j, v in enumerate(vals):
                ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Author-supplied thresholding compared with our reconstruction\nSame saved firing rates; same Toolkit LZ calculation; 200 trials per arm"
    )
    fig.savefig(OUT / "pci_lz_threshold_comparison.png", dpi=190)
    fig.savefig(OUT / "pci_lz_threshold_comparison.pdf")
    plt.close(fig)
    plot_saved_masks()
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
