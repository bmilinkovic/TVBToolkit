"""Separate trial-averaged analysis; never edits simulations or retunes PCI.

Requires the full gated trial bank. Compares the existing whole-trial/global-max
bootstrap with a paper-PROSE reconstruction: independent source-wise baseline
trial/time resampling, spatial maxima pooled over surrogate time. Pooling is an
explicit unresolved author-code detail, not claimed exact reproduction.
Also includes a whole-trial spatial-max sensitivity (preserves autocorrelation).
No pre/post swaps; no entropy floor; alpha .05, 500 surrogates for all routes.
"""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tvb-publication-mpl")
import argparse, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tvbtoolkit.complexity.pci_casali import (
    binarise_signals_casali,
    lz_complexity_2d,
    pci_norm_factor,
    sort_binJ,
    source_entropy,
)
from tvbtoolkit.complexity.pci_st import pci_st


def significance(data, pre, post, method, seed=19009, bootstrap=500):
    """Input trials × regions × time, output mask, evoked z, and null statistics."""
    if len(data) < 2:
        raise ValueError("Independent trials required")
    if method == "native_global":
        stacked = np.concatenate([data[:, :, pre], data[:, :, post]], axis=2)
        result = binarise_signals_casali(
            stacked,
            int(pre.sum()),
            n_bootstrap=bootstrap,
            alpha=0.05,
            seed=seed,
            significance_method="trial_bootstrap",
            two_sided=True,
            return_details=True,
        )
        return (
            result.binary[:, int(pre.sum()) :].astype(bool),
            result.averaged_response[:, int(pre.sum()) :],
            result.null_maxima,
            float(result.threshold),
        )
    centered = data - data[:, :, pre].mean(axis=2, keepdims=True)
    sd = np.sqrt(np.mean(centered[:, :, pre] ** 2, axis=(0, 2)))
    if np.any(sd <= 1e-12):
        raise ValueError("Zero/unresolved baseline variance")
    z = centered / sd[None, :, None]
    base = z[:, :, pre]
    observed = z[:, :, post].mean(axis=0)
    n, r, p = base.shape
    rng = np.random.default_rng(seed)
    null = np.empty((bootstrap, p))
    pool = base.transpose(1, 0, 2).reshape(r, -1)
    for b in range(bootstrap):
        if method == "paper_prose_spatial":
            # Independent indices per source destroy temporal/cross-source
            # dependence: explicit paper reconstruction, validated on sham.
            surrogate = np.empty((r, p))
            for i in range(r):
                surrogate[i] = pool[i, rng.integers(0, n * p, size=(n, p))].mean(axis=0)
        elif method == "trial_spatial":
            surrogate = base[rng.integers(0, n, size=n)].mean(axis=0)
        else:
            raise ValueError(method)
        null[b] = np.abs(surrogate).max(axis=0)
    threshold = float(np.quantile(null, 0.95))
    return np.abs(observed) > threshold, observed, null, threshold


def complexity(binary):
    ordered = sort_binJ(binary.astype(np.uint8))
    norm = pci_norm_factor(ordered)
    count = lz_complexity_2d(ordered) if norm > 0 else 0
    return dict(
        pci_lz=float(count / norm) if norm > 0 else 0.0,
        lz_count=int(count),
        entropy=float(source_entropy(binary)),
        active_fraction=float(binary.mean()),
        responsive_regions=int(binary.any(axis=1).sum()),
    )


def load_bank(root, gi, condition, n, allow_high=False):
    epochs = []
    times = None
    for trial in range(n):
        folder = root / f"g{gi}_trial{trial:03d}"
        meta = json.loads((folder / f"{condition}.json").read_text())
        if meta["status"] != "complete" or (meta["sustained_high"] and not allow_high):
            raise ValueError("Invalid trial retained: review, do not exclude silently")
        with np.load(folder / f"{condition}.npz") as d:
            select = (d["time_ms"] >= -1100) & (d["time_ms"] <= 700)
            tt = d["time_ms"][select]
            if times is not None:
                np.testing.assert_allclose(tt, times, atol=1e-8, rtol=0)
            times = tt
            epochs.append(d["state"][select, 0].T * 1000)
            labels = d["labels"].copy()
            k = int(d["source_index"])
            lengths = d["tract_lengths"].copy()
            weights = d["weights"].copy()
    return np.asarray(epochs), times, labels, k, lengths, weights


def figures(out, tag, data, t, labels, k, lengths, weights, masks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = (t >= -400) & (t < -100)
    show = (t >= -100) & (t < 500)
    evoked = (data - data[:, :, baseline].mean(axis=2, keepdims=True)).mean(axis=0)
    remote = np.arange(len(labels)) != k
    limit = max(float(np.abs(evoked[remote][:, show]).max()), 1e-8)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for i in range(len(labels)):
        axes[0, 0].plot(
            t[show], evoked[i, show], color=plt.cm.turbo(i / (len(labels) - 1)), lw=0.6
        )
    axes[0, 0].set(title="Trial-averaged baseline-centred E", ylabel="Hz")
    im = axes[0, 1].imshow(
        evoked[:, show],
        origin="upper",
        aspect="auto",
        extent=[t[show][0], t[show][-1], 89.5, -0.5],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    for y in (k - 0.5, k + 0.5):
        axes[0, 1].axhline(y, color="limegreen", lw=0.5)
    fig.colorbar(im, ax=axes[0, 1], extend="both", label="Hz; source may saturate")
    axes[0, 1].set(title="Unmasked, remote-defined scale", ylabel="AAL index")
    connected = np.flatnonzero((weights[k] > 0) & remote & (lengths[k] > 0))
    ordered = connected[np.argsort(lengths[k, connected])]
    n = min(5, len(ordered) // 2)
    for ids, label in [
        (np.array([k]), "stimulated"),
        (ordered[:n], "nearest connected"),
        (ordered[-n:] if n else np.array([], int), "farthest connected"),
    ]:
        if len(ids):
            axes[0, 2].plot(t[show], evoked[ids][:, show].mean(axis=0), label=label)
    axes[0, 2].set(title="Tract-distance groups (signed means)", ylabel="Hz")
    axes[0, 2].legend(fontsize=8)
    for ax, (method, mask, tp) in zip(axes[1], masks):
        ordering = np.argsort(-mask.sum(axis=1), kind="stable")
        ax.imshow(
            mask[ordering],
            origin="upper",
            aspect="auto",
            extent=[tp[0], tp[-1], 89.5, -0.5],
            cmap="binary",
            vmin=0,
            vmax=1,
        )
        ax.set(title=method, ylabel="Sources sorted by significant duration")
    for ax in axes.flat:
        ax.set_xlabel("Time after delivered onset (ms)")
    fig.suptitle(
        tag
        + " | first order | firing rates, not EEG | spatial-max methods not epoch-wide FWER"
    )
    fig.savefig(out / (tag + ".png"), dpi=170)
    fig.savefig(out / (tag + ".pdf"))
    plt.close(fig)
    # Anatomical centroid projections, NOT a reconstructed cortical surface.
    from tvbtoolkit.datasets.brain_act import load_subject_structural

    _, _, atlas, _ = load_subject_structural(
        subject_id="c0001",
        cohort="control",
        dataset_root=ROOT / "data/doc_data/converted_structural_invnodevol_control_max",
    )
    centres = np.asarray(atlas.centres)
    if list(atlas.labels) != list(labels):
        raise ValueError("Atlas centroids/order do not match saved regional data")
    if atlas.centres is None:
        (out / "spatial_maps_unavailable.txt").write_text(
            "The source atlas contains matching AAL90 labels but no coordinates. "
            "Centroid projections are omitted; no coordinates have been invented. "
            "PCI, butterfly and source-time figures are unaffected.\n"
        )
        return
    if centres.shape != (90, 3) or not np.isfinite(centres).all():
        raise ValueError("Invalid atlas coordinates")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    mask = masks[1][1]
    counts = mask.sum(axis=1)
    for ax, (a, b, name) in zip(
        axes, [(0, 1, "Axial"), (0, 2, "Coronal"), (1, 2, "Sagittal")]
    ):
        image = ax.scatter(
            centres[:, a],
            centres[:, b],
            c=counts,
            cmap="viridis",
            vmin=0,
            vmax=mask.shape[1],
            s=22,
        )
        if "_sham_" not in tag:
            ax.scatter(centres[k, a], centres[k, b], marker="*", c="red", s=100)
        ax.set(
            title=name + " centroid projection",
            aspect="equal",
            xlabel=f"Coordinate {a} (mm)",
            ylabel=f"Coordinate {b} (mm)",
        )
    fig.colorbar(
        image, ax=axes, label="Significant 1-ms bins (paper-prose spatial maximum)"
    )
    fig.suptitle(
        tag + " | same scale across settings; not surface-reconstructed activation"
    )
    fig.savefig(out / (tag + "_spatial.png"), dpi=170)
    fig.savefig(out / (tag + "_spatial.pdf"))
    plt.close(fig)


def synthetic(out):
    """Known coloured baseline and injected responses, not manufactured trials."""
    rng = np.random.default_rng(1971)
    data = rng.normal(size=(24, 12, 600))
    for j in range(1, 600):
        data[:, :, j] = 0.9 * data[:, :, j - 1] + 0.436 * data[:, :, j]
    pre = np.arange(600) < 300
    post = ~pre
    results = []
    for name in ("null", "local", "distributed"):
        x = data.copy()
        truth = np.zeros((12, 300), bool)
        if name == "local":
            truth[0, 20:100] = True
        if name == "distributed":
            for r in range(12):
                truth[r, 10 + r * 12 : 50 + r * 12] = True
        x[:, :, post] += truth[None] * 2
        for method in ("native_global", "paper_prose_spatial", "trial_spatial"):
            mask, _, _, _ = significance(x, pre, post, method, bootstrap=100)
            results.append(
                dict(
                    pattern=name,
                    method=method,
                    **complexity(mask),
                    injected_detection=float(mask[truth].mean())
                    if truth.any()
                    else None,
                    outside_injection_fraction=float(mask[~truth].mean()),
                )
            )
    (out / "synthetic_validation.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(exist_ok=True, parents=True)
    if args.self_test:
        synthetic(out)
        return
    if (
        not (args.root / "completion.json").exists()
        or (args.root / "REVIEW_REQUIRED.json").exists()
    ):
        raise ValueError("Full simulation bank has not passed its gate")
    manifest = json.loads((args.root / "manifest.json").read_text())
    n = manifest["trials"]
    rows = []
    methods = ("native_global", "paper_prose_spatial", "trial_spatial")
    for gi, g in enumerate(manifest["G"]):
        for condition in ("sham", "sma", "parietal"):
            data, t, labels, k, lengths, weights = load_bank(
                args.root,
                gi,
                condition,
                n,
                allow_high=manifest.get("continue_high", False),
            )
            for window, baseline, response in [
                ("current", (-400, -100), (0, 300)),
                ("extended", (-400, -100), (10, 500)),
            ]:
                pre = (t >= baseline[0]) & (t < baseline[1])
                post = (t >= response[0]) & (t < response[1])
                masks = []
                saved = {}
                for method in methods:
                    mask, z, null, threshold = significance(data, pre, post, method)
                    masks.append((method, mask, t[post]))
                    saved[method + "_mask"] = mask
                    saved[method + "_null"] = null
                    for exclude in (False, True):
                        keep = (
                            np.arange(90) != k
                            if exclude and condition != "sham"
                            else np.ones(90, bool)
                        )
                        rows.append(
                            dict(
                                G=g,
                                condition=condition,
                                window=window,
                                method=method,
                                n_trials=n,
                                exclude_source=exclude,
                                threshold=threshold,
                                **complexity(mask[keep]),
                            )
                        )
                centered = data - data[:, :, pre].mean(axis=2, keepdims=True)
                average = centered.mean(axis=0)
                for exclude in (False, True):
                    keep = (
                        np.arange(90) != k
                        if exclude and condition != "sham"
                        else np.ones(90, bool)
                    )
                    value = pci_st(
                        average[keep],
                        t,
                        baseline_window_ms=baseline,
                        response_window_ms=response,
                        k=1.2,
                        min_snr=1.1,
                        max_var_percent=99,
                        n_steps=100,
                    )
                    rows.append(
                        dict(
                            G=g,
                            condition=condition,
                            window=window,
                            method="PCI_ST",
                            n_trials=n,
                            exclude_source=exclude,
                            pci_st=float(value),
                        )
                    )
                tag = f"g{g:.3f}_{condition}_{window}"
                np.savez_compressed(
                    out / (tag + ".npz"),
                    time_ms=t,
                    evoked_hz=average,
                    labels=labels,
                    **saved,
                )
                (out / "scores.json").write_text(json.dumps(rows, indent=2))
                figures(out, tag, data, t, labels, k, lengths, weights, masks)
                print("Completed", tag, flush=True)
            # Prespecified nested trial-count sensitivity; not independent replicates.
            pre = (t >= -400) & (t < -100)
            post = (t >= 0) & (t < 300)
            convergence = []
            for number in (25, 50, 100, 150, 200):
                if number > n:
                    continue
                for method in methods:
                    mask, _, _, _ = significance(data[:number], pre, post, method)
                    convergence.append(
                        dict(n_trials=number, method=method, **complexity(mask))
                    )
            (out / f"g{g:.3f}_{condition}_convergence.json").write_text(
                json.dumps(convergence, indent=2)
            )
    synthetic(out)


if __name__ == "__main__":
    main()
