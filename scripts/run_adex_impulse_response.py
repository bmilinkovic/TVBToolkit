"""Native AAL90 impulse sweep; run from the repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "notebooks")]
os.environ.setdefault("TVB_USER_HOME", str(ROOT / ".tvb-temp"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tvbtoolkit-mpl-cache")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from brain_act_hybrid_common import BASE_PARAMETER_MODEL_NEW
from tvbtoolkit.core.config import WholeBrainConfig
from tvbtoolkit.datasets.brain_act import load_subject_structural
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import _apply_damage_parity
from tvbtoolkit.whole_brain.impulse_response import impulse_response

SUBJECTS = [("Wake", "control", "c0015"), ("EMCS", "emcs", "e0003"),
            ("MCS", "mcs", "m0075"), ("UWS", "uws", "u0020")]
B_VALUES = [5, 10, 20, 35, 45, 55]


def precision_diagnostics(result):
    source = int(result["source_index"])
    non_source = np.arange(result["response"].shape[0]) != source
    response = result["response"][non_source]
    half = result["half_epsilon_response"][non_source]
    error = response - half
    state = result["checkpoint_state"][:, :, 0]
    covariance = np.zeros((state.shape[1], 2, 2))
    covariance[:, 0, 0] = state[2]
    covariance[:, 0, 1] = covariance[:, 1, 0] = state[3]
    covariance[:, 1, 1] = state[4]
    baseline = result["baseline_khz"]
    return dict(
        source_peak_abs_response=float(np.max(np.abs(result["source_response"]))),
        source_peak_time_ms=float(result["time_ms"][np.argmax(np.abs(result["source_response"]))]),
        halving_non_source_relative_l2=float(np.linalg.norm(error) /
            max(np.linalg.norm(half), np.finfo(float).tiny)),
        halving_max_abs_non_source_disagreement=float(np.max(np.abs(error))),
        impulse_fraction_of_source_baseline=float(result["epsilon_khz"] /
            max(abs(result["control_khz"][source, 0]), np.finfo(float).tiny)),
        checkpoint_negative_excitatory_variance_regions=int((state[2] < 0).sum()),
        checkpoint_negative_inhibitory_variance_regions=int((state[4] < 0).sum()),
        checkpoint_min_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
        baseline_last_second_mean_temporal_std_hz=float(
            baseline[:, -10000:].std(axis=1).mean() * 1000),
        baseline_max_abs_region_mean_drift_hz=float(np.max(np.abs(
            baseline[:, -10000:].mean(axis=1) -
            baseline[:, -20000:-10000].mean(axis=1))) * 1000),
        responsive_non_source_by_relative_threshold={str(threshold): int(
            (np.max(np.abs(response), axis=1) > threshold).sum())
            for threshold in [.001, .01, .1]})


def build_subject_config(cohort, subject, b, dataset_root, coupling=.25,
                         monitor_variables=(0, 1)):
    """Reuse the native spontaneous configuration for both diagnostics."""
    weights, lengths, atlas, _ = load_subject_structural(
        subject_id=subject, cohort=cohort, dataset_root=dataset_root,
        validate=True, enforce_symmetry=True, zero_diagonal=True, nonfinite="raise")
    weights, lengths, zero_fraction = _apply_damage_parity(
        weights, lengths, cohort, normalize_subject_max=False)
    labels = np.asarray(atlas.labels, dtype=str)
    parameters = deepcopy(BASE_PARAMETER_MODEL_NEW)
    parameters.update(b_e=float(b), noise_alpha=0., shared_noise_mode="none")
    cfg = WholeBrainConfig(dt_ms=.1, conduction_speed=4., coupling_strength=coupling,
        model_family="adex_zerlaut", zerlaut_matteo=False, zerlaut_gk_gna=False,
        zerlaut_order=2, stochastic_integrator=True, monitor_mode="raw",
        monitor_variables=monitor_variables, weights=weights, tract_lengths=lengths,
        connectivity_normalization="none", parameter_overrides={"parameter_model": parameters})
    return cfg, labels, zero_fraction


def run_job(job):
    condition, cohort, subject, b, args = job
    cfg, labels, zero_fraction = build_subject_config(
        cohort, subject, b, args.dataset_root, args.coupling)
    weights, lengths = cfg.weights, cfg.tract_lengths
    source = args.source
    result = impulse_response(cfg, source=source, epsilon=args.epsilon,
                              baseline_ms=args.baseline_ms, seed=args.seed,
                              historical_pulse=args.historical_pulse)
    name = f"{condition}_{subject}_b{b:02d}"
    np.savez_compressed(args.output / f"{name}.npz", **result, labels=labels,
                        weights=weights, tract_lengths=lengths)
    non_source = np.arange(len(labels)) != source
    baseline = result["baseline_khz"]
    row = dict(condition=condition, cohort=cohort, subject=subject, b_pA=b,
        source_index_zero_based=source, source_region=str(labels[source]),
        absent_edge_fraction=zero_fraction,
        max_abs_non_source=float(result["peak_abs_by_region"][non_source].max()),
        responsive_regions_1pct=int(result["responsive_1pct"].sum()),
        responsive_non_source_1pct=int(result["responsive_1pct"][non_source].sum()),
        zero_impulse_max_error=float(result["zero_impulse_max_error"]),
        halving_relative_l2=float(result["halving_relative_l2"]),
        baseline_last_second_mean_hz=float(baseline[:, -10000:].mean()*1000),
        baseline_previous_second_mean_hz=float(baseline[:, -20000:-10000].mean()*1000),
        baseline_last_second_std_hz=float(baseline[:, -10000:].std()*1000),
        file=f"{name}.npz")
    row.update(precision_diagnostics(result))
    (args.output / f"{name}.json").write_text(json.dumps(row, indent=2))
    return row


def plot_results(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm, Normalize
    arrays = [np.load(output / row["file"]) for row in rows]
    maximum = max(float(np.max(np.abs(a["response"]))) for a in arrays)
    (output / "figure_settings.json").write_text(json.dumps(dict(
        colourmap="RdBu_r", common_vmin=-maximum, common_vmax=maximum,
        symlog_linthresh=1e-5, response_mask_applied=False), indent=2))
    for mode in ("linear", "symlog"):
        norm = Normalize(-maximum, maximum) if mode == "linear" else SymLogNorm(
            linthresh=1e-5, vmin=-maximum, vmax=maximum, base=10)
        fig, axes = plt.subplots(4, 6, figsize=(23, 23), sharex=True, sharey=True,
                                 layout="constrained")
        for row, data in zip(rows, arrays):
            r = [s[0] for s in SUBJECTS].index(row["condition"])
            c = B_VALUES.index(row["b_pA"])
            ax = axes[r, c]
            im = ax.imshow(data["response"], origin="lower", aspect="auto",
                extent=[0, 300, -.5, 89.5], cmap="RdBu_r", norm=norm,
                interpolation="nearest", rasterized=True)
            ax.axhline(row["source_index_zero_based"], color="#18b85a", lw=.65)
            if r == 0:
                ax.set_title(f"b = {row['b_pA']} pA")
            if c == 0:
                ax.set_ylabel(f"{row['condition']} — {row['subject']}\nAAL90 region")
                ax.set_yticks(np.arange(90), data["labels"], fontsize=4.2)
            if r == 3:
                ax.set_xlabel("Time after impulse (ms)")
        fig.colorbar(im, ax=axes, shrink=.5, label=f"Δνₑ / ε ({mode} colour scale)")
        fig.suptitle("Native second-order AdEx impulse response — matched noise and delay history\n"
                     f"Green line: {rows[0]['source_region']}; no activity threshold applied", fontsize=16)
        fig.savefig(output / f"impulse_response_{mode}.png", dpi=200)
        fig.savefig(output / f"impulse_response_{mode}.pdf")
        plt.close(fig)
    for data in arrays:
        data.close()

    fig, axes = plt.subplots(4, 6, figsize=(20, 11), sharex=True, sharey=True,
                             layout="constrained")
    for row in rows:
        with np.load(output / row["file"]) as data:
            r = [s[0] for s in SUBJECTS].index(row["condition"])
            c = B_VALUES.index(row["b_pA"])
            ax = axes[r, c]
            ax.plot(data["time_ms"], data["source_response"], color="#218c46",
                    lw=1.1, label="Stimulated region (signed)")
            ax.plot(data["time_ms"], data["max_abs_non_source_timecourse"],
                    color="#a62b8d", lw=1., label="Largest |response| elsewhere")
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.set_xlim(0, 300)
            if r == 0:
                ax.set_title(f"b = {row['b_pA']} pA")
            if c == 0:
                ax.set_ylabel(f"{row['condition']}\nΔνₑ / ε")
            if r == 3:
                ax.set_xlabel("Time after impulse (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    fig.savefig(output / "response_diagnostics.png", dpi=200)
    fig.savefig(output / "response_diagnostics.pdf")
    plt.close(fig)
    plot_resting(output, rows)


def write_summary(output, rows):
    lines = ["# Native AdEx impulse-response results", "",
        f"Source: **{rows[0]['source_region']}** (zero-based index {rows[0]['source_index_zero_based']}).",
        "See protocol.json for the settings and each NPZ for the region × time response arrays.", "",
        "[Linear heatmaps](impulse_response_linear.png) · [Weak-response view](impulse_response_symlog.png) · "
        "[Source/propagation traces](response_diagnostics.png) · [Resting dynamics](resting_dynamics.png)", "",
        "The colour scale is shared across all 24 panels. No threshold is applied to the heatmaps.",
        "Region counts use a descriptive peak |R| > 0.01, not a significance test.", "",
        "| Condition | Subject | b (pA) | Rest E (Hz) | Peak absolute R elsewhere | Non-source regions >1% | Non-source ε-halving relative L2 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {row['subject']} | {row['b_pA']} | "
            f"{row['baseline_last_second_mean_hz']:.4f} | {row['max_abs_non_source']:.6g} | "
            f"{row['responsive_non_source_1pct']} | {row['halving_non_source_relative_l2']:.3g} |")
    lines += ["", "## Interpretation safeguards", "",
        "- Epsilon-halving disagreement comparable to the non-source response means weak propagation is numerically unresolved.",
        "- Negative resting covariance variances are physically invalid; no equation or parameter was changed to conceal them.",
        "- These four illustrative subjects and one resting phase per cell do not establish a cohort-level condition effect.",
        f"- Largest zero-impulse difference: {max(r['zero_impulse_max_error'] for r in rows):.3g} kHz.",
        f"- Runs with negative excitatory variance at the checkpoint: {sum(r['checkpoint_negative_excitatory_variance_regions'] > 0 for r in rows)}/{len(rows)}.",
        f"- Maximum late-baseline per-region mean drift: {max(r['baseline_max_abs_region_mean_drift_hz'] for r in rows):.6g} Hz."]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def plot_resting(output, rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 6, figsize=(20, 11), sharex=True, sharey=True,
                             layout="constrained")
    colours = plt.get_cmap("turbo")(np.linspace(0, 1, 90))
    for row in rows:
        with np.load(output / row["file"]) as data:
            r = [s[0] for s in SUBJECTS].index(row["condition"])
            c = B_VALUES.index(row["b_pA"])
            ax = axes[r, c]
            # Display every 5 ms only; all 0.1-ms samples remain in the NPZ.
            rates = data["baseline_khz"][:, ::50] * 1000
            times = data["baseline_time_ms"][::50] / 1000
            for region in range(90):
                ax.plot(times, rates[region], color=colours[region], lw=.35,
                        alpha=.6, rasterized=True)
            if r == 0:
                ax.set_title(f"b = {row['b_pA']} pA")
            if c == 0:
                ax.set_ylabel(f"{row['condition']}\nExcitatory rate (Hz)")
            if r == 3:
                ax.set_xlabel("Resting simulation time (s)")
    fig.suptitle("Unperturbed resting dynamics — all 90 regions, unchanged native parameters")
    fig.savefig(output / "resting_dynamics.png", dpi=200)
    fig.savefig(output / "resting_dynamics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/adex_impulse_response")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/doc_data/converted_structural_invnodevol_native")
    parser.add_argument("--baseline-ms", type=float, default=8000.)
    parser.add_argument("--coupling", type=float, default=.25,
                        help="Native default .25; alternative values are explicitly separate calibration runs")
    parser.add_argument("--epsilon", type=float, default=1e-4,
                        help="Rate impulse in kHz; default 0.1 Hz avoids float32-history cancellation")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--source", type=int, default=9, help="Zero-based AAL index; default left SMA in this dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--historical-pulse", action="store_true",
                        help="Also replay the 10-ms historical 0.3 Hz/ms square drive")
    parser.add_argument("--plot-only", action="store_true", help="Rebuild figures and precision diagnostics from saved arrays")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = json.loads((args.output / "diagnostics.json").read_text())
        for row in rows:
            with np.load(args.output / row["file"]) as result:
                row.update(precision_diagnostics(result))
            (args.output / Path(row["file"]).with_suffix(".json")).write_text(json.dumps(row, indent=2))
        (args.output / "diagnostics.json").write_text(json.dumps(rows, indent=2))
        plot_results(args.output, rows)
        write_summary(args.output, rows)
        return
    jobs = [(condition, cohort, subject, b, args)
            for condition, cohort, subject in SUBJECTS for b in B_VALUES]
    if args.smoke:
        jobs = jobs[:1]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for row in pool.map(run_job, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output / "diagnostics.json").write_text(json.dumps(rows, indent=2))
    (args.output / "protocol.json").write_text(json.dumps(dict(
        baseline_ms=args.baseline_ms, response_ms=300, epsilon_khz=args.epsilon,
        seed=args.seed, dt_ms=.1, coupling_strength=args.coupling, conduction_speed_m_s=4.,
        dataset_root=str(args.dataset_root.resolve()), base_model=BASE_PARAMETER_MODEL_NEW,
        response_threshold="Diagnostic only: peak |R| > 0.01 (1% of imposed impulse); not statistical significance",
        monitor="Raw E and I; no LFP, Laplacian, or PCI threshold"), indent=2))
    if not args.smoke:
        plot_results(args.output, rows)
        write_summary(args.output, rows)


if __name__ == "__main__":
    main()
