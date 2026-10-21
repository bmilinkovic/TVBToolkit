"""Matched AAL90 G × background-drive TEP screen, separate from production.

Run each order at 100 parameter points, with a control and two 10-ms pulses.
Reuse native equations/configuration and publication TF/noise parameters. The
second-order arm attenuates derivatives experimentally; this is NOT a unit
conversion. No covariance clipping, rate ceiling, PCI or significance testing.
One common noise realization is a screening experiment, not empirical fitting.

Examples:
  python scripts/sweep_split_leak_tep_drive.py --output results/tep_drive_grid --workers 4
  python scripts/sweep_split_leak_tep_drive.py --output results/tep_drive_grid --analyse

Full trajectories are streamed, saved every 1 ms as float64, and checked at
every integration step. Finished parameter points can be resumed. Failed runs
remain in the results; no parameter point is silently dropped.
"""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("TVB_USER_HOME", str(ROOT / ".tvb-temp"))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tvb-publication-mpl")
import b_sweep_publication_run as base
from run_adex_impulse_response import build_subject_config
from experiments.gnak_order_comparison.model_adapter import scaled_rhs
from first_order_amplitude_trials import install_noise
from tvbtoolkit.whole_brain.simulation import configure_adex_simulator
from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90, leak_to_conductances


def write_json(path, value):
    """Atomic checkpoints: a killed worker must not leave half a JSON file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def make_sim(
    order,
    g,
    drive,
    amplitude,
    dataset,
    duration=10000,
    *,
    seed=None,
    onset_ms=None,
    source_label="Supp_Motor_Area_L",
    pulse_ms=10.0,
    column_normalize=False,
    b_map_pa=None,
    weight_noise=None,
    tau_ou_ms=None,
    leak_e_mv=None,
    cohort="control",
    subject="c0001",
    receptor_occupancy=0.0,
    drug_leak_e_mv=-61.2,
):
    """Preserve subject weights/delays; match the new-fit passive membrane."""
    cfg, labels, _ = build_subject_config(cohort, subject, 5, dataset, g)
    pm = base._shared_zerlaut_overrides(5, baseline_hz=drive)
    pm.pop("derivative_scale", None)
    pm["weight_noise"] = 0.001
    if weight_noise is not None:
        pm["weight_noise"] = float(weight_noise)
    if tau_ou_ms is not None:
        pm["tau_OU"] = float(tau_ou_ms)
    if leak_e_mv is not None:
        pm["E_L_e"] = float(leak_e_mv)
    if b_map_pa is not None:
        b_map_pa = np.asarray(b_map_pa, dtype=float)
        if b_map_pa.shape != (len(labels),) or not np.isfinite(b_map_pa).all() or (b_map_pa < 0).any():
            raise ValueError("b_map_pa must contain one finite nonnegative value per region")
        pm["b_e"] = b_map_pa.tolist()
    # Split-leak uses one excitatory synaptic time constant rather than the
    # standard class's two attributes; both are 5 ms in the new fit.
    pm["tau_e"] = base.TAU_E_MS
    # Native gNa/gK is already the default. Do not pass the optional
    # TVBSim-reference selector into older cluster checkouts that lack it.
    pm.pop("tvbsim_reference", None)
    if order == 1:
        for key in ("C_ee", "C_ei", "C_ii"):
            pm["initial_condition"].pop(key, None)
    if column_normalize:
        weights = np.asarray(cfg.weights, dtype=float)
        column_strength = weights.sum(axis=0)
        # Match the published notebook's *column* division exactly. In TVB's
        # target-row/source-column convention this is NOT incoming row-degree
        # normalization. A zero column stays zero; no edge is created.
        weights = np.divide(weights, column_strength[None, :], out=np.zeros_like(weights), where=column_strength[None, :] > 0)
        cfg = replace(cfg, weights=weights)
    if not 0.0 <= receptor_occupancy <= 0.766:
        raise ValueError("receptor_occupancy must be within the established [0, 0.766] range")
    receptor = get_5ht2a_aal90(target_labels=labels)
    rho = (receptor - receptor.min()) / (receptor.max() - receptor.min() + 1e-12)
    # Occupancy follows the established gNa/gK route: gNa is held fixed while
    # gK is interpolated toward the drug leak potential by occupancy × density.
    # At occupancy zero, the executed conductances match the preceding sweep.
    for pop in ("e", "i"):
        gk, gna = leak_to_conductances(50.0, -90.0, pm["E_L_" + pop], g_L=10.0)
        if receptor_occupancy:
            drug_el = float(drug_leak_e_mv) if pop == "e" else -64.4
            gk_drug, _ = leak_to_conductances(50.0, -90.0, drug_el, g_Na=gna)
            gk = gk - receptor_occupancy * rho * (gk - gk_drug)
        pm.update(
            {
                "g_K_" + pop: gk.tolist() if receptor_occupancy else float(gk),
                "g_Na_" + pop: float(gna),
                "E_K_" + pop: -90.0,
                "E_Na_" + pop: 50.0,
            }
        )
    seed = base.SEED if seed is None else int(seed)
    onset_ms = (
        (6000.0 if duration >= 6300 else duration / 2)
        if onset_ms is None
        else float(onset_ms)
    )
    indices = np.flatnonzero(labels == source_label)
    if len(indices) != 1:
        raise ValueError(f"Source label is not unique: {source_label}")
    source = int(indices[0])
    overrides = dict(parameter_model=pm, parameter_integrator={"type": "Heun"})
    if amplitude:
        overrides["parameter_stimulus"] = dict(
            stimtime=onset_ms,
            stimdur=float(pulse_ms),
            stimperiod=100000.0,
            stimval=amplitude / 1000.0,
            stimregion=[source],
            stimvariables=[0],
            stimshape="square",
        )
    cfg = replace(
        cfg,
        zerlaut_order=order,
        zerlaut_gk_gna=True,
        stochastic_integrator=True,
        simulation_length_ms=duration,
        monitor_variables=tuple(range(5 if order == 1 else 8)),
        parameter_overrides=overrides,
    )
    sim = configure_adex_simulator(cfg, seed)
    # Existing production noise driver excites only the final OU state, not
    # E/I or covariances directly. Separate seed from the external OU trace.
    install_noise(sim, seed + 10000)
    for key, expected in (("P_e", base.P_E_V2), ("P_i", base.P_I_V2)):
        if not np.array_equal(np.asarray(getattr(sim.model, key)), expected):
            raise ValueError(f"Executed {key} does not match the new fit")
    if order == 2:
        scaled_rhs(sim.model)
    sigma = base.paper_sigma_to_runner_sigma_hz(
        base.PAPER_SIGMA_HZ, base.TAU_OU_MS, 0.1
    )
    trace = base.build_shared_afferent_trace_hz(
        duration_ms=duration,
        dt_ms=0.1,
        seed=seed,
        v_drive_hz=drive,
        sigma_ou_hz=sigma,
        tau_ou_ms=base.TAU_OU_MS,
    )
    original = sim.model.dfun
    counter = [0]

    def with_drive(model, state, coupling, local_coupling=0.0):
        index = min(counter[0] // 2, len(trace) - 1)
        counter[0] += 1
        value = np.full((len(labels), 1), trace[index] / 1000.0)
        model.external_input_ex_ex = value
        model.external_input_in_ex = value.copy()
        return original(state, coupling, local_coupling)

    sim.model.dfun = MethodType(with_drive, sim.model)
    delivered = {}
    if amplitude:
        grid = np.arange(0, duration, 0.1)
        sim.stimulus.configure_time(grid)
        active = np.flatnonzero(np.asarray(sim.stimulus.temporal_pattern).ravel())
        expected_onset = onset_ms
        if not len(active) or abs(grid[active[0]] - expected_onset) > 0.100001:
            raise ValueError("Delivered pulse onset differs from requested onset")
        delivered = dict(
            actual_onset_ms=float(grid[active[0]]),
            delivered_duration_ms=float(len(active) * 0.1),
        )
    return (
        sim,
        labels,
        source,
        dict(
            model_parameters=pm,
            executed_parameters={
                key: np.asarray(getattr(sim.model, key)).tolist()
                for key in (
                    "P_e",
                    "P_i",
                    "b_e",
                    "b_i",
                    "a_e",
                    "a_i",
                    "tau_w_e",
                    "tau_w_i",
                    "T",
                    "Q_e",
                    "Q_i",
                    "tau_e",
                    "tau_i",
                    "C_m",
                    "g_K_e",
                    "g_Na_e",
                    "g_K_i",
                    "g_Na_i",
                    "K_ext_e",
                    "K_ext_i",
                    "weight_noise",
                )
            },
            weights_sha256=hashlib.sha256(
                np.ascontiguousarray(cfg.weights).tobytes()
            ).hexdigest(),
            tracts_sha256=hashlib.sha256(
                np.ascontiguousarray(cfg.tract_lengths).tobytes()
            ).hexdigest(),
            actual_drive_mean_hz=float(trace.mean()),
            actual_drive_sd_hz=float(trace.std()),
            zero_drive_fraction=float(np.mean(trace == 0)),
            **delivered,
        ),
    )


def simulate(order, g, drive, amplitude, dataset, duration):
    """Check at 0.1 ms; retain 1-ms data, including partial failed recordings."""
    started = time.monotonic()
    sim, labels, source, info = make_sim(order, g, drive, amplitude, dataset, duration)
    times, states = [], []
    peak = 0.0
    covmin = 0.0
    consecutive = np.zeros(90, int)
    longest = consecutive.copy()
    info.update(status="complete")
    try:
        for step, samples in enumerate(sim(simulation_length=duration), 1):
            t, x = samples[0]
            x = x[:, :, 0]
            if not np.isfinite(x).all():
                raise FloatingPointError(f"Nonfinite state at {t} ms")
            peak = max(peak, float(x[:2].max() * 1000))
            if t >= 3000:
                consecutive = np.where((x[:2] > 0.15).any(axis=0), consecutive + 1, 0)
                longest = np.maximum(longest, consecutive)
            if order == 2:
                eig = 0.5 * (x[2] + x[4] - np.hypot(x[2] - x[4], 2 * x[3]))
                covmin = min(covmin, float(eig.min()))
            if step % 10 == 0:
                times.append(float(t))
                states.append(x.copy())
    except Exception as exc:
        info.update(status="failed", error=repr(exc))
    info.update(
        peak_EI_hz=peak,
        min_covariance_eigenvalue_khz2=covmin if order == 2 else None,
        covariance_valid=bool(covmin >= -1e-12) if order == 2 else None,
        sustained_high=bool((longest * 0.1 >= 500).any()),
        elapsed_seconds=time.monotonic() - started,
    )
    return np.asarray(times), np.asarray(states), labels, source, info


def metrics(t, x, control, source, onset):
    """Descriptive propagation thresholds, not statistical significance."""
    pre = (t >= onset - 400) & (t < onset - 100)
    post = (t >= onset) & (t <= onset + 300)
    late = t >= 9000
    if not (pre.any() and post.any() and late.any()):
        return {}
    response = (x[:, 0] - control[:, 0]) * 1000
    peak = np.abs(response[post]).max(axis=0)
    remote = np.arange(90) != source
    threshold = max(0.1, 0.01 * peak[source])
    control_mean = control[pre, 0].mean(axis=0) * 1000
    return dict(
        baseline_mean_E_hz=float(control_mean.mean()),
        baseline_mean_I_hz=float(control[pre, 1].mean() * 1000),
        baseline_drift_hz=float(
            np.max(
                np.abs(
                    control[(t >= 5000) & (t < 6000), 0].mean(axis=0)
                    - control[(t >= 4000) & (t < 5000), 0].mean(axis=0)
                )
            )
            * 1000
        ),
        source_peak_response_hz=float(peak[source]),
        remote_max_response_hz=float(peak[remote].max()),
        remote_mean_peak_hz=float(peak[remote].mean()),
        recruited_remote=int((peak[remote] >= threshold).sum()),
        descriptive_threshold_hz=float(threshold),
        late_mean_abs_response_hz=float(np.abs(response[late]).mean()),
        source_min_post_E_hz=float(x[post, 0, source].min() * 1000),
        source_suppressed_fraction_0_300=float(
            np.mean(x[post, 0, source] * 1000 < 0.2 * control_mean[source])
        ),
    )


def plot_trial(folder, tag, t, x, control, labels, source, row):
    """Rainbow absolute E and paired response, unmasked map, and recovery."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    onset = row["actual_onset_ms"]
    keep = (t >= onset - 100) & (t <= onset + 300)
    if not keep.any():
        return
    rel = t - onset
    response = (x[:, 0] - control[:, 0]) * 1000
    fig, axes = plt.subplots(1, 4, figsize=(19, 4), layout="constrained")
    colors = plt.cm.turbo(np.linspace(0, 1, 90))
    for i, color in enumerate(colors):
        axes[0].plot(rel[keep], x[keep, 0, i] * 1000, c=color, lw=0.5, alpha=0.65)
        axes[1].plot(rel[keep], response[keep, i], c=color, lw=0.5, alpha=0.65)
    axes[0].plot(
        rel[keep], x[keep, 0, source] * 1000, c="black", lw=1.0, label="Stimulated"
    )
    axes[0].legend(fontsize=7)
    axes[0].set(title="All regions: actual E", ylabel="Firing rate (Hz)")
    axes[1].set(title="Stimulated − matched control", ylabel="Evoked E (Hz)")
    vmax = max(float(np.abs(response[keep]).max()), 1e-9)
    image = axes[2].imshow(
        response[keep].T,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[rel[keep][0], rel[keep][-1], 89.5, -0.5],
    )
    axes[2].axhline(source, color="limegreen", lw=1)
    axes[2].set(title="All 90 sources: no mask", ylabel="AAL90 index")
    fig.colorbar(image, ax=axes[2], label="Evoked E (Hz)")
    recovery = t >= onset - 100
    for i in range(90):
        axes[3].plot(
            rel[recovery] / 1000,
            x[recovery, 0, i] * 1000,
            c=colors[i],
            lw=0.4,
            alpha=0.5,
        )
    axes[3].plot(
        rel[recovery] / 1000,
        control[recovery, 0].mean(axis=1) * 1000,
        c="black",
        ls="--",
        label="Control mean",
    )
    axes[3].set(title="Full recovery: actual E", xlabel="Time after onset (s)")
    axes[3].legend(fontsize=7)
    for ax in axes[:3]:
        ax.axvspan(0, 10, color="orange", alpha=0.12)
        ax.set_xlabel("Time after delivered onset (ms)")
    fig.suptitle(
        f"Split leak, order {row['order']} | G={row['G']:.3f} | drive={row['drive_hz']:.4f} Hz | "
        f"pulse={row['amplitude_hz_per_ms']} Hz/ms × 10 ms | {labels[source]}\n"
        f"High-state flag: {row['sustained_high']}; covariance valid: {row['covariance_valid']} | map scale varies by panel"
    )
    fig.savefig(folder / (tag + ".png"), dpi=160)
    fig.savefig(folder / (tag + ".pdf"))
    plt.close(fig)


def run_point(payload):
    order, gi, di, g, drive, output, dataset, duration = payload
    folder = Path(output) / f"order{order}_g{gi:02d}_d{di:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        return json.loads((folder / "summary.json").read_text())
    rows = []
    control = None
    control_t = None
    for amp in (0.0, 0.55, 0.7):
        tag = f"a{amp:g}"
        t, x, labels, source, info = simulate(
            order, g, drive, amp, Path(dataset), duration
        )
        row = dict(
            order=order,
            gi=gi,
            di=di,
            G=g,
            drive_hz=drive,
            amplitude_hz_per_ms=amp,
            source_index=source,
            **info,
        )
        np.savez_compressed(
            folder / (tag + ".npz"),
            time_ms=t,
            state=x,
            labels=labels,
            source_index=source,
        )
        if amp == 0:
            control = x
            control_t = t
        elif (
            control is not None
            and x.shape == control.shape
            and np.array_equal(t, control_t)
            and len(t)
        ):
            onset = row["actual_onset_ms"]
            row.update(metrics(t, x, control, source, onset))
            row["control_sustained_high"] = rows[0]["sustained_high"]
            row["control_covariance_valid"] = rows[0]["covariance_valid"]
            row["prestim_max_pair_difference_hz"] = float(
                np.max(np.abs(x[t < onset, :2] - control[t < onset, :2])) * 1000
            )
            if row["prestim_max_pair_difference_hz"] > 1e-10:
                raise ValueError("Paired trajectories differ before delivered pulse")
            plot_trial(folder, tag, t, x, control, labels, source, row)
        write_json(folder / (tag + ".json"), row)
        rows.append(row)
        print(
            f"order={order} G={g:.4f} drive={drive:.4f} amp={amp}: {row['status']} peak={row['peak_EI_hz']:.2f}",
            flush=True,
        )
    write_json(folder / "summary.json", rows)
    return rows


def analyse(output):
    """Save numeric heatmaps and flag-valid candidate rankings, not fits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r
        for p in sorted(output.glob("order*/summary.json"))
        for r in json.loads(p.read_text())
    ]
    write_json(output / "summary.json", rows)
    protocol = json.loads((output / "manifest.json").read_text())
    gs, drives = protocol["G"], protocol["drive_hz"]
    candidates = []
    for order in (1, 2):
        for amp in (0.55, 0.7):
            selected = [
                r
                for r in rows
                if r["order"] == order and r["amplitude_hz_per_ms"] == amp
            ]
            names = [
                "baseline_mean_E_hz",
                "remote_mean_peak_hz",
                "recruited_remote",
                "late_mean_abs_response_hz",
                "source_suppressed_fraction_0_300",
                "peak_EI_hz",
            ]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
            for ax, name in zip(axes.flat, names):
                grid = np.full((len(drives), len(gs)), np.nan)
                for r in selected:
                    if name in r:
                        grid[r["di"], r["gi"]] = r[name]
                if len(drives) == 1:
                    ax.plot(gs, grid[0], "o-")
                    ax.set(
                        title=name.replace("_", " "),
                        xlabel="G",
                        ylabel=name.replace("_", " "),
                    )
                    continue
                im = ax.imshow(grid, origin="lower", aspect="auto")
                ax.set_xticks(range(len(gs)), [f"{v:.2f}" for v in gs], rotation=45)
                ax.set_yticks(range(len(drives)), [f"{v:.3f}" for v in drives])
                ax.set(
                    title=name.replace("_", " "),
                    xlabel="G",
                    ylabel="Requested drive (Hz)",
                )
                fig.colorbar(im, ax=ax)
            fig.suptitle(
                f"Order {order}, {amp} Hz/ms | descriptive screening; invalid runs retained"
            )
            fig.savefig(output / f"heatmaps_order{order}_a{amp}.png", dpi=180)
            fig.savefig(output / f"heatmaps_order{order}_a{amp}.pdf")
            plt.close(fig)
            for r in selected:
                valid = (
                    r["status"] == "complete"
                    and not r["sustained_high"]
                    and not r.get("control_sustained_high", True)
                    and r["covariance_valid"] is not False
                    and r.get("control_covariance_valid") is not False
                )
                if valid and "remote_mean_peak_hz" in r:
                    candidates.append(
                        dict(
                            order=order,
                            amplitude=amp,
                            G=r["G"],
                            drive_hz=r["drive_hz"],
                            remote_mean_peak_hz=r["remote_mean_peak_hz"],
                            recovery_hz=r["late_mean_abs_response_hz"],
                            suppressed_fraction=r["source_suppressed_fraction_0_300"],
                        )
                    )
    write_json(output / "flag_valid_candidates.json", candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument(
        "--drive-hz",
        type=float,
        nargs="+",
        default=None,
        help="Requested external drive values; e.g. --drive-hz 0.18",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One short point/order; separate folder required",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.analyse:
        analyse(args.output)
        return
    dataset = ROOT / "data/doc_data/converted_structural_invnodevol_control_max"
    sources = [
        Path(__file__),
        Path(base.__file__),
        ROOT / "experiments/gnak_order_comparison/model_adapter.py",
        ROOT / "src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_gK_gNa.py",
    ]
    manifest = dict(
        subject="control/c0001",
        model="native corrected split leak; new Toolkit-order TF",
        occupancy=0,
        b_pA=5,
        tau_w_ms=500,
        T_ms=20,
        dt_ms=0.1,
        saved_dt_ms=1,
        duration_ms=100 if args.smoke else 10000,
        onset_ms=6000,
        pulse_ms=10,
        amplitudes_hz_per_ms=[0.55, 0.7],
        G=np.linspace(0.3, 1, 10).tolist(),
        drive_hz=args.drive_hz
        if args.drive_hz is not None
        else np.linspace(0.001, 0.315, 10).tolist(),
        seed=base.SEED,
        noise="publication shared nonnegative OU afferent rate retained; regional internal OU enabled, weight_noise=0.001",
        internal_noise_seed=base.SEED + 10000,
        normalization=json.loads((dataset / "index.json").read_text())[
            "connectivity_normalization"
        ],
        speed_m_s=4,
        source="Supp_Motor_Area_L",
        expected_simulations=6
        if args.smoke
        else 60 * (len(args.drive_hz) if args.drive_hz is not None else 10),
        derivative_scaling="order2 J/1000 H/1e6; experimental, NOT unit conversion",
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Protocol/source changed; choose a fresh output directory")
    write_json(path, manifest)
    jobs = [
        (
            order,
            gi,
            di,
            g,
            drive,
            str(args.output),
            str(dataset),
            manifest["duration_ms"],
        )
        for gi, g in enumerate(manifest["G"])
        for di, drive in enumerate(manifest["drive_hz"])
        for order in (1, 2)
        if not args.smoke or (gi == 0 and di == 0)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_point, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            write_json(
                args.output / "progress.json",
                dict(completed_points=n, total_points=len(jobs)),
            )
    analyse(args.output)


if __name__ == "__main__":
    main()
