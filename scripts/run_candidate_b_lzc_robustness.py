#!/usr/bin/env python3
"""Candidate-matched adaptation sweep across SNN, mean-field and AAL90 scales.

The spiking network and single-region mean field are mechanistic references and
are run once per ``b_e`` value. They have no patient connectome and are therefore
not duplicated per subject. The whole-brain model is run for the seven
outcome-blind PCI calibration candidates using their native inverse-node-volume
AAL90 connectivity and matched tract lengths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import b_sweep_publication_run as reference  # noqa: E402
from brain_act_hybrid_common import BASE_PARAMETER_MODEL_NEW, COND_COLORS  # noqa: E402
from tvbtoolkit.complexity.measures import lzc_multichannel  # noqa: E402
from tvbtoolkit.core.config import WholeBrainConfig  # noqa: E402
from tvbtoolkit.datasets.brain_act import load_subject_structural  # noqa: E402
from tvbtoolkit.datasets.structural_provenance import (  # noqa: E402
    validate_native_invnodevol_dataset,
)
from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation  # noqa: E402
from tvbtoolkit.workflows.brain_act_dual_domain_parallel import (  # noqa: E402
    _apply_damage_parity,
    worker_initializer,
)


DEFAULT_SUBJECTS = (
    "control:c0015",
    "emcs:e0003",
    "emcs:e0008",
    "mcs:m0005",
    "mcs:m0009",
    "uws:u0020",
    "uws:u0038",
)
DEFAULT_B_VALUES = (5.0, 10.0, 25.0, 45.0, 65.0, 85.0, 105.0, 125.0)
CONDITION = {"control": "CNT", "emcs": "EMCS", "mcs": "MCS", "uws": "UWS"}
RATE_PERIOD_MS = 3.9
RATE_BAND_HZ = (2.0, 80.0)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 7,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=REPO_ROOT / "data/doc_data/converted_structural_invnodevol_native",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "notebooks/outputs/candidate_b_lzc_robustness",
    )
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--b-values", type=float, nargs="+", default=list(DEFAULT_B_VALUES))
    parser.add_argument("--duration-ms", type=float, default=22_000.0)
    parser.add_argument("--transient-ms", type=float, default=4_000.0)
    parser.add_argument(
        "--coupling-g", type=float, default=0.0025,
        help="Global coupling; default is the lowest calibrated native-SC value.",
    )
    parser.add_argument("--saturation-hz", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--snn-n-total", type=int, default=10_000)
    parser.add_argument("--skip-reference-scales", action="store_true")
    parser.add_argument("--skip-whole-brain", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_subject(spec: str) -> tuple[str, str]:
    try:
        cohort, subject = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Subject must be cohort:id, got {spec!r}") from exc
    cohort = cohort.strip().lower()
    if cohort == "coma":
        raise ValueError("COMA is excluded from this candidate robustness panel.")
    if cohort not in CONDITION:
        raise ValueError(f"Unsupported cohort {cohort!r}.")
    return cohort, subject.strip()


def _metadata(dataset_root: Path, subjects: list[tuple[str, str]]) -> list[dict[str, Any]]:
    index = json.loads((dataset_root / "index.json").read_text(encoding="utf-8"))
    lookup = {
        (str(row["cohort"]), str(row["subject_id"])): row
        for row in index["subjects"]
    }
    rows = []
    for cohort, subject in subjects:
        item = lookup[(cohort, subject)]
        rows.append({
            "cohort": cohort,
            "condition": CONDITION[cohort],
            "subject_id": subject,
            "stage": str(item.get("stage", "unknown") or "unknown"),
            "sedation": str(item.get("sedation", "unknown") or "unknown"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _uniform_time(x: np.ndarray, t_ms: np.ndarray, period_ms: float = RATE_PERIOD_MS) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t_ms, dtype=float).reshape(-1)
    target = np.arange(float(t[0]), float(t[-1]) + period_ms / 2.0, period_ms)
    return np.column_stack([np.interp(target, t, x[:, channel]) for channel in range(x.shape[1])])


def _rate_lzc(x: np.ndarray, t_ms: np.ndarray) -> float:
    sampled = _uniform_time(x, t_ms)
    fs = 1000.0 / RATE_PERIOD_MS
    sos = butter(4, RATE_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, sampled, axis=0)
    return float(lzc_multichannel(filtered, shuffle_seed=0))


def _reference_paths(root: Path, b: float) -> Path:
    return root / "reference_scales" / f"b_{b:g}.npz"


def _run_reference_scales(args: argparse.Namespace, b: float) -> dict[str, Any]:
    path = _reference_paths(args.output_root, b)
    if path.exists() and not args.overwrite:
        with np.load(path, allow_pickle=False) as cached:
            compatible = (
                np.isclose(float(cached["b_e_pa"][0]), b)
                and np.isclose(float(cached["duration_ms"][0]), args.duration_ms)
                and np.isclose(float(cached["transient_ms"][0]), args.transient_ms)
                and int(cached["seed"][0]) == args.seed
                and int(cached["snn_n_total"][0]) == args.snn_n_total
            )
        if compatible:
            return {"status": "cached", "path": str(path)}
        print(f"[b-lzc] incompatible reference cache will be replaced: {path}", flush=True)
    reference.SIM_DURATION_MS = float(args.duration_ms)
    reference.CUT_TRANSIENT_MS = float(args.transient_ms)
    reference.SEED = int(args.seed)
    reference.SNN_NTOT = int(args.snn_n_total)
    trace_snn, _ = reference.make_shared_ou_traces()
    snn, _ = reference.run_snn_block(int(round(b)), snn_mf_trace_hz=trace_snn)
    mf, _ = reference.run_mf_block(int(round(b)), snn_mf_trace_hz=trace_snn)
    snn_signal = np.column_stack([snn["rate_exc_hz"], snn["rate_inh_hz"]])
    mf_signal = np.column_stack([mf["ve_hz"], mf["vi_hz"]])
    lzc_snn = _rate_lzc(snn_signal, snn["time_ms"])
    lzc_mf = _rate_lzc(mf_signal, mf["time_ms"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        b_e_pa=np.array([b]),
        duration_ms=np.array([args.duration_ms]),
        transient_ms=np.array([args.transient_ms]),
        seed=np.array([args.seed]), snn_n_total=np.array([args.snn_n_total]),
        snn_time_ms=snn["time_ms"], snn_rate_hz=snn_signal,
        snn_raster_exc_t_ms=np.asarray(snn["raster_exc"][0], dtype=float),
        snn_raster_exc_i=np.asarray(snn["raster_exc"][1], dtype=int),
        snn_raster_inh_t_ms=np.asarray(snn["raster_inh"][0], dtype=float),
        snn_raster_inh_i=np.asarray(snn["raster_inh"][1], dtype=int),
        mf_time_ms=mf["time_ms"], mf_rate_hz=mf_signal, mf_adaptation_pa=mf["W_pa"],
        lzc_snn=np.array([lzc_snn]), lzc_mean_field=np.array([lzc_mf]),
    )
    return {"status": "completed", "path": str(path), "lzc_snn": lzc_snn, "lzc_mean_field": lzc_mf}


def _whole_brain_job(job: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    path = Path(job["path"])
    if path.exists() and not bool(job["overwrite"]):
        with np.load(path, allow_pickle=False) as cached:
            compatible = (
                np.isclose(float(cached["coupling_g"][0]), float(job["coupling_g"]))
                and np.isclose(float(cached["b_e_pa"][0]), float(job["b_e_pa"]))
                and int(cached["seed"][0]) == int(job["seed"])
                and np.isclose(float(cached["duration_ms"][0]), float(job["duration_ms"]))
                and np.isclose(float(cached["transient_ms"][0]), float(job["transient_ms"]))
            )
        if compatible:
            return {"status": "cached", "path": str(path)}
        print(f"[b-lzc] incompatible cache will be replaced: {path}", flush=True)
    weights, lengths, atlas, _ = load_subject_structural(
        subject_id=job["subject_id"], cohort=job["cohort"],
        dataset_root=job["dataset_root"], validate=True,
        enforce_symmetry=True, zero_diagonal=True, nonfinite="raise",
    )
    weights, lengths, zero_fraction = _apply_damage_parity(
        weights, lengths, job["cohort"], normalize_subject_max=False
    )
    model = deepcopy(BASE_PARAMETER_MODEL_NEW)
    model.update({"b_e": float(job["b_e_pa"]), "noise_alpha": 0.0, "shared_noise_mode": "none"})
    cfg = WholeBrainConfig(
        simulation_length_ms=float(job["duration_ms"]), dt_ms=0.1,
        conduction_speed=4.0, coupling_strength=float(job["coupling_g"]),
        zerlaut_order=2, stochastic_integrator=True,
        monitor_mode="temporal_average", temporal_average_period_ms=RATE_PERIOD_MS,
        monitor_variables=(0, 1), weights=weights, tract_lengths=lengths,
        connectivity_normalization="none", parameter_overrides={"parameter_model": model},
    )
    result = run_whole_brain_simulation(cfg, seed=int(job["seed"]))
    keep = np.asarray(result.time_ms) >= float(job["transient_ms"])
    time_ms = np.asarray(result.time_ms)[keep] - float(job["transient_ms"])
    rate_e_hz = np.asarray(result.raw)[keep] * 1000.0
    rate_i_hz = np.asarray(result.raw_inh)[keep] * 1000.0
    lzc_raw = _rate_lzc(rate_e_hz, time_ms)
    max_rate_hz = float(np.max(rate_e_hz))
    saturated_fraction = float(np.mean(rate_e_hz >= float(job["saturation_hz"])))
    valid_for_lzc = bool(
        max_rate_hz < float(job["saturation_hz"])
        and saturated_fraction == 0.0
    )
    lzc = lzc_raw if valid_for_lzc else float("nan")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, time_ms=time_ms, rate_e_hz=rate_e_hz, rate_i_hz=rate_i_hz,
        region_labels=np.asarray(atlas.labels), b_e_pa=np.array([job["b_e_pa"]]),
        coupling_g=np.array([job["coupling_g"]]), seed=np.array([job["seed"]]),
        duration_ms=np.array([job["duration_ms"]]),
        transient_ms=np.array([job["transient_ms"]]),
        lzc=np.array([lzc]), lzc_raw=np.array([lzc_raw]),
        valid_for_lzc=np.array([valid_for_lzc]),
        saturation_hz=np.array([job["saturation_hz"]]),
        saturated_fraction=np.array([saturated_fraction]),
        zero_fraction=np.array([zero_fraction]),
        cohort=np.array([job["cohort"]]), subject_id=np.array([job["subject_id"]]),
        sedation=np.array([job["sedation"]]), stage=np.array([job["stage"]]),
    )
    return {
        "status": "completed", "path": str(path), "cohort": job["cohort"],
        "condition": job["condition"], "subject_id": job["subject_id"],
        "sedation": job["sedation"], "stage": job["stage"],
        "b_e_pa": job["b_e_pa"], "lzc": lzc, "lzc_raw": lzc_raw,
        "valid_for_lzc": valid_for_lzc,
        "mean_rate_hz": float(np.mean(rate_e_hz)),
        "max_rate_hz": max_rate_hz, "saturated_fraction": saturated_fraction,
        "zero_fraction": zero_fraction, "runtime_s": perf_counter() - start,
    }


def _collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "whole_brain").glob("*/*/b_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            rows.append({
                "scale": "whole_brain", "cohort": str(data["cohort"][0]),
                "condition": CONDITION[str(data["cohort"][0])],
                "subject_id": str(data["subject_id"][0]),
                "sedation": str(data["sedation"][0]), "stage": str(data["stage"][0]),
                "b_e_pa": float(data["b_e_pa"][0]), "lzc": float(data["lzc"][0]),
                "lzc_raw": float(data["lzc_raw"][0]),
                "valid_for_lzc": bool(data["valid_for_lzc"][0]),
                "mean_rate_hz": float(np.mean(data["rate_e_hz"])),
                "max_rate_hz": float(np.max(data["rate_e_hz"])),
                "saturated_fraction": float(data["saturated_fraction"][0]),
            })
    for path in sorted((root / "reference_scales").glob("b_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            common = {"cohort": "reference", "condition": "reference", "subject_id": "not_subject_specific", "sedation": "not_applicable", "stage": "not_applicable", "b_e_pa": float(data["b_e_pa"][0]), "mean_rate_hz": float("nan"), "max_rate_hz": float("nan"), "lzc_raw": float("nan"), "valid_for_lzc": True, "saturated_fraction": float("nan")}
            rows.append({**common, "scale": "spiking_network", "lzc": float(data["lzc_snn"][0])})
            rows.append({**common, "scale": "single_region_mean_field", "lzc": float(data["lzc_mean_field"][0])})
    return sorted(rows, key=lambda row: (row["scale"], row["subject_id"], row["b_e_pa"]))


def _plot_lzc(rows: list[dict[str, Any]], root: Path) -> None:
    wb = [row for row in rows if row["scale"] == "whole_brain"]
    refs = [row for row in rows if row["scale"] != "whole_brain"]
    if not wb:
        return
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)
        for scale, color in [("spiking_network", "#3973AC"), ("single_region_mean_field", "#555555")]:
            data = [row for row in refs if row["scale"] == scale]
            if data:
                axes[0].plot([r["b_e_pa"] for r in data], [r["lzc"] for r in data], "o-", ms=3, color=color, label=scale.replace("_", " "))
        axes[0].set_title("A  Reference scales", loc="left")
        axes[0].set_ylabel("Normalized LZc")
        if refs:
            axes[0].legend(fontsize=5)
        for subject in sorted({row["subject_id"] for row in wb}):
            data = [row for row in wb if row["subject_id"] == subject]
            cond = data[0]["condition"]
            sed = data[0]["sedation"]
            axes[1].plot(
                [r["b_e_pa"] for r in data], [r["lzc"] for r in data],
                marker="o", ms=2.5, lw=1.0, color=COND_COLORS[cond],
                ls="--" if sed == "sedated" else "-", alpha=0.9,
                label=f"{cond}:{subject} ({sed.replace('_', ' ')})",
            )
        axes[1].set_title("B  Candidate whole-brain models", loc="left")
        axes[1].legend(fontsize=4.6, ncol=1)
        for sedation, marker, offset in [("non_sedated", "o", -1.0), ("sedated", "s", 1.0)]:
            for condition in ("UWS", "MCS", "EMCS", "CNT"):
                data = [r for r in wb if r["sedation"] == sedation and r["condition"] == condition]
                for row in data:
                    axes[2].scatter(row["b_e_pa"] + offset, row["lzc"], s=12, marker=marker, color=COND_COLORS[condition], alpha=0.75)
            for b_value in sorted({row["b_e_pa"] for row in wb}):
                values = np.asarray([
                    row["lzc"] for row in wb
                    if row["sedation"] == sedation
                    and row["b_e_pa"] == b_value
                    and np.isfinite(row["lzc"])
                ], dtype=float)
                if values.size:
                    sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
                    axes[2].errorbar(
                        b_value + offset, float(values.mean()), yerr=sem,
                        marker=marker, ms=4, color="black", mfc="white",
                        mew=.7, lw=.8, capsize=1.5, zorder=5,
                    )
        axes[2].set_title("C  Sedation-stratified values", loc="left")
        for ax in axes:
            ax.set_xlabel(r"Adaptation $b_e$ (pA)")
        fig_dir = root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "candidate_b_lzc_multiscale.pdf")
        fig.savefig(fig_dir / "candidate_b_lzc_multiscale.png")
        plt.close(fig)


def _plot_activity(root: Path) -> None:
    ref_paths = sorted((root / "reference_scales").glob("b_*.npz"), key=lambda p: float(p.stem.split("_")[1]))
    if ref_paths:
        with mpl.rc_context(RC):
            fig, axes = plt.subplots(len(ref_paths), 3, figsize=(7.2, 1.05 * len(ref_paths)), squeeze=False, sharex="col")
            for index, path in enumerate(ref_paths):
                with np.load(path, allow_pickle=False) as data:
                    b = float(data["b_e_pa"][0]); cut = float(data["transient_ms"][0])
                    keep_e = (data["snn_raster_exc_t_ms"] >= cut) & (data["snn_raster_exc_i"] < 80)
                    axes[index, 0].scatter((data["snn_raster_exc_t_ms"][keep_e] - cut) / 1000, data["snn_raster_exc_i"][keep_e], s=.25, color="#3973AC", rasterized=True)
                    axes[index, 1].plot(data["snn_time_ms"] / 1000, data["snn_rate_hz"][:, 0], color="#3973AC", lw=.6)
                    axes[index, 2].plot(data["mf_time_ms"] / 1000, data["mf_rate_hz"][:, 0], color="#555555", lw=.6)
                    axes[index, 0].set_ylabel(f"{b:g} pA")
            for ax, title in zip(axes[0], ("Raster", "Spiking population rate", "Single-region mean field")):
                ax.set_title(title)
            for ax in axes[-1]: ax.set_xlabel("Time (s)")
            (root / "figures").mkdir(parents=True, exist_ok=True)
            fig.savefig(root / "figures/reference_multiscale_activity.pdf")
            fig.savefig(root / "figures/reference_multiscale_activity.png")
            plt.close(fig)
    for subject_dir in sorted((root / "whole_brain").glob("*/*")):
        paths = sorted(subject_dir.glob("b_*.npz"), key=lambda p: float(p.stem.split("_")[1]))
        if not paths: continue
        with mpl.rc_context(RC):
            fig, axes = plt.subplots(len(paths), 1, figsize=(3.5, .9 * len(paths)), squeeze=False, sharex=True)
            for index, path in enumerate(paths):
                with np.load(path, allow_pickle=False) as data:
                    t = data["time_ms"] / 1000; rate = data["rate_e_hz"]
                    axes[index, 0].plot(t, rate, color="0.65", alpha=.16, lw=.25, rasterized=True)
                    axes[index, 0].plot(t, rate.mean(axis=1), color=COND_COLORS[CONDITION[str(data["cohort"][0])]], lw=1.0)
                    axes[index, 0].set_ylabel(f"{float(data['b_e_pa'][0]):g}")
            axes[0, 0].set_title(f"{subject_dir.parent.name.upper()}:{subject_dir.name} — regional activity", loc="left")
            axes[-1, 0].set_xlabel("Time (s)")
            fig.supylabel(r"$b_e$ (pA) / firing rate (Hz)", fontsize=6)
            fig.savefig(root / "figures" / f"whole_brain_activity_{subject_dir.parent.name}_{subject_dir.name}.pdf")
            fig.savefig(root / "figures" / f"whole_brain_activity_{subject_dir.parent.name}_{subject_dir.name}.png")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.duration_ms <= args.transient_ms:
        raise ValueError("Duration must exceed transient.")
    provenance = validate_native_invnodevol_dataset(args.dataset_root)
    subjects = [_parse_subject(item) for item in (args.subject or DEFAULT_SUBJECTS)]
    metadata = _metadata(args.dataset_root, subjects)
    if any(row["cohort"] == "coma" for row in metadata):
        raise RuntimeError("COMA is excluded from the candidate panel.")
    b_values = sorted(set(float(value) for value in args.b_values))
    planned_reference = 0 if args.skip_reference_scales else len(b_values)
    planned_wb = 0 if args.skip_whole_brain else len(metadata) * len(b_values)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "candidate_subjects.csv", metadata)
    manifest = {
        "purpose": "candidate-matched b_e robustness; not cohort inference",
        "dataset_provenance": provenance, "coma_excluded": True,
        "subjects": metadata, "b_values_pa": b_values,
        "duration_ms": args.duration_ms, "transient_ms": args.transient_ms,
        "whole_brain_coupling_g": args.coupling_g, "seed": args.seed,
        "saturation_hz": args.saturation_hz,
        "saturation_policy": "LZc is saved as lzc_raw for QC but set to NaN in the analysis column if any sample reaches the firing-rate ceiling.",
        "rate_monitor_period_ms": RATE_PERIOD_MS,
        "rate_sampling_hz_effective": 1000.0 / RATE_PERIOD_MS,
        "lzc_band_hz": list(RATE_BAND_HZ),
        "reference_scale_note": "SNN and single-region MF have no subject anatomy and are run once per b_e.",
        "sedation_analysis": "descriptive stratification only; the seven-subject calibration panel is not powered for sedation inference",
        "planned_reference_runs": planned_reference, "planned_whole_brain_runs": planned_wb,
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[b-lzc] subjects={len(metadata)} b_values={b_values}")
    print(f"[b-lzc] planned reference={planned_reference}, whole_brain={planned_wb}")
    if args.dry_run:
        return
    if not args.skip_reference_scales:
        for b in b_values:
            print(f"[b-lzc] reference scales b={b:g}", flush=True)
            _run_reference_scales(args, b)
    if not args.skip_whole_brain:
        jobs = []
        for row in metadata:
            for b in b_values:
                jobs.append({**row, "b_e_pa": b, "dataset_root": str(args.dataset_root),
                    "duration_ms": args.duration_ms, "transient_ms": args.transient_ms,
                    "coupling_g": args.coupling_g, "seed": args.seed,
                    "saturation_hz": args.saturation_hz,
                    "overwrite": args.overwrite,
                    "path": str(args.output_root / "whole_brain" / row["cohort"] / row["subject_id"] / f"b_{b:g}.npz")})
        if int(args.workers) == 1:
            worker_initializer()
            for index, job in enumerate(jobs, 1):
                result = _whole_brain_job(job)
                print(
                    f"[b-lzc] whole brain {index}/{len(jobs)} "
                    f"{result['status']} {result['path']}", flush=True,
                )
        else:
            with ProcessPoolExecutor(
                max_workers=max(1, args.workers), initializer=worker_initializer
            ) as pool:
                futures = [pool.submit(_whole_brain_job, job) for job in jobs]
                for index, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    print(
                        f"[b-lzc] whole brain {index}/{len(futures)} "
                        f"{result['status']} {result['path']}", flush=True,
                    )
    rows = _collect_rows(args.output_root)
    _write_csv(args.output_root / "candidate_b_lzc_metrics.csv", rows)
    _plot_lzc(rows, args.output_root)
    _plot_activity(args.output_root)
    print(f"[b-lzc] complete -> {args.output_root}")


if __name__ == "__main__":
    main()
