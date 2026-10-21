"""Dalla Porta-inspired second-order stimulation on subject-specific AAL90 SC.

Run from the repository root.  Simulation and PCI analysis are separate:

    python experiments/della_porta_second_order/study.py prepare
    python experiments/della_porta_second_order/study.py run --workers 6
    python experiments/della_porta_second_order/study.py status
    python experiments/della_porta_second_order/analyze.py

The 30-ms, 1-Hz/ms pulse follows the supplied *notebook code* (weight=0.001
kHz/ms), not the paper's ambiguous prose description of a 3-Hz pulse.  The
second-order RHS uses the installed-TVB-style finite-difference attenuation
already implemented by ``make_sim``; no claim of unit correctness is made.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TVB_USER_HOME", str(ROOT / ".tvb-temp"))
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import numpy as np
from tvb.simulator.monitors import TemporalAverage

from sweep_split_leak_tep_drive import make_sim

DATASET = ROOT / "data/doc_data/converted_structural_invnodevol_control_max"
BANK = ROOT / "results/della_porta_second_order_aal90_5x50_v1"
GROUPS = {"control": ("CNT", 5.0), "emcs": ("EMCS", 15.0),
          "mcs": ("MCS", 30.0), "uws": ("UWS", 40.0)}
OCCUPANCIES = (0.0, 0.25, 0.5, 0.766)
LEAKS = (-63.0, -62.0)
TRIALS = 50
MASTER_SEED = 20261021
PROTOCOL = dict(model_order=2, model="split_leak_gNa_gK", G=0.3,
                external_drive_hz=0.315, weight_noise=0.0002,
                regional_ou_tau_ms=2.0, tau_w_ms=500.0,
                source_label="Parietal_Sup_L", pulse_hz_per_ms=1.0,
                pulse_duration_ms=30.0, simulation_ms=8000.0,
                onset_range_ms=[2500, 7500], transient_discard_ms=2000.0,
                monitor_period_ms=1.0, saved_window_ms=[-500, 500],
                pci_pre_ms=[-300, 0], pci_post_ms=[0, 300],
                structural_normalization="shared_control_maximum_edge",
                structural_divisor=122.63607788085938,
                second_order_derivatives="historical TVB denominators /1000 and /1000000; experimental, not validated unit conversion",
                covariance_indexing="Toolkit corrected", covariance_clipping=False,
                b_pa={k: b for k, (_, b) in GROUPS.items()},
                occupancy=list(OCCUPANCIES), leaks_mv=list(LEAKS), trials=TRIALS,
                receptor_rule="regional min-max 5HT2A; gNa fixed, gK shifted toward E_L+1.8mV; inhibitory target -64.4mV")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def select_subjects(dataset: Path) -> dict[str, list[str]]:
    """Five damage-stratified subjects per cohort; no diagnostic labels in fit."""
    info = json.loads((dataset / "normalization_metadata.json").read_text())
    out = {}
    for cohort in GROUPS:
        candidates = sorted((r for r in info["subjects"] if r["cohort"] == cohort),
                            key=lambda r: (r["zero_entries"], r["subject_id"]))
        indices = np.rint(np.linspace(0, len(candidates) - 1, 5)).astype(int)
        out[cohort] = [candidates[i]["subject_id"] for i in indices]
        if len(set(out[cohort])) != 5:
            raise ValueError(f"Nonunique selection: {cohort}")
    return out


def manifest_for(dataset: Path) -> dict:
    index = json.loads((dataset / "index.json").read_text())
    norm = index["connectivity_normalization"]
    if norm["scheme"] != PROTOCOL["structural_normalization"] or not np.isclose(
        norm["divisor"], PROTOCOL["structural_divisor"], rtol=0, atol=1e-9
    ):
        raise ValueError("Expected the shared-control-maximum AAL90 dataset")
    subjects = select_subjects(dataset)
    cases = [dict(cohort=cohort, group=GROUPS[cohort][0], subject=subject,
                  b_pa=GROUPS[cohort][1], leak_mv=leak, occupancy=occ)
             for leak in LEAKS for cohort in GROUPS
             for subject in subjects[cohort] for occ in OCCUPANCIES]
    model_paths = [Path(__file__), ROOT / "scripts/sweep_split_leak_tep_drive.py",
                   ROOT / "scripts/b_sweep_publication_run.py",
                   ROOT / "experiments/gnak_order_comparison/model_adapter.py",
                   ROOT / "src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_gK_gNa.py",
                   ROOT / "src/tvbtoolkit/whole_brain/simulation.py"]
    return dict(protocol=PROTOCOL, subjects=subjects, cases=cases,
                dataset_index_sha256=sha256(dataset / "index.json"),
                receptor_sha256=sha256(ROOT / "data/receptors/hansen_receptors_aal90.csv"),
                source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in model_paths})


def seed_for(case: dict, trial: int) -> int:
    # Pair seed and onset across the two leak settings and four occupancies.
    text = f"{MASTER_SEED}:{case['cohort']}:{case['subject']}:{trial}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def paths(bank: Path, case_index: int, trial: int) -> tuple[Path, Path]:
    root = bank / "recordings" / f"case_{case_index:03d}"
    return root / f"trial_{trial:02d}.npz", root / f"trial_{trial:02d}.json"


def run_trial(payload: tuple) -> dict:
    bank_str, dataset_str, case_index, case, trial = payload
    bank, dataset = Path(bank_str), Path(dataset_str)
    data_path, meta_path = paths(bank, case_index, trial)
    seed = seed_for(case, trial)
    onset = int(np.random.default_rng(seed).integers(2500, 7501))
    if data_path.exists() and meta_path.exists():
        row = json.loads(meta_path.read_text())
        if row["case"] != case or row["seed"] != seed or row["onset_ms"] != onset:
            raise ValueError(f"Existing trial does not match manifest: {data_path}")
        with np.load(data_path) as saved:
            if saved["E_hz"].shape != (1000, 90):
                raise ValueError(f"Incomplete trial: {data_path}")
        return row
    sim, labels, source, delivered = make_sim(
        2, PROTOCOL["G"], PROTOCOL["external_drive_hz"],
        PROTOCOL["pulse_hz_per_ms"], dataset, duration=8000.0,
        seed=seed, onset_ms=onset, source_label=PROTOCOL["source_label"],
        pulse_ms=30.0, column_normalize=False,
        b_map_pa=np.full(90, case["b_pa"]),
        weight_noise=PROTOCOL["weight_noise"], tau_ou_ms=2.0,
        leak_e_mv=case["leak_mv"], cohort=case["cohort"],
        subject=case["subject"], receptor_occupancy=case["occupancy"],
        drug_leak_e_mv=case["leak_mv"] + 1.8)
    if abs(delivered["actual_onset_ms"] - onset) > 0.100001 or abs(
        delivered["delivered_duration_ms"] - 30.0) > 0.100001:
        raise ValueError(f"Incorrect delivered pulse: {delivered}")
    monitor = TemporalAverage(period=1.0, variables_of_interest=np.arange(8))
    monitor.config_for_sim(sim)
    sim.monitors = tuple(sim.monitors) + (monitor,)
    times, states = [], []
    peak = 0.0
    high_steps = 0
    longest_high, current_high = 0, 0
    min_cov_eigenvalue = 0.0
    cov_invalid_steps = 0
    for samples in sim(simulation_length=8000.0):
        t, state = samples[0]
        x = state[:, :, 0]
        if not np.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite second-order state: {case}, trial {trial}")
        peak = max(peak, float(x[:2].max() * 1000.0))
        if t >= 2000:
            high = bool((x[:2] > .150).any())
            high_steps += int(high)
            current_high = current_high + 1 if high else 0
            longest_high = max(longest_high, current_high)
            # Covariance PSD is checked at each 1-ms step to limit overhead.
            if round(float(t) * 10) % 10 == 0:
                minimum = .5 * (x[2] + x[4] - np.hypot(x[2] - x[4], 2*x[3]))
                local_min = float(minimum.min())
                min_cov_eigenvalue = min(min_cov_eigenvalue, local_min)
                cov_invalid_steps += int(local_min < -1e-12)
        if samples[-1] is not None:
            time_avg, averaged = samples[-1]
            if onset - 500 <= time_avg < onset + 500:
                times.append(float(time_avg) - onset)
                states.append((averaged[:2, :, 0] * 1000).astype(np.float32))
    time_ms = np.asarray(times)
    state_hz = np.asarray(states)
    if state_hz.shape != (1000, 2, 90) or np.sum((-300 <= time_ms) & (time_ms < 0)) != 300 or np.sum((0 <= time_ms) & (time_ms < 300)) != 300:
        raise ValueError(f"Incomplete peri-onset recording: {case}, {state_hz.shape}")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, time_ms=time_ms, E_hz=state_hz[:, 0],
                        I_hz=state_hz[:, 1], labels=labels, source=source,
                        seed=seed, onset_ms=onset)
    tmp.replace(data_path)
    row = dict(case=case, case_index=case_index, trial=trial, seed=seed,
               onset_ms=onset, source_index=source, source_label=str(labels[source]),
               peak_EI_hz=peak, high_rate_post_transient_steps=high_steps,
               longest_high_rate_post_transient_ms=longest_high*.1,
               min_covariance_eigenvalue_khz2=min_cov_eigenvalue,
               covariance_invalid_1ms_samples=cov_invalid_steps,
               delivered_pulse_duration_ms=delivered["delivered_duration_ms"],
               weights_sha256=delivered["weights_sha256"],
               tracts_sha256=delivered["tracts_sha256"])
    write_json(meta_path, row)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run", "status"))
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-cases", type=int, default=None,
                   help="Pilot prefix only; analysis still requires the complete bank")
    p.add_argument("--max-trials", type=int, default=TRIALS)
    args = p.parse_args()
    manifest_path = args.bank / "manifest.json"
    if args.command == "prepare":
        planned = manifest_for(args.dataset)
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != planned:
            raise ValueError("Bank already exists with different inputs/code")
        write_json(manifest_path, planned)
        print(f"{sum(map(len,planned['subjects'].values()))} subjects; {len(planned['cases'])} cases; {len(planned['cases'])*TRIALS} trials")
        print("Subjects:", planned["subjects"])
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest != manifest_for(args.dataset):
        raise ValueError("Model/dataset/source changed since prepare; use a new bank")
    complete = sum(paths(args.bank, ci, ti)[0].exists() and paths(args.bank, ci, ti)[1].exists()
                   for ci in range(len(manifest["cases"])) for ti in range(TRIALS))
    print(f"{complete}/{len(manifest['cases'])*TRIALS} trial recordings complete", flush=True)
    if args.command == "status":
        return
    if args.workers < 1 or not 1 <= args.max_trials <= TRIALS:
        p.error("workers and max-trials must be positive")
    cases = manifest["cases"][:args.max_cases] if args.max_cases else manifest["cases"]
    jobs = [(str(args.bank), str(args.dataset), ci, case, ti)
            for ci, case in enumerate(cases) for ti in range(args.max_trials)
            if not all(path.exists() for path in paths(args.bank, ci, ti))]
    print(f"Running {len(jobs)} missing trials with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trial, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 10 == 0 or i == len(futures):
                print(f"{i}/{len(futures)} newly completed", flush=True)


if __name__ == "__main__":
    main()
