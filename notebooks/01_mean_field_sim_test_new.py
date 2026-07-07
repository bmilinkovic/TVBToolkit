#!/usr/bin/env python3
"""SNN ↔ single-region MF parity sweep across the AdEx b_e parameter (v2 update).

This is the updated companion to ``01_mean_field_sim_test.py``. Functionally
the same — sweep ``b_e`` from a CLI-configurable range, run a Brian2 spiking
network and a first-order Zerlaut mean field with **shared OU-correlated
afferent drive**, and produce a 2-column figure (SNN | MF) with one row per
b_e — but with explicit lab v2 conventions:

Corrections / conventions made explicit in this script
------------------------------------------------------

1. **External drive — OU-correlated trace, NOT constant Poisson** (Sacha
   *et al.* 2025 Eq. 20). The same trace is fed to both the SNN
   (``external_rate_hz_trace=``) and the MF (``external_drive_hz=``):

       v_aff(t) = v_drive + σ · ξ(t),   dξ = -(ξ/τ_OU) dt + dW

   * ``v_drive = 0.315 Hz`` (per fiber).
   * ``σ = 3.5 Hz``, ``τ_OU = 5 ms`` (paper convention).

2. **Effective afferent EPSP rate per neuron ≈ 126 Hz** in both models
   (``K_ext_e = N_tot·(1-g)·p_con = 400`` × ``0.315 Hz`` baseline).

3. **v2 transfer-function coefficients** ``P_e``, ``P_i`` from
   ``BASE_PARAMETER_MODEL_NEW`` — the updated Berlin v2 fit used in all
   ``brain_act_*`` notebooks.

4. **v2 resting potentials** ``E_L_e = -63 mV``, ``E_L_i = -65 mV``.

5. **v2 initial conditions** for the MF: ``E = 4 Hz``, ``I = 10 Hz`` (i.e.
   the AI-state basin, ignites immediately rather than starting at 2 Hz).
   ``W`` is auto-initialised as ``b_e · v_e · τ_w`` by ``run_mf_ode``.

6. **Same MF timescale** ``T = 20 ms`` as the whole-brain runs.

Outputs
-------
* ``fig01_new_snn_vs_mf_b_sweep_rows_T<T>ms.{png,pdf}`` — 2×N rows figure
* ``summary_b_sweep_rates_T<T>ms.csv``
* ``run_manifest_T<T>ms.json``

Differences from original ``01_mean_field_sim_test.py``
-------------------------------------------------------
* MF ``init_state`` explicitly set to (4 Hz, 10 Hz) — was the default (2, 2).
* P_e / P_i, E_L_e, tau_e_*, T_ms passed in explicit MFParams construction
  rather than relying on dict packing (cleaner attribution).
* Docstring documents the v2 conventions in one place.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain_act_hybrid_common import BASE_PARAMETER_MODEL_NEW, PROJECT_ROOT, save_json

from tvbtoolkit.core.paths import legacy_results
from tvbtoolkit.brian_mf.adex.network import run_adex_network_simulation
from tvbtoolkit.workflows.mean_field_sweep import MFParams, run_mf_ode


# ---------------------------------------------------------------------------
# v2 initial-condition convention (Hz). MF defaults are (2,2), we want AI basin.
# ---------------------------------------------------------------------------

MF_INIT_E_HZ: float = 4.0
MF_INIT_I_HZ: float = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "v2 SNN ↔ single-region MF parity sweep across b_e with shared OU drive "
            "(lab v2 params: BASE_PARAMETER_MODEL_NEW)."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=legacy_results("notebooks_outputs", "01_mean_field_sim_test_new"),
    )
    p.add_argument("--b-start", type=float, default=5.0)
    p.add_argument("--b-stop", type=float, default=205.0)
    p.add_argument("--b-step", type=float, default=10.0)
    p.add_argument("--duration-ms", type=float, default=12000.0)
    p.add_argument("--transient-ms", type=float, default=2000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iext-hz", type=float, default=0.315)
    p.add_argument("--bin-width-ms", type=float, default=5.0)
    p.add_argument("--mf-dt-ms", type=float, default=0.1)
    p.add_argument("--mf-T-ms", dest="mf_t_ms", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=max(1, int(round((os.cpu_count() or 8) * 0.8))))
    p.add_argument(
        "--mf-paper-sigma",
        type=float,
        default=3.5,
        help="Paper Eq. (20) σ for v_aff(t) = v_drive + σ·ξ(t).",
    )
    p.add_argument(
        "--mf-tau-ou-ms",
        type=float,
        default=5.0,
        help="OU time constant (ms), paper uses 5 ms.",
    )
    p.add_argument("--n-tot", type=int, default=10000)
    return p.parse_args()


def paper_sigma_to_run_mf_sigma_hz(paper_sigma: float, tau_ou_ms: float, dt_ms: float) -> float:
    """Map paper Eq. (20) σ to the discrete OU-update σ used in build_shared_afferent_trace_hz."""
    tau_s = float(tau_ou_ms) / 1000.0
    target_std_hz = float(paper_sigma) * np.sqrt(max(tau_s, 1e-12) / 2.0)
    dt_over_tau = float(dt_ms) / max(float(tau_ou_ms), 1e-12)
    return float(target_std_hz * np.sqrt(max(2.0 - dt_over_tau, 1e-9)))


def build_shared_afferent_trace_hz(
    *,
    duration_ms: float,
    dt_ms: float,
    seed: int,
    v_drive_hz: float,
    sigma_ou_hz: float,
    tau_ou_ms: float,
) -> np.ndarray:
    """Generate one OU-based afferent drive trace (Hz), clipped at 0."""
    n = int(float(duration_ms) / float(dt_ms))
    rng = np.random.default_rng(int(seed))
    xi = 0.0
    out = np.empty(n, dtype=float)
    tau = max(float(tau_ou_ms), 1e-12)
    scale = float(sigma_ou_hz) * np.sqrt(float(dt_ms) / tau)
    decay = 1.0 - float(dt_ms) / tau
    base = float(v_drive_hz)
    for i in range(n):
        xi = xi * decay + scale * rng.standard_normal()
        out[i] = max(base + xi, 0.0)
    return out


def _build_v2_mfparams(b_val: float, *, n_tot: int, iext_hz: float, mf_t_ms: float) -> MFParams:
    """Construct an MFParams object using lab v2 conventions (BASE_PARAMETER_MODEL_NEW)."""
    return MFParams(
        Q_e=1.5, Q_i=5.0,
        tau_e=float(BASE_PARAMETER_MODEL_NEW["tau_e_e"]),
        tau_i=float(BASE_PARAMETER_MODEL_NEW["tau_e_i"]),
        E_e=0.0, E_i=-80.0,
        g_L=10.0, C_m=200.0,
        E_L_e=float(BASE_PARAMETER_MODEL_NEW["E_L_e"]),
        E_L_i=float(BASE_PARAMETER_MODEL_NEW["E_L_i"]),
        b_e=float(b_val), a_e=0.0, tau_w_e=500.0,
        N_tot=int(n_tot),
        p_connect_e=0.05, p_connect_i=0.05,
        g=0.2, K_ext_e=400, K_ext_i=0,
        T_ms=float(mf_t_ms),
        P_e=tuple(float(x) for x in BASE_PARAMETER_MODEL_NEW["P_e"]),
        P_i=tuple(float(x) for x in BASE_PARAMETER_MODEL_NEW["P_i"]),
        v_drive_hz=float(iext_hz),
        sigma_ou_hz=0.0,                          # internal OU disabled — external trace used
        tau_ou_ms=5.0,
    )


def run_single_b_job(
    b_val: float,
    *,
    seed: int,
    duration_ms: float,
    transient_ms: float,
    iext_hz: float,
    bin_width_ms: float,
    mf_dt_ms: float,
    mf_t_ms: float,
    mf_tau_ou_ms: float,
    mf_sigma_hz_effective: float,
    n_tot: int,
) -> dict[str, np.ndarray | float]:
    shared_aff_dt_ms = 0.1
    shared_aff_hz = build_shared_afferent_trace_hz(
        duration_ms=float(duration_ms),
        dt_ms=float(shared_aff_dt_ms),
        seed=int(seed),
        v_drive_hz=float(iext_hz),
        sigma_ou_hz=float(mf_sigma_hz_effective),
        tau_ou_ms=float(mf_tau_ou_ms),
    )

    # ---------------- SNN (Brian2) ----------------
    snn_overrides = {
        "b_e": float(b_val),
        "EL_e": float(BASE_PARAMETER_MODEL_NEW["E_L_e"]),
        "EL_i": float(BASE_PARAMETER_MODEL_NEW["E_L_i"]),
        "tau_e": float(BASE_PARAMETER_MODEL_NEW["tau_e_e"]),
        "tau_i": float(BASE_PARAMETER_MODEL_NEW["tau_e_i"]),
        "Ntot": int(n_tot),
    }
    snn = run_adex_network_simulation(
        cells="FS-RS_10",
        seed_value=int(seed),
        time_ms=float(duration_ms),
        iext_hz=float(iext_hz),
        input_hz=0.0,
        external_rate_hz_trace=shared_aff_hz,
        external_rate_dt_ms=float(shared_aff_dt_ms),
        dt_ms=0.1,
        bin_width_ms=float(bin_width_ms),
        parameter_overrides=snn_overrides,
        split_leak=False,
    )
    keep_snn = snn.time_ms >= float(transient_ms)
    t_snn = snn.time_ms[keep_snn] - float(transient_ms)
    e_snn = snn.rate_exc_hz[keep_snn]
    i_snn = snn.rate_inh_hz[keep_snn]

    # ---------------- MF (first-order Zerlaut via mean_field_sweep) ----------------
    mf_params = _build_v2_mfparams(b_val=b_val, n_tot=n_tot, iext_hz=iext_hz, mf_t_ms=mf_t_ms)
    mf = run_mf_ode(
        mf_params,
        duration_ms=float(duration_ms),
        dt_ms=float(mf_dt_ms),
        seed=int(seed),
        stim_amplitude_hz=0.0,
        transient_ms=float(transient_ms),
        sigma_ou_hz=0.0,
        tau_ou_ms=float(mf_tau_ou_ms),
        external_drive_hz=shared_aff_hz,
        external_drive_dt_ms=float(shared_aff_dt_ms),
        init_state=(MF_INIT_E_HZ, MF_INIT_I_HZ),    # v2 AI-basin start
    )

    return {
        "b": float(b_val),
        "t_snn": np.asarray(t_snn, dtype=float),
        "e_snn": np.asarray(e_snn, dtype=float),
        "i_snn": np.asarray(i_snn, dtype=float),
        "t_mf": np.asarray(mf["time_ms"], dtype=float),
        "e_mf": np.asarray(mf["ve_hz"], dtype=float),
        "i_mf": np.asarray(mf["vi_hz"], dtype=float),
        "W_mf": np.asarray(mf["W_pa"], dtype=float),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    b_values = np.arange(args.b_start, args.b_stop + 1e-9, args.b_step, dtype=float)
    rows: list[dict[str, float]] = []

    mf_sigma_hz_effective = paper_sigma_to_run_mf_sigma_hz(
        paper_sigma=float(args.mf_paper_sigma),
        tau_ou_ms=float(args.mf_tau_ou_ms),
        dt_ms=float(args.mf_dt_ms),
    )

    t_val = float(args.mf_t_ms)
    if abs(t_val - round(t_val)) < 1e-12:
        t_core = str(int(round(t_val)))
    else:
        t_core = str(t_val).replace(".", "p")
    t_label = f"T{t_core}ms"

    job_results_by_b: dict[float, dict[str, np.ndarray | float]] = {}
    worker_count = int(args.workers)
    if worker_count <= 1:
        print(f"[01-new] running {len(b_values)} b-jobs sequentially")
        n = len(b_values)
        for i, b_val in enumerate(b_values, start=1):
            job_results_by_b[float(b_val)] = run_single_b_job(
                float(b_val),
                seed=int(args.seed),
                duration_ms=float(args.duration_ms),
                transient_ms=float(args.transient_ms),
                iext_hz=float(args.iext_hz),
                bin_width_ms=float(args.bin_width_ms),
                mf_dt_ms=float(args.mf_dt_ms),
                mf_t_ms=float(args.mf_t_ms),
                mf_tau_ou_ms=float(args.mf_tau_ou_ms),
                mf_sigma_hz_effective=float(mf_sigma_hz_effective),
                n_tot=int(args.n_tot),
            )
            if i == 1 or i % max(1, n // 20) == 0 or i == n:
                print(f"[01-new] progress {i}/{n}")
    else:
        print(f"[01-new] dispatching {len(b_values)} b-jobs on {worker_count} workers")
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futs = {
                ex.submit(
                    run_single_b_job,
                    float(b_val),
                    seed=int(args.seed),
                    duration_ms=float(args.duration_ms),
                    transient_ms=float(args.transient_ms),
                    iext_hz=float(args.iext_hz),
                    bin_width_ms=float(args.bin_width_ms),
                    mf_dt_ms=float(args.mf_dt_ms),
                    mf_t_ms=float(args.mf_t_ms),
                    mf_tau_ou_ms=float(args.mf_tau_ou_ms),
                    mf_sigma_hz_effective=float(mf_sigma_hz_effective),
                    n_tot=int(args.n_tot),
                ): float(b_val)
                for b_val in b_values
            }
            n = len(futs)
            for i, fut in enumerate(as_completed(futs), start=1):
                b_val = futs[fut]
                job_results_by_b[b_val] = fut.result()
                if i == 1 or i % max(1, n // 20) == 0 or i == n:
                    print(f"[01-new] progress {i}/{n}")

    snn_series: list[dict[str, np.ndarray]] = []
    mf_series: list[dict[str, np.ndarray]] = []
    for b_val in b_values:
        r = job_results_by_b[float(b_val)]
        snn_series.append({
            "b": np.array([b_val]),
            "t": np.asarray(r["t_snn"], dtype=float),
            "e": np.asarray(r["e_snn"], dtype=float),
            "i": np.asarray(r["i_snn"], dtype=float),
        })
        mf_series.append({
            "b": np.array([b_val]),
            "t": np.asarray(r["t_mf"], dtype=float),
            "e": np.asarray(r["e_mf"], dtype=float),
            "i": np.asarray(r["i_mf"], dtype=float),
            "W": np.asarray(r["W_mf"], dtype=float),
        })
        rows.append({
            "b_e_pa": float(b_val),
            "snn_exc_mean_hz": float(np.nanmean(r["e_snn"])),
            "snn_inh_mean_hz": float(np.nanmean(r["i_snn"])),
            "mf_exc_mean_hz":  float(np.nanmean(r["e_mf"])),
            "mf_inh_mean_hz":  float(np.nanmean(r["i_mf"])),
            "mf_W_mean_pa":    float(np.nanmean(r["W_mf"])),
        })

    # ---------------- figure ----------------
    n_rows = len(b_values)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, max(2.2 * n_rows, 8.0)), sharex=False)
    if n_rows == 1:
        axes = np.array([axes])

    for idx, b_val in enumerate(b_values):
        ax_snn = axes[idx, 0]
        ax_mf  = axes[idx, 1]
        ss = snn_series[idx]
        ms = mf_series[idx]

        ax_snn.plot(ss["t"] / 1000.0, ss["e"], color="#1f77b4", lw=1.0, label="Exc")
        ax_snn.plot(ss["t"] / 1000.0, ss["i"], color="#d62728", lw=1.0, label="Inh")
        ax_snn.set_ylabel(f"b={b_val:.0f}\nRate (Hz)")
        ax_snn.grid(alpha=0.2)

        ax_mf.plot(ms["t"] / 1000.0, ms["e"], color="#1f77b4", lw=1.0, label="Exc")
        ax_mf.plot(ms["t"] / 1000.0, ms["i"], color="#d62728", lw=1.0, label="Inh")
        ax_mf.grid(alpha=0.2)

        if idx == 0:
            ax_snn.set_title("Spiking network (SNN) — Brian2, OU drive (paper Eq. 20)")
            ax_mf.set_title("Mean-field (MF) — 1st-order Zerlaut, shared OU drive")
            ax_snn.legend(frameon=False, fontsize=8, loc="upper right")
            ax_mf.legend(frameon=False, fontsize=8, loc="upper right")

        if idx == n_rows - 1:
            ax_snn.set_xlabel("Time post-transient (s)")
            ax_mf.set_xlabel("Time post-transient (s)")

    fig.tight_layout()
    fig_pdf = args.output_dir / f"fig01_new_snn_vs_mf_b_sweep_rows_{t_label}.pdf"
    fig_png = args.output_dir / f"fig01_new_snn_vs_mf_b_sweep_rows_{t_label}.png"
    fig.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output_dir / f"summary_b_sweep_rates_{t_label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json(
        args.output_dir / f"run_manifest_{t_label}.json",
        {
            "script": "01_mean_field_sim_test_new.py",
            "b_values": [float(x) for x in b_values],
            "duration_ms": float(args.duration_ms),
            "transient_ms": float(args.transient_ms),
            "seed": int(args.seed),
            "iext_hz": float(args.iext_hz),
            "mf_dt_ms": float(args.mf_dt_ms),
            "mf_T_ms": float(args.mf_t_ms),
            "mf_paper_sigma": float(args.mf_paper_sigma),
            "mf_tau_ou_ms": float(args.mf_tau_ou_ms),
            "mf_sigma_hz_effective": float(mf_sigma_hz_effective),
            "mf_init_state_hz": [MF_INIT_E_HZ, MF_INIT_I_HZ],
            "workers": int(args.workers),
            "n_tot": int(args.n_tot),
            "base_parameter_model_new": BASE_PARAMETER_MODEL_NEW,
            "v2_corrections": [
                "Shared OU afferent drive (paper Eq. 20) between SNN and MF",
                f"Explicit MF init_state=({MF_INIT_E_HZ}, {MF_INIT_I_HZ}) Hz (AI basin)",
                "MFParams constructed with explicit v2 P_e/P_i, E_L_e=-63, tau_e_*=5ms",
                "Internal OU disabled in MF (sigma_ou_hz=0); external trace is the sole noise source",
            ],
            "outputs": {
                "figure_pdf": str(fig_pdf),
                "figure_png": str(fig_png),
                "summary_csv": str(csv_path),
            },
        },
    )

    print(f"[01-new] done. outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
