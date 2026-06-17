#!/usr/bin/env python3
"""
Parity simulation: first-order vs second-order mean-field
in TVBSim and TVBToolkit.

Runs a short (2000 ms) deterministic whole-brain simulation with both orders
and compares the excitatory firing-rate time-series.

Usage:
    python3 parity_simulation.py
"""

import sys
import time
import numpy as np

# ─── Path setup ──────────────────────────────────────────────────────────────
BRAIN_ACT_ROOT = "/sessions/cool-compassionate-bardeen/mnt/brain-act"
TVBTOOLKIT_SRC = f"{BRAIN_ACT_ROOT}/../projects/TVBToolkit/src"
TVBSIM_SRC     = f"{BRAIN_ACT_ROOT}/external/TVBSim"

for p in [TVBTOOLKIT_SRC, TVBSIM_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ─── Shared simulation parameters ────────────────────────────────────────────
SIM_LENGTH_MS   = 2000.0   # short run for speed
DT_MS           = 0.1
COUPLING        = 0.3
SPEED           = 4.0
STOCHASTIC      = False    # deterministic for perfect reproducibility
SEED            = 42

# Correct Berlin/Fede P coefficients (matching TVBSim parameter_M_Berlin_new.py)
# threshold_func: P0 + P1*V + P2*S + P3*T + P4*V² + P5*S² + P6*T² + P7*VS + P8*VT + P9*ST
P_e = np.array([-0.05017034,  0.00451531, -0.00794377, -0.00208418, -0.00054697,
                  0.00341614, -0.01156433,  0.00194753,  0.00274079, -0.01066769])
P_i = np.array([-0.05184978,  0.0061593,  -0.01403522,  0.00166511, -0.0020559,
                  0.00318432, -0.03112775,  0.00656668,  0.00171829, -0.04516385])

CONNECTIVITY_ZIP = (
    f"{TVBSIM_SRC}/tvbsim/TVB/tvb_model_reference/data/connectivity/connectivity_68.zip"
)

RESULTS = {}


# ═════════════════════════════════════════════════════════════════════════════
# TVBTOOLKIT runs
# ═════════════════════════════════════════════════════════════════════════════

def run_tvbtoolkit(order: int) -> dict:
    """Run TVBToolkit whole-brain simulation with the given Zerlaut order."""
    from tvbtoolkit.core.config import WholeBrainConfig
    from tvbtoolkit.whole_brain.simulation import run_whole_brain_simulation

    cfg = WholeBrainConfig(
        model_family="adex_zerlaut",
        zerlaut_matteo=False,
        zerlaut_gk_gna=False,
        zerlaut_order=order,
        stochastic_integrator=STOCHASTIC,
        simulation_length_ms=SIM_LENGTH_MS,
        dt_ms=DT_MS,
        conduction_speed=SPEED,
        coupling_strength=COUPLING,
        connectivity_zip=CONNECTIVITY_ZIP,
        monitor_mode="raw",
        parameter_overrides={
            "parameter_model": {
                "E_L_e": -64.0,
                "E_L_i": -65.0,
                "P_e": P_e.tolist(),
                "P_i": P_i.tolist(),
            }
        },
    )
    t0 = time.perf_counter()
    result = run_whole_brain_simulation(cfg, seed=SEED)
    elapsed = time.perf_counter() - t0
    return {
        "time_ms": result.time_ms,
        "E": result.raw,          # shape (T, n_regions)
        "I": result.raw_inh,
        "elapsed_s": elapsed,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TVBSIM runs
# ═════════════════════════════════════════════════════════════════════════════

def run_tvbsim(order: int) -> dict:
    """Run TVBSim whole-brain simulation with the given Zerlaut order."""
    import importlib
    import copy

    # TVBSim uses its own parameter dict
    tvbsim_param_mod = importlib.import_module(
        "tvbsim.TVB.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin_new"
    )
    Parameters = tvbsim_param_mod.Parameters
    from tvbsim.simconfig import sim_init

    params = Parameters()
    pm = params.parameter_model
    pm["order"] = order
    pm["matteo"] = False
    pm["gK_gNa"] = False
    pm["E_L_e"] = -64.0
    pm["E_L_i"] = -65.0
    pm["P_e"] = P_e.tolist()
    pm["P_i"] = P_i.tolist()

    params.parameter_integrator["stochastic"] = STOCHASTIC
    params.parameter_integrator["dt"] = DT_MS
    params.parameter_coupling["coupling_parameter"]["a"] = COUPLING
    params.parameter_connection_between_region["speed"] = SPEED

    # Use the same connectivity file
    params.parameter_connection_between_region["from_file"] = False
    params.parameter_connection_between_region["default"] = False

    # Point to the connectivity zip via from_file path
    import os
    conn_dir  = os.path.dirname(CONNECTIVITY_ZIP)
    conn_name = os.path.basename(CONNECTIVITY_ZIP)
    params.parameter_connection_between_region["from_file"] = True
    params.parameter_connection_between_region["path"] = conn_dir
    params.parameter_connection_between_region["conn_name"] = conn_name

    # Use Raw monitor (VOI=0 excitatory only)
    params.parameter_monitor["Raw"] = True
    params.parameter_monitor["parameter_Raw"]["variables_of_interest"] = [0, 1]
    params.parameter_monitor["TemporalAverage"] = False
    params.parameter_monitor["Bold"] = False

    np.random.seed(SEED)
    t0 = time.perf_counter()
    sim = sim_init(params)
    output = sim.run(simulation_length=SIM_LENGTH_MS)
    elapsed = time.perf_counter() - t0

    t, data = output[0]
    data = np.asarray(data)   # shape (T, nvar, n_regions, 1)
    E = data[:, 0, :, 0]
    I = data[:, 1, :, 0] if data.shape[1] > 1 else None
    return {
        "time_ms": np.asarray(t).ravel(),
        "E": E,
        "I": I,
        "elapsed_s": elapsed,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main comparison
# ═════════════════════════════════════════════════════════════════════════════

def compare_timeseries(label_a, a, label_b, b):
    """Print statistics comparing two E-rate arrays (T x n_regions)."""
    # Trim to matching lengths
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    mean_a = np.mean(np.abs(a))
    rel_err = mae / (mean_a + 1e-12) * 100
    print(f"  {label_a} vs {label_b}:")
    print(f"    MAE        = {mae:.6e}  ({rel_err:.2f}% of mean |E|)")
    print(f"    RMSE       = {rmse:.6e}")
    print(f"    Max|err|   = {max_abs:.6e}")
    print(f"    Pearson r  = {corr:.6f}")
    print(f"    Mean E[{label_a}]= {np.mean(a)*1e3:.4f} Hz   Mean E[{label_b}]= {np.mean(b)*1e3:.4f} Hz")


def main():
    print("=" * 72)
    print("PARITY SIMULATION: First-order vs Second-order Mean-Field")
    print("TVBSim  ↔  TVBToolkit")
    print("=" * 72)
    print(f"  Sim length : {SIM_LENGTH_MS} ms")
    print(f"  dt         : {DT_MS} ms")
    print(f"  Coupling a : {COUPLING}")
    print(f"  Stochastic : {STOCHASTIC}")
    print()

    # ── TVBToolkit ────────────────────────────────────────────────────────────
    print("Running TVBToolkit order=1 ... ", end="", flush=True)
    try:
        tk_1 = run_tvbtoolkit(order=1)
        print(f"done in {tk_1['elapsed_s']:.1f}s  |  E shape: {tk_1['E'].shape}")
        RESULTS["TVBToolkit_order1"] = tk_1
    except Exception as e:
        print(f"FAILED: {e}")
        RESULTS["TVBToolkit_order1"] = None

    print("Running TVBToolkit order=2 ... ", end="", flush=True)
    try:
        tk_2 = run_tvbtoolkit(order=2)
        print(f"done in {tk_2['elapsed_s']:.1f}s  |  E shape: {tk_2['E'].shape}")
        RESULTS["TVBToolkit_order2"] = tk_2
    except Exception as e:
        print(f"FAILED: {e}")
        RESULTS["TVBToolkit_order2"] = None

    # ── TVBSim ────────────────────────────────────────────────────────────────
    print("Running TVBSim     order=1 ... ", end="", flush=True)
    try:
        sim_1 = run_tvbsim(order=1)
        print(f"done in {sim_1['elapsed_s']:.1f}s  |  E shape: {sim_1['E'].shape}")
        RESULTS["TVBSim_order1"] = sim_1
    except Exception as e:
        print(f"FAILED: {e}")
        RESULTS["TVBSim_order1"] = None

    print("Running TVBSim     order=2 ... ", end="", flush=True)
    try:
        sim_2 = run_tvbsim(order=2)
        print(f"done in {sim_2['elapsed_s']:.1f}s  |  E shape: {sim_2['E'].shape}")
        RESULTS["TVBSim_order2"] = sim_2
    except Exception as e:
        print(f"FAILED: {e}")
        RESULTS["TVBSim_order2"] = None

    # ── Timing summary ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("TIMING SUMMARY")
    print("-" * 72)
    for key, res in RESULTS.items():
        if res is not None:
            speedup_vs_order1 = ""
            other_key = key.replace("order1", "order2").replace("order2", "order1")
            if other_key in RESULTS and RESULTS[other_key] is not None:
                t1 = RESULTS[other_key]["elapsed_s"]
                t2 = res["elapsed_s"]
                speedup_vs_order1 = f"  (×{t2/t1:.2f} vs order=1)" if "order2" in key else ""
            print(f"  {key:<30}: {res['elapsed_s']:6.2f}s{speedup_vs_order1}")

    # ── Cross-toolkit parity ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("CROSS-TOOLKIT PARITY (TVBToolkit vs TVBSim, same order)")
    print("-" * 72)
    for order in [1, 2]:
        tk  = RESULTS.get(f"TVBToolkit_order{order}")
        sim = RESULTS.get(f"TVBSim_order{order}")
        if tk is not None and sim is not None:
            print(f"\n  Order {order}:")
            compare_timeseries(f"TVBToolkit_o{order}", tk["E"],
                               f"TVBSim_o{order}",     sim["E"])

    # ── Order comparison within each toolkit ──────────────────────────────────
    print()
    print("=" * 72)
    print("ORDER COMPARISON (first-order vs second-order, same toolkit)")
    print("-" * 72)
    for toolkit, key1, key2 in [
        ("TVBToolkit", "TVBToolkit_order1", "TVBToolkit_order2"),
        ("TVBSim",     "TVBSim_order1",     "TVBSim_order2"),
    ]:
        r1 = RESULTS.get(key1)
        r2 = RESULTS.get(key2)
        if r1 is not None and r2 is not None:
            print(f"\n  {toolkit}:")
            compare_timeseries(f"{toolkit}_o1", r1["E"], f"{toolkit}_o2", r2["E"])

    # ── Numerical summary table ───────────────────────────────────────────────
    print()
    print("=" * 72)
    print("FIRING RATE STATISTICS  (mean excitatory rate in Hz, across all regions & time)")
    print("-" * 72)
    print(f"  {'Run':<30} {'mean E (Hz)':>12}  {'std E (Hz)':>12}  {'min E (Hz)':>12}  {'max E (Hz)':>12}")
    print(f"  {'-'*30} {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
    for key, res in RESULTS.items():
        if res is not None:
            E_hz = res["E"] * 1e3   # convert kHz → Hz
            print(f"  {key:<30} {np.mean(E_hz):>12.4f}  {np.std(E_hz):>12.4f}  {np.min(E_hz):>12.4f}  {np.max(E_hz):>12.4f}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("-" * 72)
    tk2  = RESULTS.get("TVBToolkit_order2")
    sim2 = RESULTS.get("TVBSim_order2")
    if tk2 is not None and sim2 is not None:
        n = min(len(tk2["E"]), len(sim2["E"]))
        mae = np.mean(np.abs(tk2["E"][:n] - sim2["E"][:n]))
        rel = mae / (np.mean(np.abs(sim2["E"][:n])) + 1e-12) * 100
        if rel < 1.0:
            verdict = "✓ PASS  —  TVBToolkit order=2 matches TVBSim order=2 within 1% MAE"
        elif rel < 5.0:
            verdict = "⚠ MARGINAL  —  relative error < 5%"
        else:
            verdict = "✗ FAIL  —  relative error ≥ 5%"
        print(f"  {verdict}  (rel. MAE = {rel:.3f}%)")
    else:
        print("  Could not compare order=2 runs (one or both failed).")
    print()


if __name__ == "__main__":
    main()
