"""Quick reproducibility check for integrated Maria Sacha notebook pipeline.

This script performs lightweight checks (no heavy full notebook execution):
- verifies notebook and mapped modules import
- verifies paper precomputed survival arrays can be loaded
- runs a tiny one-condition, one-seed whole-brain batch and computes FC-SC summary
- writes a JSON report under the external CNRS legacy-unsorted results tree
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
from tvb.simulator import lab

from tvbtoolkit.core.paths import legacy_results
from tvbtoolkit import (
    WholeBrainConfig,
    OutputConfig,
    load_survival_arrays,
    maria_sacha_nature_conditions,
    run_condition_batch,
)
from tvbtoolkit.whole_brain.analysis import fcsc_seedwise_from_saved_batch


PAPER_ROOT = Path('/Users/borjan/CNRS/projects/paper_pipeline_hub')
NOTEBOOK = _REPO_ROOT / "notebooks" / "repro_maria_sacha_nature.ipynb"
OUT_ROOT = legacy_results("notebooks_outputs", "repro_maria_sacha_nature", "repro_check")


def main() -> None:
    report: dict[str, object] = {}

    report['notebook_exists'] = NOTEBOOK.exists()

    # survival arrays
    mean_e, taus_e, bthr_e, tau_v_e, bvals_e = load_survival_arrays(load='tau_e', precalc=True, paper_repo_root=PAPER_ROOT)
    mean_i, taus_i, bthr_i, tau_v_i, bvals_i = load_survival_arrays(load='tau_i', precalc=True, paper_repo_root=PAPER_ROOT)
    report['survival_tau_e_shape'] = list(mean_e.shape)
    report['survival_tau_i_shape'] = list(mean_i.shape)
    report['survival_tau_e_threshold_points'] = int(len(taus_e))
    report['survival_tau_i_threshold_points'] = int(len(taus_i))

    # tiny whole-brain smoke run
    out = OutputConfig(root=OUT_ROOT)
    conn_76 = PAPER_ROOT / 'TVB' / 'tvb_model_reference' / 'data' / 'connectivity' / 'connectivity_76.zip'
    cfg = WholeBrainConfig(
        model_family='adex_zerlaut',
        zerlaut_order=2,  # second-order: matches TVBSim default
        simulation_length_ms=4000.0,
        dt_ms=0.1,
        connectivity_zip=str(conn_76),
        monitor_mode='temporal_average',
        temporal_average_period_ms=1.0,
    )
    cond = [maria_sacha_nature_conditions()[0]]  # wake only
    metrics = run_condition_batch(
        base_cfg=cfg,
        conditions=cond,
        seeds=[0],
        output=out,
        post_stim_window=200,
        save_timeseries=True,
        n_jobs=1,
        use_processes=False,
        show_progress=False,
    )
    report['tiny_batch_conditions'] = list(metrics.keys())
    report['tiny_batch_lzc_n'] = int(metrics['wake']['lzc'].size)

    conn = lab.connectivity.Connectivity().from_file(str(conn_76))
    sc = np.asarray(conn.weights, dtype=float)
    np.fill_diagonal(sc, 0.0)
    fcsc = fcsc_seedwise_from_saved_batch(
        out,
        conditions=['wake'],
        seeds=[0],
        structural_connectivity=sc,
        cut_transient_ms=1000.0,
        tr_ms=500.0,
    )
    report['tiny_fcsc_legacy'] = float(np.nanmean(fcsc['wake']['legacy_r_signed_full'])) if fcsc['wake']['legacy_r_signed_full'].size else None

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUT_ROOT / 'repro_check.json'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', out_json)


if __name__ == '__main__':
    main()
