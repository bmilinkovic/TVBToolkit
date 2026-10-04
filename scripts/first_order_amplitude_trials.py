"""Independent stochastic amplitude trials. Native first-order dynamics only.

Each seed has an independent 8-s baseline; amplitudes share its checkpoint and
future noise. Thus amplitudes are paired, trials are independent. No PCI here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from first_order_adex_audit import OUT, configure, clean
from first_order_whole_brain_sweep import SUBJECTS
from tvbtoolkit.whole_brain.impulse_response import fork_checkpoint

ROOT = OUT / 'amplitude_trials_v1'
PROTOCOL = dict(version=1, G=.05, b_pA=5., baseline_ms=8000., response_ms=3000.,
                dt_ms=.1, source_index=9, pulse_ms=10., high_hz=150.,
                sustained_ms=500., tail_ms=500., saved_dt_ms=1.)


def install_noise(sim, seed, state=None):
    rng = np.random.default_rng(seed)
    if state is not None:
        rng.bit_generator.state = deepcopy(state)
    def draw(shape, **unused):
        out = np.zeros(shape)
        out[-1, :, 0] = rng.normal(size=shape[1]) * np.sqrt(PROTOCOL['dt_ms'])
        return out
    sim.integrator.noise.generate = draw
    return rng


def high_event(rates_hz, dt_ms=.1):
    """Transient sustained event and late persistence are different outcomes."""
    high = np.any(rates_hz > PROTOCOL['high_hz'], axis=1)
    count = int(round(PROTOCOL['sustained_ms'] / dt_ms))
    cs = np.concatenate([np.zeros((1, high.shape[1]), dtype=np.int64),
                         np.cumsum(high, axis=0)], axis=0)
    windows = cs[count:] - cs[:-count]
    hits = windows == count
    region_hit = hits.any(axis=0)
    first = np.full(high.shape[1], np.nan)
    first[region_hit] = (hits[:, region_hit].argmax(axis=0)+1)*dt_ms
    tail_n = int(round(PROTOCOL['tail_ms']/dt_ms))
    persistent = high[-tail_n:].all(axis=0)
    return dict(sustained_high=bool(region_hit.any()), persistent_high=bool(persistent.any()),
                high_regions=np.flatnonzero(region_hit).tolist(),
                persistent_regions=np.flatnonzero(persistent).tolist(),
                first_high_ms=[None if np.isnan(x) else float(x) for x in first])


def simulate(job, protocol=None):
    # Explicit per-experiment overrides; historical/default runs remain unchanged.
    protocol = dict(PROTOCOL if protocol is None else protocol)
    cohort, subject, seed, amplitudes, phase = job
    folder = ROOT / phase / f'{cohort}_{subject}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'summary.json'
    signature = hashlib.sha256(json.dumps(dict(protocol=protocol, amplitudes=amplitudes),
                                         sort_keys=True).encode()).hexdigest()
    if target.exists():
        cached = json.loads(target.read_text())
        if cached['signature'] != signature:
            raise ValueError(f'Protocol mismatch: {target}; use a new phase directory')
        return cached
    started = time.monotonic()
    sim, cfg, labels = configure(isolated=False, cohort=cohort, subject=subject,
                                 coupling=protocol['G'], b=protocol['b_pA'], seed=seed)
    rng = install_noise(sim, seed)
    t, base = sim.run(simulation_length=protocol['baseline_ms'])[0]
    base = base[:, :, :, 0]
    if not np.isfinite(base).all():
        raise FloatingPointError('Nonfinite baseline; never silently exclude a trial')
    rng_state = deepcopy(rng.bit_generator.state)
    np.savez_compressed(folder/'baseline.npz', time_ms=t[9::10]-t[-1],
                        state=base[9::10].astype(np.float32), labels=labels,
                        checkpoint_state=sim.current_state, history=sim.history.buffer)
    control = None
    rows = []
    for amp in sorted(set([0.]+list(amplitudes))):
        overrides = deepcopy(cfg.parameter_overrides)
        if amp:
            overrides['parameter_stimulus'] = dict(stimtime=0., stimdur=protocol['pulse_ms'], stimperiod=10000.,
                stimval=amp/1000., stimregion=[9], stimvariables=[0], stimshape='square')
        fork = fork_checkpoint(sim, replace(cfg, parameter_overrides=overrides), seed)
        assert tuple(fork.model.state_variables) == ('E','I','W_e','W_i','noise')
        for key in ['state_variable_boundaries','_integration_state_variable_boundaries',
                    'clamped_state_variable_values','_clamped_integration_state_variable_values']:
            setattr(fork.integrator, key, None)
        install_noise(fork, seed, rng_state)
        tt, a = fork.run(simulation_length=protocol['response_ms'])[0]
        a = a[:, :, :, 0]
        if not np.isfinite(a).all():
            raise FloatingPointError(f'Nonfinite response {cohort} {seed} {amp}')
        if amp == 0:
            control = a.copy()
        delta = 1000*(a[:, 0]-control[:, 0])
        remote = np.arange(90) != 9
        peaks = np.abs(delta).max(axis=0)
        row = dict(amplitude_hz_per_ms=amp, **high_event(a[:, :2]*1000),
                   max_EI_hz=float(a[:, :2].max()*1000),
                   source_peak_hz=float(peaks[9]), remote_peak_hz=float(peaks[remote].max()),
                   tail_response_rms_hz=float(np.sqrt(np.mean(delta[-5000:]**2))),
                   per_region_peak_hz=peaks.tolist())
        np.savez_compressed(folder/f'amplitude_{amp:g}.npz', time_ms=tt[9::10]-t[-1],
                            state=a[9::10].astype(np.float32),
                            delta_E_hz=delta[9::10].astype(np.float32))
        rows.append(row)
        (folder/f'amplitude_{amp:g}.json').write_text(json.dumps(clean(row), indent=2))
    result = clean(dict(cohort=cohort, subject=subject, seed=seed, phase=phase,
        protocol=protocol, signature=signature, source_region=str(labels[9]),
        baseline_mean_EI_hz=base[-20000:, :2].mean(axis=(0,2))*1000,
        baseline_max_last2s_hz=base[-20000:, :2].max()*1000,
        baseline_high=high_event(base[-20000:, :2]*1000), pulses=rows,
        elapsed_seconds=time.monotonic()-started))
    target.write_text(json.dumps(result, indent=2))
    print(cohort, seed, f'{result["elapsed_seconds"]:.1f}s',
          [(p['amplitude_hz_per_ms'],p['persistent_high']) for p in rows], flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', default='screen')
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--cohorts', nargs='+', default=[c for c,_ in SUBJECTS])
    p.add_argument('--amplitudes', nargs='+', type=float, default=[.1,.3,.45,.6,.75,.9,1.05,1.2])
    p.add_argument('--seed', type=int, default=903010)
    args = p.parse_args()
    if args.trials < 1 or min(args.amplitudes)<0:
        p.error('Positive trial count and nonnegative amplitudes required')
    ROOT.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(args.seed).generate_state(args.trials).tolist()
    jobs = [(c,s,int(seed),args.amplitudes,args.phase) for c,s in SUBJECTS
            if c in args.cohorts for seed in seeds]
    plan = dict(arguments=vars(args), protocol=PROTOCOL, seeds=seeds, jobs=len(jobs))
    manifest = ROOT/f'{args.phase}_plan.json'
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError('Existing phase has a different plan; select a new phase name')
    manifest.write_text(json.dumps(plan, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(simulate, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
    print('Completed', args.phase, len(jobs), 'independent subject/seed trials', flush=True)


if __name__ == '__main__':
    main()
