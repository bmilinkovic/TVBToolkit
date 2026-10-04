"""Subject-specific first-order b–G inference; simulations and training are separate.

Run ``python experiments/first_order_bg_sbi/workflow.py --help``. Read the
accompanying README before preparing data or submitting a large simulation bank.
No TMS, PCI, second-order dynamics or transfer-function changes occur here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
# Each worker runs one simulation; avoid multiplying BLAS threads by workers.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

from tvbtoolkit.inference.parameters import AdExParameterSpec, AdExPrior


def write_json(path, value):
    """Commit a JSON result atomically, never writing NaN/Infinity silently."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    """Fingerprint input/code contents, independent of their filesystem location."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_for(manifest):
    """Only b [pA] and dimensionless G are inferred; no group-dependent priors."""
    return AdExPrior((
        AdExParameterSpec('b', *manifest['b_bounds'], 'pA', 'parameter_model', model_keys=('b_e',)),
        AdExParameterSpec('G', *manifest['g_bounds'], 'dimensionless', 'config', target='coupling_strength'),
    ))


def draw(manifest, index):
    """Index-addressable draws: extending a bank does not change earlier draws.

    The random stream used for parameter selection is separate from the model's
    stochastic drive. Independent seeds represent baseline variability, not an
    additional fitted noise parameter.
    """
    rng = np.random.default_rng(np.random.SeedSequence([manifest['seed'], index, 712]))
    prior = prior_for(manifest)
    return rng.uniform(prior.low, prior.high), int(rng.integers(0, 2**31 - 1))


def align_bold(bold, labels, target):
    """Reorder labelled time×AAL90 BOLD explicitly; never guess a hemisphere order."""
    labels, target = list(map(str, labels)), list(map(str, target))
    x = np.asarray(bold, dtype=float)
    if len(set(labels)) != 90 or len(set(target)) != 90 or set(labels) != set(target):
        raise ValueError('Supply exactly the same 90 unique AAL labels as the SC atlas')
    if x.ndim != 2 or x.shape[1] != 90 or not np.isfinite(x).all():
        raise ValueError('BOLD must be finite time×90; do not fill missing ROIs with zeros')
    return x[:, [labels.index(label) for label in target]]


def features(bold, manifest):
    """FC/FCD distributions, labelled FC row means and BOLD autocorrelation.

    Reuse the native VBI feature extractor and its BOLD filtering identically on
    observations/simulations. Regional FC means retain some spatial information,
    but are NOT the full FC matrix. ACF values describe slow BOLD persistence,
    not the neuronal adaptation time constant directly.
    """
    from tvbtoolkit.inference.features import BOLDFeatureConfig, BOLDFeatureExtractor
    cfg = BOLDFeatureConfig(tr_seconds=manifest['tr_seconds'], include_states=False,
                            include_fcd=not manifest.get('explicit_short_fcd', False),
                            fcd_window_seconds=60., bandpass_hz=(.01, .10))
    if manifest.get('explicit_short_fcd', False):
        from vbi.feature_extraction.features import fc_stat
        filtered = preprocess_short_bold(bold, manifest['tr_seconds'])
        base, base_names = fc_stat(filtered.T, k=1, eigenvalues=False,
            pca_num_components=0, quantiles=[.05, .25, .5, .75, .95],
            features=['mean', 'std', 'skew', 'kurtosis'], verbose=False)
        base_names = [f'vbi_{name}' for name in base_names]
    else:
        extractor = BOLDFeatureExtractor(cfg)
        base = extractor.fit_transform(bold)
        base_names = extractor.feature_names_
        filtered = extractor._preprocess_connectivity(np.asarray(bold))
    if filtered.shape != np.asarray(bold).shape:
        raise ValueError('Preprocessing changed the time×region axis convention')
    fc = np.corrcoef(filtered.T)
    spatial = (fc.sum(axis=1) - 1) / (fc.shape[0] - 1)
    values = list(base) + list(spatial)
    names = list(base_names) + [f'FC_mean_{x}' for x in manifest['labels']]
    if manifest.get('explicit_short_fcd', False):
        fcd_values, fcd_names = short_fcd_features(filtered, manifest['tr_seconds'])
        values.extend(fcd_values); names.extend(fcd_names)
    for lag in (1, 2, 4):
        acf = np.array([np.corrcoef(filtered[:-lag, i], filtered[lag:, i])[0, 1]
                        for i in range(filtered.shape[1])])
        values.extend([acf.mean(), acf.std()])
        names.extend([f'ACF_lag{lag}_mean', f'ACF_lag{lag}_std'])
    if manifest.get('extended_features', False):
        from scipy.signal import welch
        freq, power = welch(filtered, fs=1 / manifest['tr_seconds'], axis=0,
                            nperseg=min(128, len(filtered)))
        band = (freq >= .01) & (freq <= .10)
        slow = (freq >= .01) & (freq < .04)
        fraction = power[slow].sum(axis=0) / power[band].sum(axis=0)
        peaks = freq[band][np.argmax(power[band], axis=0)]
        values.extend([fraction.mean(), fraction.std(), peaks.mean(), peaks.std()])
        names.extend(['BOLD_slow_power_fraction_mean', 'BOLD_slow_power_fraction_std',
                      'BOLD_peak_hz_mean', 'BOLD_peak_hz_std'])
    if 'state_template' in manifest:
        extra, extra_names = state_features(bold, manifest)
        values.extend(extra); names.extend(extra_names)
    result = np.asarray(values, dtype=np.float32)
    if len(result) != len(names):
        raise ValueError('Feature values and names differ in length')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite features: inspect constant signals, filter or FCD')
    return result, tuple(names)


def preprocess_short_bold(bold, tr_seconds):
    """Filter over TIME explicitly even when 50 volumes < 90 regions.

    Matches the existing z-score + Butterworth/filtfilt calculation without its
    downstream shape heuristic, which would transpose a short time×90 recording.
    This local adapter leaves the production BOLD utilities untouched.
    """
    from scipy.signal import iirfilter, filtfilt
    from scipy.stats import zscore
    x = np.asarray(bold, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(x.std(axis=0) <= 1e-12):
        raise ValueError('Expected finite nonconstant time×region BOLD')
    b, a = iirfilter(2, [.01 / (.5/tr_seconds), .10 / (.5/tr_seconds)],
                     btype='bandpass', ftype='butter', output='ba')
    return filtfilt(b, a, zscore(x, axis=0), axis=0)


def short_fcd_features(filtered, tr_seconds):
    """60-s Pearson-FC windows, one-TR stride, all off-diagonal FCD entries.

    Explicit seconds-to-samples conversion avoids VBI 0.4.3 get_fcd's length
    guard comparing sample count with seconds. For 50 samples and a 25-sample
    window there are 26 windows and 325 distinct window-pair similarities.
    These overlapping pairs are NOT independent observations. We exclude only
    the diagonal: excluding the first 25 diagonals would leave one pair and
    undefined skew/kurtosis on this short recording.
    """
    from scipy.stats import skew, kurtosis
    x = np.asarray(filtered)
    width = int(round(60. / tr_seconds))
    if width < 3 or len(x) < 2 * width:
        raise ValueError('This FCD protocol needs at least two 60-s windows')
    upper = np.triu_indices(x.shape[1], 1)
    edges = np.array([np.corrcoef(x[start:start + width].T)[upper]
                      for start in range(len(x) - width + 1)])
    matrix = np.corrcoef(edges)
    pairs = matrix[np.triu_indices(len(matrix), 1)]
    values = [pairs.mean(), pairs.std(), skew(pairs), kurtosis(pairs)]
    values += np.quantile(pairs, [.05, .25, .5, .75, .95]).tolist()
    names = ['FCD_mean', 'FCD_std', 'FCD_skew', 'FCD_kurtosis',
             'FCD_q05', 'FCD_q25', 'FCD_q50', 'FCD_q75', 'FCD_q95']
    return values, names


def state_patterns(bold, manifest):
    """Phase-coherence edges using fixed preprocessing and nine-volume edge trim."""
    from tvbtoolkit.analysis.brain_states import phase_patterns
    return phase_patterns(bold, pipeline='brain_act_legacy', trim_edge_samples=9,
                          tr_seconds=manifest['tr_seconds'], bandpass_hz=(.01, .10),
                          filter_order=2)[0]


def make_state_template(bold, weights, manifest):
    """Fit five empirical states ONCE, ordered by their SC correlation.

    Sorting is performed only here, never independently for each simulation.
    Missing simulated states subsequently get zero occupancy, not relabelled.
    """
    from sklearn.cluster import KMeans
    patterns = state_patterns(bold, manifest)
    centers = KMeans(n_clusters=5, n_init=20, random_state=11).fit(patterns).cluster_centers_
    sc = np.asarray(weights)[np.triu_indices(90, 1)]
    correlations = np.array([np.corrcoef(c, sc)[0, 1] for c in centers])
    if not np.isfinite(correlations).all():
        raise ValueError('Undefined empirical state–SC coupling')
    order = np.argsort(correlations)
    return dict(centers=centers[order].tolist(), sc_edges=sc.tolist(),
                empirical_sc_correlation=correlations[order].tolist(),
                order='ascending empirical centroid–SC Pearson correlation',
                representation='upper-triangle instantaneous phase coherence')


def state_features(bold, manifest):
    """Occupancy versus state-specific SC–phase-FC coupling on frozen templates.

    Coupling is recalculated from each recording's conditional mean pattern;
    appending the fixed template correlations alone would be uninformative.
    An absent state has coupling=0 by convention and occupancy=0 identifies it.
    These are phase-FC states, not sliding-window Pearson-FC states.
    """
    from scipy.spatial.distance import cdist
    template = manifest['state_template']
    centers = np.asarray(template['centers'])
    patterns = state_patterns(bold, manifest)
    assignment = cdist(patterns, centers).argmin(axis=1)
    values, names = [], []
    for k in range(len(centers)):
        active = assignment == k
        occupancy = float(active.mean())
        correlation = float(np.corrcoef(patterns[active].mean(axis=0), template['sc_edges'])[0, 1]) if active.any() else 0.
        values.extend([occupancy, correlation])
        names.extend([f'state_{k}_occupancy', f'state_{k}_SC_phaseFC'])
    return values, names


def configure_bold_kernel(sim, manifest):
    """Use TVB's native double-gamma convolution, leaving neural dynamics alone.

    'Digamma' is interpreted as double gamma, NOT the special digamma function.
    Explicit SPM-style parameters avoid TVB's different default gamma mixture.
    Reconfigure only the BOLD monitor before stepping the simulator.
    """
    if manifest.get('bold_hrf') != 'spm_double_gamma':
        return  # Compatibility for explicitly older Volterra manifests.
    from tvb.datatypes.equations import MixtureOfGammas
    monitor = sim.monitors[1]
    monitor.hrf_kernel = MixtureOfGammas(parameters=dict(
        a_1=6., a_2=16., l=1., c=1/6, gamma_a_1=1., gamma_a_2=1.))
    monitor.hrf_length = 32000.
    monitor.config_for_sim(sim)
    # TVB uses an unnormalised discrete convolution. Unit DC gain changes only
    # scale (not z-scored features) and makes the convention explicit.
    monitor.hemodynamic_response_function /= monitor.hemodynamic_response_function.sum()


def base_configuration(manifest, dataset):
    """Reuse production first-order split-leak configuration, at occupancy zero.

    No structural renormalisation, Laplacian or TMS is introduced. Personalised
    lengths and the existing coupling equation are retained. The b placeholder
    is overwritten by the prior draw before any simulation.
    """
    from experiments.first_order_gnak.simulate import configuration
    p = manifest['production_protocol']
    cfg, labels, _, _ = configuration(dict(cohort=manifest['cohort'],
        subject=manifest['subject'], occupancy=0.), p, dataset)
    cfg.parameter_overrides.pop('parameter_stimulus', None)
    return replace(cfg, monitor_mode='temporal_average', temporal_average_period_ms=10.,
                   monitor_variables=(0, 1, 2), include_bold_monitor=True,
                   bold_monitor_period_ms=manifest['tr_seconds'] * 1000,
                   bold_monitor_variables=(0,)), labels


def prepare(args):
    """Freeze one participant's data, model settings and common prior bounds."""
    if args.output.exists():
        raise FileExistsError('Use a new output directory; prepare never overwrites a bank')
    if args.tr <= 0 or .10 >= .5 / args.tr or args.transient_seconds < 32:
        raise ValueError('Need positive TR, Nyquist >0.10 Hz, and >=32 s warm-up for HRF history')
    p = json.loads((ROOT / 'experiments/first_order_gnak/protocol.json').read_text())
    index = json.loads((args.dataset_root / 'index.json').read_text())
    normalization = index['connectivity_normalization']
    if normalization['scheme'] != p['normalization_scheme'] or not np.isclose(normalization['divisor'], p['normalization_divisor']):
        raise ValueError('Wrong SC normalization: require shared control maximum edge')
    if args.subject not in index['cohorts'][args.cohort]['subject_ids']:
        raise ValueError('Subject is not in the specified cohort')
    # Remove condition-dependent adaptation from the inherited protocol.
    p['b_pA'] = {c: 0. for c in ('control', 'emcs', 'mcs', 'uws')}
    p['tau_w_ms'] = 500.
    m = dict(version=4, subject=args.subject, cohort=args.cohort, seed=args.seed,
             b_bounds=[0., 80.], g_bounds=[args.g_min, args.g_max],
             tr_seconds=args.tr, transient_seconds=args.transient_seconds,
             production_protocol=p, normalization=normalization,
             empirical_sha256=sha(args.bold), preprocessing_note=args.preprocessing_note,
             bold_hrf='spm_double_gamma', extended_features=True,
             storage_policy='features and diagnostics only; no simulated neural or BOLD time series',
             explicit_short_fcd=True,
             fcd_protocol=dict(window_seconds=60., stride_volumes=1, exclude_diagonal_only=True))
    if args.g_min < 0:
        raise ValueError('G must be nonnegative')
    prior_for(m)  # Validate bounds before constructing the model.
    cfg, labels = base_configuration(m, args.dataset_root)
    with np.load(args.bold, allow_pickle=False) as data:
        bold = align_bold(data['bold'], data['labels'], labels)
    full_bold = bold.copy()
    retained = getattr(args, 'retained_seconds', None)
    if retained is not None:
        n = int(round(retained / args.tr))
        if not np.isclose(n * args.tr, retained) or n > len(bold):
            raise ValueError('Retained duration must match whole volumes and fit empirical recording')
        bold = bold[:n]
    if len(bold) * args.tr < 120 - 1e-9:
        raise ValueError('Need >=120 s for the short-recording FCD protocol')
    m.update(labels=list(map(str, labels)), n_volumes=len(bold))
    m.update(full_empirical_n_volumes=len(full_bold), retained_seconds=len(bold)*args.tr,
             empirical_selection='all matched-length segments, with end-anchored final segment; window-marginalized inference, not independent likelihood multiplication')
    from experiments.first_order_gnak.simulate import digest
    m.update(weights_sha256=digest(cfg.weights), tracts_sha256=digest(cfg.tract_lengths))
    if getattr(args, 'state_features', False):
        m['state_template'] = make_state_template(bold, cfg.weights, m)
    from importlib.metadata import version, PackageNotFoundError
    m['package_versions'] = {}
    for package in ('numpy', 'scipy', 'tvb-library', 'vbi', 'sbi', 'torch'):
        try:
            m['package_versions'][package] = version(package)
        except PackageNotFoundError:
            m['package_versions'][package] = 'not-installed'
    # Include imported native model/configuration sources, not only this driver.
    paths = sorted((ROOT / 'src/tvbtoolkit').rglob('*.py'))
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'experiments/first_order_gnak/simulate.py',
              ROOT / 'scripts/run_adex_impulse_response.py',
              ROOT / 'scripts/brain_states_new_doc_bold_audited.py',
              ROOT / 'scripts/first_order_amplitude_trials.py',
              ROOT / 'notebooks/brain_act_hybrid_common.py']
    m['source_hashes'] = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    values, names = features(bold, m)
    m['feature_names'] = names
    args.output.mkdir(parents=True)
    starts, segment_weights = empirical_segments(len(full_bold), len(bold))
    segment_features = np.array([features(full_bold[i:i+len(bold)], m)[0] for i in starts])
    np.savez_compressed(args.output / 'observation.npz', bold=bold, full_bold=full_bold,
                        features=values, labels=labels, segment_starts=starts,
                        segment_weights=segment_weights, segment_features=segment_features)
    write_json(args.output / 'manifest.json', m)
    print(f'Prepared {args.subject}: {len(bold)} volumes; {len(values)} features. No simulations run.')


def empirical_segments(total, width):
    """Cover every empirical volume; downweight overlap in the final segment.

    For 297 volumes and width 50, starts are 0,50,100,150,200,247. The three
    overlapping volumes contribute half to each segment's weighting. This is
    window coverage weighting, not a claim of independent segment likelihoods.
    """
    starts = np.unique(np.r_[np.arange(0, total-width+1, width), total-width]).astype(int)
    coverage = np.zeros(total)
    for start in starts:
        coverage[start:start+width] += 1
    weights = np.array([(1/coverage[start:start+width]).sum() for start in starts]) / total
    return starts, weights


def check_sources(manifest):
    """Do not silently mix datasets made with different model/analysis code."""
    for name, expected in manifest['source_hashes'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Code changed since preparation: {name}; use a new bank')


def run_draw(payload):
    """Generate one stochastic baseline and retain numerical failures explicitly.

    Ten-ms neural averages and BOLD are used in RAM, not saved. Save only features,
    parameter/seed provenance and QC summaries. No high-rate case is discarded
    or clipped. Training refuses flagged banks until their prior is reviewed.
    """
    bank, dataset, index, theta_override = payload
    bank = Path(bank)
    m = json.loads((bank / 'manifest.json').read_text())
    folder = bank / ('draws' if theta_override is None else 'predictive') / f'{index:07d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'metadata.json').exists():
        return json.loads((folder / 'metadata.json').read_text())
    theta, seed = draw(m, index)
    if theta_override is not None:
        theta = np.asarray(theta_override)
        seed = int(np.random.default_rng(np.random.SeedSequence([m['seed'], index, 991])).integers(2**31 - 1))
    row = dict(index=index, theta=theta.tolist(), seed=seed, status='failed')
    started = time.monotonic()
    try:
        from experiments.first_order_gnak.simulate import simulator, digest
        from tvbtoolkit.inference.adex import extract_bold_monitor
        cfg, _ = base_configuration(m, dataset)
        if digest(cfg.weights) != m['weights_sha256'] or digest(cfg.tract_lengths) != m['tracts_sha256']:
            raise ValueError('Subject connectome changed since preparation')
        cfg = prior_for(m).apply(cfg, theta)
        period = m['tr_seconds'] * 1000
        # An integer number of discarded BOLD volumes guarantees matched lengths.
        warm = int(np.ceil(m['transient_seconds'] * 1000 / period))
        duration = (warm + m['n_volumes']) * period
        if not np.isclose(period / cfg.dt_ms, round(period / cfg.dt_ms)):
            raise ValueError('TR must be an integer multiple of integration timestep')
        sim = simulator(cfg, seed)
        configure_bold_kernel(sim, m)
        if sim.stimulus is not None or not np.isclose(sim.monitors[0].period, 10.):
            raise ValueError('Unexpected stimulation or neural monitor period')
        row['model_module'] = type(sim.model).__module__
        row['model_parameters'] = {name: np.asarray(getattr(sim.model, name)).tolist()
                                  for name in ('b_e', 'tau_w_e', 'weight_noise', 'tau_OU',
                                               'external_input_ex_ex', 'external_input_in_ex')}
        output = sim.run(simulation_length=duration)
        t, bold = extract_bold_monitor(output, expected_period_ms=period)
        if len(bold) != warm + m['n_volumes']:
            raise ValueError(f'Unexpected BOLD sample count: {len(bold)}')
        bold, t = bold[warm:], t[warm:]
        nt, neural = output[0]
        neural = neural[nt > warm*period, :, :, 0]
        row.update(neural_timeseries_saved=False, bold_timeseries_saved=False,
                   discarded_until_ms=warm*period, neural_QC_monitor_ms=10.)
        if not np.isfinite(neural).all() or not np.isfinite(bold).all():
            raise FloatingPointError('Nonfinite dynamics')
        rates = neural[:, :2] * 1000  # Native kHz -> reporting Hz only.
        row.update(mean_E_hz=float(rates[:, 0].mean()), mean_I_hz=float(rates[:, 1].mean()),
                   max_saved_EI_hz=float(rates.max()), min_saved_EI_hz=float(rates.min()),
                   mean_W_pA=float(neural[:, 2].mean()))
        value, names = features(bold, m)
        if list(names) != m['feature_names']:
            raise ValueError('Feature ordering changed')
        np.savez_compressed(folder / 'features.npz', features=value, theta=theta, seed=seed)
        # A conservative review flag, NOT a validated physiological boundary/PFP test.
        row['status'] = 'review' if rates.max() > 100 or rates.min() < 0 else 'complete'
    except Exception as exc:
        row.update(error=str(exc), traceback=traceback.format_exc())
    row['elapsed_seconds'] = time.monotonic() - started
    write_json(folder / 'metadata.json', row)
    print(index, row['status'], round(row['elapsed_seconds'], 1), flush=True)
    return row


def simulate(args):
    """Run/resume an index range; increasing --stop extends the same bank."""
    m = json.loads((args.bank / 'manifest.json').read_text())
    check_sources(m)
    if args.start < 0 or args.stop <= args.start or args.workers < 1:
        raise ValueError('Require 0 <= start < stop and positive workers')
    # One campaign per bank at a time: no racing resumptions or duplicate writers.
    lock = args.bank / 'RUNNING'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_draw, [(str(args.bank), str(args.dataset_root), i, None)
                                           for i in range(args.start, args.stop)]))
        write_json(args.bank / f'run_{args.start}_{args.stop}.json', rows)
        if any(row['status'] != 'complete' for row in rows):
            raise RuntimeError('Some draws failed or require physiological review; inspect metadata before training')
    finally:
        lock.unlink()


def fit_scores(simulated, observations, weights, names):
    """Descriptive fit score, NOT likelihood, posterior density or a PCI score.

    Standardise features by their spread across the valid prior simulations,
    then give FC, spatial FC, FCD, ACF and spectra equal total weight. Average
    discrepancy over every empirical segment with coverage weights. Higher
    1/(1+RMS discrepancy) is better; a star marks the maximum observed score,
    not a unique parameter truth. Near-constant features contribute no weight.
    """
    scale = np.std(simulated, axis=0)
    usable = scale > 1e-8
    scale = np.where(usable, scale, 1.)
    prefixes = ['vbi_', 'FC_mean_', 'FCD_', 'ACF_', 'BOLD_']
    groups = [np.array([name.startswith(prefix) for name in names]) & usable for prefix in prefixes]
    groups = [group for group in groups if group.any()]
    if not groups:
        raise ValueError('No variable feature families for fit score')
    loss = np.zeros(len(simulated))
    for observation, weight in zip(observations, weights):
        squared = ((simulated-observation)/scale)**2
        loss += weight * np.mean([squared[:, group].mean(axis=1) for group in groups], axis=0)
    return 1/(1+np.sqrt(loss)), scale, usable


def report(args):
    """Plot b–G coloured by fit; mark the highest-scoring VALID draw with a star.

    This report is separate from SBI and can be run after the pilot or full bank.
    Invalid draws stay in the numerical table and appear as grey crosses.
    """
    import csv
    import matplotlib.pyplot as plt
    m = json.loads((args.bank / 'manifest.json').read_text())
    check_sources(m)
    rows, vectors, valid_rows = [], [], []
    for i in range(args.budget):
        folder = args.bank / 'draws' / f'{i:07d}'
        path = folder / 'metadata.json'
        if not path.exists():
            raise ValueError(f'Missing draw {i}; reduce report budget or finish simulations')
        row = json.loads(path.read_text())
        rows.append(row)
        if row['status'] == 'complete':
            with np.load(folder / 'features.npz') as data:
                vectors.append(data['features'])
            valid_rows.append(i)
    if len(vectors) < 2:
        raise ValueError('Need at least two valid draws for descriptive scoring')
    with np.load(args.bank / 'observation.npz') as data:
        scores, scale, usable = fit_scores(np.array(vectors), data['segment_features'],
                                           data['segment_weights'], m['feature_names'])
    score_by_index = dict(zip(valid_rows, scores))
    winner = valid_rows[int(np.argmax(scores))]
    output = args.bank / f'fit_map_{args.budget}'
    output.mkdir(exist_ok=False)
    with (output / 'fit_scores.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['index', 'b_pA', 'G', 'seed', 'status', 'fit_score'])
        for i, row in enumerate(rows):
            writer.writerow([i, *row['theta'], row['seed'], row['status'], score_by_index.get(i, '')])
    write_json(output / 'best_fit.json', dict(subject=m['subject'], cohort=m['cohort'],
        draw=winner, b_pA=rows[winner]['theta'][0], G=rows[winner]['theta'][1],
        seed=rows[winner]['seed'], score=float(scores.max()),
        interpretation='maximum observed descriptive feature-fit score among valid draws; not a MAP estimate'))
    np.savez_compressed(output / 'score_normalisation.npz', scale=scale, usable=usable)
    theta = np.array([rows[i]['theta'] for i in valid_rows])
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    points = ax.scatter(theta[:, 0], theta[:, 1], c=scores, cmap='viridis', s=13, rasterized=True)
    invalid = np.array([row['theta'] for row in rows if row['status'] != 'complete'])
    if len(invalid):
        ax.scatter(invalid[:, 0], invalid[:, 1], marker='x', color='.7', s=9, label='Invalid / review')
    ax.scatter(*rows[winner]['theta'], marker='*', s=230, color='red', edgecolor='white', linewidth=1,
               label='Highest observed fit score', zorder=5)
    ax.set(xlabel='Spike-triggered adaptation b (pA)', ylabel='Global coupling G',
           title=f'{m["cohort"]} / {m["subject"]}: {len(vectors)} valid draws')
    ax.legend(); fig.colorbar(points, ax=ax, label='Descriptive fit score (higher is better)')
    fig.savefig(output / 'b_G_fit_map.png', dpi=300)
    fig.savefig(output / 'b_G_fit_map.pdf'); plt.close(fig)


def train(args):
    """Fit one participant; reserve the last 10% for synthetic recovery checks.

    Never drop failed/high-rate draws to produce an implicitly truncated prior.
    If a bank fails QC, inspect it and prospectively redefine a common prior in
    a new bank. Held-out coverage is diagnostic, not proof of empirical validity.
    """
    import torch
    from tvbtoolkit.inference.sbi import SimulationDataset, train_vbi_posterior, sample_vbi_posterior
    m = json.loads((args.bank / 'manifest.json').read_text())
    check_sources(m)
    if args.budget < 100 or args.recovery_cases < 1:
        raise ValueError('Require >=100 draws and positive recovery_cases (not a sufficiency claim)')
    theta, values, seeds = [], [], []
    for i in range(args.budget):
        folder = args.bank / 'draws' / f'{i:07d}'
        row = json.loads((folder / 'metadata.json').read_text())
        if row['status'] != 'complete':
            raise ValueError(f'Draw {i} is {row["status"]}: review bank; no silent exclusions')
        with np.load(folder / 'features.npz') as data:
            theta.append(data['theta']); values.append(data['features']); seeds.append(data['seed'])
    theta, values, seeds = np.array(theta), np.array(values), np.array(seeds)
    rng = np.random.default_rng(args.training_seed)
    order = rng.permutation(len(theta))
    split = int(.9 * len(order))
    use, hold = order[:split], order[split:]
    output = args.bank / f'fit_{args.budget}_seed{args.training_seed}'
    output.mkdir()  # Deliberately refuse overwriting an inference result.
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    ds = SimulationDataset(theta[use], values[use], ('b', 'G'), tuple(m['feature_names']), seeds[use])
    ds.save(output / 'training.npz')
    posterior = train_vbi_posterior(ds, prior_for(m), num_threads=args.threads)
    # Only load this file if you trust its source: torch posterior objects use pickle.
    torch.save(posterior, output / 'posterior.pt')
    with np.load(args.bank / 'observation.npz') as data:
        observed = data['features']
        segments, weights = data['segment_features'], data['segment_weights']
    segment_samples = np.array([sample_vbi_posterior(posterior, x, num_samples=10000) for x in segments])
    choices = rng.choice(len(segments), size=10000, p=weights)
    samples = segment_samples[choices, rng.integers(10000, size=10000)]
    np.savez_compressed(output / 'segment_posteriors.npz', samples=segment_samples, weights=weights)
    selected = hold[:args.recovery_cases]
    recovery = np.array([sample_vbi_posterior(posterior, values[i], num_samples=1000) for i in selected])
    lo, hi = np.quantile(recovery, [.05, .95], axis=1)
    coverage = ((theta[selected] >= lo) & (theta[selected] <= hi)).mean(axis=0)
    np.savez_compressed(output / 'posterior_samples.npz', samples=samples, parameter_names=['b', 'G'])
    np.savez_compressed(output / 'recovery.npz', truth=theta[selected], samples=recovery,
                        heldout_indices=hold, tested_indices=selected)
    write_json(output / 'summary.json', dict(budget=args.budget, training_draws=len(use),
        recovery_cases=len(selected), marginal_90pct_coverage=coverage.tolist(),
        median=np.median(samples, axis=0).tolist(), interval_90pct=np.quantile(samples, [.05, .95], axis=0).tolist(),
        posterior_correlation=float(np.corrcoef(samples.T)[0, 1]),
        inference_definition='coverage-weighted mixture of segment-conditioned posteriors; not an all-segments joint likelihood posterior',
        warning='Provisional until recovery, seed/budget stability and posterior prediction pass'))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    axes[0].hexbin(samples[:, 0], samples[:, 1], gridsize=40, mincnt=1)
    axes[0].set(xlabel='b (pA)', ylabel='G', title='Joint posterior')
    for j, name in enumerate(('b (pA)', 'G')):
        pred = np.median(recovery[:, :, j], axis=1)
        axes[j+1].scatter(theta[selected, j], pred)
        limits = [prior_for(m).low[j], prior_for(m).high[j]]
        axes[j+1].plot(limits, limits, 'k--')
        axes[j+1].set(xlabel=f'True {name}', ylabel=f'Posterior median {name}', title='Held-out recovery')
    fig.savefig(output / 'posterior_and_recovery.pdf'); plt.close(fig)


def predict(args):
    """New stochastic simulations at joint-posterior draws, never at separate medians."""
    m = json.loads((args.bank / 'manifest.json').read_text())
    check_sources(m)
    with np.load(args.fit / 'posterior_samples.npz') as data:
        samples = data['samples']
    if args.fit.parent.resolve() != args.bank.resolve() or not 0 < args.count <= len(samples):
        raise ValueError('Fit must belong to this bank and count must be positive/in range')
    # One predictive set per bank avoids mistakenly mixing fits on resume.
    target = args.bank / 'predictive'
    target.mkdir()
    write_json(target / 'manifest.json', dict(fit=str(args.fit), posterior_sha256=sha(args.fit / 'posterior_samples.npz')))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_draw, [(str(args.bank), str(args.dataset_root), i, samples[i].tolist())
                                       for i in range(args.count)]))
    write_json(target / 'summary.json', rows)


def main():
    """CLI separates data preparation, expensive simulations, inference and prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Audit/freeze labelled empirical data and model settings')
    p.add_argument('--bold', type=Path, required=True, help='NPZ: bold[time,90], labels[90]')
    p.add_argument('--subject', required=True)
    p.add_argument('--cohort', choices=['control', 'emcs', 'mcs', 'uws'], required=True)
    p.add_argument('--tr', type=float, required=True, help='Empirical sampling interval in seconds')
    p.add_argument('--g-min', type=float, required=True)
    p.add_argument('--g-max', type=float, required=True)
    p.add_argument('--transient-seconds', type=float, default=60.)
    p.add_argument('--retained-seconds', type=float, default=120., help='Matched empirical/simulated BOLD duration after warm-up')
    p.add_argument('--state-features', action='store_true', help='Fit five frozen empirical states; add occupancy and SC–phase-FC coupling')
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--preprocessing-note', required=True, help='Motion/confound/dummy-volume processing provenance')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    for name in ('simulate', 'train', 'predict', 'report'):
        q = commands.add_parser(name)
        q.add_argument('--bank', type=Path, required=True)
        if name in ('simulate', 'predict'):
            q.add_argument('--dataset-root', type=Path, required=True)
            q.add_argument('--workers', type=int, default=1)
        if name == 'simulate':
            q.add_argument('--start', type=int, default=0)
            q.add_argument('--stop', type=int, default=16, help='Exclusive draw index; initial benchmark defaults to 16')
        elif name == 'report':
            q.add_argument('--budget', type=int, required=True)
        elif name == 'train':
            q.add_argument('--budget', type=int, required=True)
            q.add_argument('--training-seed', type=int, default=1)
            q.add_argument('--threads', type=int, default=1)
            q.add_argument('--recovery-cases', type=int, default=100)
        else:
            q.add_argument('--fit', type=Path, required=True)
            q.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
