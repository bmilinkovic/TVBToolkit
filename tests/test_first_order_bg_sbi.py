"""Fast safeguards for the modular b–G inference driver; no scientific fit claimed."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/first_order_bg_sbi/workflow.py'
spec = importlib.util.spec_from_file_location('bg_workflow', PATH)
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)


def test_prior_and_indexed_draws():
    m = dict(b_bounds=[0, 80], g_bounds=[0, .1], seed=73)
    p = bg.prior_for(m)
    assert p.names == ('b', 'G')
    first, seed = bg.draw(m, 0)
    bg.draw(m, 100000)
    again, seed2 = bg.draw(m, 0)
    np.testing.assert_array_equal(first, again)
    assert seed == seed2
    assert bg.draw(m, 1)[1] != seed
    assert np.all(first >= p.low) and np.all(first <= p.high)


def test_explicit_region_alignment():
    labels = np.array([f'region_{i}' for i in range(90)])
    x = np.arange(180).reshape(2, 90)
    np.testing.assert_array_equal(bg.align_bold(x[:, ::-1], labels[::-1], labels), x)
    with pytest.raises(ValueError):
        bg.align_bold(x, ['duplicate'] * 90, labels)
    with pytest.raises(ValueError):
        bg.align_bold(x.T, labels, labels)


def test_parameter_mapping_does_not_mutate_original():
    from tvbtoolkit.core.config import WholeBrainConfig
    original = WholeBrainConfig(zerlaut_order=1, coupling_strength=.05,
        parameter_overrides={'parameter_model': {'b_e': 5., 'tau_w_e': 500., 'weight_noise': .0001}})
    changed = bg.prior_for(dict(b_bounds=[0, 80], g_bounds=[0, .1])).apply(original, [60, .08])
    assert changed.coupling_strength == .08
    assert changed.parameter_overrides['parameter_model']['b_e'] == 60
    assert changed.parameter_overrides['parameter_model']['tau_w_e'] == 500
    assert original.parameter_overrides['parameter_model']['b_e'] == 5


def test_feature_order_and_reproducibility():
    pytest.importorskip('vbi')
    x = np.random.default_rng(7).normal(size=(180, 90))
    m = dict(tr_seconds=2., labels=[f'r{i}' for i in range(90)])
    a, names = bg.features(x, m)
    b, again = bg.features(x, m)
    np.testing.assert_array_equal(a, b)
    assert names == again and len(a) == len(names)
    assert np.isfinite(a).all() and 'FC_mean_r0' in names


def test_native_monitor_configuration():
    """Catch monitor-setting shadowing and accidental second-order/TMS activation."""
    import json
    from experiments.first_order_gnak.simulate import simulator
    root = PATH.parents[2]
    dataset = root / 'data/doc_data/converted_structural_invnodevol_control_max'
    if not dataset.exists():
        pytest.skip('Local normalized connectomes unavailable')
    p = json.loads((root / 'experiments/first_order_gnak/protocol.json').read_text())
    m = dict(production_protocol=p, cohort='control', subject='c0005', tr_seconds=2.)
    cfg, labels = bg.base_configuration(m, dataset)
    sim = simulator(cfg, 7)
    assert len(labels) == 90
    assert list(sim.model.state_variables) == ['E', 'I', 'W_e', 'W_i', 'noise']
    assert 'gK_gNa' in type(sim.model).__module__
    assert sim.stimulus is None
    assert sim.monitors[0].period == 10
    assert sim.monitors[1].period == 2000


def test_prepare_freezes_subject_bank(tmp_path):
    """Exercise the real preparation stage using explicitly synthetic BOLD."""
    import json
    from argparse import Namespace
    root = PATH.parents[2]
    dataset = root / 'data/doc_data/converted_structural_invnodevol_control_max'
    if not dataset.exists():
        pytest.skip('Local normalized connectomes unavailable')
    p = json.loads((root / 'experiments/first_order_gnak/protocol.json').read_text())
    _, labels = bg.base_configuration(dict(production_protocol=p, cohort='control',
                                         subject='c0005', tr_seconds=2.), dataset)
    source = tmp_path / 'synthetic.npz'
    np.savez_compressed(source, bold=np.random.default_rng(19).normal(size=(180, 90)), labels=labels)
    args = Namespace(output=tmp_path / 'bank', tr=2., transient_seconds=60.,
        dataset_root=dataset, subject='c0005', cohort='control', seed=19,
        g_min=0., g_max=.1, bold=source, preprocessing_note='Synthetic unit test, NOT participant data')
    bg.prepare(args)
    m = json.loads((args.output / 'manifest.json').read_text())
    assert m['n_volumes'] == 180
    assert set(m['production_protocol']['b_pA'].values()) == {0.}
    bg.check_sources(m)
    assert not (args.output / 'draws').exists()
    with pytest.raises(FileExistsError):
        bg.prepare(args)


def test_double_gamma_matches_declared_formula():
    from scipy.stats import gamma
    from tvb.simulator.monitors import Bold
    from types import SimpleNamespace
    monitor = Bold()
    # Configure kernel without a full simulation; real monitor integration is
    # separately checked in test_native_monitor_configuration.
    monitor.config_for_sim = lambda sim: monitor.compute_hrf()
    monitor.dt = .1
    bg.configure_bold_kernel(SimpleNamespace(monitors=[None, monitor]), {'bold_hrf': 'spm_double_gamma'})
    t = monitor._stock_time
    expected = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    expected /= expected.sum()
    np.testing.assert_allclose(monitor.hemodynamic_response_function[0], expected[::-1], atol=1e-14)
    assert monitor.hrf_length == 32000
    assert expected.min() < 0 and expected.max() > 0


def test_state_template_is_fixed_and_occupancy_sums_to_one():
    rng = np.random.default_rng(11)
    bold = rng.normal(size=(180, 90))
    w = rng.uniform(size=(90, 90)); w = (w + w.T) / 2
    m = dict(tr_seconds=2.)
    m['state_template'] = bg.make_state_template(bold, w, m)
    before = np.array(m['state_template']['centers'])
    values, names = bg.state_features(bold, m)
    assert len(names) == 10 and np.isclose(np.sum(values[::2]), 1.)
    np.testing.assert_array_equal(before, m['state_template']['centers'])


def test_two_minute_fcd_is_finite_and_matches_manual_windows():
    from scipy.stats import skew, kurtosis
    x = np.random.default_rng(8).normal(size=(50, 90))
    got, names = bg.short_fcd_features(x, 2.4)
    edges = np.array([np.corrcoef(x[i:i+25].T)[np.triu_indices(90, 1)] for i in range(26)])
    pairs = np.corrcoef(edges)[np.triu_indices(26, 1)]
    assert len(pairs) == 325 and len(names) == 9
    np.testing.assert_allclose(got[:4], [pairs.mean(), pairs.std(), skew(pairs), kurtosis(pairs)])
    m = dict(tr_seconds=2.4, labels=[f'r{i}' for i in range(90)],
             extended_features=True, explicit_short_fcd=True)
    result, names = bg.features(x, m)
    assert np.isfinite(result).all() and not any('state_' in name for name in names)
    assert len(result) == len(names) == 119
    assert len([n for n in names if n.startswith('FC_mean_')]) == 90
    filtered = bg.preprocess_short_bold(x, 2.4)
    assert filtered.shape == (50, 90)
    from scipy.signal import iirfilter, filtfilt
    from scipy.stats import zscore
    b, a = iirfilter(2, [.01/(.5/2.4), .1/(.5/2.4)], btype='bandpass', ftype='butter')
    np.testing.assert_allclose(filtered, filtfilt(b, a, zscore(x, axis=0), axis=0))


def test_all_empirical_volumes_are_covered():
    starts, weights = bg.empirical_segments(297, 50)
    np.testing.assert_array_equal(starts, [0, 50, 100, 150, 200, 247])
    coverage = np.zeros(297)
    for start in starts:
        coverage[start:start+50] += 1
    assert np.all(coverage > 0) and np.isclose(weights.sum(), 1.)


def test_fit_score_marks_exact_match_best():
    x = np.array([[1., 1.], [0., 0.], [2., 2.]])
    scores, _, _ = bg.fit_scores(x, np.array([[1., 1.]]), np.array([1.]), ['FC_mean_a', 'ACF_a'])
    assert np.argmax(scores) == 0 and scores[0] == 1.


def test_no_raw_simulation_storage_policy():
    import inspect
    source = inspect.getsource(bg.run_draw)
    assert 'signals.npz' not in source and 'bold.npz' not in source and 'neural.npy' not in source
    assert 'features.npz' in source


def test_report_synthetic_bank(tmp_path):
    """Smoke-test CSV, best-fit marker selection and both figure formats."""
    import json
    from argparse import Namespace
    m = dict(subject='synthetic', cohort='control', source_hashes={}, feature_names=['FC_mean_a', 'ACF_a'])
    bg.write_json(tmp_path/'manifest.json', m)
    np.savez_compressed(tmp_path/'observation.npz', segment_features=[[1.,1.]], segment_weights=[1.])
    for i, value in enumerate([1., 0., 2.]):
        folder = tmp_path/'draws'/f'{i:07d}'; folder.mkdir(parents=True)
        bg.write_json(folder/'metadata.json', dict(theta=[i*20., .025+i*.02], seed=i, status='complete'))
        np.savez_compressed(folder/'features.npz', features=[value,value])
    bg.report(Namespace(bank=tmp_path, budget=3))
    assert json.loads((tmp_path/'fit_map_3/best_fit.json').read_text())['draw'] == 0
    assert (tmp_path/'fit_map_3/b_G_fit_map.png').exists()


def test_run_draw_saves_summaries_only(tmp_path, monkeypatch):
    """Exercise the complete recording-to-feature path with fake TVB monitors."""
    from types import SimpleNamespace
    from tvbtoolkit.core.config import WholeBrainConfig
    import experiments.first_order_gnak.simulate as native
    weights = np.eye(90)
    m = dict(subject='synthetic', cohort='control', seed=1, b_bounds=[0,80], g_bounds=[.025,.075],
        tr_seconds=2.4, transient_seconds=60., n_volumes=50, labels=[f'r{i}' for i in range(90)],
        extended_features=True, explicit_short_fcd=True,
        weights_sha256=native.digest(weights), tracts_sha256=native.digest(weights))
    rng = np.random.default_rng(9)
    bold = rng.normal(size=(75,1,90,1))
    m['feature_names'] = list(bg.features(bold[25:,0,:,0], m)[1])
    bg.write_json(tmp_path/'manifest.json', m)
    cfg = WholeBrainConfig(weights=weights, tract_lengths=weights, zerlaut_order=1)
    monkeypatch.setattr(bg, 'base_configuration', lambda *args: (cfg, m['labels']))
    model = SimpleNamespace(**{k:1. for k in ('b_e','tau_w_e','weight_noise','tau_OU','external_input_ex_ex','external_input_in_ex')})
    raw = np.full((4,3,90,1), .005)
    output = [(np.array([10.,60000.,70000.,180000.]),raw), (np.arange(1,76)*2400.,bold)]
    fake = SimpleNamespace(stimulus=None, monitors=[SimpleNamespace(period=10.)], model=model,
                           run=lambda **kwargs: output)
    monkeypatch.setattr(native, 'simulator', lambda *args: fake)
    row = bg.run_draw((str(tmp_path),'unused',0,None))
    assert row['status'] == 'complete'
    assert not row['neural_timeseries_saved'] and not row['bold_timeseries_saved']
    assert {p.name for p in (tmp_path/'draws/0000000').iterdir()} == {'features.npz','metadata.json'}
