from __future__ import annotations

import numpy as np
import pytest

from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90


def test_get_5ht2a_aal90_loads_static_atlas() -> None:
    receptor_map = get_5ht2a_aal90()

    assert receptor_map.shape == (90,)
    assert np.isfinite(receptor_map).all()
    assert receptor_map.min() >= 0.0
    assert receptor_map.max() <= 1.0


def test_get_5ht2a_aal90_rejects_unknown_tracer() -> None:
    with pytest.raises(ValueError, match="tracer must be one of"):
        get_5ht2a_aal90("unknown")


def test_get_5ht2a_aal90_aligns_values_by_region_label() -> None:
    source_order = get_5ht2a_aal90()

    # Use the real table labels while requesting their reverse order.  This
    # catches accidental positional assignment without hard-coding PET values.
    import csv
    from pathlib import Path

    table_path = Path(__file__).resolve().parents[1] / "data" / "receptors" / "hansen_receptors_aal90.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source_labels = np.asarray([row[""] for row in rows], dtype=object)
    requested = source_labels[::-1]

    aligned = get_5ht2a_aal90(target_labels=requested)
    np.testing.assert_allclose(aligned, source_order[::-1])


def test_get_5ht2a_aal90_rejects_unmatched_target_labels() -> None:
    labels = np.asarray([f"not_a_region_{i}" for i in range(90)], dtype=object)
    with pytest.raises(ValueError, match="Could not align"):
        get_5ht2a_aal90(target_labels=labels)


def test_get_5ht2a_aal90_rejects_duplicate_target_labels() -> None:
    labels = np.asarray(["Precentral_L"] * 90, dtype=object)
    with pytest.raises(ValueError, match="must be unique"):
        get_5ht2a_aal90(target_labels=labels)


def test_production_atlas_matches_unshifted_raw_parcels():
    import pandas as pd
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / 'data/receptors'
    raw = pd.read_csv(folder / 'hansen_receptors_aal90_raw.csv', index_col=0)
    production = pd.read_csv(folder / 'hansen_receptors_aal90.csv', index_col=0)
    assert raw.shape == production.shape == (90, 37)
    assert list(raw.index) == list(production.index)
    assert list(raw.columns) == list(production.columns)
    assert raw.index[0] == 'Precentral_L'
    assert raw.index[-1] == 'Temporal_Inf_R'
    # Independent values from direct Cimbi PET re-extraction, not the old CSV.
    np.testing.assert_allclose(raw['5HT2a_cimbi_hc29_beliveau'].iloc[:3],
        [16.404373168945312, 14.332442283630371, 18.761463165283203], rtol=1e-7)
    np.testing.assert_allclose(production, raw / raw.max(), rtol=1e-13, atol=1e-14)
    for tracer, col in [('cimbi','5HT2a_cimbi_hc29_beliveau'),
                        ('savli','5HT2a_alt_hc19_savli'),
                        ('talbot','5HT2a_mdl_hc3_talbot')]:
        np.testing.assert_allclose(get_5ht2a_aal90(tracer), (raw[col]/raw[col].max()))


def test_brian_entry_point_uses_same_aligned_map():
    # Load the lightweight file without importing optional Brian2 simulations.
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'src/tvbtoolkit/brian_mf/receptors.py'
    spec = importlib.util.spec_from_file_location('receptor_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = module.get_hansen_receptors_aal90().index.to_numpy()[::-1]
    for tracer in ('cimbi', 'savli', 'talbot'):
        np.testing.assert_array_equal(module.get_5ht2a_aal90(tracer, target_labels=labels),
                                      get_5ht2a_aal90(tracer, target_labels=labels))


def test_known_bad_atlas_is_rejected_before_loading(tmp_path, monkeypatch):
    import hashlib
    from tvbtoolkit.workflows import pharmacology
    path = tmp_path/'bad.csv'
    path.write_text('historical fixture')
    monkeypatch.setattr(pharmacology, '_MISALIGNED_ATLAS_SHA256',
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='one-region-shifted'):
        get_5ht2a_aal90(csv_path=path)


def test_gnak_configuration_receives_corrected_map(monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    from tvbtoolkit.core.config import WholeBrainConfig
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('gnak_sim', root/'experiments/first_order_gnak/simulate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import pandas as pd
    labels = pd.read_csv(root/'data/receptors/hansen_receptors_aal90_raw.csv',index_col=0).index.to_numpy()[::-1]
    def build(*args):
        return WholeBrainConfig(parameter_overrides={'parameter_model':{'E_L_e':-63.,'E_L_i':-65.}}),labels,None
    monkeypatch.setattr(module,'build_subject_config',build)
    protocol=json.loads((root/'experiments/first_order_gnak/protocol.json').read_text())
    baseline=None
    for occ in (0.,.25,.5,.766):
        cfg, actual_labels, _, rho=module.configuration(dict(cohort='control',subject='test',occupancy=occ),protocol,root)
        x=get_5ht2a_aal90(target_labels=labels)
        expected=(x-x.min())/(np.ptp(x)+1e-12)
        np.testing.assert_array_equal(actual_labels,labels)
        np.testing.assert_allclose(rho,expected)
        model=cfg.parameter_overrides['parameter_model']
        if baseline is None:baseline=model
        for pop in ('e','i'):
            gk,gna=module.leak_to_conductances(50.,-90.,model['E_L_'+pop],g_L=10.)
            drug,_=module.leak_to_conductances(50.,-90.,protocol['drug_E_L_'+pop+'_mV'],g_Na=gna)
            np.testing.assert_allclose(model['g_K_'+pop],gk-occ*expected*(gk-drug))
            if occ==0:np.testing.assert_allclose(model['g_K_'+pop],gk)
