"""Export four matched Liege BOLD/SC examples and prepare a portable SBI pilot.

Selection is deterministic, independent of FC/PCI: retain control c0005 and UWS
u0033 from earlier work, choose the first chronic non-sedated EMCS/MCS with finite
BOLD. Match via the structural source map, condition/stage/sedation and original
within-file index. No COMA data are loaded. The supplied matched file order is
the available identity evidence; missing subject names are explicitly recorded.

Run with --help. This prepares data/configurations only, never simulations.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from argparse import Namespace
import numpy as np

from workflow import ROOT, prepare, sha, write_json
sys.path.insert(0, str(ROOT / 'scripts'))
from brain_states_new_doc_bold_audited import (
    FILE_SPECS, _load_mat_mapping, _select_3d_numeric, _to_subject_roi_time,
    _to_subject_roi_roi, build_roi_order_reference, _decode_subject_names,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Directory containing CNT_send and FC_send')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tr', type=float, default=2.4, help='TR in seconds; user confirmed 2.4 s')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_map = args.dataset_root / 'source_subject_map.csv'
    with source_map.open() as stream:
        mapping = list(csv.DictReader(stream))
    labels = np.array(build_roi_order_reference(args.data_root).interleaved_names)
    jobs, audits = [], []
    for group in ('control', 'emcs', 'mcs', 'uws'):
        rows = [r for r in mapping if r['cohort'] == group and r['sedation'] == 'non_sedated'
                and r['stage'] == ('control' if group == 'control' else 'chronic')]
        preferred = {'control': 'c0005', 'uws': 'u0033'}.get(group)
        rows.sort(key=lambda r: (r['subject_id'] != preferred, r['subject_id']))
        selected = None
        for row in rows:
            spec = next(s for s in FILE_SPECS if all(s[k] == row[k] for k in ('cohort', 'stage', 'sedation')))
            fc_path = args.data_root / spec['fc_path']
            raw = _load_mat_mapping(fc_path)
            var, array = _select_3d_numeric(raw, spec['fc_var'])
            timeseries = _to_subject_roi_time(array)
            position = int(row['source_subject_index'])
            # Verify the matched count against the exact inverse-volume SC source.
            sc_raw = _load_mat_mapping(args.data_root / row['source_sc_file'])
            _, sc_array = _select_3d_numeric(sc_raw, 'SC')
            if len(_to_subject_roi_roi(sc_array)) != len(timeseries):
                raise ValueError('SC/BOLD subject count mismatch')
            bold = timeseries[position].T
            if not np.isfinite(bold).all() or np.any(bold.std(axis=0) <= 1e-12):
                audits.append(dict(subject=row['subject_id'], cohort=group, status='excluded_nonfinite_or_constant'))
                continue
            fc_names = _decode_subject_names(raw.get('subj_names'))
            sc_names = _decode_subject_names(sc_raw.get('subj_names'))
            if fc_names and sc_names and fc_names[position] != sc_names[position]:
                raise ValueError('SC/BOLD subject-name mismatch')
            selected = row
            break
        if selected is None:
            raise ValueError(f'No eligible matched BOLD for {group}')
        subject = selected['subject_id']
        exported = args.output / f'{group}_{subject}_bold.npz'
        np.savez_compressed(exported, bold=bold, labels=labels)
        provenance = dict(subject=subject, cohort=group, stage=selected['stage'],
            sedation=selected['sedation'], source_subject_index=position,
            source_fc_file=str(fc_path), source_fc_sha256=sha(fc_path), fc_variable=var,
            source_sc_file=selected['source_sc_file'], n_volumes=len(bold),
            matching='names and matched index' if fc_names and sc_names else 'provided matched file index; names unavailable in one or both modalities',
            tr_seconds=args.tr, tr_evidence='user confirmed 2.4 s; also used in previous audited analysis',
            preprocessing_status='upstream temporal filtering/motion processing not documented in supplied MAT variables',
            roi_mapping='ROI_MNI_V4_90 interleaved labels -> normalized atlas labels')
        bank = args.output / f'{group}_{subject}'
        prepare(Namespace(output=bank, tr=args.tr, transient_seconds=60., retained_seconds=120.,
            dataset_root=args.dataset_root, subject=subject, cohort=group, seed=20261004,
            g_min=.025, g_max=.075, bold=exported, state_features=False,
            preprocessing_note=json.dumps(provenance)))
        audits.append(provenance)
        jobs.append(dict(cohort=group, subject=subject, bank=bank.name, simulations=10000))
    write_json(args.output / 'pilot.json', dict(jobs=jobs, total_simulations=40000,
        metadata_review_required=False, tr_confirmed=True, state_features=False, retained_seconds=120.,
        source_map_sha256=sha(source_map),
        explanation='TR=2.4 s confirmed by user; no state occupancy or SC–FC fit; upstream preprocessing provenance remains documented in input audit'))
    write_json(args.output / 'input_audit.json', audits)
    print('Prepared four banks: TR=2.4 s, no state features. No simulations run.')


if __name__ == '__main__':
    main()
