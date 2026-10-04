"""Prepare clean cluster banks from the small exported empirical-input bundle.

Run after git pull, using the SAME checkout that will run SLURM. This deliberately
does not reuse local manifests fingerprinted against uncommitted research code.
Only empirical NPZ inputs and the subject list are transferred, never model
source or simulation output. No simulation occurs during preparation.
"""
import argparse
from argparse import Namespace
import json
from pathlib import Path
from workflow import prepare, write_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup-seconds', type=float, default=60., help='Conservative default; assess shorter warm-up in a separate pilot bank')
    args = p.parse_args()
    spec = json.loads((args.inputs / 'pilot.json').read_text())
    if len(spec['jobs']) != 4 or any(j['cohort'] == 'coma' for j in spec['jobs']):
        raise ValueError('Expected four non-COMA pilot subjects')
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for job in spec['jobs']:
        group, subject = job['cohort'], job['subject']
        source = args.inputs / f'{group}_{subject}_bold.npz'
        bank = f'{group}_{subject}'
        prepare(Namespace(output=args.output / bank, tr=2.4, transient_seconds=args.warmup_seconds,
            retained_seconds=120., dataset_root=args.dataset_root, subject=subject,
            cohort=group, seed=20261004, g_min=.025, g_max=.075,
            bold=source, state_features=False,
            preprocessing_note='User-confirmed TR=2.4 s; exported matched Liege BOLD, upstream temporal preprocessing undocumented; source SHA256='+sha(source)))
        jobs.append(dict(cohort=group, subject=subject, bank=bank, simulations=10000))
    write_json(args.output / 'pilot.json', dict(jobs=jobs, total_simulations=40000,
        retained_seconds=120., warmup_seconds=args.warmup_seconds, tr_confirmed=True,
        state_features=False, storage_policy='summaries_only', version=4))


if __name__ == '__main__':
    main()
