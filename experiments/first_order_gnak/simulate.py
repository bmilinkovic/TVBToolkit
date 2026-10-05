"""First-order gNa/gK recordings only; no PCI imports or computation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from run_adex_impulse_response import build_subject_config
from first_order_amplitude_trials import install_noise
from tvbtoolkit.whole_brain.simulation import configure_adex_simulator
from tvbtoolkit.workflows.pharmacology import get_5ht2a_aal90,leak_to_conductances

def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write_json(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(path)

def configuration(job,p,dataset,split=True):
    cfg,labels,_=build_subject_config(job['cohort'],job['subject'],p['b_pA'][job['cohort']],Path(dataset),p['G'],tuple(range(5)))
    cfg=replace(cfg,zerlaut_order=1,zerlaut_gk_gna=split,dt_ms=p['dt_ms'],conduction_speed=p['conduction_speed'])
    m=cfg.parameter_overrides['parameter_model'];m.pop('derivative_scale',None)
    for k in ('C_ee','C_ei','C_ii'):m.get('initial_condition',{}).pop(k,None)
    m.update(tau_w_e=p['tau_w_ms'],external_input_ex_ex=p['drive_hz']/1000,external_input_in_ex=p['drive_hz']/1000)
    cfg.parameter_overrides['parameter_integrator']={'type':'Heun'}
    receptor=get_5ht2a_aal90(target_labels=labels)
    rho=(receptor-receptor.min())/(receptor.max()-receptor.min()+1e-12)
    if split:
        for pop in ('e','i'):
            gk,gna=leak_to_conductances(50.,-90.,float(m['E_L_'+pop]),g_L=10.)
            drug,_=leak_to_conductances(50.,-90.,p['drug_E_L_'+pop+'_mV'],g_Na=gna)
            m['g_K_'+pop]=(gk-job['occupancy']*rho*(gk-drug)).tolist()
            m['g_Na_'+pop]=float(gna)
            m['E_K_'+pop]=-90.;m['E_Na_'+pop]=50.
    indices=np.flatnonzero(labels==p['source_label'])
    if len(indices)!=1:raise ValueError('Source label must match exactly once')
    return cfg,labels,int(indices[0]),rho

def simulator(cfg,seed):
    sim=configure_adex_simulator(cfg,seed)
    assert tuple(sim.model.state_variables)==('E','I','W_e','W_i','noise')
    for k in ('state_variable_boundaries','_integration_state_variable_boundaries','clamped_state_variable_values','_clamped_integration_state_variable_values'):setattr(sim.integrator,k,None)
    install_noise(sim,seed)
    return sim

def run_one(payload):
    job,p,dataset,out=payload
    name=f"{job['cohort']}_{job['subject']}_o{job['occupancy']:.3f}_r{job['trial']:03d}_warm{job['warmup_ms']}"
    folder=Path(out)/'trials'/name;folder.mkdir(parents=True,exist_ok=True)
    # Protect direct worker calls too: changed receptor inputs cannot reuse an
    # old trial merely because its subject/protocol/seed are unchanged.
    receptor_csv_sha256=hashlib.sha256((ROOT/'data/receptors/hansen_receptors_aal90.csv').read_bytes()).hexdigest()
    sig=signature(dict(job=job,protocol=p,receptor_csv_sha256=receptor_csv_sha256));target=folder/'metadata.json'
    if target.exists():
        previous=json.loads(target.read_text())
        if previous['signature']!=sig:raise ValueError('Cache signature mismatch')
        return previous
    row=dict(job=job,signature=sig,status='complete',receptor_csv_sha256=receptor_csv_sha256,
             receptor_tracer='5HT2a_cimbi_hc29_beliveau',receptor_normalization='minmax')
    try:
        cfg,labels,k,rho=configuration(job,p,dataset)
        cfg.parameter_overrides['parameter_stimulus']=dict(stimtime=float(job['onset_ms']),stimdur=p['pulse_ms'],stimperiod=100000.,stimval=p['pulse_hz_per_ms']/1000,stimregion=[k],stimvariables=[0],stimshape='square')
        sim=simulator(cfg,job['seed'])
        # Inspect the actual discrete waveform before running the uninterrupted trial.
        grid=np.arange(0,job['total_ms'],p['dt_ms']);sim.stimulus.configure_time(grid)
        temporal=np.asarray(sim.stimulus.temporal_pattern).reshape(-1)
        active=np.flatnonzero(temporal!=0)
        if len(active)==0:raise ValueError('No delivered pulse')
        onset=float(grid[active[0]]);last=float(grid[active[-1]])
        if abs(onset-job['onset_ms'])>p['dt_ms']+1e-6:raise ValueError('Unexpected pulse timing')
        times,raw=sim.run(simulation_length=job['total_ms'])[0]
        if not np.isfinite(raw).all():raise FloatingPointError('Nonfinite trajectory')
        # Raw state after each integrator step; retain actual timestamps.
        times=times[9::10];state=raw[9::10,:3,:,0].astype(np.float32)
        keep=times>=job['warmup_ms']
        arrays=dict(time_ms=times[keep],state=state[keep],labels=labels,receptor=rho,
                    onset_ms=onset,source_index=k)
        if job['paired_control']:
            sham=deepcopy(cfg);sham.parameter_overrides.pop('parameter_stimulus',None)
            control=simulator(sham,job['seed'])
            ct,cr=control.run(simulation_length=job['total_ms'])[0]
            np.testing.assert_allclose(ct[9::10],times)
            arrays['control']=cr[9::10,:3,:,0][keep].astype(np.float32)
            before=arrays['time_ms']<onset
            np.testing.assert_allclose(arrays['state'][before],arrays['control'][before],atol=1e-7,rtol=0)
        keys=('b_e','tau_w_e','a_e','a_i','b_i','tau_w_i','T','weight_noise','tau_OU','P_e','P_i','g_K_e','g_K_i','g_Na_e','g_Na_i')
        row.update(source_label=str(labels[k]),source_index=k,actual_onset_ms=onset,last_nonzero_pulse_sample_ms=last,
                   delivered_nonzero_duration_ms=len(active)*p['dt_ms'],weights_sha256=digest(cfg.weights),tracts_sha256=digest(cfg.tract_lengths),
                   model_class=type(sim.model).__name__,model_parameters={key:np.asarray(getattr(sim.model,key)).tolist() for key in keys},
                   receptor_sha256=digest(rho),max_EI_hz=float(state[:,:2].max()*1000))
        tmp=folder/'recordings.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(folder/'recordings.npz')
    except Exception as exc:
        row.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    write_json(target,row);print(name,row['status'],flush=True);return row

def main():
    a=argparse.ArgumentParser();a.add_argument('--dataset-root',type=Path,default=ROOT/'data/doc_data/converted_structural_invnodevol_control_max')
    a.add_argument('--output',type=Path,required=True);a.add_argument('--workers',type=int,default=6)
    a.add_argument('--pilot',action='store_true');a.add_argument('--plan-only',action='store_true');args=a.parse_args()
    p=json.loads((HERE/'protocol.json').read_text());idx=json.loads((args.dataset_root/'index.json').read_text())
    n=idx['connectivity_normalization']
    if n['scheme']!=p['normalization_scheme'] or not np.isclose(n['divisor'],p['normalization_divisor']):raise ValueError('Wrong SC normalization')
    jobs=[];examples={'control':'c0005','emcs':'e0009','mcs':'m0060','uws':'u0033'}
    for c in p['b_pA']:
        subjects=[examples[c]] if args.pilot else idx['cohorts'][c]['subject_ids']
        for s in subjects:
            for trial in range(2 if args.pilot else p['trials']):
                seed=int.from_bytes(hashlib.sha256(f"{p['master_seed']}:{c}:{s}:{trial}".encode()).digest()[:4],'little')
                onset=6000 if args.pilot else int(np.random.default_rng(seed^123456).integers(3400,8001))
                for occ in p['occupancies']:
                    jobs.append(dict(cohort=c,subject=s,trial=trial,seed=seed,occupancy=occ,onset_ms=onset,total_ms=10000,warmup_ms=3000,paired_control=args.pilot))
                if args.pilot and trial==0:
                    jobs.append(dict(cohort=c,subject=s,trial=trial,seed=seed,occupancy=0,onset_ms=11000,total_ms=15000,warmup_ms=8000,paired_control=True))
    args.output.mkdir(parents=True,exist_ok=True)
    provenance={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),HERE/'protocol.json',ROOT/'src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut_gK_gNa.py',ROOT/'src/tvbtoolkit/whole_brain/simulation.py',ROOT/'data/receptors/hansen_receptors_aal90.csv']}
    manifest=dict(protocol=p,pilot=args.pilot,jobs=jobs,dataset_index_sha256=hashlib.sha256((args.dataset_root/'index.json').read_bytes()).hexdigest(),source_hashes=provenance)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Changed manifest/code: use a new output directory')
    if not path.exists():write_json(path,manifest)
    print(f'{len(jobs)} trials; pilot={args.pilot}',flush=True)
    if args.plan_only:return
    lock=args.output/'RUNNING'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(run_one,[(j,p,str(args.dataset_root),str(args.output)) for j in jobs]))
        failures=sum(r['status']!='complete' for r in results)
        write_json(args.output/'completion.json',dict(total=len(jobs),failed=failures))
        if failures:raise RuntimeError(f'{failures} failed trials; inspect metadata')
    finally:lock.unlink()

if __name__=='__main__':main()
