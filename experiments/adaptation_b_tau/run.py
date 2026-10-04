"""Resumable, isolated first-order b x tau campaign using native TVB functions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src'),str(HERE)]
os.environ.setdefault('MPLCONFIGDIR','/tmp/adaptation-b-tau-mpl')
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import pandas as pd
from first_order_adex_audit import configure,DATA
from first_order_amplitude_trials import install_noise
from tvbtoolkit.whole_brain.simulation import configure_adex_simulator
from tvbtoolkit.whole_brain.impulse_response import fork_checkpoint
from metrics import regional_baseline,off_metrics,validity,pci_metrics

DEFAULT_OUT=REPO/'results/adaptation_b_tau_v1'

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False));temp.replace(path)

def identifier(job):
    return f"{job['cohort']}_{job['subject']}_b{job['b']:03g}_tau{job['tau']:04g}_r{job['replicate']:03d}"

def unclamp(sim):
    assert tuple(sim.model.state_variables)==('E','I','W_e','W_i','noise')
    for key in ['state_variable_boundaries','_integration_state_variable_boundaries','clamped_state_variable_values','_clamped_integration_state_variable_values']:
        setattr(sim.integrator,key,None)

def get_segment(sim,duration,keep=True):
    """Chunk only unstimulated simulation; a pulse branch is run uninterrupted."""
    pieces=[];peak=0.
    for start in np.arange(0,duration,500):
        _,a=sim.run(simulation_length=min(500,duration-start))[0];a=a[:,:,:,0]
        if not np.isfinite(a).all():raise FloatingPointError('Nonfinite integration state')
        peak=max(peak,float(abs(a[:,:2]).max()*1000))
        if peak>1e4:raise FloatingPointError('Catastrophic finite firing rate above 10000 Hz')
        if keep:pieces.append(a[9::10,:3].astype(np.float32))
    return (np.concatenate(pieces) if pieces else None),peak

def make_epoch(pre,post):
    # Checkpoint itself is t=0. First saved continuation is t=1 ms.
    return np.concatenate([pre[-401:,0],post[:299,0]],axis=0).T.astype(float)*1000

def measure(job,pre,control,pulse,p,folder):
    dt=p['saved_dt_ms'];common=regional_baseline(pre,dt,p)
    base_valid=validity(pre,dt,p);rows=[];regional=[];pci_arrays={}
    mean_fields=['E_mean_hz','I_mean_hz','E_variance_hz2','I_variance_hz2','W_mean_pA','W_peak_pA',
        'up_fraction','down_fraction','unclassified_fraction','up_mean_ms','up_median_ms','down_mean_ms','down_median_ms',
        'dominant_frequency_hz','slow_power_fraction','bimodal','slow_switching','complete_cycles']
    baseline_df=pd.DataFrame(common)
    summary={k:float(baseline_df[k].mean()) for k in mean_fields}
    summary['baseline_stationarity_E_mean_shift_hz']=float(abs(pre[:len(pre)//2,0].mean(axis=0)-pre[len(pre)//2:,0].mean(axis=0)).max()*1000)
    for condition,data in [('spontaneous',control),('TMS',pulse)]:
        check=validity(data,dt,p);off=off_metrics(pre[:,0]*1000,data[:,0]*1000,dt,p)
        source=off[p['source_index']]
        epoch=make_epoch(pre,data)
        pm,arrays=pci_metrics(epoch[None],p)
        pci_arrays.update({condition+'_'+k:v for k,v in arrays.items()})
        row=dict(subject=job['subject'],diagnostic_group=job['cohort'],b_pA=job['b'],tau_w_ms=job['tau'],
            replicate=job['replicate'],seed=job['seed'],stimulation_condition=condition,
            **summary,**pm,valid=bool(base_valid['valid'] and check['valid']),
            invalid_reasons=';'.join(base_valid['invalid_reasons']+check['invalid_reasons']),PFP=bool(base_valid['PFP'] or check['PFP']),
            condition_E_mean_hz=float(data[:,0].mean()*1000),condition_I_mean_hz=float(data[:,1].mean()*1000),
            condition_E_variance_hz2=float(data[:,0].var(axis=0).mean()*1e6),condition_I_variance_hz2=float(data[:,1].var(axis=0).mean()*1e6),
            condition_max_EI_hz=check['max_EI_hz'],condition_W_peak_pA=float(data[:,2].max()),condition_W_mean_pA=float(data[:,2].mean()),
            off_region_fraction=float(np.mean([r['off_present'] for r in off])),
            off_mean_duration_ms=float(np.mean([r['off_duration_ms'] for r in off])),
            off_mean_depth_fraction=float(np.mean([r['off_depth_fraction'] for r in off])),
            **{'source_'+k:v for k,v in source.items() if k!='region_index'},
            slow_switching_proxy=bool(summary['slow_switching']>=.2),
            mathematical_bistability_established=False,metrics_status='complete')
        rows.append(row)
        for bl,ev in zip(common,off):regional.append(dict(subject=job['subject'],diagnostic_group=job['cohort'],b_pA=job['b'],tau_w_ms=job['tau'],replicate=job['replicate'],seed=job['seed'],stimulation_condition=condition,**{**bl,**ev}))
    for key in ['source_off_duration_ms','source_off_depth_fraction','off_region_fraction']:
        rows[1][key+'_excess_over_control']=rows[1][key]-rows[0][key]
        rows[0][key+'_excess_over_control']=0.
    rows[1]['operational_transition']=bool(rows[1]['valid'] and (rows[1]['slow_switching_proxy'] or (rows[1]['source_off_present'] and rows[1]['source_off_duration_ms_excess_over_control']>=20)))
    rows[0]['operational_transition']=bool(rows[0]['valid'] and rows[0]['slow_switching_proxy'])
    pd.DataFrame(regional).to_csv(folder/'regional_metrics.csv',index=False)
    np.savez_compressed(folder/'pci_masks.npz',**pci_arrays)
    return rows

def run_job(payload):
    job,p,out=payload;folder=Path(out)/'cells'/identifier(job);folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(dict(job=job,protocol=p),sort_keys=True).encode()).hexdigest()
    target=folder/'result.json'
    if target.exists():
        cached=json.loads(target.read_text())
        if cached['signature']!=signature:raise ValueError('Existing job signature differs: '+str(folder))
        if cached['status']!='execution_error':return cached
        if (folder/'recordings.npz').exists():
            with np.load(folder/'recordings.npz') as saved:
                rows=measure(job,saved['baseline'],saved['control'],saved['TMS'],p,folder)
            atomic_json(folder/'initial_execution_error.json',cached)
            cached.update(status='complete',rows=rows)
            cached['metadata']['analysis_recovered_from_saved_recordings']=True
            atomic_json(target,cached)
            return cached
    started=time.monotonic();arrays={};metadata={};status='complete'
    try:
        _,cfg,labels=configure(isolated=False,cohort=job['cohort'],subject=job['subject'],b=job['b'],coupling=p['G'],seed=job['seed'])
        cfg.parameter_overrides['parameter_model']['tau_w_e']=float(job['tau'])
        sim=configure_adex_simulator(cfg,job['seed']);unclamp(sim)
        assert float(np.asarray(sim.model.tau_w_e).item())==job['tau']
        assert float(np.asarray(sim.model.b_e).item())==job['b']
        assert float(np.asarray(sim.model.b_i).item())==0.
        rng=install_noise(sim,job['seed'])
        keys=['b_e','tau_w_e','b_i','tau_w_i','a_e','a_i','T','external_input_ex_ex','external_input_in_ex','weight_noise','tau_OU','P_e','P_i']
        metadata=dict(model_class=type(sim.model).__name__,integrator=type(sim.integrator).__name__,
            dt_ms=cfg.dt_ms,source_region=str(labels[p['source_index']]),source_index=p['source_index'],
            model_parameters={k:getattr(sim.model,k) for k in keys},
            weights_sha256=hashlib.sha256(np.ascontiguousarray(cfg.weights).tobytes()).hexdigest(),
            tracts_sha256=hashlib.sha256(np.ascontiguousarray(cfg.tract_lengths).tobytes()).hexdigest(),
            zero_edge_fraction=float((cfg.weights==0).mean()),incoming_strength=cfg.weights.sum(axis=1).tolist())
        _,metadata['equilibration_peak_EI_hz']=get_segment(sim,p['equilibration_ms'],keep=False)
        pre,metadata['baseline_peak_EI_hz']=get_segment(sim,p['baseline_ms'])
        future=deepcopy(rng.bit_generator.state);branches={}
        for condition,amp in [('spontaneous',0.),('TMS',p['pulse_hz_per_ms'])]:
            overrides=deepcopy(cfg.parameter_overrides)
            if amp:overrides['parameter_stimulus']=dict(stimtime=0.,stimdur=p['pulse_ms'],stimperiod=100000.,stimval=amp/1000.,stimregion=[p['source_index']],stimvariables=[0],stimshape='square')
            branch=fork_checkpoint(sim,replace(cfg,parameter_overrides=overrides),job['seed']);unclamp(branch)
            np.testing.assert_array_equal(branch.current_state,sim.current_state)
            np.testing.assert_array_equal(branch.history.buffer,sim.history.buffer)
            install_noise(branch,job['seed'],future)
            tt,raw=branch.run(simulation_length=p['response_ms'])[0];raw=raw[:,:,:,0]
            if not np.isfinite(raw).all():raise FloatingPointError('Nonfinite '+condition+' response')
            branches[condition]=raw[9::10,:3].astype(np.float32)
        arrays=dict(baseline=pre,control=branches['spontaneous'],TMS=branches['TMS'],labels=labels,
                    baseline_time_ms=np.arange(-len(pre)+1,1),response_time_ms=np.arange(1,len(branches['TMS'])+1))
        np.savez_compressed(folder/'recordings.npz',**arrays)
        rows=measure(job,pre,branches['spontaneous'],branches['TMS'],p,folder)
    except Exception as exc:
        status='numerical_failure' if isinstance(exc,FloatingPointError) else 'execution_error'
        metadata.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        rows=[dict(subject=job['subject'],diagnostic_group=job['cohort'],b_pA=job['b'],tau_w_ms=job['tau'],replicate=job['replicate'],seed=job['seed'],stimulation_condition=c,valid=False,invalid_reasons=status,metrics_status=status) for c in ['spontaneous','TMS']]
    result=clean(dict(signature=signature,job=job,status=status,elapsed_seconds=time.monotonic()-started,metadata=metadata,rows=rows))
    atomic_json(target,result)
    print(identifier(job),status,f"{result['elapsed_seconds']:.1f}s",flush=True)
    return result

def inventory(p):
    index=json.loads((DATA/'index.json').read_text());jobs=[]
    for c in p['cohorts']:
        for s in index['cohorts'][c]['subject_ids']:
            for rep in range(p['replicates']):
                seed=int.from_bytes(hashlib.sha256(f"{p['seed']}:{c}:{s}:{rep}".encode()).digest()[:4],'little')
                for b in p['b_pA']:
                    for tau in p['tau_w_ms']:jobs.append(dict(cohort=c,subject=s,b=b,tau=tau,replicate=rep,seed=seed))
    # Complete exemplar grids first, then interleave cohort/subjects at each pair.
    priority={'c0005':0,'m0060':1,'u0033':2,'e0009':3}
    jobs.sort(key=lambda j:(0 if j['subject'] in priority else 1,priority.get(j['subject'],99),j['b'],j['tau'],j['replicate'],j['cohort'],j['subject']))
    return jobs,index

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['plan','run','launch','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--subjects',nargs='*');parser.add_argument('--b',nargs='*',type=float);parser.add_argument('--tau',nargs='*',type=float)
    parser.add_argument('--limit',type=int);parser.add_argument('--replicates',type=int);args=parser.parse_args()
    p=json.loads((HERE/'protocol.json').read_text())
    if args.replicates:p['replicates']=args.replicates
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    jobs,index=inventory(p)
    manifest=dict(protocol=p,jobs=jobs,total_pairs=len(jobs),total_recordings=len(jobs)*2,
        cohort_counts={c:len(index['cohorts'][c]['subject_ids']) for c in p['cohorts']},
        SC_normalization=index['connectivity_normalization'],
        source_hashes={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [HERE/'metrics.py',HERE/'run.py',HERE/'protocol.json',REPO/'src/tvbtoolkit/whole_brain/legacy_engine/src/Zerlaut.py']})
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old['protocol']!=p:raise ValueError('Protocol differs; choose a new output directory')
    else:atomic_json(out/'manifest.json',manifest)
    if args.command=='plan':print(json.dumps({k:v for k,v in manifest.items() if k not in ['jobs','protocol']},indent=2));return
    if args.command=='report':
        from report import report
        report(out);return
    if args.command=='launch':
        command=[sys.executable,str(HERE/'run.py'),'run','--output',str(out),'--workers',str(args.workers)]
        with (out/'campaign.log').open('ab') as stream:
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=REPO,start_new_session=True)
        atomic_json(out/'process.json',dict(pid=proc.pid,command=command,started_unix=time.time()))
        print(json.dumps(dict(pid=proc.pid,output=str(out),log=str(out/'campaign.log'))));return
    if args.subjects:jobs=[j for j in jobs if j['subject'] in args.subjects]
    if args.b:jobs=[j for j in jobs if j['b'] in args.b]
    if args.tau:jobs=[j for j in jobs if j['tau'] in args.tau]
    if args.limit:jobs=jobs[:args.limit]
    lock=out/'running.lock'
    if lock.exists():
        pid=int(lock.read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError(f'Campaign already running with PID {pid}')
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    started=time.time();done=0;errors=0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_job,(j,p,str(out))) for j in jobs]
            for future in as_completed(futures):
                r=future.result();done+=1;errors+=int(r['status']=='execution_error')
                atomic_json(out/'progress.json',dict(completed_invocation=done,planned_invocation=len(jobs),campaign_total=manifest['total_pairs'],execution_errors=errors,elapsed_seconds=time.time()-started,pid=os.getpid(),running=done<len(jobs)))
                if done%25==0:
                    from report import collect
                    collect(out)
        from report import report
        report(out)
    finally:
        if lock.exists() and int(lock.read_text())==os.getpid():lock.unlink()

if __name__=='__main__':main()
