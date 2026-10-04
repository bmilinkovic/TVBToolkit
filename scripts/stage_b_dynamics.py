"""Sequential diagnostics 2/3: instrument native dynamics; no rate clipping."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import numpy as np
from run_adex_impulse_response import ROOT, build_subject_config
from stage_b_transfer_domain import evaluate
from tvbtoolkit.whole_brain.simulation import configure_adex_simulator

BASE=ROOT/'results/adex_staged_diagnostics'
DATA=ROOT/'data/doc_data/converted_structural_invnodevol_control_max'
FIELDS=['E_khz','I_khz','Cee_khz2','Cei_khz2','Cii_khz2','We_pA','Wi_pA','noise',
        'long_range_khz','external_E_khz','external_I_khz','FE_khz','FI_khz',
        'JE_E','JE_I','JI_E','JI_I','HE_EE_per_khz','HE_EI_per_khz','HE_II_per_khz',
        'HI_EE_per_khz','HI_EI_per_khz','HI_II_per_khz','correctionE_khz','correctionI_khz','min_cov_eigenvalue_khz2']

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def run(job, *, b=5, derivative_scale=1, output_override=None, model_overrides=None):
    stage,cohort,subject,gain,drive,stochastic,duration,seed=job[:8]
    matched_noise=bool(job[8]) if len(job)>8 else False
    output=output_override or BASE/(f'diagnostic{stage}'+('_matched' if matched_noise else ''));output.mkdir(parents=True,exist_ok=True)
    name=f'{cohort}_{subject}_G{gain:g}_drive{drive:g}_noise{int(stochastic)}_seed{seed}'
    if output_override is not None:name+=f'_b{b:g}_scale{derivative_scale:g}'
    if model_overrides and output_override is None:
        raise ValueError('Experimental model overrides require a separate output directory')
    if (output/f'{name}.json').exists():
        cached=json.loads((output/f'{name}.json').read_text())
        if cached.get('model_parameter_overrides',{}) != (model_overrides or {}):
            raise ValueError('Cached run used different model overrides; choose another output directory')
        return cached
    if not stochastic:raise ValueError('New diagnostics require stochastic externally driven simulations')
    cfg,labels,_=build_subject_config(cohort,subject,b,DATA,gain,monitor_variables=tuple(range(8)))
    if stage==2:
        cfg=replace(cfg,weights=np.zeros((1,1)),tract_lengths=np.zeros((1,1)))
        labels=np.array(['isolated_EI_node'])
    cfg=replace(cfg,stochastic_integrator=stochastic)
    cfg.parameter_overrides['parameter_model'].update(external_input_ex_ex=drive/1000,external_input_in_ex=drive/1000)
    cfg.parameter_overrides['parameter_model'].update(model_overrides or {})
    sim=configure_adex_simulator(cfg,seed)
    sim.model.derivative_scale=np.array([float(derivative_scale)])
    if stochastic and matched_noise:
        # Delay-history initialization consumes different numbers of RNG draws.
        # Match subsequent innovations, without altering the initial history.
        sim.integrator.noise.random_stream.seed(seed)
    # Observation experiment explicitly disables the existing E/I zero bounds.
    sim.integrator.state_variable_boundaries=None
    sim.integrator._integration_state_variable_boundaries=None
    sim.integrator.clamped_state_variable_values=None
    sim.integrator._clamped_integration_state_variable_values=None
    m=sim.model;original=m.dfun;n=len(labels);calls=0;records=[];times=[];events={}
    maxrates=np.zeros(n);maxinput=np.zeros(n);highcounts=np.zeros(n,int);maxhighcounts=np.zeros(n,int)
    sample_every=10 # accepted step every 1 ms at dt=.1 ms
    def observed(state,coupling,local_coupling=0.):
        nonlocal calls
        call=calls;calls+=1;time=(call//2+call%2)*cfg.dt_ms;accepted=call%2==0
        s=state[:,:,0];lr=coupling[0,:,0]
        with np.errstate(all='ignore'):
            d=original(state,coupling,local_coupling)
            noise=m.weight_noise*m._mixed_noise(s[7])
            de=np.maximum(lr+m.external_input_ex_ex+noise,0).reshape(n)
            di=np.maximum(m.inh_factor*lr+m.external_input_in_ex+noise,0).reshape(n)
            fe=np.asarray(m.TF_excitatory(s[0],s[1],de,m.external_input_ex_in,s[5])).reshape(n)
            fi=np.asarray(m.TF_inhibitory(s[0],s[1],di,m.external_input_in_in,s[6])).reshape(n)
            ft=np.array([fe,fi])*float(np.asarray(m.T).item())
            correction=float(np.asarray(m.T).item())*d[:2,:,0]-(np.array([fe,fi])-s[:2])
            # Analytic smaller eigenvalue avoids eigensolver failure on NaN.
            eig=.5*(s[2]+s[4]-np.sqrt((s[2]-s[4])**2+4*s[3]**2))
        masks=dict(negative_rate=(s[:2]<-1e-9).any(axis=0),invalid_FT=(ft>1).any(axis=0),
            negative_covariance=eig < -1e-12,
            large_correction=(np.abs(correction)>np.maximum(np.abs([fe,fi]),.001)).any(axis=0),
            nonfinite=(~np.isfinite(s)).any(axis=0)|(~np.isfinite(d[:,:,0])).any(axis=0),
            catastrophic_magnitude=(np.abs(s[:2])>1000).any(axis=0)|(np.abs(s[2:5])>1e6).any(axis=0))
        newevent=False
        for key,mask in masks.items():
            if key not in events and mask.any():
                idx=int(np.flatnonzero(mask)[0]);newevent=True
                events[key]=dict(time_ms=time,stage='accepted' if accepted else 'predictor',region=str(labels[idx]),index=idx,
                    rates_hz=(s[:2,idx]*1000).tolist(),long_range_hz=float(lr[idx]*1000),
                    transfer_hz=[float(fe[idx]*1000),float(fi[idx]*1000)],
                    correction_hz=(correction[:,idx]*1000).tolist(),min_cov_eig_hz2=float(eig[idx]*1e6))
        if accepted:
            maxrates[:]=np.maximum(maxrates,np.nan_to_num(np.max(s[:2],axis=0)*1000,nan=0,posinf=1e300))
            maxinput[:]=np.maximum(maxinput,np.nan_to_num(lr*1000,nan=0,posinf=1e300))
            highcounts[:]=np.where((s[:2]>.1).any(axis=0),highcounts+1,0)
            maxhighcounts[:]=np.maximum(maxhighcounts,highcounts)
            if 'sustained_high' not in events and (highcounts>=1000).any():
                idx=int(np.flatnonzero(highcounts>=1000)[0]);events['sustained_high']=dict(time_ms=time,index=idx,region=str(labels[idx]))
        stop=masks['nonfinite'].any() or masks['catastrophic_magnitude'].any()
        if (accepted and (call//2)%sample_every==0) or newevent or stop:
            with np.errstate(all='ignore'):
                # inh_factor=1 and equal drive are retained in these protocols.
                tf=evaluate(m,s[0],s[1],de,s[5],s[6])
            record=np.concatenate([s,lr[None],de[None],di[None],tf[:,0],tf[:,1:3].reshape(4,n),
                                  tf[:,3:].reshape(6,n),correction,eig[None]],axis=0)
            records.append(record.copy());times.append(time)
        if stop:raise FloatingPointError('Observed nonfinite derivative/state or catastrophic magnitude; no clipping')
        return d
    m.dfun=observed
    reason=None
    try:
        with np.errstate(all='ignore'):sim.run(simulation_length=duration)
    except FloatingPointError as exc:reason=str(exc)
    arr=np.asarray(records);ts=np.asarray(times)
    late=arr[ts>duration/2]
    finite=reason is None
    validity=not any(k in events for k in ['negative_rate','invalid_FT','negative_covariance','nonfinite','catastrophic_magnitude'])
    high=bool((maxhighcounts>=1000).any())
    settled=False
    if len(late)>20:
        means1=late[:len(late)//2,:2].mean(axis=0);means2=late[len(late)//2:,:2].mean(axis=0)
        settled=bool(np.max(np.abs(means2-means1)/(np.maximum(np.abs(means2),.001)))<.2)
    category=4 if not finite else 2 if high else 1 if validity and settled else 3
    row=dict(stage=stage,cohort=cohort,subject=subject,G=gain,external_drive_hz=drive,stochastic=stochastic,seed=seed,
        duration_requested_ms=duration,duration_observed_ms=float(ts[-1]),b_pA=b,T_ms=20,dt_ms=.1,
        derivative_scale=derivative_scale,
        model_parameter_overrides=model_overrides or {},
        category=category,classification={1:'low-rate stable candidate',2:'finite sustained high-rate',3:'oscillatory/other transition or invalid finite state',4:'numerical failure'}[category],
        mathematically_valid=validity,finite_completed=finite,stop_reason=reason,events=events,
        maximum_rate_hz=float(maxrates.max()),maximum_long_range_hz=float(maxinput.max()),
        region_maximum_rate_hz=maxrates.tolist(),region_maximum_long_range_hz=maxinput.tolist(),
        sustained_high_duration_ms=(maxhighcounts*.1).tolist(),
        region_incoming_strength=cfg.weights.sum(axis=1).tolist(),region_G_times_strength=(gain*cfg.weights.sum(axis=1)).tolist(),
        late_mean_E_hz=float(late[:,0].mean()*1000) if len(late) else None,
        original_fit_domain_status='unknown',rate_clipping=False,noise_reseeded_after_history=matched_noise)
    row=clean(row)
    np.savez_compressed(output/f'{name}.npz',time_ms=ts,records=arr,fields=np.array(FIELDS),labels=labels,
                        weights=cfg.weights,tract_lengths=cfg.tract_lengths,derivative_scale=derivative_scale)
    row['file']=name+'.npz'
    (output/f'{name}.json').write_text(json.dumps(row,indent=2,allow_nan=False))
    print(json.dumps(dict(stage=stage,cohort=cohort,subject=subject,G=gain,drive=drive,noise=stochastic,category=category,last_ms=float(ts[-1]))),flush=True)
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',type=int,choices=[2,3]);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--matched-noise',action='store_true')
    parser.add_argument('--duration-ms',type=float,default=2000);args=parser.parse_args()
    assert (BASE/'diagnostic1/completion.json').exists()
    if args.stage==3:assert (BASE/'diagnostic2/completion.json').exists()
    output=BASE/(f'diagnostic{args.stage}'+('_matched' if args.matched_noise else ''));output.mkdir(parents=True,exist_ok=True)
    if args.stage==2:
        drives=[0,.05,.1,.2,.315,.5,.75,1,1.5,2,3,5,10,30,100,300,1000,10000]
        jobs=[(2,'control','c0015',0.,d,True,args.duration_ms,20260928) for d in drives]
    else:
        metadata=json.loads((DATA/'normalization_metadata.json').read_text())['subjects']
        reps=[]
        for cohort in sorted({s['cohort'] for s in metadata}):
            group=[s for s in metadata if s['cohort']==cohort];median=np.median([s['zero_entries'] for s in group])
            chosen=min(group,key=lambda s:(abs(s['zero_entries']-median),s['subject_id']))
            reps.append((cohort,chosen['subject_id']))
        gains=[0,.001,.003,.01,.03,.05,.075,.1,.15,.2,.25,.3,.4,.5,.75,1.,1.5,2.,3.,5.,10.,20.,30.,40.,60.,100.]
        jobs=[(3,c,s,g,.315,True,args.duration_ms,20260928) for c,s in reps for g in gains]
        (output/'representatives.json').write_text(json.dumps(dict(selection='closest to cohort median zero-entry count, lexicographic ties',subjects=reps),indent=2))
    protocol=dict(stage=args.stage,duration_ms=args.duration_ms,seed=20260928,b_pA=5,T_ms=20,dt_ms=.1,
        clipping='disabled integrator E/I bounds only for these diagnostics; native external input nonnegativity retained',
        classification='1 valid settled low-rate candidate; 2 finite >100 Hz for >=100 ms; 3 other including invalid finite; 4 nonfinite or catastrophic >1e6 Hz / covariance >1e12 Hz²',
        original_training_domain='unresolved; no fabricated extrapolation labels',equations_modified=False,tf_refitted=False,
        scalar_c=json.loads((BASE/'stage_a/completion.json').read_text())['c'])
    protocol['grid'] = dict(external_drive_hz=drives) if args.stage==2 else dict(initial_G=gains,refinement='three interior points where category/validity changes')
    protocol['noise_reseeded_after_history']=args.matched_noise
    if args.matched_noise:jobs=[j+(True,) for j in jobs]
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):rows.append(future.result())
    if args.stage==3:
        refinements=[]
        for cohort,subject in reps:
            group=sorted([r for r in rows if r['cohort']==cohort],key=lambda r:r['G'])
            for a,b in zip(group,group[1:]):
                if (a['category'],a['mathematically_valid']) != (b['category'],b['mathematically_valid']):
                    refinements += [(3,cohort,subject,float(g),.315,True,args.duration_ms,20260928,args.matched_noise) for g in np.linspace(a['G'],b['G'],5)[1:-1]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run,j) for j in refinements]):rows.append(future.result())
    (output/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (output/'completion.json').write_text(json.dumps(dict(runs_completed=len(rows),original_fit_domain_verified=False),indent=2))

if __name__=='__main__':main()
