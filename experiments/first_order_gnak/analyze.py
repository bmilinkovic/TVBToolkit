"""Read-only simulation inputs; separate PCI analysis and pilot diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'experiments/adaptation_b_tau'),str(HERE)]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
from simulate import write_json

def epoch(times,state,onset):
    target=onset+np.arange(-400.,300.)
    if target[0]<times[0]-1e-6 or target[-1]>times[-1]+1e-6:raise ValueError('Incomplete epoch')
    return np.array([np.interp(target,times,state[:,0,k])*1000 for k in range(90)])

def analyze_group(payload):
    key,files,protocol,out=payload
    from metrics import pci_metrics
    from first_order_doc16_pci import sampling,lz_score
    from tvbtoolkit.complexity.pci_casali import binarise_signals
    data=[];seed_ids=[]
    for f in files:
        meta=json.loads(f.read_text());seed_ids.append(meta['job']['seed'])
        with np.load(f.with_name('recordings.npz')) as z:
            data.append(epoch(z['time_ms'],z['state'],float(z['onset_ms'])))
    if len(set(seed_ids))!=len(seed_ids):raise ValueError('Duplicate trial seeds')
    if len(data)!=protocol['trials']:raise ValueError(f'Expected {protocol["trials"]} trials for {key}, got {len(data)}')
    p=dict(protocol,source_index=meta['source_index'])
    scores,arrays=pci_metrics(np.array(data),p)
    x,t=sampling(np.array(data),p['pci']['sampling_ms']);pre=(t>=-400)&(t<-100);post=(t>=0)&(t<300)
    centered=x-x[:,:,pre].mean(axis=2,keepdims=True)
    arrays['evoked_hz']=centered.mean(axis=0)[:,post]
    arrays['activation_order']=np.argsort(-np.max(abs(arrays['evoked_hz']),axis=1))
    arrays['significance_order']=np.argsort(-arrays['all90_mask'].sum(axis=1))
    # Historical threshold algorithm retained, but slicing is explicitly in bins.
    for scope,ids in [('all90',np.arange(90)),('remote89',np.delete(np.arange(90),p['source_index']))]:
        xx=x[:,ids];stack=np.concatenate([xx[:,:,pre],xx[:,:,post]],axis=2)
        old=np.random.get_state()
        try:
            np.random.seed(p['pci']['seed']);mask=binarise_signals(stack,int(pre.sum()),nshuffles=10,percentile=100.)[:,:,int(pre.sum()):]
        finally:np.random.set_state(old)
        vals=np.array([lz_score(m)['PCI_LZ'] for m in mask])
        scores[scope+'_legacy_trial_mean']=float(vals.mean());scores[scope+'_legacy_trial_sd']=float(vals.std(ddof=1))
        arrays[scope+'_legacy_trial_values']=vals;arrays[scope+'_legacy_masks']=mask
    tag='_'.join(map(str,key));dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    scores.update(cohort=key[0],subject=key[1],occupancy=key[2],historical_interpretation='TVBSim threshold/per-trial averaging with corrected sample-index windows')
    np.savez_compressed(dest/(tag+'.npz'),**arrays);write_json(dest/(tag+'.json'),scores)
    return scores

def pilot_report(files,manifest,out):
    from metrics import regional_baseline,off_metrics,validity
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    p=json.loads((ROOT/'experiments/adaptation_b_tau/protocol.json').read_text())
    rows=[];fig,axes=plt.subplots(4,4,figsize=(17,12),layout='constrained')
    statefig,stateaxes=plt.subplots(4,3,figsize=(15,11),layout='constrained')
    cohorts=list(manifest['protocol']['b_pA']);occupied=manifest['protocol']['occupancies']
    for f in files:
        m=json.loads(f.read_text());j=m['job'];p['source_index']=m['source_index'];p['pulse_ms']=manifest['protocol']['pulse_ms']
        with np.load(f.with_name('recordings.npz')) as z:
            tt=z['time_ms']-float(z['onset_ms']);a=z['state'];control=z['control'];k=m['source_index']
            pre=a[tt<0];post=a[tt>0];bl=regional_baseline(pre,1,p);off=off_metrics(pre[:,0]*1000,post[:,0]*1000,1,p)[k]
            checks=validity(a,1,p);e=a[:,0]*1000;delta=(a[:,0]-control[:,0])*1000
            # Matched-noise distance is a diagnostic only, never the PCI input.
            rms=np.sqrt(np.mean(delta**2,axis=1));tail=(tt>=tt[-1]-500)
            peak=float(rms[(tt>=0)&(tt<=300)].max());cut=max(.1,.05*peak)
            from metrics import runs
            intervals=[(aa,bb) for aa,bb in runs((tt>=300)&(rms<cut)) if bb-aa>=200]
            recovery=float(tt[intervals[0][0]]) if intervals else None
            baseline_sd=pre[:,0].std(axis=0)*1000
            late_mean=e[tail].mean(axis=0);early_mean=pre[:,0].mean(axis=0)*1000
            frac=float(np.mean(abs(late_mean-early_mean)<=np.maximum(.1,baseline_sd)))
            row=dict(**j,**off,valid=checks['valid'],PFP=checks['PFP'],max_EI_hz=checks['max_EI_hz'],
                     E_mean_hz=float(pre[:,0].mean()*1000),I_mean_hz=float(pre[:,1].mean()*1000),
                     slow_switching_fraction=float(np.mean([b['slow_switching'] for b in bl])),
                     down_fraction=float(np.mean([b['down_fraction'] for b in bl])),
                     recovery_200ms_paired_RMS_threshold_hz=cut,recovery_first_200ms_below_threshold_ms=recovery,
                     source_up_median_ms=bl[k]['up_median_ms'],source_down_median_ms=bl[k]['down_median_ms'],
                     tail_paired_RMS_hz=float(np.sqrt(np.mean(delta[tail]**2))),
                     tail_regions_within_baseline_1sd_or_01Hz=frac,
                     baseline_max_half_mean_shift_hz=float(np.max(abs(pre[:len(pre)//2,0].mean(axis=0)-pre[len(pre)//2:,0].mean(axis=0)))*1000))
            rows.append(row)
            pd.DataFrame(bl).to_csv(out/(f.parent.name+'_regional.csv'),index=False)
            if j['trial']==0 and j['warmup_ms']==3000:
                ax=axes[cohorts.index(j['cohort']),occupied.index(j['occupancy'])]
                for node in range(90):ax.plot(tt,e[:,node],color=plt.cm.turbo(node/89),lw=.35,alpha=.45)
                ax.plot(tt,e[:,k],color='black',lw=.8);ax.axvspan(0,p['pulse_ms'],color='red',alpha=.2)
                ax.set(xlim=(-500,1500),title=f"{j['cohort']} b={manifest['protocol']['b_pA'][j['cohort']]} / occupancy {j['occupancy']}",xlabel='Time from onset (ms)',ylabel='E firing rate (Hz)')
                if j['occupancy']==0:
                    for variable,title,factor in [(0,'E (Hz)',1000),(1,'I (Hz)',1000),(2,'W (pA)',1)]:
                        ax2=stateaxes[cohorts.index(j['cohort']),variable]
                        ax2.plot(tt,a[:,variable]*factor,lw=.3,alpha=.3)
                        ax2.plot(tt,a[:,variable,k]*factor,color='black',lw=.8)
                        ax2.axvline(0,color='red',lw=.7);ax2.set(title=j['cohort']+' '+title,xlabel='Time from onset (ms)')
    fig.suptitle('First-order gNa/gK: all 90 regional trajectories; black = stimulated source; one seed shown')
    fig.savefig(out/'regional_trajectories.png',dpi=180);fig.savefig(out/'regional_trajectories.svg');plt.close(fig)
    statefig.suptitle('Zero occupancy: complete retained baseline and response; all regions, stimulated source in black')
    statefig.savefig(out/'baseline_E_I_W.png',dpi=180);statefig.savefig(out/'baseline_E_I_W.svg');plt.close(statefig)
    d=pd.DataFrame(rows);d.to_csv(out/'pilot_metrics.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for c,g in d[d.warmup_ms==3000].groupby('cohort'):
        for ax,key in zip(axes,['off_duration_ms','tail_paired_RMS_hz','slow_switching_fraction']):
            agg=g.groupby('occupancy')[key].agg(['mean','min','max']);ax.plot(agg.index,agg['mean'],'o-',label=c);ax.fill_between(agg.index,agg['min'],agg['max'],alpha=.15);ax.set(xlabel='Occupancy',ylabel=key.replace('_',' '))
    axes[0].legend();fig.savefig(out/'recovery_and_off.png',dpi=180);fig.savefig(out/'recovery_and_off.svg');plt.close(fig)
    print(d.groupby(['cohort','occupancy'])[['E_mean_hz','off_duration_ms','tail_paired_RMS_hz','PFP']].mean().to_string())

def main():
    a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--pilot-report',action='store_true');a.add_argument('--workers',type=int,default=6);args=a.parse_args()
    if args.input.resolve()==args.output.resolve() or args.input.resolve() in args.output.resolve().parents:raise ValueError('Analysis output must be separate from simulation tree')
    manifest=json.loads((args.input/'manifest.json').read_text());files=sorted((args.input/'trials').glob('*/metadata.json'))
    if len(files)!=len(manifest['jobs']):raise ValueError('Simulation campaign incomplete')
    groups={}
    for f in files:
        m=json.loads(f.read_text())
        if m['status']!='complete':raise ValueError(f'Failed simulation: {f}')
        j=m['job'];groups.setdefault((j['cohort'],j['subject'],j['occupancy']),[]).append(f)
    if args.pilot_report:
        if not manifest['pilot']:raise ValueError('Not a paired pilot')
        pilot_report(files,manifest,args.output);return
    if manifest['pilot']:raise ValueError('Two-trial pilot is not production PCI')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'analysis_provenance.json',dict(simulation_manifest=manifest,analysis_file=str(Path(__file__))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(analyze_group,[(k,v,manifest['protocol'],str(args.output)) for k,v in groups.items()]))
    table=pd.DataFrame(rows);table.to_csv(args.output/'pci_subject_occupancy.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key in zip(axes,['all90_PCI_LZ','all90_PCI_ST','all90_legacy_trial_mean']):
        for c,g in table.groupby('cohort'):
            q=g.groupby('occupancy')[key].quantile([.25,.5,.75]).unstack()
            ax.plot(q.index,q[.5],'o-',label=c);ax.fill_between(q.index,q[.25],q[.75],alpha=.15)
        ax.set(xlabel='5-HT2A occupancy',ylabel=key)
    axes[0].legend();fig.suptitle('Subject medians and interquartile ranges; methods are not numerically interchangeable')
    fig.savefig(args.output/'pci_occupancy.png',dpi=200);fig.savefig(args.output/'pci_occupancy.svg');plt.close(fig)

if __name__=='__main__':main()
