"""Same-trial empirical-style/legacy TVBSim PCI-LZ and published PCI-ST.

This orchestrates existing estimators; it does not change their defaults or
the neural model. Method adaptations are explicit sensitivity analyses.
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,resample_poly
from first_order_trial_pci import ROOT,load_trials,save_figure
from tvbtoolkit.complexity.pci_casali import (
    binarise_signals_casali,binarise_signals,sort_binJ,lz_complexity_2d,pci_norm_factor,source_entropy)
from tvbtoolkit.complexity.pci_st import pci_st_from_trials

PHASE='doc16_amp055'
COHORTS=['control','emcs','mcs','uws']
ANALYSIS_PREFIX='analysis_baseline400_100_response_onset0_300_n'
PULSE_OFFSET_MS=10.
TIMES=np.arange(-400.,300.)


def lz_score(binary):
    ordered=sort_binJ(binary);norm=pci_norm_factor(ordered)
    count=lz_complexity_2d(ordered) if norm>0 else 0
    return dict(PCI_LZ=float(count/norm) if norm>0 else 0.,lz_count=int(count),
                normalizer=float(norm),entropy=source_entropy(binary),
                active_fraction=float(binary.mean()))


def sampling(data,dt):
    if dt==1:return data,TIMES.copy()
    if dt!=3:raise ValueError('Only prespecified 1-ms and 3-ms grids')
    # Start at -399 ms so t=0 is exactly on the output grid. Polyphase FIR
    # anti-aliasing is analysis-only; never alters integrated model dynamics.
    out=resample_poly(data[:,:,1:],1,3,axis=2,padtype='line')
    t=np.arange(-399.,300.,3.)
    assert out.shape[2]==len(t)
    return out,t


def evaluate(data,dt=3,response_start=0.,seed=17001):
    data,t=sampling(data,dt)
    pre=(t>=-400)&(t<-100);post=(t>=0)&(t<300);keep=t[post]>=response_start
    stacked=np.concatenate([data[:,:,pre],data[:,:,post]],axis=2)
    onset=int(pre.sum())
    sig=binarise_signals_casali(stacked,onset,n_bootstrap=2000,alpha=.01,
        seed=seed,significance_method='trial_bootstrap',return_details=True)
    binary=sig.binary[:,onset:][:,keep]
    native=lz_score(binary)
    pvalues=sig.corrected_p_values[:,onset:][:,keep]
    # Preserve and restore global RNG because the historical API uses it.
    state=np.random.get_state()
    try:
        np.random.seed(seed)
        legacy=binarise_signals(stacked,onset,nshuffles=10,percentile=100.)[:,:,onset:][:,:,keep]
    finally:
        np.random.set_state(state)
    legacy_scores=np.array([lz_score(b)['PCI_LZ'] for b in legacy])
    st=pci_st_from_trials(data,t,baseline_window_ms=(-400.,-100.),
        response_window_ms=(response_start,300.),k=1.2,min_snr=1.1,
        max_var_percent=99.,n_steps=100,return_details=True)
    centered=data-data[:,:,pre].mean(axis=2,keepdims=True)
    evoked=centered.mean(axis=0)[:,post][:,keep]
    activation_order=np.argsort(-np.max(abs(evoked),axis=1),kind='stable')
    binary_order=np.sum(binary,axis=1).argsort()[::-1]
    occupancy=legacy.mean(axis=0)
    legacy_order=np.argsort(-occupancy.sum(axis=1),kind='stable')
    row=dict(n_trials=len(data),dt_ms=dt,response_start_ms=response_start,
        baseline_window_ms=[-400.,-100.],response_window_ms=[response_start,300.],
        baseline_reference='pulse_onset',response_reference='pulse_onset',
        pulse_offset_after_onset_ms=PULSE_OFFSET_MS,
        baseline_sample_range_ms=[float(t[pre][0]),float(t[pre][-1])],
        response_sample_range_ms=[float(t[post][keep][0]),float(t[post][keep][-1])],
        response_n_bins=int(keep.sum()),
        PCI_LZ_native=native['PCI_LZ'],PCI_LZ_legacy_mean=float(legacy_scores.mean()),
        PCI_LZ_legacy_sd=float(legacy_scores.std(ddof=1)),PCI_ST=float(st.pci_st),
        native_active_fraction=native['active_fraction'],legacy_active_fraction=float(occupancy.mean()),
        native_significant_regions=int(binary.any(axis=1).sum()),
        native_significant_remote_regions=int(binary[np.arange(len(binary))!=9].any(axis=1).sum()),
        native_entropy=native['entropy'],native_lz_count=native['lz_count'],
        threshold_baseline_sd=float(sig.threshold),ST_components=st.n_components,
        alpha_sensitivity=[dict(alpha=a,**lz_score(pvalues<=a)) for a in [.005,.01,.05]],
        ST_snr_sensitivity=[dict(min_snr=s,PCI_ST=float(st.component_contributions[st.retained_snrs>s].sum()),
            n_components=int((st.retained_snrs>s).sum())) for s in [1.1,1.5,2.,3.]])
    arrays=dict(time_ms=t[post][keep],
        time_after_onset_ms=t[post][keep],evoked_hz=evoked,binary_native=binary,
        native_p=pvalues,native_null_maxima=sig.null_maxima,
        legacy_binary_trials=legacy,legacy_trial_scores=legacy_scores,
        legacy_occupancy=occupancy,activation_order=activation_order,
        significance_order=binary_order,legacy_order=legacy_order,
        ST_component_time_ms=t[(t>=-400)&(t<300)],ST_components=st.component_signals,
        ST_contributions=st.component_contributions,ST_snrs=st.retained_snrs)
    return row,arrays


def filtered_epochs(cohort,subject,seeds):
    """45-Hz low-pass sensitivity, applied to full 11-s records before cropping."""
    outputs=[[],[]];sos=butter(4,45.,fs=1000.,output='sos')
    for seed in seeds:
        folder=ROOT/PHASE/f'{cohort}_{subject}_seed{seed}'
        with np.load(folder/'baseline.npz') as a:base=a['state'][:,0].T.astype(float)*1000
        for output,amplitude in zip(outputs,[.55,0.]):
            with np.load(folder/f'amplitude_{amplitude:g}.npz') as a:post=a['state'][:,0].T.astype(float)*1000
            full=sosfiltfilt(sos,np.concatenate([base,post],axis=1),axis=1)
            # Baseline ends at index7999 (onset t=0); retain [-400,299] ms.
            start=base.shape[1]-1-400
            output.append(full[:,start:start+700])
    return [np.array(x) for x in outputs]


def plot_subject(folder,name,rows,arrays,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    vmax=max(max(abs(a['evoked_hz']).ravel()) for a in arrays)
    for j,(condition,a,r) in enumerate(zip(['stimulated','unstimulated'],arrays,rows)):
        matrices=[a['evoked_hz'][a['activation_order']],a['binary_native'][a['significance_order']],
                  a['legacy_occupancy'][a['legacy_order']]]
        for k,m in enumerate(matrices):
            options=dict(cmap='RdBu_r',norm=SymLogNorm(linthresh=.01,vmin=-vmax,vmax=vmax)) if k==0 else dict(cmap='binary',vmin=0,vmax=1)
            im=axes[j,k].imshow(m,origin='upper',aspect='auto',extent=[0,300,90,0],**options)
            fig.colorbar(im,ax=axes[j,k],label=['E response (Hz), shared symlog','Significant: 0 or 1','Fraction of trials (NOT p value)'][k],shrink=.65)
            order=a[['activation_order','significance_order','legacy_order'][k]]
            axes[j,k].axhline(np.flatnonzero(order==9)[0]+.5,color='#1b9e77',lw=.7)
            axes[j,k].set(title=['Unthresholded trial average','Empirical-style bootstrap mask','Legacy threshold exceedance'][k],
                xlabel='Time after pulse onset (ms)',ylabel=f'{condition}: regions ranked high to low')
            ticks=np.arange(0,90,15);axes[j,k].set_yticks(ticks+.5,[labels[q] for q in order[ticks]],fontsize=5)
        axes[j,3].axis('off');axes[j,3].text(0,1,
            f"N = {r['n_trials']}\n\nNative PCI-LZ = {r['PCI_LZ_native']:.4f}\nLegacy mean PCI-LZ = {r['PCI_LZ_legacy_mean']:.4f}\nPCI-ST = {r['PCI_ST']:.3f}\n\nNative active bins = {100*r['native_active_fraction']:.2f}%\nNative significant regions = {r['native_significant_regions']}/90\nNative binary entropy = {r['native_entropy']:.4f}\nLegacy exceedance = {100*r['legacy_active_fraction']:.2f}%\nST components = {r['ST_components']}\n\nLegacy: one binary map / PCI\nper trial, then mean scores.\nST: continuous trial average,\nnot either significance map.",va='top',fontsize=9)
    fig.suptitle(name+'; 0.55 Hz/ms, 10-ms pulse; G=0.05, b=5 pA\nBaseline −400 to −100 ms; response 0–300 ms from pulse onset (pulse included)')
    save_figure(fig,folder,'method_comparison');plt.close(fig)
    # Show an actual single-trial legacy binary map, not a thresholded occupancy
    # map invented to resemble the empirical-style statistic.
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,a,condition in zip(axes,arrays,['stimulated','unstimulated']):
        b=a['legacy_binary_trials'][0];order=b.sum(axis=1).argsort()[::-1]
        ax.imshow(b[order],aspect='auto',extent=[0,300,90,0],cmap='binary',vmin=0,vmax=1)
        ax.set(title=f"{condition}: first saved trial\nPCI-LZ={a['legacy_trial_scores'][0]:.4f}",xlabel='Time after pulse onset (ms)',ylabel='Ranked regions')
    save_figure(fig,folder,'legacy_single_trial_masks');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for j,(a,condition) in enumerate(zip(arrays,['stimulated','unstimulated'])):
        if len(a['ST_contributions']):
            axes[j,0].plot(a['ST_component_time_ms'],a['ST_components'].T,lw=.6)
            axes[j,1].bar(np.arange(len(a['ST_contributions']))+1,a['ST_contributions'])
        axes[j,0].axvline(0,color='k',lw=.6)
        axes[j,0].set(xlabel='Time after onset (ms)',ylabel='SVD component',title=condition)
        axes[j,1].set(xlabel='Component',ylabel='PCI-ST contribution',title='Components are not anatomical sources')
    fig.suptitle(name+' — actual PCI-ST processing, separate from binary activation maps')
    save_figure(fig,folder,'PCI_ST_components');plt.close(fig)


def analyse_subject(job):
    subject,n=job;c=subject['cohort'];s=subject['subject']
    folder=ROOT/PHASE/f'{ANALYSIS_PREFIX}{n}'/f'{c}_{s}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    # Failed physiological gates are surfaced, never repaired by excluding trials.
    try:
        trials,sham,paired,labels,seeds=load_trials(PHASE,c,.55,subject=s,post_stop_ms=300)
    except ValueError as exc:
        row=dict(cohort=c,subject=s,status='physiological_gate_failed',reason=str(exc))
        (folder/'summary.json').write_text(json.dumps(row,indent=2));return row
    wanted=json.loads((ROOT/PHASE/'status.json').read_text())['seeds'][:n]
    selected=np.isin(seeds,wanted)
    trials,sham,paired,seeds=trials[selected],sham[selected],paired[selected],seeds[selected]
    if len(trials)!=n:raise ValueError(f'{c}/{s}: expected {n}, found {len(trials)}')
    rows=[];primary_arrays=[]
    for condition,data in [('stimulated',trials),('unstimulated',sham)]:
        r,a=evaluate(data);r.update(condition=condition,variant='primary_3ms_onset0_300')
        rows.append(r);primary_arrays.append(a)
        np.savez_compressed(folder/f'{condition}_primary.npz',**a,labels=labels,seeds=seeds,
                            paired_mean_hz=paired.mean(axis=0))
    plot_subject(folder,f'{c}/{s}',rows,primary_arrays,labels)
    sensitivity=[]
    filtered=filtered_epochs(c,s,seeds)
    for condition,data,lowpass in zip(['stimulated','unstimulated'],[trials,sham],filtered):
        for variant,x,dt,start in [('native_1ms',data,1,0.),('lowpass45_3ms',lowpass,3,0.)]:
            r,_=evaluate(x,dt=dt,response_start=start)
            sensitivity.append(dict(condition=condition,variant=variant,**r))
    # Trial convergence is assessed on the same subject, not by pooling cohorts.
    convergence=[];rng=np.random.default_rng(44219)
    for count in [v for v in [10,20,50,100] if v<=n]:
        for repeat in range(1 if count==n else 3):
            ix=np.arange(n) if count==n else np.sort(rng.choice(n,count,replace=False))
            for condition,data in [('stimulated',trials),('unstimulated',sham)]:
                r,_=evaluate(data[ix])
                convergence.append(dict(condition=condition,repeat=repeat,**r))
    row=dict(cohort=c,subject=s,n_trials=n,status='diagnostic_scores_not_clinically_validated',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection=subject,primary=rows,sensitivity=sensitivity,convergence=convergence,
        input='AAL90 excitatory firing rates in Hz; no LFP/EEG forward model',
        empirical_route='Explicit trial_bootstrap, 2000 resamples, alpha .01, two sided, no entropy floor',
        legacy_route='10 shuffles; percentile100 indexes minimum shuffled maximum; positive-only trial masks; mean of trial LZ scores',
        primary_dt_ms=3,baseline_ms=[-400,-100],baseline_reference='pulse_onset',
        response_ms=[0,300],response_reference='pulse_onset',response_after_onset_ms=[0,300],
        interpretation='Compare sham specificity, sampling/filtering sensitivity and convergence before interpreting group differences.')
    (folder/'summary.json').write_text(json.dumps(row,indent=2));print(c,s,n,[(r['condition'],r['PCI_LZ_native'],r['PCI_LZ_legacy_mean'],r['PCI_ST']) for r in rows],flush=True)
    return row


def overview(rows,n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/PHASE/f'{ANALYSIS_PREFIX}{n}'
    usable=[r for r in rows if 'primary' in r]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,title in zip(axes,['PCI_LZ_native','PCI_LZ_legacy_mean','PCI_ST'],['Empirical-style PCI-LZ','Legacy TVBSim mean PCI-LZ','PCI-ST']):
        for ci,c in enumerate(COHORTS):
            rr=sorted([r for r in usable if r['cohort']==c],key=lambda r:r['subject'])
            for j,r in enumerate(rr):
                x=ci+(j-1.5)*.07;a,b=r['primary']
                ax.plot([x,x],[b[key],a[key]],color=f'C{ci}',alpha=.5,lw=.8)
                ax.scatter(x,a[key],color=f'C{ci}',s=30)
                ax.scatter(x,b[key],facecolors='none',edgecolors=f'C{ci}',s=30)
                ax.annotate(r['subject'],(x,a[key]),xytext=(2,3),textcoords='offset points',fontsize=5)
        ax.set_xticks(range(4),['Wake','EMCS','MCS','UWS']);ax.set_title(title);ax.set_ylabel('Score (method-specific scale)')
    fig.suptitle(f'0.55 Hz/ms; {n} trials per subject; filled=stimulated, open=matched unstimulated\nFour structurally stratified subjects per cohort; fixed first-order parameters; preliminary, not population inference')
    save_figure(fig,folder,'cohort_PCI_comparison');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for r in usable:
        color=f'C{COHORTS.index(r["cohort"])}'
        for ax,key in zip(axes,['PCI_LZ_native','PCI_LZ_legacy_mean','PCI_ST']):
            for condition,style in [('stimulated','-'),('unstimulated','--')]:
                data=[q for q in r['convergence'] if q['condition']==condition]
                counts=sorted({q['n_trials'] for q in data})
                ax.plot(counts,[np.median([q[key] for q in data if q['n_trials']==v]) for v in counts],style,color=color,alpha=.6)
                ax.set(xlabel='Trials averaged/used',ylabel=key)
    fig.suptitle('Trial-count sensitivity; solid=stimulated, dashed=unstimulated; colour=cohort')
    save_figure(fig,folder,'cohort_trial_convergence');plt.close(fig)
    # Three 4x4 source-time montages, preserving actual anatomical permutations.
    from matplotlib.colors import SymLogNorm
    loaded={}
    for r in usable:
        path=folder/f"{r['cohort']}_{r['subject']}"/'stimulated_primary.npz'
        with np.load(path) as a:loaded[(r['cohort'],r['subject'])]={k:a[k] for k in a.files}
    if loaded:
        vmax=max(np.abs(a['evoked_hz']).max() for a in loaded.values())
        for field,order,title in [('evoked_hz','activation_order','Unthresholded trial-averaged response (Hz)'),
                                  ('binary_native','significance_order','Empirical-style significant activation (0/1)'),
                                  ('legacy_occupancy','legacy_order','Legacy trial exceedance fraction (not significance probability)')]:
            fig,axes=plt.subplots(4,4,figsize=(16,14),layout='constrained')
            for ci,c in enumerate(COHORTS):
                rr=sorted([r for r in usable if r['cohort']==c],key=lambda r:r['selection']['structural_rank'])
                for j,ax in enumerate(axes[ci]):
                    if j>=len(rr):ax.axis('off');continue
                    r=rr[j];a=loaded[(c,r['subject'])]
                    options=dict(cmap='RdBu_r',norm=SymLogNorm(linthresh=.01,vmin=-vmax,vmax=vmax)) if field=='evoked_hz' else dict(cmap='binary',vmin=0,vmax=1)
                    im=ax.imshow(a[field][a[order]],aspect='auto',origin='upper',extent=[0,300,90,0],**options)
                    ax.set(title=f"{c}/{r['subject']}",xlabel='Time after pulse onset (ms)',ylabel='Source rank (largest at top)')
                    ax.axhline(np.flatnonzero(a[order]==9)[0]+.5,color='#1b9e77',lw=.6)
            fig.colorbar(im,ax=axes,label=title,shrink=.6)
            fig.suptitle(f'{title}; {n} trials/subject, 0.55 Hz/ms\nCommon colour scale; anatomical row order differs between panels and is saved')
            save_figure(fig,folder,'all_subjects_'+field);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for r in usable:
        colour=f'C{COHORTS.index(r["cohort"])}'
        for q in r['primary']:
            style='-' if q['condition']=='stimulated' else '--'
            for ax,key in zip(axes[:2],['active_fraction','PCI_LZ']):
                ax.plot([a['alpha'] for a in q['alpha_sensitivity']],[a[key] for a in q['alpha_sensitivity']],style,color=colour,alpha=.5)
                ax.set(xlabel='Significance alpha',ylabel=key,xscale='log')
            axes[2].plot([a['min_snr'] for a in q['ST_snr_sensitivity']],[a['PCI_ST'] for a in q['ST_snr_sensitivity']],style,color=colour,alpha=.5)
    axes[2].set(xlabel='PCI-ST minimum SNR',ylabel='PCI-ST')
    fig.suptitle('Prespecified sensitivity, not automatic tuning: solid=stimulated, dashed=unstimulated\nNative primary alpha=.01; PCI-ST primary minimum SNR=1.1')
    save_figure(fig,folder,'threshold_sensitivity');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for r in usable:
        q=r['primary'][0];color=f'C{COHORTS.index(r["cohort"])}'
        count=q['native_active_fraction']*90*q['response_n_bins']
        axes[0].scatter(count,q['native_lz_count'],color=color)
        axes[1].scatter(q['native_entropy'],q['PCI_LZ_native'],color=color)
        axes[2].scatter(q['native_significant_regions'],q['PCI_LZ_native'],color=color)
    axes[0].set(xlabel='Number of significant source × time cells',ylabel='Raw LZ phrase count')
    axes[1].set(xlabel='Binary source entropy H',ylabel='Normalized PCI-LZ')
    axes[2].set(xlabel='Number of significant regions (including source)',ylabel='Normalized PCI-LZ')
    fig.suptitle('Sparse-mask normalization diagnostic: PCI-LZ = c log₂(L)/(L H)\nA nonzero score with only one active region is not evidence of distributed propagation; no entropy floor applied')
    save_figure(fig,folder,'sparse_mask_normalization');plt.close(fig)
    (folder/'summary.json').write_text(json.dumps(rows,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();selected=json.loads((ROOT/PHASE/'subject_selection.json').read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(analyse_subject,[(s,args.trials) for s in selected]))
    overview(rows,args.trials)


if __name__=='__main__':main()
