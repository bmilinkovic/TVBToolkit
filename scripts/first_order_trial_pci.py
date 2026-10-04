"""Trial-averaged PCI diagnostics, separate from dynamics and amplitude selection.

No pre/post swaps, entropy floor, LFP conversion or rate clipping. PCI-LZ uses
the existing baseline whole-trial bootstrap sensitivity route, NOT a claim of
exact equivalence to the unavailable Casali 2013 supplementary implementation.
PCI-ST is calculated on continuous trial-averaged activity, not the LZ mask.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tvbtoolkit.complexity.pci_casali import (
    binarise_signals_casali, sort_binJ, source_entropy, lz_complexity_2d, pci_norm_factor)
from tvbtoolkit.complexity.pci_st import pci_st_from_trials

ROOT = Path(__file__).resolve().parents[1]/'results/first_order_adex_validation/amplitude_trials_v1'
TIMES = np.arange(-400., 300.)
BASE = (TIMES >= -400) & (TIMES < -50)
LZ_BASE = (TIMES >= -350) & (TIMES < -50)
POST = (TIMES >= 0) & (TIMES < 300)
METHOD = dict(pci_lz='TVBSim/native 2D LZ; baseline whole-trial bootstrap sensitivity analysis',
    significance_method='trial_bootstrap', alpha=.01, n_bootstrap=2000,
    baseline_window_lz_ms=[-350,-50], baseline_window_st_ms=[-400,-50],
    response_window_ms=[0,300], dt_ms=1., k=1.2, min_snr=1.1, max_var_percent=99.,
    n_steps=100, embed=False, average_reference=False, entropy_floor=None,
    signal='regional excitatory firing rates in Hz, not EEG/source-current units')


def load_trials(phase, cohort, amplitude, subject=None, post_stop_ms=300):
    epochs=[]; shams=[]; ids=[]; labels=None; paired=[]
    pattern=f'{cohort}_{subject}_seed*/summary.json' if subject else f'{cohort}_*/summary.json'
    subjects_seen=set()
    for path in sorted((ROOT/phase).glob(pattern)):
        row=json.loads(path.read_text()); folder=path.parent
        subjects_seen.add(row['subject'])
        if len(subjects_seen)>1:
            raise ValueError('Specify one subject; never pool subjects as trials')
        pulse=next((r for r in row['pulses'] if np.isclose(r['amplitude_hz_per_ms'],amplitude)),None)
        if pulse is None:
            continue
        if row['baseline_high']['sustained_high'] or pulse['sustained_high']:
            raise ValueError('Baseline/TMS validation failed: do not silently discard high-state trials')
        with np.load(folder/'baseline.npz') as b, np.load(folder/f'amplitude_{amplitude:g}.npz') as p, np.load(folder/'amplitude_0.npz') as c:
            pre=b['state'][-401:,0].T.astype(float)*1000
            ep=np.concatenate([pre,p['state'][:post_stop_ms-1,0].T.astype(float)*1000],axis=1)
            sham=np.concatenate([pre,c['state'][:post_stop_ms-1,0].T.astype(float)*1000],axis=1)
            actual=np.concatenate([b['time_ms'][-401:],p['time_ms'][:post_stop_ms-1]])
            np.testing.assert_allclose(actual,np.arange(-400.,float(post_stop_ms)),atol=1e-7)
            epochs.append(ep);shams.append(sham);paired.append(ep-sham)
            labels=b['labels'];ids.append(row['seed'])
    if not epochs:
        raise ValueError(f'No complete eligible trials for {cohort}, {amplitude}')
    return np.array(epochs),np.array(shams),np.array(paired),labels,np.array(ids)


def evaluate(trials, seed=101, bootstrap=2000):
    sig_input=np.concatenate([trials[:,:,LZ_BASE],trials[:,:,POST]],axis=2)
    sig=binarise_signals_casali(sig_input,300,n_bootstrap=bootstrap,alpha=.01,
        seed=seed,significance_method='trial_bootstrap',return_details=True)
    binary=sig.binary[:,300:]
    ordered=sort_binJ(binary)
    norm=pci_norm_factor(ordered)
    lz_count=lz_complexity_2d(ordered) if norm>0 else 0
    lz=float(lz_count/norm) if norm>0 else 0.
    st=pci_st_from_trials(trials,TIMES,return_details=True,
        baseline_window_ms=(-400.,-50.),response_window_ms=(0.,300.),k=1.2,
        min_snr=1.1,max_var_percent=99.,n_steps=100,embed=False,average_reference=False)
    corrected=trials-trials[:,:,BASE].mean(axis=2,keepdims=True)
    evoked=corrected.mean(axis=0)
    # Display order only: never feed activation-sorted continuous data into LZ.
    activation_order=np.argsort(-np.max(np.abs(evoked[:,POST]),axis=1),kind='stable')
    significance_order=np.sum(binary,axis=1).argsort()[::-1]  # exactly sort_binJ
    result=dict(n_trials=len(trials),PCI_LZ=lz,PCI_ST=float(st.pci_st),
        lz_count=int(lz_count),lz_normalizer=float(norm),source_entropy=source_entropy(binary),
        significant_fraction=float(binary.mean()),significant_regions=int(binary.any(axis=1).sum()),
        threshold_baseline_sd=float(sig.threshold),n_st_components=st.n_components,
        threshold_seed=seed,n_bootstrap=bootstrap)
    arrays=dict(time_ms=TIMES,evoked_hz=evoked,binary=binary,
        corrected_p=sig.corrected_p_values[:,300:],standardized_response=sig.averaged_response[:,300:],
        null_maxima=sig.null_maxima,activation_order=activation_order,
        significance_order=significance_order,st_components=st.component_signals,
        st_contributions=st.component_contributions,st_snrs=st.retained_snrs,
        st_thresholds=st.thresholds,st_nst_difference=st.nst_difference)
    return result,arrays


def save_figure(fig,folder,name):
    fig.savefig(folder/f'{name}.png',dpi=300)
    fig.savefig(folder/f'{name}.svg')


def plot_analysis(folder,name,result,arrays,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    a=arrays;post=a['evoked_hz'][:,POST];vmax=max(abs(post).max(),1e-6)
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained',gridspec_kw={'width_ratios':[1.4,1.4,1]})
    image=axes[0].imshow(post[a['activation_order']],aspect='auto',origin='upper',
        extent=[0,300,90,0],cmap='RdBu_r',norm=SymLogNorm(linthresh=.001,vmin=-vmax,vmax=vmax))
    fig.colorbar(image,ax=axes[0],label='Trial-averaged baseline-corrected E (Hz)')
    axes[0].set(title='A  Response: largest peak at top',xlabel='Time after onset (ms)',ylabel='AAL90 regions, ranked')
    axes[1].imshow(a['binary'][a['significance_order']],aspect='auto',origin='upper',
        extent=[0,300,90,0],cmap='binary',vmin=0,vmax=1)
    axes[1].set(title='B  PCI-LZ mask: most significant bins at top',xlabel='Time after onset (ms)',ylabel='AAL90 regions, re-ranked')
    for ax,order in zip(axes[:2],[a['activation_order'],a['significance_order']]):
        ticks=np.arange(0,90,10);ax.set_yticks(ticks+.5,[labels[i] for i in order[ticks]],fontsize=5)
        ax.axhline(float(np.flatnonzero(order==9)[0])+.5,color='#1b9e77',lw=.6)
    axes[2].axis('off')
    axes[2].text(0,1,f"C  Scores (different scales)\n\nPCI-LZ: {result['PCI_LZ']:.4f}\nPCI-ST: {result['PCI_ST']:.3f}\n\nN = {result['n_trials']} trials\nSignificant bins: {100*result['significant_fraction']:.2f}%\nPCI-ST components: {result['n_st_components']}\n\nPCI-ST uses continuous activity,\nnot panel B's binary mask.\nGreen line: stimulated source.\n\nDiagnostic results; not clinical cutoffs.",va='top',fontsize=10)
    fig.suptitle(name)
    save_figure(fig,folder,name);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    if len(a['st_contributions']):
        axes[0].plot(TIMES,a['st_components'].T,lw=.65)
        axes[1].bar(np.arange(len(a['st_contributions']))+1,a['st_contributions'])
    axes[0].axvline(0,color='k',lw=.6);axes[0].set(xlabel='Time (ms)',ylabel='SVD component',title='PCI-ST retained continuous components')
    axes[1].set(xlabel='Component',ylabel='Contribution to PCI-ST',title='State-transition contributions')
    save_figure(fig,folder,name+'_ST_components');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',default='confirm');p.add_argument('--amplitude',type=float,required=True)
    p.add_argument('--cohort',default='control');p.add_argument('--min-trials',type=int,default=100)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--subset-repeats',type=int,default=5)
    args=p.parse_args()
    trials,shams,paired,labels,seeds=load_trials(args.phase,args.cohort,args.amplitude)
    if len(trials)<args.min_trials:
        raise ValueError(f'Only {len(trials)} complete trials; require {args.min_trials}')
    folder=ROOT/'PCI'/f'{args.phase}_{args.cohort}_amp{args.amplitude:g}'
    folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,data in [('stimulated',trials),('sham',shams)]:
        r,a=evaluate(data,bootstrap=args.bootstrap)
        r['condition']=name;results.append(r)
        np.savez_compressed(folder/f'{name}.npz',**a,trial_seeds=seeds,labels=labels,
                            matched_response_hz=paired.mean(axis=0))
        plot_analysis(folder,name,r,a,labels)
    # Trial-count convergence: repeated subsets without replacement, NOT
    # independent experimental replications or bootstrap confidence intervals.
    convergence=[];rng=np.random.default_rng(20260930)
    for n in sorted(set([n for n in [10,20,50,100,200] if n<=len(trials)]+[len(trials)])):
        for rep in range(1 if n==len(trials) else args.subset_repeats):
            ix=np.arange(n) if n==len(trials) else np.sort(rng.choice(len(trials),n,replace=False))
            for name,data in [('stimulated',trials),('sham',shams)]:
                r,_=evaluate(data[ix],seed=101,bootstrap=args.bootstrap)
                convergence.append(dict(condition=name,subset_repeat=rep,**r))
                print(args.cohort,name,n,rep,r['PCI_LZ'],r['PCI_ST'],flush=True)
    (folder/'summary.json').write_text(json.dumps(dict(method=METHOD,results=results,
        trial_seeds=seeds.tolist(),convergence=convergence,
        interpretation='Diagnostic scores. Inspect sham specificity and convergence before biological interpretation.'),indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key in zip(axes,['PCI_LZ','PCI_ST','significant_fraction']):
        for name,color in [('stimulated','C0'),('sham','C1')]:
            rr=[r for r in convergence if r['condition']==name]
            ax.scatter([r['n_trials'] for r in rr],[r[key] for r in rr],s=15,alpha=.6,color=color,label=name)
        ax.set(xlabel='Trials averaged',ylabel=key);ax.legend()
    fig.suptitle('Trial-count sensitivity: repeated subsets, not independent replications')
    save_figure(fig,folder,'trial_count_convergence');plt.close(fig)


if __name__=='__main__':
    main()
