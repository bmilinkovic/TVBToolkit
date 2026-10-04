"""Explicit regional state, OFF and PCI metrics; independent of simulation."""
import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
from sklearn.mixture import GaussianMixture


def runs(mask):
    """Half-open intervals, including boundary-censored intervals."""
    mask=np.asarray(mask,dtype=bool)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def durations(mask,dt,minimum=0,complete=True):
    return np.array([(b-a)*dt for a,b in runs(mask)
                     if (b-a)*dt>=minimum and (not complete or (a>0 and b<len(mask)))],float)


def smooth(x,dt,p):
    return uniform_filter1d(np.asarray(x,float),max(1,round(p['states']['smoothing_ms']/dt)),axis=0,mode='nearest')


def state_metrics(trace,dt,p):
    """Bimodality is evidence for switching, not proof of two stable attractors.

No quantile cut can create DOWN states in a unimodal trace. Duration estimates
exclude censored boundary runs; brief excursions are marked unclassified.
"""
    trace=np.asarray(trace,float);cfg=p['states']
    x=smooth(trace,dt,p);y=np.log1p(np.maximum(x,0))
    fit=y[::max(1,round(cfg['fit_step_ms']/dt)),None]
    status=np.ones(len(x),dtype=int);bic_gain=0.;separation=0.;accepted=False
    low=high=float(np.mean(x));threshold=None
    if np.std(fit)>1e-6 and len(fit)>=30:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            one=GaussianMixture(1,random_state=0,n_init=1,reg_covar=1e-6).fit(fit)
            two=GaussianMixture(2,random_state=0,n_init=3,reg_covar=1e-6).fit(fit)
        order=np.argsort(two.means_[:,0]);means=two.means_[order,0]
        variances=two.covariances_[order,0,0]
        bic_gain=float(one.bic(fit)-two.bic(fit))
        separation=float(np.sqrt(2)*(means[1]-means[0])/np.sqrt(variances.sum()))
        low,high=np.expm1(means)
        accepted=bool(bic_gain>=cfg['bic_improvement'] and separation>=cfg['ashman_d']
                      and two.weights_.min()>=cfg['minimum_component_fraction']
                      and low<=cfg['low_high_ratio']*high and high>=.1)
        if accepted:
            # Midpoint between fitted log-rate modes, frozen for this trace.
            threshold=float(np.expm1(means.mean()));status=(x>threshold).astype(int)
            for value in [0,1]:
                for a,b in runs(status==value):
                    if (b-a)*dt<cfg['minimum_dwell_ms']:status[a:b]=-1
    quiescent=bool(np.mean(x)<.1 and np.quantile(x,.95)<.2)
    if quiescent:status[:]=0
    up=durations(status==1,dt,cfg['minimum_dwell_ms'])
    down=durations(status==0,dt,cfg['minimum_dwell_ms'])
    cycles=int(min(len(up),len(down)))
    # Welch frequency is modulation of firing rate, never the rate's Hz value.
    f,psd=welch(trace,fs=1000/dt,nperseg=min(len(trace),int(4000/dt)),detrend='constant')
    band=(f>=.25)&(f<=45)
    total=float(psd[band].sum())
    dominant=float(f[band][np.argmax(psd[band])]) if total>1e-15 else None
    slow=float(psd[(f>=.25)&(f<=4)].sum()/total) if total>1e-15 else None
    def avg(a):return float(np.mean(a)) if len(a) else None
    def med(a):return float(np.median(a)) if len(a) else None
    return dict(bimodal=accepted,slow_switching=bool(accepted and cycles>=cfg['minimum_complete_cycles'] and dominant is not None and dominant<=4),
                quiescent=quiescent,up_fraction=float(np.mean(status==1)),down_fraction=float(np.mean(status==0)),
                unclassified_fraction=float(np.mean(status<0)),up_mean_ms=avg(up),up_median_ms=med(up),
                down_mean_ms=avg(down),down_median_ms=med(down),complete_cycles=cycles,
                up_complete_count=len(up),down_complete_count=len(down),state_threshold_hz=threshold,
                low_mode_hz=float(low),high_mode_hz=float(high),bic_gain=bic_gain,ashman_d=separation,
                dominant_frequency_hz=dominant,slow_power_fraction=slow),status


def regional_baseline(state,dt,p):
    """state: time x [E kHz, I kHz, W pA] x region."""
    e=state[:,0]*1000;i=state[:,1]*1000;w=state[:,2];rows=[]
    for k in range(e.shape[1]):
        m,_=state_metrics(e[:,k],dt,p)
        rows.append(dict(region_index=k,E_mean_hz=float(e[:,k].mean()),I_mean_hz=float(i[:,k].mean()),
            E_variance_hz2=float(e[:,k].var()),I_variance_hz2=float(i[:,k].var()),E_peak_hz=float(e[:,k].max()),
            I_peak_hz=float(i[:,k].max()),W_mean_pA=float(w[:,k].mean()),W_peak_pA=float(w[:,k].max()),**m))
    return rows


def off_metrics(baseline,post,dt,p):
    """Per-region OFF proxy after a positive initial response, frozen baseline cut.

Return to threshold ends an OFF event. A run touching the 1-s endpoint is
right-censored; its duration is a lower bound. No division by zero in quiescence.
"""
    base=smooth(baseline,dt,p);response=smooth(post,dt,p);cfg=p['off'];rows=[]
    t=(np.arange(len(post))+1)*dt
    for k in range(base.shape[1]):
        mean=float(base[:,k].mean());sd=float(base[:,k].std());q=float(np.quantile(base[:,k],cfg['baseline_quantile']))
        threshold=min(q,cfg['baseline_mean_fraction']*mean)
        early=np.flatnonzero(t<=cfg['initial_window_ms']);peak=int(early[np.argmax(response[early,k])])
        excursion=float(response[peak,k]-mean)
        initial=bool(excursion>=max(cfg['initial_peak_sd']*sd,cfg['initial_minimum_hz']))
        window=(t>=max(p['pulse_ms'],t[peak]))&(t<=cfg['end_ms'])
        intervals=[(a,b) for a,b in runs(window&(response[:,k]<threshold)) if (b-a)*dt>=cfg['minimum_duration_ms']]
        present=bool(initial and intervals)
        if present:
            a,b=intervals[0];duration=(b-a)*dt;depth=float(mean-response[a:b,k].min())
            onset=float(t[a]);censored=bool(b==len(post) or t[min(b,len(post)-1)]>cfg['end_ms'])
        else:duration=0.;depth=0.;onset=None;censored=False
        rows.append(dict(region_index=k,initial_response_hz=excursion,initial_response_sd=excursion/max(sd,1e-12),initial_response_detected=initial,
            off_present=present,off_threshold_hz=threshold,off_duration_ms=float(duration),off_depth_hz=depth,
            off_depth_fraction=float(depth/max(mean,1e-12)) if present else 0.,off_onset_ms=onset,
            off_right_censored=censored,maximum_suppression_hz=float(max(0,mean-response[window,k].min()))))
    return rows


def validity(state,dt,p):
    cfg=p['validity'];reasons=[]
    if not np.isfinite(state).all():return dict(valid=False,invalid_reasons=['nonfinite'],PFP=False)
    rates=state[:,:2]*1000
    if rates.min() < -cfg['negative_tolerance_hz']:reasons.append('negative_rates')
    for j,name in enumerate(['E','I']):
        means=rates[:,j].mean(axis=0);lo,hi=cfg[name+'_mean_hz']
        if (means<lo).any():reasons.append(name+'_region_mean_below_bound')
        if (means>hi).any():reasons.append(name+'_region_mean_above_bound')
    high=np.any(rates>cfg['high_hz'],axis=1);pfp=np.any(rates>cfg['PFP_hz'],axis=1)
    high_event=any(len(durations(high[:,k],dt,cfg['high_duration_ms'],False)) for k in range(high.shape[1]))
    pfp_event=any(len(durations(pfp[:,k],dt,cfg['PFP_duration_ms'],False)) for k in range(pfp.shape[1]))
    if high_event:reasons.append('sustained_implausible_rate')
    if pfp_event:reasons.append('PFP_episode')
    return dict(valid=not reasons,invalid_reasons=reasons,PFP=bool(pfp_event),max_EI_hz=float(rates.max()))


def pci_metrics(epochs,p):
    """Existing estimators; one-trial fallback explicitly distinguished."""
    from first_order_doc16_pci import sampling,lz_score
    from tvbtoolkit.complexity.pci_casali import binarise_signals_casali
    from tvbtoolkit.complexity.pci_st import pci_st_from_trials
    cfg=p['pci'];x,t=sampling(np.asarray(epochs),cfg['sampling_ms'])
    pre=(t>=cfg['baseline_ms'][0])&(t<cfg['baseline_ms'][1]);post=(t>=0)&(t<cfg['response_ms'][1]);npre=int(pre.sum())
    result={};arrays={}
    for scope,ids in [('all90',np.arange(90)),('remote89',np.delete(np.arange(90),p['source_index']))]:
        data=x[:,ids];stack=np.concatenate([data[:,:,pre],data[:,:,post]],axis=2)
        sig=binarise_signals_casali(stack,npre,n_bootstrap=cfg['bootstrap'],alpha=cfg['alpha'],seed=cfg['seed'],
            significance_method='trial_bootstrap',single_trial='baseline_resample' if len(x)==1 else 'raise',return_details=True)
        mask=sig.binary[:,npre:];lz=lz_score(mask)
        st=pci_st_from_trials(data,t,baseline_window_ms=tuple(cfg['baseline_ms']),response_window_ms=tuple(cfg['response_ms']),
            k=cfg['k'],min_snr=cfg['min_snr'],max_var_percent=cfg['max_var_percent'],n_steps=cfg['n_steps'],return_details=True)
        result.update({scope+'_PCI_ST':float(st.pci_st),scope+'_ST_components':int(st.n_components),
            scope+'_PCI_LZ':lz['PCI_LZ'] if len(x)>1 else None,
            scope+'_LZ_single_trial_sensitivity':lz['PCI_LZ'] if len(x)==1 else None,
            scope+'_recruited_regions':int(mask.any(axis=1).sum()),scope+'_active_fraction':float(mask.mean()),scope+'_lz_count':lz['lz_count']})
        arrays[scope+'_mask']=mask;arrays[scope+'_p']=sig.corrected_p_values[:,npre:]
    result.update(PCI_trials=len(x),recruitment_method='single_trial_baseline_resample_exploratory' if len(x)==1 else 'trial_bootstrap',PCI_LZ_native_available=len(x)>1)
    arrays['time_ms']=t[post]
    return result,arrays
