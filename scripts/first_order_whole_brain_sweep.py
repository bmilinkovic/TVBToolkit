"""Exploratory first-order G sweep; no PCI or automatic baseline selection."""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from first_order_adex_audit import OUT,configure,clean

SUBJECTS=[('control','c0013'),('emcs','e0003'),('mcs','m0075'),('uws','u0020')]

def run(job, freeze_inputs=False):
    cohort,subject,g=job;folder=OUT/'G_sweep';folder.mkdir(parents=True,exist_ok=True)
    name=f'{cohort}_{subject}_G{g:g}'
    if freeze_inputs:name+='_frozen_initial_input'
    if (folder/f'{name}.json').exists():return json.loads((folder/f'{name}.json').read_text())
    sim,cfg,labels=configure(isolated=False,cohort=cohort,subject=subject,coupling=g)
    original=sim.model.dfun;calls=0;inputs=[];initial_input=None
    def drift(x,c,lc=0.):
        nonlocal calls,initial_input
        if initial_input is None:initial_input=c.copy()
        if freeze_inputs:c=initial_input
        # Native Heun first-stage input at 1-ms spacing, not a zero-delay proxy.
        if calls%20==0:inputs.append(c[0,:,0].copy())
        calls+=1
        value=original(x,c,lc)
        if not np.isfinite(value).all():raise FloatingPointError('Nonfinite drift')
        return value
    sim.model.dfun=drift
    t,s=sim.run(simulation_length=3000.)[0]
    s=s[:,:,:,0];saved=s[9::10];times=t[9::10];lr=np.array(inputs)*1000
    rates=s[:,:2]*1000;high=(rates>150).any(axis=1)
    first=[]
    for i in range(90):
        # Require >150 Hz continuously for 100 ms; distinct from nonfinite failure.
        hits=np.flatnonzero(np.convolve(high[:,i].astype(int),np.ones(1000,dtype=int),mode='valid')==1000)
        first.append(float(t[hits[0]]) if len(hits) else None)
    found=[i for i,v in enumerate(first) if v is not None]
    earliest=min(found,key=lambda i:first[i]) if found else None
    row=clean(dict(cohort=cohort,subject=subject,G=g,b_pA=5.,duration_ms=3000.,seed=1230263469,
        mean_EI_last_second_hz=rates[-10000:].mean(axis=(0,2)),
        max_EI_hz=rates.max(),max_long_range_hz=lr.max(),
        mean_long_range_last_second_hz=lr[-1000:].mean(),first_high_region=None if earliest is None else str(labels[earliest]),
        transition_times_ms=first,first_high_ms=None if earliest is None else first[earliest],
        frozen_initial_long_range_input=freeze_inputs,
        note='Exploratory one-seed G sweep. High event means >150 Hz for 100ms, not proof of an asymptotic PFP.'))
    np.savez_compressed(folder/f'{name}.npz',time_ms=times,state=saved,
        input_time_ms=np.arange(len(lr)),long_range_hz=lr,labels=labels,
        incoming_strength=cfg.weights.sum(axis=1),effective_strength=g*cfg.weights.sum(axis=1),
        weights=cfg.weights,delay_steps=sim.connectivity.idelays)
    (folder/f'{name}.json').write_text(json.dumps(row,indent=2));print(cohort,g,row['first_high_region'],flush=True)
    return row

if __name__=='__main__':
    jobs=[(c,s,g) for c,s in SUBJECTS for g in [0.,.01,.05,.1,.25,.5,1.,2.,5.]]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,jobs))
    refinement=[]
    for c,s in SUBJECTS:
        rr=sorted([r for r in rows if r['cohort']==c],key=lambda r:r['G'])
        hits=[r for r in rr if r['first_high_ms'] is not None]
        if hits:
            upper=hits[0]['G'];lower=max(r['G'] for r in rr if r['G']<upper)
            refinement.extend((c,s,float(round(g,6))) for g in np.linspace(lower,upper,6)[1:-1])
    with ProcessPoolExecutor(max_workers=4) as pool:rows.extend(pool.map(run,refinement))
    (OUT/'G_sweep_summary.json').write_text(json.dumps(rows,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for c,_ in SUBJECTS:
        rr=sorted([r for r in rows if r['cohort']==c],key=lambda r:r['G'])
        for ax,key in zip(axes,['max_EI_hz','mean_long_range_last_second_hz','first_high_ms']):
            ax.plot([r['G'] for r in rr],[np.nan if r[key] is None else r[key] for r in rr],'o-',label=c)
    for ax,label in zip(axes,['Maximum E/I rate (Hz)','Mean long-range input (Hz)','First sustained high event (ms)']):
        ax.set(xlabel='G',ylabel=label);ax.legend()
    fig.suptitle('First-order exploratory G sweep, b=5 pA, 3 seconds, one subject per cohort\nNot a selected baseline; missing event markers mean no event within this window')
    fig.savefig(OUT/'whole_brain_G_sweep.png',dpi=200)
