"""Tidy tables, heatmaps and explicitly provisional phenotype selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COHORTS=['control','emcs','mcs','uws']
METRICS=['E_mean_hz','I_mean_hz','E_variance_hz2','I_variance_hz2','dominant_frequency_hz',
         'up_fraction','down_fraction','up_median_ms','down_median_ms','W_mean_pA','W_peak_pA','slow_switching',
         'source_off_depth_fraction','source_off_duration_ms','off_region_fraction',
         'remote89_recruited_regions','all90_PCI_ST','all90_LZ_single_trial_sensitivity']

def collect(out):
    rows=[];manifest=json.loads((out/'manifest.json').read_text());seen=set()
    for file in sorted((out/'cells').glob('*/result.json')):
        item=json.loads(file.read_text());rows.extend(item['rows']);j=item['job']
        seen.add((j['cohort'],j['subject'],j['b'],j['tau'],j['replicate']))
    for j in manifest['jobs']:
        if (j['cohort'],j['subject'],j['b'],j['tau'],j['replicate']) not in seen:
            for condition in ['spontaneous','TMS']:
                rows.append(dict(subject=j['subject'],diagnostic_group=j['cohort'],b_pA=j['b'],tau_w_ms=j['tau'],replicate=j['replicate'],seed=j['seed'],stimulation_condition=condition,metrics_status='pending',valid=None))
    frame=pd.DataFrame(rows);temp=out/'metrics.tmp.csv';frame.to_csv(temp,index=False);temp.replace(out/'metrics.csv')
    # Companion region-level metrics remain per cell until final aggregation.
    status=dict(completed_pairs=len(seen),total_pairs=manifest['total_pairs'],pending_pairs=manifest['total_pairs']-len(seen),
                table_rows=len(frame),analysis_complete=len(seen)==manifest['total_pairs'])
    (out/'table_status.json').write_text(json.dumps(status,indent=2));return frame,manifest,status

def heatmaps(frame,out,p,title,prefix):
    complete=frame[frame.metrics_status=='complete']
    metrics=[m for m in METRICS if m in complete and complete[m].notna().any()]
    if not metrics:return
    fig,axes=plt.subplots(int(np.ceil(len(metrics)/3)),3,figsize=(14,3.25*np.ceil(len(metrics)/3)),layout='constrained')
    for ax,m in zip(axes.flat,metrics):
        condition='TMS' if m.startswith(('source_','off_','remote','all90')) else 'spontaneous'
        sub=complete[complete.stimulation_condition==condition]
        z=sub.groupby(['b_pA','tau_w_ms'])[m].median().unstack().reindex(index=p['b_pA'],columns=p['tau_w_ms'])
        arr=z.to_numpy(float);im=ax.imshow(np.ma.masked_invalid(arr),origin='lower',aspect='auto',cmap='viridis')
        ax.set(title=m.replace('_',' '),xticks=range(len(p['tau_w_ms'])),xticklabels=p['tau_w_ms'],yticks=range(len(p['b_pA'])),yticklabels=p['b_pA'],xlabel='tau_w (ms)',ylabel='b (pA)')
        fig.colorbar(im,ax=ax,shrink=.8)
    for ax in list(axes.flat)[len(metrics):]:ax.set_visible(False)
    fig.suptitle(title+'\nMedians of available runs; blank = missing; invalid runs retained in raw maps. Single-trial LZ is exploratory.')
    fig.savefig(out/(prefix+'.png'),dpi=220);fig.savefig(out/(prefix+'.svg'));plt.close(fig)

def rank_parameters(frame,p):
    """Ordinal template ranking; labels affect scoring ONLY, never simulations.

No measured empirical targets exist. These scores are a prespecified heuristic,
not a likelihood, a posterior, or evidence of clinical discrimination.
"""
    d=frame[(frame.stimulation_condition=='TMS')&(frame.metrics_status=='complete')].copy()
    if d.empty:return d
    # Parameter preference has modest influence after phenotype scoring.
    prior=((d.b_pA-p['selection']['reference_b_pA'])/80)**2+np.log(d.tau_w_ms/p['selection']['reference_tau_w_ms'])**2
    d['parameter_departure']=prior
    # Scale PCI by its fixed 0..90 component-count context only as a bounded
    # heuristic; this is NOT a published PCI-ST normalization.
    complex_fraction=np.clip(d.all90_PCI_ST/90,0,1)
    off_depth=np.clip(d.source_off_depth_fraction,0,1)
    off_duration=np.clip(d.source_off_duration_ms/300,0,1)
    initial=d.source_initial_response_detected.astype(bool).astype(float)
    d['pathology_index']=(.25*d.slow_switching+.25*off_depth+.25*off_duration
                         +.125*(1-np.clip(d.remote89_recruited_regions/89,0,1))+.125*(1-complex_fraction))
    anchors=d.diagnostic_group.map(p['selection']['ordinal_anchors'])
    # Broad ordinal anchors are declared analyst assumptions, not measured targets.
    d['phenotype_score']=1-abs(d.pathology_index-anchors)-.25*(1-initial)
    d['selection_score']=d.phenotype_score-p['selection']['prior_weight']*prior
    d.loc[d.valid!=True,'selection_score']=np.nan
    # An absent initial response cannot qualify as the evoked DoC phenotype.
    d.loc[~d.source_initial_response_detected.astype(bool),'selection_score']=np.nan
    d['selection_interpretation']='provisional_ordinal_template_no_empirical_fit'
    return d

def transitions(frame,p):
    rows=[];d=frame[(frame.stimulation_condition=='TMS')&(frame.metrics_status=='complete')]
    if 'operational_transition' not in d:return pd.DataFrame()
    for (c,s,tau),group in d.groupby(['diagnostic_group','subject','tau_w_ms']):
        state=group.groupby('b_pA').agg(event=('operational_transition','mean'),valid_fraction=('valid','mean'))
        valid=state[state.valid_fraction==1];positive=valid.index[valid.event>=.5].tolist()
        if positive:
            upper=min(positive);below=[b for b in valid.index if b<upper and valid.loc[b,'event']<.5]
            lower=max(below) if below else None
            censor='left' if upper==min(p['b_pA']) else ('interval' if lower is not None else 'unknown_lower')
        else:upper=None;lower=max(valid.index) if len(valid) else None;censor='right' if len(valid)==len(p['b_pA']) else 'incomplete_or_invalid'
        rows.append(dict(diagnostic_group=c,subject=s,tau_w_ms=tau,bcrit_lower_pA=lower,bcrit_upper_pA=upper,
            censoring=censor,valid_b_values=len(valid),tested_b_values=len(state),
            positive_b_values=';'.join(map(str,positive)),nonmonotonic=bool(positive and any(b>min(positive) and valid.loc[b,'event']<.5 for b in valid.index))))
    return pd.DataFrame(rows)

def select_and_compare(ranked,out,p):
    if ranked.empty:return {}
    cell=ranked.groupby(['diagnostic_group','subject','b_pA','tau_w_ms'],as_index=False).agg(
        selection_score=('selection_score','mean'),phenotype_score=('phenotype_score','mean'),pathology_index=('pathology_index','mean'),
        valid_fraction=('valid','mean'),parameter_departure=('parameter_departure','mean'))
    cell.loc[cell.valid_fraction<1,'selection_score']=np.nan
    required=len(p['b_pA'])*len(p['tau_w_ms'])
    # Never rank an incomplete subject against only the early cells that finished.
    complete=cell.groupby('subject').size();eligible=complete.index[complete==required]
    cell=cell[cell.subject.isin(eligible)];best=[];near=[]
    for s,g in cell.groupby('subject'):
        g=g.dropna(subset=['selection_score'])
        if g.empty:
            best.append(dict(subject=s,status='no_valid_parameter_pair'));continue
        pick=g.sort_values(['selection_score','parameter_departure'],ascending=[False,True]).iloc[0]
        best.append({**pick.to_dict(),'status':'provisional_not_empirical_fit'})
        near.extend(g[g.selection_score>=g.selection_score.max()-p['selection']['near_optimal_tolerance']].to_dict('records'))
    pd.DataFrame(best).to_csv(out/'subject_selected_parameters.csv',index=False)
    pd.DataFrame(near).to_csv(out/'subject_near_optimal_regions.csv',index=False)
    if cell.empty:return dict(complete_subjects=0,selection_ready=False)
    group=cell.groupby(['diagnostic_group','b_pA','tau_w_ms'],as_index=False).agg(score=('selection_score','mean'),valid_subject_fraction=('valid_fraction','mean'),subjects=('subject','nunique'))
    group.loc[group.valid_subject_fraction<1,'score']=np.nan
    group.to_csv(out/'condition_parameter_scores.csv',index=False)
    groupbest=[];groupnear=[];boot=[];rng=np.random.default_rng(p['selection']['seed'])
    for c,g in group.groupby('diagnostic_group'):
        g=g.dropna(subset=['score'])
        if g.empty:continue
        pick=g.sort_values('score',ascending=False).iloc[0];groupbest.append(pick.to_dict())
        groupnear.extend(g[g.score>=g.score.max()-p['selection']['near_optimal_tolerance']].to_dict('records'))
        matrix=cell[cell.diagnostic_group==c].pivot(index='subject',columns=['b_pA','tau_w_ms'],values='selection_score')
        for iteration in range(p['selection']['bootstrap_subjects']):
            ids=rng.integers(0,len(matrix),len(matrix));sample=matrix.iloc[ids]
            means=sample.mean(axis=0).where(sample.notna().all(axis=0))
            if not means.notna().any():continue
            b,tau=means.idxmax();boot.append(dict(diagnostic_group=c,bootstrap=iteration,b_pA=b,tau_w_ms=tau,n_subjects=len(matrix)))
    pd.DataFrame(groupbest).to_csv(out/'condition_selected_parameters.csv',index=False)
    pd.DataFrame(groupnear).to_csv(out/'condition_near_optimal_regions.csv',index=False)
    pd.DataFrame(boot).to_csv(out/'condition_bootstrap_selections.csv',index=False)
    # Common pair must be valid for every included subject; balance groups equally.
    common=group.groupby(['b_pA','tau_w_ms'],as_index=False).agg(score=('score','mean'),cohorts=('score','count'),valid_fraction=('valid_subject_fraction','min'))
    nc=group.diagnostic_group.nunique();common.loc[(common.cohorts<nc)|(common.valid_fraction<1),'score']=np.nan
    common.to_csv(out/'common_parameter_scores.csv',index=False)
    commonbest=common.dropna(subset=['score']).sort_values('score',ascending=False)
    chosen=commonbest.iloc[0].to_dict() if len(commonbest) else None
    # Direct group contrasts at each common tuple, with no group-specific b/tau.
    structure=cell[cell.valid_fraction==1].groupby(['diagnostic_group','b_pA','tau_w_ms']).pathology_index.mean().unstack(0)
    for c in ['mcs','uws','emcs']:
        if c in structure and 'control' in structure:structure[c+'_minus_control']=structure[c]-structure['control']
    structure.to_csv(out/'common_parameter_structural_contrasts.csv')
    # Subject-held-out comparison, fitting templates on training subjects only.
    cv=[]
    for subject in cell.subject.unique():
        test=cell[cell.subject==subject];train=cell[cell.subject!=subject]
        if train[train.diagnostic_group==test.diagnostic_group.iloc[0]].subject.nunique()<1:continue
        tr=train.groupby(['diagnostic_group','b_pA','tau_w_ms']).selection_score.mean().unstack(0)
        tr=tr.where(train.groupby(['diagnostic_group','b_pA','tau_w_ms']).selection_score.count().unstack(0)==train.groupby('diagnostic_group').subject.nunique(),np.nan)
        shared=tr.mean(axis=1).where(tr.notna().all(axis=1));specific=tr[test.diagnostic_group.iloc[0]]
        for name,series in [('common',shared),('condition_specific',specific)]:
            if not series.notna().any():continue
            b,tau=series.idxmax();z=test[(test.b_pA==b)&(test.tau_w_ms==tau)].iloc[0]
            cv.append(dict(subject=subject,diagnostic_group=z.diagnostic_group,solution=name,b_pA=b,tau_w_ms=tau,heldout_template_score=z.selection_score,valid_fraction=z.valid_fraction))
    pd.DataFrame(cv).to_csv(out/'heldout_common_vs_condition.csv',index=False)
    result=dict(complete_subjects=int(len(eligible)),common_selected=chosen,condition_selected=groupbest,
                interpretation='Provisional ordinal templates. Bootstrap measures subject-sampling dispersion, not empirical-fit confidence; one subject per group gives no meaningful population uncertainty.')
    (out/'selected_parameters.json').write_text(json.dumps(result,indent=2));return result

def report(out):
    out=Path(out);frame,manifest,status=collect(out);p=manifest['protocol'];plots=out/'figures';plots.mkdir(exist_ok=True)
    complete=frame[frame.metrics_status=='complete']
    if complete.empty:
        (out/'REPORT.md').write_text('# Adaptation sweep\n\nNo successfully measured parameter pairs yet. See metrics.csv and table_status.json.\n');return
    for c,g in complete.groupby('diagnostic_group'):heatmaps(g,plots,p,f'{c}: {g.subject.nunique()} subjects; {status["completed_pairs"]}/{status["total_pairs"]} campaign pairs','group_'+c)
    # Subject maps only for complete grids, avoiding a deluge of unfinished panels.
    for s,g in complete.groupby('subject'):
        if len(g[['b_pA','tau_w_ms']].drop_duplicates())==len(p['b_pA'])*len(p['tau_w_ms']):heatmaps(g,plots,p,s,'subject_'+s)
    ranked=rank_parameters(frame,p);ranked.to_csv(out/'phenotype_scores.csv',index=False)
    trans=transitions(frame,p);trans.to_csv(out/'bcrit_intervals.csv',index=False)
    selection=select_and_compare(ranked,out,p)
    t=complete[complete.stimulation_condition=='TMS'];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,(x,y) in zip(axes.flat,[('source_off_duration_ms','all90_PCI_ST'),('source_off_depth_fraction','remote89_recruited_regions'),('source_off_duration_ms','remote89_recruited_regions'),('source_off_depth_fraction','all90_LZ_single_trial_sensitivity')]):
        for c,g in t[t.valid==True].groupby('diagnostic_group'):ax.scatter(g[x],g[y],s=8,alpha=.35,label=c)
        ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '));ax.legend(fontsize=7)
    fig.suptitle('Covariation across valid grid cells; repeated cells within subjects are dependent')
    fig.savefig(plots/'off_propagation_pci.png',dpi=220);fig.savefig(plots/'off_propagation_pci.svg');plt.close(fig)
    if not trans.empty:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        for c,g in trans.groupby('diagnostic_group'):
            for s,ss in g.groupby('subject'):ax.plot(ss.tau_w_ms,ss.bcrit_upper_pA,'o-',alpha=.25,lw=.7,label=c if s==g.subject.iloc[0] else None)
        ax.set(xlabel='tau_w (ms)',ylabel='First observed transition b (pA)',title='Operational slow-switching / evoked-OFF onset; missing transitions retained as censoring in table');ax.legend()
        fig.savefig(plots/'bcrit_vs_tau.png',dpi=220);plt.close(fig)
    report_text=f'''# Adaptation sweep — {'complete' if status['analysis_complete'] else 'IN PROGRESS'}

Completed {status['completed_pairs']} of {status['total_pairs']} parameter pairs.
All requested pairs and both recording conditions are represented in metrics.csv;
unfinished rows are explicitly marked pending. Invalid and failed pairs remain.

No individual empirical targets were supplied. Selected parameters, if available,
are provisional ordinal phenotype rankings, not patient fits. Simulation parameters
are identical across groups at each grid point. Group labels enter only the declared
template comparison. A clinical ordering obtained after this selection cannot serve
as independent validation of that ordering.

Native trial-averaged PCI-LZ is undefined for one trial. The separate single-trial
LZ sensitivity and its recruitment masks are exploratory. PCI-ST is calculated by
the existing implementation. Repeated trials and longer stationary baselines are
required to validate selected settings. Slow switching is not proof of mathematical
bistability. Empirical EEG OFF periods are only approximated by rate suppression here.

See bcrit_intervals.csv for interval/right/left censoring, parameter score tables for
common versus group-specific solutions, near-optimal sets and bootstrap selections.
Current complete-subject count for parameter selection: {selection.get('complete_subjects',0)}.
'''
    (out/'REPORT.md').write_text(report_text)
    print(json.dumps(status),flush=True)
