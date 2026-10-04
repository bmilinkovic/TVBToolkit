"""Diagnostic 1: unchanged native TFs and native finite-difference derivatives."""
import json
from run_adex_impulse_response import ROOT, BASE_PARAMETER_MODEL_NEW
import numpy as np
from tvbtoolkit.whole_brain.legacy_engine.src.Zerlaut import Zerlaut_adaptation_second_order
from tvbtoolkit.whole_brain.legacy_engine.parameter.parameter_M_Berlin_new import Parameters

OUT=ROOT/'results/adex_staged_diagnostics/diagnostic1'

def native_model():
    p=Parameters().parameter_model
    p.update(BASE_PARAMETER_MODEL_NEW)
    m=Zerlaut_adaptation_second_order()
    for k,v in p.items():
        if hasattr(m,k) and k not in ('initial_condition','noise_alpha','shared_noise_mode'):
            setattr(m,k,np.asarray(v))
    return m

def evaluate(m,e,i,ext,we,wi,h=1e-7):
    outputs=[]
    for tf,w,inh in [(m.TF_excitatory,we,m.external_input_ex_in),(m.TF_inhibitory,wi,m.external_input_in_in)]:
        def f(x,y):return np.asarray(tf(x,y,ext,inh,w)).reshape(np.broadcast_arrays(x,y)[0].shape)
        f0=f(e,i);ep=f(e+h,i);em=f(e-h,i);ip=f(e,i+h);im=f(e,i-h)
        outputs.append(np.array([f0,(ep-em)/(2*h),(ip-im)/(2*h),
            (ep-2*f0+em)/h**2,(f(e+h,i+h)-f(e+h,i-h)-f(e-h,i+h)+f(e-h,i-h))/(4*h*h),
            (ip-2*f0+im)/h**2]))
    return np.array(outputs)

def main():
    assert json.loads((ROOT/'results/adex_staged_diagnostics/stage_a/completion.json').read_text())['complete']
    OUT.mkdir(parents=True,exist_ok=True)
    m=native_model()
    rates_e=np.unique(np.r_[0,np.linspace(.1,30,61),np.linspace(35,200,34)])
    rates_i=np.unique(np.r_[0,np.linspace(.1,30,61),np.linspace(35,250,44)])
    adaptations=np.array([-100,0,50,100,250,500,1000,3000,6000.])
    external=np.array([0,.1,.315,1,3,10,30,100,300,1000,10000.])
    e,i=np.meshgrid(rates_e/1000,rates_i/1000,indexing='ij')
    rows=[];slices=None
    for ext in external:
        maps=np.array([evaluate(m,e,i,ext/1000,w,w) for w in adaptations])
        np.savez_compressed(OUT/f'tf_external_{ext:g}Hz.npz',values=maps,
            excitatory_input_hz=rates_e,inhibitory_input_hz=rates_i,adaptation_pA=adaptations,
            field_names=np.array(['F_khz','dF_dE','dF_dI','d2F_dE2_per_khz','d2F_dEdI_per_khz','d2F_dI2_per_khz']),
            original_fit_domain_mask=np.full(e.shape,-1,dtype=np.int8))
        for population in range(2):
            a=maps[:,population]
            rows.append(dict(external_hz=float(ext),population=['E','I'][population],
                max_rate_hz=float(np.nanmax(a[:,0])*1000),nonfinite_count=int((~np.isfinite(a)).sum()),
                fraction_FT_above_one=float((a[:,0]*20>1).mean()),fraction_rate_above_180hz=float((a[:,0]>.18).mean()),
                max_abs_first_derivative=float(np.nanmax(np.abs(a[:,1:3]))),
                max_abs_second_derivative_per_hz=float(np.nanmax(np.abs(a[:,3:]))/1000)))
        if ext==.315:slices=maps[2]
    metadata=dict(status='evaluated; exact training domain unresolved',
        coefficients_e=np.asarray(m.P_e).tolist(),coefficients_i=np.asarray(m.P_i).tolist(),
        original_fit_domain=None,extrapolation_status='unknown, not silently treated as inside fit',domain_mask_unknown=-1,
        related_coefficients='/Users/borjan/CNRS/projects/TVBSim/brian_MF/Tf_calc/data/dV_git_{RS,FS}-cell_CONFIG1_fit.npy',
        provenance_note='Related 11-entry files match current rounded 10-entry vectors after excluding entry 4 (~0.001); exact training grid/fit mask not established. No coefficient changed.',
        unverified_other_script_grid_hz=[.1,30],
        unverified_grid_warning='TVBSim tf_simulation.py defaults, NOT confirmed original fitting domain for these vectors',
        scan=dict(E_hz=rates_e.tolist(),I_hz=rates_i.tolist(),external_hz=external.tolist(),adaptation_pA=adaptations.tolist()),
        notes=['TF_E and TF_I evaluated separately; adaptation axis applied independently to each population.',
               'In subsequent simulations inhibitory adaptation remains its native value (zero).',
               'Extending scan to 10000 Hz external input covers previously observed saturated invalid states; not a physiological claim.',
               'No equations, fits or coefficients changed; no trajectory simulations in Diagnostic 1.'],summary=rows)
    (OUT/'report.json').write_text(json.dumps(metadata,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for p in range(2):
        for col,values in enumerate([slices[p,0]*1000,np.max(np.abs(slices[p,1:3]),axis=0),np.max(np.abs(slices[p,3:]),axis=0)/1000]):
            im=axes[p,col].pcolormesh(rates_e,rates_i,values.T,shading='auto',
                 norm=None if col==0 else LogNorm(vmin=1e-4,vmax=max(1e-3,float(np.nanmax(values)))))
            axes[p,col].set(xlabel='Local excitatory input (Hz)',ylabel='Local inhibitory input (Hz)',
                title=f"TF {['E','I'][p]}: {['output Hz','max |first derivative|','max |curvature| per Hz'][col]}")
            fig.colorbar(im,ax=axes[p,col])
    fig.suptitle('Unchanged transfer functions: external drive 0.315 Hz, adaptation 50 pA\nExact original fitting domain unresolved; no inferred training boundary shown')
    fig.savefig(OUT/'transfer_maps.png',dpi=200);fig.savefig(OUT/'transfer_maps.pdf');plt.close(fig)
    (OUT/'completion.json').write_text(json.dumps(dict(evaluation_complete=True,training_domain_verified=False,
        limitations='Cannot definitively label extrapolation without exact training provenance',no_refit=True),indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
