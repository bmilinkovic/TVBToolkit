"""First-order-only local equilibrium audit; never runs a second-order model.

Uses the existing subject configuration and native first-order RHS. Fixed
points refer to the zero-OU drift, not an equilibrium of a noisy sample path.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from run_adex_impulse_response import ROOT, build_subject_config
from stage_b_dynamics import DATA
from tvbtoolkit.whole_brain.simulation import configure_adex_simulator

OUT=ROOT/'results/first_order_adex_validation'

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value

def configure(b=5., drive_hz=.315, dt=.1, method='Heun', isolated=True, seed=1230263469,
              cohort='control',subject='c0013',coupling=0.):
    cfg,labels,_=build_subject_config(cohort,subject,b,DATA,coupling,tuple(range(5)))
    cfg=replace(cfg,zerlaut_order=1,dt_ms=dt)
    pm=cfg.parameter_overrides['parameter_model']
    pm.pop('derivative_scale',None)
    for k in ['C_ee','C_ei','C_ii']:
        pm.get('initial_condition',{}).pop(k,None)
    pm.update(external_input_ex_ex=drive_hz/1000.,external_input_in_ex=drive_hz/1000.)
    cfg.parameter_overrides['parameter_integrator']={'type':method}
    if isolated:
        cfg=replace(cfg,weights=np.zeros((1,1)),tract_lengths=np.zeros((1,1)),coupling_strength=0.)
        labels=np.array(['isolated_EI'])
    sim=configure_adex_simulator(cfg,seed)
    if tuple(sim.model.state_variables)!=('E','I','W_e','W_i','noise'):
        raise AssertionError('Only first-order five-state model permitted')
    for key in ['state_variable_boundaries','_integration_state_variable_boundaries',
                'clamped_state_variable_values','_clamped_integration_state_variable_values']:
        setattr(sim.integrator,key,None)
    return sim,cfg,labels

def scalar(m,key):return float(np.asarray(getattr(m,key)).ravel()[0])

def state_at_rates(m,rates_hz):
    if scalar(m,'a_e')!=0 or scalar(m,'a_i')!=0:
        raise ValueError('Reduced equilibrium search assumes a_e=a_i=0')
    e,i=np.asarray(rates_hz)/1000.
    return np.array([e,i,scalar(m,'tau_w_e')*scalar(m,'b_e')*e,
                     scalar(m,'tau_w_i')*scalar(m,'b_i')*i,0.])[:,None,None]

def residual(m,r):
    if np.any(np.asarray(r)<0):
        # Native TF is evaluated only at physical rates; bounded root transform
        # below supplies nonnegative rates.
        raise ValueError('negative rate')
    return m.dfun(state_at_rates(m,r),np.zeros((1,1,1)))[:2,0,0]*scalar(m,'T')*1000.

def equilibria(m):
    solutions=[]
    # Log coordinates allow near-zero roots without negative TF evaluations.
    for e in [.001,.1,1,5,20,60,120,195,230]:
        for i in [.1,5,30,100,195]:
            def f(z):
                if np.any(abs(z)>30):return np.full(2,1e6)
                with np.errstate(all='ignore'):return residual(m,np.exp(z))
            sol=root(f,np.log([e,i]),method='hybr',options={'xtol':1e-8})
            if not np.isfinite(sol.x).all() or np.any(abs(sol.x)>30):continue
            rates=np.exp(sol.x)
            if max(abs(f(sol.x)))>1e-5 or max(rates)>1000:continue
            if any(np.linalg.norm(rates-s['rates_hz'])<.001 for s in solutions):continue
            x=state_at_rates(m,rates);jac=np.empty((4,4))
            # Full E,I,We,Wi drift Jacobian at fixed OU=0. Noise is exogenous;
            # clipping makes the OU derivative nonsmooth at zero long-range drive.
            for k,h in enumerate([1e-8,1e-8,1e-3,1e-3]):
                xp=x.copy();xm=x.copy();xp[k]+=h;xm[k]-=h
                jac[:,k]=((m.dfun(xp,np.zeros((1,1,1)))-m.dfun(xm,np.zeros((1,1,1))))/(2*h))[:4,0,0]
            eig=np.linalg.eigvals(jac)
            solutions.append(dict(rates_hz=rates,adaptation_pa=x[2:4,0,0],
                residual_hz=f(sol.x),jacobian=jac,eigen_real_per_ms=eig.real,
                eigen_imag_per_ms=eig.imag,stable=bool(eig.real.max()<-1e-8)))
    return solutions

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    drives=[0.,.1,.315,.5,1.,2.,5.,10.,20.]
    rows=[]
    for b in [5.,55.]:
        for drive in drives:
            sim,_,_=configure(b,drive);m=sim.model
            sols=equilibria(m)
            rows.append(clean(dict(b_pA=b,drive_hz=drive,equilibria=sols)))
            print(b,drive,[(s['rates_hz'].round(3).tolist(),s['stable']) for s in sols],flush=True)
            (OUT/'local_equilibria.json').write_text(json.dumps(rows,indent=2))
    sim,cfg,_=configure();m=sim.model
    keys=['T','g_L','C_m','E_L_e','E_L_i','E_e','E_i','Q_e','Q_i','Q_i_e',
          'tau_e_e','tau_e_i','tau_i','N_tot','g','p_connect_e','p_connect_i',
          'K_ext_e','K_ext_i','b_e','b_i','a_e','a_i','tau_w_e','tau_w_i',
          'tau_OU','weight_noise','P_e','P_i','external_input_ex_ex','external_input_in_ex',
          'external_input_ex_in','external_input_in_in']
    (OUT/'executed_parameters.json').write_text(json.dumps(clean({k:getattr(m,k) for k in keys} | {
        'model_class':str(type(m)),'integrator':str(type(sim.integrator)),
        'dt_ms':cfg.dt_ms,'noise_nsig':sim.integrator.noise.nsig,
        'initial_state':sim.current_state,'SC_scalar':122.63607788085938}),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row in rows:
        ax=axes[0 if row['b_pA']==5 else 1]
        for s in row['equilibria']:
            color='#287caf' if s['stable'] else '#c44330'
            for k in range(2):ax[k].scatter(row['drive_hz'],s['rates_hz'][k],color=color,s=18)
            ax[2].scatter(row['drive_hz'],max(s['eigen_real_per_ms']),color=color,s=18)
    for j,axrow in enumerate(axes):
        for k,ax in enumerate(axrow):ax.set(xlabel='External drive (Hz)',ylabel=['E equilibrium (Hz)','I equilibrium (Hz)','Largest real eigenvalue (1/ms)'][k],title=f'b={[5,55][j]} pA')
        axrow[2].axhline(0,color='k',lw=.5)
    fig.suptitle('First-order zero-OU drift equilibria: blue stable, red unstable\nMultistart root search; incomplete branches are possible')
    fig.savefig(OUT/'local_fixed_points.png',dpi=220)
    plt.close(fig)
    # At equilibria W depends on E; these are adaptation-equilibrated nullclines,
    # not a transient phase portrait with frozen W.
    for b in [5.,55.]:
        sim,_,_=configure(b);m=sim.model
        axis=np.linspace(.001,220,221);ee,ii=np.meshgrid(axis,axis)
        e=ee.ravel()/1000;i=ii.ravel()/1000;w=scalar(m,'tau_w_e')*b*e
        ext=np.full_like(e,.000315);zero=np.zeros_like(e)
        fe=m.TF_excitatory(e,i,ext,zero,w)*1000
        fi=m.TF_inhibitory(e,i,ext,zero,zero)*1000
        np.savez_compressed(OUT/f'TF_surface_b{b:g}.npz',E_hz=ee,I_hz=ii,We_pa=w.reshape(ee.shape),FE_hz=fe.reshape(ee.shape),FI_hz=fi.reshape(ee.shape))
        fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ax,f,label in zip(axs[:2],[fe,fi],['FE','FI']):
            im=ax.pcolormesh(ee,ii,f.reshape(ee.shape),vmin=0,vmax=200,cmap='viridis');fig.colorbar(im,ax=ax,label='Hz');ax.set_title(label)
        axs[2].contour(ee,ii,fe.reshape(ee.shape)-ee,levels=[0],colors=['blue'])
        axs[2].contour(ee,ii,fi.reshape(ee.shape)-ii,levels=[0],colors=['orange'])
        axs[2].set_title('Nullclines: E blue, I orange')
        for ax in axs:ax.set(xlabel='E (Hz)',ylabel='I (Hz)')
        fig.suptitle(f'First order, b={b:g} pA; W=tau_w*b*E, external drive 0.315 Hz')
        fig.savefig(OUT/f'TF_nullclines_b{b:g}.png',dpi=220);plt.close(fig)

if __name__=='__main__':main()
