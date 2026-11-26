"""Isolated AdEx single-neuron TF refit. No production-model modifications."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc
from brian2 import (start_scope, seed, defaultclock, NeuronGroup, PoissonGroup,
                    Synapses, SpikeMonitor, StateMonitor, ms, mV, nS, pA, pF,
                    Hz, second)

def inputs(n):
    # Front-load the low-input regime; include 0 explicitly.
    u = np.linspace(0, 1, n)
    return np.r_[0.0, 0.1 + 199.9 * u**3]

def simulate(pop, leak, fe, fi, duration, random_seed):
    start_scope()
    seed(int(random_seed))
    defaultclock.dt = 0.1 * ms
    el = -64.0 if pop == "RS" else -65.0
    delta = 2.0 if pop == "RS" else 0.5
    b = 30.0 if pop == "RS" else 0.0
    # gK + gNa = 10 nS and weighted reversal equals EL.
    gna = 10.0 * (el + 90.0) / 140.0
    gk = 10.0 - gna
    leak_term = ("gL*(EL-v)" if leak == "original"
                 else "gK*(EK-v)+gNa*(ENa-v)")
    eq = f"""
    dv/dt = ({leak_term}+gL*Delta*exp((v-VT)/Delta)
             +Ge*(Ee-v)+Gi*(Ei-v)-w)/C : volt (unless refractory)
    dw/dt = -w/tauw : amp
    dGe/dt = -Ge/taue : siemens
    dGi/dt = -Gi/taui : siemens
    gL : siemens
    gK : siemens
    gNa : siemens
    EL : volt
    EK : volt
    ENa : volt
    Ee : volt
    Ei : volt
    C : farad
    Delta : volt
    VT : volt
    Vr : volt
    Vcut : volt
    tauw : second
    taue : second
    taui : second
    b : amp
    """
    cell = NeuronGroup(1, eq, threshold="v>Vcut", reset="v=Vr;w+=b",
                       refractory=5*ms, method="heun")
    cell.gL=10*nS;cell.gK=gk*nS;cell.gNa=gna*nS
    cell.EL=el*mV;cell.EK=-90*mV;cell.ENa=50*mV
    cell.Ee=0*mV;cell.Ei=-80*mV;cell.C=200*pF
    cell.Delta=delta*mV;cell.VT=-50*mV;cell.Vr=-65*mV
    cell.Vcut=-30*mV;cell.tauw=500*ms
    cell.taue=5*ms;cell.taui=5*ms;cell.b=b*pA
    cell.v=-60*mV;cell.w=0*pA;cell.Ge=0*nS;cell.Gi=0*nS
    exc=PoissonGroup(400, rates=fe*Hz)
    inh=PoissonGroup(100, rates=fi*Hz)
    se=Synapses(exc,cell,on_pre="Ge_post+=1.5*nS")
    si=Synapses(inh,cell,on_pre="Gi_post+=5*nS")
    se.connect();si.connect()
    spikes=SpikeMonitor(cell)
    adap=StateMonitor(cell,"w",record=True,dt=10*ms)
    from brian2 import run
    run(duration*second)
    cutoff=duration/2
    rate=float(np.sum(np.asarray(spikes.t/second)>=cutoff)/(duration-cutoff))
    t=np.asarray(adap.t/second)
    w=float(np.mean(np.asarray(adap.w[0]/pA)[t>=cutoff]))
    return rate,w

def features(fe,fi,w):
    # Analytic voltage moments in mV, ms, nS, pA, pF.
    ge=400*fe*5*1.5/1000
    gi=100*fi*5*5/1000
    g=10+ge+gi
    tm=200/g
    # Note: EL differs between RS and FS; supplied in fit_features below.
    return ge,gi,g,tm

def fit_features(fe,fi,w,el):
    ge,gi,g,tm=features(fe,fi,w)
    muv=(gi*(-80)+10*el-w)/g
    ue=1.5/g*(0-muv);ui=5/g*(-80-muv)
    re=400*fe/1000;ri=100*fi/1000
    var=(re*(ue*5)**2/(2*(tm+5))+ri*(ui*5)**2/(2*(tm+5)))
    sig=np.sqrt(np.maximum(var,1e-12))
    # Identical 5-ms synaptic time constants imply Tv=tm+5.
    tv=tm+5
    v=(muv+60)/10;s=(sig-4)/6;t=(tv*10/200-0.5)
    x=np.column_stack([np.ones_like(v),v,s,t,v*v,s*s,t*t,v*s,v*t,s*t])
    return x,muv,sig,tv

def predict(p,x,muv,sig,tv):
    threshold=1000*(x@p)
    return 1000*erfc((threshold-muv)/(np.sqrt(2)*sig))/(2*tv)

def score(actual,pred):
    actual=np.asarray(actual);pred=np.asarray(pred)
    out={"rmse_hz":float(np.sqrt(np.mean((pred-actual)**2))),
         "mae_hz":float(np.mean(abs(pred-actual))),
         "log1p_rmse":float(np.sqrt(np.mean((np.log1p(pred)-np.log1p(actual))**2)))}
    for name,mask in [("low_0_2",actual<2),("mid_2_50",(actual>=2)&(actual<50)),
                      ("high_50_plus",actual>=50)]:
        out[name]={"n":int(mask.sum()),"mae_hz":float(np.mean(abs(pred[mask]-actual[mask])))
                   if mask.any() else None}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["original","split","both"],default="both")
    ap.add_argument("--n-grid",type=int,default=15)
    ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--duration-s",type=float,default=12)
    ap.add_argument("--output",default="experiments/adex_tf_refit/results")
    a=ap.parse_args()
    if a.n_grid<2 or a.seeds<1 or a.duration_s<=0:ap.error("invalid grid/seeds/duration")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import csv
    grid=inputs(a.n_grid)
    modes=["original","split"] if a.mode=="both" else [a.mode]
    rows=[]
    for mode in modes:
        for pop in ["RS","FS"]:
            for fe in grid:
                for fi in grid:
                    for split in ["train","test"]:
                        for rep in range(a.seeds):
                            # Distinct held-out stochastic realisations.
                            seed_id=100000*int(mode=="split")+10000*int(pop=="FS")+(
                                5000 if split=="test" else 0)+rep
                            rate,w=simulate(pop,mode,float(fe),float(fi),
                                            a.duration_s,seed_id)
                            rows.append(dict(mode=mode,pop=pop,fe=float(fe),fi=float(fi),
                                             split=split,seed=seed_id,rate_hz=rate,w_pA=w))
    with (out/"observations.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={}
    for mode in modes:
        for pop in ["RS","FS"]:
            subset=[r for r in rows if r["mode"]==mode and r["pop"]==pop]
            tr=[r for r in subset if r["split"]=="train"]
            te=[r for r in subset if r["split"]=="test"]
            def data(rs):
                fe=np.array([r["fe"] for r in rs]);fi=np.array([r["fi"] for r in rs])
                w=np.array([r["w_pA"] for r in rs]);y=np.array([r["rate_hz"] for r in rs])
                return (*fit_features(fe,fi,w,-64 if pop=="RS" else -65),y)
            x,mu,sig,tv,y=data(tr)
            xt,mut,sigt,tvt,yt=data(te)
            p0=np.array([-0.05,0,0,0,0,0,0,0,0,0],dtype=float)
            # Use log1p to retain zero-rate observations and balance low/high rates.
            def residual(p):
                pred=predict(p,x,mu,sig,tv)
                return np.r_[np.log1p(pred)-np.log1p(y), 0.01*(p-p0)]
            res=least_squares(residual,p0,max_nfev=2500)
            pred=predict(res.x,xt,mut,sigt,tvt)
            key=f"{mode}_{pop}"
            np.save(out/f"{key}_coefficients.npy",res.x)
            report[key]={"coefficients":res.x.tolist(),"optimiser_success":bool(res.success),
                         "optimiser_message":res.message,"held_out":score(yt,pred),
                         "training_n":len(tr),"held_out_n":len(te),
                         "note":"Not validated for whole-brain stimulation or equilibrium stability"}
            with (out/f"{key}_predictions.csv").open("w",newline="") as f:
                wr=csv.writer(f);wr.writerow(["fe_hz","fi_hz","observed_hz","predicted_hz"])
                wr.writerows(zip([r["fe"] for r in te],[r["fi"] for r in te],yt,pred))
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v["held_out"] for k,v in report.items()},indent=2))
if __name__=="__main__":main()
