"""Zero-occupancy implementation checks; stochastic, first order only."""
import argparse
import json
from pathlib import Path
import numpy as np
from simulate import HERE,ROOT,configuration,simulator,write_json

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    p=json.loads((HERE/'protocol.json').read_text());rows=[]
    for c,s in [('control','c0005'),('emcs','e0009'),('mcs','m0060'),('uws','u0033')]:
        j=dict(cohort=c,subject=s,occupancy=0)
        original,labels,k,_=configuration(j,p,ROOT/'data/doc_data/converted_structural_invnodevol_control_max',split=False)
        split,_,_,_=configuration(j,p,ROOT/'data/doc_data/converted_structural_invnodevol_control_max',split=True)
        first=simulator(original,1900);second=simulator(split,1900)
        x=first.current_state.copy();coupling=np.full((1,90,1),.0002)
        rhs1=first.model.dfun(x.copy(),coupling.copy());rhs2=second.model.dfun(x.copy(),coupling.copy())
        np.testing.assert_allclose(rhs1,rhs2,rtol=1e-10,atol=1e-12)
        t,y=first.run(simulation_length=200)[0];tt,yy=second.run(simulation_length=200)[0]
        np.testing.assert_allclose(y,yy,rtol=1e-7,atol=1e-9)
        rows.append(dict(cohort=c,max_rhs_difference=float(abs(rhs1-rhs2).max()),max_trajectory_difference=float(abs(y-yy).max())))
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,dict(check='matched-noise 200-ms trajectory and RHS at zero occupancy',results=rows))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
