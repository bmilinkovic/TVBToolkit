"""Run directly: python experiments/adaptation_b_tau/test_metrics.py."""
import json
from pathlib import Path
import numpy as np
from metrics import durations, state_metrics, off_metrics, validity

p=json.loads(Path(__file__).with_name('protocol.json').read_text())
assert durations([1,1,0,1,1,0,1],1).tolist()==[2.]
tonic,_=state_metrics(np.full(6000,5.),1,p)
assert not tonic['bimodal'] and tonic['up_fraction']==1
alternating=np.tile(np.r_[np.full(500,.2),np.full(500,10.)],6)
switching,_=state_metrics(alternating,1,p)
assert switching['bimodal'] and switching['slow_switching']
assert .4<switching['down_fraction']<.6
baseline=np.full((6000,1),5.)
response=np.full((2000,1),5.);response[:30]=15;response[80:280]=.1
off=off_metrics(baseline,response,1,p)[0]
assert off['off_present'] and 170<off['off_duration_ms']<220
assert off['off_depth_fraction']>.9
response[:30]=5
assert not off_metrics(baseline,response,1,p)[0]['off_present']
state=np.zeros((2000,3,1));state[:,0]=.005;state[:,1]=.01
assert validity(state,1,p)['valid']
state[:,0]=.199
assert validity(state,1,p)['PFP'] and not validity(state,1,p)['valid']
state[:,0]=-.001
assert 'negative_rates' in validity(state,1,p)['invalid_reasons']
print('PASS: censoring, tonic/switching states, evoked OFF, validity and finite PFP checks')
