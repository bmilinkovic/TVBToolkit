"""Direct smoke tests without launching a production campaign."""
import json
import numpy as np
from analyze import epoch,HERE
from metrics import pci_metrics
p=json.loads((HERE/'protocol.json').read_text())
assert p['b_pA']==dict(control=5,emcs=15,mcs=30,uws=60)
assert 'coma' not in p['b_pA'] and p['trials']==50
t=np.arange(3000.,10001.);state=np.zeros((len(t),3,90));state[:,0]=t[:,None]/1000
aligned=epoch(t,state,3400.1)
np.testing.assert_allclose(aligned[0],3400.1+np.arange(-400,300))
try:epoch(t,state,3300)
except ValueError:pass
else:raise AssertionError('Incomplete baseline accepted')
# Use the actual production trial count with cheap bootstrap/ST settings only
# for API/shape validation; these override values do NOT enter production.
p['source_index']=9;p['pci']['bootstrap']=20;p['pci']['n_steps']=10
rng=np.random.default_rng(14);x=rng.normal(5,.1,(50,90,700))
x[:,9,400:430]+=3
score,arrays=pci_metrics(x,p)
assert score['PCI_LZ_native_available'] and score['PCI_trials']==50
assert arrays['all90_mask'].shape==(90,100)
assert arrays['remote89_mask'].shape==(89,100)
print('PASS: cohort lock, 50 trials, onset alignment, boundary checks, ensemble PCI API')
