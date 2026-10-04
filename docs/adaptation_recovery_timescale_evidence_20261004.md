# Adaptation strength versus recovery timescale

There is evidence that adaptation-related kinetics and duration vary; a single
universal time constant is a modelling simplification. This is not equivalent to
evidence for a particular AdEx tau_w in each diagnostic group.

Primary cellular evidence: Descalzo, Gallego and Sanchez-Vives (2014), *Adaptation
in the Visual Cortex: Influence of Membrane Trajectory and Neuronal Firing Pattern
on Slow Afterpotentials*, compared stimuli producing similar spike counts.
Afterhyperpolarization duration and amplitude depended on firing pattern and
membrane trajectory, and adaptation developed with different time courses.
These were cortical animal experiments, not measurements of AdEx tau_w in DoC.
Their prolonged-stimulation afterpotentials last much longer than our 500-ms
effective adaptation timescale; do not transplant those values into this model.
https://pmc.ncbi.nlm.nih.gov/articles/PMC4224415/

Human evidence: Rosanova et al. (2018) linked post-TMS OFF periods to disrupted
causal interactions in UWS. This supports measuring suppression duration and
recovery, but does not identify an intrinsic adaptation decay constant separately
from excitation, inhibition, recurrent activity or strength of adaptation.
https://www.nature.com/articles/s41467-018-06871-1

Mechanistic interpretation in our model (a_e=0):

    dW/dt = -W/tau_w + b E

If E is approximately zero during an OFF period, W(t) approximately equals
W0*exp(-t/tau_w). If reactivation occurs near an effective threshold Wcrit, then

    t_recovery approximately tau_w * log(W0/Wcrit).

This illustrates why longer OFF periods can arise from larger W0 (including a
larger b), slower decay, or altered network-dependent Wcrit. The approximation
is explanatory, not a fitted recovery law for a recurrent stochastic network.
OFF duration is not interchangeable with tau_w. Nor do UP/DOWN transitions by
themselves establish two coexisting stable attractors.

Decision: user-selected b = 5/15/30/60 pA for control/EMCS/MCS/UWS, tau_w=500 ms
for all. Test trajectories, episode durations and evoked recovery at these values;
retain tau_w uncertainty for later inference/sensitivity work. Do not claim that
the literature establishes this exact condition-to-parameter mapping.

The first-order gNa/gK workflow and cluster instructions are documented in
hpc/README_first_order_gnak.md. VBI changes are deferred to the next task.
