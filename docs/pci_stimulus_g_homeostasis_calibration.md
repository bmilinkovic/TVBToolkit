# PCI stimulus, global coupling, and homeostasis calibration

This calibration must be completed before another full 189-subject PCI run.
It is exploratory and its two subjects must not be treated as an inferential
sample.

## What was wrong with the earlier stimulus sweep

The embedded analysis in `notebooks/07_pci_stim_sensitivity_tutorial.ipynb`
used one control connectome, set `b_e=35 pA`, targeted legacy zero-based index
18 rather than anatomically resolving left supplementary motor area, used two
trials, and labelled every setting as spreading to all 90 regions.  Its
one-hertz spread criterion was therefore saturated and could not select a
pulse on the basis of spatial propagation.

## New staged calibration

`scripts/calibrate_pci_stimulus_g_homeostasis.py` does the following:

1. Selects one control and one UWS subject (or accepts explicit subjects).
2. Resolves `Supp_Motor_Area_L` against the converted dataset atlas; no numeric
   target is assumed.
3. Aligns the 5-HT2A map to that same label order.
4. Tests square, raised-cosine, and finite Gaussian pulses across duration and
   peak input, at occupancy 0 and 0.766 with matched random seeds.
   Each trial is cut relative to its stimulation onset; matched trials are
   time-locked and averaged before pulse safety, propagation, and recovery are
   scored. Per-trial metrics are also retained as diagnostics.
5. Rejects pulses reaching the configured 100-Hz safety threshold, then ranks
   remaining pulses by visible amplitude, non-saturated propagation, and late
   recovery.
6. Sweeps global coupling `G` with the selected pulse. Both subjects use the
   same `b_e=5 pA` by default. The diagnosis-specific `b_e` gradient is disabled,
   so any control--UWS difference arises from connectome damage,
   receptor-weighted modulation, and their interaction with `G`, not a
   hard-coded diagnosis shift.
7. Compares no homeostasis, pre-fitted/frozen inhibition, and online
   homeostasis acting within the post-stimulation trial.

The rate monitor uses an exact 3-ms period (333.33 Hz). An exact 340-Hz monitor
cannot be represented with the model's 0.1-ms integration step; 3 ms is close,
reproducible, and lies between the principal empirical PCI sampling rates.

## Homeostatic rule and its interpretation

Coronel-Oliveros et al. use a Jansen-Rit model and learn the inhibitory feedback
parameter `C4` with a rate rule proportional to

`inhibitory rate * (excitatory rate - target rate)`.

Their exact code is not an AdEx implementation. In this toolkit the analogous
inhibitory efficacy is a new inhibitory-to-excitatory quantal conductance
`Q_i_e`. It is separate from `Q_i`, so the rule alters inhibitory feedback onto
excitatory/pyramidal cells without also changing inhibitory-to-inhibitory
input. For regional scale `h = Q_i_e / Q_i_e,baseline`, this implementation uses

`d log(h) = (dt/tau) * (rI/rho) * ((rE-rho)/rho) * h^(beta-1)`.

Thus, a region above its target strengthens inhibition, a region below target
weakens it, and no inhibitory activity produces no learning. Log-space
updates preserve positivity, and each update is capped before applying bounds
of 0.25--4 times baseline `Q_i_e`. The normalization by the target rate is the
explicit dimensional translation needed because the AdEx model stores rates
in kHz and `Q_i` in nS.

The default target is the regional unstimulated baseline for each subject,
occupancy, and `G` condition, with a 0.5-Hz numerical floor. Consequently the
online error is approximately zero before stimulation and emerges when the
pulse produces persistent excess firing. This preserves each condition's
operating point instead of forcing control and UWS brains to the same 2.5-Hz
state. The paper's fixed 2.5-Hz target remains available as a sensitivity
option.

The online model adds filtered excitatory and inhibitory rate states with a
default 50-ms detector constant. Plasticity is gated off below a 20-Hz filtered
excitatory rate, preventing initial settling or ordinary baseline fluctuations
from changing inhibition. The pulse therefore remains fast, but persistent
high, above-target firing progressively strengthens `Q_i_e` during the
following 1.5-s observation window. The 2-s learning constant follows the
cited model and is tested against two controls: no homeostasis and inhibition
learned before then frozen during stimulation. PCI itself remains calculated
from the first 300 ms beginning at the 8-ms response offset; the extended
window is used to establish whether persistent explosive activity is brought
back down rather than to redefine PCI.

### Online AdEx equations

For node `r`, the second-order Zerlaut AdEx rate equation remains

```text
T dnu_mu/dt = F_mu - nu_mu
              + 1/2 sum(lambda,eta) C_lambda_eta
                    d2F_mu/(dnu_lambda dnu_eta),  mu in {e,i}.
```

The long-range excitatory input is

```text
L_r(t) = G sum_s SC_rs nu_e,s(t - delay_rs),
```

and the stimulus adds `A p(t-t0)` only to left SMA. Receptor occupancy changes
regional potassium conductance according to

```text
gK_r(o) = gK_control - o R_r (gK_control - gK_drug),
```

where `R_r` is the label-aligned, normalized 5-HT2A density.

Online homeostasis introduces three states per region:

```text
tau_d dR_e/dt = nu_e - R_e
tau_d dR_i/dt = nu_i - R_i
Q_i_e(t) = Q_i_e,0 H(t)
```

With `epsilon=(R_e-rho)/rho`, `gamma=max(R_i,0)/rho`, and runaway gate
`A=1[R_e >= r_activation]`, the efficacy state is

```text
dH/dt = A gamma epsilon B(H,epsilon) / tau_h,
```

where the two-sided soft bound is

```text
B = ((Hmax-H)/(Hmax-1))^beta,  epsilon >= 0
B = ((H-Hmin)/(1-Hmin))^beta, epsilon < 0.
```

`Q_i_e(t)` is used only by the excitatory transfer function and excitatory
membrane-voltage/adaptation calculation. The inhibitory transfer function
continues to use the fixed `Q_i`, so the controller specifically changes
inhibitory-to-excitatory feedback. Defaults are `tau_d=50 ms`, `tau_h=2000 ms`,
`r_activation=20 Hz`, `beta=1`, `Hmin=0.25`, and `Hmax=4`.

## Running on the cluster

```bash
sbatch hpc/submit_pci_stimulus_g_calibration.sh
```

To choose named candidates:

```bash
sbatch hpc/submit_pci_stimulus_g_calibration.sh \
  --subject control:c0001 --subject uws:u0020
```

Run `--dry-run` first to verify subject IDs, atlas target, sampling rate, and
planned simulation counts. Results and figures are written below the output
root and are ignored by Git.

For a reduced local run (one matched seed, 112 simulations with the default
two subjects and occupancies):

```bash
python scripts/calibrate_pci_stimulus_g_homeostasis.py \
  --dataset-root /path/to/converted_structural \
  --quick-local --workers 8
```

## References

- Coronel-Oliveros et al. (2026), *A multi-frequency whole-brain neural mass
  model with homeostatic feedback inhibition*, PLOS Computational Biology,
  https://doi.org/10.1371/journal.pcbi.1013463
- Authors' released implementation,
  https://github.com/carlosmig/EEG-Dementias
