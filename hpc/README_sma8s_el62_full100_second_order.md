# Second-order SMA/5-HT2A full-cohort HPC run

This is a **separate bank** from the existing first-order run. It uses the
same 179 non-COMA subjects, 4 occupancies, 100 trials, seeds, onsets,
connectomes, delays, noise, b, G, pulse, 8-s duration, 2-s transient and
1-ms temporal averaging: 716 cases and 71,600 simulations. The only model
switch is `order=2`, using the current native split-leak gNa/gK model with
corrected mixed covariance equation and the existing `scaled_rhs` adapter.
The adapter divides the first derivatives by 1000 and second derivatives by
1,000,000; this is **experimental derivative attenuation**, not an established
unit conversion. Covariances are directly integrated without clipping or PSD
projection. Every finite trial is retained. >150-Hz episodes and negative
covariance eigenvalues are flagged in each trial JSON and propagated into PCI
trial/subject tables. Nonfinite model states fail the trial rather than being
silently analysed.

The 10-trial pilot had substantial covariance-invalid and high-rate incidence.
Therefore this full run is a *model-comparison analysis*, not a validated
physiological second-order result. Inspect the flags alongside PCI before
interpreting group or occupancy differences.

After `git pull` on the cluster, from `/home/bmilinkovic/TVBToolkit`:

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/home/bmilinkovic/TVBToolkit/data/doc_data/converted_structural_invnodevol_control_max
export SMA8S_SECOND_ORDER_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_second_order_scaled_v1
export SMA8S_SECOND_ORDER_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_second_order_pci_v1

python experiments/sma8s_full100_second_order/simulate.py prepare \
  --dataset-root "$TVB_DATASET_ROOT" --bank "$SMA8S_SECOND_ORDER_BANK"
sbatch hpc/submit_sma8s_el62_full100_second_order_sim.sh
```

`prepare` must report `179 subjects; 716 subject×occupancy cases; 71600
simulations; array=0-715`. The array uses the same 48-CPU `workq` allocation
as the first-order bank and prints progress every 10 completed trials. Check:

```bash
python experiments/sma8s_full100_second_order/simulate.py status \
  --dataset-root "$TVB_DATASET_ROOT" --bank "$SMA8S_SECOND_ORDER_BANK"
squeue -u "$USER"
```

After all 716 cases complete, submit separate downstream PCI jobs:

```bash
sbatch --dependency=afterok:SIM_ARRAY_JOB_ID hpc/submit_sma8s_el62_full100_second_order_pci.sh
```

Replace `SIM_ARRAY_JOB_ID` with the actual simulation job number. PCI element
0 runs the selected per-poststim-trial `hybrid` method (500 Gaglioti bootstrap
draws, two-sided threshold, alpha .01); element 1 runs the two-sided
TVBSim-minimum comparator. Neither drops high-rate/covariance-invalid trials,
and neither uses the 0.08 entropy floor. Both take [-300,0) and [0,300) ms
from the saved [-500,+500) ms peri-onset recording. The output root contains
separate method directories with `trial_scores.csv`, `subject_summary.csv`,
binary masks, figures, and methods/completion metadata. The bank itself has
one compressed regional E trace and one JSON diagnostics file per trial.

To retry an interrupted case, resubmit only its array index; completed trial
files are verified and reused. Do not mix this bank with first-order files.
