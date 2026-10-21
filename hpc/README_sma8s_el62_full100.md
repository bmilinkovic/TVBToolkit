# First-order SMA/5-HT2A full-cohort bank and separate PCI analyses

This workflow uses **all 179 non-COMA subjects** (35 CNT, 18 EMCS, 75 MCS,
51 UWS), four occupancies (0, .25, .5, .766) and 100 stochastic trials per
subject/occupancy: **71,600 eight-second simulations** in 716 SLURM array
elements. Each element runs 100 trials on a 48-CPU node; `%1` allows only one
array element at a time, matching the established 48-worker HPC allocation.
This may be a long HPC
campaign; the array is restartable and completed recordings are reused.

There is **one** simulation bank, not one run per PCI method. Simulations do
not calculate PCI. The PCI array has two independent elements, each reading
exactly that bank. The analyses write to distinct output directories.

## Frozen model and recording protocol

The same setup as the 10-trial −62 mV pilot is fixed in `common.py`: first-
order stochastic split-leak gNa/gK AdEx; subject-specific shared-control-
maximum-scaled invnodevol SC and tracts, no Laplacian or row normalisation;
G=0.3; external drive=.315 Hz; weight_noise=.0002; regional OU tau=2 ms;
tau_w=500 ms; b=(5,15,35,40) pA for CNT/EMCS/MCS/UWS; E_L,e=−62 mV and
drug target=−60.2 mV; regionally min–max-normalised 5-HT2A occupancy changes
gK while gNa remains fixed. The left SMA receives the existing 10-ms square
pulse, 0.55 Hz/ms into excitatory rate. The 8-s simulation is continuous;
the first 2 s are equilibration, and the onset is reproducibly randomized
within 2.5–7.5 s. Seeds/onsets are paired across occupancies. TVB's
TemporalAverage monitor samples E at 1 ms. The bank saves the **−500 to
+500 ms peri-onset segment**, not all 8 s, plus full-run E/I peak and
>150-Hz incidence/duration diagnostics. Both PCI methods use [−300,0) and
[0,300) ms. High-rate incidence is recorded both over the full 8 s and after
the initial 2-s transient. No trial is silently excluded for high rates.

The second-order model is **not submitted**. Its pending validation should
be reviewed before a separate, versioned second-order bank is enabled; the
analysis entry point accepts a bank of matching trial recordings later.

## On the cluster

From `/home/bmilinkovic/TVBToolkit`, after pulling the new scripts and
confirming the transferred dataset. Check that the HPC checkout contains the
same model/transfer-function changes as the local −62 mV pilot; the frozen
manifest hashes the model sources on the HPC, but cannot make an older HPC
checkout identical to the local one automatically.

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/home/bmilinkovic/TVBToolkit/data/doc_data/converted_structural_invnodevol_control_max
export SMA8S_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_first_order_v1
export SMA8S_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_pci_v1

python experiments/sma8s_full100/simulate.py prepare \
  --dataset-root "$TVB_DATASET_ROOT" --bank "$SMA8S_BANK"
sbatch hpc/submit_sma8s_el62_full100_sim.sh
```

The prepare step must print `179 subjects; 716 subject×occupancy cases;
71600 simulations; array=0-715`. Save the returned simulation array job ID.
The `hpc/logs` directory must exist (it is tracked in this repository).

To watch progress:

```bash
python experiments/sma8s_full100/simulate.py status \
  --dataset-root "$TVB_DATASET_ROOT" --bank "$SMA8S_BANK"
squeue -u "$USER"
```

Only after the **entire** simulation array succeeds, submit PCI:

```bash
sbatch --dependency=afterok:SIM_ARRAY_JOB_ID hpc/submit_sma8s_el62_full100_pci.sh
```

Replace `SIM_ARRAY_JOB_ID` with the actual number, without angle brackets.
If any simulation array element fails, inspect its SLURM log, then resubmit
only those array indices using `sbatch --array=INDEX hpc/submit_sma8s_el62_full100_sim.sh`.
The trial files that passed signature checks are reused. After all 716 case
markers exist, submit the PCI script without a dependency if the previous
dependent job was cancelled.

The PCI array element 0 writes `hybrid/`; element 1 writes
`tvbsim_two_sided_min/`. Each has `methods.json`, `trial_scores.csv`,
`subject_summary.csv`, all individual 90×300 masks under `masks/`,
`figures/pci_by_occupancy.{pdf,png}`, and `completion.json`. The hybrid uses
500 Gaglioti bootstrap draws, pooled regional baseline and the trial-mean
surrogate cutoff on individual two-sided post responses. The TVBSim variant
uses its ten-shuffle **minimum** surrogate maximum and changes its final
positive-only test to two-sided. Both compute one Toolkit PCI-LZ per trial
and average 100 scores per subject, with **no 0.08 entropy guard**.

If the cluster's `workq` concurrent-array policy differs, adjust the `%1`
limit at submission (`sbatch --array=0-715%N ...`) after consulting its
resource policy; each simultaneous element then uses another 48-CPU node.
Do not point the two analyses or future second-order runs
at the same output directory.
