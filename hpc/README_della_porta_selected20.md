# Resume the paused Dalla Porta study on the HPC

This is the **same scientific experiment** as the paused local
`della_porta_second_order_aal90_5x50_v1` run: five selected subjects from each
of CNT, EMCS, MCS and UWS; two excitatory leaks (−63 and −62 mV); four 5-HT2A
occupancies; and 50 trials per subject/leak/occupancy. Total: **8,000 trials
in 160 cases**. No COMA subjects are included. The existing local `run_trial`
function is called without changes, so the subject selection, trial seeds,
randomized pulse times, 30-ms × 1-Hz/ms left superior parietal pulse,
second-order split-leak model, G=.3, b=(5,15,30,40) pA, noise and 8-second
protocol match the local run. One 48-CPU Slurm array task runs one 50-trial
case. Saved 1-ms peri-onset E/I recordings and diagnostics have the same format
as the local bank.

This creates a **new HPC bank**. It does not move or delete the 406 completed
local recordings, nor resume their in-memory workers. The local run remains
paused; do not restart it accidentally while comparing outputs.

From `/home/bmilinkovic/TVBToolkit` after `git pull`:

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/home/bmilinkovic/TVBToolkit/data/doc_data/converted_structural_invnodevol_control_max
export DELLA_SELECTED20_BANK=/home/bmilinkovic/TVBToolkit/results/della_porta_second_order_selected20_hpc_v1
export DELLA_SELECTED20_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/della_porta_second_order_selected20_hpc_pci_v1

/home/bmilinkovic/.conda/envs/tvbtoolkit/bin/python \
  experiments/della_porta_selected20_hpc/simulate.py prepare \
  --dataset "$TVB_DATASET_ROOT" --bank "$DELLA_SELECTED20_BANK"
sbatch hpc/submit_della_porta_selected20_sim.sh
```

`prepare` must print `20 subjects; 160 cases; 8000 trials; SLURM array=0-159`.
Use the `tvbtoolkit` Python for this interactive command; `base` lacks NumPy.
Slurm activates that environment itself. Each task prints progress at 10,
20, 30, 40 and 50 completed trials. Monitor completed cases with:

```bash
/home/bmilinkovic/.conda/envs/tvbtoolkit/bin/python \
  experiments/della_porta_selected20_hpc/simulate.py status \
  --dataset "$TVB_DATASET_ROOT" --bank "$DELLA_SELECTED20_BANK"
```

Only after all 160 cases complete, submit the **separate** Dalla Porta
trial-averaged PCI calculation using the simulation array's job ID:

```bash
sbatch --dependency=afterok:SIM_ARRAY_JOB_ID hpc/submit_della_porta_selected20_pci.sh
```

Its result directory contains a subject/leak/occupancy PCI table, the
source-by-time binary masks, zero-occupancy cohort mask figures, an
occupancy-vs-PCI figure and methods metadata. The original local study's PCI
code is reused, including 1,000 bootstrap draws and the [−300,0) and
[0,300) ms windows. No new statistical definition is introduced here.

If a case fails, inspect its log, then resubmit just that array index using
`sbatch --array=INDEX hpc/submit_della_porta_selected20_sim.sh`. Completed
trial files are checked and reused. Avoid pointing this job at the much
larger, separate 179-subject Dalla Porta bank.
