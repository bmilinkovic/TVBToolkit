# Full-cohort Dalla Porta-inspired second-order HPC study

This is the exact local `experiments/della_porta_second_order/study.py` trial
routine, now scheduled over **all 179 non-COMA subjects** rather than the
20 selected subjects. It keeps both excitatory leak settings (−63, −62 mV),
four 5-HT2A occupancies (0, .25, .5, .766), and **50 trials per case**:

`179 subjects × 2 leaks × 4 occupancies × 50 trials = 71,600 simulations`

Every simulation is 8 s with the first 2 s treated as equilibration; a
randomized 2.5–7.5 s onset delivers a 30-ms, 1-Hz/ms square pulse to left
superior parietal cortex. The model is stochastic second-order split-leak
gNa/gK AdEx, G=.3, external drive=.315 Hz, regional noise weight=.0002,
regional OU tau=2 ms, tau_w=500 ms. The local Dalla Porta adaptation values
are CNT 5, EMCS 15, MCS **30**, UWS 40 pA. The shared-control-maximum AAL90
connectomes and individual tract lengths are retained. The existing
experimental first-/second-derivative attenuation is used; covariances are
integrated directly, with no clipping. High-rate and invalid-covariance
events are flagged, and all finite trials are retained.

This differs deliberately from the SMA full100 banks: SMA rather than
parietal stimulation, 10 ms × .55 Hz/ms rather than 30 ms × 1 Hz/ms, MCS b=35
rather than 30, only −62 mV, and 100 rather than 50 trials. Thus comparing
these banks tests **protocols**, not one isolated causal parameter. For the
cleanest protocol comparison use the Dalla Porta −62 mV panel and the same
downstream PCI rule for every bank. The first- and second-order SMA trials
are paired with each other, but their seeds are **not** paired to the Dalla
Porta trials. Within Dalla Porta, seeds are paired across leak and occupancy.

## Cluster submission

From `/home/bmilinkovic/TVBToolkit` after pulling the new commit:

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/home/bmilinkovic/TVBToolkit/data/doc_data/converted_structural_invnodevol_control_max
export DELLA_FULL_BANK=/home/bmilinkovic/TVBToolkit/results/della_porta_second_order_full50_v1
export DELLA_FULL_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/della_porta_second_order_full50_pci_v1

python experiments/della_porta_full_cohort/simulate.py prepare \
  --dataset "$TVB_DATASET_ROOT" --bank "$DELLA_FULL_BANK"
sbatch hpc/submit_della_porta_full_cohort_sim.sh
```

`prepare` must report `179 subjects; 1432 cases; 71600 trials` and an array of
`0-715`, **two cases per array task**. This avoids a possible cluster array-size
limit. Each task uses 48 workers on one `workq` node, sequentially runs its
two cases, and logs progress every 10 completed trials. Monitor it with:

```bash
python experiments/della_porta_full_cohort/simulate.py status \
  --dataset "$TVB_DATASET_ROOT" --bank "$DELLA_FULL_BANK"
squeue -u "$USER"
```

After all 1,432 cases complete, use the returned simulation job ID:

```bash
sbatch --dependency=afterok:SIM_ARRAY_JOB_ID hpc/submit_della_porta_full_cohort_pci.sh
```

PCI is a **separate job**. On each set of identical recordings it computes:

1. Dalla Porta's trial-averaged binarization and LZ76-normalized PCI;
2. the selected Gaglioti-derived hybrid, thresholding each poststimulus trial
   and averaging 50 PCI-LZ values;
3. the two-sided TVBSim minimum-surrogate comparator, also per trial.

All use [-300,0) ms baseline and [0,300) ms response. The Dalla Porta method
uses 1,000 bootstrap draws; hybrid uses 500 at nominal alpha=.01; TVBSim uses
ten shuffles and their smallest maximum. Outputs are separate from the trial
bank: `subject_summary.csv`, `trial_scores.csv`, every case's binary masks
and cutoff, case JSONs, high-rate/covariance counts, occupancy-vs-PCI PDF/PNG
figures for each leak, zero-occupancy example masks, and completion metadata.
The bank stores each trial's 1-ms, 90-region E and I peri-onset recordings
plus JSON diagnostics. No full eight-second neural trace is retained.

The SMA PCI jobs already compute hybrid and TVBSim on their 100-trial banks.
To add the same Dalla Porta trial-mean PCI rule to either completed SMA bank,
set `SMA8S_BANK` to that bank and `DELLA_ON_SMA_OUTPUT` to a fresh analysis
directory, then run `sbatch hpc/submit_della_pci_on_sma_bank.sh`. Do this once
for first-order and once for second-order, with different output directories.
This scores existing recordings only. It does not rerun the model, and retains
all 100 trials for the SMA trial average; Dalla Porta's own bank averages 50.

Each array task is restartable: resubmitting an index reuses completed trials
from both assigned cases after checks. Never point these scripts at an SMA
bank or at the local 20-subject Dalla Porta bank.
