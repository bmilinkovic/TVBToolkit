# Original per-trial Gaglioti-style PCI on the saved SMA banks

This is a **downstream analysis only**. It reads each completed 100-trial,
8-second SMA recording bank and does not rerun AdEx or alter the hybrid,
TVBSim, or Dalla Porta output directories.

For every trial and every AAL90 region, baseline mean and sample SD come from
that trial's [-300, 0) ms interval. The original one-trial specialization of
the supplied 500-bootstrap Gaglioti spatial-maximum statistic gives a
two-sided alpha=.01 threshold. The individual [0, 300) ms response is
binarised against its **own** threshold. One Toolkit PCI-LZ is computed per
trial, then 100 trial scores are averaged per subject and occupancy. This is
the per-trial rule used in the earlier SMA validation figures. It is **not**
Gaglioti's literal trial-averaged empirical statistic.

No 0.08 entropy floor, high-rate exclusion, or covariance exclusion is used.
All such trial flags are retained in the output table for quality review.
Only the display-independent original masks are scored; the saved figure
ordering does not change any PCI result.

After `git pull` on the cluster, from `/home/bmilinkovic/TVBToolkit`:

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export SMA8S_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_first_order_v1
export SMA8S_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_pci_v1
sbatch hpc/submit_sma8s_el62_full100_original_gaglioti_pci.sh
```

The first-order simulation bank must already report 716/716 complete cases.
The PCI root may already contain `hybrid` and `tvbsim_two_sided_min`; the new
job writes only to its fresh `gaglioti_per_trial` subdirectory. It prints
progress every 20 of 716 subject-by-occupancy cases.

For the second-order bank, after its simulations complete (or submit with an
`afterok` dependency on simulation array job 2175490):

```bash
export SMA8S_SECOND_ORDER_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_second_order_scaled_v1
export SMA8S_SECOND_ORDER_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_second_order_pci_v1
sbatch hpc/submit_sma8s_el62_full100_second_order_original_gaglioti_pci.sh
```

Each method directory contains `methods.json`, `completion.json`,
`trial_scores.csv` (71,600 rows), `subject_summary.csv` (716 rows), one case
JSON and compressed 100-by-90-by-300 binary mask array per case, plus
`figures/pci_by_occupancy.pdf` and `.png`. Each trial has its own threshold,
saved in the trial table and NPZ array. There is deliberately no single
subject-wide threshold; `subject_summary.csv` leaves `threshold_z` blank.

Successful completion requires `completion.json` with `cases=716` and
`trials=71600`. The score job is restartable: complete case JSON/NPZ pairs
are checked and reused. Never point first- and second-order jobs at the same
bank or PCI output directory.
