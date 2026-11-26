# MCS b=32 pA: first-order sensitivity run only

This is **not** another 179-subject simulation. The original CNT (35), EMCS
(18) and UWS (51) subjects retain their completed first-order b=5/15/40 pA
recordings and hybrid-PCI scores. Only the 75 MCS subjects change from
**b=35 to b=32 pA**. Four occupancies and 100 trials yield 300 array cases and
**30,000 new eight-second simulations**. The model order, SC scaling, G=0.3,
tau_w=500 ms, E_L=-62 mV, noise, receptor mapping, 10-ms 0.55-Hz/ms left-SMA
pulse, 2-s discarded transient, random pulse onset, 1-ms temporal averaging,
and [-300,0)/[0,300)-ms PCI windows remain those of the original bank.
The existing hybrid PCI algorithm is called directly: 500 bootstraps,
two-sided alpha=.01 threshold, one PCI-LZ value per trial, 100-trial mean per
subject, no high-rate exclusion or entropy guard.

The prepare step **refuses to run** if the original bank's hashed model source,
dataset, receptor CSV, subjects, or protocol no longer match the checkout.
Do not bypass this check: otherwise a difference in code could be mistaken for
an effect of b. If it fails, restore the exact original code revision in a
separate checkout or ask for a controlled version comparison.

## Cluster commands

From `/home/bmilinkovic/TVBToolkit` after pulling the new files:

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/home/bmilinkovic/TVBToolkit/data/doc_data/converted_structural_invnodevol_control_max
export SMA8S_ORIGINAL_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_first_order_v1
export SMA8S_ORIGINAL_HYBRID_PCI=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_pci_v1/hybrid
export SMA8S_MCS_B32_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_mcs_b32_first_order_v1
export SMA8S_MCS_B32_PCI=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_mcs_b32_hybrid_v1
export SMA8S_MCS_B32_COMBINED=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_mcs_b32_combined_hybrid_v1

python experiments/sma8s_mcs_b32_full100/simulate.py prepare \
  --dataset-root "$TVB_DATASET_ROOT" \
  --reference-bank "$SMA8S_ORIGINAL_BANK" \
  --bank "$SMA8S_MCS_B32_BANK"
sbatch hpc/submit_sma8s_mcs_b32_full100_sim.sh
```

Prepare must report `75 subjects, 300 cases, 30000 trials; array=0-299`.
Keep the returned simulation job ID. After it completes successfully, run:

```bash
python experiments/sma8s_mcs_b32_full100/simulate.py status \
  --dataset-root "$TVB_DATASET_ROOT" \
  --reference-bank "$SMA8S_ORIGINAL_BANK" \
  --bank "$SMA8S_MCS_B32_BANK"
sbatch hpc/submit_sma8s_mcs_b32_full100_hybrid_pci.sh
```

Status must report `300/300`. When PCI finishes, verify its
`completion.json` says 300 cases and 30,000 trials. Then merge the new MCS
scores with the original, untouched CNT/EMCS/UWS scores:

```bash
sbatch hpc/submit_sma8s_mcs_b32_merge.sh
```

The merge step checks original/new manifest linkage, the same trial seeds and
stimulus onsets, all 300 matching MCS subject/occupancy keys, and complete PCI
tables. It writes combined 179-subject tables and an overview figure. It does
not duplicate raw trials or masks; `provenance.json` points to the original
and new directories holding them. Do not use old MCS masks when plotting the
new MCS b=32 condition.

The simulation output includes 30,000 peri-onset NPZ recordings and metadata
JSON files. The MCS PCI output includes `subject_summary.csv` (300 rows),
`trial_scores.csv` (30,000 rows), 300 mask NPZ files, `methods.json`,
`completion.json`, and `figures/mcs_pci_by_occupancy.{pdf,png}`. The merged
output has 716 summary rows, 71,600 trial rows, `provenance.json`, and
`figures/pci_by_occupancy_combined.{pdf,png}`.
