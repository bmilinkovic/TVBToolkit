# Full-cohort MCS/UWS b=27/42 pA sensitivity analysis

This is a **separate sensitivity run** of the frozen SMA8s protocol. The only
parameter changes relative to the matching original first- or second-order
bank are **MCS b: 35 -> 27 pA** and **UWS b: 40 -> 42 pA**. CNT=5, EMCS=15,
tau_w=500 ms, G=0.3, the split-leak gNa/gK model, shared-control-
maximum-scaled subject SC, 5-HT2A mapping, pulse, seeds and randomized onset
rule are unchanged. The four occupancy levels remain 0, 0.25, 0.50 and 0.766.

Each model order runs 179 non-COMA subjects x 4 occupancies x 100 trials =
**71,600 simulations** in 716 array tasks. Each trial lasts 8 s, with the
first 2 s treated as equilibration. Stimulus onset is randomized between
2.5 and 7.5 s; the saved E trace is TVB's 1-ms TemporalAverage from -500 to
+500 ms around onset. All PCI methods use [-300,0) ms baseline and [0,300)
ms response. One recording bank serves all downstream PCI analyses. The
first-order run is the primary manuscript sensitivity; order 2 is optional
and remains scientifically separate.

## Install code and set paths on the cluster

From a **clean** cluster checkout, fetch and switch to the pushed branch:

```bash
cd /home/bmilinkovic/TVBToolkit
git fetch origin experiment/adex-tf-refit-validation
git switch --track origin/experiment/adex-tf-refit-validation
```

If that local branch already exists, use
`git switch experiment/adex-tf-refit-validation` followed by
`git pull --ff-only`. Do not discard local cluster changes to switch branches.
Confirm the dataset already transferred earlier contains `index.json`:

```bash
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=/home/bmilinkovic/TVBToolkit/data/doc_data/converted_structural_invnodevol_control_max
test -f "$TVB_DATASET_ROOT/index.json"
```

## First-order simulation bank

```bash
export SMA8S_B27_ORDER=1
export SMA8S_B27_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_b27_u42_first_order_v1
export SMA8S_B27_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_b27_u42_first_order_pci_v1

python experiments/sma8s_b27_full100/simulate.py prepare \
  --order "$SMA8S_B27_ORDER" --dataset-root "$TVB_DATASET_ROOT" \
  --bank "$SMA8S_B27_BANK"
sbatch hpc/submit_sma8s_b27_full100_sim.sh
```

The prepare output must say `179 subjects; 716 subject×occupancy cases;
71600 simulations; array=0-715`. Save the returned simulation job ID.

```bash
python experiments/sma8s_b27_full100/simulate.py status \
  --order "$SMA8S_B27_ORDER" --dataset-root "$TVB_DATASET_ROOT" \
  --bank "$SMA8S_B27_BANK"
```

Only after all **716/716** cases are complete and the simulation array has
finished successfully, submit PCI:

```bash
sbatch hpc/submit_sma8s_b27_full100_pci.sh
```

Alternatively, to queue PCI immediately but ensure it starts only after the
simulation array succeeds, run `sbatch --dependency=afterok:SIM_JOB_ID
hpc/submit_sma8s_b27_full100_pci.sh`, replacing `SIM_JOB_ID` with the real
number. If any simulation task fails, inspect its log, then resubmit only the
failed case IDs using `sbatch --array=INDEX
hpc/submit_sma8s_b27_full100_sim.sh`; completed trials with matching
signatures are reused. Do not run PCI on a partial bank.

## Optional second-order bank

The same launchers support order 2. Use **different bank and PCI directories**
and do not reuse the first-order environment variables when submitting PCI:

```bash
export SMA8S_B27_ORDER=2
export SMA8S_B27_BANK=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_b27_u42_second_order_v1
export SMA8S_B27_PCI_ROOT=/home/bmilinkovic/TVBToolkit/results/sma8s_el62_full100_b27_u42_second_order_pci_v1
python experiments/sma8s_b27_full100/simulate.py prepare \
  --order "$SMA8S_B27_ORDER" --dataset-root "$TVB_DATASET_ROOT" \
  --bank "$SMA8S_B27_BANK"
sbatch hpc/submit_sma8s_b27_full100_sim.sh
```

Check `status` with `--order 2`, then submit the same PCI launcher. The
second-order implementation and covariance diagnostics are unchanged; its
scientific validity must be assessed separately from the first-order run.

## Outputs and completion checks

Each bank contains `manifest.json`, 71,600 peri-onset `recordings/*.npz`
with companion per-trial JSON diagnostics, and 716 `case_complete/*.json`
markers. The manifest records every cohort's b, occupancy, seed and source
code hashes. The simulation scripts fail rather than mixing different model
orders or parameters in the same bank.

The PCI array has three sequential tasks (48 CPUs each):

0. `hybrid`: the selected trial-wise hybrid Gaglioti threshold;
1. `tvbsim_two_sided_min`: historical ten-shuffle minimum with a two-sided mask;
2. `gaglioti_per_trial`: the earlier independent per-trial Gaglioti-style rule.

Each analysis subdirectory has `completion.json`, `methods.json`,
`trial_scores.csv` (71,600 rows plus header), `subject_summary.csv` (716
rows plus header), saved 90x300 binary masks, and
`figures/pci_by_occupancy.{pdf,png}`. No analysis excludes high-rate trials
or applies an entropy floor; trial flags remain available for sensitivity
checks. On the cluster, verify each method:

```bash
for method in hybrid tvbsim_two_sided_min gaglioti_per_trial; do
  cat "$SMA8S_B27_PCI_ROOT/$method/completion.json"
  wc -l "$SMA8S_B27_PCI_ROOT/$method/trial_scores.csv" \
    "$SMA8S_B27_PCI_ROOT/$method/subject_summary.csv"
done
```

## Exploratory baseline cutoff after hybrid PCI

Casarotto's empirical PCI*=0.31 was validated against subjective reports
using **PCImax across stimulation sites**. We do not transplant that value
to simulated single-site firing-rate PCI. Instead, after hybrid PCI finishes,
estimate a **model-specific, descriptive** cutoff separating behavioural
MCS and UWS at **zero occupancy**:

```bash
sbatch hpc/submit_sma8s_b27_cutoff.sh
```

The output `exploratory_cutoff_hybrid/` contains `cutoff_summary.json`,
`subject_classification.csv`, `cross_validation_folds.csv`, and the PDF/PNG
figure `baseline_cutoff`. The cutoff maximises Youden's J within this cohort;
the report also gives the 0.28, 0.31 and 0.44 *descriptive* crossings and
the threshold that would retain at least Casarotto's 36/38 MCS sensitivity.
Repeated five-fold **subject-level** cross-validation and subject bootstrap
quantify instability/optimism. Private clinical annotations are not shipped
with this public code. After transferring a pseudonymised CSV with
`subject,evidence` columns, rerun the cutoff script with
`--clinical-annotations /path/to/clinical_annotations.csv` to add the overlap.
Neither a high PCI nor a behavioural label alone establishes consciousness.
This cutoff must
not be described as an independently validated diagnostic threshold.

The original 35/40-pA MCS/UWS banks and their PCI directories are not modified.
