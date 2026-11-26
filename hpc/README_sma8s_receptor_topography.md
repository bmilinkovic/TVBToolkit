# AAL90 5-HT2A topography controls (first-order MCS/UWS)

This is a **separate control experiment**, not a change to the production
simulation or PCI banks. It uses every non-COMA MCS (75) and UWS (51) subject,
each one's own shared-control-maximum-normalised invnodevol SC, and the
original first-order split-leak SMA protocol (`b=35/40` pA, `G=0.3`,
`tau_w=500` ms, `E_L,e=-62` mV, 8 s simulation, 2 s transient, 10 ms pulse at
0.55 Hz/ms, 100 matched stochastic trials, TVB 1 ms TemporalAverage).

Only occupancy **0** and **0.766** are used. There is one newly simulated
zero-occupancy case per subject, shared across its high-dose map comparisons,
because at zero occupancy the receptor map does not enter gK. Seeds and random
pulse onsets are identical across that subject's baseline and every map.
High-dose maps are the original Cimbi AAL90 map, one uniform map at the mean
of the *effective min-max-normalised* density, `n` cortical-centroid spin
surrogates, and `n` full 90-region permutations. Each map is applied identically
to all patients. Scrambled maps retain the 90 regional values exactly; the
uniform map changes variance by design but retains their mean. **No structural
connectome, stimulus, noise, or other model parameter is changed.**

## Important spin limitation

The spin control is a *parcel-centroid approximation* to a cortical surface
spin: AAL90 cortical centroids are projected onto a hemisphere-centred unit
sphere, rotated, then one-to-one reassigned by Hungarian matching. The seven
non-cortical parcels per hemisphere cannot be surface-spun and are permuted
within hemisphere. This is spatially constrained, not an exact vertex-wise
surface spin or an exact preservation of every spatial autocorrelation value.
The map manifest records each assignment, its correlation with the original,
and a fixed-graph cortical Moran's I diagnostic. Inspect the spin versus
permutation Moran's I distributions before treating the spin as a convincing
spatial-autocorrelation-preserving null.
Do not label its p-values as canonical surface-spin p-values in the manuscript.
For a definitive surface-spin test, an AAL90 cortical surface-to-parcel mapping
would be needed. Full permutations are deliberately *not* spatially constrained.

## Computational scale

The default 99 spins and 99 full permutations yield 200 high-dose maps and
one shared zero-dose case per subject: **25,326 subject/map cases and
2,532,600 eight-second simulations**. This is expensive. A five-case local
smoke sample consumed about **320 KB per compressed trial**, implying roughly
**0.8 TB** of recordings before masks/metadata, subject to compression and
quota differences on the cluster. Do not submit the 99+99 production campaign
without checking quota and wall-time estimates. At 99 null maps,
the smallest Monte Carlo p is 0.01. A pilot with `--n-spin 19
--n-permutation 19` has 5,166 cases and 516,600 simulations, with p-resolution
only 0.05. If you want a smaller code smoke test, prepare an **entirely
different bank** with 1+1 null maps and run `case --max-trials 2`; that does
not produce scientifically interpretable PCI. Do not prepare over an existing
bank with different null counts or code.

**Simpler permutation-only option:** set `--n-spin 0 --n-permutation 19`.
The original map, uniform comparator, and one shared zero-dose baseline remain
in the design. This gives `126 × (1 baseline + 1 original + 1 uniform + 19
permutations) × 100 = 277,200` simulations; the smallest attainable
permutation p-value is `1/20 = 0.05`. For p-resolution 0.01, use 99
permutations: 1,285,200 simulations. A single permutation is a wiring smoke
test, **not** a valid spatial-null distribution. Full permutations test generic
map specificity but do **not** preserve spatial autocorrelation; a later spin
analysis is needed for that stronger claim.
For a one-permutation **workflow check**, use `--n-spin 0 --n-permutation 1`:
504 cases and 50,400 simulations. It produces a paired patient figure comparing
zero dose with high dose under the real versus shuffled maps, but a single
permutation cannot establish a null distribution or a meaningful p-value.

The wrapper deliberately runs only one 48-core case at a time (`%1`), to
match the existing cluster allocation. At an observed mean of *m* minutes per
100-trial case, the 99+99 campaign takes about `25,326*m/1,440` days before
queue delays and PCI analysis (for example, 3 minutes/case means ~53 days).
Measure a small pilot first and decide whether this cost is acceptable; do
not infer that the job is quick merely because each array task is short.

## Cluster commands

After pulling the committed scripts and confirming the existing dataset:

```bash
cd /home/bmilinkovic/TVBToolkit
export TVB_REPO=/home/bmilinkovic/TVBToolkit
export TVB_DATASET_ROOT=$TVB_REPO/data/doc_data/converted_structural_invnodevol_control_max
export SMA8S_TOPO_BANK=$TVB_REPO/results/sma8s_el62_b35_u40_receptor_topography_v1
export SMA8S_TOPO_PCI=$TVB_REPO/results/sma8s_el62_b35_u40_receptor_topography_hybrid_v1

/home/bmilinkovic/.conda/envs/tvbtoolkit/bin/python \
  experiments/sma8s_receptor_topography/simulate.py prepare \
  --dataset-root "$TVB_DATASET_ROOT" --bank "$SMA8S_TOPO_BANK" \
  --n-spin 99 --n-permutation 99

SMA8S_TOPO_DRY_RUN=1 bash hpc/submit_sma8s_receptor_topography_chunks.sh
bash hpc/submit_sma8s_receptor_topography_chunks.sh
```

For the permutation-only pilot, choose a distinct bank/output directory and
replace the two null-count flags with `--n-spin 0 --n-permutation 19`. Do not
change the counts after `prepare` has created a bank.

For the requested **one-permutation first run**, use distinct paths such as
`results/sma8s_el62_b35_u40_receptor_perm1_v1` for `SMA8S_TOPO_BANK` and
`results/sma8s_el62_b35_u40_receptor_perm1_hybrid_v1` for `SMA8S_TOPO_PCI`,
and prepare with `--n-spin 0 --n-permutation 1`. Then use the same dry-run,
submission, status and dependent PCI commands above. The extra figure is
`figures/one_permutation_paired_comparison.{pdf,png}`.

The wrapper submits 500-case chunks sequentially at the existing 48-worker
allocation and prints `LAST_SIM_JOB_ID`. Save that ID. It prevents an overly
large single Slurm array. A failed chunk blocks later chunks (`afterok`),
without deleting finished trials. For status:

```bash
/home/bmilinkovic/.conda/envs/tvbtoolkit/bin/python \
  experiments/sma8s_receptor_topography/simulate.py status \
  --dataset-root "$TVB_DATASET_ROOT" --bank "$SMA8S_TOPO_BANK"
```

Only once every case is complete, run the **separate** hybrid PCI job:

```bash
sbatch --dependency=afterok:LAST_SIM_JOB_ID \
  hpc/submit_sma8s_receptor_topography_pci.sh
```

Replace `LAST_SIM_JOB_ID` with the actual numeric ID. The analyzer also checks
every case-completion marker and will refuse an incomplete bank. It uses the
same 500-bootstrap, pooled regional prestimulus, two-sided hybrid binary mask
and per-trial PCI-LZ as the original first-order run; 100 trial scores are
averaged per patient/map. No high-rate trials are silently excluded. The
analysis records high-rate counts and should be interpreted alongside them.

## Outputs and inference

The bank contains `manifest.json`, `maps.npz`, 1-s peri-onset 1-ms E traces
and trial JSON under `recordings/`, plus one `case_complete/` marker per case.
The separate PCI output contains `subject_summary.csv`, `trial_scores.csv`,
`delta_by_subject_and_map.csv`, `null_map_group_summary.csv`,
`null_map_gradient_summary.csv`, `null_tests.csv`, all trial masks under
`masks/`, `figures/original_vs_spatial_nulls.{pdf,png}` and `completion.json`.
The patient is the unit of the group mean; each **same null map** is applied
to every patient. The null test compares the original map's mean ΔPCI in MCS,
UWS, and the MCS-minus-UWS gradient with the corresponding map-null
distribution using an inclusive Monte Carlo p-value. A separate uniform
comparator tests whether regional heterogeneity matters at all. This is a
topography-control experiment, not a clinical drug-effect validation.

**Storage cleanup:** the PCI Slurm script defaults to deleting only this
experiment bank's `recordings/` directory, *after* it has written and verified
all 100-trial scores, patient tables, all case masks, null summaries and figures,
and a completion file with the matching bank hash. It leaves those derived
outputs, the manifest/maps and case-completion markers intact, and writes
`recordings_deleted_after_pci.json` in the bank. To retain recordings instead,
export `SMA8S_TOPO_RETAIN_RECORDINGS=1` before submission. Once deleted, the
PCI thresholds and other raw-signal analyses cannot be recomputed without
rerunning the simulations; copy any desired raw examples *before* the PCI job.
