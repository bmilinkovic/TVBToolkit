# Cluster deployment — four subjects, 48 workers, summaries only

This is the authoritative launch guide for version 4. **No cluster jobs have been
submitted by the assistant.** Only code is on GitHub, not participant data.

## Design and remaining scientific qualification

Wake c0005, EMCS e0003, MCS m0011 and UWS u0033; 10,000 stochastic first-order
gNa/gK draws each; b uniform 0–80 pA, G uniform 0.025–0.075; tau_w=500 ms,
speed=4 m/s; occupancy=0; no TMS, no second-order model. Personal shared-control-
maximum SC and tract lengths. 120 retained seconds, TR=2.4 s, double-gamma BOLD.
No state occupancy or SC–FC fitting. The 179-subject inventory is not reduced:
five flagged BOLD records still need individual investigation before a cohort run.

**Warm-up is configurable at preparation.** Default 60 s remains conservative,
not an experimentally established requirement. The 32-s convolution needs history.
Use a separate pilot output directory with `--warmup-seconds 33.6` to compare
that shorter choice against 60 s; do not silently change already-generated banks.
Inspect initial operating regimes and summary stability before committing to 40k.
All timings previously reported assume 60+120 s; a shorter validated warm-up would
reduce cost. This release does not claim that the warm-up comparison has been run.

## 1. Transfer the small empirical inputs from your Mac

Replace `HPC_HOST` and `/scratch/bmilinkovic` with your actual cluster hostname and
allocated storage. No hostname or scratch allocation is assumed to exist.

```bash
ssh bmilinkovic@HPC_HOST 'mkdir -p /scratch/bmilinkovic/bg_sbi/inputs'
rsync -av --include='/*_bold.npz' --include='/pilot.json' \
  --include='/input_audit.json' --exclude='*' \
  /Users/borjan/CNRS/projects/TVBToolkit/results/first_order_bg_sbi_four_2min/ \
  bmilinkovic@HPC_HOST:/scratch/bmilinkovic/bg_sbi/inputs/

# Skip if you already transferred this SAME normalized dataset intact:
rsync -av \
  /Users/borjan/CNRS/projects/TVBToolkit/data/doc_data/converted_structural_invnodevol_control_max/ \
  bmilinkovic@HPC_HOST:/scratch/bmilinkovic/converted_structural_invnodevol_control_max/
```

Do NOT transfer old local subject-bank manifests as production banks. The four
small `*_bold.npz` exports contain the full empirical series, not simulated data.
Subject identifiers are pseudonymous; transfer via your institution-approved SSH.

## 2. Update code and prepare banks ON the cluster

```bash
cd /home/bmilinkovic/TVBToolkit
git pull --ff-only
export TVB_REPO=/home/bmilinkovic/TVBToolkit
source hpc/slurm_env.sh
# Only if the existing environment does not already have these dependencies:
python -m pip install -e '.[inference]'
python -c 'import vbi,sbi,torch; print(vbi.__version__,sbi.__version__,torch.__version__)'

export TVB_DATASET_ROOT=/scratch/bmilinkovic/converted_structural_invnodevol_control_max
export BG_SBI_PILOT=/scratch/bmilinkovic/bg_sbi/banks_v4
python experiments/first_order_bg_sbi/prepare_cluster.py \
  --inputs /scratch/bmilinkovic/bg_sbi/inputs \
  --dataset-root "$TVB_DATASET_ROOT" --output "$BG_SBI_PILOT"
mkdir -p hpc/logs
```

Use VBI 0.4.3 / SBI 0.23.2 from the existing inference extra, not the older editable
`~/code/python/vbi`. Preparation refuses to overwrite a bank. If changing code or
protocol, create a new bank; input/code hashes intentionally prevent mixed runs.
The existing bootstrap supports `TVB_CONDA_ENV` / `TVB_CONDA_PREFIX` overrides.

## 3. Benchmark first, then extend the SAME banks

```bash
# Four array tasks, limited to ONE node concurrently, 48 workers per node.
# Sixteen draws use at most 16 simultaneous workers in each preflight task.
BG_SBI_STOP=16 sbatch hpc/submit_first_order_bg_sbi_four.sh
squeue -u "$USER"
```

Inspect logs, `run_0_16.json`, validity statuses and actual elapsed times.
The existing 100-Hz/negative-rate QC flag is a conservative review rule, not a
PFP classifier. Review flags/failures are kept, not discarded. A job exits nonzero
if any draw needs review; posterior training refuses flagged banks. Never widen
QC silently just to obtain training data. Stale `RUNNING` files after scheduler
termination must be cleared only after confirming that job is no longer running.

```bash
# After satisfactory preflight; existing completed draws are reused:
sbatch hpc/submit_first_order_bg_sbi_four.sh
```

Resources match the earlier scripts: workq, one node, one task, 48 CPUs, mem=0,
14-day limit per array task, array 0–3%1. This is **48 workers total at a time**,
not four simultaneous nodes. Override array concurrency only if desired and
allocated. Do not submit competing runs to the same bank.

## 4. Fit-coloured plots and SBI are separate jobs

```bash
# After all four banks have finished (or use BG_SBI_BUDGET=16 for pilot maps):
BG_STAGE=report sbatch hpc/submit_first_order_bg_sbi_four.sh
BG_STAGE=train sbatch hpc/submit_first_order_bg_sbi_four.sh
```

Do not launch training before simulation completion. Each report writes
`fit_map_10000/b_G_fit_map.png` and `.pdf`, coloured by a descriptive feature-fit
score. A red **star marks the highest-scoring valid simulated (b,G)**; grey
crosses show invalid/reviewed cases. A numerical CSV and `best_fit.json` identify
the exact draw and seed. Existing report/fit directories are never overwritten.

Score definition: standardise features by their prior-simulation SD; give FC,
spatial-FC, FCD, ACF and spectral feature families equal weight; average squared
discrepancy across every empirical segment with overlap-coverage weights;
score=1/(1+sqrt(discrepancy)). Higher is better. **This is neither a likelihood nor
the posterior density/MAP.** The star may reflect a favourable stochastic draw;
validate candidates with additional seeds rather than declaring it unique truth.

Each fit uses 90% of the bank for training (with SBI's internal validation split),
10% held out; up to 100 cases tested for recovery. `fit_10000_seed1/` saves joint
posterior samples, per-segment posteriors, uncertainty, recovery diagnostics and
figures. All 297 empirical volumes contribute through starts 0,50,100,150,200,247.
The three overlapping volumes are downweighted. The subject summary is a
coverage-weighted **mixture of segment-conditioned posteriors**, deliberately
not a product of correlated segment likelihoods or a jointly inferred whole-scan
posterior. Inspect segment-to-segment uncertainty; no uniqueness is required.

## What is saved (and NOT saved)

- Per draw: `features.npz` (119 features, b, G, seed), `metadata.json` (QC,
  timing, model provenance, failures). No neural/BOLD simulation time series.
- Per participant: manifest, small empirical observation/segment data, run
  summaries, fit-score maps/tables, trained posterior, samples and recovery plots.
- Optional prediction stage also saves only features and diagnostics.

The numerical feature payload for 40k draws is roughly 19 MB before container,
metadata and filesystem overhead. Expect considerably more on disk due to many
small files, JSON and neural-network/posterior objects, but not terabytes. Measure
actual usage on the pilot. Pickling is not a compression solution for raw traces.
Rerunning from saved seed/parameters is required to recover traces later; preserve
code/environment versions too. Do not load untrusted posterior `.pt` pickle files.

```bash
# From your Mac, retrieve compact results (choose your local destination):
rsync -av bmilinkovic@HPC_HOST:/scratch/bmilinkovic/bg_sbi/banks_v4/ \
  /Users/borjan/CNRS/projects/TVBToolkit/results/bg_sbi_cluster_v4/
```
