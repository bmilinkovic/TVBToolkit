#!/usr/bin/env bash
#SBATCH --job-name=tvb-sero-pci
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh
DATASET_ROOT="$(resolve_tvb_dataset_root)"

# Baseline root is still accepted for aggregate-only reuse, but this full run
# simulates occupancy 0.0 by default so all four occupancy scenarios are fresh.
BASELINE_ROOT="${SEROTONERGIC_BASELINE_ROOT:-${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/condition_b/sims_pci}"
OUTPUT_ROOT="${SEROTONERGIC_OUTPUT_ROOT:-${TVB_REPO}/notebooks/outputs/serotonergic_pci_full_50trials}"
SCENARIO="${SEROTONERGIC_SCENARIO:-private_alpha0}"

# Pilot dose schedule was four occupancies total. This submitter simulates all
# four by passing --simulate-baseline to the Python runner below.
read -r -a OCCUPANCIES <<< "${SEROTONERGIC_OCCUPANCIES:-0 0.25 0.5 0.766}"
TRIAL_SEEDS=($(seq 0 49))

echo "[sero-pci] dataset_root=${DATASET_ROOT}"
echo "[sero-pci] baseline_root=${BASELINE_ROOT}"
echo "[sero-pci] output_root=${OUTPUT_ROOT}"
echo "[sero-pci] scenario=${SCENARIO}"
echo "[sero-pci] occupancies=${OCCUPANCIES[*]}"
echo "[sero-pci] n_trial_seeds=${#TRIAL_SEEDS[@]}"
echo "[sero-pci] workers=${SLURM_CPUS_PER_TASK}"

python scripts/run_serotonergic_pci_full.py \
  --dataset-root "${DATASET_ROOT}" \
  --baseline-root "${BASELINE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --scenario "${SCENARIO}" \
  --trial-seeds "${TRIAL_SEEDS[@]}" \
  --occupancies "${OCCUPANCIES[@]}" \
  --simulate-baseline \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "$@"

python scripts/plot_serotonergic_pci_publishable.py \
  --input "${OUTPUT_ROOT}/tables/serotonergic_pci_subject_metrics_with_rescue.csv" \
  --output-dir "${OUTPUT_ROOT}/figures/publishable" \
  --prefix "serotonergic_pci_rescue_full_50trials"

echo "[sero-pci] full serotonergic PCI run complete"
