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
require_native_invnodevol_dataset "${DATASET_ROOT}"

# Baseline root is still accepted for aggregate-only reuse, but this full run
# simulates occupancy 0.0 by default so all four occupancy scenarios are fresh.
BASELINE_ROOT="${SEROTONERGIC_BASELINE_ROOT:-${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/condition_b/sims_pci}"
OUTPUT_ROOT="${SEROTONERGIC_OUTPUT_ROOT:-${TVB_REPO}/notebooks/outputs/serotonergic_pci_v6_dual_100trials_invnodevol_native}"
SCENARIO="private_alpha0"
COUPLING_STRENGTH="${SEROTONERGIC_COUPLING_STRENGTH:-}"
if [ -z "${COUPLING_STRENGTH}" ]; then
  echo "ERROR: set SEROTONERGIC_COUPLING_STRENGTH to the calibrated G value." >&2
  echo "       The legacy G=0.25 is not assumed after SC normalization changed." >&2
  exit 5
fi

# The corrected production protocol fixes four occupancies and 100 matched
# trial seeds. Smaller or alternate runs belong in the pilot runner.
OCCUPANCIES=(0 0.25 0.5 0.766)
TRIAL_SEEDS=($(seq 0 99))
if [ "${#TRIAL_SEEDS[@]}" -ne 100 ]; then
  echo "ERROR: corrected production protocol requires exactly 100 trials." >&2
  exit 4
fi

echo "[sero-pci] dataset_root=${DATASET_ROOT}"
echo "[sero-pci] baseline_root=${BASELINE_ROOT}"
echo "[sero-pci] output_root=${OUTPUT_ROOT}"
echo "[sero-pci] scenario=${SCENARIO}"
echo "[sero-pci] coupling_strength=${COUPLING_STRENGTH}"
echo "[sero-pci] occupancies=${OCCUPANCIES[*]}"
echo "[sero-pci] n_trial_seeds=${#TRIAL_SEEDS[@]}"
echo "[sero-pci] workers=${SLURM_CPUS_PER_TASK}"

python scripts/run_serotonergic_pci_full.py \
  "$@" \
  --dataset-root "${DATASET_ROOT}" \
  --baseline-root "${BASELINE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --scenario "${SCENARIO}" \
  --occupancies "${OCCUPANCIES[@]}" \
  --transient-ms 4000 \
  --t-analysis-ms 300 \
  --trial-sim-ms 8000 \
  --rate-monitor-period-ms 3 \
  --stim-amplitude 0.00030 \
  --stim-duration-ms 10 \
  --coupling-strength "${COUPLING_STRENGTH}" \
  --stim-onset-seed 0 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --trial-seeds "${TRIAL_SEEDS[@]}" \
  --stim-region-label Supp_Motor_Area_L \
  --receptor-tracer cimbi \
  --receptor-csv "${TVB_REPO}/data/receptors/hansen_receptors_aal90.csv" \
  --pci-binarise-method casali \
  --pci-significance-method pre_post_swap \
  --pci-permutation-replicates 1000 \
  --pci-alpha 0.05 \
  --pci-permutation-seed 0 \
  --pci-response-start-ms 8 \
  --pci-min-source-entropy 0.08 \
  --pci-st-baseline-window-ms -300 -50 \
  --pci-st-response-window-ms 8 300 \
  --pci-st-k 1.2 \
  --pci-st-min-snr 1.1 \
  --pci-st-max-var-percent 99 \
  --pci-st-n-steps 100 \
  --e-l-e-drug -61.2 \
  --e-l-i-drug -64.4 \
  --simulate-baseline

echo "[sero-pci] full serotonergic PCI run complete"
