#!/usr/bin/env bash
#SBATCH --job-name=tvb-sero-emcs-robust
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

# shellcheck source=hpc/slurm_env.sh
source hpc/slurm_env.sh
DATASET_ROOT="$(resolve_tvb_dataset_root)"

OUTPUT_ROOT="${SEROTONERGIC_ROBUSTNESS_OUTPUT_ROOT:-${TVB_REPO}/notebooks/outputs/serotonergic_pci_emcs_robustness_100trials}"
SUBJECT_ID="e0001"
SCENARIO="private_alpha0"
OCCUPANCIES=(0 0.25 0.5 0.766)

TRIAL_SEEDS=($(seq 0 99))
EXPECTED_TRIAL_SEEDS=($(seq 0 99))
if [ "${TRIAL_SEEDS[*]}" != "${EXPECTED_TRIAL_SEEDS[*]}" ]; then
  echo "ERROR: corrected robustness protocol requires trial seeds 0..99 in order." >&2
  exit 4
fi

echo "[sero-pci-emcs-robustness] dataset_root=${DATASET_ROOT}"
echo "[sero-pci-emcs-robustness] output_root=${OUTPUT_ROOT}"
echo "[sero-pci-emcs-robustness] subject_id=${SUBJECT_ID}"
echo "[sero-pci-emcs-robustness] scenario=${SCENARIO}"
echo "[sero-pci-emcs-robustness] occupancies=${OCCUPANCIES[*]}"
echo "[sero-pci-emcs-robustness] n_trial_seeds=${#TRIAL_SEEDS[@]}"
echo "[sero-pci-emcs-robustness] workers=${SLURM_CPUS_PER_TASK}"

python scripts/run_serotonergic_pci_emcs_robustness.py \
  "$@" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --subject-id "${SUBJECT_ID}" \
  --scenario "${SCENARIO}" \
  --occupancies "${OCCUPANCIES[@]}" \
  --transient-ms 4000 \
  --t-analysis-ms 300 \
  --trial-sim-ms 8000 \
  --stim-amplitude 0.00030 \
  --stim-duration-ms 10 \
  --stim-onset-seed 0 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --trial-seeds "${TRIAL_SEEDS[@]}" \
  --stim-region-label Supp_Motor_Area_L \
  --receptor-tracer cimbi \
  --receptor-csv "${TVB_REPO}/data/receptors/hansen_receptors_aal90.csv" \
  --pci-binarise-method casali \
  --pci-bootstrap-replicates 500 \
  --pci-alpha 0.01 \
  --pci-bootstrap-seed 0 \
  --e-l-e-drug -61.2 \
  --e-l-i-drug -64.4 \
  --variation-fraction 0.20

echo "[sero-pci-emcs-robustness] run complete"
