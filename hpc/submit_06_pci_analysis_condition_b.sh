#!/usr/bin/env bash
#SBATCH --job-name=tvb06-pci-condb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh

CONDITION_B_SCENARIOS=(private_alpha0 global_alpha_025 sc_alpha_045)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${#CONDITION_B_SCENARIOS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} outside 0..$((${#CONDITION_B_SCENARIOS[@]} - 1))" >&2
    exit 6
  fi
  SCENARIO="${CONDITION_B_SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
else
  SCENARIO="${SCENARIO:-private_alpha0}"
fi

SIM_PCI_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/condition_b/sims_pci"
OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/06_pci_condition_b_${SCENARIO}"

echo "[06] condition_b scenario=${SCENARIO}"
echo "[06] sim_pci_root=${SIM_PCI_ROOT}"
echo "[06] output_dir=${OUTPUT_DIR}"

python notebooks/06_pci_analysis_pub.py \
  --sim-pci-root "${SIM_PCI_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scenario "${SCENARIO}" \
  --n-trials 100 \
  --min-trials 100 \
  "$@"
