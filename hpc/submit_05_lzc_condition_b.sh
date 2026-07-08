#!/usr/bin/env bash
#SBATCH --job-name=tvb05-lzc-condb
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

SIM_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/condition_b/sims"
OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/05_lzc_condition_b_${SCENARIO}"

echo "[05] condition_b scenario=${SCENARIO}"
echo "[05] sim_root=${SIM_ROOT}"
echo "[05] output_dir=${OUTPUT_DIR}"

python notebooks/05_lzc_analysis_pub.py \
  --sim-root "${SIM_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scenario "${SCENARIO}" \
  "$@"
