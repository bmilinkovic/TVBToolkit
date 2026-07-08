#!/usr/bin/env bash
#SBATCH --job-name=tvb05-lzc-condb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh

CONDITION_B_SCENARIOS=(
  private_alpha0
  global_alpha_005
  global_alpha_010
  global_alpha_015
  global_alpha_020
  global_alpha_025
  global_alpha_030
  global_alpha_035
  global_alpha_040
  global_alpha_045
  global_alpha_050
  sc_alpha_005
  sc_alpha_010
  sc_alpha_015
  sc_alpha_020
  sc_alpha_025
  sc_alpha_030
  sc_alpha_035
  sc_alpha_040
  sc_alpha_045
  sc_alpha_050
)

SIM_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/condition_b/sims"

echo "[05] sim_root=${SIM_ROOT}"
echo "[05] scenarios=${#CONDITION_B_SCENARIOS[@]}"
echo "[05] running as one exclusive single-node job; walltime is partition max"

for SCENARIO in "${CONDITION_B_SCENARIOS[@]}"; do
  OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/05_lzc_condition_b_${SCENARIO}"
  echo "[05] START scenario=${SCENARIO} output_dir=${OUTPUT_DIR}"
  python notebooks/05_lzc_analysis_pub.py \
    --sim-root "${SIM_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --scenario "${SCENARIO}" \
    "$@"
  echo "[05] DONE scenario=${SCENARIO}"
done

echo "[05] all condition_b LZc scenarios complete"
