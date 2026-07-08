#!/usr/bin/env bash
#SBATCH --job-name=tvb04-bs-condb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
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

# Brain-state fitting uses sklearn KMeans; allow its OpenMP kernels to use the
# full single node requested by this job. Scenarios are processed sequentially
# within this job so this analysis occupies one node while LZc/PCI use others.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

SIM_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/condition_b/sims"
DATASET_ROOT="$(resolve_tvb_dataset_root)"

echo "[04] sim_root=${SIM_ROOT}"
echo "[04] dataset_root=${DATASET_ROOT}"
echo "[04] scenarios=${#CONDITION_B_SCENARIOS[@]}"
echo "[04] running as one exclusive single-node job; walltime is partition max"

for SCENARIO in "${CONDITION_B_SCENARIOS[@]}"; do
  OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/04_brain_states_condition_b_${SCENARIO}"
  echo "[04] START scenario=${SCENARIO} output_dir=${OUTPUT_DIR}"
  python notebooks/04_brain_states_analysis_pub.py \
    --sim-root "${SIM_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --scenario "${SCENARIO}" \
    "$@"
  echo "[04] DONE scenario=${SCENARIO}"
done

echo "[04] all condition_b brain-state scenarios complete"
