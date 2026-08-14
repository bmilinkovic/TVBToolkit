#!/usr/bin/env bash
#SBATCH --job-name=b-lzc-robust
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh
DATASET_ROOT="$(resolve_tvb_dataset_root)"
require_native_invnodevol_dataset "${DATASET_ROOT}"
OUTPUT_ROOT="${B_LZC_OUTPUT_ROOT:-${TVB_REPO}/notebooks/outputs/candidate_b_lzc_robustness}"

python scripts/run_candidate_b_lzc_robustness.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "$@"
