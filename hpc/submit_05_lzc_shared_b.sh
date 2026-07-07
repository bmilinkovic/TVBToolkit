#!/usr/bin/env bash
#SBATCH --job-name=tvb05-lzc-shared
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh
CNRS_ROOT="${CNRS_DATA_ROOT:-/Volumes/ex_data/cnrs}"

python notebooks/05_lzc_analysis_pub.py \
  --sim-root "${CNRS_ROOT}/data_doc_liege/results/notebooks_outputs/ba_sim_hybrid/shared_b/sims" \
  --output-dir "${CNRS_ROOT}/data_doc_liege/results/notebooks_outputs/05_lzc_shared_b" \
  "$@"
