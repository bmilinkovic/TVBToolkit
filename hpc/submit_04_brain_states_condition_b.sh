#!/usr/bin/env bash
#SBATCH --job-name=tvb04-bs-condb
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

python notebooks/04_brain_states_analysis_pub.py \
  --sim-root notebooks/outputs/ba_sim_hybrid/condition_b/sims \
  --output-dir notebooks/outputs/04_brain_states_condition_b \
  "$@"
