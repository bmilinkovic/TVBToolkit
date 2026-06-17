#!/usr/bin/env bash
#SBATCH --job-name=tvb02-condb
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

python notebooks/02_full_noise_sims_rates_bold.py \
  --sweep-mode condition_b \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "$@"
