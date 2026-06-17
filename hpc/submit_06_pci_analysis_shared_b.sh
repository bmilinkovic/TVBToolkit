#!/usr/bin/env bash
#SBATCH --job-name=tvb06-pci-shared
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh

python notebooks/06_pci_analysis_pub.py \
  --sim-pci-root notebooks/outputs/ba_sim_hybrid/shared_b/sims_pci \
  --output-dir notebooks/outputs/06_pci_shared_b \
  "$@"
