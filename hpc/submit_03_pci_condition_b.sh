#!/usr/bin/env bash
#SBATCH --job-name=tvb03-pci-condb
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
DATASET_ROOT="$(resolve_tvb_dataset_root)"

echo "[03] dataset_root=${DATASET_ROOT}"

python notebooks/03_pci_trial_sims_hybrid.py \
  --dataset-root "${DATASET_ROOT}" \
  --sweep-mode condition_b \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "$@"
