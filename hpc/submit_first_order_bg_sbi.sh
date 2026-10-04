#!/usr/bin/env bash
#SBATCH --job-name=tvb-bg-sbi
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err
# Submit from repository root after mkdir -p hpc/logs.
# The bank must already be prepared with labelled empirical BOLD.
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${TVB_DATASET_ROOT:?Set normalized structural dataset location}"
: "${BG_SBI_BANK:?Set prepared participant bank directory}"
# Default is a small benchmark; a large budget requires an explicit override.
python experiments/first_order_bg_sbi/workflow.py simulate \
  --bank "${BG_SBI_BANK}" --dataset-root "${TVB_DATASET_ROOT}" \
  --start "${BG_SBI_START:-0}" --stop "${BG_SBI_STOP:-16}" \
  --workers "${SLURM_CPUS_PER_TASK}"
