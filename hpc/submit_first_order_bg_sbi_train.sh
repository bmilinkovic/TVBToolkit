#!/usr/bin/env bash
#SBATCH --job-name=tvb-bg-inference
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err
# Separate from forward simulations; CPU resources match existing cluster scripts.
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
: "${BG_SBI_BANK:?Set prepared participant bank directory}"
: "${BG_SBI_BUDGET:?Set completed simulation budget to use for training}"
python experiments/first_order_bg_sbi/workflow.py train \
  --bank "${BG_SBI_BANK}" --budget "${BG_SBI_BUDGET}" \
  --training-seed "${BG_SBI_TRAINING_SEED:-1}" --threads "${SLURM_CPUS_PER_TASK}"
