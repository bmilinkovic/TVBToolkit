#!/usr/bin/env bash
#SBATCH --job-name=tvb-sma8s-2gag1
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SMA8S_SECOND_ORDER_BANK:?Set SMA8S_SECOND_ORDER_BANK to the completed second-order SMA bank}"
: "${SMA8S_SECOND_ORDER_PCI_ROOT:?Set SMA8S_SECOND_ORDER_PCI_ROOT to its separate PCI analysis root}"
python experiments/sma8s_full100/score_original_per_trial.py \
  --bank "${SMA8S_SECOND_ORDER_BANK}" \
  --output "${SMA8S_SECOND_ORDER_PCI_ROOT}/gaglioti_per_trial" \
  --workers "${SLURM_CPUS_PER_TASK}"
