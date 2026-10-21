#!/usr/bin/env bash
#SBATCH --job-name=tvb-della-sma-pci
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
: "${SMA8S_BANK:?Set SMA8S_BANK to a completed first- or second-order SMA full100 bank}"
: "${DELLA_ON_SMA_OUTPUT:?Set DELLA_ON_SMA_OUTPUT to a distinct PCI analysis directory}"
python experiments/della_porta_full_cohort/score_sma_bank.py \
  --bank "${SMA8S_BANK}" --output "${DELLA_ON_SMA_OUTPUT}" \
  --workers "${SLURM_CPUS_PER_TASK}"
