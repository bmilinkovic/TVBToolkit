#!/usr/bin/env bash
#SBATCH --job-name=tvb-della-pci
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
: "${DELLA_FULL_BANK:?Set DELLA_FULL_BANK to the completed recording bank}"
: "${DELLA_FULL_PCI_ROOT:?Set DELLA_FULL_PCI_ROOT to a separate analysis directory}"
python experiments/della_porta_full_cohort/analyze.py \
  --bank "${DELLA_FULL_BANK}" --output "${DELLA_FULL_PCI_ROOT}" \
  --workers "${SLURM_CPUS_PER_TASK}"
