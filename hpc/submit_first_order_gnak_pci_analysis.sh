#!/usr/bin/env bash
#SBATCH --job-name=tvb-fo-gnak-pci
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
: "${GNAK_SIM_OUTPUT:?Set GNAK_SIM_OUTPUT to completed simulation directory}"
: "${GNAK_PCI_OUTPUT:?Set a separate GNAK_PCI_OUTPUT directory}"
python experiments/first_order_gnak/analyze.py \
  --input "${GNAK_SIM_OUTPUT}" --output "${GNAK_PCI_OUTPUT}" \
  --workers "${SLURM_CPUS_PER_TASK}"
