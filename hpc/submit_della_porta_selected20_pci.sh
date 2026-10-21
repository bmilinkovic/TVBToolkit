#!/usr/bin/env bash
#SBATCH --job-name=tvb-della-20-pci
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${DELLA_SELECTED20_BANK:?Set DELLA_SELECTED20_BANK to the completed bank}"
: "${DELLA_SELECTED20_PCI_ROOT:?Set DELLA_SELECTED20_PCI_ROOT to a separate analysis directory}"
python experiments/della_porta_second_order/analyze.py \
  --bank "${DELLA_SELECTED20_BANK}" --output "${DELLA_SELECTED20_PCI_ROOT}" \
  --nboot 1000
