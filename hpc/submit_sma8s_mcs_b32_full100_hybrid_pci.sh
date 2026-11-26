#!/usr/bin/env bash
#SBATCH --job-name=tvb-pci-mcs32
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
: "${SMA8S_MCS_B32_BANK:?Set SMA8S_MCS_B32_BANK to the completed new bank}"
: "${SMA8S_MCS_B32_PCI:?Set SMA8S_MCS_B32_PCI to a NEW hybrid PCI directory}"
python experiments/sma8s_mcs_b32_full100/analyze.py \
  --bank "${SMA8S_MCS_B32_BANK}" \
  --output "${SMA8S_MCS_B32_PCI}" \
  --workers "${SLURM_CPUS_PER_TASK}"
