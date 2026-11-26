#!/usr/bin/env bash
#SBATCH --job-name=tvb-merge-mcs32
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SMA8S_ORIGINAL_BANK:?Set SMA8S_ORIGINAL_BANK}"
: "${SMA8S_ORIGINAL_HYBRID_PCI:?Set SMA8S_ORIGINAL_HYBRID_PCI}"
: "${SMA8S_MCS_B32_BANK:?Set SMA8S_MCS_B32_BANK}"
: "${SMA8S_MCS_B32_PCI:?Set SMA8S_MCS_B32_PCI}"
: "${SMA8S_MCS_B32_COMBINED:?Set SMA8S_MCS_B32_COMBINED}"
python experiments/sma8s_mcs_b32_full100/merge.py \
  --original-bank "${SMA8S_ORIGINAL_BANK}" \
  --original-pci "${SMA8S_ORIGINAL_HYBRID_PCI}" \
  --mcs-bank "${SMA8S_MCS_B32_BANK}" \
  --mcs-pci "${SMA8S_MCS_B32_PCI}" \
  --output "${SMA8S_MCS_B32_COMBINED}"
