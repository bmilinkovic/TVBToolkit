#!/usr/bin/env bash
#SBATCH --job-name=tvb-sma8s-pci
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SMA8S_BANK:?Set SMA8S_BANK to the completed simulation bank}"
: "${SMA8S_PCI_ROOT:?Set SMA8S_PCI_ROOT to a distinct analysis root}"
methods=(hybrid tvbsim_two_sided_min)
method="${methods[$SLURM_ARRAY_TASK_ID]}"
python experiments/sma8s_full100/analyze.py \
  --bank "${SMA8S_BANK}" --output "${SMA8S_PCI_ROOT}/${method}" \
  --method "${method}" --workers "${SLURM_CPUS_PER_TASK}"
