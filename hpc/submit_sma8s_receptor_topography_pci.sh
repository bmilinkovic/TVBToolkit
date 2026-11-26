#!/usr/bin/env bash
#SBATCH --job-name=tvb-5ht2a-null-pci
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
: "${SMA8S_TOPO_BANK:?Set SMA8S_TOPO_BANK to the completed simulation bank}"
: "${SMA8S_TOPO_PCI:?Set SMA8S_TOPO_PCI to a new analysis directory}"
analysis_args=(--bank "${SMA8S_TOPO_BANK}" --output "${SMA8S_TOPO_PCI}" \
  --workers "${SLURM_CPUS_PER_TASK}")
if [[ "${SMA8S_TOPO_RETAIN_RECORDINGS:-0}" != 1 ]]; then
  analysis_args+=(--delete-recordings-after-analysis)
fi
python experiments/sma8s_receptor_topography/analyze.py \
  "${analysis_args[@]}"
