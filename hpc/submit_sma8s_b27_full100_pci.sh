#!/usr/bin/env bash
#SBATCH --job-name=tvb-pci-2742
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SMA8S_B27_BANK:?Set SMA8S_B27_BANK to a completed b27 recording bank}"
: "${SMA8S_B27_PCI_ROOT:?Set SMA8S_B27_PCI_ROOT to a new PCI analysis root}"
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    method=hybrid
    python experiments/sma8s_full100/analyze.py \
      --bank "${SMA8S_B27_BANK}" --output "${SMA8S_B27_PCI_ROOT}/${method}" \
      --method "${method}" --workers "${SLURM_CPUS_PER_TASK}"
    ;;
  1)
    method=tvbsim_two_sided_min
    python experiments/sma8s_full100/analyze.py \
      --bank "${SMA8S_B27_BANK}" --output "${SMA8S_B27_PCI_ROOT}/${method}" \
      --method "${method}" --workers "${SLURM_CPUS_PER_TASK}"
    ;;
  2)
    python experiments/sma8s_full100/score_original_per_trial.py \
      --bank "${SMA8S_B27_BANK}" \
      --output "${SMA8S_B27_PCI_ROOT}/gaglioti_per_trial" \
      --workers "${SLURM_CPUS_PER_TASK}"
    ;;
  *) echo "Unexpected PCI array index" >&2; exit 2;;
esac
