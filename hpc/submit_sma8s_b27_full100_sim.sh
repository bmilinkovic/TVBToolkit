#!/usr/bin/env bash
#SBATCH --job-name=tvb-sma-2742
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --array=0-715%1
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${TVB_DATASET_ROOT:?Set TVB_DATASET_ROOT to converted_structural_invnodevol_control_max}"
: "${SMA8S_B27_BANK:?Set SMA8S_B27_BANK to a new, order-specific output directory}"
: "${SMA8S_B27_ORDER:?Set SMA8S_B27_ORDER to 1 or 2}"
case "${SMA8S_B27_ORDER}" in 1|2) ;; *) echo "SMA8S_B27_ORDER must be 1 or 2" >&2; exit 2;; esac
test -f "${SMA8S_B27_BANK}/manifest.json" || {
  echo "Run prepare before sbatch; manifest missing" >&2
  exit 2
}
python experiments/sma8s_b27_full100/simulate.py case \
  --order "${SMA8S_B27_ORDER}" \
  --dataset-root "${TVB_DATASET_ROOT}" --bank "${SMA8S_B27_BANK}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"
