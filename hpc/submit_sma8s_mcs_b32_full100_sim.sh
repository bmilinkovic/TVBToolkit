#!/usr/bin/env bash
#SBATCH --job-name=tvb-sma-mcs32
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --array=0-299%1
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${TVB_DATASET_ROOT:?Set TVB_DATASET_ROOT to converted_structural_invnodevol_control_max}"
: "${SMA8S_ORIGINAL_BANK:?Set SMA8S_ORIGINAL_BANK to the completed original first-order bank}"
: "${SMA8S_MCS_B32_BANK:?Set SMA8S_MCS_B32_BANK to a NEW bank directory}"
test -f "${SMA8S_MCS_B32_BANK}/manifest.json" || {
  echo "Run prepare before sbatch; manifest missing" >&2
  exit 2
}
python experiments/sma8s_mcs_b32_full100/simulate.py case \
  --dataset-root "${TVB_DATASET_ROOT}" \
  --reference-bank "${SMA8S_ORIGINAL_BANK}" \
  --bank "${SMA8S_MCS_B32_BANK}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}"
