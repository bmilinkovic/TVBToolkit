#!/usr/bin/env bash
#SBATCH --job-name=tvb-sma8s-100
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
: "${SMA8S_BANK:?Set SMA8S_BANK to a new output directory}"
test -f "${SMA8S_BANK}/manifest.json" || {
  echo "Run the prepare command before sbatch; manifest missing" >&2
  exit 2
}
python experiments/sma8s_full100/simulate.py case \
  --dataset-root "${TVB_DATASET_ROOT}" --bank "${SMA8S_BANK}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"
