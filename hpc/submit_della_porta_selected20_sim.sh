#!/usr/bin/env bash
#SBATCH --job-name=tvb-della-20
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --array=0-159%1
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${TVB_DATASET_ROOT:?Set TVB_DATASET_ROOT to the shared-control-maximum dataset}"
: "${DELLA_SELECTED20_BANK:?Set DELLA_SELECTED20_BANK to a new output directory}"
test -f "${DELLA_SELECTED20_BANK}/manifest.json" || {
  echo "Prepare the 20-subject Dalla Porta bank before sbatch" >&2
  exit 2
}
python experiments/della_porta_selected20_hpc/simulate.py case \
  --dataset "${TVB_DATASET_ROOT}" --bank "${DELLA_SELECTED20_BANK}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"
