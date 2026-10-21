#!/usr/bin/env bash
#SBATCH --job-name=tvb-della-full
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
: "${TVB_DATASET_ROOT:?Set TVB_DATASET_ROOT to the shared-control-maximum dataset}"
: "${DELLA_FULL_BANK:?Set DELLA_FULL_BANK to a new output directory}"
test -f "${DELLA_FULL_BANK}/manifest.json" || {
  echo "Prepare the Dalla Porta full-cohort bank before sbatch" >&2
  exit 2
}
# Each array element handles two cases, avoiding common SLURM 1,000-element
# array limits while keeping a single dependency job ID for downstream PCI.
for case_index in "${SLURM_ARRAY_TASK_ID}" "$((SLURM_ARRAY_TASK_ID + 716))"; do
  python experiments/della_porta_full_cohort/simulate.py case \
    --dataset "${TVB_DATASET_ROOT}" --bank "${DELLA_FULL_BANK}" \
    --case-index "${case_index}" --workers "${SLURM_CPUS_PER_TASK}"
done
