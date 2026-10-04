#!/usr/bin/env bash
#SBATCH --job-name=tvb-fo-gnak50
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
: "${TVB_DATASET_ROOT:?Set TVB_DATASET_ROOT to shared_control_maximum_edge dataset}"
: "${GNAK_SIM_OUTPUT:?Set a new GNAK_SIM_OUTPUT directory}"
python experiments/first_order_gnak/simulate.py \
  --dataset-root "${TVB_DATASET_ROOT}" --output "${GNAK_SIM_OUTPUT}" \
  --workers "${SLURM_CPUS_PER_TASK}"
