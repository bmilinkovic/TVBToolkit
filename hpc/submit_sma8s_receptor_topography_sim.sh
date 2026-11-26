#!/usr/bin/env bash
#SBATCH --job-name=tvb-5ht2a-null
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${TVB_DATASET_ROOT:?Set TVB_DATASET_ROOT to the shared-control-maximum dataset}"
: "${SMA8S_TOPO_BANK:?Set SMA8S_TOPO_BANK to a new, empty result directory}"
test -f "${SMA8S_TOPO_BANK}/manifest.json" || {
  echo "Run prepare before sbatch; manifest missing" >&2
  exit 2
}
python experiments/sma8s_receptor_topography/simulate.py case \
  --dataset-root "${TVB_DATASET_ROOT}" --bank "${SMA8S_TOPO_BANK}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"
