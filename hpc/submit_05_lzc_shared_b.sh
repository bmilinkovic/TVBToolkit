#!/usr/bin/env bash
#SBATCH --job-name=tvb05-lzc-shared
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh

B_TAG="${B_TAG:-b005}"
SCENARIO="${SCENARIO:-private_alpha0}"
case "${B_TAG}" in
  b005|b015|b025) ;;
  *)
    echo "ERROR: shared_b spontaneous is only complete enough for B_TAG=b005,b015,b025." >&2
    echo "       Got B_TAG=${B_TAG}" >&2
    exit 4
    ;;
esac
case "${SCENARIO}" in
  private_alpha0|global_alpha_005|global_alpha_010|global_alpha_015|global_alpha_020|global_alpha_025|global_alpha_030|global_alpha_035|global_alpha_040) ;;
  *)
    echo "ERROR: shared_b spontaneous common usable scenarios are private_alpha0 and global_alpha_005..global_alpha_040." >&2
    echo "       Got SCENARIO=${SCENARIO}" >&2
    exit 5
    ;;
esac

SIM_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/shared_b/sims"
OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/05_lzc_shared_b_${B_TAG}_${SCENARIO}"
DATASET_ROOT="$(resolve_tvb_dataset_root)"

echo "[05] dataset_root=${DATASET_ROOT}"

python notebooks/05_lzc_analysis_pub.py \
  --sim-root "${SIM_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --b-tag "${B_TAG}" \
  --scenario "${SCENARIO}" \
  "$@"
