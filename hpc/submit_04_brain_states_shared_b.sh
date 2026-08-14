#!/usr/bin/env bash
#SBATCH --job-name=tvb04-bs-shared
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

B_TAG="${B_TAG:-b010}"
SCENARIO="${SCENARIO:-private_alpha0}"
case "${B_TAG}" in
  b[0-9][0-9][0-9]|condb_*) ;;
  *)
    echo "ERROR: invalid B_TAG=${B_TAG}" >&2
    exit 4
    ;;
esac
case "${SCENARIO}" in
  private_alpha0|global_alpha_0[0-5][0-9]|sc_alpha_0[0-5][0-9]) ;;
  *)
    echo "ERROR: invalid SCENARIO=${SCENARIO}" >&2
    exit 5
    ;;
esac

SIM_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_native_invnodevol/shared_b/sims"
OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/04_brain_states_native_invnodevol_shared_b_${B_TAG}_${SCENARIO}"
DATASET_ROOT="$(resolve_tvb_dataset_root)"
require_native_invnodevol_dataset "${DATASET_ROOT}"

echo "[04] dataset_root=${DATASET_ROOT}"

python notebooks/04_brain_states_analysis_pub.py \
  --sim-root "${SIM_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --b-tag "${B_TAG}" \
  --scenario "${SCENARIO}" \
  "$@"
