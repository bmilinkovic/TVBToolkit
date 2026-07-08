#!/usr/bin/env bash
#SBATCH --job-name=tvb06-pci-shared
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
    echo "ERROR: shared_b PCI is complete for B_TAG=b005,b015,b025 only in the current archive." >&2
    echo "       Got B_TAG=${B_TAG}" >&2
    exit 4
    ;;
esac
case "${SCENARIO}" in
  private_alpha0|global_alpha_025|global_alpha_045|sc_alpha_025|sc_alpha_045) ;;
  *)
    echo "ERROR: shared_b PCI complete scenarios are private_alpha0, global_alpha_025, global_alpha_045, sc_alpha_025, sc_alpha_045." >&2
    echo "       Got SCENARIO=${SCENARIO}" >&2
    exit 5
    ;;
esac

SIM_PCI_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/shared_b/sims_pci"
OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/06_pci_shared_b_${B_TAG}_${SCENARIO}"

python notebooks/06_pci_analysis_pub.py \
  --sim-pci-root "${SIM_PCI_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --b-tag "${B_TAG}" \
  --scenario "${SCENARIO}" \
  --n-trials 100 \
  --min-trials 100 \
  "$@"
