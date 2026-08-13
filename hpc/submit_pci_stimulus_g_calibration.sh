#!/usr/bin/env bash
#SBATCH --job-name=pci-stim-g
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh
DATASET_ROOT="$(resolve_tvb_dataset_root)"
OUTPUT_ROOT="${PCI_CALIBRATION_OUTPUT_ROOT:-${TVB_REPO}/notebooks/outputs/pci_stimulus_g_calibration}"

echo "[pci-calibration] dataset_root=${DATASET_ROOT}"
echo "[pci-calibration] output_root=${OUTPUT_ROOT}"
require_native_invnodevol_dataset "${DATASET_ROOT}"
echo "[pci-calibration] left-SMA target; control + MCS + UWS; diagnosis b_e retained"

python scripts/calibrate_pci_stimulus_g_homeostasis.py \
  "$@" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --occupancies 0 0.766 \
  --trial-seeds 0 1 2 \
  --pulse-shapes square raised_cosine gaussian \
  --durations-ms 1 5 10 \
  --amplitudes-khz 0.00010 0.00020 0.00030 0.00050 \
  --g-values 0.0025 0.005 0.01 0.025 0.05 \
  --reference-g 0.01 \
  --monitor-period-ms 3 \
  --response-start-ms 8 \
  --stim-region-label Supp_Motor_Area_L \
  --homeostasis off \
  --homeostatic-target baseline \
  --homeostatic-epochs 6 \
  --homeostatic-epoch-ms 1000 \
  --homeostatic-tau-s 2 \
  --homeostatic-detector-tau-ms 50 \
  --homeostatic-activation-sd 5 \
  --homeostatic-post-ms 1500 \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "[pci-calibration] complete; figures are in ${OUTPUT_ROOT}/figures"
