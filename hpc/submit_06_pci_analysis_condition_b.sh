#!/usr/bin/env bash
#SBATCH --job-name=tvb06-pci-condb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh

CONDITION_B_SCENARIOS=(private_alpha0 global_alpha_025 sc_alpha_045)

SIM_PCI_ROOT="${TVB_REPO}/notebooks/outputs/ba_sim_hybrid/condition_b/sims_pci"

echo "[06] sim_pci_root=${SIM_PCI_ROOT}"
echo "[06] scenarios=${#CONDITION_B_SCENARIOS[@]}"
echo "[06] running as one exclusive single-node job; walltime is partition max"

for SCENARIO in "${CONDITION_B_SCENARIOS[@]}"; do
  OUTPUT_DIR="${TVB_REPO}/notebooks/outputs/06_pci_condition_b_${SCENARIO}"
  echo "[06] START scenario=${SCENARIO} output_dir=${OUTPUT_DIR}"
  python notebooks/06_pci_analysis_pub.py \
    --sim-pci-root "${SIM_PCI_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --scenario "${SCENARIO}" \
    --n-trials 100 \
    --min-trials 100 \
    "$@"
  echo "[06] DONE scenario=${SCENARIO}"
done

echo "[06] all condition_b PCI scenarios complete"
