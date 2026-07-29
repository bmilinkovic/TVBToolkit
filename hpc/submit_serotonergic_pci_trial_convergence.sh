#!/usr/bin/env bash
#SBATCH --job-name=tvb-sero-pci-conv
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs

# shellcheck source=hpc/slurm_env.sh
source hpc/slurm_env.sh

# This job only reads the corrected simulation cache. All checkpoints, tables,
# logs, and figures are written to the separate convergence output root.
CACHE_ROOT="${SEROTONERGIC_PCI_CACHE_ROOT:-${TVB_REPO}/notebooks/outputs/serotonergic_pci_full_100trials_corrected}"
OUTPUT_ROOT="${SEROTONERGIC_PCI_CONVERGENCE_OUTPUT_ROOT:-${TVB_REPO}/notebooks/outputs/serotonergic_pci_trial_convergence}"
SUBJECTS_PER_COHORT="${SEROTONERGIC_PCI_CONVERGENCE_SUBJECTS_PER_COHORT:-1}"
REPEATS="${SEROTONERGIC_PCI_CONVERGENCE_REPEATS:-5}"
SELECTION_SEED="${SEROTONERGIC_PCI_CONVERGENCE_SELECTION_SEED:-0}"
SUBSET_SEED="${SEROTONERGIC_PCI_CONVERGENCE_SUBSET_SEED:-0}"
WORKERS="${SLURM_CPUS_PER_TASK:-1}"

echo "[pci-convergence] immutable_cache_root=${CACHE_ROOT}"
echo "[pci-convergence] separate_output_root=${OUTPUT_ROOT}"
echo "[pci-convergence] subjects_per_cohort=${SUBJECTS_PER_COHORT}"
echo "[pci-convergence] repeats=${REPEATS}"
echo "[pci-convergence] workers=${WORKERS}"

python scripts/analyze_serotonergic_pci_trial_convergence.py \
  --cache-root "${CACHE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --cohorts coma uws mcs emcs control \
  --subjects-per-cohort "${SUBJECTS_PER_COHORT}" \
  --selection-seed "${SELECTION_SEED}" \
  --subset-seed "${SUBSET_SEED}" \
  --subset-sizes 20 40 60 80 \
  --repeats "${REPEATS}" \
  --pci-significance-method pre_post_swap \
  --pci-permutation-replicates 1000 \
  --pci-alpha 0.05 \
  --pci-seed 0 \
  --pci-response-start-ms 8 \
  --pci-min-source-entropy 0.08 \
  --pci-st-baseline-window-ms -300 -50 \
  --pci-st-response-window-ms 0 300 \
  --pci-st-k 1.2 \
  --pci-st-min-snr 1.1 \
  --pci-st-max-var-percent 99 \
  --pci-st-n-steps 100 \
  --workers "${WORKERS}" \
  "$@"

echo "[pci-convergence] analysis complete"
