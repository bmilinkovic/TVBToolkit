#!/usr/bin/env bash
#SBATCH --job-name=momi-lap-pci
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err

set -euo pipefail
mkdir -p hpc/logs
source hpc/slurm_env.sh
DATASET_ROOT="$(resolve_tvb_dataset_root)"
OUTPUT_ROOT="${MOMI_LAPLACIAN_PCI_OUTPUT_ROOT:-${TVB_REPO}/results/momi_laplacian_pci_two_subject}"

require_native_invnodevol_dataset "${DATASET_ROOT}"
python experiments/momi_laplacian_pci/run_two_subject_robustness.py \
  "$@" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --subject control:c0015 \
  --subject uws:u0001 \
  --subject uws:u0020 \
  --trial-seeds 0 1 2 3 4 5 6 7 8 9 \
  --workers "${SLURM_CPUS_PER_TASK}"
