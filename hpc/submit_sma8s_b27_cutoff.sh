#!/usr/bin/env bash
#SBATCH --job-name=tvb-cut-2742
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --error=hpc/logs/%x-%j.err
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
: "${SMA8S_B27_PCI_ROOT:?Set SMA8S_B27_PCI_ROOT to the completed PCI analysis root}"
summary="${SMA8S_B27_PCI_ROOT}/hybrid/subject_summary.csv"
test -f "${SMA8S_B27_PCI_ROOT}/hybrid/completion.json" || {
  echo "Hybrid PCI must finish before cutoff analysis" >&2
  exit 2
}
python experiments/sma8s_b27_full100/cutoff.py \
  --input-csv "${summary}" \
  --output "${SMA8S_B27_PCI_ROOT}/exploratory_cutoff_hybrid"
