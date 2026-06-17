#!/usr/bin/env bash
# Run on login node after environment setup and data transfer. Does not launch simulations.
set -euo pipefail
source hpc/slurm_env.sh

python -m py_compile \
  notebooks/02_full_noise_sims_rates_bold.py \
  notebooks/03_pci_trial_sims_hybrid.py \
  notebooks/04_brain_states_analysis_pub.py \
  notebooks/05_lzc_analysis_pub.py \
  notebooks/06_pci_analysis_pub.py \
  src/tvbtoolkit/complexity/measures.py

python notebooks/02_full_noise_sims_rates_bold.py --dry-run --workers 1 --output-root /tmp/tvb_hpc_02_dry_run
python notebooks/03_pci_trial_sims_hybrid.py --dry-run --workers 1 --output-root /tmp/tvb_hpc_03_dry_run

echo "Smoke test complete. If dry-run counts look correct, submit SLURM jobs."
