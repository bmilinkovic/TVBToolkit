#!/usr/bin/env bash
#SBATCH --job-name=tvb-bg-four
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --array=0-3%1
#SBATCH --output=hpc/logs/%x-%A_%a.out
#SBATCH --error=hpc/logs/%x-%A_%a.err
# Four subject-specific banks, 10,000 draws each. %1 uses one 48-CPU node at a
# time; override with sbatch --array=0-3%4 only if four nodes are desired/available.
set -euo pipefail
source "${TVB_REPO:-/home/bmilinkovic/TVBToolkit}/hpc/slurm_env.sh"
: "${TVB_DATASET_ROOT:?Set normalized structural dataset location}"
: "${BG_SBI_PILOT:?Set transferred pilot folder containing pilot.json and banks}"
# Reject superseded pilot manifests rather than quietly submitting the old
# state-feature/provisional-TR configuration. Current TR was confirmed by user.
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("tr_confirmed") and p.get("state_features") is False and p.get("retained_seconds")==120 and p.get("storage_policy")=="summaries_only", "Run prepare_cluster.py for the current summaries-only pilot"' \
  "${BG_SBI_PILOT}/pilot.json"
BG_BANK_NAME=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["jobs"][int(sys.argv[2])]["bank"])' \
  "${BG_SBI_PILOT}/pilot.json" "${SLURM_ARRAY_TASK_ID}")
case "${BG_STAGE:-simulate}" in
  simulate)
    python experiments/first_order_bg_sbi/workflow.py simulate \
      --bank "${BG_SBI_PILOT}/${BG_BANK_NAME}" --dataset-root "${TVB_DATASET_ROOT}" \
      --stop "${BG_SBI_STOP:-10000}" --workers "${SLURM_CPUS_PER_TASK}"
    ;;
  train)
    python experiments/first_order_bg_sbi/workflow.py train \
      --bank "${BG_SBI_PILOT}/${BG_BANK_NAME}" --budget "${BG_SBI_BUDGET:-10000}" \
      --training-seed "${BG_SBI_TRAINING_SEED:-1}" --threads "${SLURM_CPUS_PER_TASK}"
    ;;
  report)
    python experiments/first_order_bg_sbi/workflow.py report \
      --bank "${BG_SBI_PILOT}/${BG_BANK_NAME}" --budget "${BG_SBI_BUDGET:-10000}"
    ;;
  *) echo 'BG_STAGE must be simulate, report or train' >&2; exit 2 ;;
esac
