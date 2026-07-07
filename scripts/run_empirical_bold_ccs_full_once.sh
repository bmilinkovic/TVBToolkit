#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/borjan/CNRS/projects/TVBToolkit"
CNRS_ROOT="${CNRS_DATA_ROOT:-/Volumes/ex_data/cnrs}"
PHIID_ROOT="${CNRS_ROOT}/data_doc_liege/results/phiid_empirical_bold"
PYTHON_BIN="/Users/borjan/miniconda3/bin/python3"
WORKERS="${1:-12}"
LOG_FILE="${2:-$PHIID_ROOT/logs/tvbtoolkit.phiid.ccs.once.log}"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] CCS one-shot wrapper starting"
echo "Root: $ROOT"
echo "Python: $PYTHON_BIN"
echo "Workers: $WORKERS"
echo "Log: $LOG_FILE"

cd "$ROOT"

"$PYTHON_BIN" "$ROOT/scripts/run_empirical_bold_phiid.py" \
  --redundancy ccs \
  --run-matlab \
  --require-complete \
  --matlab-parallel \
  --matlab-workers "$WORKERS"

"$PYTHON_BIN" "$ROOT/scripts/run_luppi2022_doc_downstream.py" \
  --phiid-root "$PHIID_ROOT/phiid/ccs" \
  --averages-root "$PHIID_ROOT/averages/ccs" \
  --output-root "$PHIID_ROOT/downstream_luppi2022/ccs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] CCS one-shot wrapper completed"
