#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/borjan/CNRS/projects/TVBToolkit"
CNRS_ROOT="${CNRS_DATA_ROOT:-/Volumes/ex_data/cnrs}"
PHIID_ROOT="${CNRS_ROOT}/data_doc_liege/results/phiid_empirical_bold"
INPUT_DIR="$PHIID_ROOT/inputs"
OUTPUT_DIR="$PHIID_ROOT/phiid/mmi"
WORKERS="${1:-12}"
LOG_FILE="${2:?log file path required}"
MATLAB_BIN="/Applications/MATLAB_R2023b.app/bin/matlab"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launchd wrapper starting"
  echo "Workers: $WORKERS"
  echo "Input: $INPUT_DIR"
  echo "Output: $OUTPUT_DIR"
} >>"$LOG_FILE"

exec /usr/bin/caffeinate -dimsu "$MATLAB_BIN" -batch \
  "addpath(genpath('/Users/borjan/code/matlab/elph')); addpath('$ROOT/scripts'); phiid_empirical_bold_aal90('$INPUT_DIR', '$OUTPUT_DIR', 'mmi', true, ${WORKERS})" \
  >>"$LOG_FILE" 2>&1
