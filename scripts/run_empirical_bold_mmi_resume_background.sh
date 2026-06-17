#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/borjan/CNRS/projects/TVBToolkit"
INPUT_DIR="$ROOT/results/phiid_empirical_bold/inputs"
OUTPUT_DIR="$ROOT/results/phiid_empirical_bold/phiid/mmi"
LOG_DIR="$ROOT/results/phiid_empirical_bold/logs"
MATLAB_BIN="/Applications/MATLAB_R2023b.app/bin/matlab"
WORKERS="${1:-12}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/matlab_mmi_resume_${STAMP}.log"
PID_FILE="$LOG_DIR/matlab_mmi_resume_${STAMP}.pid"

CMD="addpath(genpath('/Users/borjan/code/matlab/elph')); addpath('$ROOT/scripts'); phiid_empirical_bold_aal90('$INPUT_DIR', '$OUTPUT_DIR', 'mmi', true, ${WORKERS})"

nohup caffeinate -dimsu "$MATLAB_BIN" -batch "$CMD" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Note: caffeinate prevents idle sleep, but a closed laptop lid can still sleep the Mac depending on system setup."
