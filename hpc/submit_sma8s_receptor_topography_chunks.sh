#!/usr/bin/env bash
# Submit manageable 500-task sequential chunks; avoids site MaxArraySize limits.
# Requires prepare and the exported environment variables in README.
set -euo pipefail
: "${SMA8S_TOPO_BANK:?Set SMA8S_TOPO_BANK to the prepared result bank}"
test -f "${SMA8S_TOPO_BANK}/manifest.json" || {
  echo "Prepare the bank before submitting chunks" >&2
  exit 2
}
case_count="$(python - "${SMA8S_TOPO_BANK}/manifest.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["cases"]))
PY
)"
chunk_size="${SMA8S_TOPO_CHUNK_SIZE:-500}"
if (( chunk_size < 1 || chunk_size > 500 )); then
  echo "SMA8S_TOPO_CHUNK_SIZE must be 1..500" >&2
  exit 2
fi
dry_run="${SMA8S_TOPO_DRY_RUN:-0}"
if [[ "$dry_run" != 0 && "$dry_run" != 1 ]]; then
  echo "SMA8S_TOPO_DRY_RUN must be 0 or 1" >&2
  exit 2
fi
previous=""
for ((start=0; start<case_count; start+=chunk_size)); do
  end=$((start + chunk_size - 1))
  if (( end >= case_count )); then end=$((case_count - 1)); fi
  if [[ "$dry_run" == 1 ]]; then
    printf 'Would submit cases %s-%s (array %s-%s%%1)\n' "$start" "$end" "$start" "$end"
    continue
  fi
  if [[ -z "${previous}" ]]; then
    job="$(sbatch --parsable --array="${start}-${end}%1" \
      hpc/submit_sma8s_receptor_topography_sim.sh)"
  else
    job="$(sbatch --parsable --dependency="afterok:${previous}" \
      --array="${start}-${end}%1" hpc/submit_sma8s_receptor_topography_sim.sh)"
  fi
  previous="${job%%;*}"
  printf 'Submitted cases %s-%s as job %s\n' "$start" "$end" "$previous"
done
if [[ "$dry_run" == 1 ]]; then
  printf 'DRY_RUN: %s cases, %s chunks; no jobs submitted\n' "$case_count" \
    "$(( (case_count + chunk_size - 1) / chunk_size ))"
  exit 0
fi
printf 'LAST_SIM_JOB_ID=%s\n' "$previous"
