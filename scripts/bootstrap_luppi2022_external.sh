#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXT="$ROOT/external/luppi2022"

mkdir -p "$EXT"

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "present: $dest"
    return 0
  fi
  echo "cloning: $url -> $dest"
  git clone "$url" "$dest"
}

clone_if_missing "https://github.com/gpreti/GSP_StructuralDecouplingIndex.git" \
  "$EXT/GSP_StructuralDecouplingIndex"
clone_if_missing "https://github.com/frantisekvasa/rotate_parcellation.git" \
  "$EXT/rotate_parcellation"

cat <<EOF

External Luppi 2022 resources prepared under:
  $EXT

Notes:
  - rsHRF is not cloned here because the paper cites a NITRC distribution rather than a GitHub repo.
  - Brain Connectivity Toolbox is not cloned here because the current plan prefers local Python graph metrics first,
    with optional BCT validation later.
EOF
