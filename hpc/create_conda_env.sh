#!/usr/bin/env bash
# Run once on the login node after transferring the repository.
set -euo pipefail

ENV_NAME="${TVB_CONDA_ENV:-tvbtoolkit}"

if command -v module >/dev/null 2>&1; then
  module load anaconda/3.11 2>/dev/null || true
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found." >&2
  exit 2
fi

conda create -y -n "${ENV_NAME}" python=3.11 pip
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
python - <<'PY'
import sys
print(sys.version)
import numpy, scipy, sklearn, matplotlib
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('sklearn', sklearn.__version__)
print('matplotlib', matplotlib.__version__)
PY
