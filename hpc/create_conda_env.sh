#!/usr/bin/env bash
# Run once on the login node after transferring the repository.
set -euo pipefail

ENV_NAME="${TVB_CONDA_ENV:-tvbtoolkit}"
ENV_PREFIX="${TVB_CONDA_PREFIX:-${HOME}/.conda/envs/${ENV_NAME}}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  if command -v module >/dev/null 2>&1; then
    module load anaconda/3.11 2>/dev/null || true
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found." >&2
    exit 2
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if [ -d "${ENV_PREFIX}" ]; then
  echo "Using existing Conda environment: ${ENV_PREFIX}"
else
  conda create -y -p "${ENV_PREFIX}" python=3.11 pip
fi
conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
python - <<'PY'
import sys
print(sys.version)
import numpy, scipy, sklearn, matplotlib, pandas, tvb
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('sklearn', sklearn.__version__)
print('matplotlib', matplotlib.__version__)
print('pandas', pandas.__version__)
print('tvb', tvb.__file__)
PY
