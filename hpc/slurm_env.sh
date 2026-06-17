#!/usr/bin/env bash
# Shared SLURM environment bootstrap for TVBToolkit jobs.
# Override these when submitting if needed, e.g.:
#   export TVB_REPO=/home/bmilinkovic/path/to/TVBToolkit
#   export TVB_CONDA_ENV=tvbtoolkit

set -euo pipefail

export TVB_REPO="${TVB_REPO:-/home/bmilinkovic/TVBToolkit}"
export TVB_CONDA_ENV="${TVB_CONDA_ENV:-tvbtoolkit}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TVB_REPO}/.cache/matplotlib}"
export TVB_USER_HOME="${TVB_USER_HOME:-${TVB_REPO}/.tvb-temp}"

mkdir -p "${MPLCONFIGDIR}" "${TVB_USER_HOME}"
cd "${TVB_REPO}"

if command -v module >/dev/null 2>&1; then
  module load anaconda/3.11 2>/dev/null || true
fi

# Make conda activation work in non-interactive SLURM shells.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${TVB_CONDA_ENV}"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${TVB_CONDA_ENV}"
else
  echo "ERROR: conda not found. Load anaconda/3.11 or install Miniconda first." >&2
  exit 2
fi

python --version
which python
