#!/usr/bin/env bash
# Shared SLURM environment bootstrap for TVBToolkit jobs.
# Override these when submitting if needed, e.g.:
#   export TVB_REPO=/home/bmilinkovic/path/to/TVBToolkit
#   export TVB_CONDA_ENV=tvbtoolkit

set -euo pipefail

export TVB_REPO="${TVB_REPO:-/home/bmilinkovic/TVBToolkit}"
export TVB_CONDA_ENV="${TVB_CONDA_ENV:-tvbtoolkit}"
export TVB_CONDA_PREFIX="${TVB_CONDA_PREFIX:-${HOME}/.conda/envs/${TVB_CONDA_ENV}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TVB_REPO}/.cache/matplotlib}"
export TVB_USER_HOME="${TVB_USER_HOME:-${TVB_REPO}/.tvb-temp}"

mkdir -p "${MPLCONFIGDIR}" "${TVB_USER_HOME}"
cd "${TVB_REPO}"

# Make conda activation work in non-interactive SLURM shells.
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
    echo "ERROR: conda not found. Load anaconda/3.11 or install Miniconda first." >&2
    exit 2
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if [ -x "${TVB_CONDA_PREFIX}/bin/python" ]; then
  conda activate "${TVB_CONDA_PREFIX}"
else
  conda activate "${TVB_CONDA_ENV}"
fi

python --version
which python
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"

case "$(which python)" in
  "${CONDA_PREFIX:-__missing__}"/bin/python) ;;
  *)
    echo "ERROR: python is not coming from the active Conda environment." >&2
    echo "       which python: $(which python)" >&2
    echo "       CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}" >&2
    exit 3
    ;;
esac
