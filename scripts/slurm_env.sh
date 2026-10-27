#!/usr/bin/env bash
# Source this from a Slurm job submitted in the release root.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
test -f scripts/run_publication_experiments.py || {
  echo 'Submit from 20261026, or set PROJECT_DIR to its absolute path.' >&2
  exit 2
}
if [[ -n "${EXPERIMENT_MODULES:-}" ]]; then
  read -r -a requested_modules <<< "$EXPERIMENT_MODULES"
  module load "${requested_modules[@]}"
fi
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
resume_flags=()
if [[ "${RESUME:-1}" == 1 ]]; then resume_flags=(--resume); fi
output_flags=(--output-root "${EXPERIMENT_OUTPUT_ROOT:-$PROJECT_DIR/outputs}")
