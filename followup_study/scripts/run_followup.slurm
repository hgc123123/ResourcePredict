#!/bin/bash
#SBATCH --job-name=routing-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=routing-calibration-%j.out
#SBATCH --error=routing-calibration-%j.err

set -euo pipefail
TASK_FOLLOWUP="${FOLLOWUP_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f "$TASK_FOLLOWUP/followup_study/run_followup.py" ]]; then
  TASK_FOLLOWUP="$TASK_FOLLOWUP/followup_study"
fi
if [[ ! -f "$TASK_FOLLOWUP/run_followup.py" ]]; then
  echo "Submit from the release root or followup_study, or export FOLLOWUP_DIR." >&2
  exit 2
fi
cd "$TASK_FOLLOWUP"
if [[ -n "${EXPERIMENT_MODULES:-}" ]]; then
  read -r -a module_args <<< "$EXPERIMENT_MODULES"
  module load "${module_args[@]}"
fi
# CPU only: do not reserve GPUs and do not expose an existing training GPU.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

PROJECT_DIR="${PROJECT_DIR:-$PWD/..}"
STUDY_ROOT="${STUDY_ROOT:-$PROJECT_DIR/outputs/study}"
FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-$PROJECT_DIR/outputs/followup}"
FOLLOWUP_STAGE="${FOLLOWUP_STAGE:-all}"
FOLLOWUP_SEEDS="${FOLLOWUP_SEEDS:-40 41 42 43 44}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-10000}"
read -r -a seed_args <<< "$FOLLOWUP_SEEDS"
extra=()
if [[ -n "${JOB_EFFICIENCY_CSV:-}" ]]; then extra+=(--data-csv "$JOB_EFFICIENCY_CSV"); fi
if [[ -n "${SPLIT_MANIFEST:-}" ]]; then extra+=(--split-manifest "$SPLIT_MANIFEST"); fi
if [[ "${RESUME:-1}" == 1 ]]; then extra+=(--resume); fi
python run_followup.py \
  --project-dir "$PROJECT_DIR" --study-root "$STUDY_ROOT" \
  --output-root "$FOLLOWUP_ROOT" --stage "$FOLLOWUP_STAGE" \
  --seeds "${seed_args[@]}" --variants full no_code \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" "${extra[@]}"
