#!/bin/bash
#SBATCH --job-name=hardware-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=hardware-analysis-%j.out
#SBATCH --error=hardware-analysis-%j.err
set -euo pipefail
TASK_ANALYSIS_DIR="${HARDWARE_ANALYSIS_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f "$TASK_ANALYSIS_DIR/hardware_study/run_capacity_analysis.py" ]]; then
  TASK_ANALYSIS_DIR="$TASK_ANALYSIS_DIR/hardware_study"
fi
if [[ ! -f "$TASK_ANALYSIS_DIR/run_capacity_analysis.py" ]]; then
  echo "Submit from release root or hardware_study, or set HARDWARE_ANALYSIS_DIR." >&2
  exit 2
fi
cd "$TASK_ANALYSIS_DIR"
if [[ -n "${EXPERIMENT_MODULES:-}" ]]; then
  read -r -a module_args <<< "$EXPERIMENT_MODULES"
  module load "${module_args[@]}"
fi
TASK_PROJECT="${PROJECT_DIR:-$TASK_ANALYSIS_DIR/..}"
TASK_STUDY="${STUDY_ROOT:-$TASK_PROJECT/outputs/study}"
TASK_OUTPUT="${HARDWARE_OUTPUT:-$TASK_PROJECT/outputs/hardware}"
: "${JOB_EFFICIENCY_CSV:?Set JOB_EFFICIENCY_CSV to the private original CSV}"
: "${GPU_CAPACITY_MAP:?Set GPU_CAPACITY_MAP to the external capacity registry}"
: "${GPU_MODEL_MAP:?Set GPU_MODEL_MAP to the external GPU model registry}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
extra_args=()
if [[ "${HISTORICAL_MAPPING_CONFIRMED:-0}" == 1 ]]; then
  extra_args+=(--historical-mapping-confirmed)
fi
if [[ -n "${REGISTRY_PROVENANCE:-}" ]]; then
  extra_args+=(--registry-provenance "$REGISTRY_PROVENANCE")
fi
python -u run_capacity_analysis.py \
  --project-root "$TASK_PROJECT" \
  --study-root "$TASK_STUDY" \
  --data-csv "$JOB_EFFICIENCY_CSV" \
  --capacity-map "$GPU_CAPACITY_MAP" \
  --model-map "$GPU_MODEL_MAP" \
  --capacity-unit "${CAPACITY_UNIT:-as registered}" \
  --output-dir "$TASK_OUTPUT" \
  --seeds 40 41 42 43 44 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 2026 \
  "${extra_args[@]}"
