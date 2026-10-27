#!/usr/bin/env bash
#SBATCH --job-name=hpc-primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/primary-%j.out
#SBATCH --error=logs/primary-%j.err
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm_env.sh"
: "${JOB_EFFICIENCY_CSV:?Set JOB_EFFICIENCY_CSV to the external private CSV}"
: "${MODEL_DIR:?Set MODEL_DIR to the local Qwen2.5-Coder-1.5B-Instruct directory}"
if [[ "${STAGE:-primary}" == primary ]]; then
  stages=(prepare base cascade components diagnostics coverage primary-export)
else
  stages=("$STAGE")
fi
for stage in "${stages[@]}"; do
  "$EXPERIMENT_PYTHON" scripts/run_publication_experiments.py --stage "$stage" "${output_flags[@]}" "${resume_flags[@]}"
done
