#!/usr/bin/env bash
#SBATCH --job-name=hpc-online
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/online-%j.out
#SBATCH --error=logs/online-%j.err
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm_env.sh"
: "${JOB_EFFICIENCY_CSV:?Set JOB_EFFICIENCY_CSV}"
: "${MODEL_DIR:?Set MODEL_DIR}"
"$EXPERIMENT_PYTHON" scripts/run_publication_experiments.py --stage online "${output_flags[@]}" "${resume_flags[@]}"
