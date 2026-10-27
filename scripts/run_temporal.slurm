#!/usr/bin/env bash
#SBATCH --job-name=hpc-temporal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/temporal-%j.out
#SBATCH --error=logs/temporal-%j.err
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm_env.sh"
: "${JOB_EFFICIENCY_CSV:?Set JOB_EFFICIENCY_CSV}"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${STAGE:?Set STAGE to temporal-0600, temporal-0800 or temporal-1000}"
case "$STAGE" in temporal-0600|temporal-0800|temporal-1000) ;; *) echo 'Invalid temporal STAGE' >&2; exit 2 ;; esac
"$EXPERIMENT_PYTHON" scripts/run_publication_experiments.py --stage "$STAGE" "${output_flags[@]}" "${resume_flags[@]}"
