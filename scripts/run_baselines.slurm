#!/usr/bin/env bash
#SBATCH --job-name=hpc-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/baselines-%j.out
#SBATCH --error=logs/baselines-%j.err
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm_env.sh"
: "${JOB_EFFICIENCY_CSV:?Set JOB_EFFICIENCY_CSV}"
"$EXPERIMENT_PYTHON" scripts/run_publication_experiments.py --stage baselines "${output_flags[@]}" "${resume_flags[@]}"
