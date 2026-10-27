#!/usr/bin/env bash
#SBATCH --job-name=hpc-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/analysis-%j.out
#SBATCH --error=logs/analysis-%j.err
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm_env.sh"
stage="${STAGE:-followup}"
case "$stage" in followup|hardware|diagnostics|components|primary-export|temporal-prepare|report|smoke) ;; *) echo 'Invalid CPU STAGE' >&2; exit 2 ;; esac
"$EXPERIMENT_PYTHON" scripts/run_publication_experiments.py --stage "$stage" "${output_flags[@]}" "${resume_flags[@]}"
