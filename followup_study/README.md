# Independent Calibration and Routing Attribution

This directory contains two supplementary CPU studies reported separately in the manuscript: routing-rule comparisons under equal budgets and independent residual calibration. Both reuse frozen Full / No-Code base predictions from the primary study's independent training-tail checkpoint protocol. They refit lightweight adaptive gates and select policies without retraining Qwen.

This is an **adaptive-routing comparison under a 70% / 15% / 15% outer-validation protocol**. It differs from the simplified primary method, which uses a 70% / 30% validation protocol and bypasses inference based directly on exact-history eligibility. Results from this routing or calibration study must not replace primary-table results without identifying the protocol difference.

## Dependencies and Inputs

Run the following commands from the release root, `20261026`. Python 3.12 is recommended. CPU dependencies are NumPy, pandas, scikit-learn, joblib, and PyYAML.

```bash
python -m pip install -r followup_study/requirements.txt
python followup_study/verify_source.py
```

The default input root is `outputs/study`. Complete the primary Full, No-Code, and Last-1 runs for all five seeds first. The inputs for each seed include:

```text
outputs/study/
  protocol/outer_split.csv
  train_tail/base/qwen/tokens_4096/{full,no_code}/seed_40...seed_44/
    config.resolved.yaml
    predictions.csv
    run_manifest.json
  train_tail/base/last-1/completed_before_submission/seed_40...seed_44/
    predictions.csv
    run_manifest.json
```

Preparation also requires the original private CSV to reconstruct history using the original identifiers, hashes, and completion times. A different CSV cannot be substituted. **Qwen weights and a GPU are not required.** Production per-job results are not bundled with the source release; the required predictions must come from the preceding primary runs.

## Running the Studies

Inspect the execution plan first. `--dry-run` neither checks input completeness nor produces experimental results:

```bash
python followup_study/run_followup.py --stage all --dry-run
```

Run the complete studies in an environment with the required private inputs:

```bash
export JOB_EFFICIENCY_CSV=/path/to/private/original.csv
python followup_study/run_followup.py \
  --stage all --study-root outputs/study --output-root outputs/followup \
  --data-csv "$JOB_EFFICIENCY_CSV" \
  --seeds 40 41 42 43 44 --variants full no_code --resume
```

Alternatively, submit a CPU Slurm job from the release root. Select a partition appropriate to the site, for example by adding `--partition=your_cpu_partition`. If cluster modules are needed, set `EXPERIMENT_MODULES` before submission; the script does not automatically load modules from the authors' site.

```bash
export JOB_EFFICIENCY_CSV=/path/to/private/original.csv
sbatch followup_study/scripts/run_followup.slurm
```

For separate submissions, finish preparation before running the gate and allocation stages. The latter two stages may run concurrently.

```bash
FOLLOWUP_STAGE=prepare sbatch followup_study/scripts/run_followup.slurm
# Run after prepare completes successfully:
FOLLOWUP_STAGE=gate sbatch followup_study/scripts/run_followup.slurm
FOLLOWUP_STAGE=allocation sbatch followup_study/scripts/run_followup.slurm
```

Override paths with `PROJECT_DIR`, `STUDY_ROOT`, and `FOLLOWUP_ROOT`. Submit the script from either the release root or `followup_study`; use an absolute path for the original data. Do not launch the same stage and seed twice concurrently. `--resume` accepts only completed stages whose requests, source, inputs, and output checksums match. Use a new output directory after changing code or parameters.

## Experimental Design

The outer training/validation/test split and frozen base predictions remain unchanged. Split outer validation chronologically into approximately 70% gate fitting, 15% policy selection, and 15% independent calibration. Jobs with identical submission times remain in the same block. Labels whose completion times cross the start of the next block are purged from the current block's fitting, selection, or calibration role. Test labels are used only for evaluation, not for fitting, policy selection, or calibration. History follows the strict condition `completion_time < submit_time`; completed earlier test jobs may supply history for later jobs.

Study A compares learned, recent_history, low_dispersion, many_observations, random_hash, bypass-all, and bypass-none rules using the same selected history experts and post-fusion models. Default budgets are 0, 25%, 50%, 75%, and 100% of exact-history-eligible validation jobs. Thresholds are frozen using only the policy-selection block, and realized test bypass rates are reported separately. The primary equal-budget comparison uses 50%. Encoder demand is computed by causal LRU-1024 replay; it is not measured latency. No-Code has no code encoder, so encoder-cost attribution is performed only for Full.

Study B compares normalized sampled-peak upper bounds for Full Cascade, No-Code Cascade, Hierarchical Same-Script Last-1, and standalone Qwen. Calibration uses finite-sample upper quantiles of residuals `y-p`, with global, history-group, and day-maximum residual rules. The nominal joint shortfall targets are 5% and 1%, with half the risk budget assigned to each resource. The implemented upper bound is `min(1, p + max(0,q))`. Insufficient calibration support returns a unit upper bound of 1. Test labels are not used to select more favorable margins.

This study measures shortfall and additional headroom for observed normalized peaks. These quantities are not OOM probabilities, physical resource savings, or future safety guarantees. Exchangeability cannot be assumed to hold under temporal drift. Zero observed shortfall from a unit bound is not evidence of zero future risk.

## Outputs and Verification

```text
outputs/followup/
  prepared/{full,no_code}/seed_40...seed_44/
    predictions.csv  selected_policy.json  protocol.json
    selection_sweep.csv  cascade_models.joblib
    request.json  completion.json
  gate/full/seed_40...seed_44/
  gate/gate_metrics_{per_seed,summary}.csv
  gate/gate_paired_intervals_per_seed.csv
  allocation/seed_40...seed_44/
  allocation/joint_bounds_metrics_{per_seed,summary}.csv
  allocation/resource_bounds_metrics_{per_seed,summary}.csv
```

Per-job predictions, `PRIVATE_test_bounds.csv`, identifiers, and script hashes are private runtime artifacts and are not distributed with the source. Summaries report the mean and sample standard deviation across five seeds; five seeds do not turn one test period into five independent datasets. Stage manifests record source, input, and output hashes and reject mismatched provenance.

Run synthetic tests without private data:

```bash
python -m unittest discover -s followup_study/tests -v
```

These tests cover role isolation, boundary purging, exclusion of test labels from calibration, frozen thresholds, cache start boundaries, missing labels, finite-sample quantiles, and synthetic end-to-end analyses. Passing the synthetic tests verifies workflow execution; it does not reproduce production accuracy.
