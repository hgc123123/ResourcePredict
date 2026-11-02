# GPU Capacity and Model Stratification

This directory reuses frozen Full and No-Code predictions from the primary protocol to analyze GPU capacity/model strata, hardware composition, and paired day-block uncertainty. **It does not retrain models, reselect policies, change labels, or require a GPU.** The primary workflow must first verify that the frozen Full policy has zero thresholds and is equivalent to the simplified exact-history bypass rule. No-Code retains its independently selected adaptive routing, so comparisons remain at the pipeline level.

## Inputs and Privacy

The default input is `outputs/study` under the release root. Required inputs are the data identity, population index, and outer split in `protocol/`, together with sealed Full / No-Code Cascade predictions for five seeds. Only `jobid`, `nodelist`, and `gpu` are read from the original CSV; the analysis verifies the full-file SHA-256, original row positions, string identifiers, and split identity.

Production node lists, hardware registries, and experimental results are not distributed with the source. Supply two external JSON files:

- Capacity registry: `{"node_name": 24576}`. Values are positive integer capacities per GPU; do not multiply them by the job's requested GPU count.
- Model registry: `{"node_name": {"gres_type": "model_name", "installed_gpu_count": 8}}`. Set the model type to `null` when it cannot be confirmed. The installed GPU count is not the job's requested GPU count.

`examples/*.synthetic.json` contains fictional schema examples, not production nodes. Reproducing the production hardware strata requires the corresponding private registries. The optional `--registry-provenance` argument accepts an external provenance JSON file recording capacity units, registry sources, and applicable dates. The analysis hashes the supplied registries and provenance file separately; the release does not include production hardware registries.

A current hardware inventory does not establish that configurations were unchanged during the historical study period. Use `--historical-mapping-confirmed` only when supporting evidence exists. This flag records the caller's confirmation; it does not replace historical hardware logs. Specify capacity units with `--capacity-unit`; values are not automatically converted or rounded.

## Running the Analysis

Run the following commands from the release root, `20261026`. Python 3.12 is recommended:

```bash
python -m pip install -r hardware_study/requirements.txt
python hardware_study/run_capacity_analysis.py --help
export JOB_EFFICIENCY_CSV=/path/to/private/original.csv
export GPU_CAPACITY_MAP=/path/to/private/capacities.json
export GPU_MODEL_MAP=/path/to/private/models.json
python hardware_study/run_capacity_analysis.py \
  --study-root outputs/study --data-csv "$JOB_EFFICIENCY_CSV" \
  --capacity-map "$GPU_CAPACITY_MAP" --model-map "$GPU_MODEL_MAP" \
  --output-dir outputs/hardware --seeds 40 41 42 43 44 \
  --bootstrap-replicates 10000 --bootstrap-seed 2026
```

When provenance records are available, add `--registry-provenance /path/to/private/provenance.json`. Use `--capacity-unit MiB` only if the registered units are MiB. The output directory must be absent or empty. Use a new directory for a rerun; nonempty results cannot be overwritten. If Matplotlib is unavailable, add `--no-plots` to produce statistics and manifests without figures.

Submit a CPU Slurm job as follows. Select a site partition with `--partition=your_cpu_partition` if needed, and set cluster modules through `EXPERIMENT_MODULES`.

```bash
sbatch hardware_study/run_capacity_analysis.slurm
```

Submit the script from either the release root or `hardware_study`. By default, `PROJECT_DIR` is the release root, `STUDY_ROOT` is its `outputs/study` directory, and `HARDWARE_OUTPUT` is its `outputs/hardware` directory. Use absolute paths for the external CSV and registries. Pass the corresponding options through `REGISTRY_PROVENANCE`, `CAPACITY_UNIT`, and `HISTORICAL_MAPPING_CONFIRMED=1`.

## Strata and Statistics

Capacity statistics use exact registered values and five prespecified bands: <=16384, (16384,32768], (32768,65536], (65536,86016], and >86016. GPU models are grouped using case-normalized GRES labels, without inferring models from capacities or node names.

Unknown nodes, unparseable node lists, unspecified models, and inconsistent mappings are reported separately while retaining the full valid-job population. These samples are not silently dropped. The site's homogeneous-allocation assumption and the quality of registry mappings are separate issues. Registry checks identify input anomalies; differences in a current registry do not establish that historical jobs actually used mixed hardware.

Within each hardware group, paired bootstrap resamples whole nonempty UTC submission-day blocks 10,000 times with pseudorandom seed 2026. Differences are Full minus reference: negative MAE/MSE/RMSE differences favor Full, while positive Acc@10% differences favor Full. Groups with fewer than 30 jobs or fewer than 5 day blocks are flagged as small; intervals are not calculated for a single day block. Intervals are pointwise descriptive results without multiple-comparison adjustment.

Stratum means and error bars show the mean and sample standard deviation across five training seeds, not confidence intervals. `n` counts jobs from the same test period and is not multiplied by the number of seeds. All targets are observed sampled-peak utilization ratios. They are neither renormalized nor converted into OOM risks or physical resource benefits.

## Outputs and Verification

Primary outputs include `capacity_population_coverage.csv`, `hardware_monthly_composition.csv`, `capacity_metrics_per_seed.csv`, `capacity_metrics_summary.csv`, `capacity_paired_intervals_per_seed.csv`, `capacity_paired_support_summary.csv`, mapping-anomaly tables, and `analysis_manifest.json`. SVG, PDF, and PNG figures are also produced when Matplotlib is installed. Runtime artifacts such as `PRIVATE_unknown_nodes.csv` contain site identifiers and must not be published with the source.

Run tests without the original data, production registries, or training results:

```bash
python -m unittest discover -s hardware_study -p 'test_capacity_analysis.py' -v
```

Tests use fictional nodes, labels, and sealed five-seed results. They cover node-list expansion, capacity/model schemas, unknown and anomalous mappings, string identifiers, integrity verification, and the complete five-seed analysis. Passing these tests verifies the workflow, not production accuracy. The optional check against real sealed results requires explicitly setting `HARDWARE_REAL_SMOKE=1` and `STUDY_ROOT`; no production results are read by default.
