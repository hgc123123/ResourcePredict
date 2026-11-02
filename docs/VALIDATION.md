# Validation scope

Source validation covers software integrity, chronological data roles, synthetic workflow execution, and aggregation of existing experimental outputs. These checks are distinct from production model training and independent reproduction with the restricted trace.

## Recorded validation

- The CPU-capable main test subset passed, with 376 tests passing and four dependency-dependent tests skipped. Six test modules requiring PyTorch and the separate environment/inventory module were excluded from that subset; environment/inventory checks and release-specific checks were run separately after finalizing the inventory.
- Follow-up: 15 tests passed. They exercise chronological role separation, target-independent scores and cutoffs, missing-label behavior, finite-sample calibration, immutable resume checks, cache boundaries, and synthetic prepared-prediction gate/allocation analyses. These tests do not train Qwen or independently validate the production preparation stage.
- Hardware: 32 tests passed; one optional private-result test was skipped. Tests include exact string identifiers, sealed synthetic five-seed prediction inputs, node-list parsing, per-GPU capacities, unresolved hardware handling, and paired analysis through the real sealed-file reader.
- The synthetic CPU pipeline completed with real loader, chronological purging, optimization-only preprocessing, causal history replay and actual lightweight cascade fit/select/apply. It used 320 fictional rows, 316 effective jobs and 632 job-resource checks; four completed boundary-purged history uses were verified. Base estimates were deterministic stand-ins, not trained Qwen outputs.
- Read-only primary-export checks against five sealed reference Full runs confirmed zero thresholds for both resources and exact validation/test prediction and bypass-decision equivalence: zero differing rows across 7,726 job-resource rows per seed. No-Code artifacts were unchanged.
- Read-only aggregation against existing results, with primary-export verification and figures disabled, found all required numerical inputs and generated 27 aggregate CSV tables and an evidence index. Table 5 contained 40 seed/resource/subgroup comparisons. These production results and validation outputs are not included in the source distribution.
- The canonical all-stage dry-run, standalone entry help/plan paths, Python syntax, source checksums and source-only archive contents were checked. Slurm scripts use LF line endings and no site-specific account, partition or user paths.

## Validation environment

The recorded local checks used Windows Python 3.12.14, NumPy 2.3.5 and pandas 3.0.1. Validation-only packages included PyYAML 6.0.3, scikit-learn 1.9.0, SciPy 1.18.0 and pytest 9.1.1. These versions differ from the Linux training environment specified by `requirements-paper.txt`. Use the documented CUDA/PyTorch setup for production runs.

Windows file permissions required an external validation wrapper to create accessible test-fixture directories. The wrapper did not change model, data, metric, routing, or selection code. Test fixtures cover dtype validation, external XGBoost detection, binary newline byte counts, and nested checksum inventories.

PyTorch, CUDA, Transformers/PEFT and Matplotlib were unavailable locally. GPU training, live online inference, context-tokenizer coverage, full neural tests, and actual vector-figure rendering were not rerun. The release includes their source and dependencies; local syntax and plan checks are not reported as execution of those experiments. Slurm submission itself was not performed, and local Git Bash could not start under this Windows environment, so Slurm shell behavior was reviewed rather than executed here.

## Reproducibility limits

Source checksums establish file identity. They do not independently validate historical telemetry acquisition, sample completeness, hardware mappings, or pretrained-weight provenance. The data protocol documents these limits and the distinction between sampled peaks and unobserved execution maxima. Production reproduction requires authorized access to the original trace, model files, and applicable hardware registries. Synthetic workflow success is not production-accuracy evidence.

## Portability checks

An additional validation snapshot parsed all 159 Python files, verified the root and routing/calibration checksum inventories, and passed 29 targeted tests covering variant names, source/environment inventories, workflow planning, and primary exact-bypass behavior. The main experiment planner, routing/calibration planner, and aggregate reporter completed dry runs. These recorded counts describe the corresponding validation snapshots; they do not imply that production training or GPU inference was rerun after every documentation change.

Use the commands in the project [README](../README.md) to validate a local installation. A successful dry run checks the execution plan only; complete experimental verification requires the relevant inputs, dependencies, and runtime resources.

Targeted CPU checks also passed for the public prediction runner, temporal-window preparation, paired prediction diagnostics, synthetic pipeline, workflow planning, and primary exact-bypass export. These checks validate module references and the covered behaviors without requiring private production inputs or GPU execution.
