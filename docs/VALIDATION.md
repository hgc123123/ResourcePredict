# Release validation

This records preparation of the source release on 2026-10-26 and its English documentation revision on 2026-10-27. Neither operation is a new production training study or independent reproduction without the private trace.

## Checks performed

- The CPU-capable main test subset passed, with 376 tests passing and four dependency-dependent tests skipped. Six test modules requiring PyTorch and the separate environment/inventory module were excluded from that subset; environment/inventory checks and release-specific checks were run separately after finalizing the inventory.
- Follow-up: 15 tests passed. They exercise chronological role separation, target-independent scores and cutoffs, missing-label behavior, finite-sample calibration, immutable resume checks, cache boundaries, and synthetic prepared-prediction gate/allocation analyses. These tests do not train Qwen or independently validate the production preparation stage.
- Hardware: 32 tests passed; one optional private-result test was skipped. Tests include exact string identifiers, sealed synthetic five-seed prediction inputs, node-list parsing, per-GPU capacities, unresolved hardware handling, and paired analysis through the real sealed-file reader.
- The synthetic CPU pipeline completed with real loader, chronological purging, optimization-only preprocessing, causal history replay and actual lightweight cascade fit/select/apply. It used 320 fictional rows, 316 effective jobs and 632 job-resource checks; four completed boundary-purged history uses were verified. Base estimates were deterministic stand-ins, not trained Qwen outputs.
- The new primary exporter was checked read-only against the existing five sealed revised Full runs. All five had zero thresholds for both resources. Every exported validation/test prediction and bypass decision matched the reference exactly (zero differing rows; 7,726 job-resource rows per seed). No-Code was not modified.
- The aggregate reporter was checked read-only against existing results with primary-export provenance verification and figures disabled. All required numerical inputs were found; it generated 27 aggregate CSV tables and an evidence index in a QA directory outside the source release. Table 5 contains 40 seed/resource/subgroup comparisons. No historical results or QA outputs are supplied in this source release.
- The canonical all-stage dry-run, standalone entry help/plan paths, Python syntax, source checksums and source-only archive contents were checked. Slurm scripts use LF line endings and no site-specific account, partition or user paths.

## Environment and limits

Local checks used Windows Python 3.12.14, NumPy 2.3.5 and pandas 3.0.1. Cached validation-only packages included PyYAML 6.0.3, scikit-learn 1.9.0, SciPy 1.18.0 and pytest 9.1.1. These are not the paper's pinned Linux training environment. The production dependency specifications remain unchanged except for release orchestration; use `requirements-paper.txt` with the documented CUDA/PyTorch setup for new production runs.

Local Windows file permissions prevented access to some pytest fixture directories created with mode 0700. A validation wrapper outside this release used Windows-accessible permissions for fixture directory creation. Model, data, metric, routing, and selection code were unchanged. Source test fixtures explicitly cover string corruption dtype, the external XGBoost distribution, binary newline byte counts, and the nested follow-up inventory, without weakening their assertions.

PyTorch, CUDA, Transformers/PEFT and Matplotlib were unavailable locally. GPU training, live online inference, context-tokenizer coverage, full neural tests, and actual vector-figure rendering were not rerun. The release includes their source and dependencies; local syntax and plan checks are not reported as execution of those experiments. Slurm submission itself was not performed, and local Git Bash could not start under this Windows environment, so Slurm shell behavior was reviewed rather than executed here.

No source packaging check can establish correctness of the historical telemetry acquisition, missing-sample coverage, historical hardware mappings, or pretrained-weight provenance. Those limits remain stated in the manuscript and data protocol. Production reproduction still requires authorized external inputs; synthetic workflow success is not accuracy evidence.

## English documentation revision

On 2026-10-27, the project and experiment guides were standardized in English. A content audit covered 193 source text files, including all eight Markdown documents. Shell commands in the documentation were compared with the preceding source archive, and relative documentation links were checked. Prediction algorithms, experimental settings, and data-role assignments were unchanged. The non-ASCII path-rejection fixture now uses an accented Latin word while retaining its original validation purpose.

All 159 Python files passed syntax parsing. The root and follow-up checksum inventories were regenerated and verified. Twenty-nine targeted tests passed, covering portable variant names, environment/source inventories, publication workflow planning, and primary exact-bypass behavior. The main experiment planner, follow-up planner, and aggregate reporter also completed dry runs. These checks did not rerun production training or GPU inference.
