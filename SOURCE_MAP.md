# Source provenance and changes

This is a standalone source release assembled from the existing experiment implementations. Historical source directories and their experimental outputs were not modified.

| Release location | Existing source | Scope |
|---|---|---|
| `common`, `qwen`, `last-1`, `hybrid`, `reliability_cascade`, `analysis`, `baseline_study`, `online_study` | `20261021/reviewer_experiments/` | Dataset/replay, model training, revised checkpoint selection, comparative fitting, diagnostics, matched baselines and measured live inference |
| Other baseline/fusion modules, configs, tests and Python utilities | `20261021/reviewer_experiments/` | Dependencies and experiment utilities retained with tested APIs |
| `followup_study/` | `20261021/reviewer_followup_20261023/` | Equal-budget routing and independent calibration; paths adapted to integrated release |
| `hardware_study/` | `20261021/reviewer_hardware_20261023/` | Capacity/type strata; external registries replace local node inventories; synthetic schema examples only |
| `scripts/run_publication_experiments.py`, release Slurm scripts | Added for this release | Final-protocol orchestration, explicit seeds/batches, independent stages |
| `scripts/export_primary_exact_bypass.py` | Added for this release | Frozen classifier-free primary export with zero-threshold and prediction/route equivalence assertions |
| `publication_reporting/`, `scripts/report_publication_results.py` | Added for this release | Parameterized aggregate tables, data-backed figures, missing-input and provenance checks |
| Release documentation, license, provenance and checksum tools | Added/adapted for this release | Portable instructions and auditable source-only packaging |

The primary config uses independent train-tail checkpoint selection. Its original preprocessing/architecture/hyperparameter settings are retained; the official entry point explicitly uses 3 × 2 × 2 training batches for Full and No-Code. Conventional baselines retain their own model-specific settings and share the new optimization/early-stopping sample roles.

The primary export applies the existing `online_study` exact-bypass rule to frozen offline artifacts. It does not refit, use targets for routing, alter No-Code, or copy a different route's cache/forward counters. Paper-mode export fails if the new Full run does not reproduce the manuscript's zero-threshold simplification.

Historical result constants, old acceptance-table scripts, their obsolete tests, old dataset example configs, and six/eight-GPU batch-24/32 Slurm workflows were excluded. The lower-level matrix runner's process default is also aligned to three. Test fixtures were adapted for pandas-3 dtype behavior, correct detection of the external XGBoost distribution, platform-independent byte counts, and nested checksum inventories; model algorithms were not changed by those fixture fixes. No private trace, original executable scripts, trained weights, real node registries or historical results were copied. The target contract records the final manuscript's 10-second description without asserting recovery of historical sample-level telemetry or a complete original aggregation lineage.

`SOURCE_PROVENANCE.json` records per-file original and release hashes, distinguishing unchanged copies, adapted files and new files. Source roots are represented by relative provenance names; execution does not access those directories. `SOURCE_SHA256SUMS.txt` records the final release content. Generated test caches, run artifacts and archives are excluded from both source packaging and public input materials.

Dependencies and Qwen model weights remain subject to their upstream licenses; the repository's MIT license covers project source, not the private trace or external third-party weights. The recorded source audit does not retroactively supply missing pretrained-weight provenance from historical primary runs. New runtime/model inventories can be captured with `scripts/capture_environment.py`.
