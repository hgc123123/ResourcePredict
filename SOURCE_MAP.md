# Source organization and provenance

ResourcePredict integrates training, chronological replay, prediction diagnostics, and serving benchmarks in a standalone source distribution. Runtime execution uses this repository and authorized external inputs; it does not depend on the original development directories.

| Location | Source collection | Function |
|---|---|---|
| `common/`, `qwen/`, `last-1/`, `hybrid/`, `reliability_cascade/`, `analysis/`, `baseline_study/`, `online_study/` | `prediction-experiments` | Data processing, history replay, model training, independent checkpoint selection, cascade fitting, evaluation, and serial inference |
| Other baseline modules, configuration, tests, and execution utilities | `prediction-experiments` | Shared model interfaces and experiment support |
| `followup_study/` | `routing-calibration` | Equal-budget routing comparisons and independent residual calibration |
| `hardware_study/` | `hardware-analysis` | Capacity and device-model stratification using external registries |
| `scripts/run_publication_experiments.py`, Slurm entry points | Project integration | Experiment orchestration, explicit seed and batch settings, and independent execution stages |
| `scripts/export_primary_exact_bypass.py` | Project integration | Frozen classifier-free export with threshold, prediction, and routing equivalence checks |
| `publication_reporting/`, `scripts/report_publication_results.py` | Project integration | Aggregate tables, figures, input-completeness checks, and export verification |
| Documentation and source-inventory tools | Project integration | Installation, protocol reference, source provenance, and distribution integrity |

Collection identifiers are logical provenance labels, not filesystem paths. `SOURCE_PROVENANCE.json` records each file's current SHA-256 digest and, where available, its original digest and source collection. It distinguishes unchanged copies, adapted files, and files introduced during integration. Original digests are retained when modules are renamed. `scripts/build_release_provenance.py --baseline-provenance SOURCE_PROVENANCE.json` refreshes current digests without discarding those recorded origins.

## Experiment settings

The primary configuration reserves a chronological training tail for checkpoint selection. Full and No-Code use three processes, batch size two per process, and two gradient-accumulation steps, for an effective batch size of twelve. Conventional baselines use the same optimization and early-stopping samples with their own model-specific settings.

The primary export applies the `online_study` exact-history bypass rule to frozen offline artifacts. Export verification requires zero reference thresholds and equivalent predictions and routing decisions. Cache-hit and encoder-forward counters are measured separately because they depend on the executed routing and caching operations.

Run inventories bind outputs to their source code, inputs, and configuration. Use a new output root after modifying code or configuration; existing studies must retain their original inventories.

## Distribution boundaries

`SOURCE_SHA256SUMS.txt` verifies the distributed source files. Source archives exclude version-control metadata, generated caches, runtime outputs, private traces, executable user scripts, trained weights, and production hardware registries. Synthetic examples demonstrate input schemas and workflow behavior without representing production records.

The telemetry contract describes sampled-peak targets and the limits of the retained job-level export; it does not reconstruct raw monitoring streams. Source checksums establish file identity, not experimental correctness or historical pretrained-weight provenance. `scripts/capture_environment.py` can record the software and model inventories for a new run.

The [MIT License](LICENSE) covers project source. External data, models, and dependencies retain their own access conditions and licenses.
