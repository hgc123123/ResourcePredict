# Data, targets and event-time protocol

## External inputs and identity

The production trace is private and contains executable user scripts. It is not shipped, downloaded, or synthesized from paper tables by this release. Production training requires the original CSV and a local Qwen2.5-Coder-1.5B-Instruct directory; hardware slicing additionally requires the site's capacity/type registries.

Paper-mode CSV SHA-256: `7967f923899dbad19ab79416bdfd999ca5174e1dd4086d0a97fa43296a8379cf`.

Effective split fingerprint: `85d00ce3a9fb6fe39e1be66ff91169daad85d7b217bc0b971c6dd251d845e3ce`.

Hashes identify inputs and assignments, not public availability or scientific correctness. The release has no bundled per-job labels, predictions, split memberships, production scripts, raw node registry or trained checkpoint. The preprocessing and schema-compatible synthetic generator are included.

## Export schema

`configs/paper.local.yaml` is the executable schema. It expects UTF-8-SIG CSV, unique `jobid`, timestamps interpreted in `Asia/Shanghai`, and rows retained for successful `COMPLETED` jobs with exit code `0:0`. Fields include:

| Role | Columns |
|---|---|
| Identity and replay | `jobid`, `user`, `submit`, `start1`, `end1`, `elapsed` |
| Submission-visible model inputs | `script_content`, `cpu`, `mem`, `gpu`, `time`; derived `mem/cpu` |
| Exported target ratios | `mem_max`, `vram_max` in [0,1], not percentage units |
| Hardware-analysis metadata | `nodelist`, requested `gpu` count |
| Other required export columns, excluded from learned metadata | `jobname`, `partition`, `state1`, `exitcode`, `cpu_eff`, `mem_eff`, `mem_p95`, `mem_capacity_max`, `gpu_util`, `vram_eff`, `vram_p95`, `vram_capacity_max`, `workdir` |

Node names, job IDs, timestamps, observed efficiency fields, absolute observed peaks, and execution metadata are not added as learned request features. User vocabulary and numeric scalers are fitted on optimization jobs only. Canonical script identity is retained for history even in No-Code. Site path prefixes in script redaction are preserved as part of original preprocessing, not as local deployment paths to open or execute.

The loader validates finite in-range labels and resource applicability under its configured validity rules. In the retained production splits, valid VRAM eligibility coincides with requesting GPUs; the coverage change is a CPU/GPU workload-composition change. The export does not establish the operational cause of that change.

## Sampled-peak target meaning

The final manuscript describes a **10-second sampling interval**. Experiment code consumes the exported ratio labels; it does not regenerate historical Prometheus samples or prove scrape completeness. `configs/telemetry_contract.json` records this distinction.

For memory, let `A_j` be total job-allocated memory from Slurm AllocTRES, `N_j` the allocated node count, and `P_j` the maximum observed node usage over valid samples and nodes. The target is `N_j * P_j / A_j`. Under equal allocated memory per node this is the largest observed per-node ratio. It is not a reconstructed simultaneous aggregate memory demand. Numerator and denominator must be in consistent units.

For VRAM, normalize each observed device usage by that device's registered physical capacity, then take the maximum ratio over allocated GPUs and observed sample times. GPUs within a job are homogeneous; capacities across jobs may differ. A GPU-count request is not a VRAM-capacity denominator. Registry values are per-device capacities and must use units consistent with observed usage.

Missing observations do not contribute to a sampled peak; a resource with no valid observations has an invalid label and is excluded from that resource's objective and metrics. Unobserved intervals can hide spikes. The retained export lacks sample-level coverage indicators and an independent label-finalization timestamp; this code cannot retrospectively certify detection of every delayed sample. No new whole-job timeout filter is applied to the historical CSV by the experiment code.

## Chronological roles and purging

| Role | Production count | Use |
|---|---:|---|
| Outer train | 15,458 | Internal chronological base optimization/selection |
| Base optimization | 13,880 | Gradient fitting, preprocessing vocabulary/scalers |
| Base early stopping | 1,546 | Checkpoint selection only |
| Internal crossing exclusions | 32 | Excluded from optimization at the inner completion boundary |
| Outer validation | 1,927 | Subsequent lightweight fitting and policy selection |
| Primary gate/fusion fitting | 1,302 | Leading approximately 70%, after boundary purge |
| Primary policy selection | 585 | Trailing approximately 30%, no base checkpoint selection |
| Validation crossing exclusions | 40 | Removed from the leading fitting block |
| Outer test | 1,936 | Report-only prediction evaluation, no fitting/selection |

The outer 80/10/10 split remains an assignment by submission time with completion-boundary purging; it is not contradicted by subdivisions inside train and validation. Fractions refer to grouping before purging, not exact fractions of final retained counts. Identical submission timestamps stay together. Preprocessing is never fitted on the early-stop or outer-validation blocks.

History replay exposes a label only after its recorded completion is strictly before the current submission. A boundary-purged job can later contribute completed history when eligible, while remaining excluded from the fitting/evaluation role it crossed. Test labels may update later eligible history after completion; they never train model parameters or select policies. Availability at recorded completion is an assumption because separate finalization times were not retained.

The residual-calibration study separately divides outer validation into approximately 70/15/15: 1,302 fitting, 239 selection and 291 calibration jobs after 40/55 completion exclusions. It freezes fitted models/policies before the calibration block. These results are not substituted for primary 70/30 results.

## Uncertainty and operational limits

Paired bootstrap conditions on frozen predictions and resamples submission days, nonoverlapping three-/seven-day blocks, users or canonical scripts separately. It recomputes job-weighted metrics with 10,000 resamples, seed 2026. These are separate clustered sensitivities, not moving-block or multiway bootstrap, independent datasets, pooled-seed inference, or multiplicity-adjusted confirmation.

Signed error is prediction minus target. Positive shortfall is `max(target - prediction, 0)` with zeros retained for other valid jobs. High utilization is target ≥.8. Additive margins are prespecified normalized-ratio diagnostics. Joint calibrated shortfall means that at least one applicable resource exceeds its bound. A unit bound is the normalized upper bound 1, not demonstrated useful physical capacity savings.

Recorded request features are legitimate submission-time predictors, but retrospective utilization prediction under those requests does not identify the effect of changing them. This release supports the paper's single-cluster retrospective study and local serial prediction-path measurements; it does not establish OOM probabilities, safe allocation, resource savings under a scheduler, untouched future-period confirmation, or cross-cluster generalization.
