# Final-manuscript experiment map

The canonical runner is `scripts/run_publication_experiments.py`. Paths below are relative to its output root, default `outputs/`. All production runs start from external data and a local base model, rather than bundled historical checkpoints or predictions.

| Manuscript evidence | Stage / implementation | Protocol and output |
|---|---|---|
| Dataset, label eligibility, temporal boundaries; Tables 1–2 | `prepare`; `common/data.py`, `qwen/base_selection.py` | Strict original CSV identity, immutable outer split, optimization/early-stop roles; `study/protocol/` |
| Full/No-Code Qwen Base, Last-1, Fixed Hybrid | `base`; `qwen/`, `last-1/`, `hybrid/` | Five seeds; optimization-only preprocessing; train-tail checkpoint selection; `study/train_tail/base/` |
| Frozen experts, post-blending, comparative pre-classifiers | `cascade`; `reliability_cascade/` | Five seeds × Full/No-Code; approximately 70/30 outer validation; `study/train_tail/cascade/` |
| Primary Full exact-history bypass | `primary-export`; `export_primary_exact_bypass.py` | Frozen Full only; require zero thresholds and route/prediction equivalence; `primary_exact_bypass/` |
| Matched conventional baselines; Table 3 | `baselines`; `baseline_study/` | XGBoost, TextCNN, BiLSTM; five seeds; same optimization/early-stop jobs, own architecture/training criteria; `baselines/` |
| Paired day/user/script/longer-block uncertainty; Tables 4–5, Fig. 2 | `diagnostics`; `analysis/reviewer_diagnostics.py` | Paired Full–Hierarchical and Full–No-Code, seed-specific estimates; `study/train_tail/analysis/` |
| History subgroups; Table 5, Fig. 3 | `diagnostics`; `analysis/cascade_subgroups.py` | Most-recent same-user script matches, older exact-script history, user history without exact script, no eligible user history; target-independent membership |
| Token coverage; Fig. 4 | `coverage`; `analysis/token_coverage.py` | Tokenizer-only budgets 2048/4096/8192; only context 4096 is trained |
| Pipeline component comparisons; Table 6 | `components`; `run_reviewer_experiments.py` | Independently refit/select exact-bypass, no-bypass, pre-route-only, no-trust variants; `study/train_tail/components/` |
| Frozen component interventions, classifier diagnostics, threshold curves, bounded LRU replay; Fig. 5 | `diagnostics` | Freeze selected experts/models, intervene on operations; report separately from refitted pipelines |
| Temporal sensitivity; Fig. 6 | `temporal-prepare`, `temporal-0600/0800/1000` | Expanding prefixes .6/.8/1.0, disjoint test intervals, seed 42; independently retrain Full/No-Code and retain adaptive selected policies; `temporal/` |
| Hardware population/capacity/model strata; Fig. 7 | `hardware`; `hardware_study/` | Frozen primary-equivalent Full and independent No-Code; external per-node registries; `hardware/` |
| Signed error, under/overprediction, P95/P99 positive shortfall, high-utilization jobs; Table 7 | `diagnostics` | Evaluate valid labels; high utilization ≥.8; `risk_metrics_*.csv` |
| Fixed additive margins; Fig. 8 | `diagnostics` | Margins 0/.05/.10/.20, prespecified; `margin_diagnostics_*.csv` |
| Equal-budget learned/recency/dispersion/count/random routing comparisons | `followup`; `followup_study/gate_study.py` | Five seeds, lightweight adaptive refits on frozen base predictions; primary validation budget .5; `followup/gate/` |
| Independent residual calibration; Table 8 | `followup`; `followup_study/allocation_study.py` | Separate 70/15/15 validation, frozen before calibration; global/history-group/day-maximum residual margins at α=.05/.01; `followup/allocation/` |
| Measured serial online path; Fig. 9 | `online`; `online_study/` | Seven cases × five checkpoints × three technical repeats = 105 runs; actual conditional encoder calls; `online/` |
| Aggregate manuscript tables and data-backed figures | `report`; `publication_reporting/` | Tables 1–8, Fig. 2–9 in SVG/PDF/300-dpi PNG plus source CSV; `publication/` |
| Fictional end-to-end plumbing verification | `smoke`; `smoke_reviewer_pipeline.py` | Real loader, split/purge, histories and lightweight fits; deterministic stand-in base predictions, not trained Qwen |

## Primary path versus comparative fitting

The archived revised Full study selected zero pre-thresholds for both resources in all five seeds. The primary route therefore needs only submission-time applicability and eligible exact history; it does not evaluate the reliability classifier. The fitted expert and post-inference models remain frozen. The release reproduces comparative fitting first, then checks and exports this simplification. It does not silently force a newly selected nonzero policy to match the paper.

No-Code retains its independently selected adaptive routes. Pipeline ablations also refit and select their own policies. Frozen interventions disable individual operations on already-selected models. These answer different questions and are labelled separately in the exported evidence.

Temporal experiments and residual calibration retain adaptive-routing policies and their separate validation roles. They do not establish the behavior of the simplified primary rule under new training windows or calibrated allocation.

## Online measurements

The seven cases are `full_lru_0`, `full_lru_128`, `full_lru_1024`, `no_bypass_lru_1024`, `no_post_lru_1024`, `no_trust_lru_1024`, and `exact_bypass_lru_1024`. Here `full` means the classifier-equipped comparison; `exact_bypass` is the simplified primary path. Caches start at validation; capacity counts embeddings, not bytes. Use four CPU threads and ten warmup jobs. A balanced shuffled order controls technical repeat ordering; cases execute serially on one GPU.

Timing includes completion-label activation, script hashing, history features, route-specific gate operations, cache lookup/update, request/user preparation, tokenization, conditional encoder execution, prediction heads and post-fusion. Standardized/redacted job objects are already available. Raw Slurm parsing, process startup, initial history preload, RPC, queueing, failures and scheduler actions are outside scope. Batch-one numerical checks and offline differences are recorded rather than hidden by claiming bitwise equality.

Technical repeats are not new training seeds or independent datasets. Hardware and runtime changes can change latency and numerical outputs. Online results must be read from the measured execution files, not inferred from offline forward rates.

## Other utilities

`run_paper_experiments.py` provides plan/completion primitives used by the current runner. Generic historical model/fusion APIs remain to preserve tested dependencies and support comparisons. Their standalone defaults are not a second formal manuscript protocol. Old result acceptance constants and old six/eight-GPU batch-24/32 Slurm workflows are omitted from this release. The canonical entry point explicitly specifies the final three-process batch-12 setting.

The package contains no reported-result substitutions. The reporter flags unavailable inputs and verifies primary export provenance before using the primary-method interpretation. Missing experiments are not filled with manuscript constants.
