# Publication tables and figures

Run from the repository root:

```bash
python scripts/report_publication_results.py --dry-run
python scripts/report_publication_results.py --require-complete
```

Complete the relevant experiments before aggregating results. Input directories can be changed with the following command-line options:

| Option | Default | Experiment scope |
|---|---|---|
| `--study-root` | `outputs/study` | Five-seed primary study, components, and diagnostics under independent training-tail checkpoint selection |
| `--primary-root` | `outputs/primary_exact_bypass` | Five-seed exports with zero-threshold and prediction/route equivalence checks |
| `--baselines-root` | `outputs/baselines` | XGBoost, TextCNN, and BiLSTM using the same optimization/early-stopping jobs |
| `--followup-root` | `outputs/followup` | Budget-controlled routing comparisons and independent residual calibration |
| `--windows-root` | `outputs/temporal` | Independently retrained 60%, 80%, and 100% prefixes at seed 42 |
| `--hardware-root` | `outputs/hardware` | GPU capacity/type strata using frozen primary-study predictions |
| `--online-root` | `outputs/online` | Measured serial conditional inference and bounded LRU caching |
| `--output-dir` | `outputs/publication` | Output tables, figures, and evidence index |

`--help` and `--dry-run` use only the Python standard library and require no private data, model, or completed results. Aggregation requires pandas and NumPy; plotting additionally requires Matplotlib from `requirements-analysis.txt`. Use `--no-figures` to produce tables only. Missing inputs are listed in `report_manifest.json` and the output `README.md`; reported-result constants are not substituted for missing results. `--require-complete` returns a nonzero status when required inputs are missing.

The reporter reads numerical summaries. Each experiment's own entry point must first verify data identity, artifact integrity, and training-role assignments. Metrics use resource-valid masks and job weighting. Cross-seed standard deviations in figures are distinguished from confidence intervals.

## Table and figure map

| Report item | Exported content |
|---|---|
| Table 1 | Resource-label eligibility and chronological split composition |
| Table 2 | Model, training, and routing settings from the resolved `protocol/train_tail.yaml`, with path fields omitted |
| Table 3 | Primary, history, Fixed Hybrid, and matched conventional models, retaining means, sample standard deviations, and actual seed counts |
| Table 4 | Seed-specific paired day-block intervals for Full versus Hierarchical |
| Table 5 | Seed-specific history-subgroup MAE differences and paired day-block intervals for Full versus No-Code |
| Table 6 | Base/Cascade results and component pipelines with repeated fitting/selection; frozen interventions are exported separately |
| Table 7 | Signed error, underprediction tails, and high-utilization diagnostics |
| Table 8 | Joint shortfall and resource bounds from the independent calibration study |
| Fig. 2 | Daily VRAM MAE differences against two references, with five-seed means and sample standard deviations |
| Fig. 3 | MAE for history subgroups defined without current target values |
| Fig. 4 | Tokenizer-only coverage at budgets of 2048, 4096, and 8192 tokens |
| Fig. 5 | Frozen threshold sensitivity for the classifier comparison and offline bounded-LRU replay |
| Fig. 6 | Seed-42 temporal prefix retraining of the adaptive-routing extension |
| Fig. 7 | Frozen primary prediction MAE by registered GPU capacity |
| Fig. 8 | Prespecified additive margins and sampled-peak shortfall rates |
| Fig. 9 | Measured serial prediction-path latency, live batch-one MAE, and encoder forwards |

Figures are exported as SVG, PDF, and 300-dpi PNG, together with the aggregate CSV data used for plotting. They preserve the numerical inputs and statistical definitions, while their layout may differ from typeset versions. Missing results are identified explicitly and omitted from curves.

The primary Full path uses exact-history eligibility without evaluating a learned pre-inference classifier. Before reporting Full as the primary method, the reporter verifies five-seed export manifests, frozen source hashes, and prediction/route equivalence, then aggregates the primary exports. Without this evidence, the case is labelled as a classifier reference and treated as missing in strict mode. Failed verification stops aggregation. No-Code, temporal windows, and residual calibration retain their own adaptive-policy scopes and are reported separately.

Table 5 intervals are reported separately by seed and subgroup, without pooling seeds or adjusting for multiple subgroup comparisons. Fig. 6 uses only seed 42 and has no training-seed error bars. For Fig. 9, technical repeats are averaged within each seed before calculating cross-seed sample standard deviations. P95/P99 values are means of within-run quantiles, rather than quantiles of all pooled requests.

The reporter does not copy per-job predictions, original user scripts, node inventories, `PRIVATE_test_bounds`, or `PRIVATE_unknown_nodes`. Primary-path provenance checks compute prediction-file hashes. Output reports retain local input paths for traceability; review those paths under institutional disclosure rules before publishing the reports.
