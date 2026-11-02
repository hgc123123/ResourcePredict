# ResourcePredict

Source code for **History-First Fusion for Sampled-Peak System and GPU Memory Utilization Prediction in HPC Clusters**.

ResourcePredict provides training, causal history replay, inference, and evaluation tools for sampled-peak system-memory and GPU-memory utilization prediction. It includes the primary exact-history bypass method, adaptive-routing comparisons, and statistical and runtime analyses. Production traces, executable user scripts, trained weights, and production result files are not included.

## Method and evaluation protocol

- **Primary method:** jobs with eligible exact-script history use the selected frozen history expert directly. Other jobs invoke Qwen and resource-specific post-inference fusion. The primary prediction path does not evaluate a learned pre-inference classifier.
- **Comparisons:** learned pre-inference routing, trust projection, and independently selected No-Code policies are evaluated separately. No-Code removes learned script-content encoding while retaining script identity for history. Full versus No-Code is a pipeline comparison, rather than a code-only intervention with fixed routing.
- **Primary experiments:** seeds 40-44; Full and No-Code use three processes, batch size 2 per process, and two gradient-accumulation steps, giving effective batch size 12. The outer chronological split is 80/10/10. Base training and checkpoint selection use separate blocks within the outer training period. The outer validation period is divided approximately 70/30 for lightweight model fitting and policy selection, followed by freezing.
- **Additional studies:** temporal retraining retains independently selected adaptive policies. Residual calibration uses a separate 70/15/15 validation split. These studies are reported separately from the primary method.

The classifier-equipped Full engine fits the history experts and post-inference models and supplies the classifier comparison. The `primary-export` stage exports the classifier-free rule from frozen Full artifacts after checking zero thresholds and exact prediction/route equivalence. Export under the configured study protocol requires these checks to pass. No-Code retains its independently selected policies. The live primary path is implemented by the `exact_bypass` operation in `online_study`.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the experiment map and [docs/DATA_AND_PROTOCOL.md](docs/DATA_AND_PROTOCOL.md) for the schema, targets, and chronological roles.

## Installation and checks without private data

Linux and Python 3.12 are recommended. Install a CUDA-compatible PyTorch build for GPU training before installing the project dependencies. `requirements-paper.txt` records the principal training versions; numerical results may differ across software and hardware environments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-paper.txt -r requirements-analysis.txt
python scripts/build_source_manifest.py --syntax-check
python scripts/run_publication_experiments.py --stage all --dry-run
python -m pytest tests
python -m unittest discover -s followup_study/tests -v
python -m unittest discover -s hardware_study -p 'test_capacity_analysis.py' -v
python scripts/run_publication_experiments.py --stage smoke
```

The `smoke` stage uses fictional data with the actual loader, chronological splitting, causal history replay, and lightweight fusion fitting. Base predictions are explicitly identified deterministic stand-ins. This check does not train Qwen or reproduce production accuracy. It requires NumPy, pandas, PyYAML, scikit-learn, and joblib, but no GPU. Use a new output directory for each repeat, for example `--output-root outputs/smoke_02`.

## Production inputs

Run commands from the repository root. Set the following paths to authorized external inputs:

```bash
export JOB_EFFICIENCY_CSV=/absolute/path/to/private/original.csv
export MODEL_DIR=/absolute/path/to/Qwen2.5-Coder-1.5B-Instruct
export GPU_CAPACITY_MAP=/absolute/path/to/private/capacity.json
export GPU_MODEL_MAP=/absolute/path/to/private/models.json
mkdir -p logs
```

Base training requires the first two inputs. Hardware stratification also requires the capacity and model registries. Their schemas are documented in [hardware_study/README.md](hardware_study/README.md). The example registries contain fictional nodes and are intended only for schema and workflow checks. Models are loaded locally and are not downloaded automatically.

`configs/paper.local.yaml` verifies the production trace's CSV SHA-256, row counts, and split fingerprints. Reproducing production results requires authorized access to the original inputs. Experiments on another trace require a separate configuration and newly established data and split identities; they constitute a separate study.

## Running the experiments

The entry point is `scripts/run_publication_experiments.py`. The following sequence covers the reported experiment families. `--resume` reuses only verified results with matching inputs. Use a new output root after changing source code or configuration.

```bash
# Data preparation, base training, cascade comparisons, and primary export
python scripts/run_publication_experiments.py --stage prepare --resume
python scripts/run_publication_experiments.py --stage base --resume
python scripts/run_publication_experiments.py --stage cascade --resume
python scripts/run_publication_experiments.py --stage components --resume
python scripts/run_publication_experiments.py --stage diagnostics --resume
python scripts/run_publication_experiments.py --stage coverage --resume
python scripts/run_publication_experiments.py --stage primary-export --resume

# XGBoost, TextCNN, and BiLSTM on the same optimization/early-stopping jobs
python scripts/run_publication_experiments.py --stage baselines --resume

# Equal-budget routing and independent calibration; no Qwen retraining
python scripts/run_publication_experiments.py --stage followup --resume

# Independently trained temporal windows, each using seed 42
python scripts/run_publication_experiments.py --stage temporal-prepare --resume
python scripts/run_publication_experiments.py --stage temporal-0600 --resume
python scripts/run_publication_experiments.py --stage temporal-0800 --resume
python scripts/run_publication_experiments.py --stage temporal-1000 --resume

# Hardware strata, measured serial inference, tables, and figures
python scripts/run_publication_experiments.py --stage hardware
python scripts/run_publication_experiments.py --stage online --resume
python scripts/run_publication_experiments.py --stage report --resume
```

With all dependencies and external inputs available, `--stage all --resume` executes the production stages serially. The default output root is `outputs/`. Use the same `--output-root` throughout a study. Completed hardware analysis is reused only after its inputs, source, configuration, and outputs pass verification; incomplete or changed results require a new output directory.

The `report` stage produces aggregate data for Tables 1-8 and Figs. 2-9, with CSV, SVG, PDF, and 300-dpi PNG output. It exports numerical tables and experimental plots; the architecture illustration is outside its scope. Figures preserve the input values and statistical definitions, while their layout may differ from typeset versions. Missing experiments are listed and cause a nonzero exit status. Primary-method reporting also requires successful export provenance checks. Per-job predictions, derived data, and checkpoints remain within the output root and are not included in the source distribution.

## Slurm submission

The scripts leave partition, account, node selection, and personal paths to the deployment environment. Add `--partition` and `--account` to `sbatch` as required by the site. Resource requests and time limits can be adjusted. Install dependencies in the selected Python environment before submission. Set `EXPERIMENT_MODULES` for required cluster modules and `EXPERIMENT_PYTHON` for a specific interpreter.

```bash
# Primary study: three GPUs; prepare through primary-export
sbatch scripts/run_primary.slurm

# After the primary study, submit these independently as resources permit
sbatch scripts/run_baselines.slurm                         # One GPU
sbatch --export=ALL,STAGE=followup scripts/run_cpu.slurm    # CPU
sbatch scripts/run_online.slurm                           # One GPU; 105 serial runs
sbatch --export=ALL,STAGE=hardware scripts/run_cpu.slurm    # CPU

# Prepare temporal windows, then submit three separate three-GPU jobs
python scripts/run_publication_experiments.py --stage temporal-prepare --resume
sbatch --export=ALL,STAGE=temporal-0600 scripts/run_temporal.slurm
sbatch --export=ALL,STAGE=temporal-0800 scripts/run_temporal.slurm
sbatch --export=ALL,STAGE=temporal-1000 scripts/run_temporal.slurm

# Aggregate results after the required experiments finish
sbatch --export=ALL,STAGE=report scripts/run_cpu.slurm
```

Stages have different resource requirements, so a single-GPU allocation is insufficient for `--stage all`. Temporal windows can run concurrently when each has its own allocation. Online cases run serially on one GPU in a balanced order; concurrent execution would change the interpretation of the latency measurements.

## Source scope and reproducibility

Core modules include `common/`, `qwen/`, `last-1/`, `hybrid/`, `reliability_cascade/`, `baseline_study/`, `analysis/`, `followup_study/`, `hardware_study/`, and `online_study/`. General experiment utilities are retained for API compatibility. The documented entry point defines the supported study protocol. Production result constants and obsolete workflows are excluded.

[SOURCE_MAP.md](SOURCE_MAP.md) describes the source origins and adaptations. `SOURCE_PROVENANCE.json` records per-file source hashes, and `SOURCE_SHA256SUMS.txt` verifies release integrity. Checksums establish file identity, not experimental correctness. See [docs/VALIDATION.md](docs/VALIDATION.md) for the validation scope.

The study is retrospective and uses one cluster and a previously inspected test period. Sampled peaks need not capture unobserved execution maxima. Margin and calibration diagnostics do not establish safe allocation, OOM probabilities, cross-cluster generalization, or scheduler-level resource savings. Restricted inputs and unavailable trained weights limit independent reproduction of production results.

Project source is distributed under the [MIT License](LICENSE). Third-party models, dependencies, and external data retain their own licenses. The project license does not grant access or redistribution rights to private traces or third-party weights.
