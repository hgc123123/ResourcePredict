#!/usr/bin/env python3
"""Run the conventional baselines with the primary train-tail selection roles.

Only new baselines are fitted. Primary Qwen/cascade outputs remain read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_study.protocol import (MODELS, SCHEMA, digest, load_primary,
                                     locked_json, project_path, read_json,
                                     seal_run, verify_seal, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-study", default="outputs/reviewer_20261021")
    parser.add_argument("--output-root", default="outputs/baselines_train_tail_20261023")
    parser.add_argument("--config", help="Default: reference-study/protocol/train_tail.yaml")
    parser.add_argument("--data-csv", default=os.environ.get("JOB_EFFICIENCY_CSV"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--stage", choices=["prepare", "train", "aggregate", "all"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip verified completed runs; archive and restart incomplete runs")
    parser.add_argument("--dry-run", action="store_true", help="Print plan; no data imports or writes")
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        parser.error("models must be unique")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be unique nonnegative integers")
    args.reference_study = project_path(args.reference_study)
    args.output_root = project_path(args.output_root)
    args.config = project_path(args.config) if args.config else None
    if (args.output_root == ROOT or args.output_root in ROOT.parents
            or args.output_root == args.reference_study
            or args.reference_study in args.output_root.parents
            or args.output_root in args.reference_study.parents):
        parser.error("output-root must be separate from the project root and reference study")
    if args.data_csv:
        args.data_csv = project_path(args.data_csv)
    if not args.dry_run and args.stage != "aggregate" and not args.data_csv:
        parser.error("Set JOB_EFFICIENCY_CSV or --data-csv")
    return args


def plan(args):
    print(f"Reference (read-only): {args.reference_study}")
    print(f"New outputs: {args.output_root}")
    if args.stage != "aggregate":
        print("[PLAN] Verify raw CSV, frozen outer split and exact primary optimization/early-stop roles")
    if args.stage in {"train", "all"}:
        for model in args.models:
            for seed in args.seeds:
                print(f"[PLAN] train {model} seed={seed}; outputs={args.output_root/model/f'seed_{seed}'}")
    if args.stage in {"aggregate", "all"}:
        print("[PLAN] Verify every requested run; recompute metrics; summarize mean and sample SD")


def prepare(args):
    config, bundle, selection, identity = load_primary(
        args.reference_study, args.data_csv, args.config)
    locked_json(args.output_root / "protocol/study.json", identity)
    locked_json(args.output_root / "protocol/base_training_selection.json", selection.manifest)
    # The reference files are never changed or copied over.
    locked_json(args.output_root / "protocol/fitting_counts.json", {
        "outer_counts": bundle.frame.groupby("split").size().to_dict(),
        "optimization_count": len(selection.optimization),
        "early_stopping_count": len(selection.early_stopping),
        "purged_optimization_count": selection.manifest["purged_optimization_count"],
        "optimization_valid": selection.manifest["optimization_valid_by_resource"],
        "early_stopping_valid": selection.manifest["early_stopping_valid_by_resource"],
    })
    print(f"[AUDIT] optimization={len(selection.optimization)}, "
          f"early_stopping={len(selection.early_stopping)}, "
          f"inner_purge={selection.manifest['purged_optimization_count']}; "
          f"outer validation={len(bundle.split('validation'))}, test={len(bundle.split('test'))}", flush=True)
    return config, bundle, selection, identity


def run_one(args, model, seed, config, bundle, selection, identity):
    from common.config import serializable_config, set_value
    output = args.output_root / model / f"seed_{seed}"
    config = deepcopy(config)
    set_value(config, "seed", seed)
    set_value(config, "outputs.root", str(args.output_root))
    set_value(config, "baseline_study.model", model)
    if args.device is not None:
        set_value(config, "training.device", args.device)
    contract = {"study": identity, "model": model, "seed": seed,
                "config": serializable_config(config)}
    # Complete results are immutable. A changed contract or damaged seal fails
    # rather than silently overwriting a usable result.
    if output.exists() and any(output.iterdir()):
        request = output / "request.json"
        if not request.is_file() or read_json(request) != contract:
            raise FileExistsError(f"Different/unmanaged baseline output: {output}")
        if not args.resume:
            raise FileExistsError(f"Run already exists: {output}; use identical arguments with --resume")
        if (output / "completion.json").exists():
            verify_seal(output, contract)
            print(f"[SKIP] {model} seed={seed}: verified completion", flush=True)
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = args.output_root / "interrupted" / model / f"seed_{seed}_{stamp}"
        if output.is_symlink() or args.output_root not in output.resolve().parents:
            raise ValueError(f"Refuse to move unmanaged output: {output}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(destination))
        print(f"[RECOVERY] Incomplete run preserved at {destination}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    locked_json(output / "request.json", contract)
    locked_json(output / "base_training_selection.json", selection.manifest)
    print(f"[RUN] {model} seed={seed}", flush=True)
    if model == "xgboost":
        from baseline_study.xgb import train
    else:
        from baseline_study.neural import train
    train(config, bundle, selection, output)
    validate_metrics(output)
    seal_run(output, contract)
    print(f"[DONE] {output}", flush=True)
    gc.collect()


def validate_metrics(output):
    import numpy as np
    import pandas as pd
    from common.metrics import compute_metrics
    predictions = pd.read_csv(output / "predictions.csv")
    reported = pd.read_csv(output / "metrics.csv")
    config = read_json(output / "request.json")["config"]
    actual = compute_metrics(predictions, tolerance=config.get("evaluation", {}).get("acc_tolerance", .10))
    columns = ["n", "mae", "mse", "rmse", "acc_at_10"]
    key = ["split", "resource"]
    left = reported.set_index(key).sort_index()
    right = actual.set_index(key).sort_index()
    if not left.index.equals(right.index) or not np.allclose(
            left[columns].to_numpy(float), right[columns].to_numpy(float),
            atol=1e-12, rtol=0):
        raise ValueError(f"Metrics disagree with predictions: {output}")
    return reported


def aggregate(args):
    import pandas as pd
    from common.metrics import coerce_valid_mask
    identity = read_json(args.output_root / "protocol/study.json")
    rows, population = [], None
    for model in args.models:
        for seed in args.seeds:
            output = args.output_root / model / f"seed_{seed}"
            seal = verify_seal(output)
            contract = seal["contract"]
            if (contract["study"] != identity or contract["model"] != model
                    or contract["seed"] != seed):
                raise ValueError(f"Run belongs to another study/model/seed: {output}")
            metrics = validate_metrics(output)
            # Assert identical outer job IDs, masks and targets across every run.
            prediction = pd.read_csv(output / "predictions.csv", dtype={"sample_id": str})
            columns = ["sample_id", "split", "resource", "target", "valid"]
            current = prediction[columns].copy()
            current["valid"] = coerce_valid_mask(current["valid"])
            current = current.sort_values(columns[:3]).reset_index(drop=True)
            if population is None:
                population = current
            else:
                pd.testing.assert_frame_equal(population, current)
            for row in metrics.to_dict("records"):
                rows.append({"model": model, "seed": seed, **row})
    frame = pd.DataFrame(rows)
    metric_names = ["n", "mae", "mse", "rmse", "acc_at_10"]
    summary = frame.groupby(["model", "split", "resource"])[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{'sd' if stat == 'std' else stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["seed_n"] = len(args.seeds)
    # A singleton run has no sample SD; retain NaN instead of reporting zero.
    frame.to_csv(args.output_root / "metrics_per_seed.csv", index=False)
    summary.to_csv(args.output_root / "metrics_summary.csv", index=False)
    table = summary[summary["split"] == "test"].copy()
    labels = {"xgboost": "XGBoost", "textcnn": "TextCNN", "lstm": "BiLSTM"}
    lines = ["# Baselines under the primary training-tail protocol", "",
             "Mean and sample SD across the explicitly listed seeds; accuracy SD is in percentage points.",
             f"Seeds: {', '.join(map(str, args.seeds))}.", "",
             "| Resource | Method | n | MAE | MSE | RMSE | Acc@10% |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in table.to_dict("records"):
        values = [f"{row[name+'_mean']:.4f} ± {row[name+'_sd']:.4f}" for name in ["mae", "mse", "rmse"]]
        accuracy = f"{100*row['acc_at_10_mean']:.2f}% ± {100*row['acc_at_10_sd']:.2f} pp"
        lines.append(f"| {row['resource']} | {labels[row['model']]} | {row['n_mean']:.0f} | "
                     + " | ".join(values + [accuracy]) + " |")
    (args.output_root / "table_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(f"[SUMMARY] {args.output_root / 'table_baselines.md'}")


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        plan(args)
        return 0
    if args.stage != "aggregate":
        config, bundle, selection, identity = prepare(args)
        if args.stage in {"train", "all"}:
            # Exclusive file per model prevents duplicate scheduler submissions
            # from training the same destination concurrently. Other models can
            # use the same root in independent jobs.
            for model in args.models:
                lock = args.output_root / f".{model}.running"
                with lock.open("x", encoding="utf-8") as handle:
                    json.dump({"pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, handle)
                try:
                    for seed in args.seeds:
                        run_one(args, model, seed, config, bundle, selection, identity)
                finally:
                    lock.unlink()
    if args.stage in {"aggregate", "all"}:
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
