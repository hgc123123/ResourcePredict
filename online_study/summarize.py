"""Summarize technical repeats within seed before comparing training seeds.

Each run's P95/P99 is a within-run request percentile. Its repeated-run average
and between-seed SD are reported explicitly; requests are never pooled to
manufacture a different percentile or an increased number of training seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

from online_study.suite import canonical_hash, read_json, report_setting, verify_report


def _numeric_metrics(value: Any, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            result.update(_numeric_metrics(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite benchmark metric: {prefix}")
        result[prefix] = number
    return result


def metrics_from_report(report: dict[str, Any]) -> dict[str, float]:
    metrics = _numeric_metrics(report["all_requests"], "all_requests")
    for route, value in sorted(report.get("by_route", {}).items()):
        metrics.update(_numeric_metrics(value, f"by_route.{route}"))
    for scope in ("live_accuracy", "archived_accuracy"):
        for row in report.get(scope, []):
            resource = str(row["resource"])
            metrics.update(_numeric_metrics({key: value for key, value in row.items() if key != "resource"}, f"{scope}.{resource}"))
    metrics.update(_numeric_metrics(report.get("offline_parity", {}), "offline_parity"))
    metrics.update(_numeric_metrics(report.get("archived_execution_comparison", {}), "archived_execution_comparison"))
    metrics.update(_numeric_metrics(report.get("base_batch_one_parity", {}), "base_batch_one_parity"))
    for field in ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
                  "max_cached_tensor_bytes_including_pending", "max_resident_cache_entries",
                  "max_pending_same_timestamp_entries", "cache_evictions_total_including_prefix"):
        if field in report:
            metrics.update(_numeric_metrics(report[field], field))
    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def summarize_plan(plan_path: Path, output: Path) -> dict[str, list[dict[str, Any]]]:
    plan = read_json(plan_path)
    saved_hash = plan.get("plan_sha256")
    identity = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if saved_hash != canonical_hash(identity):
        raise ValueError("Timing plan checksum mismatch")
    runs = plan.get("runs", [])
    if not runs:
        raise ValueError("The plan contains no runs")
    expected_matrix = {(str(case["case"]), int(seed), repeat)
                       for case in plan["cases"] for seed in plan["settings"]["seeds"]
                       for repeat in range(1, int(plan["settings"]["technical_repeats"]) + 1)}
    observed_matrix = [(str(run["case"]), int(run["seed"]), int(run["repeat"])) for run in runs]
    if len(observed_matrix) != len(set(observed_matrix)) or set(observed_matrix) != expected_matrix:
        raise ValueError("The plan is not a complete, unique seed/case/technical-repeat matrix")
    per_run: list[dict[str, Any]] = []
    environment_signature: str | None = None
    data_signature: str | None = None
    source_signature: str | None = None
    count_by_seed: dict[int, int] = {}
    for run in runs:
        report = verify_report(run, require_receipt=True)
        if not report.get("hardware") or not report.get("software"):
            raise ValueError("Hardware and software provenance must be recorded for every run")
        current_environment = canonical_hash({"hardware": report["hardware"], "software": report["software"],
                                              "num_threads": report_setting(report, "num_threads") or report.get("torch_num_threads")})
        current_data = canonical_hash({"data_sha256": report.get("data_sha256"), "split_fingerprint": report.get("split_fingerprint")})
        if not report.get("data_sha256") or not report.get("split_fingerprint"):
            raise ValueError("Dataset and split fingerprints are required")
        if not report.get("source_contract", {}).get("sha256"):
            raise ValueError("Benchmark source contract is required")
        current_source = report["source_contract"]["sha256"]
        if environment_signature is not None and environment_signature != current_environment:
            raise ValueError("Mixed hardware/software/thread settings cannot be summarized as one controlled study")
        if data_signature is not None and data_signature != current_data:
            raise ValueError("Mixed datasets or splits cannot be summarized")
        if source_signature is not None and source_signature != current_source:
            raise ValueError("Mixed benchmark source contracts cannot be summarized")
        environment_signature, data_signature, source_signature = current_environment, current_data, current_source
        seed = int(run["seed"])
        count = int(report["all_requests"]["n"])
        if count <= 0:
            raise ValueError("No test requests measured")
        if seed in count_by_seed and count_by_seed[seed] != count:
            raise ValueError(f"Different test request counts within seed {seed}")
        count_by_seed[seed] = count
        for metric, value in metrics_from_report(report).items():
            per_run.append({"case": run["case"], "operation": run["operation"],
                            "cache_capacity_entries": run["cache_capacity_entries"], "cache_start": run["cache_start"],
                            "seed": seed, "repeat": int(run["repeat"]), "metric": metric, "value": value})
    if len(set(count_by_seed.values())) != 1:
        raise ValueError("Different test request populations across training seeds")
    by_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in per_run:
        key = tuple(row[field] for field in ("case", "operation", "cache_capacity_entries", "cache_start", "seed", "metric"))
        by_seed.setdefault(key, []).append(row)
    per_seed: list[dict[str, Any]] = []
    expected_repeats = int(plan["settings"]["technical_repeats"])
    for key, rows in sorted(by_seed.items()):
        repeats = [int(row["repeat"]) for row in rows]
        if len(set(repeats)) != len(repeats):
            raise ValueError("Duplicate technical repeat in a seed/case/metric group")
        values = [float(row["value"]) for row in rows]
        # A route may have zero observations in a run: no artificial zero
        # latency is inserted for the unobserved route. Its n is still present.
        row = dict(zip(("case", "operation", "cache_capacity_entries", "cache_start", "seed", "metric"), key))
        row.update(mean_over_technical_repeats=statistics.mean(values),
                   sd_over_technical_repeats=statistics.stdev(values) if len(values) > 1 else 0.0,
                   observed_repeats=len(values), expected_repeats=expected_repeats)
        per_seed.append(row)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in per_seed:
        key = tuple(row[field] for field in ("case", "operation", "cache_capacity_entries", "cache_start", "metric"))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    expected_seeds = set(int(seed) for seed in plan["settings"]["seeds"])
    for key, rows in sorted(groups.items()):
        values = [float(row["mean_over_technical_repeats"]) for row in rows]
        observed_seeds = {int(row["seed"]) for row in rows}
        row = dict(zip(("case", "operation", "cache_capacity_entries", "cache_start", "metric"), key))
        row.update(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else 0.0,
                   n_training_seeds=len(observed_seeds), expected_training_seeds=len(expected_seeds),
                   all_seeds_and_repeats_observed=observed_seeds == expected_seeds and all(r["observed_repeats"] == expected_repeats for r in rows))
        summary.append(row)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_run.csv", per_run, ["case", "operation", "cache_capacity_entries", "cache_start", "seed", "repeat", "metric", "value"])
    _write_csv(output / "per_seed.csv", per_seed, ["case", "operation", "cache_capacity_entries", "cache_start", "seed", "metric", "mean_over_technical_repeats", "sd_over_technical_repeats", "observed_repeats", "expected_repeats"])
    _write_csv(output / "summary.csv", summary, ["case", "operation", "cache_capacity_entries", "cache_start", "metric", "mean", "sd", "n_training_seeds", "expected_training_seeds", "all_seeds_and_repeats_observed"])
    wanted = ["all_requests.total_ms.mean", "all_requests.total_ms.p95", "all_requests.total_ms.p99",
              "all_requests.encoder_forward_rate", "all_requests.cache_hit_rate_all_jobs", "all_requests.history_bypass_job_rate",
              "all_requests.serial_service_throughput_jobs_per_second"]
    wanted.extend(f"live_accuracy.{resource}.{metric}" for resource in ("memory", "vram")
                  for metric in ("mae", "rmse", "acc_at_10"))
    lines = ["# Complete local prediction-path timing", "",
             "Technical repeats are averaged within each training seed first. The reported SD is the sample SD across training-seed means.", "",
             "P95/P99 entries average within-run request percentiles; they are not percentiles of pooled requests. Fractions are in [0,1].", "",
             "Frozen diagnostic operations reuse selected experts/models without reselecting policy. This in-process benchmark does not measure scheduler placement, RPC queueing, or utilization gains.", "",
             "Prediction parity uses an independent uncached batch-one reference and a canonical offline cascade. Archived batched accuracy is reported separately; archived execution differences must not be described as exact reproduction.", "",
             "| Case | Metric | Mean | Between-seed SD | Seeds |", "|---|---|---:|---:|---:|"]
    for row in summary:
        if row["metric"] in wanted:
            lines.append(f"| {row['case']} | {row['metric']} | {row['mean']:.6g} | {row['sd']:.6g} | {row['n_training_seeds']} |")
    lines.extend(["", f"Plan SHA256: `{saved_hash}`", f"Source contract SHA256: `{source_signature}`", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return {"per_run": per_run, "per_seed": per_seed, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    plan = Path(args.plan).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else plan.parent / "summary"
    summarize_plan(plan, output)
    print(f"Verified timing summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
