#!/usr/bin/env python3
"""Aggregate the sealed Hierarchical Same-Script Last-1 cascade reference."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_significance import (  # noqa: E402
    pair_prediction_frames,
    validate_artifact_identity,
)
from common.data import sha256_file  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
from reliability_cascade.integrity import canonical_sha256, validate_variant  # noqa: E402
from scripts.compare_cascade_variants import (  # noqa: E402
    _run_dir,
    _validate_matrix_plan,
    _verify_sealed_run,
)


FORMAL_SEEDS = (40, 41, 42, 43, 44)
EVALUATION_SPLITS = ("validation", "test")
RESOURCES = ("memory", "vram")
METRICS = ("mae", "mse", "rmse", "acc_at_10")
NUMERIC_COLUMNS = (
    "n",
    "mae",
    "mse",
    "rmse",
    "acc_tolerance",
    "hits_at_tolerance",
    "acc_at_10",
)
REFERENCE_COLUMN = "hierarchical_reference_prediction"
METHOD = "Hierarchical Same-Script Last-1"
RUNS_NAME = "hierarchical_same_script_last1_reference_runs.csv"
SUMMARY_NAME = "hierarchical_same_script_last1_reference_summary.csv"
MANIFEST_NAME = "hierarchical_same_script_last1_reference_manifest.json"
OUTPUT_NAMES = (RUNS_NAME, SUMMARY_NAME, MANIFEST_NAME)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and aggregate the five-seed Hierarchical Same-Script Last-1 "
            "reference embedded in reliability-cascade artifacts"
        )
    )
    parser.add_argument("--root", required=True, help="Reliability-cascade matrix root")
    parser.add_argument("--output-dir")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object: {path}")
    return value


def _validate_formal_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    duplicates = sorted({seed for seed in values if values.count(seed) > 1})
    if duplicates:
        raise ValueError(f"Duplicate seeds are not allowed: {duplicates}")
    if values != FORMAL_SEEDS:
        raise ValueError(
            "Formal reference aggregation requires --seeds 40 41 42 43 44 "
            "in this order"
        )
    return values


def _plan_request_hashes(
    plan: dict[str, Any], seeds: Sequence[int]
) -> dict[int, str]:
    runs = plan.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Cascade matrix plan has no runs list")
    output: dict[int, str] = {}
    for entry in runs:
        if not isinstance(entry, dict):
            raise ValueError("Cascade matrix plan contains a non-object run entry")
        try:
            seed = int(entry["seed"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Cascade matrix plan contains an invalid seed") from exc
        if seed in output:
            raise ValueError(f"Cascade matrix plan contains duplicate seed {seed}")
        fingerprint = str(entry.get("run_request_sha256", ""))
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint.lower()
        ):
            raise ValueError(f"Cascade matrix plan has an invalid request hash for seed {seed}")
        output[seed] = fingerprint.lower()
    if set(output) != set(seeds) or len(output) != len(seeds):
        raise ValueError(
            "Cascade matrix plan run seeds do not exactly match the requested seeds"
        )
    return output


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("_config_path", None)
    normalized.pop("seed", None)
    normalized.pop("outputs", None)
    normalized.pop("analysis", None)
    return normalized


def _reference_contract(
    request: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    resolved = request.get("resolved_config")
    details = manifest.get("details")
    if not isinstance(resolved, dict):
        raise ValueError("Cascade run request has no resolved configuration")
    if not isinstance(details, dict):
        raise ValueError("Cascade run manifest details must be an object")
    history = details.get("history_contract")
    effective = details.get("qwen_effective_parameters")
    if not isinstance(history, dict) or not isinstance(effective, dict):
        raise ValueError("Cascade manifest lacks history or Qwen reference semantics")
    return {
        "protocol": request.get("protocol"),
        "variant": request.get("variant"),
        "model": manifest.get("model"),
        "resolved_config": _normalized_config(resolved),
        "qwen_backbone": details.get("qwen_backbone"),
        "qwen_effective_parameters": effective,
        "history_contract": history,
        "reference_prediction_column": REFERENCE_COLUMN,
    }


def _artifact_identity(manifest: dict[str, Any], run_dir: Path) -> dict[str, str]:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        protocol = {}
    values = {
        "data_sha256": manifest.get("data_sha256"),
        "split_fingerprint": manifest.get("split_fingerprint")
        or protocol.get("split_fingerprint"),
        "order_fingerprint": manifest.get("order_fingerprint")
        or protocol.get("order_fingerprint"),
        "split_manifest_sha256": manifest.get("split_manifest_sha256")
        or protocol.get("split_manifest_sha256"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Cascade manifest lacks provenance fields {missing}: {run_dir}")
    return {name: str(value) for name, value in values.items()}


def _validate_metric_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"split", "resource", *NUMERIC_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing metric columns: {missing}")
    output = frame.copy()
    for name in NUMERIC_COLUMNS:
        output[name] = pd.to_numeric(output[name], errors="coerce")
        if output[name].isna().any() or not np.isfinite(output[name].to_numpy(float)).all():
            raise ValueError(f"{path} contains invalid numeric values in {name}")
    if output.duplicated(["split", "resource"]).any():
        raise ValueError(f"{path} contains duplicate split/resource rows")
    expected = {(split, resource) for split in EVALUATION_SPLITS for resource in RESOURCES}
    observed = set(zip(output["split"].astype(str), output["resource"].astype(str)))
    if observed != expected or len(output) != len(expected):
        raise ValueError(f"{path} must contain exactly validation/test x memory/VRAM")
    if (output["n"] <= 0).any() or not np.array_equal(
        output["n"].to_numpy(float), np.round(output["n"].to_numpy(float))
    ):
        raise ValueError(f"{path} contains invalid evaluation counts")
    if not np.array_equal(
        output["hits_at_tolerance"].to_numpy(float),
        np.round(output["hits_at_tolerance"].to_numpy(float)),
    ):
        raise ValueError(f"{path} contains non-integer hit counts")
    if (
        (output["hits_at_tolerance"] < 0)
        | (output["hits_at_tolerance"] > output["n"])
    ).any():
        raise ValueError(f"{path} contains hit counts outside [0,n]")
    if (output[["mae", "mse", "rmse"]] < 0).any().any():
        raise ValueError(f"{path} contains negative error metrics")
    if ((output["acc_at_10"] < 0) | (output["acc_at_10"] > 1)).any():
        raise ValueError(f"{path} contains Acc@10% outside [0,1]")
    if not np.allclose(
        output["acc_at_10"],
        output["hits_at_tolerance"] / output["n"],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(f"{path} has inconsistent hit counts and Acc@10%")
    if not np.allclose(
        output["rmse"], np.sqrt(output["mse"]), rtol=1e-9, atol=1e-12
    ):
        raise ValueError(f"{path} has inconsistent MSE and RMSE")
    return output


def _recompute_reference_metrics(
    predictions: pd.DataFrame, *, tolerance: float
) -> pd.DataFrame:
    if REFERENCE_COLUMN not in predictions:
        raise ValueError(f"Cascade predictions lack {REFERENCE_COLUMN}")
    reference = predictions.copy()
    reference["prediction"] = pd.to_numeric(
        reference[REFERENCE_COLUMN], errors="coerce"
    )
    return compute_metrics(
        reference,
        tolerance=tolerance,
        splits=EVALUATION_SPLITS,
    )


def _assert_saved_matches_recomputed(
    saved: pd.DataFrame, recomputed: pd.DataFrame, path: Path
) -> None:
    keys = ["split", "resource"]
    left = saved.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = recomputed.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError(f"Saved reference metric cells differ from recomputation: {path}")
    for name in NUMERIC_COLUMNS:
        left_values = left[name].to_numpy(float)
        right_values = right[name].to_numpy(float)
        if name in {"n", "hits_at_tolerance"}:
            matches = np.array_equal(left_values, right_values)
        else:
            matches = np.allclose(left_values, right_values, rtol=1e-12, atol=1e-12)
        if not matches:
            raise ValueError(
                f"Saved reference {name} differs from predictions.csv recomputation: {path}"
            )


def _summarize(runs: pd.DataFrame, expected_seed_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, resource), part in runs.groupby(["split", "resource"], sort=True):
        if part["seed"].duplicated().any():
            raise ValueError(f"Duplicate seeds in reference metrics for {split}/{resource}")
        if len(part) != expected_seed_count:
            raise ValueError(
                f"Expected {expected_seed_count} seeds for {split}/{resource}, "
                f"observed {len(part)}"
            )
        if part["n"].nunique() != 1:
            raise ValueError(f"Evaluation counts differ across seeds for {split}/{resource}")
        row: dict[str, Any] = {
            "method": METHOD,
            "split": split,
            "resource": resource,
            "n": int(part["n"].iloc[0]),
            "seeds": int(len(part)),
        }
        for metric in METRICS:
            values = part[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "resource"]).reset_index(drop=True)


def _write_outputs(
    output_dir: Path,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "Reference aggregate outputs already exist: "
            + ", ".join(conflicts)
            + "; use --overwrite to replace only this known artifact set"
        )
    staged: list[tuple[Path, Path]] = []
    try:
        output_hashes: dict[str, str] = {}
        for name, frame in ((RUNS_NAME, runs), (SUMMARY_NAME, summary)):
            temporary = output_dir / f".{name}.tmp-{uuid4().hex}"
            frame.to_csv(temporary, index=False, lineterminator="\n")
            output_hashes[name] = sha256_file(temporary)
            staged.append((temporary, output_dir / name))
        manifest = {**manifest, "outputs": output_hashes}
        temporary = output_dir / f".{MANIFEST_NAME}.tmp-{uuid4().hex}"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True, allow_nan=False)
        staged.append((temporary, output_dir / MANIFEST_NAME))
        for temporary, destination in staged:
            temporary.replace(destination)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = _validate_formal_seeds(args.seeds)
    variant = validate_variant(args.variant)
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    root = _resolve(args.root)
    plan = _validate_matrix_plan(
        root,
        variant=variant,
        seeds=seeds,
        max_tokens=args.max_tokens,
    )
    if plan.get("protocol") != "history-first-reliability-cascade/v1":
        raise ValueError("Cascade matrix uses an unexpected protocol")
    expected_requests = _plan_request_hashes(plan, seeds)

    rows: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    shared_identity: dict[str, str] | None = None
    shared_contract_sha256: str | None = None
    baseline_predictions: pd.DataFrame | None = None

    for seed in seeds:
        run_dir = _run_dir(
            root,
            variant=variant,
            max_tokens=args.max_tokens,
            seed=seed,
        )
        manifest, verification_mode = _verify_sealed_run(
            run_dir,
            plan=plan,
            seed=seed,
            variant=variant,
            max_tokens=args.max_tokens,
        )
        identity = _artifact_identity(manifest, run_dir)
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError(f"Seed {seed} uses a different dataset or frozen split")

        request_path = run_dir / "run_request.json"
        request_document = _read_json(request_path, "cascade run request")
        request = request_document.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"Cascade run request has no request object: {request_path}")
        request_sha256 = str(request_document.get("run_request_sha256", "")).lower()
        if request_sha256 != expected_requests[seed]:
            raise ValueError(f"Seed {seed} run request does not match its matrix plan")
        contract_sha256 = canonical_sha256(_reference_contract(request, manifest))
        if shared_contract_sha256 is None:
            shared_contract_sha256 = contract_sha256
        elif contract_sha256 != shared_contract_sha256:
            raise ValueError(
                f"Seed {seed} uses a different resolved configuration or reference contract"
            )

        predictions_path = run_dir / "predictions.csv"
        predictions = pd.read_csv(
            predictions_path, dtype={"sample_id": str, "job_id": str}
        )
        validate_artifact_identity(predictions, manifest, f"cascade seed {seed}")
        if baseline_predictions is None:
            baseline_predictions = predictions
        else:
            pair_prediction_frames(
                baseline_predictions, predictions, splits=EVALUATION_SPLITS
            )

        resolved = request["resolved_config"]
        tolerance = float((resolved.get("evaluation") or {}).get("acc_tolerance", 0.10))
        reference_path = run_dir / "reference_metrics.csv"
        saved = _validate_metric_frame(pd.read_csv(reference_path), reference_path)
        recomputed = _validate_metric_frame(
            _recompute_reference_metrics(predictions, tolerance=tolerance),
            predictions_path,
        )
        _assert_saved_matches_recomputed(saved, recomputed, reference_path)
        recomputed.insert(0, "seed", seed)
        recomputed.insert(0, "method", METHOD)
        recomputed["run"] = str(run_dir.relative_to(root)).replace("\\", "/")
        rows.append(recomputed)
        input_records.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "run_request_sha256": request_sha256,
                "predictions_sha256": sha256_file(predictions_path),
                "reference_metrics_sha256": sha256_file(reference_path),
                "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
                "artifact_integrity_sha256": sha256_file(
                    run_dir / "artifact_integrity.json"
                ),
                "completion_record_sha256": sha256_file(
                    run_dir / "completion_record.json"
                ),
                "verification_mode": verification_mode,
            }
        )

    if shared_identity is None or shared_contract_sha256 is None:
        raise AssertionError("No reference runs were loaded")
    runs = pd.concat(rows, ignore_index=True).sort_values(
        ["split", "resource", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    expected_rows = len(seeds) * len(EVALUATION_SPLITS) * len(RESOURCES)
    if len(runs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} reference rows, observed {len(runs)}")
    summary = _summarize(runs, len(seeds))

    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else root / "analysis" / "hierarchical_same_script_last1_reference"
    )
    plan_path = root / "cascade_experiment_plan.json"
    manifest = {
        "schema": "hierarchical-same-script-last1-reference-aggregate/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "reference_prediction_column": REFERENCE_COLUMN,
        "reference_scope": (
            "Exact same-user script Last-1 when eligible, ordinary same-user Last-1 "
            "otherwise, and the seed-specific frozen Qwen prediction at cold start."
        ),
        "statistics": "mean and sample standard deviation (ddof=1) across seeds",
        "seeds": list(seeds),
        "variant": variant,
        "max_tokens": args.max_tokens,
        **shared_identity,
        "reference_contract_sha256": shared_contract_sha256,
        "matrix_plan": str(plan_path),
        "matrix_plan_sha256": sha256_file(plan_path),
        "inputs": input_records,
    }
    _write_outputs(
        output_dir,
        runs,
        summary,
        manifest,
        overwrite=bool(args.overwrite),
    )
    print("Hierarchical Same-Script Last-1 five-seed summary:")
    print(summary.to_string(index=False))
    print(f"Reference aggregate artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
