#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import get_value, load_config, resolve_path  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402


REQUIRED_COLUMNS = {
    "split",
    "resource",
    "n",
    "mae",
    "mse",
    "rmse",
    "acc_tolerance",
    "hits_at_tolerance",
    "acc_at_10",
}
NUMERIC_COLUMNS = [
    "n",
    "mae",
    "mse",
    "rmse",
    "acc_tolerance",
    "hits_at_tolerance",
    "acc_at_10",
]
METRIC_KEYS = ["split", "resource"]
DIAGNOSTIC_MANIFEST_FLAGS = (
    "uses_targets",
    "diagnostic_only",
    "not_a_deployable_model",
)


def _normalized_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove run-location identity while retaining all experimental semantics."""
    normalized = copy.deepcopy(config)
    normalized.pop("_config_path", None)
    normalized.pop("seed", None)
    normalized.pop("outputs", None)
    # Analysis settings do not affect model fitting or prediction generation.
    normalized.pop("analysis", None)
    return normalized


def _manifest_experiment_semantics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return outcome-independent manifest fields that define a replicate family."""
    model = str(manifest.get("model", ""))
    details = manifest.get("details") or {}
    if model.startswith("qwen_"):
        return {
            "model": model,
            "variant": details.get("variant"),
            "enabled_modalities": details.get("enabled_modalities"),
            "fusion_mode": details.get("fusion_mode"),
            "effective_parameters": details.get("effective_parameters"),
        }
    if model.startswith("reliability_cascade_"):
        calibration = details.get("calibration_protocol") or {}
        selection = details.get("selection_policy") or {}
        return {
            "model": model,
            "variant": details.get("variant"),
            "qwen_backbone": details.get("qwen_backbone"),
            "qwen_effective_parameters": details.get("qwen_effective_parameters"),
            "history_contract": details.get("history_contract"),
            "calibration_protocol": calibration,
            "selection_objective": selection.get("objective"),
            "selection_constraints": selection.get("constraints"),
            "route_feature_whitelist": details.get("route_feature_whitelist"),
            "route_decision_uses_qwen_output": details.get(
                "route_decision_uses_qwen_output"
            ),
        }
    if "hybrid" in model:
        return {
            "model": model,
            "qwen_variant": manifest.get("qwen_variant"),
            "qwen_effective_parameters": manifest.get("qwen_effective_parameters"),
            "last1_protocol": manifest.get("last1_protocol"),
            "selection_protocol_version": manifest.get("selection_protocol_version"),
            "selection_objective": manifest.get("selection_objective"),
            "selection_constraints": manifest.get("selection_constraints"),
            "same_script_beta_grid": manifest.get("same_script_beta_grid"),
        }
    if model == "last-1":
        return {
            "model": model,
            "protocol": details.get("protocol"),
            "causal_online_protocol": details.get("causal_online_protocol"),
            "fallback": details.get("fallback"),
        }
    return {"model": model}


def _experiment_family_signature(path: Path, manifest: dict[str, Any]) -> str:
    """Fingerprint the resolved, seed-independent configuration for one run."""
    config_path = path.parent / "config.resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Metrics have no resolved configuration for replicate audit: {config_path}"
        )
    config = load_config(config_path)
    payload = {
        "resolved_config": _normalized_experiment_config(config),
        "manifest_semantics": _manifest_experiment_semantics(manifest),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate measured model metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", action="append", default=[], dest="models")
    parser.add_argument("--allow-mixed-data", action="store_true")
    parser.add_argument(
        "--group-by-context",
        action="store_true",
        help="Treat checkpoint max_tokens as an experiment dimension",
    )
    parser.add_argument(
        "--include-noncausal",
        action="store_true",
        help="Include ordered-row Last-1 smoke runs in the aggregate",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Metrics have no run manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Run manifest must be a JSON object: {manifest_path}")
    for required in ("model", "seed", "data_sha256"):
        if value.get(required) is None:
            raise ValueError(f"Run manifest is missing {required}: {manifest_path}")
    return value


def _split_fingerprint(manifest: dict[str, Any], path: Path) -> str:
    protocol = manifest.get("protocol") or {}
    value = manifest.get("split_fingerprint") or protocol.get("split_fingerprint")
    if not value:
        raise ValueError(f"Run manifest has no split fingerprint: {path}")
    return str(value)


def _is_forbidden_diagnostic(manifest: dict[str, Any]) -> bool:
    """Keep target-using/post-hoc artifacts out of formal model aggregation."""
    details = manifest.get("details") or {}
    return any(
        manifest.get(name) is True or details.get(name) is True
        for name in DIAGNOSTIC_MANIFEST_FLAGS
    )


def _context_max_tokens(manifest: dict[str, Any]) -> int | None:
    details = manifest.get("details") or {}
    effective = details.get("effective_parameters")
    if not isinstance(effective, dict):
        effective = manifest.get("qwen_effective_parameters")
    if not isinstance(effective, dict) or effective.get("max_tokens") is None:
        return None
    value = int(effective["max_tokens"])
    if value <= 0:
        raise ValueError("Run manifest contains a non-positive Qwen max_tokens value")
    return value


def _validate_metrics(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing metric columns: {sorted(missing)}")
    output = frame.copy()
    for name in NUMERIC_COLUMNS:
        original_missing = output[name].isna()
        output[name] = pd.to_numeric(output[name], errors="coerce")
        newly_invalid = output[name].isna() & ~original_missing
        if newly_invalid.any() or output[name].isna().any():
            raise ValueError(f"{path} contains non-numeric or missing values in {name}")
        if not np.isfinite(output[name].to_numpy(float)).all():
            raise ValueError(f"{path} contains non-finite values in {name}")
    if (output["n"] <= 0).any():
        raise ValueError(f"{path} contains non-positive evaluation counts")
    if not np.allclose(output["n"], np.round(output["n"]), rtol=0.0, atol=0.0):
        raise ValueError(f"{path} contains non-integer evaluation counts")
    if not np.allclose(
        output["hits_at_tolerance"],
        np.round(output["hits_at_tolerance"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(f"{path} contains non-integer hit counts")
    if ((output["hits_at_tolerance"] < 0) | (output["hits_at_tolerance"] > output["n"])).any():
        raise ValueError(f"{path} contains hit counts outside [0,n]")
    if (output[["mae", "mse", "rmse"]] < 0).any().any():
        raise ValueError(f"{path} contains negative error metrics")
    if ((output["acc_at_10"] < 0) | (output["acc_at_10"] > 1)).any():
        raise ValueError(f"{path} contains Acc@10% outside [0,1]")
    expected_acc = output["hits_at_tolerance"] / output["n"]
    if not np.allclose(output["acc_at_10"], expected_acc, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{path} has inconsistent hit counts and Acc@10%")
    if not np.allclose(output["rmse"], np.sqrt(output["mse"]), rtol=1e-9, atol=1e-12):
        raise ValueError(f"{path} has inconsistent MSE and RMSE")
    if output.duplicated(["split", "resource"]).any():
        raise ValueError(f"{path} contains duplicate split/resource metric rows")
    return output


def _recompute_and_compare(
    saved: pd.DataFrame,
    metrics_path: Path,
    tolerance: float,
    manifest: dict[str, Any],
) -> None:
    predictions_path = metrics_path.parent / "predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Metrics cannot be verified without predictions: {predictions_path}"
        )
    predictions = pd.read_csv(
        predictions_path,
        dtype={"sample_id": str, "job_id": str},
    )
    if "model" not in predictions or "seed" not in predictions:
        raise ValueError(f"Predictions have no model/seed identity: {predictions_path}")
    if predictions["model"].isna().any():
        raise ValueError(f"Predictions contain missing model identities: {predictions_path}")
    models = predictions["model"].astype(str).unique().tolist()
    seeds = pd.to_numeric(predictions["seed"], errors="coerce")
    if len(models) != 1 or models[0] != str(manifest["model"]):
        raise ValueError(
            f"Prediction model disagrees with run manifest: {predictions_path}"
        )
    manifest_seed = int(manifest["seed"])
    if (
        seeds.isna().any()
        or len(seeds.unique()) != 1
        or int(seeds.iloc[0]) != manifest_seed
        or not (seeds == manifest_seed).all()
    ):
        raise ValueError(
            f"Prediction seed disagrees with run manifest: {predictions_path}"
        )
    recomputed = _validate_metrics(
        compute_metrics(predictions, tolerance=tolerance),
        predictions_path,
    )
    left = saved.sort_values(METRIC_KEYS, kind="mergesort").reset_index(drop=True)
    right = recomputed.sort_values(METRIC_KEYS, kind="mergesort").reset_index(drop=True)
    if not left[METRIC_KEYS].equals(right[METRIC_KEYS]):
        raise ValueError(f"Saved metrics do not cover the recomputed keys: {metrics_path}")
    for column in NUMERIC_COLUMNS:
        left_values = left[column].to_numpy(float)
        right_values = right[column].to_numpy(float)
        if column in {"n", "hits_at_tolerance"}:
            matches = np.array_equal(left_values, right_values)
        else:
            matches = np.allclose(left_values, right_values, rtol=1e-9, atol=1e-12)
        if not matches:
            raise ValueError(
                f"Saved {column} differs from predictions.csv recomputation: {metrics_path}"
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    output_root = resolve_path(config, get_value(config, "outputs.root", "outputs"))
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {output_root}")
    destination = resolve_path(
        config, args.output or str(output_root / "aggregate_metrics.csv")
    )

    rows = []
    data_hashes = set()
    split_fingerprints = set()
    tolerance = float(get_value(config, "evaluation.acc_tolerance", 0.10))
    for path in sorted(output_root.rglob("metrics.csv")):
        manifest = _manifest(path)
        relative_run = str(path.parent.relative_to(output_root)).replace("\\", "/")
        if _is_forbidden_diagnostic(manifest):
            print(
                f"[SKIP] diagnostic/non-deployable result: {path.parent}",
                file=sys.stderr,
            )
            continue
        model = str(manifest["model"])
        if args.models and model not in set(args.models):
            continue
        details = manifest.get("details") or {}
        is_noncausal_last1 = model == "last-1" and not bool(
            details.get("causal_online_protocol", False)
        )
        if (
            is_noncausal_last1
            and not args.include_noncausal
        ):
            print(f"[SKIP] noncausal Last-1 smoke run: {path.parent}", file=sys.stderr)
            continue
        frame = _validate_metrics(pd.read_csv(path), path)
        _recompute_and_compare(frame, path, tolerance, manifest)
        seed = manifest["seed"]
        data_hash = str(manifest["data_sha256"])
        split_fingerprint = _split_fingerprint(manifest, path)
        data_hashes.add(data_hash)
        split_fingerprints.add(split_fingerprint)
        aggregate_model = model
        if model == "last-1" and args.include_noncausal:
            protocol_name = str(details.get("protocol", "")).strip()
            if not protocol_name:
                raise ValueError(f"Last-1 manifest has no protocol: {path}")
            aggregate_model = f"last-1_{protocol_name}"
        frame.insert(0, "model", aggregate_model)
        frame.insert(1, "seed", seed)
        max_tokens = _context_max_tokens(manifest)
        if (
            model.startswith("qwen_")
            or "hybrid" in model
            or model.startswith("reliability_cascade_")
        ) and max_tokens is None:
            raise ValueError(f"Qwen max_tokens is absent from the run manifest: {path}")
        frame.insert(2, "max_tokens", max_tokens)
        frame["experiment_family_sha256"] = _experiment_family_signature(
            path, manifest
        )
        frame["run"] = relative_run
        frame["metrics_path"] = str(path)
        frame["data_sha256"] = data_hash
        frame["split_fingerprint"] = split_fingerprint
        rows.append(frame)

    if not rows:
        raise FileNotFoundError("No measured metrics.csv files matched the requested models")
    if len(data_hashes) > 1 and not args.allow_mixed_data:
        raise ValueError(
            "Runs reference multiple dataset hashes; refuse to aggregate incomparable "
            "results without --allow-mixed-data"
        )
    if len(split_fingerprints) > 1 and not args.allow_mixed_data:
        raise ValueError(
            "Runs use different frozen splits; refuse to aggregate incomparable "
            "results without --allow-mixed-data"
        )

    combined = pd.concat(rows, ignore_index=True)
    context_dimension = ["max_tokens"] if args.group_by_context else []
    comparison_scope = (
        ["data_sha256", "split_fingerprint"] if args.allow_mixed_data else []
    )
    if not args.group_by_context:
        context_scope = comparison_scope + ["model"]
        context_variation = combined.groupby(context_scope, dropna=False)[
            "max_tokens"
        ].nunique(dropna=True)
        if (context_variation > 1).any():
            raise ValueError(
                "Runs for the same model use multiple Qwen context lengths; rerun "
                "with --group-by-context instead of treating them as random-seed replicates"
            )
    combined = combined.sort_values(
        ["model", *context_dimension, "seed", "split", "resource"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    run_identity = comparison_scope + [
        "model",
        *context_dimension,
        "seed",
        "split",
        "resource",
    ]
    duplicate_runs = combined.duplicated(run_identity)
    if duplicate_runs.any():
        examples = combined.loc[
            duplicate_runs,
            ["model", *context_dimension, "seed", "split", "resource", "run"],
        ].head(10)
        raise ValueError(
            "Multiple output directories claim the same model/seed evaluation:\n"
            + examples.to_string(index=False)
        )
    summary_groups = comparison_scope + [
        "model",
        *context_dimension,
        "split",
        "resource",
    ]
    signature_variation = combined.groupby(summary_groups, dropna=False)[
        "experiment_family_sha256"
    ].nunique()
    if (signature_variation > 1).any():
        mismatched = signature_variation[signature_variation > 1]
        raise ValueError(
            "Resolved training/evaluation configurations differ within a replicate "
            "group; do not pool these runs as random seeds:\n"
            + mismatched.to_string()
        )
    count_variation = combined.groupby(summary_groups, dropna=False)["n"].nunique()
    if (count_variation > 1).any():
        raise ValueError("Evaluation counts differ across seeds for the same model/resource")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    summary = (
        combined.groupby(summary_groups, as_index=False, dropna=False)
        .agg(
            runs=("seed", "count"),
            n=("n", "first"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            acc_at_10_mean=("acc_at_10", "mean"),
            acc_at_10_std=("acc_at_10", "std"),
        )
        .sort_values(summary_groups, kind="mergesort")
        .reset_index(drop=True)
    )
    summary_path = destination.with_name(
        f"{destination.stem}_summary{destination.suffix or '.csv'}"
    )
    summary.to_csv(summary_path, index=False)
    print(combined.to_string(index=False))
    print(f"Aggregated measured metrics: {destination}")
    print(f"Across-seed mean/std summary: {summary_path}")


if __name__ == "__main__":
    main()
