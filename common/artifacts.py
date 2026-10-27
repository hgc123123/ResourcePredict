from __future__ import annotations

import json
import platform
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .config import get_value, resolve_path, write_resolved_config
from .data import DatasetBundle
from .metrics import (
    add_error_columns,
    coerce_valid_mask,
    compute_metrics,
    validate_prediction_frame,
)


def dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "xgboost",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def output_directory(config: Dict[str, Any], model_name: str, suffix: Optional[str] = None) -> Path:
    root = resolve_path(config, get_value(config, "outputs.root", "outputs"))
    result = root / model_name
    if suffix:
        result = result / suffix
    result.mkdir(parents=True, exist_ok=True)
    return result


def build_prediction_frame(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    model_name: str,
    seed: int,
    clip_predictions: bool,
    extra_columns: Optional[Mapping[str, np.ndarray]] = None,
) -> pd.DataFrame:
    rows = []
    for resource, target_column, valid_column in (
        ("memory", "target_memory", "valid_memory"),
        ("vram", "target_vram", "valid_vram"),
    ):
        raw_prediction = np.asarray(predictions[resource], dtype=float)
        if len(raw_prediction) != len(frame):
            raise ValueError(f"Prediction length mismatch for {resource}")
        prediction = np.clip(raw_prediction, 0.0, 1.0) if clip_predictions else raw_prediction
        part = pd.DataFrame(
            {
                "sample_id": frame["sample_id"].astype(str).to_numpy(),
                "job_id": frame["job_id"].astype(str).to_numpy(),
                "user_id": frame["user_id"].astype(str).to_numpy(),
                "split": frame["split"].astype(str).to_numpy(),
                "resource": resource,
                "target": frame[target_column].to_numpy(float),
                "raw_prediction": raw_prediction,
                "prediction": prediction,
                "valid": frame[valid_column].to_numpy(bool),
                "model": model_name,
                "seed": int(seed),
            }
        )
        context_columns = (
            "job_name",
            "partition",
            "submit_time",
            "start_time",
            "completion_time",
            "job_state",
            "exit_code",
            "gpu_count",
        )
        for name in context_columns:
            if name in frame.columns and frame[name].notna().any():
                part[name] = frame[name].to_numpy()
        if extra_columns:
            for name, values in extra_columns.items():
                array = np.asarray(values)
                if len(array) != len(frame):
                    raise ValueError(f"Extra column length mismatch: {name}")
                part[name] = array
        rows.append(part)
    return add_error_columns(pd.concat(rows, ignore_index=True))


def _validate_evaluation_predictions(
    bundle: DatasetBundle,
    predictions: pd.DataFrame,
) -> None:
    validate_prediction_frame(predictions)
    allowed_splits = {"train", "validation", "test"}
    allowed_resources = {"memory", "vram"}
    observed_splits = set(predictions["split"].astype(str))
    observed_resources = set(predictions["resource"].astype(str))
    if not observed_splits.issubset(allowed_splits):
        raise ValueError(f"Prediction frame contains unknown splits: {sorted(observed_splits)}")
    if not observed_resources.issubset(allowed_resources):
        raise ValueError(
            f"Prediction frame contains unknown resources: {sorted(observed_resources)}"
        )

    evaluation = bundle.frame.loc[
        bundle.frame["split"].isin(["validation", "test"])
    ].copy()
    expected_parts = []
    for resource, target_column, valid_column in (
        ("memory", "target_memory", "valid_memory"),
        ("vram", "target_vram", "valid_vram"),
    ):
        expected_parts.append(
            pd.DataFrame(
                {
                    "sample_id": evaluation["sample_id"].astype(str).to_numpy(),
                    "job_id_expected": evaluation["job_id"].astype(str).to_numpy(),
                    "split": evaluation["split"].astype(str).to_numpy(),
                    "resource": resource,
                    "target_expected": evaluation[target_column].to_numpy(float),
                    "valid_expected": evaluation[valid_column].to_numpy(bool),
                }
            )
        )
    expected = pd.concat(expected_parts, ignore_index=True)
    actual = predictions.loc[
        predictions["split"].astype(str).isin(["validation", "test"])
    ].copy()
    actual["sample_id"] = actual["sample_id"].astype(str)
    actual["split"] = actual["split"].astype(str)
    actual["resource"] = actual["resource"].astype(str)
    keys = ["sample_id", "split", "resource"]
    expected_keys = pd.MultiIndex.from_frame(expected[keys])
    actual_keys = pd.MultiIndex.from_frame(actual[keys])
    missing = expected_keys.difference(actual_keys)
    extra = actual_keys.difference(expected_keys)
    if len(missing) or len(extra):
        raise ValueError(
            "Evaluation prediction coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    paired = expected.merge(actual, on=keys, how="inner", validate="one_to_one")
    if not (paired["job_id_expected"].astype(str) == paired["job_id"].astype(str)).all():
        raise ValueError("Evaluation predictions contain mismatched job_id values")
    actual_valid = coerce_valid_mask(paired["valid"], "prediction valid")
    expected_valid = paired["valid_expected"].to_numpy(bool)
    if not np.array_equal(actual_valid, expected_valid):
        raise ValueError("Evaluation prediction validity mask does not match the dataset")

    expected_target = paired["target_expected"].to_numpy(float)
    actual_target = pd.to_numeric(paired["target"], errors="coerce").to_numpy(float)
    expected_finite = np.isfinite(expected_target)
    actual_finite = np.isfinite(actual_target)
    if not np.array_equal(expected_finite, actual_finite):
        raise ValueError("Evaluation prediction target missingness does not match the dataset")
    same_nonfinite = (
        (np.isnan(expected_target) & np.isnan(actual_target))
        | (np.isposinf(expected_target) & np.isposinf(actual_target))
        | (np.isneginf(expected_target) & np.isneginf(actual_target))
    )
    mismatch = (
        expected_finite
        & ~np.isclose(expected_target, actual_target, rtol=0.0, atol=1e-12)
    ) | (~expected_finite & ~same_nonfinite)
    if mismatch.any():
        raise ValueError(
            f"Evaluation prediction targets differ from the dataset for {int(mismatch.sum())} rows"
        )


def write_evaluation(
    config: Dict[str, Any],
    bundle: DatasetBundle,
    model_name: str,
    predictions: pd.DataFrame,
    output_dir: Path,
    details: Optional[Dict[str, Any]] = None,
    run_seed: Optional[int] = None,
) -> pd.DataFrame:
    tolerance = float(get_value(config, "evaluation.acc_tolerance", 0.10))
    _validate_evaluation_predictions(bundle, predictions)
    metrics = compute_metrics(predictions, tolerance=tolerance)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    write_resolved_config(config, output_dir / "config.resolved.yaml")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "seed": int(get_value(config, "seed", 42) if run_seed is None else run_seed),
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
        "data_path": str(bundle.data_path),
        "data_sha256": bundle.data_sha256,
        "split_manifest": str(bundle.manifest_path) if bundle.manifest_path else None,
        "split_manifest_sha256": bundle.protocol.get("split_manifest_sha256"),
        "split_fingerprint": bundle.protocol.get("split_fingerprint"),
        "order_fingerprint": bundle.protocol.get("order_fingerprint"),
        "protocol": bundle.protocol,
        "details": details or {},
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    return metrics
