#!/usr/bin/env python3
"""Seal and verify one completed conventional-baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import (  # noqa: E402
    get_value,
    load_config,
    resolved_serializable_config,
)
from common.data import sha256_file  # noqa: E402
from scripts.aggregate_results import (  # noqa: E402
    _recompute_and_compare,
    _split_fingerprint,
    _validate_metrics,
)


BASELINE_MODELS = ("xgboost", "textcnn", "lstm")
COMPLETION_FILE = "baseline_completion.json"
COMPLETION_SCHEMA = "conventional-baseline-completion/v1"
COMMON_ARTIFACTS = (
    "predictions.csv",
    "metrics.csv",
    "run_manifest.json",
    "config.resolved.yaml",
)
MODEL_ARTIFACTS = {
    "xgboost": (
        "preprocessor.joblib",
        "feature_names.json",
        "model_memory.joblib",
        "model_vram.joblib",
    ),
    "textcnn": (
        "model.pt",
        "training_history.csv",
        "architecture.json",
        "token_vocabulary.json",
        "user_vocabulary.json",
    ),
    "lstm": (
        "model.pt",
        "training_history.csv",
        "architecture.json",
        "token_vocabulary.json",
        "user_vocabulary.json",
    ),
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a baseline run before treating it as complete on resume"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-model", required=True, choices=BASELINE_MODELS)
    parser.add_argument("--expected-seed", required=True, type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument(
        "--write-seal",
        action="store_true",
        help="validate the completed run and atomically write its completion seal",
    )
    return parser.parse_args(argv)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def _require_nonempty_files(run_dir: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Baseline run is missing required artifacts: {', '.join(missing)}"
        )
    empty = [name for name in names if (run_dir / name).stat().st_size <= 0]
    if empty:
        raise ValueError(f"Baseline run has empty artifacts: {', '.join(empty)}")


def _read_json(path: Path, expected_type: type) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, expected_type):
        raise ValueError(
            f"JSON artifact must contain {expected_type.__name__}: {path}"
        )
    return value


def _normalized_sha256(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not HEX_SHA256.fullmatch(normalized):
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256 value")
    return normalized


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, str) and str(integer) != value.strip():
        raise ValueError(f"{field} must be an integer")
    return integer


def _expected_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    data_sha256 = _normalized_sha256(
        get_value(dict(config), "data.expected_sha256"),
        "data.expected_sha256",
    )
    split_fingerprint = _normalized_sha256(
        get_value(dict(config), "split.expected_fingerprint"),
        "split.expected_fingerprint",
    )
    expected_counts = get_value(dict(config), "split.expected_counts")
    valid_counts = get_value(dict(config), "data.expected_valid_by_split")
    if not isinstance(expected_counts, dict):
        raise ValueError("Configuration must define split.expected_counts")
    if not isinstance(valid_counts, dict):
        raise ValueError("Configuration must define data.expected_valid_by_split")
    evaluation_counts: dict[str, int] = {}
    expected_valid: dict[tuple[str, str], int] = {}
    for split in ("validation", "test"):
        evaluation_counts[split] = _integer(
            expected_counts.get(split), f"split.expected_counts.{split}"
        )
        for resource in ("memory", "vram"):
            resource_counts = valid_counts.get(resource)
            if not isinstance(resource_counts, dict):
                raise ValueError(
                    f"Configuration must define data.expected_valid_by_split.{resource}"
                )
            expected_valid[(split, resource)] = _integer(
                resource_counts.get(split),
                f"data.expected_valid_by_split.{resource}.{split}",
            )
    return {
        "data_sha256": data_sha256,
        "split_fingerprint": split_fingerprint,
        "evaluation_counts": evaluation_counts,
        "valid_counts": expected_valid,
        "tolerance": float(get_value(dict(config), "evaluation.acc_tolerance", 0.10)),
    }


def _verify_resolved_config(
    path: Path,
    *,
    expected_config: Mapping[str, Any],
    expected_seed: int,
    expected_data_sha256: str,
    expected_split_fingerprint: str,
) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            resolved = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid resolved configuration: {path}") from error
    if not isinstance(resolved, dict):
        raise ValueError(f"Resolved configuration must be a mapping: {path}")
    if _integer(resolved.get("seed"), "resolved config seed") != expected_seed:
        raise ValueError("Resolved configuration seed does not match the requested seed")
    if (
        _normalized_sha256(
            get_value(resolved, "data.expected_sha256"),
            "resolved config data.expected_sha256",
        )
        != expected_data_sha256
    ):
        raise ValueError("Resolved configuration data SHA-256 does not match the contract")
    if (
        _normalized_sha256(
            get_value(resolved, "split.expected_fingerprint"),
            "resolved config split.expected_fingerprint",
        )
        != expected_split_fingerprint
    ):
        raise ValueError("Resolved configuration split fingerprint does not match the contract")
    expected_resolved = resolved_serializable_config(dict(expected_config))
    if resolved != expected_resolved:
        raise ValueError(
            "Resolved configuration differs from the normalized requested experiment"
        )
    return _canonical_sha256(resolved)


def _verify_prediction_coverage(
    predictions: pd.DataFrame,
    evaluation_counts: Mapping[str, int],
) -> None:
    required = {"sample_id", "split", "resource"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing coverage columns: {missing}")
    for split, expected in evaluation_counts.items():
        for resource in ("memory", "vram"):
            count = int(
                (
                    predictions["split"].astype(str).eq(split)
                    & predictions["resource"].astype(str).eq(resource)
                ).sum()
            )
            if count != expected:
                raise ValueError(
                    "Prediction coverage differs from the configured split count: "
                    f"split={split}, resource={resource}, expected={expected}, actual={count}"
                )


def _verify_metric_counts(
    metrics: pd.DataFrame,
    expected_valid: Mapping[tuple[str, str], int],
) -> None:
    indexed = metrics.set_index(["split", "resource"], verify_integrity=True)
    for key, expected in expected_valid.items():
        if key not in indexed.index:
            raise ValueError(f"Metrics are missing split/resource row: {key}")
        actual = int(indexed.loc[key, "n"])
        if actual != expected:
            raise ValueError(
                "Metric evaluation count differs from the configured valid-label count: "
                f"split={key[0]}, resource={key[1]}, expected={expected}, actual={actual}"
            )


def _verify_model_artifacts(
    run_dir: Path,
    model: str,
    manifest: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_data_sha256: str,
) -> None:
    if model == "xgboost":
        features = _read_json(run_dir / "feature_names.json", list)
        if not features or any(not isinstance(name, str) for name in features):
            raise ValueError("feature_names.json must contain a non-empty string list")
        return

    for name in (
        "architecture.json",
        "token_vocabulary.json",
        "user_vocabulary.json",
    ):
        value = _read_json(run_dir / name, dict)
        if not value:
            raise ValueError(f"Neural baseline JSON artifact is empty: {run_dir / name}")
    details = manifest.get("details")
    identity = details.get("checkpoint_identity") if isinstance(details, dict) else None
    if not isinstance(identity, dict):
        raise ValueError("Neural baseline manifest has no checkpoint identity")
    numeric_features = identity.get("numeric_features")
    if not isinstance(numeric_features, list):
        raise ValueError("Neural baseline checkpoint identity has no numeric_features list")
    if _integer(identity.get("seed"), "checkpoint identity seed") != expected_seed:
        raise ValueError("Neural baseline checkpoint identity has the wrong seed")
    if _normalized_sha256(
        identity.get("data_sha256"), "checkpoint identity data_sha256"
    ) != expected_data_sha256:
        raise ValueError("Neural baseline checkpoint identity has the wrong data SHA-256")
    # common.neural.checkpoint_identity uses a JSON-list serialization for its
    # split hash, whereas common.data uses the protocol's tab/newline encoding.
    # Both are provenance fields, but they are intentionally not comparable.
    _normalized_sha256(
        identity.get("split_fingerprint"),
        "checkpoint identity split_fingerprint",
    )
    if numeric_features:
        _require_nonempty_files(run_dir, ("numeric_scaler.json",))
        scaler = _read_json(run_dir / "numeric_scaler.json", dict)
        if scaler.get("columns") != numeric_features:
            raise ValueError("numeric_scaler.json does not match checkpoint numeric features")

    history_path = run_dir / "training_history.csv"
    history = pd.read_csv(history_path)
    required_history = {"epoch", "train_loss", "validation_mean_mae"}
    if history.empty or not required_history.issubset(history.columns):
        raise ValueError(f"Training history is incomplete: {history_path}")
    numeric = history[list(required_history)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"Training history contains invalid numeric values: {history_path}")


def _sealed_artifact_names(run_dir: Path, model: str) -> tuple[str, ...]:
    names = [*COMMON_ARTIFACTS, *MODEL_ARTIFACTS[model]]
    scaler = "numeric_scaler.json"
    if model in {"textcnn", "lstm"} and (run_dir / scaler).is_file():
        names.append(scaler)
    return tuple(names)


def _verify_completion_seal(
    run_dir: Path,
    *,
    model: str,
    seed: int,
    normalized_config_sha256: str,
) -> None:
    path = run_dir / COMPLETION_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline run has no completion seal: {path}; rerun and seal this baseline"
        )
    seal = _read_json(path, dict)
    if seal.get("schema") != COMPLETION_SCHEMA:
        raise ValueError("Unknown conventional-baseline completion-seal schema")
    if seal.get("model") != model:
        raise ValueError("Baseline completion seal belongs to a different model")
    if _integer(seal.get("seed"), "completion seal seed") != seed:
        raise ValueError("Baseline completion seal belongs to a different seed")
    sealed_config = _normalized_sha256(
        seal.get("normalized_config_sha256"),
        "completion seal normalized_config_sha256",
    )
    if sealed_config != normalized_config_sha256:
        raise ValueError("Baseline completion seal has a configuration hash mismatch")

    expected_names = _sealed_artifact_names(run_dir, model)
    files = seal.get("files_sha256")
    if not isinstance(files, dict) or set(files) != set(expected_names):
        raise ValueError("Baseline completion seal does not bind the required artifact set")
    for name in expected_names:
        expected_hash = _normalized_sha256(
            files.get(name), f"completion seal files_sha256.{name}"
        )
        actual_hash = sha256_file(run_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(f"Baseline artifact hash mismatch: {name}")
    recorded_set_hash = _normalized_sha256(
        seal.get("artifact_set_sha256"),
        "completion seal artifact_set_sha256",
    )
    if recorded_set_hash != _canonical_sha256(files):
        raise ValueError("Baseline completion seal artifact-set hash is inconsistent")


def verify_baseline_run(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    expected_model: str,
    expected_seed: int,
    _require_completion_seal: bool = True,
) -> dict[str, Any]:
    """Validate one baseline run without modifying it or loading model binaries."""
    run_dir = Path(run_dir).resolve()
    if expected_model not in BASELINE_MODELS:
        raise ValueError(f"Unsupported baseline model: {expected_model}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Baseline run directory does not exist: {run_dir}")

    artifacts = (*COMMON_ARTIFACTS, *MODEL_ARTIFACTS[expected_model])
    _require_nonempty_files(run_dir, artifacts)
    contract = _expected_contract(config)
    manifest = _read_json(run_dir / "run_manifest.json", dict)

    if manifest.get("model") != expected_model:
        raise ValueError(
            f"Manifest model mismatch: expected={expected_model}, actual={manifest.get('model')}"
        )
    manifest_seed = _integer(manifest.get("seed"), "run manifest seed")
    if manifest_seed != expected_seed:
        raise ValueError(
            f"Manifest seed mismatch: expected={expected_seed}, actual={manifest_seed}"
        )
    manifest_data_sha256 = _normalized_sha256(
        manifest.get("data_sha256"), "run manifest data_sha256"
    )
    if manifest_data_sha256 != contract["data_sha256"]:
        raise ValueError("Run manifest data SHA-256 does not match the experiment contract")
    manifest_split = _normalized_sha256(
        _split_fingerprint(manifest, run_dir / "run_manifest.json"),
        "run manifest split_fingerprint",
    )
    if manifest_split != contract["split_fingerprint"]:
        raise ValueError("Run manifest split fingerprint does not match the experiment contract")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Run manifest protocol must be a JSON object")
    if _normalized_sha256(
        protocol.get("data_sha256"), "run manifest protocol.data_sha256"
    ) != contract["data_sha256"]:
        raise ValueError("Run protocol data SHA-256 disagrees with the experiment contract")
    if _normalized_sha256(
        protocol.get("split_fingerprint"),
        "run manifest protocol.split_fingerprint",
    ) != contract["split_fingerprint"]:
        raise ValueError("Run protocol split fingerprint disagrees with the experiment contract")

    normalized_config_sha256 = _verify_resolved_config(
        run_dir / "config.resolved.yaml",
        expected_config=config,
        expected_seed=expected_seed,
        expected_data_sha256=contract["data_sha256"],
        expected_split_fingerprint=contract["split_fingerprint"],
    )

    metrics_path = run_dir / "metrics.csv"
    metrics = _validate_metrics(pd.read_csv(metrics_path), metrics_path)
    _verify_metric_counts(metrics, contract["valid_counts"])
    _recompute_and_compare(
        metrics,
        metrics_path,
        tolerance=contract["tolerance"],
        manifest=manifest,
    )
    predictions = pd.read_csv(
        run_dir / "predictions.csv", dtype={"sample_id": str, "job_id": str}
    )
    _verify_prediction_coverage(predictions, contract["evaluation_counts"])
    _verify_model_artifacts(
        run_dir,
        expected_model,
        manifest,
        expected_seed=expected_seed,
        expected_data_sha256=contract["data_sha256"],
    )
    if _require_completion_seal:
        _verify_completion_seal(
            run_dir,
            model=expected_model,
            seed=expected_seed,
            normalized_config_sha256=normalized_config_sha256,
        )

    return {
        "run_dir": str(run_dir),
        "model": expected_model,
        "seed": expected_seed,
        "data_sha256": contract["data_sha256"],
        "split_fingerprint": contract["split_fingerprint"],
        "normalized_config_sha256": normalized_config_sha256,
    }


def seal_baseline_run(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    expected_model: str,
    expected_seed: int,
) -> dict[str, Any]:
    """Validate a completed run, atomically seal it, then verify the seal."""
    result = verify_baseline_run(
        run_dir,
        config=config,
        expected_model=expected_model,
        expected_seed=expected_seed,
        _require_completion_seal=False,
    )
    resolved_run_dir = Path(result["run_dir"])
    names = _sealed_artifact_names(resolved_run_dir, expected_model)
    files = {name: sha256_file(resolved_run_dir / name) for name in names}
    payload = {
        "schema": COMPLETION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": expected_model,
        "seed": expected_seed,
        "normalized_config_sha256": result["normalized_config_sha256"],
        "files_sha256": files,
        "artifact_set_sha256": _canonical_sha256(files),
    }
    _atomic_write_json(resolved_run_dir / COMPLETION_FILE, payload)
    return verify_baseline_run(
        resolved_run_dir,
        config=config,
        expected_model=expected_model,
        expected_seed=expected_seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, args.overrides)
        operation = seal_baseline_run if args.write_seal else verify_baseline_run
        result = operation(
            Path(args.run_dir),
            config=config,
            expected_model=args.expected_model,
            expected_seed=args.expected_seed,
        )
    except Exception as error:
        print(f"[INVALID] {error}", file=sys.stderr)
        return 1
    action = "sealed and verified" if args.write_seal else "verified"
    print(f"[OK] {action} {result['model']} seed {result['seed']}: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
