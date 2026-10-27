#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.artifacts import write_evaluation  # noqa: E402
from common.config import get_value, load_config, resolve_path  # noqa: E402
from common.data import sha256_file  # noqa: E402
from reliability_cascade.evaluation import apply_selected_cascade  # noqa: E402
from reliability_cascade.integrity import (  # noqa: E402
    CHECKPOINT_BOUND_FILES,
    CHECKPOINT_SCHEMA,
    REQUIRED_RUN_FILES,
    build_run_request,
    read_json,
    validate_variant,
    verify_run_directory,
    write_artifact_integrity,
    write_completion_record,
)
from reliability_cascade.pipeline import (  # noqa: E402
    PROTOCOL_VERSION,
    cascade_model_config,
    chronological_calibration_masks,
    prepare_cascade,
    resolve_component_paths,
)
from reliability_cascade.selection import select_cascade  # noqa: E402


EXPECTED_OUTPUTS = REQUIRED_RUN_FILES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the history-first reliability-gated cascade"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=True, allow_nan=False)


def _output_dir(config: dict[str, Any], value: str | None) -> Path:
    seed = int(get_value(config, "seed", 42))
    variant = validate_variant(
        get_value(config, "reliability_cascade.qwen_variant", "full")
    )
    max_tokens = int(get_value(config, "qwen.max_tokens", 4096))
    if max_tokens <= 0:
        raise ValueError("qwen.max_tokens must be positive")
    return resolve_path(
        config,
        value
        or get_value(
            config,
            "reliability_cascade.output_dir",
            f"outputs/reliability_cascade/tokens_{max_tokens}/{variant}/seed_{seed}",
        ),
    )


def _assert_writable_destination(path: Path, overwrite: bool) -> None:
    if path.is_symlink():
        raise FileExistsError(f"Refusing to publish through a symbolic link: {path}")
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Cascade output path is not a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Cascade output already exists: {path}; use a new directory or pass "
            "--overwrite"
        )


def _publish_directory(staging: Path, destination: Path, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Cascade output already exists: {destination}")
        backup = destination.with_name(
            f".{destination.name}.backup-{uuid4().hex}"
        )
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            print(
                f"WARNING: new run was published, but old backup remains at "
                f"{backup}: {exc}",
                file=sys.stderr,
            )


def _recover_interrupted_publish(destination: Path) -> None:
    backups = sorted(destination.parent.glob(f".{destination.name}.backup-*"))
    if destination.exists():
        if backups:
            print(
                "WARNING: verified destination exists, but an older publish backup "
                f"also remains: {backups}",
                file=sys.stderr,
            )
        return
    if len(backups) > 1:
        raise RuntimeError(
            f"Multiple interrupted-publish backups require manual review: {backups}"
        )
    if backups:
        backups[0].rename(destination)
        print(
            f"Recovered interrupted publish backup: {backups[0]} -> {destination}",
            file=sys.stderr,
        )


def _acquire_output_lock(destination: Path) -> Any:
    lock_path = destination.parent / f".{destination.name}.lock"
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        handle.close()
        raise RuntimeError(
            f"Another process is publishing the same cascade output: {destination}"
        ) from exc
    return handle


def _release_output_lock(handle: Any) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.overrides)
    output_dir = _output_dir(config, args.output_dir)
    qwen_path, last1_path = resolve_component_paths(
        config, args.qwen_predictions, args.last1_predictions
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "config": str(Path(args.config).expanduser().resolve()),
                    "qwen_predictions": str(qwen_path),
                    "last1_predictions": str(last1_path),
                    "output_dir": str(output_dir),
                    "writes_files": False,
                    "test_labels_used_for_selection": False,
                    "route_decision_uses_qwen_output": False,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = _acquire_output_lock(output_dir)
    try:
        _recover_interrupted_publish(output_dir)
        _assert_writable_destination(output_dir, bool(args.overwrite))
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
            )
        )
    except BaseException:
        _release_output_lock(lock_handle)
        raise
    try:
        run_request = build_run_request(
            config,
            qwen_path,
            last1_path,
            protocol=PROTOCOL_VERSION,
        )
        _write_json(staging_dir / "run_request.json", run_request)
        checkpoint_dir = staging_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        prepared = prepare_cascade(config, str(qwen_path), str(last1_path))
        fit_mask, selection_mask, calibration_protocol = chronological_calibration_masks(
            prepared, config
        )
        seed = int(get_value(config, "seed", 42))
        model_config = cascade_model_config(config, seed=seed)
        selected = select_cascade(
            prepared, config, fit_mask, selection_mask, model_config
        )
        variant = str(get_value(config, "reliability_cascade.qwen_variant", "full"))
        model_name = f"reliability_cascade_{variant}"
        (
            predictions,
            metrics,
            reference_metrics,
            routing_metrics,
            importance,
            route_sha256,
        ) = apply_selected_cascade(
            prepared,
            selected,
            model_name=model_name,
            seed=seed,
        )

        selected.sweep.to_csv(staging_dir / "selection_sweep.csv", index=False)
        _write_json(staging_dir / "selected_policy.json", selected.policy)
        reference_metrics.to_csv(staging_dir / "reference_metrics.csv", index=False)
        routing_metrics.to_csv(staging_dir / "routing_metrics.csv", index=False)
        importance.to_csv(staging_dir / "feature_importance.csv", index=False)
        model_path = checkpoint_dir / "cascade_models.joblib"
        joblib.dump(selected.models, model_path, compress=3)

        component_contract = prepared.components.checkpoint_contract()
        qwen_details = prepared.components.qwen_manifest.get("details") or {}
        details = {
            "protocol": PROTOCOL_VERSION,
            "variant": variant,
            "qwen_backbone": prepared.qwen_backbone,
            "qwen_predictions_path": str(prepared.components.qwen_path),
            "last1_predictions_path": str(prepared.components.last1_path),
            "qwen_effective_parameters": qwen_details.get("effective_parameters"),
            "effective_parameters": qwen_details.get("effective_parameters"),
            "component_contract": component_contract,
            "calibration_protocol": calibration_protocol,
            "history_contract": prepared.features.history_contract,
            "selection_policy": selected.policy,
            "route_feature_whitelist": selected.models["memory"].pre_names,
            "route_decision_uses_qwen_output": False,
            "test_labels_used_for_fit_or_selection": False,
            "offline_counterfactual_qwen_predictions": True,
            "routing_rates_are_simulated": True,
            "lazy_serving_latency_measured": False,
            "route_mask_sha256": route_sha256,
            "models_sha256": sha256_file(model_path),
            "run_request_sha256": run_request["run_request_sha256"],
            "routing_summary": routing_metrics.to_dict(orient="records"),
            "validation_metrics_are_descriptive": True,
        }
        saved_metrics = write_evaluation(
            config,
            prepared.bundle,
            model_name,
            predictions,
            staging_dir,
            details=details,
            run_seed=seed,
        )
        if not np.allclose(
            saved_metrics[["mae", "mse", "rmse", "acc_at_10"]].to_numpy(float),
            metrics[["mae", "mse", "rmse", "acc_at_10"]].to_numpy(float),
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError(
                "Saved evaluation metrics differ from pre-write recomputation"
            )

        checkpoint_contract = {
            "schema": CHECKPOINT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL_VERSION,
            "run_request_sha256": run_request["run_request_sha256"],
            "qwen_backbone": prepared.qwen_backbone,
            "component_contract": component_contract,
            "calibration_protocol": calibration_protocol,
            "history_contract": prepared.features.history_contract,
            "selection_contract": selected.policy,
            "cascade_model_config": asdict(model_config),
            "route_mask_sha256": route_sha256,
            "route_decision_uses_qwen_output": False,
            "models_sha256": sha256_file(model_path),
            "artifact_sha256": {
                name: sha256_file(staging_dir / name)
                for name in CHECKPOINT_BOUND_FILES
            },
        }
        contract_path = checkpoint_dir / "checkpoint_contract.json"
        _write_json(contract_path, checkpoint_contract)
        write_artifact_integrity(
            staging_dir,
            run_request_sha256=run_request["run_request_sha256"],
        )

        manifest_path = staging_dir / "run_manifest.json"
        manifest = read_json(manifest_path)
        manifest_details = manifest.setdefault("details", {})
        manifest_details["checkpoint_contract_sha256"] = sha256_file(contract_path)
        manifest_details["artifact_integrity_sha256"] = sha256_file(
            staging_dir / "artifact_integrity.json"
        )
        _write_json(manifest_path, manifest)
        write_completion_record(
            staging_dir,
            run_request_sha256=run_request["run_request_sha256"],
        )
        verify_run_directory(
            staging_dir,
            expected_run_request_sha256=run_request["run_request_sha256"],
            expected_seed=seed,
            expected_model=model_name,
        )
        _publish_directory(staging_dir, output_dir, bool(args.overwrite))
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        _release_output_lock(lock_handle)
        raise

    _release_output_lock(lock_handle)

    print(saved_metrics.to_string(index=False))
    print(routing_metrics.to_string(index=False))
    print(f"Predictions: {output_dir / 'predictions.csv'}")
    print(f"Selected policy: {output_dir / 'selected_policy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
