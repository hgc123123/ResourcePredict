#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.artifacts import write_evaluation  # noqa: E402
from common.config import get_value, load_config, resolve_path  # noqa: E402
from common.data import sha256_file  # noqa: E402
from common.history import causal_history_contract  # noqa: E402
from common.metrics import add_error_columns  # noqa: E402
from dynamic_hybrid.features import RobustFeatureScaler  # noqa: E402
from dynamic_hybrid.model import RESOURCES, DynamicHybridModel  # noqa: E402
from dynamic_hybrid.pipeline import gate_fit_and_selection_masks, prepare_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen causal dynamic hybrid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _device(config: Dict[str, Any]) -> torch.device:
    value = str(get_value(config, "dynamic_hybrid.device", "auto"))
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    checkpoint = resolve_path(config, args.checkpoint)
    training_manifest = _load_json(checkpoint / "training_manifest.json")
    model_config = _load_json(checkpoint / "model_config.json")
    if model_config != training_manifest.get("model_config"):
        raise ValueError("model_config.json differs from the signed training manifest")
    expected_checkpoint_hashes = training_manifest.get("checkpoint_files_sha256")
    if not isinstance(expected_checkpoint_hashes, dict):
        raise ValueError("Training manifest has no checkpoint file-integrity record")
    for name in (
        "dynamic_hybrid.pt",
        "model_config.json",
        "scaler_memory.json",
        "scaler_vram.json",
    ):
        expected_hash = expected_checkpoint_hashes.get(name)
        path = checkpoint / name
        if not expected_hash or not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Dynamic-hybrid checkpoint integrity check failed: {name}")
    prepared = prepare_inputs(config, args.qwen_predictions, args.last1_predictions)
    component_seed = int(prepared.components.seed)
    gate_seed = int(get_value(config, "dynamic_hybrid.seed", component_seed))
    if int(training_manifest.get("gate_seed", training_manifest.get("seed", -1))) != gate_seed:
        raise ValueError("Configured dynamic-hybrid gate seed differs from checkpoint")
    if int(training_manifest.get("component_seed", component_seed)) != component_seed:
        raise ValueError("Checkpoint component seed differs from Qwen/Last-1 artifacts")
    qwen_variant = str(
        (prepared.components.qwen_manifest.get("details") or {}).get("variant")
    )
    model_name = f"dynamic_hybrid_{qwen_variant}"
    if str(training_manifest.get("model")) != model_name:
        raise ValueError("Checkpoint model identity differs from the Qwen variant")
    expected_contract = training_manifest.get("component_contract")
    actual_contract = prepared.components.checkpoint_contract()
    if expected_contract != actual_contract:
        raise ValueError(
            "Evaluation components differ from the exact dataset/split/prediction files "
            "used to train the dynamic gate"
        )
    if bool(training_manifest.get("test_labels_used_for_fit_or_selection", True)):
        raise ValueError("Checkpoint does not attest that test labels were excluded from selection")
    _, _, current_fit_protocol = gate_fit_and_selection_masks(prepared, config)
    if training_manifest.get("fit_protocol") != current_fit_protocol:
        raise ValueError("Configured gate fit/selection protocol differs from checkpoint")
    expected_refit = bool(
        get_value(config, "dynamic_hybrid.refit_on_full_development", True)
    )
    final_fit_protocol = training_manifest.get("final_fit_protocol") or {}
    if bool(final_fit_protocol.get("enabled", False)) != expected_refit:
        raise ValueError("Configured final-refit policy differs from checkpoint")
    current_history_protocol = causal_history_contract(
        int(get_value(config, "dynamic_hybrid.history_k", 5)),
        float(get_value(config, "dynamic_hybrid.history_ema_alpha", 0.5)),
    )
    if training_manifest.get("history_protocol") != current_history_protocol:
        raise ValueError("Configured causal-history feature protocol differs from checkpoint")
    if model_config.get("feature_names") != prepared.features.names:
        raise ValueError("Checkpoint feature whitelist differs from current features")
    if not bool(model_config.get("strict_qwen_no_history", False)):
        raise ValueError("Checkpoint does not enforce the Qwen-only cold-start fallback")

    model = DynamicHybridModel(
        int(model_config["input_dim"]),
        hidden_dims=tuple(int(value) for value in model_config["hidden_dims"]),
        dropout=float(model_config["dropout"]),
        residual_max=float(model_config["residual_max"]),
        initial_alphas={
            key: float(value) for key, value in model_config["initialized_alphas"].items()
        },
        input_normalization=str(model_config.get("input_normalization", "layernorm")),
    )
    weights_path = checkpoint / "dynamic_hybrid.pt"
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    device = _device(config)
    model.to(device).eval()

    frame = prepared.features.paired.copy()
    count = len(frame)
    raw_prediction = np.full(count, np.nan, dtype=np.float64)
    prediction = np.full(count, np.nan, dtype=np.float64)
    gate = np.full(count, np.nan, dtype=np.float64)
    residual = np.full(count, np.nan, dtype=np.float64)
    batch_size = int(get_value(config, "dynamic_hybrid.eval_batch_size", 4096))
    if batch_size <= 0:
        raise ValueError("dynamic_hybrid.eval_batch_size must be positive")

    with torch.inference_mode():
        for resource in RESOURCES:
            indices = np.flatnonzero(frame["resource"].astype(str).to_numpy() == resource)
            scaler = RobustFeatureScaler.load(checkpoint / f"scaler_{resource}.json")
            scaled = scaler.transform(prepared.features.values, prepared.features.names)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                qwen = torch.as_tensor(
                    frame.iloc[batch_indices]["qwen_prediction"].to_numpy(np.float32),
                    device=device,
                )
                available = torch.as_tensor(
                    frame.iloc[batch_indices]["history_available"].to_numpy(bool),
                    device=device,
                )
                last1_values = frame.iloc[batch_indices]["last1_prediction"].to_numpy(
                    np.float32
                )
                last1_values = np.where(available.cpu().numpy(), last1_values, qwen.cpu().numpy())
                output = model(
                    resource,
                    torch.as_tensor(scaled[batch_indices], device=device),
                    qwen,
                    torch.as_tensor(last1_values, device=device),
                    available,
                )
                raw_prediction[batch_indices] = output["raw_prediction"].cpu().numpy()
                prediction[batch_indices] = output["prediction"].cpu().numpy()
                gate[batch_indices] = output["gate"].cpu().numpy()
                residual[batch_indices] = output["residual"].cpu().numpy()

    if not np.isfinite(prediction).all():
        raise RuntimeError("Evaluation did not produce one finite prediction per row")
    no_history = ~frame["history_available"].to_numpy(bool)
    qwen_values = frame["qwen_prediction"].to_numpy(float)
    if not (
        np.allclose(prediction[no_history], qwen_values[no_history], rtol=0.0, atol=1e-6)
        and np.allclose(raw_prediction[no_history], qwen_values[no_history], rtol=0.0, atol=1e-6)
        and np.array_equal(gate[no_history], np.zeros(int(no_history.sum())))
        and np.array_equal(residual[no_history], np.zeros(int(no_history.sum())))
    ):
        raise AssertionError("No-history rows did not use the exact Qwen-only fallback")

    evaluation_mask = frame["split"].astype(str).isin(["validation", "test"]).to_numpy()
    frame = frame.loc[evaluation_mask].reset_index(drop=True)
    raw_prediction = raw_prediction[evaluation_mask]
    prediction = prediction[evaluation_mask]
    gate = gate[evaluation_mask]
    residual = residual[evaluation_mask]
    job_column = "job_id_qwen" if "job_id_qwen" in frame else "job_id_last1"
    user_column = "user_id_qwen" if "user_id_qwen" in frame else "user_id_last1"
    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "job_id": frame[job_column].astype(str),
            "user_id": frame[user_column].astype(str),
            "split": frame["split"].astype(str),
            "resource": frame["resource"].astype(str),
            "target": frame["target"].to_numpy(float),
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "valid": frame["valid"].to_numpy(bool),
            "model": model_name,
            "seed": gate_seed,
            "qwen_prediction": frame["qwen_prediction"].to_numpy(float),
            "last1_prediction": frame["last1_prediction"].to_numpy(float),
            "history_available": frame["history_available"].to_numpy(bool),
            "dynamic_gate": gate,
            "bounded_residual": residual,
            "used_qwen_only": ~frame["history_available"].to_numpy(bool),
        }
    )
    output = add_error_columns(output)
    output_dir = resolve_path(config, args.output_dir) if args.output_dir else checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_protocol = current_fit_protocol
    if bool(final_fit_protocol.get("validation_labels_used_for_refit", False)):
        validation_role = (
            "descriptive development-fit metric; validation labels were used only "
            "after checkpoint/hyperparameter selection for the frozen final refit"
        )
    else:
        validation_role = (
            "independent checkpoint-selection split"
            if "out-of-fold train" in str(fit_protocol.get("fit_source", ""))
            else "descriptive calibration metric; leading validation jobs fitted the gate"
        )
    metrics = write_evaluation(
        config,
        prepared.bundle,
        model_name,
        output,
        output_dir,
        details={
            "checkpoint": str(checkpoint),
            "checkpoint_weights_sha256": sha256_file(weights_path),
            "checkpoint_files_sha256": expected_checkpoint_hashes,
            "component_contract": actual_contract,
            "component_seed": component_seed,
            "gate_seed": gate_seed,
            "fit_protocol": fit_protocol,
            "final_fit_protocol": final_fit_protocol,
            "feature_whitelist": model_config["feature_names"],
            "strict_qwen_no_history": True,
            "selection_metric": training_manifest["selection_metric"],
            "selection_results_scope": training_manifest.get(
                "selection_results_scope"
            ),
            "resource_model_kind": training_manifest.get("resource_model_kind"),
            "selection_constraints": training_manifest.get("selection_constraints"),
            "best_selection_metrics": training_manifest.get("best_selection_metrics"),
            "test_labels_used_for_fit_or_selection": False,
            "evaluated_splits": ["validation", "test"],
            "metric_roles": {
                "validation": validation_role,
                "test": "held-out evaluation",
            },
        },
        run_seed=gate_seed,
    )
    print(metrics.to_string(index=False))
    print(f"Predictions: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
