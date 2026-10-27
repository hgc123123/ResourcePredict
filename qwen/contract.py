from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from common.config import get_value, set_value
from common.protocols import (
    BOUNDARY_COMPLETION_PROTOCOL_VERSION,
    normalize_boundary_completion_policy,
)
from qwen.base_selection import base_selection_settings, LEGACY_MODE


CONTRACT_VERSION = 4
RUNTIME_ONLY_QWEN_KEYS = {"eval_batch_size", "num_workers"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Experiment-contract value is not JSON serializable: {type(value)!r}")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def contract_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _metadata_semantics(config: Dict[str, Any]) -> Dict[str, Any]:
    feature_specs = copy.deepcopy(get_value(config, "data.metadata_features", []) or [])
    if feature_specs:
        active_features: Dict[str, Any] = {
            "mode": "engineered",
            "features": feature_specs,
        }
    else:
        active_features = {
            "mode": "numeric_columns",
            "columns": list(get_value(config, "data.metadata_numeric", []) or []),
        }
    return {
        **active_features,
        "categorical": list(get_value(config, "data.metadata_categorical", []) or []),
        "standardization": str(
            get_value(config, "data.metadata_standardization", "zscore")
        ),
        "require_complete": bool(
            get_value(config, "data.require_complete_metadata", False)
        ),
    }


def _normalized_lora(config: Dict[str, Any]) -> Dict[str, Any]:
    configured = copy.deepcopy(get_value(config, "qwen.lora", {}) or {})
    if not isinstance(configured, dict):
        raise ValueError("qwen.lora must be a mapping")
    configured.setdefault("enabled", False)
    configured.setdefault("r", 16)
    configured.setdefault("alpha", 32)
    configured.setdefault("dropout", 0.10)
    configured.setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    configured["enabled"] = bool(configured["enabled"])
    configured["r"] = int(configured["r"])
    configured["alpha"] = int(configured["alpha"])
    configured["dropout"] = float(configured["dropout"])
    configured["target_modules"] = [str(item) for item in configured["target_modules"]]
    return configured


def _qwen_semantics(
    config: Dict[str, Any],
    effective_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    encoder_learning_rate = float(get_value(config, "qwen.learning_rate", 1e-4))
    attention_implementation = effective_parameters.get(
        "attention_implementation",
        get_value(config, "qwen.attention_implementation"),
    )
    semantics = {
        "variant": str(get_value(config, "qwen.variant", "full")),
        "enabled_modalities": list(
            get_value(config, "qwen.enabled_modalities", ["code", "user", "metadata"])
        ),
        "fusion_mode": str(get_value(config, "qwen.fusion_mode", "gated")),
        "model_name_or_path": str(get_value(config, "qwen.model_name_or_path", "")),
        "max_tokens": int(get_value(config, "qwen.max_tokens", 2048)),
        "head_fraction": float(get_value(config, "qwen.head_fraction", 0.60)),
        "prompt_prefix": str(get_value(config, "qwen.prompt_prefix", "")),
        "prompt_suffix": str(get_value(config, "qwen.prompt_suffix", "")),
        "fusion_dim": int(get_value(config, "qwen.fusion_dim", 256)),
        "user_embedding_dim": int(get_value(config, "qwen.user_embedding_dim", 64)),
        "head_hidden_dim": int(get_value(config, "qwen.head_hidden_dim", 128)),
        "dropout": float(get_value(config, "qwen.dropout", 0.30)),
        "learning_rate": encoder_learning_rate,
        "head_learning_rate": float(
            get_value(config, "qwen.head_learning_rate", encoder_learning_rate)
        ),
        "weight_decay": float(get_value(config, "qwen.weight_decay", 0.05)),
        "batch_size": int(get_value(config, "qwen.batch_size", 16)),
        "gradient_accumulation_steps": int(
            get_value(config, "qwen.gradient_accumulation_steps", 1)
        ),
        "epochs": int(get_value(config, "qwen.epochs", 20)),
        "patience": int(get_value(config, "qwen.patience", 5)),
        "precision": str(get_value(config, "qwen.precision", "bf16")),
        "quantization": str(get_value(config, "qwen.quantization", "none")),
        "gradient_checkpointing": bool(
            get_value(config, "qwen.gradient_checkpointing", True)
        ),
        "attention_implementation": (
            None if attention_implementation is None else str(attention_implementation)
        ),
        "trust_remote_code": bool(get_value(config, "qwen.trust_remote_code", True)),
        "user_dropout": float(
            get_value(config, "qwen.user_dropout", get_value(config, "training.user_dropout", 0.20))
        ),
        "smooth_l1_beta": float(get_value(config, "qwen.smooth_l1_beta", 0.10)),
        "vram_loss_weight": float(
            get_value(
                config,
                "qwen.vram_loss_weight",
                get_value(config, "training.vram_loss_weight", 2.0),
            )
        ),
        "gradient_clip": float(get_value(config, "qwen.gradient_clip", 1.0)),
        "early_stopping_min_delta": float(
            get_value(config, "qwen.early_stopping_min_delta", 0.0)
        ),
        "lora": _normalized_lora(config),
        "pooling": "last_non_padding_token",
        "metadata_residual_conditioning": True,
    }
    selection = base_selection_settings(config)
    # Absence retains the byte-identical version-4 legacy payload.
    if selection["mode"] != LEGACY_MODE:
        semantics["base_selection"] = selection
    return semantics


def experiment_contract(
    config: Dict[str, Any],
    effective_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "seed": int(get_value(config, "seed", 42)),
        "data_semantics": {
            "label_scale": str(get_value(config, "data.label_scale", "ratio")),
            "label_out_of_range": str(
                get_value(config, "data.label_out_of_range", "error")
            ),
            "csv_encoding": str(get_value(config, "data.csv_encoding", "utf-8-sig")),
            "csv_field_size_limit_bytes": int(
                get_value(config, "data.csv_field_size_limit_bytes", 64 * 1024 * 1024)
            ),
            "expected_columns": copy.deepcopy(
                get_value(config, "data.expected_columns", None)
            ),
            "columns": copy.deepcopy(get_value(config, "data.columns", {}) or {}),
            "filters": {
                "version": 1,
                "settings": copy.deepcopy(get_value(config, "data.filters", {}) or {}),
                "state_normalization": "strip_upper",
                "exit_code_normalization": "strip_status_colon_signal",
                "application_order": ["state", "exit_code"],
            },
            "code_redaction": {
                "version": 1,
                "settings": copy.deepcopy(
                    get_value(config, "data.code_redaction", {}) or {}
                ),
                "replacement": "<USER>",
                "scope": "immediate_segment_after_each_path_prefix",
            },
            "forbidden_metadata_columns": copy.deepcopy(
                get_value(config, "data.forbidden_metadata_columns", []) or []
            ),
            "timestamp_timezone": str(
                get_value(config, "data.timestamp_timezone", "UTC")
            ),
            "time_validation": copy.deepcopy(
                get_value(config, "data.time_validation", {}) or {}
            ),
            "require_unique_job_ids": bool(
                get_value(config, "data.require_unique_job_ids", False)
            ),
            "metadata": _metadata_semantics(config),
        },
        "split_semantics": {
            "train_ratio": float(get_value(config, "split.train_ratio", 0.8)),
            "validation_ratio": float(get_value(config, "split.validation_ratio", 0.1)),
            "order_type": str(get_value(config, "split.order_type", "datetime")),
            "order_semantics": str(
                get_value(config, "split.order_semantics", "unspecified")
            ),
            "expected_counts": copy.deepcopy(
                get_value(config, "split.expected_counts", None)
            ),
            "require_completed_labels_at_boundaries": bool(
                get_value(
                    config,
                    "split.require_completed_labels_at_boundaries",
                    False,
                )
            ),
            "boundary_completion_policy": normalize_boundary_completion_policy(
                get_value(config, "split.boundary_completion_policy", "error")
            ),
            "boundary_protocol_version": BOUNDARY_COMPLETION_PROTOCOL_VERSION,
            "create_if_missing": bool(
                get_value(config, "split.create_if_missing", False)
            ),
        },
        "evaluation_semantics": {
            "clip_predictions": bool(
                get_value(config, "evaluation.clip_predictions", True)
            ),
            "acc_tolerance": float(get_value(config, "evaluation.acc_tolerance", 0.10)),
        },
        "qwen_semantics": _qwen_semantics(config, effective_parameters),
        "effective_qwen_parameters": copy.deepcopy(dict(effective_parameters)),
    }
    normalized = _jsonable(payload)
    return {
        "version": CONTRACT_VERSION,
        "sha256": contract_sha256(normalized),
        "payload": normalized,
    }


def _difference_paths(expected: Any, observed: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(observed, dict):
        paths: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in observed:
                paths.append(child)
            else:
                paths.extend(_difference_paths(expected[key], observed[key], child))
        return paths
    if isinstance(expected, list) and isinstance(observed, list):
        if expected == observed:
            return []
        return [prefix]
    return [] if expected == observed else [prefix]


def _validated_saved_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    saved = manifest.get("experiment_contract")
    if not isinstance(saved, dict):
        raise ValueError(
            "Checkpoint has no experiment_contract; retrain it with the current Qwen code"
        )
    if int(saved.get("version", -1)) != CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported experiment contract version: {saved.get('version')!r}"
        )
    payload = saved.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint experiment contract has no valid payload")
    expected_sha = contract_sha256(payload)
    if saved.get("sha256") != expected_sha:
        raise ValueError("Checkpoint experiment contract fingerprint is corrupt or modified")
    if _jsonable(manifest.get("effective_parameters", {})) != payload.get(
        "effective_qwen_parameters"
    ):
        raise ValueError("Checkpoint effective parameters disagree with its experiment contract")
    qwen = payload.get("qwen_semantics", {})
    manifest_checks = {
        "seed": (manifest.get("seed"), payload.get("seed")),
        "variant": (manifest.get("variant"), qwen.get("variant")),
        "enabled_modalities": (
            manifest.get("enabled_modalities"),
            qwen.get("enabled_modalities"),
        ),
        "fusion_mode": (manifest.get("fusion_mode", "gated"), qwen.get("fusion_mode")),
    }
    inconsistent = [
        name for name, (observed, expected) in manifest_checks.items() if observed != expected
    ]
    if inconsistent:
        raise ValueError(
            "Checkpoint manifest fields disagree with its experiment contract: "
            + ", ".join(inconsistent)
        )
    return copy.deepcopy(saved)


def _restore_checkpoint_switches(
    config: Dict[str, Any],
    saved_contract: Mapping[str, Any],
) -> None:
    payload = saved_contract["payload"]
    qwen = payload["qwen_semantics"]
    config["seed"] = int(payload["seed"])
    set_value(config, "qwen.variant", qwen["variant"])
    set_value(config, "qwen.enabled_modalities", copy.deepcopy(qwen["enabled_modalities"]))
    set_value(config, "qwen.fusion_mode", qwen["fusion_mode"])
    current_lora = copy.deepcopy(get_value(config, "qwen.lora", {}) or {})
    if not isinstance(current_lora, dict):
        raise ValueError("qwen.lora must be a mapping")
    current_lora["enabled"] = bool(qwen["lora"]["enabled"])
    set_value(config, "qwen.lora", current_lora)


def restore_effective_config(
    config: Dict[str, Any],
    saved_contract: Mapping[str, Any],
) -> None:
    payload = saved_contract["payload"]
    config["seed"] = int(payload["seed"])
    for key, value in payload["qwen_semantics"].items():
        if key in {"pooling", "metadata_residual_conditioning"}:
            continue
        if key in RUNTIME_ONLY_QWEN_KEYS:
            continue
        set_value(config, f"qwen.{key}", copy.deepcopy(value))


def validate_and_restore_contract(
    config: Dict[str, Any],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    saved = _validated_saved_contract(manifest)
    _restore_checkpoint_switches(config, saved)
    observed = experiment_contract(config, manifest["effective_parameters"])
    if observed["sha256"] != saved["sha256"]:
        differences = _difference_paths(saved["payload"], observed["payload"])
        rendered = ", ".join(differences[:20]) or "unknown canonical payload change"
        if len(differences) > 20:
            rendered += f", ... (+{len(differences) - 20} more)"
        raise ValueError(
            "Evaluation configuration does not match the checkpoint experiment contract; "
            f"changed fields: {rendered}"
        )
    restore_effective_config(config, saved)
    return saved


def validate_base_model_override(
    saved_contract: Mapping[str, Any], override: str | None
) -> str | None:
    """Allow a LoRA base-model override only when it names the trained base model."""
    if override is None:
        return None
    payload = saved_contract.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Saved experiment contract has no payload")
    qwen = payload.get("qwen_semantics")
    if not isinstance(qwen, Mapping):
        raise ValueError("Saved experiment contract has no Qwen semantics")
    expected = str(qwen.get("model_name_or_path", ""))
    observed = str(override)
    if observed != expected:
        raise ValueError(
            "--base-model differs from the checkpoint experiment contract; loading "
            "different base weights would invalidate prediction/benchmark provenance "
            f"(expected {expected!r}, got {observed!r})"
        )
    return observed
