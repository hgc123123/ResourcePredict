from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.artifacts import build_prediction_frame, write_evaluation
from common.config import get_value, load_config, resolve_path
from common.data import load_dataset
from qwen.base_selection import base_selection_settings, verify_base_training_selection, LEGACY_MODE
from qwen.contract import validate_and_restore_contract, validate_base_model_override
from qwen.dataset import (
    QwenBatchCollator,
    QwenJobDataset,
    QwenPreprocessors,
    ensure_tokenizer_padding,
    known_user_flags,
    script_sha256,
    split_fingerprint,
    validate_qwen_bundle,
)
from qwen.model import GatedQwenRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Qwen gated checkpoint on validation and test"
    )
    parser.add_argument("--config", required=True, help="The same shared YAML used for training")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory from train.py")
    parser.add_argument("--output-dir", help="Defaults to the checkpoint's parent run directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-model", help="Override the saved LoRA base-model path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; may be repeated",
    )
    return parser.parse_args()


def _resolve_device(request: str) -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if request == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dtype(config: Dict[str, Any], device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    precision = str(get_value(config, "qwen.precision", "bf16")).lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_training_manifest(checkpoint: Path) -> Dict[str, Any]:
    path = checkpoint / "training_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint training manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    manifest = _load_training_manifest(checkpoint)
    saved_contract = validate_and_restore_contract(config, manifest)
    base_model_override = validate_base_model_override(saved_contract, args.base_model)

    bundle = load_dataset(config)
    validate_qwen_bundle(bundle)
    if bundle.data_sha256 != manifest["data_sha256"]:
        raise ValueError("Evaluation CSV hash does not match the checkpoint training data")
    current_split_fingerprint = split_fingerprint(bundle.frame)
    if current_split_fingerprint != manifest["split_fingerprint"]:
        raise ValueError("Evaluation split assignments do not match the checkpoint")
    if list(bundle.numeric_features) != list(manifest["numeric_features"]):
        raise ValueError("Evaluation metadata feature order does not match training")
    base_training_selection = manifest.get("base_training_selection")
    if base_selection_settings(config)["mode"] != LEGACY_MODE:
        base_training_selection = verify_base_training_selection(
            bundle, config, base_training_selection
        )
        expected_selection_hash = manifest["effective_parameters"].get(
            "base_training_selection_sha256"
        )
        if base_training_selection["sha256"] != expected_selection_hash:
            raise ValueError("Base selection provenance disagrees with checkpoint contract")
    elif base_training_selection is not None:
        raise ValueError("Legacy checkpoint unexpectedly contains base selection provenance")

    preprocessors = QwenPreprocessors.load(checkpoint)
    if preprocessors.scaler.columns != bundle.numeric_features:
        raise ValueError("Saved numeric scaler columns do not match the configured metadata")

    tokenizer_path = checkpoint / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=bool(get_value(config, "qwen.trust_remote_code", True)),
        local_files_only=True,
    )
    ensure_tokenizer_padding(tokenizer)
    tokenizer.padding_side = "right"

    device = _resolve_device(args.device)
    dtype = _model_dtype(config, device)
    attention_implementation = get_value(config, "qwen.attention_implementation")
    if not attention_implementation:
        attention_implementation = "eager" if device.type == "cpu" else "sdpa"
    model = GatedQwenRegressor.from_checkpoint(
        checkpoint,
        torch_dtype=dtype,
        model_name_override=base_model_override,
        attention_implementation=str(attention_implementation),
    )
    if model.num_users != len(preprocessors.user_vocabulary):
        raise ValueError("Saved user vocabulary size does not match the model checkpoint")
    expected_qwen = saved_contract["payload"]["qwen_semantics"]
    architecture_checks = {
        "enabled_modalities": (
            list(model.enabled_modalities),
            expected_qwen["enabled_modalities"],
        ),
        "fusion_mode": (model.fusion_mode, expected_qwen["fusion_mode"]),
        "metadata_dim": (model.metadata_dim, len(bundle.numeric_features)),
        "fusion_dim": (model.fusion_dim, expected_qwen["fusion_dim"]),
        "user_embedding_dim": (
            model.user_embedding_dim,
            expected_qwen["user_embedding_dim"],
        ),
        "head_hidden_dim": (model.head_hidden_dim, expected_qwen["head_hidden_dim"]),
        "dropout": (model.dropout_probability, expected_qwen["dropout"]),
        "lora_enabled": (model.lora_enabled, expected_qwen["lora"]["enabled"]),
    }
    architecture_mismatches = [
        name
        for name, (observed, expected) in architecture_checks.items()
        if observed != expected
    ]
    if architecture_mismatches:
        raise ValueError(
            "Loaded model architecture disagrees with the checkpoint experiment contract: "
            + ", ".join(architecture_mismatches)
        )
    model.to(device)
    model.eval()

    effective = dict(manifest.get("effective_parameters") or {})
    max_tokens = int(effective.get("max_tokens", get_value(config, "qwen.max_tokens", 2048)))
    head_fraction = float(effective.get("head_fraction", 0.60))
    prompt_prefix = str(effective.get("prompt_prefix", ""))
    prompt_suffix = str(effective.get("prompt_suffix", ""))
    eval_batch_size = int(
        get_value(config, "qwen.eval_batch_size", get_value(config, "qwen.batch_size", 16))
    )
    num_workers = int(get_value(config, "qwen.num_workers", get_value(config, "training.num_workers", 0)))
    if max_tokens <= 0 or eval_batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid max_tokens, eval_batch_size, or num_workers")

    evaluation_frame = bundle.frame.loc[
        bundle.frame["split"].isin(("validation", "test"))
    ].reset_index(drop=True)
    dataset = QwenJobDataset(
        evaluation_frame,
        preprocessors,
        training=False,
        require_any_label=False,
    )
    collator = QwenBatchCollator(
        tokenizer,
        max_tokens=max_tokens,
        head_fraction=head_fraction,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )

    row_count = len(evaluation_frame)
    memory_predictions = np.full(row_count, np.nan, dtype=np.float64)
    vram_predictions = np.full(row_count, np.nan, dtype=np.float64)
    gates = np.full((row_count, 3), np.nan, dtype=np.float64)

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            user_ids = batch["user_ids"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            with _autocast_context(device, dtype):
                output = model(input_ids, attention_mask, user_ids, metadata)
            row_index = batch["row_index"].numpy()
            batch_predictions = output["predictions"].float().cpu().numpy()
            memory_predictions[row_index] = batch_predictions[:, 0]
            vram_predictions[row_index] = batch_predictions[:, 1]
            gates[row_index] = output["gate_weights"].float().cpu().numpy()

    if not np.isfinite(memory_predictions).all() or not np.isfinite(vram_predictions).all():
        raise RuntimeError("Evaluation did not produce one prediction for every row")

    known_users = known_user_flags(
        preprocessors.user_vocabulary, evaluation_frame["user_id"].astype(str)
    )
    code_hashes = np.asarray(
        [script_sha256(code) for code in evaluation_frame["code"]], dtype=object
    )
    model_name = f"qwen_{manifest['variant']}"
    prediction_frame = build_prediction_frame(
        evaluation_frame,
        {"memory": memory_predictions, "vram": vram_predictions},
        model_name=model_name,
        seed=int(manifest["seed"]),
        clip_predictions=bool(get_value(config, "evaluation.clip_predictions", True)),
        extra_columns={
            "gate_code": gates[:, 0],
            "gate_user": gates[:, 1],
            "gate_metadata": gates[:, 2],
            "user_known": known_users,
            "script_sha256": code_hashes,
        },
    )

    if args.output_dir:
        output_dir = resolve_path(config, args.output_dir)
    else:
        output_dir = checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = write_evaluation(
        config,
        bundle,
        model_name,
        prediction_frame,
        output_dir,
        details={
            "checkpoint": str(checkpoint),
            "variant": manifest["variant"],
            "enabled_modalities": manifest["enabled_modalities"],
            "fusion_mode": manifest.get("fusion_mode", "gated"),
            "selection_metric": manifest["selection_metric"],
            "base_training_selection": base_training_selection,
            "best_epoch": manifest["best_epoch"],
            "best_validation_mean_mae": manifest["best_validation_mean_mae"],
            "effective_parameters": manifest.get("effective_parameters"),
            "training_runtime": manifest.get("training_runtime"),
            "experiment_contract_sha256": saved_contract["sha256"],
            "base_model_override": base_model_override,
            "evaluated_splits": ["validation", "test"],
            "script_encoder_input": "script_content_only",
            "device": str(device),
            "dtype": str(dtype),
        },
    )
    print(metrics.to_string(index=False))
    print(f"Wrote per-job predictions and metrics to {output_dir}")


if __name__ == "__main__":
    main()
