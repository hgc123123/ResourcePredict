from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.artifacts import output_directory
from common.config import (
    get_value,
    load_config,
    resolve_path,
    set_value,
    write_resolved_config,
)
from common.data import load_dataset, write_data_report
from qwen.base_selection import base_preprocessing_bundle, select_base_training_frames
from qwen.contract import experiment_contract, restore_effective_config
from qwen.dataset import (
    QwenBatchCollator,
    QwenJobDataset,
    QwenPreprocessors,
    ensure_tokenizer_padding,
    split_fingerprint,
)
from qwen.model import GatedQwenRegressor, masked_smooth_l1_loss


VARIANT_MODALITIES = {
    "full": ("code", "user", "metadata"),
    "code_only": ("code",),
    "no_code": ("user", "metadata"),
    "no_user": ("code", "metadata"),
    "no_metadata": ("code", "user"),
    "concat_fusion": ("code", "user", "metadata"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the script-only Qwen gated memory/VRAM peak regressor"
    )
    parser.add_argument("--config", required=True, help="Shared YAML experiment configuration")
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_MODALITIES),
        default="full",
        help="Strict modality ablation; disabled modalities are excluded from softmax",
    )
    parser.add_argument("--output-dir", help="Run directory; defaults under outputs/qwen/<variant>")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and fine-tune the encoder")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing checkpoint")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; may be repeated",
    )
    return parser.parse_args()


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _invalidate_overwrite_artifacts(run_dir: Path) -> None:
    """Remove completion sentinels/results before restarting this exact run."""
    for name in (
        "training_complete.json",
        "predictions.csv",
        "metrics.csv",
        "run_manifest.json",
        "base_training_selection.json",
    ):
        (run_dir / name).unlink(missing_ok=True)


def _precision_settings(config: Dict[str, Any], force_cpu: bool) -> tuple[str, torch.dtype]:
    precision = str(get_value(config, "qwen.precision", "bf16")).lower()
    if force_cpu or not torch.cuda.is_available():
        return "no", torch.float32
    if precision in {"fp32", "float32", "no"}:
        return "no", torch.float32
    if precision in {"bf16", "bfloat16"}:
        return "bf16", torch.bfloat16
    if precision in {"fp16", "float16"}:
        return "fp16", torch.float16
    raise ValueError("qwen.precision must be fp32, fp16, or bf16")


def _make_loader(
    dataset: QwenJobDataset,
    collator: QwenBatchCollator,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collator,
        generator=generator if shuffle else None,
    )


def validation_mae(
    model: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
) -> tuple[float, float, float]:
    model.eval()
    absolute_sums = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["user_ids"],
                batch["metadata"],
            )
            gathered_predictions, gathered_targets, gathered_mask = accelerator.gather_for_metrics(
                (output["predictions"], batch["targets"], batch["target_mask"])
            )
            errors = torch.abs(gathered_predictions.float() - gathered_targets.float())
            mask = gathered_mask.bool()
            for resource_index in range(2):
                valid = mask[:, resource_index]
                absolute_sums[resource_index] += float(errors[valid, resource_index].sum().item())
                counts[resource_index] += int(valid.sum().item())

    if np.any(counts == 0):
        raise ValueError(f"Validation split has zero valid labels; counts={counts.tolist()}")
    resource_mae = absolute_sums / counts
    return float(resource_mae[0]), float(resource_mae[1]), float(resource_mae.mean())


def _training_manifest(
    *,
    config: Dict[str, Any],
    bundle: Any,
    variant: str,
    seed: int,
    enabled_modalities: Sequence[str],
    best_epoch: int,
    best_validation_mae: float,
    effective_parameters: Dict[str, Any],
    base_training_selection: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "seed": seed,
        "enabled_modalities": list(enabled_modalities),
        "fusion_mode": "concat" if variant == "concat_fusion" else "gated",
        "data_path": str(bundle.data_path),
        "data_sha256": bundle.data_sha256,
        "split_fingerprint": split_fingerprint(bundle.frame),
        "split_manifest": str(bundle.manifest_path) if bundle.manifest_path else None,
        "numeric_features": list(bundle.numeric_features),
        "best_epoch": int(best_epoch),
        "best_validation_mean_mae": float(best_validation_mae),
        "selection_metric": "mean(memory_mae, vram_mae) on validation",
        "effective_parameters": effective_parameters,
        "experiment_contract": experiment_contract(config, effective_parameters),
    }
    if base_training_selection is not None:
        manifest["base_training_selection"] = base_training_selection
        manifest["selection_metric"] = (
            "mean(memory_mae, vram_mae) on "
            + str(base_training_selection["early_stopping_source"])
        )
    return manifest


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(get_value(config, "seed", 42))
    set_value(config, "qwen.variant", args.variant)
    set_value(config, "qwen.enabled_modalities", list(VARIANT_MODALITIES[args.variant]))
    set_value(
        config,
        "qwen.fusion_mode",
        "concat" if args.variant == "concat_fusion" else "gated",
    )
    if args.no_lora:
        set_value(config, "qwen.lora.enabled", False)

    if args.output_dir:
        run_dir = resolve_path(config, args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = output_directory(config, "qwen", args.variant) / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoint"
    if args.overwrite and int(os.environ.get("RANK", "0")) == 0:
        _invalidate_overwrite_artifacts(run_dir)
    if (checkpoint_dir / "model_config.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_dir}. Use --overwrite or a new output directory."
        )

    device_request = str(get_value(config, "training.device", "auto")).lower()
    force_cpu = device_request == "cpu"
    if device_request not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")
    if device_request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is unavailable")
    mixed_precision, encoder_dtype = _precision_settings(config, force_cpu)

    gradient_accumulation = _positive_int(
        get_value(config, "qwen.gradient_accumulation_steps", 1),
        "qwen.gradient_accumulation_steps",
    )
    accelerator = Accelerator(
        cpu=force_cpu,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation,
    )
    set_seed(seed, device_specific=False)

    bundle = load_dataset(config)
    base_selection = select_base_training_frames(bundle, config)
    metadata_mode = str(get_value(config, "data.metadata_standardization", "zscore"))
    if metadata_mode not in {"zscore", "none"}:
        raise ValueError("data.metadata_standardization must be zscore or none")
    preprocessors = QwenPreprocessors.fit(
        base_preprocessing_bundle(bundle, base_selection),
        standardize_metadata=metadata_mode == "zscore",
    )
    train_frame = base_selection.optimization
    validation_frame = base_selection.early_stopping

    model_name = str(get_value(config, "qwen.model_name_or_path"))
    if not model_name or model_name == "None":
        raise ValueError("qwen.model_name_or_path is required")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(get_value(config, "qwen.trust_remote_code", True)),
    )
    ensure_tokenizer_padding(tokenizer)
    tokenizer.padding_side = "right"

    max_tokens = _positive_int(get_value(config, "qwen.max_tokens", 2048), "qwen.max_tokens")
    batch_size = _positive_int(get_value(config, "qwen.batch_size", 16), "qwen.batch_size")
    num_workers = int(get_value(config, "qwen.num_workers", get_value(config, "training.num_workers", 0)))
    if num_workers < 0:
        raise ValueError("qwen.num_workers must be non-negative")
    user_dropout = float(
        get_value(config, "qwen.user_dropout", get_value(config, "training.user_dropout", 0.20))
    )
    head_fraction = float(get_value(config, "qwen.head_fraction", 0.60))
    prompt_prefix = str(get_value(config, "qwen.prompt_prefix", ""))
    prompt_suffix = str(get_value(config, "qwen.prompt_suffix", ""))
    collator = QwenBatchCollator(
        tokenizer,
        max_tokens=max_tokens,
        head_fraction=head_fraction,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        pad_to_multiple_of=8 if accelerator.device.type == "cuda" else None,
    )
    train_dataset = QwenJobDataset(
        train_frame,
        preprocessors,
        training=True,
        user_dropout=user_dropout,
        require_any_label=True,
    )
    validation_dataset = QwenJobDataset(
        validation_frame,
        preprocessors,
        training=False,
        require_any_label=False,
    )
    train_loader = _make_loader(
        train_dataset,
        collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    validation_loader = _make_loader(
        validation_dataset,
        collator,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    quantization = str(get_value(config, "qwen.quantization", "none")).lower()
    if quantization != "none":
        raise ValueError(
            "This reproducible implementation currently requires qwen.quantization=none; "
            "add and validate a quantized path before reporting it"
        )
    lora_config = dict(get_value(config, "qwen.lora", {}) or {})
    attention_implementation = get_value(config, "qwen.attention_implementation")
    if not attention_implementation:
        attention_implementation = "eager" if accelerator.device.type == "cpu" else "sdpa"

    model = GatedQwenRegressor(
        model_name_or_path=model_name,
        num_users=len(preprocessors.user_vocabulary),
        metadata_dim=len(bundle.numeric_features),
        fusion_dim=_positive_int(get_value(config, "qwen.fusion_dim", 256), "qwen.fusion_dim"),
        user_embedding_dim=_positive_int(
            get_value(config, "qwen.user_embedding_dim", 64), "qwen.user_embedding_dim"
        ),
        head_hidden_dim=_positive_int(
            get_value(config, "qwen.head_hidden_dim", 128), "qwen.head_hidden_dim"
        ),
        dropout=float(get_value(config, "qwen.dropout", 0.30)),
        enabled_modalities=VARIANT_MODALITIES[args.variant],
        fusion_mode="concat" if args.variant == "concat_fusion" else "gated",
        torch_dtype=encoder_dtype,
        attention_implementation=str(attention_implementation),
        trust_remote_code=bool(get_value(config, "qwen.trust_remote_code", True)),
        gradient_checkpointing=bool(get_value(config, "qwen.gradient_checkpointing", True)),
        lora_config=lora_config,
    )

    encoder_lr = float(get_value(config, "qwen.learning_rate", 1e-4))
    head_lr = float(get_value(config, "qwen.head_learning_rate", encoder_lr))
    weight_decay = float(
        get_value(config, "qwen.weight_decay", 0.05)
    )
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("code_encoder.") and parameter.requires_grad
    ]
    other_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("code_encoder.") and parameter.requires_grad
    ]
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": encoder_lr})
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": head_lr})
    if not parameter_groups:
        raise ValueError("The selected variant has no trainable parameters")
    optimizer = AdamW(parameter_groups, weight_decay=weight_decay)

    model, optimizer, train_loader, validation_loader = accelerator.prepare(
        model, optimizer, train_loader, validation_loader
    )
    optimizer.zero_grad(set_to_none=True)

    epochs = _positive_int(get_value(config, "qwen.epochs", 20), "qwen.epochs")
    patience = _positive_int(get_value(config, "qwen.patience", 5), "qwen.patience")
    smooth_l1_beta = float(get_value(config, "qwen.smooth_l1_beta", 0.10))
    vram_loss_weight = float(
        get_value(
            config,
            "qwen.vram_loss_weight",
            get_value(config, "training.vram_loss_weight", 2.0),
        )
    )
    if vram_loss_weight <= 0:
        raise ValueError("qwen.vram_loss_weight must be positive")
    gradient_clip = float(get_value(config, "qwen.gradient_clip", 1.0))
    min_delta = float(get_value(config, "qwen.early_stopping_min_delta", 0.0))

    effective_parameters = {
        "max_tokens": max_tokens,
        "script_token_budget": collator.script_token_budget,
        "head_fraction": head_fraction,
        "prompt_prefix": prompt_prefix,
        "prompt_suffix": prompt_suffix,
        "batch_size_per_process": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "world_size": accelerator.num_processes,
        "effective_global_batch_size": (
            batch_size * gradient_accumulation * accelerator.num_processes
        ),
        "epochs": epochs,
        "patience": patience,
        "encoder_learning_rate": encoder_lr,
        "head_learning_rate": head_lr,
        "weight_decay": weight_decay,
        "user_dropout": user_dropout,
        "smooth_l1_beta": smooth_l1_beta,
        "vram_loss_weight": vram_loss_weight,
        "precision": mixed_precision,
        "attention_implementation": attention_implementation,
        "lora": lora_config,
    }
    if base_selection.manifest is not None:
        effective_parameters["base_training_selection_sha256"] = base_selection.manifest["sha256"]

    if accelerator.is_main_process:
        write_resolved_config(config, run_dir / "config.resolved.yaml")
        write_data_report(bundle, run_dir / "data_report.json")
        if base_selection.manifest is not None:
            with (run_dir / "base_training_selection.json").open("w", encoding="utf-8") as handle:
                json.dump(base_selection.manifest, handle, indent=2, ensure_ascii=True)
        print(
            f"Training qwen/{args.variant}: train={len(train_dataset)}, "
            f"validation={len(validation_dataset)}, device={accelerator.device}, "
            f"world_size={accelerator.num_processes}"
        )
        if base_selection.manifest is not None:
            print(
                "Checkpoint selection source="
                + str(base_selection.manifest["early_stopping_source"])
                + "; original outer validation/test splits remain unchanged"
            )

    history: list[Dict[str, Any]] = []
    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        local_loss_sum = 0.0
        local_batches = 0
        local_gate_sum = torch.zeros(3, device=accelerator.device, dtype=torch.float64)
        local_gate_count = 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                output = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["user_ids"],
                    batch["metadata"],
                )
                loss = masked_smooth_l1_loss(
                    output["predictions"],
                    batch["targets"],
                    batch["target_mask"],
                    beta=smooth_l1_beta,
                    resource_weights=(1.0, vram_loss_weight),
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients and gradient_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            local_loss_sum += float(loss.detach().float().item())
            local_batches += 1
            local_gate_sum += output["gate_weights"].detach().double().sum(dim=0)
            local_gate_count += int(output["gate_weights"].shape[0])

        epoch_stats = torch.cat(
            (
                torch.tensor(
                    [local_loss_sum, float(local_batches)],
                    device=accelerator.device,
                    dtype=torch.float64,
                ),
                local_gate_sum,
                torch.tensor(
                    [float(local_gate_count)],
                    device=accelerator.device,
                    dtype=torch.float64,
                ),
            )
        )
        epoch_stats = accelerator.reduce(epoch_stats, reduction="sum")
        train_loss = float((epoch_stats[0] / epoch_stats[1].clamp_min(1.0)).item())
        mean_gates = (epoch_stats[2:5] / epoch_stats[5].clamp_min(1.0)).cpu().tolist()
        memory_mae, vram_mae, validation_mean_mae = validation_mae(
            model, validation_loader, accelerator
        )
        improved = validation_mean_mae < (best_validation_mae - min_delta)

        if improved:
            best_validation_mae = validation_mean_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_checkpoint(checkpoint_dir, tokenizer=tokenizer)
                preprocessors.save(checkpoint_dir)
                manifest = _training_manifest(
                    config=config,
                    bundle=bundle,
                    variant=args.variant,
                    seed=seed,
                    enabled_modalities=VARIANT_MODALITIES[args.variant],
                    best_epoch=best_epoch,
                    best_validation_mae=best_validation_mae,
                    effective_parameters=effective_parameters,
                    base_training_selection=base_selection.manifest,
                )
                restore_effective_config(config, manifest["experiment_contract"])
                with (checkpoint_dir / "training_manifest.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(manifest, handle, indent=2, ensure_ascii=True)
                write_resolved_config(config, checkpoint_dir / "config.resolved.yaml")
                write_resolved_config(config, run_dir / "config.resolved.yaml")
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_memory_mae": memory_mae,
            "validation_vram_mae": vram_mae,
            "validation_mean_mae": validation_mean_mae,
            "train_gate_code_mean": float(mean_gates[0]),
            "train_gate_user_mean": float(mean_gates[1]),
            "train_gate_metadata_mean": float(mean_gates[2]),
            "is_best": improved,
        }
        history.append(row)
        if accelerator.is_main_process:
            pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.6f} "
                f"val_memory_mae={memory_mae:.6f} val_vram_mae={vram_mae:.6f} "
                f"val_mean_mae={validation_mean_mae:.6f} "
                f"gates=({mean_gates[0]:.3f},{mean_gates[1]:.3f},{mean_gates[2]:.3f}) "
                f"best={improved}"
            )
        accelerator.wait_for_everyone()

        if epochs_without_improvement >= patience:
            if accelerator.is_main_process:
                print(
                    f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                    f"validation mean MAE={best_validation_mae:.6f}"
                )
            break

    if best_epoch == 0:
        raise RuntimeError("Training completed without producing a checkpoint")
    accelerator.wait_for_everyone()
    elapsed_seconds = time.perf_counter() - training_started
    peak_allocated = (
        float(torch.cuda.max_memory_allocated(accelerator.device))
        if accelerator.device.type == "cuda"
        else 0.0
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved(accelerator.device))
        if accelerator.device.type == "cuda"
        else 0.0
    )
    runtime_values = torch.tensor(
        [elapsed_seconds, peak_allocated, peak_reserved],
        device=accelerator.device,
        dtype=torch.float64,
    )
    runtime_values = accelerator.gather(runtime_values.unsqueeze(0)).max(dim=0).values
    if accelerator.is_main_process:
        manifest_path = checkpoint_dir / "training_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["training_runtime"] = {
            "elapsed_seconds": float(runtime_values[0].item()),
            "completed_epochs": int(len(history)),
            "early_stopped": bool(len(history) < epochs),
            "world_size": int(accelerator.num_processes),
            "effective_global_batch_size": int(
                effective_parameters["effective_global_batch_size"]
            ),
            "peak_cuda_memory_allocated_bytes_per_process_max": int(
                runtime_values[1].item()
            ),
            "peak_cuda_memory_reserved_bytes_per_process_max": int(
                runtime_values[2].item()
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(accelerator.device)
                if accelerator.device.type == "cuda"
                else None
            ),
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True)
        with (run_dir / "training_complete.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "complete",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "best_epoch": int(best_epoch),
                    "completed_epochs": int(len(history)),
                    "training_manifest": str(manifest_path),
                    "experiment_contract_sha256": manifest["experiment_contract"][
                        "sha256"
                    ],
                },
                handle,
                indent=2,
                ensure_ascii=True,
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
