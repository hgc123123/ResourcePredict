#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.artifacts import dependency_versions  # noqa: E402
from common.config import get_value, load_config, resolve_path, write_resolved_config  # noqa: E402
from common.data import sha256_file  # noqa: E402
from common.history import causal_history_contract  # noqa: E402
from dynamic_hybrid.dataset import ResourceArrays, ResourceGateDataset  # noqa: E402
from dynamic_hybrid.features import RobustFeatureScaler  # noqa: E402
from dynamic_hybrid.model import (  # noqa: E402
    RESOURCES,
    DynamicHybridModel,
    dynamic_hybrid_loss,
)
from dynamic_hybrid.pipeline import (  # noqa: E402
    gate_fit_and_selection_masks,
    prepare_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train causal resource-specific dynamic gates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-predictions")
    parser.add_argument("--last1-predictions")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: Dict[str, Any]) -> torch.device:
    name = str(get_value(config, "dynamic_hybrid.device", "auto"))
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _initial_alpha(frame: pd.DataFrame, mask: np.ndarray, step: float) -> float:
    if not 0.0 < step <= 1.0:
        raise ValueError("dynamic_hybrid.initial_alpha_grid_step must be in (0,1]")
    part = frame.loc[mask]
    qwen = part["qwen_prediction"].to_numpy(float)
    last1 = part["last1_prediction"].to_numpy(float)
    history = part["history_available"].to_numpy(bool)
    target = part["target"].to_numpy(float)
    grid = np.unique(np.append(np.arange(0.0, 1.0 + step * 0.5, step), 1.0))
    losses = []
    for alpha in grid:
        prediction = np.where(history, alpha * last1 + (1.0 - alpha) * qwen, qwen)
        losses.append(float(np.mean(np.abs(np.clip(prediction, 0.0, 1.0) - target))))
    return float(grid[int(np.argmin(losses))])


def _last1_selection_metrics(frame: pd.DataFrame, mask: np.ndarray) -> Dict[str, float]:
    part = frame.loc[mask]
    prediction = np.clip(part["last1_prediction"].to_numpy(float), 0.0, 1.0)
    target = part["target"].to_numpy(float)
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "mse": float(np.square(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "acc_at_10": float(np.mean(absolute <= 0.10)),
    }


def _arrays(
    prepared_values: np.ndarray,
    frame: pd.DataFrame,
    mask: np.ndarray,
) -> ResourceArrays:
    indices = np.flatnonzero(mask)
    history = frame.loc[mask, "history_available"].to_numpy(bool)
    qwen = frame.loc[mask, "qwen_prediction"].to_numpy(np.float32)
    last1_raw = frame.loc[mask, "last1_prediction"].to_numpy(np.float32)
    return ResourceArrays(
        features=prepared_values[mask],
        qwen_prediction=qwen,
        last1_prediction=np.where(history, last1_raw, qwen).astype(np.float32),
        history_available=history,
        target=frame.loc[mask, "target"].to_numpy(np.float32),
        row_index=indices,
    )


@torch.inference_mode()
def _selection_metrics(
    model: DynamicHybridModel,
    resource: str,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    predictions, targets = [], []
    model.eval()
    for batch in loader:
        output = model(
            resource,
            batch["features"].to(device),
            batch["qwen_prediction"].to(device),
            batch["last1_prediction"].to(device),
            batch["history_available"].to(device),
        )
        predictions.append(output["prediction"].cpu().numpy())
        targets.append(batch["target"].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "mse": float(np.square(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "acc_at_10": float(np.mean(absolute <= 0.10)),
    }


def _fit_resource_for_epochs(
    model: DynamicHybridModel,
    resource: str,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    loss_parameters: Dict[str, float],
) -> list[float]:
    """Fit one resource for a fixed, validation-selected number of epochs."""
    optimizer = torch.optim.AdamW(
        model.resource_gates[resource].parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device)
            qwen = batch["qwen_prediction"].to(device)
            last1 = batch["last1_prediction"].to(device)
            available = batch["history_available"].to(device)
            target = batch["target"].to(device)
            output = model(resource, features, qwen, last1, available)
            loss, _ = dynamic_hybrid_loss(
                output,
                target,
                qwen,
                last1,
                available,
                **loss_parameters,
            )
            loss.backward()
            if gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.resource_gates[resource].parameters(), gradient_clip
                )
            optimizer.step()
            size = len(target)
            total_loss += float(loss.detach().cpu()) * size
            total_rows += size
        epoch_losses.append(total_loss / max(total_rows, 1))
    return epoch_losses


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    component_seed = int(get_value(config, "seed", 42))
    gate_seed = int(get_value(config, "dynamic_hybrid.seed", component_seed))
    _seed_everything(gate_seed)
    prepared = prepare_inputs(config, args.qwen_predictions, args.last1_predictions)
    qwen_variant = str(
        (prepared.components.qwen_manifest.get("details") or {}).get("variant")
    )
    model_name = f"dynamic_hybrid_{qwen_variant}"
    fit_base, selection_base, fit_protocol = gate_fit_and_selection_masks(prepared, config)
    frame = prepared.features.paired

    output_dir = resolve_path(
        config,
        args.output_dir
        or get_value(
            config,
            "dynamic_hybrid.output_dir",
            "outputs/dynamic_hybrid/completed_before_submission/"
            f"{get_value(config, 'dynamic_hybrid.qwen_variant', get_value(config, 'hybrid.qwen_variant', 'full'))}/"
            + (
                f"seed_{component_seed}"
                if gate_seed == component_seed
                else f"seed_{component_seed}_gate_{gate_seed}"
            ),
        ),
    )
    checkpoint = output_dir / "checkpoint"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint}; choose a new output or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, output_dir / "config.resolved.yaml")

    device = _device(config)
    hidden_dims = tuple(
        int(value)
        for value in get_value(config, "dynamic_hybrid.hidden_dims", [64, 32])
    )
    dropout = float(get_value(config, "dynamic_hybrid.dropout", 0.10))
    residual_max = float(get_value(config, "dynamic_hybrid.residual_max", 0.05))
    input_normalization = str(
        get_value(config, "dynamic_hybrid.input_normalization", "layernorm")
    )
    batch_size = int(get_value(config, "dynamic_hybrid.batch_size", 256))
    epochs = int(get_value(config, "dynamic_hybrid.epochs", 150))
    patience = int(get_value(config, "dynamic_hybrid.patience", 15))
    min_epochs = int(get_value(config, "dynamic_hybrid.min_epochs", 30))
    learning_rate = float(get_value(config, "dynamic_hybrid.learning_rate", 0.001))
    weight_decay = float(get_value(config, "dynamic_hybrid.weight_decay", 0.0001))
    gradient_clip = float(get_value(config, "dynamic_hybrid.gradient_clip", 1.0))
    if batch_size <= 0 or epochs <= 0 or patience <= 0 or min_epochs < 0:
        raise ValueError(
            "batch_size, epochs, and patience must be positive; min_epochs must "
            "be non-negative"
        )
    if min_epochs > epochs:
        raise ValueError("dynamic_hybrid.min_epochs cannot exceed epochs")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("dynamic_hybrid.learning_rate must be finite and positive")
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("dynamic_hybrid.weight_decay must be finite and non-negative")
    if not np.isfinite(gradient_clip) or gradient_clip < 0.0:
        raise ValueError("dynamic_hybrid.gradient_clip must be finite and non-negative")

    loss_parameters = {
        "smooth_l1_beta": float(
            get_value(config, "dynamic_hybrid.smooth_l1_beta", 0.05)
        ),
        "mse_weight": float(get_value(config, "dynamic_hybrid.mse_weight", 0.25)),
        "tolerance_weight": float(
            get_value(config, "dynamic_hybrid.tolerance_weight", 0.50)
        ),
        "tolerance": 0.10,
        "tolerance_temperature": float(
            get_value(config, "dynamic_hybrid.tolerance_temperature", 0.02)
        ),
        "underprediction_weight": float(
            get_value(config, "dynamic_hybrid.underprediction_weight", 0.05)
        ),
        "routing_weight": float(
            get_value(config, "dynamic_hybrid.routing_weight", 0.05)
        ),
        "residual_weight": float(
            get_value(config, "dynamic_hybrid.residual_weight", 0.01)
        ),
    }
    if not np.isfinite(list(loss_parameters.values())).all():
        raise ValueError("Dynamic-hybrid loss parameters must all be finite")
    if loss_parameters["smooth_l1_beta"] <= 0.0:
        raise ValueError("dynamic_hybrid.smooth_l1_beta must be positive")
    if loss_parameters["tolerance_temperature"] <= 0.0:
        raise ValueError("dynamic_hybrid.tolerance_temperature must be positive")
    for name in (
        "mse_weight",
        "tolerance_weight",
        "underprediction_weight",
        "routing_weight",
        "residual_weight",
    ):
        if loss_parameters[name] < 0.0:
            raise ValueError(f"dynamic_hybrid.{name} must be non-negative")

    fit_masks: Dict[str, np.ndarray] = {}
    selection_masks: Dict[str, np.ndarray] = {}
    scalers: Dict[str, RobustFeatureScaler] = {}
    original_alphas: Dict[str, float] = {}
    initialized_alphas: Dict[str, float] = {}
    step = float(get_value(config, "dynamic_hybrid.initial_alpha_grid_step", 0.01))
    valid = frame["valid"].to_numpy(bool)
    resources = frame["resource"].astype(str).to_numpy()
    for resource in RESOURCES:
        fit_mask = fit_base & valid & (resources == resource)
        selection_mask = selection_base & valid & (resources == resource)
        if int(fit_mask.sum()) < 32 or int(selection_mask.sum()) < 16:
            raise ValueError(
                f"Insufficient {resource} gate rows: fit={int(fit_mask.sum())}, "
                f"selection={int(selection_mask.sum())}"
            )
        fit_masks[resource] = fit_mask
        selection_masks[resource] = selection_mask
        scalers[resource] = RobustFeatureScaler.fit(
            prepared.features.values[fit_mask], prepared.features.names
        )
        scalers[resource].save(checkpoint / f"scaler_{resource}.json")
        original = _initial_alpha(frame, fit_mask, step)
        original_alphas[resource] = original
        initialized_alphas[resource] = float(np.clip(original, 1e-4, 1.0 - 1e-4))

    model = DynamicHybridModel(
        len(prepared.features.names),
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual_max=residual_max,
        initial_alphas=initialized_alphas,
        input_normalization=input_normalization,
    ).to(device)

    history_rows = []
    best_metrics: Dict[str, Dict[str, float]] = {}
    best_epochs: Dict[str, int] = {}
    constraint_records: Dict[str, Dict[str, Any]] = {}
    max_acc_drop = float(
        get_value(config, "dynamic_hybrid.max_acc_drop_vs_last1", 0.005)
    )
    max_rmse_increase = float(
        get_value(config, "dynamic_hybrid.max_rmse_increase_vs_last1", 0.0)
    )
    if (
        not np.isfinite(max_acc_drop)
        or not np.isfinite(max_rmse_increase)
        or max_acc_drop < 0.0
        or max_rmse_increase < 0.0
    ):
        raise ValueError("Dynamic-hybrid selection constraint tolerances must be non-negative")
    constraint_failure_policy = str(
        get_value(config, "dynamic_hybrid.constraint_failure_policy", "error")
    )
    if constraint_failure_policy not in {"error", "lowest_mae"}:
        raise ValueError(
            "dynamic_hybrid.constraint_failure_policy must be error or lowest_mae"
        )
    for resource in RESOURCES:
        scaler = scalers[resource]
        scaled = scaler.transform(prepared.features.values, prepared.features.names)
        fit_loader = DataLoader(
            ResourceGateDataset(_arrays(scaled, frame, fit_masks[resource])),
            batch_size=batch_size,
            shuffle=True,
        )
        selection_loader = DataLoader(
            ResourceGateDataset(_arrays(scaled, frame, selection_masks[resource])),
            batch_size=batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(
            model.resource_gates[resource].parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        last1_metrics = _last1_selection_metrics(frame, selection_masks[resource])
        epoch_zero_metrics = _selection_metrics(model, resource, selection_loader, device)
        epoch_zero_state = copy.deepcopy(model.resource_gates[resource].state_dict())
        best_unconstrained_state = epoch_zero_state
        best_unconstrained_metrics = epoch_zero_metrics
        best_unconstrained_epoch = 0
        best_constrained_state = None
        best_constrained_metrics = None
        best_constrained_epoch = None

        def admissible(metrics: Dict[str, float]) -> bool:
            return bool(
                metrics["acc_at_10"] >= last1_metrics["acc_at_10"] - max_acc_drop
                and metrics["rmse"] <= last1_metrics["rmse"] + max_rmse_increase
            )

        if admissible(epoch_zero_metrics):
            best_constrained_state = epoch_zero_state
            best_constrained_metrics = epoch_zero_metrics
            best_constrained_epoch = 0
        history_rows.append(
            {
                "phase": "selection_fit",
                "resource": resource,
                "epoch": 0,
                "fit_loss": float("nan"),
                **{
                    f"selection_{name}": value
                    for name, value in epoch_zero_metrics.items()
                },
                "constraint_admissible": admissible(epoch_zero_metrics),
                "is_best_unconstrained_so_far": True,
                "is_best_constrained_so_far": admissible(epoch_zero_metrics),
            }
        )
        best_mae = epoch_zero_metrics["mae"]
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_rows = 0
            for batch in fit_loader:
                optimizer.zero_grad(set_to_none=True)
                features = batch["features"].to(device)
                qwen = batch["qwen_prediction"].to(device)
                last1 = batch["last1_prediction"].to(device)
                available = batch["history_available"].to(device)
                target = batch["target"].to(device)
                output = model(resource, features, qwen, last1, available)
                loss, _ = dynamic_hybrid_loss(
                    output,
                    target,
                    qwen,
                    last1,
                    available,
                    **loss_parameters,
                )
                loss.backward()
                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.resource_gates[resource].parameters(), gradient_clip
                    )
                optimizer.step()
                size = len(target)
                total_loss += float(loss.detach().cpu()) * size
                total_rows += size

            metrics = _selection_metrics(model, resource, selection_loader, device)
            improved = metrics["mae"] < best_mae
            if improved:
                best_mae = metrics["mae"]
                best_unconstrained_state = copy.deepcopy(
                    model.resource_gates[resource].state_dict()
                )
                best_unconstrained_metrics = metrics
                best_unconstrained_epoch = epoch
            is_admissible = admissible(metrics)
            constrained_improved = is_admissible and (
                best_constrained_metrics is None
                or metrics["mae"] < best_constrained_metrics["mae"]
            )
            if constrained_improved:
                best_constrained_state = copy.deepcopy(
                    model.resource_gates[resource].state_dict()
                )
                best_constrained_metrics = metrics
                best_constrained_epoch = epoch
            if improved or constrained_improved:
                stale = 0
            else:
                stale += 1
            history_rows.append(
                {
                    "phase": "selection_fit",
                    "resource": resource,
                    "epoch": epoch,
                    "fit_loss": total_loss / max(total_rows, 1),
                    **{f"selection_{name}": value for name, value in metrics.items()},
                    "constraint_admissible": is_admissible,
                    "is_best_unconstrained_so_far": improved,
                    "is_best_constrained_so_far": constrained_improved,
                }
            )
            if epoch >= min_epochs and stale >= patience:
                break
        constraint_fallback = best_constrained_state is None
        if constraint_fallback:
            if constraint_failure_policy == "error":
                failure = {
                    "resource": resource,
                    "reason": "no checkpoint satisfies validation-only Acc/RMSE constraints",
                    "last1_selection_metrics": last1_metrics,
                    "max_acc_drop_vs_last1": max_acc_drop,
                    "max_rmse_increase_vs_last1": max_rmse_increase,
                    "best_unconstrained_epoch": best_unconstrained_epoch,
                    "best_unconstrained_metrics": best_unconstrained_metrics,
                    "test_labels_used": False,
                }
                pd.DataFrame(history_rows).to_csv(
                    output_dir / "training_history.csv", index=False
                )
                with (output_dir / "constraint_failure.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(failure, handle, indent=2, ensure_ascii=True)
                raise RuntimeError(
                    f"No admissible {resource} checkpoint; see "
                    f"{output_dir / 'constraint_failure.json'}"
                )
            chosen_state = best_unconstrained_state
            chosen_metrics = best_unconstrained_metrics
            chosen_epoch = best_unconstrained_epoch
        else:
            chosen_state = best_constrained_state
            chosen_metrics = best_constrained_metrics
            chosen_epoch = best_constrained_epoch
        model.resource_gates[resource].load_state_dict(chosen_state)
        best_metrics[resource] = chosen_metrics
        best_epochs[resource] = int(chosen_epoch)
        constraint_records[resource] = {
            "last1_selection_metrics": last1_metrics,
            "max_acc_drop_vs_last1": max_acc_drop,
            "max_rmse_increase_vs_last1": max_rmse_increase,
            "constraint_fallback": constraint_fallback,
            "constraint_failure_policy": constraint_failure_policy,
            "chosen_epoch": int(chosen_epoch),
            "epoch_zero_metrics": epoch_zero_metrics,
        }

    refit_on_full_development = bool(
        get_value(config, "dynamic_hybrid.refit_on_full_development", True)
    )
    final_fit_masks = dict(fit_masks)
    final_alphas = dict(original_alphas)
    final_initialized_alphas = dict(initialized_alphas)
    final_fit_protocol: Dict[str, Any] = {
        "enabled": False,
        "source": fit_protocol["fit_source"],
        "validation_labels_used_for_refit": False,
        "test_labels_used": False,
    }
    if refit_on_full_development:
        development_base = fit_base | selection_base
        if np.any(development_base & frame["split"].eq("test").to_numpy()):
            raise AssertionError("Test rows entered final dynamic-hybrid refit")
        final_fit_masks = {}
        final_scalers: Dict[str, RobustFeatureScaler] = {}
        final_alphas = {}
        final_initialized_alphas = {}
        for resource in RESOURCES:
            mask = development_base & valid & (resources == resource)
            if int(mask.sum()) < 48:
                raise ValueError(
                    f"Insufficient {resource} rows for final refit: {int(mask.sum())}"
                )
            final_fit_masks[resource] = mask
            final_scalers[resource] = RobustFeatureScaler.fit(
                prepared.features.values[mask], prepared.features.names
            )
            final_scalers[resource].save(checkpoint / f"scaler_{resource}.json")
            alpha = _initial_alpha(frame, mask, step)
            final_alphas[resource] = alpha
            final_initialized_alphas[resource] = float(
                np.clip(alpha, 1e-4, 1.0 - 1e-4)
            )

        refit_seed = gate_seed + 1_000_003
        _seed_everything(refit_seed)
        refit_model = DynamicHybridModel(
            len(prepared.features.names),
            hidden_dims=hidden_dims,
            dropout=dropout,
            residual_max=residual_max,
            initial_alphas=final_initialized_alphas,
            input_normalization=input_normalization,
        ).to(device)
        for resource in RESOURCES:
            scaler = final_scalers[resource]
            scaled = scaler.transform(
                prepared.features.values, prepared.features.names
            )
            loader = DataLoader(
                ResourceGateDataset(
                    _arrays(scaled, frame, final_fit_masks[resource])
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            refit_losses = _fit_resource_for_epochs(
                refit_model,
                resource,
                loader,
                device,
                epochs=best_epochs[resource],
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                gradient_clip=gradient_clip,
                loss_parameters=loss_parameters,
            )
            for epoch, fit_loss in enumerate(refit_losses, start=1):
                history_rows.append(
                    {
                        "phase": "final_refit",
                        "resource": resource,
                        "epoch": epoch,
                        "fit_loss": fit_loss,
                        "constraint_admissible": np.nan,
                        "is_best_unconstrained_so_far": np.nan,
                        "is_best_constrained_so_far": np.nan,
                    }
                )
        model = refit_model
        final_fit_protocol = {
            "enabled": True,
            "source": "union of gate fit and checkpoint-selection development rows",
            "selected_epoch_source": fit_protocol["selection_source"],
            "refit_seed": refit_seed,
            "validation_labels_used_for_refit": bool(
                np.any(development_base & frame["split"].eq("validation").to_numpy())
            ),
            "test_labels_used": False,
        }

    torch.save(model.state_dict(), checkpoint / "dynamic_hybrid.pt")
    model_config = {
        "input_dim": len(prepared.features.names),
        "feature_names": prepared.features.names,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "residual_max": residual_max,
        "input_normalization": input_normalization,
        "original_fixed_alphas": final_alphas,
        "initialized_alphas": final_initialized_alphas,
        "selection_original_fixed_alphas": original_alphas,
        "selection_initialized_alphas": initialized_alphas,
        "strict_qwen_no_history": True,
    }
    with (checkpoint / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2, ensure_ascii=True)

    checkpoint_files_sha256 = {
        name: sha256_file(checkpoint / name)
        for name in (
            "dynamic_hybrid.pt",
            "model_config.json",
            "scaler_memory.json",
            "scaler_vram.json",
        )
    }

    training_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "seed": gate_seed,
        "gate_seed": gate_seed,
        "component_seed": component_seed,
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
        "component_contract": prepared.components.checkpoint_contract(),
        "fit_protocol": fit_protocol,
        "final_fit_protocol": final_fit_protocol,
        "history_protocol": causal_history_contract(
            int(get_value(config, "dynamic_hybrid.history_k", 5)),
            float(get_value(config, "dynamic_hybrid.history_ema_alpha", 0.5)),
        ),
        "feature_whitelist": prepared.features.names,
        "fit_rows": {resource: int(fit_masks[resource].sum()) for resource in RESOURCES},
        "final_fit_rows": {
            resource: int(final_fit_masks[resource].sum()) for resource in RESOURCES
        },
        "selection_rows": {
            resource: int(selection_masks[resource].sum()) for resource in RESOURCES
        },
        "selection_metric": "resource-specific validation/calibration-tail MAE",
        "selection_results_scope": (
            "selection-stage checkpoints before optional full-development refit"
        ),
        "best_epoch": best_epochs,
        "resource_model_kind": {
            resource: (
                "dynamic_gate" if best_epochs[resource] > 0 else "static_alpha_fallback"
            )
            for resource in RESOURCES
        },
        "best_selection_metrics": best_metrics,
        "selection_constraints": constraint_records,
        "loss_parameters": loss_parameters,
        "optimization_protocol": {
            "routing_supervision": "continuous_line_segment_projection_alpha",
            "early_stopping_tracks": [
                "best_unconstrained_mae",
                "best_constraint-admissible_mae",
            ],
            "min_epochs": min_epochs,
            "patience": patience,
            "input_normalization": input_normalization,
            "rare_binary_features_preserved_during_scaling": True,
        },
        "test_labels_used_for_fit_or_selection": False,
        "model_config": model_config,
        "checkpoint_files_sha256": checkpoint_files_sha256,
    }
    with (checkpoint / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(training_manifest, handle, indent=2, ensure_ascii=True)
    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
    print(json.dumps(training_manifest["best_selection_metrics"], indent=2))
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
