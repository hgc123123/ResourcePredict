"""Train the unchanged neural baselines using the independent training tail.

The original baseline entry points keep their historical behavior. This module
changes only which rows fit preprocessing/parameters and select the checkpoint.
No pretrained language-model checkpoint is loaded.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.artifacts import build_prediction_frame, write_evaluation
from common.config import get_value
from common.data import DatasetBundle
from common.preprocessing import NumericScaler, TokenVocabulary, UserVocabulary
from qwen.base_selection import BaseTrainingSelection, verify_base_training_selection


def fit_preprocessors(config: dict, optimization: pd.DataFrame, numeric_features: list[str]):
    """Fit all learned input transformations exclusively on optimization rows."""
    vocabulary = TokenVocabulary.fit(
        optimization["code"].astype(str),
        max_size=int(get_value(config, "text.max_vocabulary", 30000)),
        min_frequency=int(get_value(config, "text.min_frequency", 2)),
    )
    users = UserVocabulary.fit(optimization["user_id"].astype(str))
    standardization = str(get_value(config, "data.metadata_standardization", "zscore"))
    if standardization not in {"zscore", "none"}:
        raise ValueError("data.metadata_standardization must be zscore or none")
    scaler = (
        NumericScaler.fit(optimization, numeric_features, standardize=standardization == "zscore")
        if numeric_features else None
    )
    return vocabulary, users, scaler


def architecture(config: dict, name: str):
    """Use the same builders and settings as the historical baseline scripts."""
    if name == "textcnn":
        from textcnn.model import build_model

        kernels = list(get_value(config, "textcnn.kernel_sizes", [3, 4, 5]))
        arguments = {
            "embedding_dim": int(get_value(config, "text.embedding_dim", 128)),
            "kernel_sizes": kernels,
            "filters_per_kernel": int(get_value(config, "textcnn.filters_per_kernel", 64)),
            "user_embedding_dim": int(get_value(config, "textcnn.user_embedding_dim", 16)),
            "metadata_hidden_dim": int(get_value(config, "textcnn.metadata_hidden_dim", 32)),
            "hidden_dim": int(get_value(config, "textcnn.hidden_dim", 128)),
            "dropout": float(get_value(config, "textcnn.dropout", 0.5)),
        }
        return build_model, arguments, max(kernels)
    if name == "lstm":
        from lstm.model import build_model

        arguments = {
            "embedding_dim": int(get_value(config, "text.embedding_dim", 128)),
            "hidden_dim": int(get_value(config, "lstm.hidden_dim", 128)),
            "layers": int(get_value(config, "lstm.layers", 1)),
            "bidirectional": bool(get_value(config, "lstm.bidirectional", True)),
            "user_embedding_dim": int(get_value(config, "lstm.user_embedding_dim", 16)),
            "metadata_hidden_dim": int(get_value(config, "lstm.metadata_hidden_dim", 32)),
            "head_hidden_dim": int(get_value(config, "lstm.head_hidden_dim", 128)),
            "dropout": float(get_value(config, "lstm.dropout", 0.3)),
        }
        return build_model, arguments, 1
    raise ValueError(f"Unsupported neural baseline: {name}")


def train(
    config: dict,
    bundle: DatasetBundle,
    selection: BaseTrainingSelection,
    output_dir: Path,
) -> None:
    import torch
    from torch.utils.data import DataLoader
    from common.neural import (
        ScriptRegressionDataset, _model_forward, _move_batch,
        checkpoint_identity, collate_script_batch, masked_smooth_l1,
        predict_loader, resolve_device, set_seed, validation_mean_mae,
    )

    manifest = verify_base_training_selection(bundle, config, selection.manifest)
    if manifest["mode"] != "train_tail" or manifest["validation_used_for_early_stopping"]:
        raise ValueError("Matched baselines require independent train-tail checkpoint selection")
    # Check the supplied frames as well as the manifest; no outer-validation
    # example may silently enter optimization or early stopping.
    for role, frame in (("optimization", selection.optimization), ("early_stopping", selection.early_stopping)):
        if frame.sample_id.astype(str).tolist() != manifest[f"{role}_sample_ids"]:
            raise ValueError(f"{role} rows do not match the verified selection")
        source = bundle.frame.set_index("sample_id").loc[frame.sample_id].reset_index()
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), source[frame.columns], check_dtype=False)

    name = str(get_value(config, "baseline_study.model"))
    builder, arguments, minimum_length = architecture(config, name)
    seed = int(get_value(config, "seed", 42))
    set_seed(seed)
    device = resolve_device(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_tokens = int(get_value(config, "text.max_tokens", 512))
    vocabulary, users, scaler = fit_preprocessors(config, selection.optimization, bundle.numeric_features)
    vocabulary.save(output_dir / "token_vocabulary.json")
    users.save(output_dir / "user_vocabulary.json")
    if scaler is not None:
        scaler.save(output_dir / "numeric_scaler.json")
    arguments.update(vocabulary_size=len(vocabulary), user_count=len(users), numeric_dim=len(bundle.numeric_features))
    model = builder(arguments).to(device)
    batch_size = int(get_value(config, "training.batch_size", 64))
    workers = int(get_value(config, "training.num_workers", 2))
    collate = partial(collate_script_batch, minimum_length=minimum_length)

    def loader(frame: pd.DataFrame, *, shuffle: bool = False):
        dataset = ScriptRegressionDataset(frame, vocabulary, users, scaler, max_tokens=max_tokens)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=device.type == "cuda", collate_fn=collate)

    fitting = loader(selection.optimization, shuffle=True)
    stopping = loader(selection.early_stopping)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(get_value(config, "training.learning_rate", 1e-3)),
        weight_decay=float(get_value(config, "training.weight_decay", 0.01)),
    )
    beta = float(get_value(config, "training.smooth_l1_beta", 0.10))
    vram_weight = float(get_value(config, "training.vram_loss_weight", 2.0))
    user_dropout = float(get_value(config, "training.user_dropout", 0.20))
    patience = int(get_value(config, "training.patience", 5))
    epochs = int(get_value(config, "training.epochs", 30))
    if min(batch_size, epochs, patience, max_tokens) < 1:
        raise ValueError("Batch size, epochs, patience, and max_tokens must be positive")
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_state = None
    best_epoch = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in fitting:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = _model_forward(model, batch, user_dropout=user_dropout)
            loss = masked_smooth_l1(predictions, batch["targets"], batch["mask"], beta, vram_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        predictions, targets, masks = predict_loader(model, stopping, device)
        metric = validation_mean_mae(predictions, targets, masks)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "early_stopping_mean_mae": metric})
        print(f"{name} seed={seed} epoch={epoch} train_loss={np.mean(losses):.6f} early_stopping_mean_mae={metric:.6f}", flush=True)
        if metric < best_metric - 1e-12:
            best_metric, best_epoch = metric, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite early-stopping checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    identity = checkpoint_identity(bundle, seed, max_tokens)
    torch.save({"model_state": best_state, "model_kwargs": arguments,
                "best_early_stopping_mean_mae": best_metric, "best_epoch": best_epoch,
                "numeric_features": bundle.numeric_features, "identity": identity,
                "base_training_selection": manifest}, output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "architecture.json").write_text(json.dumps(arguments, indent=2), encoding="utf-8")

    # Outer validation is evaluated only after the chosen checkpoint is frozen.
    evaluation = pd.concat([bundle.split("validation"), bundle.split("test")], ignore_index=True)
    predicted, _, _ = predict_loader(model, loader(evaluation), device)
    prediction_frame = build_prediction_frame(
        evaluation, {"memory": predicted[:, 0], "vram": predicted[:, 1]}, name, seed,
        clip_predictions=bool(get_value(config, "evaluation.clip_predictions", True)),
    )
    metrics = write_evaluation(config, bundle, name, prediction_frame, output_dir, details={
        "base_training_selection": manifest, "preprocessing_fit": "optimization_only",
        "checkpoint_selection_source": "outer_train_tail", "outer_validation_used_for_selection": False,
        "refit_after_checkpoint_selection": False, "best_epoch": best_epoch,
        "best_early_stopping_mean_mae": best_metric, "checkpoint_identity": identity,
        "smooth_l1_beta": beta, "vram_loss_weight": vram_weight, "user_dropout": user_dropout,
    })
    print(metrics.to_string(index=False), flush=True)
