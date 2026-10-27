from __future__ import annotations

import hashlib
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import build_prediction_frame, output_directory, write_evaluation
from .config import get_value
from .data import DatasetBundle
from .preprocessing import NumericScaler, TokenVocabulary, UserVocabulary


ModelBuilder = Callable[[Dict[str, Any]], nn.Module]
CHECKPOINT_IDENTITY_FIELDS = (
    "data_sha256",
    "standardized_input_fingerprint",
    "split_fingerprint",
    "seed",
    "max_tokens",
    "numeric_features",
)


def split_fingerprint(frame: pd.DataFrame) -> str:
    """Hash the ordered sample-to-split assignments without delimiter ambiguity."""
    required = {"sample_id", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot fingerprint split assignments; missing columns: {missing}")
    assignments = [
        [str(sample_id), str(split)]
        for sample_id, split in frame[["sample_id", "split"]].itertuples(
            index=False, name=None
        )
    ]
    payload = json.dumps(
        assignments, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def standardized_input_fingerprint(bundle: DatasetBundle) -> str:
    """Bind a checkpoint to the configured columns, labels, and engineered inputs."""
    candidates = [
        "sample_id",
        "split",
        "code",
        "user_id",
        "target_memory",
        "target_vram",
        "valid_memory",
        "valid_vram",
        *bundle.numeric_features,
        *bundle.categorical_features,
    ]
    columns = list(dict.fromkeys(column for column in candidates if column in bundle.frame))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(columns, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    )
    for values in bundle.frame[columns].itertuples(index=False, name=None):
        normalized = []
        for value in values:
            if pd.isna(value):
                normalized.append(["missing", None])
            elif isinstance(value, (bool, np.bool_)):
                normalized.append(["bool", bool(value)])
            elif isinstance(value, (int, np.integer)):
                normalized.append(["int", int(value)])
            elif isinstance(value, (float, np.floating)):
                normalized.append(["float", format(float(value), ".17g")])
            else:
                normalized.append(["string", str(value)])
        payload = json.dumps(
            normalized, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def checkpoint_identity(
    bundle: DatasetBundle,
    seed: int,
    max_tokens: int,
) -> Dict[str, Any]:
    return {
        "data_sha256": str(bundle.data_sha256),
        "standardized_input_fingerprint": standardized_input_fingerprint(bundle),
        "split_fingerprint": split_fingerprint(bundle.frame),
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "numeric_features": list(bundle.numeric_features),
    }


def validate_checkpoint_identity(
    stored: Any,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(stored, dict):
        raise ValueError(
            "Checkpoint has no provenance identity; retrain it with the current code"
        )
    mismatches = []
    for field in CHECKPOINT_IDENTITY_FIELDS:
        if field not in stored:
            mismatches.append(f"{field}=<missing>")
        elif stored[field] != expected[field]:
            mismatches.append(
                f"{field}: checkpoint={stored[field]!r}, current={expected[field]!r}"
            )
    if mismatches:
        raise ValueError("Checkpoint provenance mismatch: " + "; ".join(mismatches))


def validate_training_label_coverage(bundle: DatasetBundle) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "validation"):
        part = bundle.split(split)
        for resource, valid_column in (
            ("memory", "valid_memory"),
            ("vram", "valid_vram"),
        ):
            count = int(part[valid_column].fillna(False).astype(bool).sum())
            counts[f"{split}_{resource}"] = count
            if count == 0:
                raise ValueError(
                    f"{split.capitalize()} split has no valid {resource} targets"
                )
    return counts


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: Dict[str, Any]) -> torch.device:
    requested = str(get_value(config, "training.device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


class ScriptRegressionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        token_vocabulary: TokenVocabulary,
        user_vocabulary: UserVocabulary,
        numeric_scaler: Optional[NumericScaler],
        max_tokens: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.token_ids = [
            token_vocabulary.encode(text, max_tokens=max_tokens)
            for text in self.frame["code"].astype(str)
        ]
        self.user_ids = user_vocabulary.encode(self.frame["user_id"].astype(str))
        if numeric_scaler is None:
            self.numeric = np.zeros((len(self.frame), 0), dtype=np.float32)
        else:
            self.numeric = numeric_scaler.transform(self.frame)
        targets = self.frame[["target_memory", "target_vram"]].to_numpy(float)
        masks = self.frame[["valid_memory", "valid_vram"]].to_numpy(bool)
        self.targets = np.where(np.isfinite(targets), targets, 0.0).astype(np.float32)
        self.masks = masks.astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "token_ids": self.token_ids[index],
            "user_id": int(self.user_ids[index]),
            "numeric": self.numeric[index],
            "targets": self.targets[index],
            "mask": self.masks[index],
        }


def collate_script_batch(items: List[Dict[str, Any]], minimum_length: int = 1) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([len(item["token_ids"]) for item in items], dtype=torch.long)
    padded_length = max(int(lengths.max()), int(minimum_length))
    token_ids = torch.zeros((len(items), padded_length), dtype=torch.long)
    for row, item in enumerate(items):
        ids = torch.tensor(item["token_ids"], dtype=torch.long)
        token_ids[row, : len(ids)] = ids
    return {
        "token_ids": token_ids,
        "lengths": lengths,
        "user_ids": torch.tensor([item["user_id"] for item in items], dtype=torch.long),
        "numeric": torch.tensor(np.stack([item["numeric"] for item in items]), dtype=torch.float32),
        "targets": torch.tensor(np.stack([item["targets"] for item in items]), dtype=torch.float32),
        "mask": torch.tensor(np.stack([item["mask"] for item in items]), dtype=torch.float32),
    }


def masked_smooth_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    vram_weight: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(predictions, targets, reduction="none", beta=beta)
    task_weights = predictions.new_tensor([1.0, vram_weight]).view(1, 2)
    weights = mask * task_weights
    denominator = weights.sum().clamp_min(1.0)
    return (loss * weights).sum() / denominator


def _model_forward(model: nn.Module, batch: Dict[str, torch.Tensor], user_dropout: float) -> torch.Tensor:
    user_ids = batch["user_ids"]
    if model.training and user_dropout > 0:
        replace = torch.rand(user_ids.shape, device=user_ids.device) < user_dropout
        user_ids = user_ids.masked_fill(replace, 0)
    logits = model(
        token_ids=batch["token_ids"],
        lengths=batch["lengths"],
        user_ids=user_ids,
        numeric=batch["numeric"],
    )
    return torch.sigmoid(logits)


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        predicted = _model_forward(model, batch, user_dropout=0.0)
        predictions.append(predicted.cpu().numpy())
        targets.append(batch["targets"].cpu().numpy())
        masks.append(batch["mask"].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)


def validation_mean_mae(predictions: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    values = []
    for task in range(2):
        valid = masks[:, task] > 0.5
        if valid.any():
            values.append(float(np.mean(np.abs(predictions[valid, task] - targets[valid, task]))))
    if not values:
        raise ValueError("Validation split has no valid labels")
    return float(np.mean(values))


def _fit_preprocessors(
    bundle: DatasetBundle,
    config: Dict[str, Any],
) -> Tuple[TokenVocabulary, UserVocabulary, Optional[NumericScaler]]:
    train = bundle.split("train")
    vocabulary = TokenVocabulary.fit(
        train["code"].astype(str),
        max_size=int(get_value(config, "text.max_vocabulary", 30000)),
        min_frequency=int(get_value(config, "text.min_frequency", 2)),
    )
    users = UserVocabulary.fit(train["user_id"].astype(str))
    metadata_mode = str(get_value(config, "data.metadata_standardization", "zscore"))
    if metadata_mode not in {"zscore", "none"}:
        raise ValueError("data.metadata_standardization must be zscore or none")
    scaler = (
        NumericScaler.fit(
            train,
            bundle.numeric_features,
            standardize=metadata_mode == "zscore",
        )
        if bundle.numeric_features
        else None
    )
    return vocabulary, users, scaler


def _loaders(
    bundle: DatasetBundle,
    config: Dict[str, Any],
    vocabulary: TokenVocabulary,
    users: UserVocabulary,
    scaler: Optional[NumericScaler],
    minimum_length: int,
) -> Dict[str, DataLoader]:
    batch_size = int(get_value(config, "training.batch_size", 64))
    workers = int(get_value(config, "training.num_workers", 2))
    max_tokens = int(get_value(config, "text.max_tokens", 512))
    collate = partial(collate_script_batch, minimum_length=minimum_length)
    result: Dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        dataset = ScriptRegressionDataset(
            bundle.split(split), vocabulary, users, scaler, max_tokens=max_tokens
        )
        result[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
    return result


def train_script_model(
    config: Dict[str, Any],
    bundle: DatasetBundle,
    model_name: str,
    model_builder: ModelBuilder,
    architecture: Dict[str, Any],
    minimum_length: int = 1,
) -> Path:
    seed = int(get_value(config, "seed", 42))
    max_tokens = int(get_value(config, "text.max_tokens", 512))
    set_seed(seed)
    label_counts = validate_training_label_coverage(bundle)
    device = resolve_device(config)
    output_dir = output_directory(config, model_name, f"seed_{seed}")
    identity = checkpoint_identity(bundle, seed, max_tokens)

    vocabulary, users, scaler = _fit_preprocessors(bundle, config)
    vocabulary.save(output_dir / "token_vocabulary.json")
    users.save(output_dir / "user_vocabulary.json")
    if scaler is not None:
        scaler.save(output_dir / "numeric_scaler.json")

    model_kwargs = dict(architecture)
    model_kwargs.update(
        {
            "vocabulary_size": len(vocabulary),
            "user_count": len(users),
            "numeric_dim": len(bundle.numeric_features),
        }
    )
    model = model_builder(model_kwargs).to(device)
    loaders = _loaders(bundle, config, vocabulary, users, scaler, minimum_length)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get_value(config, "training.learning_rate", 1e-3)),
        weight_decay=float(get_value(config, "training.weight_decay", 0.01)),
    )
    beta = float(get_value(config, "training.smooth_l1_beta", 0.10))
    vram_weight = float(get_value(config, "training.vram_loss_weight", 2.0))
    user_dropout = float(get_value(config, "training.user_dropout", 0.20))
    patience = int(get_value(config, "training.patience", 5))
    epochs = int(get_value(config, "training.epochs", 30))

    best_metric = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = _model_forward(model, batch, user_dropout=user_dropout)
            loss = masked_smooth_l1(
                predictions, batch["targets"], batch["mask"], beta, vram_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_predictions, val_targets, val_masks = predict_loader(
            model, loaders["validation"], device
        )
        val_mae = validation_mean_mae(val_predictions, val_targets, val_masks)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mean_mae": val_mae}
        )
        print(
            f"epoch={epoch} train_loss={np.mean(losses):.6f} "
            f"validation_mean_mae={val_mae:.6f}"
        )
        if val_mae < best_metric - 1e-12:
            best_metric = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save(
        {
            "model_state": best_state,
            "model_kwargs": model_kwargs,
            "best_validation_mean_mae": best_metric,
            "numeric_features": bundle.numeric_features,
            "identity": identity,
        },
        output_dir / "model.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    with (output_dir / "architecture.json").open("w", encoding="utf-8") as handle:
        json.dump(model_kwargs, handle, indent=2)

    validation_predictions, _, _ = predict_loader(model, loaders["validation"], device)
    test_predictions, _, _ = predict_loader(model, loaders["test"], device)
    eval_frame = pd.concat(
        [bundle.split("validation"), bundle.split("test")], ignore_index=True
    )
    all_predictions = np.concatenate([validation_predictions, test_predictions], axis=0)
    prediction_frame = build_prediction_frame(
        eval_frame,
        {"memory": all_predictions[:, 0], "vram": all_predictions[:, 1]},
        model_name=model_name,
        seed=seed,
        clip_predictions=bool(get_value(config, "evaluation.clip_predictions", True)),
    )
    write_evaluation(
        config,
        bundle,
        model_name,
        prediction_frame,
        output_dir,
        details={
            "best_validation_mean_mae": best_metric,
            "smooth_l1_beta": beta,
            "user_dropout": user_dropout,
            "preprocessing_fit": "train_only",
            "checkpoint_identity": identity,
            "training_label_counts": label_counts,
        },
    )
    return output_dir


def evaluate_script_checkpoint(
    config: Dict[str, Any],
    bundle: DatasetBundle,
    model_name: str,
    model_builder: ModelBuilder,
    minimum_length: int = 1,
) -> Path:
    seed = int(get_value(config, "seed", 42))
    max_tokens = int(get_value(config, "text.max_tokens", 512))
    set_seed(seed)
    device = resolve_device(config)
    output_dir = output_directory(config, model_name, f"seed_{seed}")
    checkpoint_path = output_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {checkpoint_path}")
    identity = checkpoint_identity(bundle, seed, max_tokens)
    validate_checkpoint_identity(checkpoint.get("identity"), identity)
    vocabulary = TokenVocabulary.load(output_dir / "token_vocabulary.json")
    users = UserVocabulary.load(output_dir / "user_vocabulary.json")
    scaler_path = output_dir / "numeric_scaler.json"
    scaler = NumericScaler.load(scaler_path) if scaler_path.exists() else None
    model = model_builder(checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    loaders = _loaders(bundle, config, vocabulary, users, scaler, minimum_length)
    validation_predictions, _, _ = predict_loader(model, loaders["validation"], device)
    test_predictions, _, _ = predict_loader(model, loaders["test"], device)
    eval_frame = pd.concat([bundle.split("validation"), bundle.split("test")], ignore_index=True)
    all_predictions = np.concatenate([validation_predictions, test_predictions], axis=0)
    prediction_frame = build_prediction_frame(
        eval_frame,
        {"memory": all_predictions[:, 0], "vram": all_predictions[:, 1]},
        model_name=model_name,
        seed=seed,
        clip_predictions=bool(get_value(config, "evaluation.clip_predictions", True)),
    )
    write_evaluation(
        config,
        bundle,
        model_name,
        prediction_frame,
        output_dir,
        details={
            "evaluation_only": True,
            "checkpoint": str(checkpoint_path),
            "checkpoint_identity": identity,
        },
    )
    return output_dir
