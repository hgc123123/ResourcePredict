from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.data import DatasetBundle
from common.preprocessing import NumericScaler, UserVocabulary


EMPTY_SCRIPT = "<EMPTY_SCRIPT>"


def split_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered sample-to-split assignments, not just the source CSV."""
    digest = hashlib.sha256()
    for sample_id, split in frame[["sample_id", "split"]].itertuples(index=False, name=None):
        digest.update(f"{sample_id}\t{split}\n".encode("utf-8"))
    return digest.hexdigest()


def script_encoder_input(text: Any) -> str:
    """Return the exact string passed to the tokenizer, without content normalization."""
    code = str(text)
    return code if code.strip() else EMPTY_SCRIPT


def script_sha256(text: Any) -> str:
    return hashlib.sha256(script_encoder_input(text).encode("utf-8")).hexdigest()


def validate_qwen_bundle(bundle: DatasetBundle) -> None:
    if bundle.categorical_features:
        raise ValueError(
            "The Qwen metadata tower currently accepts numeric metadata only. "
            f"Remove or encode data.metadata_categorical first: {bundle.categorical_features}"
        )
    if not bundle.numeric_features:
        raise ValueError("At least one data.metadata_numeric or time_limit feature is required")

    train = bundle.split("train")
    validation = bundle.split("validation")
    for resource in ("memory", "vram"):
        valid_column = f"valid_{resource}"
        if not train[valid_column].astype(bool).any():
            raise ValueError(f"Training split has no valid {resource} labels")
        if not validation[valid_column].astype(bool).any():
            raise ValueError(f"Validation split has no valid {resource} labels")


@dataclass
class QwenPreprocessors:
    scaler: NumericScaler
    user_vocabulary: UserVocabulary

    @classmethod
    def fit(
        cls,
        bundle: DatasetBundle,
        standardize_metadata: bool = True,
    ) -> "QwenPreprocessors":
        validate_qwen_bundle(bundle)
        train = bundle.split("train")
        return cls(
            scaler=NumericScaler.fit(
                train,
                bundle.numeric_features,
                standardize=standardize_metadata,
            ),
            user_vocabulary=UserVocabulary.fit(train["user_id"].astype(str)),
        )

    def save(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.scaler.save(output / "numeric_scaler.json")
        self.user_vocabulary.save(output / "user_vocab.json")

    @classmethod
    def load(cls, directory: str | Path) -> "QwenPreprocessors":
        source = Path(directory)
        return cls(
            scaler=NumericScaler.load(source / "numeric_scaler.json"),
            user_vocabulary=UserVocabulary.load(source / "user_vocab.json"),
        )


class QwenJobDataset(Dataset):
    """A job dataset whose code field contains no injected user or metadata text."""

    def __init__(
        self,
        frame: pd.DataFrame,
        preprocessors: QwenPreprocessors,
        *,
        training: bool,
        user_dropout: float = 0.0,
        require_any_label: bool = False,
    ) -> None:
        if not 0.0 <= user_dropout <= 1.0:
            raise ValueError("user_dropout must be in [0,1]")

        selected = frame.copy()
        if require_any_label:
            selected = selected.loc[
                selected["valid_memory"].astype(bool) | selected["valid_vram"].astype(bool)
            ].copy()
        if selected.empty:
            raise ValueError("QwenJobDataset received no usable rows")

        self.frame = selected.reset_index(drop=False).rename(columns={"index": "bundle_row"})
        self.training = bool(training)
        self.user_dropout = float(user_dropout)
        self.metadata = preprocessors.scaler.transform(self.frame)
        self.user_ids = preprocessors.user_vocabulary.encode(self.frame["user_id"].astype(str))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        user_id = int(self.user_ids[index])
        if self.training and user_id != 0 and torch.rand(()) < self.user_dropout:
            user_id = 0

        code = script_encoder_input(row["code"])

        valid = np.asarray(
            [bool(row["valid_memory"]), bool(row["valid_vram"])], dtype=np.bool_
        )
        raw_targets = np.asarray(
            [row["target_memory"], row["target_vram"]], dtype=np.float32
        )
        targets = np.where(valid, raw_targets, 0.0).astype(np.float32)

        return {
            "code": code,
            "user_id": user_id,
            "metadata": self.metadata[index],
            "targets": targets,
            "target_mask": valid,
            "row_index": int(row["bundle_row"]),
        }


class QwenBatchCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_tokens: int,
        head_fraction: float = 0.60,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        pad_to_multiple_of: int | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= head_fraction <= 1.0:
            raise ValueError("head_fraction must be in [0,1]")
        self.tokenizer = tokenizer
        self.max_tokens = int(max_tokens)
        self.head_fraction = float(head_fraction)
        self.prompt_prefix = str(prompt_prefix)
        self.prompt_suffix = str(prompt_suffix)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.prefix_ids = self.tokenizer.encode(
            self.prompt_prefix, add_special_tokens=False
        )
        self.suffix_ids = self.tokenizer.encode(
            self.prompt_suffix, add_special_tokens=False
        )
        special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self.script_token_budget = (
            self.max_tokens - len(self.prefix_ids) - len(self.suffix_ids) - special_tokens
        )
        if self.script_token_budget <= 0:
            raise ValueError(
                "qwen.max_tokens is too small after prompt and special-token reservation"
            )

    def _encode_script(self, code: str) -> Dict[str, Any]:
        script_ids = self.tokenizer.encode(code, add_special_tokens=False)
        if len(script_ids) > self.script_token_budget:
            head_count = int(self.head_fraction * self.script_token_budget)
            tail_count = self.script_token_budget - head_count
            head = script_ids[:head_count]
            tail = script_ids[-tail_count:] if tail_count else []
            script_ids = head + tail
        combined = self.prefix_ids + script_ids + self.suffix_ids
        return self.tokenizer.prepare_for_model(
            combined,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prepared = [self._encode_script(example["code"]) for example in examples]
        encoded = self.tokenizer.pad(
            prepared,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.max_tokens:
            raise ValueError(
                "Padding to a multiple exceeded qwen.max_tokens; choose a compatible max_tokens"
            )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "user_ids": torch.as_tensor(
                [example["user_id"] for example in examples], dtype=torch.long
            ),
            "metadata": torch.as_tensor(
                np.stack([example["metadata"] for example in examples]),
                dtype=torch.float32,
            ),
            "targets": torch.as_tensor(
                np.stack([example["targets"] for example in examples]),
                dtype=torch.float32,
            ),
            "target_mask": torch.as_tensor(
                np.stack([example["target_mask"] for example in examples]),
                dtype=torch.bool,
            ),
            "row_index": torch.as_tensor(
                [example["row_index"] for example in examples], dtype=torch.long
            ),
        }


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has neither a pad token nor an EOS token")
    tokenizer.pad_token = tokenizer.eos_token


def known_user_flags(
    user_vocabulary: UserVocabulary,
    users: Iterable[str],
) -> np.ndarray:
    return user_vocabulary.encode(users) != 0
